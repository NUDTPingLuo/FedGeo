"""Shared stochastic quantization utilities with real sub-byte packing.

The old implementation stored every 2/4/8-bit value in an ``int8`` tensor.
That is numerically valid but means 2-bit and 4-bit payloads occupy exactly the
same tensor storage as 8-bit payloads.  This module packs 2-bit and 4-bit codes
into bytes and provides one encoder/decoder for both uplink and downlink.
"""

import torch


SUPPORTED_BITS = (2, 4, 8, 16, 32)


def _validate_bits(bits):
    bits = int(bits)
    if bits not in SUPPORTED_BITS:
        raise ValueError("q_bits must be one of {}, got {}".format(SUPPORTED_BITS, bits))
    return bits


@torch.no_grad()
def quantize_tensor(tensor, bits):
    """Stochastically quantize a tensor and return a portable CPU payload."""
    bits = _validate_bits(bits)
    source = tensor.detach()
    shape = tuple(source.shape)
    numel = source.numel()

    if bits == 32:
        return {
            "format": "dense",
            "values": source.to(dtype=torch.float32, device="cpu").contiguous(),
            "scale": torch.tensor(1.0, dtype=torch.float32),
            "bits": 32,
            "shape": shape,
            "numel": numel,
        }

    # Symmetric stochastic quantization.  For b bits this uses the signed
    # levels [-(2^(b-1)-1), ..., +(2^(b-1)-1)].
    levels = (1 << (bits - 1)) - 1
    max_value = source.abs().max() if numel else source.new_tensor(0.0)
    safe_max_value = max_value.clamp_min(torch.finfo(torch.float32).eps)
    scaled = source.reshape(-1).float() * (float(levels) / safe_max_value)
    lower = torch.floor(scaled)
    rounded = lower + (torch.rand_like(scaled) < (scaled - lower)).to(scaled.dtype)
    rounded = rounded.clamp(-levels, levels).to(torch.int64)

    if bits in (2, 4):
        codes_per_byte = 8 // bits
        encoded = rounded + levels
        padding = (-numel) % codes_per_byte
        if padding:
            encoded = torch.cat((encoded, encoded.new_zeros(padding)))
        encoded = encoded.reshape(-1, codes_per_byte)
        packed = torch.zeros(encoded.size(0), dtype=torch.uint8, device=encoded.device)
        mask = (1 << bits) - 1
        for offset in range(codes_per_byte):
            packed |= ((encoded[:, offset] & mask) << (offset * bits)).to(torch.uint8)
        values = packed.cpu().contiguous()
        payload_format = "packed"
    elif bits == 8:
        values = rounded.to(torch.int8).cpu().contiguous()
        payload_format = "integer"
    else:  # 16 bit
        values = rounded.to(torch.int16).cpu().contiguous()
        payload_format = "integer"

    return {
        "format": payload_format,
        "values": values,
        "scale": max_value.to(dtype=torch.float32, device="cpu"),
        "bits": bits,
        "shape": shape,
        "numel": numel,
    }


def _decode_legacy(payload, device):
    """Decode the three-item tuples produced by earlier experiment code."""
    q_tensor, max_value, bits = payload
    if bits >= 32:
        return q_tensor.to(device)
    levels = (1 << (bits - 1)) - 1 if bits > 1 else 1
    scale = max_value.to(device) if isinstance(max_value, torch.Tensor) else max_value
    return q_tensor.to(device).float() * (scale / float(levels))


@torch.no_grad()
def dequantize_tensor(payload, device=None):
    """Decode a payload created by :func:`quantize_tensor`.

    Legacy tuple payloads remain readable so existing checkpoints/results are
    not invalidated by this fix.
    """
    device = torch.device("cpu") if device is None else torch.device(device)
    if isinstance(payload, (tuple, list)):
        return _decode_legacy(payload, device)

    bits = int(payload["bits"])
    shape = tuple(payload["shape"])
    numel = int(payload["numel"])
    values = payload["values"].to(device)
    if bits == 32:
        return values.reshape(shape)

    levels = (1 << (bits - 1)) - 1
    if payload.get("format") == "packed":
        codes_per_byte = 8 // bits
        mask = (1 << bits) - 1
        unpacked = []
        integer_values = values.to(torch.int64)
        for offset in range(codes_per_byte):
            unpacked.append((integer_values >> (offset * bits)) & mask)
        # Packing groups consecutive values per byte.  Stack then flatten in
        # row-major order to reconstruct the original ordering.
        quantized = torch.stack(unpacked, dim=1).reshape(-1)[:numel] - levels
    else:
        quantized = values.reshape(-1).to(torch.int64)[:numel]

    scale = payload["scale"].to(device).float()
    return (quantized.float() * (scale / float(levels))).reshape(shape)


def payload_nbytes(obj):
    """Return tensor storage plus compact scalar metadata bytes.

    This is a wire-size estimate, not Python-object memory usage.  It counts
    packed tensor storage and numeric metadata, which is the relevant quantity
    for communication comparisons.
    """
    if isinstance(obj, torch.Tensor):
        return obj.numel() * obj.element_size()
    if isinstance(obj, dict):
        return sum(payload_nbytes(value) for value in obj.values())
    if isinstance(obj, (tuple, list)):
        return sum(payload_nbytes(value) for value in obj)
    if isinstance(obj, (int, float, bool)):
        return 8
    return 0


def state_dict_nbytes(state_dict):
    return sum(tensor.numel() * tensor.element_size() for tensor in state_dict.values())
