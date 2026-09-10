"""Quantized-delta aggregation for the FedAvg-Q control baseline."""

from fed_baselines.server_fedgeo import GeoServer


class FedAvgQServer(GeoServer):
    def __init__(self, client_list, dataset_id, model_name, batch_size, alpha,
                 q_bits=32, run_tag=None):
        if run_tag is None:
            run_tag = "FedAvgQ_{}_alpha{}".format(dataset_id, alpha)
        super().__init__(
            client_list, dataset_id, model_name, batch_size, alpha, q_bits, run_tag
        )
