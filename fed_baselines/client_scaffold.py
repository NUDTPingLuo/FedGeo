from fed_baselines.client_base import FedClient
import copy
from utils.models import *
import math

from torch.utils.data import DataLoader
from utils.fed_utils import init_model


class ScaffoldClient(FedClient):
    def __init__(self, name, epoch, dataset_id, model_name, lr, batch_size, momentum):
        super().__init__(name, epoch, dataset_id, model_name, lr, batch_size, momentum)
        # server control variate
        self.scv = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        # client control variate
        self.ccv = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        self.init_lr = self._lr
        ccv_state = self.ccv.state_dict()
        for key in ccv_state:
            ccv_state[key] = torch.zeros_like(ccv_state[key])
        self.ccv.load_state_dict(ccv_state)

    def update(self, total_round, global_round, model_state_dict, scv_state):
        """
        SCAFFOLD client updates local models and server control variate
        :param model_state_dict:
        :param scv_state:
        """
        # 余弦退火公式: lr_t = eta_min + 0.5 * (lr_max - eta_min) * (1 + cos(pi * current_round / total_round))
        eta_min = 0.0
        self._lr = eta_min + 0.5 * (self.init_lr - eta_min) * (
                1 + math.cos(math.pi * global_round / total_round)
        )

        self.model = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        self.model.load_state_dict(model_state_dict)
        self.scv = init_model(model_name=self.model_name, num_class=self._num_class, image_channel=self._image_channel)
        self.scv.load_state_dict(scv_state)

    def train(self):
        """
        Client trains the model on local dataset using SCAFFOLD
        :return: Local updated model, number of local data points, training loss, updated client control variate
        """
        train_loader = DataLoader(self.trainset, batch_size=self._batch_size, shuffle=True, num_workers=self._num_workers, pin_memory=torch.cuda.is_available(), persistent_workers=False)

        self.model.to(self._device)
        self.ccv.to(self._device)
        self.scv.to(self._device)
        global_state_dict = copy.deepcopy(self.model.state_dict())
        scv_state = self.scv.state_dict()
        ccv_state = self.ccv.state_dict()
        cnt = 0

        optimizer = torch.optim.SGD(self.model.parameters(), lr=self._lr, momentum=self._momentum)
        # optimizer = torch.optim.Adam(self.model.parameters(), lr=self._lr, weight_decay=1e-4)
        loss_func = nn.CrossEntropyLoss()

        epoch_loss_collector = []

        # Training process
        for epoch in range(self._epoch):
            for step, (x, y) in enumerate(train_loader):
                with torch.no_grad():
                    b_x = x.to(self._device)  # Tensor on GPU
                    b_y = y.to(self._device)  # Tensor on GPU

                with torch.enable_grad():
                    self.model.train()
                    output = self.model(b_x)
                    loss = loss_func(output, b_y.long())
                    optimizer.zero_grad()

                    loss.backward()
                    optimizer.step()

                    with torch.no_grad():
                        for name, param in self.model.named_parameters():
                            # 注意：只修正参数，不碰 BatchNorm 统计量
                            param.data = param.data - self._lr * (
                                        scv_state[name].to(self._device) - ccv_state[name].to(self._device))

                    cnt += 1
                    epoch_loss_collector.append(loss.item())

        delta_model_state = copy.deepcopy(self.model.state_dict())

        new_ccv_state = copy.deepcopy(self.ccv.state_dict())
        delta_ccv_state = copy.deepcopy(new_ccv_state)
        state_dict = self.model.state_dict()
        for key in state_dict:
            new_ccv_state[key] = ccv_state[key] - scv_state[key] + (global_state_dict[key] - state_dict[key]) / (cnt * self._lr)
            delta_ccv_state[key] = new_ccv_state[key] - ccv_state[key]
            delta_model_state[key] = state_dict[key] - global_state_dict[key]

        self.ccv.load_state_dict(new_ccv_state)

        return state_dict, self.n_data, loss.item(), delta_ccv_state
