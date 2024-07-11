import torch
import torch.nn as nn
import copy

class EWC:
    def __init__(self, dataloader, device):
        # self.model = model
        self.dataloader = dataloader
        self.device = device
        # self.original_model_state = {name: param.clone().detach() for name, param in model.named_parameters()}
        # self.prev_model_state = copy.deepcopy(model.state_dict())
        self.prev_model_state = None
        # self.prev_model = copy.deepcopy(model)
        # self.prev_model = copy.deepcopy(model)
        self.fisher_information = {}
        # self._store_fisher_information(self.prev_model)

    def _store_fisher_information(self, model):
        model.eval()
        self.fisher_information = {}
        for name, param in model.named_parameters():
            self.fisher_information[name] = torch.zeros_like(param)

        for inputs, targets in self.dataloader:
            # inputs, targets = inputs.to(self.device), targets.to(self.device)
            #
            # model.zero_grad()
            # outputs, _ = model(inputs)
            # loss = nn.CrossEntropyLoss()(outputs, targets)
            #
            # loss.backward()

            for name, param in model.named_parameters():
                if param.grad is not None:
                    self.fisher_information[name] += (param.grad ** 2) / len(self.dataloader)
                # else:
                #     self.fisher_information[name] += torch.zeros_like(param)

    def penalty(self, model):
        loss = 0
        for name, param in model.named_parameters():
            if name in self.fisher_information:
                # loss += (self.fisher_information[name] * (param - model.state_dict()[name]) ** 2).sum()
                loss += (self.fisher_information[name] * (param - self.prev_model_state[name]) ** 2).sum()
                # loss += (self.fisher_information[name] * (param - self.prev_model.state_dict()[name]) ** 2).sum()
        return loss / 2
