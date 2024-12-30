import torch
import torch.nn as nn
import torch.nn.functional as F

class SpikeKernelLoss(nn.Module):
    def __init__(self, tau_s=5.0):
        super(SpikeKernelLoss, self).__init__()
        self.tau_s = tau_s

    def forward(self, outputs, target):
        target_psp = self._psp(target, self.tau_s)
        delta = outputs - target_psp
        return 0.5 * torch.sum(delta**2)

    def _psp(self, inputs, tau_s):
        n_steps = inputs.shape[-1]
        shape = inputs.shape
        syn = torch.zeros(shape[:-1], dtype=inputs.dtype, device=inputs.device)
        syns = torch.zeros(*shape, dtype=inputs.dtype, device=inputs.device)

        for t in range(n_steps):
            syn = syn - syn / tau_s + inputs[..., t]
            syns[..., t] = syn / tau_s

        return syns

class SpikeCountLossFunction(torch.autograd.Function):
    @staticmethod
    def forward(ctx, outputs, target, desired_count, undesired_count):
        n_steps = outputs.shape[4]
        out_count = torch.sum(outputs, dim=4)

        delta = (out_count - target) / n_steps

        delta = delta.unsqueeze_(-1).repeat(1, 1, 1, 1, n_steps)

        return delta

    @staticmethod
    def backward(ctx, grad_output):
        return grad_output, None, None, None, None, None

class SpikeCountLoss(nn.Module):
    def __init__(self, desired_count, undesired_count):
        super(SpikeCountLoss, self).__init__()
        self.desired_count = desired_count
        self.undesired_count = undesired_count
        
    def forward(self, outputs, labels):
        
        target = torch.ones_like(outputs[...,0])*self.undesired_count
        
        target.scatter_(1, 
                        labels.unsqueeze(1).unsqueeze(2).unsqueeze(3),
                        torch.full_like(target, self.desired_count))
        
        delta = SpikeCountLossFunction.apply(outputs, target, self.desired_count, self.undesired_count)
                
        return 1 / 2 * torch.sum(delta**2)

class SpikeCumulativeLoss(nn.Module):
    def __init__(self, gamma = 0.0):
        super(SpikeCumulativeLoss, self).__init__()
        self.gamma = gamma
        
    def forward(self, outputs, labels):
        
        target = torch.zeros_like(outputs)
        target.scatter_(1, 
                labels.unsqueeze(1).unsqueeze(2).unsqueeze(3).unsqueeze(4).expand(-1, -1, -1, -1, outputs.shape[4]),  # T (시간 차원) 확장
                torch.full_like(target, 1))
        
        output_cumsum = torch.cumsum(outputs, dim=-1)
        target_cumsum = torch.cumsum(target, dim=-1)
        
        err = abs(target_cumsum - output_cumsum)
        focal_weight = (1-err) ** self.gamma
        return torch.sum(focal_weight*(err**2))

class SpikeSoftmaxLoss(nn.Module):
    def __init__(self):
        super(SpikeSoftmaxLoss, self).__init__()
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, outputs, target):
        delta = F.log_softmax(outputs.sum(dim=4).squeeze(-1).squeeze(-1), dim=1)
        return self.criterion(delta, target)
