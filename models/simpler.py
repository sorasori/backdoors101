import torch.nn as nn
import torch.nn.functional as F

from models.model import Model


class SimplerNet(Model):
    def __init__(self, num_classes):
        super().__init__()
        self.fc1 = nn.Linear(784, 500)
        self.fc2 = nn.Linear(500, num_classes)

    def forward(self, x, latent=False):
        self.forward_passes += 1
        x = x.reshape([-1, 784])
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        out = F.log_softmax(x, dim=1)
        if latent:
            return out, x
        else:
            return out
