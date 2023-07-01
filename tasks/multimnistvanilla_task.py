from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from dataset.multi_mnist_loader import MNIST
from models.simpler import SimplerNet
from tasks.mnist_task import MNISTTask
from tasks.multimnist_task import MultiMNISTTask


class MultiMNISTVanillaTask(MultiMNISTTask):

    def build_model(self):
        return SimplerNet(num_classes=len(self.classes))
