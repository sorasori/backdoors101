import torch.utils.data
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms

from dataset.multi_mnist_loader import MNIST
from tasks.mnist_task import MNISTTask


class MultiMNISTTask(MNISTTask):

    def load_data(self):
        transform = transforms.Compose([transforms.ToTensor(),
                                        self.normalize])
        train = MNIST(root=self.params.data_path, train=True, download=True,
                                   transform=transform,
                                   multi=True)
        train, val = torch.utils.data.random_split(train, [54000, 6000])

        #self.train_dataset = MNIST(root=self.params.data_path, train=True, download=True,
        #                           transform=transform,
        #                           multi=True)
        self.train_dataset = train
        self.val_dataset = val
        self.train_loader = DataLoader(train,
                                       batch_size=self.params.batch_size,
                                       shuffle=True,
                                       # TODO: num_workers war 4
                                       num_workers=8)
        self.val_loader = DataLoader(val,
                                       batch_size=self.params.val_batch_size,
                                       shuffle=True,
                                       # TODO: num_workers war 4
                                       num_workers=8)

        self.test_dataset = MNIST(root=self.params.data_path, train=False, download=True,
                                  transform=transform,
                                  multi=True)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.params.test_batch_size, shuffle=False,
                                      num_workers=8)
        self.classes = list(range(100))
