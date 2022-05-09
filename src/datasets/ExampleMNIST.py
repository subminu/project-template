from torchvision import datasets as dset
from torchvision.transforms import transforms


class ExampleMNIST(dset.MNIST):
    def __init__(
        self,
        data_dir: str,
        download: bool,
        train: bool,
    ):
        self.transforms = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Resize((1024, 1024)),
                transforms.Normalize((0.5,), (0.5,)),
            ]
        )
        super().__init__(
            root=data_dir, train=train, transform=self.transforms, download=download
        )
