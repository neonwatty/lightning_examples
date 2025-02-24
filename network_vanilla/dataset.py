import torchvision.datasets as datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torch.utils.data import Subset
from torch.utils.data import random_split
import pytorch_lightning as pl


class DataModule(pl.LightningDataModule):
    def __init__(self, data_dir, batch_size, num_workers, sample_size=None):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.sample_size = sample_size  # None will load the entire dataset

    def prepare_data(self):
        # Download the full dataset regardless of sample_size
        datasets.MNIST(self.data_dir, train=True, download=True)
        datasets.MNIST(self.data_dir, train=False, download=True)

    def setup(self, stage=None):
        entire_dataset = datasets.MNIST(
            root=self.data_dir,
            train=True,
            transform=transforms.ToTensor(),
            download=False,
        )

        # If sample_size is provided, use a subset of the data
        if self.sample_size is not None:
            self.train_ds = Subset(entire_dataset, list(range(self.sample_size)))
            self.val_ds = Subset(entire_dataset, list(range(self.sample_size)))
            self.test_ds = Subset(entire_dataset, list(range(self.sample_size)))
        else:
            # If no sample_size is provided, use the entire dataset
            self.train_ds, self.val_ds = random_split(entire_dataset, [50000, 10000])
            self.test_ds = datasets.MNIST(
                root=self.data_dir,
                train=False,
                transform=transforms.ToTensor(),
                download=False,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=True,
        )

    def val_dataloader(self):
        # Here we are using the training data for validation, adjust as needed
        return DataLoader(
            self.train_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_ds,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            shuffle=False,
        )
    
    def sample_dataloader(self):
        # Return the sample dataloader for testing purposes
        return self.train_dataloader()  # This will use the subset or full dataset

    @property
    def shapes(self):
        # Return the shapes of the dataset
        x, y = self.train_ds[0]
        input_shape = x.shape

        # input_size is vectorized shape of a single datapoint
        input_size = input_shape[0] * input_shape[1] * input_shape[2]
        num_classes = 10
        return {"input_size": input_size, "num_classes": num_classes}
