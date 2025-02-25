import torch
import datasets
from torch.utils.data import DataLoader, Dataset
import pytorch_lightning as pl
from torchvision import transforms


class HuggingFaceDataset(Dataset):
    def __init__(self, hf_dataset, transform=None):
        self.dataset = hf_dataset
        self.transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, idx):
        item = self.dataset[idx]
        image = item["image"].convert("L")  # Convert to grayscale if needed
        label = torch.tensor(item["label"], dtype=torch.long)

        if self.transform:
            image = self.transform(image)

        return image, label


class DataModule(pl.LightningDataModule):
    def __init__(self, dataset_name="mnist", batch_size=64, sample_size=None, cache_dir="./dataset", num_workers=0):
        """
        :param dataset_name: Name of the Hugging Face dataset (e.g., "mnist").
        :param batch_size: Number of samples per batch.
        :param sample_size: Number of samples to use (if None, use full dataset).
        :param cache_dir: Directory to store cached datasets.
        :param num_workers: Number of worker threads for data loading.
        """
        super().__init__()
        self.dataset_name = dataset_name
        self.batch_size = batch_size
        self.sample_size = sample_size
        self.cache_dir = cache_dir
        self.num_workers = num_workers  # Added num_workers
        self.transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,)),  # Normalize to [-1, 1]
            ]
        )

    def prepare_data(self):
        # Downloads the dataset (only needed once), specifying cache directory
        datasets.load_dataset(self.dataset_name, cache_dir=self.cache_dir)

    def setup(self, stage=None):
        # Load the dataset from cache
        dataset = datasets.load_dataset(self.dataset_name, cache_dir=self.cache_dir)

        # Apply sampling if sample_size is specified
        if self.sample_size:
            train_data = dataset["train"].select(range(min(self.sample_size, len(dataset["train"]))))
            val_data = dataset["test"].select(range(min(self.sample_size // 10, len(dataset["test"]))))
            test_data = dataset["test"].select(range(min(self.sample_size // 10, len(dataset["test"]))))
        else:
            train_data = dataset["train"]
            val_data = dataset["test"]
            test_data = dataset["test"]

        # Wrap in custom dataset class
        self.train_dataset = HuggingFaceDataset(train_data, transform=self.transform)
        self.val_dataset = HuggingFaceDataset(val_data, transform=self.transform)
        self.test_dataset = HuggingFaceDataset(test_data, transform=self.transform)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=self.num_workers)

    @property
    def shapes(self):
        # Return the shapes of the dataset
        x, y = self.train_dataset[0]

        # flatten input_shape into input_size
        input_size = torch.flatten(x).shape[0]
        num_classes = 10
        return {"input_size": input_size, "num_classes": num_classes}


# class HFDataset(Dataset):
#     def __init__(self, hf_dataset, transform=None):
#         self.dataset = hf_dataset
#         self.transform = transform

#     def __len__(self):
#         return len(self.dataset)

#     def __getitem__(self, idx):
#         item = self.dataset[idx]
#         image = item["image"].convert("L")  # Convert to grayscale
#         label = item["label"]

#         if self.transform:
#             image = self.transform(image)

#         return image, label


# class DataModule(pl.LightningDataModule):
#     def __init__(self, data_dir, batch_size, num_workers, sample_size=None):
#         super().__init__()
#         self.data_dir = data_dir
#         self.batch_size = batch_size
#         self.num_workers = num_workers
#         self.sample_size = sample_size  # None will load the entire dataset

#     def prepare_data(self):
#         # Download the full dataset regardless of sample_size
#         datasets.load_dataset("mnist")

#     def setup(self, stage=None):
#         entire_dataset = datasets.MNIST(
#             root=self.data_dir,
#             train=True,
#             transform=transforms.ToTensor(),
#             download=False,
#         )

#         # If sample_size is provided, use a subset of the data
#         if self.sample_size is not None:
#             self.train_ds = Subset(entire_dataset, list(range(self.sample_size)))
#             self.val_ds = Subset(entire_dataset, list(range(self.sample_size)))
#             self.test_ds = Subset(entire_dataset, list(range(self.sample_size)))
#         else:
#             # If no sample_size is provided, use the entire dataset
#             self.train_ds, self.val_ds = random_split(entire_dataset, [50000, 10000])
#             self.test_ds = datasets.MNIST(
#                 root=self.data_dir,
#                 train=False,
#                 transform=transforms.ToTensor(),
#                 download=False,
#             )

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_ds,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=True,
#         )

#     def val_dataloader(self):
#         # Here we are using the training data for validation, adjust as needed
#         return DataLoader(
#             self.train_ds,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=False,
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.test_ds,
#             batch_size=self.batch_size,
#             num_workers=self.num_workers,
#             shuffle=False,
#         )

#     def sample_dataloader(self):
#         # Return the sample dataloader for testing purposes
#         return self.train_dataloader()  # This will use the subset or full dataset

#     @property
#     def shapes(self):
#         # Return the shapes of the dataset
#         x, y = self.train_ds[0]

#         # flatten input_shape into input_size
#         input_size = torch.flatten(x).shape[0]
#         num_classes = 10
#         return {"input_size": input_size, "num_classes": num_classes}
