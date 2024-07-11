import random
from torchvision import transforms
import torch
from torch.utils.data import Dataset

class HyperspectralImageAugmentation:
    def __init__(self, crop_size, spectral_band_dropout_prob=0.1):
        self.crop_size = crop_size
        self.spectral_band_dropout_prob = spectral_band_dropout_prob

    def __call__(self, hyperspectral_image):
        # Unpack the input sample into hyperspectral image tensor and label
        # hyperspectral_image, label = sample

        # Perform hyperspectral image augmentation
        # augmented_hyperspectral_image = self.random_crop(hyperspectral_image)
        augmented_hyperspectral_image = self.random_spectral_band_dropout(hyperspectral_image)

        # return augmented_hyperspectral_image, label
        return augmented_hyperspectral_image

    def random_crop(self, hyperspectral_image):
        # print(hyperspectral_image.size()) #200.7.7
        _, h, w = hyperspectral_image.shape
        top = random.randint(0, h - self.crop_size)
        left = random.randint(0, w - self.crop_size)
        return hyperspectral_image[:, top:top + self.crop_size, left:left + self.crop_size]

    def random_spectral_band_dropout(self, hyperspectral_image):
        num_spectral_bands = hyperspectral_image.shape[0]
        dropout_mask = torch.rand(num_spectral_bands) > self.spectral_band_dropout_prob
        return hyperspectral_image * dropout_mask[:, None, None]


class CustomDataset(Dataset):
    def __init__(self, input_data, labels, transform=None):
        self.input_data = input_data
        self.labels = labels
        self.transform = transform

    def __len__(self):
        return len(self.input_data)

    def __getitem__(self, idx):

        if self.transform:
            data_aug = self.transform(self.input_data[idx])
            data_aug = data_aug.view(data_aug.size(0), -1)
            # print(data_aug.size()) #200,49

        sample = data_aug, self.labels[idx]

        return sample

def hsi_aug(x_train, y_train):
    # Create a custom dataset with the HyperspectralImageAugmentation transformation
    transform = transforms.Compose([
        HyperspectralImageAugmentation(crop_size=7, spectral_band_dropout_prob=0.1),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(degrees=30),
        # transforms.ToTensor(),
    ])
    # Create the dataset
    dataset = CustomDataset(x_train, y_train, transform=transform)

    return dataset