import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class FashionMNISTGeneratedDataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None):
        """
        Initialize Fashion-MNIST generated dataset.

        Args:
            data_dir (str): Directory containing images, e.g., 'outputs/fmnist-samples'.
            labels_file (str): Path to train_labels.json file.
            transform (callable, optional): Image preprocessing transforms.
        """
        self.data_dir = data_dir
        self.transform = transform

        # Read train_labels.json
        with open(labels_file, 'r') as f:
            self.labels_dict = json.load(f)

        # Collect all image paths and labels
        self.image_paths = []
        self.hard_labels = []
        self.soft_labels = []

        # Fashion-MNIST classes
        self.classes = [
            "T-shirt/top", "Trouser", "Pullover", "Dress", "Coat",
            "Sandal", "Shirt", "Sneaker", "Bag", "Ankle_boot"
        ]

        for class_name in self.classes:
            class_dir = os.path.join(data_dir, "samples", class_name)
            if not os.path.exists(class_dir):
                continue
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                # Ensure image path exists in labels_dict
                if image_path in self.labels_dict[class_name]:
                    self.image_paths.append(image_path)
                    self.hard_labels.append(self.labels_dict[class_name][image_path]["label"])
                    self.soft_labels.append(self.labels_dict[class_name][image_path]["soft_label"])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # Load image
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("L")  # Grayscale for Fashion-MNIST

        # Apply transforms
        if self.transform:
            image = self.transform(image)

        # Get soft label (compatible with training code)
        soft_label = torch.tensor(self.soft_labels[idx], dtype=torch.float32)

        return image, soft_label


if __name__ == "__main__":
    # Define data directory and labels file
    data_dir = "outputs/fmnist-samples"
    labels_file = "outputs/fmnist-samples/train_labels.json"

    # Define preprocessing transforms
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 255] -> [0, 1]
        transforms.Normalize((0.5,), (0.5,))  # [-1, 1], single channel for grayscale
    ])

    # Initialize dataset
    dataset = FashionMNISTGeneratedDataset(
        data_dir=data_dir,
        labels_file=labels_file,
        transform=transform
    )

    # Print dataset information
    print(f"Dataset size: {len(dataset)}")
    print(f"Classes: {dataset.classes}")

    # Get first sample
    image, soft_label = dataset[0]
    print(f"First sample image shape: {image.shape}")  # Should be [1, 28, 28]
    print(f"First sample soft label: {soft_label.tolist()}")  # Should be list of 10 floats
    print(f"First sample image path: {dataset.image_paths[0]}")
    print(f"First sample hard label: {dataset.hard_labels[0]}")