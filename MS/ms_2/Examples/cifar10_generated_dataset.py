import os
import json
import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms as transforms


class CIFAR10GeneratedDataset(Dataset):
    def __init__(self, data_dir, labels_file, transform=None):
        """
        初始化 CIFAR-10 生成数据集。

        Args:
            data_dir (str): 图像目录，例如 'outputs/cifar10-samples'。
            labels_file (str): train_labels.json 文件路径。
            transform (callable, optional): 图像预处理变换。
        """
        self.data_dir = data_dir
        self.transform = transform

        # 读取 train_labels.json
        with open(labels_file, 'r') as f:
            self.labels_dict = json.load(f)

        # 收集所有图像路径和标签
        self.image_paths = []
        self.hard_labels = []
        self.soft_labels = []

        # CIFAR-10 类别
        self.classes = [
            "airplane", "automobile", "bird", "cat", "deer",
            "dog", "frog", "horse", "ship", "truck"
        ]

        for class_name in self.classes:
            class_dir = os.path.join(data_dir, "samples", class_name)
            if not os.path.exists(class_dir):
                continue
            for image_name in os.listdir(class_dir):
                image_path = os.path.join(class_dir, image_name)
                # 确保图像路径在 labels_dict 中
                if image_path in self.labels_dict[class_name]:
                    self.image_paths.append(image_path)
                    self.hard_labels.append(self.labels_dict[class_name][image_path]["label"])
                    self.soft_labels.append(self.labels_dict[class_name][image_path]["soft_label"])

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        # 加载图像
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")

        # 应用变换
        if self.transform:
            image = self.transform(image)

        # 获取软标签（与训练代码兼容）
        soft_label = torch.tensor(self.soft_labels[idx], dtype=torch.float32)

        return image, soft_label



if __name__ == "__main__":
    # 定义数据目录和标签文件
    data_dir = "outputs/cifar10-samples"
    labels_file = "outputs/cifar10-samples/train_labels.json"

    # 定义预处理变换
    transform = transforms.Compose([
        transforms.ToTensor(),  # [0, 255] -> [0, 1]
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))  # [-1, 1]
    ])

    # 初始化数据集
    dataset = CIFAR10GeneratedDataset(
        data_dir=data_dir,
        labels_file=labels_file,
        transform=transform
    )

    # 打印数据集信息
    print(f"数据集大小: {len(dataset)}")
    print(f"类别: {dataset.classes}")

    # 获取第一个样本
    image, soft_label = dataset[0]
    print(f"第一个样本图像形状: {image.shape}")  # 应为 [3, 32, 32]
    print(f"第一个样本软标签: {soft_label.tolist()}")  # 应为 [10] 浮点数列表
    print(f"第一个样本图像路径: {dataset.image_paths[0]}")
    print(f"第一个样本硬标签: {dataset.hard_labels[0]}")