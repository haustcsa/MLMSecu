from torch.utils.data import Dataset
from PIL import Image
import os

from torchvision import transforms

count = 0
# ---train&val data---
class SingleInputDataset(Dataset):

    def __init__(self, txt_path, train_transform=None, valid_transform=None):
        # 使用 with 语句打开文件，确保文件会被正确关闭
        with open(txt_path, 'r') as f:
            lines = f.readlines()
            print('textpath:' + txt_path)

        imgs = []
        for line in lines:
            line = line.rstrip()
            words = line.split()
            # 使用 os.path.join 处理路径，确保在不同操作系统上的兼容性
            img_path = os.path.join(words[0])
            imgs.append((img_path, int(words[1])))

        self.imgs = imgs  # generate the global list
        self.train_transform = train_transform
        self.valid_transform = valid_transform
        print('数据集大小:', len(self.imgs))  # 打印数据集的大小

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        global count
        try:
            img = Image.open(img_path).convert('RGB')
            count += 1
            print(f'\r已加载图片数量：{count}, 图片位置: {img_path}, 标签: {label}', end='')  # 用 \r 覆盖行

            # 执行 transform
            if self.train_transform is not None:
                img = self.train_transform(img)
            if self.valid_transform is not None:
                img = self.valid_transform(img)

            return img, label

        except FileNotFoundError:
            print(f"文件未找到，跳过: {img_path}")
            # 跳过该图片，递归调用 __getitem__ 获取下一个有效数据
            return self.__getitem__((index + 1) % len(self.imgs))

    def __len__(self):
        return len(self.imgs)


# ---test data---
class TestDataset(Dataset):
    def __init__(self, txt_path, test_transform=None):
        with open(txt_path, 'r') as f:
            lines = f.readlines()

        imgs = []
        for line in lines:
            line = line.rstrip()
            words = line.split()
            # 使用 os.path.join 处理路径
            img_path = os.path.join(words[0])
            imgs.append((img_path, int(words[1])))

        self.imgs = imgs  # generate the global list
        self.test_transform = test_transform

    def __getitem__(self, index):
        img_path, label = self.imgs[index]
        img = Image.open(img_path).convert('RGB')

        # transform
        if self.test_transform is not None:
            img = self.test_transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)