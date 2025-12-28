# coding:utf8
import torch
from torch import nn
import torchvision.utils as vutils
import matplotlib.pyplot as plt
import numpy as np
from torch.utils.data import DataLoader

from BAM_Code.DatasetLoader import DatasetLoader
from net.dcgan import NetG


# 假设的配置类
class Opt:
    def __init__(self):
        self.nz = 100  # 噪声维度
        self.ngf = 64  # 生成器特征图基数
        self.ndf = 64  # 判别器特征图基数



def main():
    # 初始化配置
    opt = Opt()

    # 创建生成器实例
    netG = NetG(opt)

    # 加载预训练权重（如果有的话）
    netG.load_state_dict(torch.load('net/netg_200.pth'))

    # 设置为评估模式
    netG.eval()

    # 生成随机噪声作为输入
    noise = torch.randn(64, opt.nz, 1, 1)  # 批量生成64张图片

    # 生成图像
    with torch.no_grad():
        fake_images = netG(noise)

    # 将图像数据转换为0-1范围
    fake_images = (fake_images + 1) / 2  # 从tanh的(-1,1)映射到(0,1)

    # 创建一个网格来显示图像
    img_grid = vutils.make_grid(fake_images, padding=2, normalize=False, nrow=8)

    # 转换为numpy数组并调整通道顺序
    np_img = img_grid.numpy().transpose((1, 2, 0))

    # 显示图像
    plt.figure(figsize=(10, 10))
    plt.imshow(np_img)
    plt.axis('off')
    plt.title("Generated Images")
    plt.show()

    # 保存图像到文件
    vutils.save_image(fake_images, 'generated_images.png', padding=2, normalize=True)


if __name__ == '__main__':
    main()
    dataset = DatasetLoader('G:\ssd\SaveDataset\Batches\AlexnetSurrogate\\40enerations', file_size=20000)
    dataloader = DataLoader(
        dataset, batch_size=64, shuffle=False, pin_memory=True, num_workers=32
    )
    print("sss")
    print(len(dataloader))