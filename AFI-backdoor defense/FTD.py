import argparse
from urllib import request
from torchvision.utils import save_image
import main
import torch
from PIL import Image
from torchvision import models, transforms
from pycocotools.coco import COCO
from scipy.spatial.distance import euclidean
import shutil
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import os
import cv2

def main():
    print("===Setup running===")
    parser = argparse.ArgumentParser()#创建解析器：使用 argparse 的第一步是创建一个 ArgumentParser 对象。ArgumentParser 对象包含将命令行解析成 Python 数据类型所需的全部信息。
    parser.add_argument("--config", default="./config/baseline_ftd.yaml")#添加参数：给一个 ArgumentParser 添加程序参数信息是通过调用 add_argument() 方法完成的。
    parser.add_argument("--gpu", default="0", type=str)
    parser.add_argument(
        "--resume",#摘要/概述
        default="",
        type=str,
        help="checkpoint name (empty string means the latest checkpoint)\
            or False (means training from scratch).",#数据集部分现在提供了最终的测试集（包括真实数据）。在那里，您将找到使用GTSRB数据集时应该参考哪篇论文的信息。
    )

    #在/root/autodl-tmp/badnets-MNIST/attack_after路径下的图片中，含有clean的所有图片进行特征提取，选出特征反差最大的两张图片，分别保存到指定文件夹下。
    input_directory = "/root/autodl-tmp/badnets-MNIST/attack_after"
    output_directory = "/root/autodl-tmp/badnets-MNIST/dissimilar_pairs"

    # #融合
    input_folder = "/root/autodl-tmp/badnets-MNIST/attack_after"  # 替换为输入图片目录
    image1_path = "/root/autodl-tmp/badnets-MNIST/dissimilar_pairs/most_dissimilar_1.png"  # 基础图片1
    image2_path = "/root/autodl-tmp/badnets-MNIST/dissimilar_pairs/most_dissimilar_2.png"  # 基础图片2
    output_folder = "/root/autodl-tmp/badnets-MNIST/ronghe"  # 保存融合后图片的文件夹
    overlay_images(input_folder, image1_path, image2_path, output_folder, overlay_alpha=0.3)


def set_transparency_and_save(image_path, output_path=None, alpha=1, replace_original=False):
    """
    将图片透明度设置为 alpha 并保存为 PNG
    :param image_path: 输入图片路径
    :param output_path: 输出图片路径（None则自动生成）
    :param alpha: 透明度（0.0透明 - 1.0不透明）
    :param replace_original: 是否替换原文件
    """
    try:
        # 打开图片并转为 RGBA 模式
        img = Image.open(image_path).convert("RGBA")
        
        # 分离通道
        r, g, b, a = img.split()
        
        # 修改 Alpha 通道（透明度）
        a = a.point(lambda x: int(x * alpha))
        
        # 合并通道
        img_transparent = Image.merge("RGBA", (r, g, b, a))
        
        # 处理输出路径
        if replace_original:
            output_path = image_path  # 覆盖原文件
        elif output_path is None:
            base, ext = os.path.splitext(image_path)
            output_path = f"{base}_transparent.png"
        
        # 保存为PNG（自动处理透明背景）
        img_transparent.save(output_path, "PNG")
        print(f"处理成功: {image_path} -> {output_path}")
        
    except Exception as e:
        print(f"处理失败 {image_path}: {str(e)}")



# 保存图像的函数
def save_images(data_loader, folder_path, prefix="image"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # 如果文件夹不存在，创建它

    image_count = 0
    for i, (images, labels) in enumerate(data_loader):
        for j in range(images.size(0)):
            img = images[j]  # 获取当前图片
            img_filename = os.path.join(folder_path, f"{prefix}_{image_count}.png")
            save_image(img, img_filename)  # 保存图像
            image_count += 1
            print(f"Saved {img_filename}")






def overlay_images(input_folder, image1_path, image2_path, output_folder, overlay_alpha=0.3):
    """
    将input_folder中的每张图片直接叠加到image1和image2上
    :param input_folder: 待处理图片目录
    :param image1_path: 基础图片1路径
    :param image2_path: 基础图片2路径
    :param output_folder: 输出目录
    :param overlay_alpha: 叠加透明度 (0-1)
    """
    # 创建输出目录
    os.makedirs(output_folder, exist_ok=True)
    
    # 加载基础图片
    base_img1 = cv2.imread(image1_path, cv2.IMREAD_UNCHANGED)
    base_img2 = cv2.imread(image2_path, cv2.IMREAD_UNCHANGED)
    
    # 验证基础图片
    if base_img1 is None:
        raise ValueError(f"无法读取基础图片1: {image1_path}")
    if base_img2 is None:
        raise ValueError(f"无法读取基础图片2: {image2_path}")

    # 确保基础图片有alpha通道
    if base_img1.shape[2] == 3:
        base_img1 = cv2.cvtColor(base_img1, cv2.COLOR_BGR2BGRA)
    if base_img2.shape[2] == 3:
        base_img2 = cv2.cvtColor(base_img2, cv2.COLOR_BGR2BGRA)

    # 处理每张图片
    for filename in os.listdir(input_folder):
        if not filename.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.tiff', '.webp')):
            continue
            
        input_path = os.path.join(input_folder, filename)
        overlay_img = cv2.imread(input_path, cv2.IMREAD_UNCHANGED)
        
        if overlay_img is None:
            print(f"⚠️ 跳过无法读取的文件: {input_path}")
            continue
            
        try:
            # 确保叠加图片有alpha通道
            if overlay_img.shape[2] == 3:
                overlay_img = cv2.cvtColor(overlay_img, cv2.COLOR_BGR2BGRA)
            
            # 调整尺寸匹配基础图片
            overlay_resized1 = cv2.resize(overlay_img, (base_img1.shape[1], base_img1.shape[0]))
            overlay_resized2 = cv2.resize(overlay_img, (base_img2.shape[1], base_img2.shape[0]))
            
            # 直接叠加（不再融合）
            result1 = base_img1.copy()
            result2 = base_img2.copy()
            
            # 创建叠加蒙版
            overlay_mask = (overlay_resized1[:,:,3] > 0).astype(np.uint8) * 255
            
            # 叠加到基础图片1
            for c in range(0, 3):
                result1[:,:,c] = np.where(
                    overlay_mask == 255,
                    overlay_resized1[:,:,c] * overlay_alpha + base_img1[:,:,c] * (1 - overlay_alpha),
                    base_img1[:,:,c]
                )
            
            # 叠加到基础图片2
            for c in range(0, 3):
                result2[:,:,c] = np.where(
                    overlay_mask == 255,
                    overlay_resized2[:,:,c] * overlay_alpha + base_img2[:,:,c] * (1 - overlay_alpha),
                    base_img2[:,:,c]
                )
            
            # 保存结果（带透明通道）
            basename = os.path.splitext(filename)[0]
            output_path1 = os.path.join(output_folder, f"{basename}_overlay_base1.png")
            output_path2 = os.path.join(output_folder, f"{basename}_overlay_base2.png")
            
            cv2.imwrite(output_path1, result1)
            cv2.imwrite(output_path2, result2)
            
            print(f"✅ 叠加完成: {output_path1}")
            print(f"✅ 叠加完成: {output_path2}")
            
        except Exception as e:
            print(f"❌ 处理失败 {filename}: {str(e)}")

def find_clean_images(input_dir):
    """查找所有包含'clean'的图片路径"""
    image_paths = []
    for root, _, files in os.walk(input_dir):
        for f in files:
            if "clean" in f.lower() and f.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, f))
    return image_paths

def extract_features(image_path):
    """提取图片特征"""
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return None
    img = cv2.resize(img, (28, 28))  # MNIST标准尺寸
    return img.flatten()

def find_most_dissimilar(input_dir, output_dir):
    """主处理函数"""
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 查找所有clean图片
    image_paths = find_clean_images(input_dir)
    print(f"找到 {len(image_paths)} 张含'clean'的图片")
    
    if len(image_paths) < 2:
        available_files = "\n".join(os.listdir(input_dir)[:10])  # 显示前10个文件供检查
        raise ValueError(
            f"需要至少2张含'clean'的图片，但只找到 {len(image_paths)} 张\n"
            f"目录内容示例:\n{available_files}\n"
            f"请检查:\n1. 路径是否正确: {input_dir}\n"
            "2. 文件名是否包含'clean'\n"
            "3. 文件扩展名是否为.png/.jpg/.jpeg"
        )
    
    # 提取特征
    features = []
    valid_paths = []
    for path in tqdm(image_paths, desc="提取特征"):
        feat = extract_features(path)
        if feat is not None:
            features.append(feat)
            valid_paths.append(path)
    
    features = np.array(features)
    scaler = StandardScaler()
    features = scaler.fit_transform(features)
    
    # 计算差异
    pca = PCA(n_components=min(50, len(features)-1))
    features_pca = pca.fit_transform(features)
    
    max_dist = -1
    best_pair = (0, 1)
    for i in range(len(features_pca)):
        for j in range(i+1, len(features_pca)):
            dist = np.linalg.norm(features_pca[i] - features_pca[j])
            if dist > max_dist:
                max_dist = dist
                best_pair = (i, j)
    
    # 保存结果
    for idx, img_idx in enumerate(best_pair, 1):
        img = cv2.imread(valid_paths[img_idx])
        output_path = os.path.join(output_dir, f"most_dissimilar_{idx}.png")
        cv2.imwrite(output_path, img)
        print(f"保存结果: {output_path}")


if __name__ == "__main__":
    main()








