import argparse
import os
import pathlib
import re
import time
import datetime
import random
import pandas as pd
import torch
from torch.utils.data import DataLoader
from analyze_predictions import *
from dataset import build_poisoned_training_set, build_testset
from deeplearning import evaluate_badnets, optimizer_picker, train_one_epoch, predict_single_image
from models import BadNet
import glob
import os
import json
from torchvision.utils import save_image

parser = argparse.ArgumentParser(description='Reproduce the basic backdoor attack in "Badnets: Identifying vulnerabilities in the machine learning model supply chain".')
parser.add_argument('--dataset', default='ImageNet', help='Which dataset to use (MNIST or CIFAR10 or ImageNet, default: MNIST)')
parser.add_argument('--nb_classes', default=10, type=int, help='number of the classification types')
parser.add_argument('--load_local', action='store_true', help='train model or directly load model (default true, if you add this param, then load trained local model to evaluate the performance)')
parser.add_argument('--loss', default='mse', help='Which loss function to use (mse or cross, default: mse)')
parser.add_argument('--optimizer', default='sgd', help='Which optimizer to use (sgd or adam, default: sgd)')
parser.add_argument('--epochs', default=100, help='Number of epochs to train backdoor model, default: 100')
parser.add_argument('--batch_size', type=int, default=64, help='Batch size to split dataset, default: 64')
parser.add_argument('--num_workers', type=int, default=0, help='Batch size to split dataset, default: 64')
parser.add_argument('--lr', type=float, default=0.001, help='Learning rate of the model, default: 0.001')
parser.add_argument('--download', action='store_true', help='Do you want to download data ( default false, if you add this param, then download)')
parser.add_argument('--data_path', default='./data/', help='Which dataset to use (badnets-mnist/dataset/imagenet100/ or ./dataset/ Place to load dataset (default: ./dataset/)')
parser.add_argument('--device', default='cpu', help='device to use for training / testing (cpu, or cuda:1, default: cpu)')
# poison settings
parser.add_argument('--poisoning_rate', type=float, default=0.1, help='poisoning portion (float, range from 0 to 1, default: 0.1)')
parser.add_argument('--trigger_label', type=int, default=1, help='The NO. of trigger label (int, range from 0 to 10, default: 0)')
parser.add_argument('--trigger_path', default="/root/badnets-mnist/triggers/trigger_white.png", help='Trigger Path (default: ./triggers/trigger_white.png)')
parser.add_argument('--trigger_size', type=int, default=5, help='Trigger Size (int, default: 5)')

args = parser.parse_args()

def main():
    print("{}".format(args).replace(', ', ',\n'))

    if re.match('cuda:\d', args.device):
        cuda_num = args.device.split(':')[1]
        os.environ['CUDA_VISIBLE_DEVICES'] = cuda_num
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu") # if you're using MBP M1, you can also use "mps"

    # create related path
    pathlib.Path("./checkpoints/").mkdir(parents=True, exist_ok=True)
    pathlib.Path("./logs/").mkdir(parents=True, exist_ok=True)

    print("\n# load dataset: %s " % args.dataset)
    dataset_train, args.nb_classes = build_poisoned_training_set(is_train=True, args=args)
    dataset_val_clean, dataset_val_poisoned = build_testset(is_train=False, args=args)
    
    data_loader_train        = DataLoader(dataset_train,         batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_loader_val_clean    = DataLoader(dataset_val_clean,     batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers)
    data_loader_val_poisoned = DataLoader(dataset_val_poisoned,  batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers) # shuffle 随机化

    # 1、保存攻击后的图片
    print("Saving clean images...")
    save_images(data_loader_val_clean, "/root/autodl-tmp/ImageNet/attack_after", prefix="clean")
    print("Saving poisoned images...")
    save_images(data_loader_val_poisoned, "/root/autodl-tmp/ImageNet/attack_after", prefix="poisoned")
    print("Saving 100 clean images per class...")
            # MNIST数据集的分类
    class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    save_sampled_images(data_loader_val_clean, "/root/autodl-tmp/ImageNet/attack_after", class_names, num_samples_per_class=100, prefix="clean")
    print("Saving 100 poisoned images per class...")
    save_sampled_images(data_loader_val_poisoned, "/root/autodl-tmp/ImageNet/attack_after", class_names, num_samples_per_class=100, prefix="poisoned")
  
    model = BadNet(input_channels=dataset_train.channels, output_num=args.nb_classes).to(device)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = optimizer_picker(args.optimizer, model.parameters(), lr=args.lr)

    basic_model_path = "./checkpoints/badnet-%s.pth" % args.dataset
    start_time = time.time()
    if os.path.exists(basic_model_path):  
        print("✅ 发现已训练的模型，加载中...")
        model.load_state_dict(torch.load(basic_model_path), strict=True)
        test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)
        if test_stats['clean_acc'] < 0.85:  
            print("⚠️ 虽然加载了模型，但准确率过低，可能仍需训练！")
            args.load_local = False
        else:
            print(f"✅ 发现已训练模型，当前准确率: {test_stats['clean_acc']:.2f}")
            args.load_local = True
    else:
        print("⚠️ 未发现已训练模型，开始新训练...")
        args.load_local = False

    if args.load_local:
        print("## Load model from : %s" % basic_model_path)
        model.load_state_dict(torch.load(basic_model_path), strict=True)
        device = "cuda"  # 或 "cpu"

        # MNIST数据集的分类
        class_names = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]

        # # 2、过滤掉未被攻击成功的中毒图像
        # folder_path = "/root/autodl-tmp/badnets-MNIST/attack_after"  # 替换成你的图片文件夹路径
        # filter_images_by_prediction(folder_path, model, device, class_names)
        

        # 4、设定参数
        folder_path = "/root/autodl-tmp/badnets-MNIST/ronghe"  # 替换为你的图片文件夹路径
        output_file = "/root/badnets-mnist/prediction_results.json"  # 预测结果保存路径
        predict_images_in_folder(folder_path, model, device, class_names, output_file)
        # 使用示例
        json_file = "/root/badnets-mnist/prediction_results.json"  # 替换为你的 JSON 文件路径
        results = analyze_predictions(json_file)
        print(results)
        folder = "/root/autodl-tmp/badnets-MNIST/attack_after"
        results, acc = predict_clean_images_and_calc_acc(folder, model, device, class_names)
    else:
        print(f"Start training for {args.epochs} epochs")
        stats = []
        for epoch in range(args.epochs):
            train_stats = train_one_epoch(data_loader_train, model, criterion, optimizer, args.loss, device)
            test_stats = evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device)
            print(f"# EPOCH {epoch}   loss: {train_stats['loss']:.4f} Test Acc: {test_stats['clean_acc']:.4f}, ASR: {test_stats['asr']:.4f}\n")
            
            # save model 
            torch.save(model.state_dict(), basic_model_path)

            log_stats = {**{f'train_{k}': v for k, v in train_stats.items()},
                            **{f'test_{k}': v for k, v in test_stats.items()},
                            'epoch': epoch,
            }

            # save training stats
            stats.append(log_stats)
            df = pd.DataFrame(stats)
            df.to_csv("./logs/%s_trigger%d.csv" % (args.dataset, args.trigger_label), index=False, encoding='utf-8')

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))
    print('Training time {}'.format(total_time_str))

def coco_collate_fn(batch):
    """
    自定义 collate_fn 用于处理 COCO 数据集的批次数据。
    """
    images = []
    targets = []

    for img, target in batch:
        images.append(img)
        targets.append(target)

    # 将图像堆叠成一个批次
    images = torch.stack(images, dim=0)

    return images, targets

def predict_images_in_folder(folder_path, model, device, class_names, output_file):
    image_paths = glob.glob(os.path.join(folder_path, "*.png"))  # 获取所有 PNG 图片路径
    results = []
    
    for image_path in image_paths:
        result = predict_single_image(image_path, model, device, print_perform=True, class_names=class_names)
        result_info = {
            "image": os.path.basename(image_path),
            "prediction": result["index"],
            "class_name": class_names[result["index"]]
        }
        results.append(result_info)
        print(f"predict result: {result_info}")
    
    # 保存结果到 JSON 文件
    with open(output_file, "w") as f:
        json.dump(results, f, indent=4)
    
    print(f"Results saved to {output_file}")


def filter_images_by_prediction(folder_path, model, device, class_names):
    """遍历文件夹中的所有图片，删除 prediction 不是 1 的图片"""
    
    image_paths = glob.glob(os.path.join(folder_path, "*.png"))  # 获取所有 PNG 图片路径
    keep_count = 0  # 统计保留的图片数量
    delete_count = 0  # 统计删除的图片数量
    
    for image_path in image_paths:
        # 进行模型预测
        result = predict_single_image(image_path, model, device, print_perform=False, class_names=class_names)
        prediction = result["index"]

  # 获取文件名
        file_name = os.path.basename(image_path)
        # 如果 prediction 不是 1，则删除该图片
        if prediction != 1 and "poisoned" in file_name:
            os.remove(image_path)
            delete_count += 1
            print(f"🗑️ 已删除图片: {os.path.basename(image_path)} (prediction={prediction})")
        else:
            keep_count += 1  # 保留的图片数量
    
    print(f"✅ 处理完成：保留 {keep_count} 张图片，删除 {delete_count} 张图片")


def save_sampled_images(data_loader, folder_path, class_names, num_samples_per_class=100, prefix="image"):
    # 确保目标文件夹存在
    os.makedirs(folder_path, exist_ok=True)

    # 组织数据，按类别存储图像
    class_dict = {cls: [] for cls in class_names}

    # 遍历数据集，将每张图片按类别存入字典
    for images, labels in data_loader:
        for img, label in zip(images, labels):
            class_name = class_names[label.item()]
            class_dict[class_name].append(img)

    # 统一存放图片
    image_count = 0

    # 为每个类别随机选取 num_samples_per_class 张图片
    for class_name, images in class_dict.items():
        sampled_images = random.sample(images, min(num_samples_per_class, len(images)))  # 防止数据不足

        # 保存图像，所有图像放在同一文件夹
        for img in sampled_images:
            img_filename = os.path.join(folder_path, f"{prefix}_{class_name}_{image_count:05d}.png")
            save_image(img, img_filename)
            print(f"✅ Saved: {img_filename}")
            image_count += 1  # 计数递增，确保文件名唯一

    print("🎉 All images saved successfully!")



def  data_loader_val_clean_ronghe (folder, keyword="clean"):
    """
    从文件夹中加载包含特定关键字的图片文件名，并创建对应的 DataLoader。

    :param folder: 图像文件夹路径
    :param keyword: 需要匹配的关键字
    :return: 包含匹配关键字的图像路径列表
    """
    image_paths = []
    for filename in os.listdir(folder):
        if keyword in filename.lower():  # 只选取包含 'clean' 的图像文件名
            image_paths.append(os.path.join(folder, filename))
    return image_paths

def  data_loader_val_poisoned_ronghe (folder, keyword="poisoned"):
    """
    从文件夹中加载包含特定关键字的图片文件名，并创建对应的 DataLoader。

    :param folder: 图像文件夹路径
    :param keyword: 需要匹配的关键字
    :return: 包含匹配关键字的图像路径列表
    """
    image_paths = []
    for filename in os.listdir(folder):
        if keyword in filename.lower():  # 只选取包含 'clean' 的图像文件名
            image_paths.append(os.path.join(folder, filename))
    return image_paths

# 保存全部图像的函数
def save_images(data_loader, folder_path, prefix="image"):
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)  # 如果文件夹不存在，创建它

    image_count = 0
    for i, (images, labels) in enumerate(data_loader):
        for j in range(images.size(0)):
            img = images[j]  # 获取当前图片
            label = labels[j]  # 获取当前标签

            # 保存图像
            img_filename = os.path.join(folder_path, f"{prefix}_{image_count}.png")
            save_image(img, img_filename)  # 保存图像

            # 保存标签
            label_filename = os.path.join(folder_path, f"{prefix}_{image_count}_label.txt")
            with open(label_filename, 'w') as label_file:
                # 假设标签是一个单一的类索引，你可以根据需要调整为多类标签等格式
                label_file.write(str(label.item()))  # 保存标签（转换为数字）

            image_count += 1
            print(f"Saved {img_filename} and {label_filename}")

def predict_clean_images_and_calc_acc(folder_path, model, device, class_names):
    image_paths = glob.glob(os.path.join(folder_path, "*.png"))  # 获取所有 PNG 图片路径
    results = []
    total = 0
    correct = 0

    for image_path in image_paths:
        filename = os.path.basename(image_path)
        if 'clean' not in filename:
            continue  # 跳过不含 "clean" 的图片

        result = predict_single_image(image_path, model, device, print_perform=False, class_names=class_names)
        predicted_index = result["index"]
        predicted_class_name = class_names[predicted_index]

        total += 1
        if predicted_class_name in filename:
            correct += 1

        result_info = {
            "image": filename,
            "prediction": predicted_index,
            "predicted_class_name": predicted_class_name,
            "correct": predicted_class_name in filename
        }
        results.append(result_info)
        # print(f"Predict result: {result_info}")

    acc = correct / total if total > 0 else 0
    print(f"\nTotal clean images: {total}")
    print(f"Correct predictions: {correct}")
    print(f"Accuracy: {acc:.4f}")

    return results, acc

if __name__ == "__main__":
    main()
