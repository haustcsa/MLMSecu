from __future__ import print_function

import scipy.stats
from torch.nn.functional import softmax


from disguide.generatorensemble import GeneratorEnsemble
from my_utils import *

import numpy as np
import matplotlib.pyplot as plt

import torch
from typing import Tuple


# 创建generator实例
generator = network.gan.GeneratorA(nz=256, nc=3, img_size=32, activation=torch.tanh,
                                       grayscale=18).cuda()
generator.load_state_dict(torch.load('checkpoint/generator/single/generator_best.pt'))
generator.eval()


# 创建generatorEnsemble实例,使用该生成器输出的数据为[batchsize,ensemble_size, num_channels, img_size, img_size]
ensemble_size=2
generatorEnsemble = GeneratorEnsemble(nz=32, num_channels=3, img_size=32, G_activation=torch.tanh, grayscale=18, ensemble_size=2).cuda()
generatorEnsemble.load_state_dict(torch.load('checkpoint/generator/ensemble/generator_best.pt'))
generatorEnsemble.eval()

# 创建teacher模型
teacher = network.resnet_8x.ResNet18_8x(num_classes=10).cuda()
teacher.load_state_dict(torch.load('checkpoint/teacher/cifar10-resnet18_8x.pt'))
teacher.eval()


# 2. 生成样本并获取教师模型预测
def get_predictions(generator, teacher, num_samples=256, is_ensemble=False):
    if is_ensemble:
        z = torch.randn(num_samples, 32).cuda()  # ensemble的nz=32
        generated_samples = generator(z)  # [batch, ensemble_size, C, H, W]
        generated_samples = generated_samples.view(-1, 3, 32, 32)  # 合并ensemble维度
    else:
        z = torch.randn(num_samples, 256).cuda()  # 单一生成器nz=256
        generated_samples = generator(z)

    with torch.no_grad():
        predictions = teacher(generated_samples)
        probas = softmax(predictions, dim=1)
    return probas.cpu().numpy()


# 3. 计算指标
def calculate_metrics(probas):
    epsilon = 1e-10
    entropy = -np.sum(probas * np.log(probas + epsilon), axis=1).mean()
    variance = np.var(probas, axis=0).mean()
    return entropy, variance


def js_divergence(p, q):
    # 确保p和q的样本数相同（随机选择）
    min_samples = min(p.shape[0], q.shape[0])
    p_sampled = p[np.random.choice(p.shape[0], min_samples, replace=False)]
    q_sampled = q[np.random.choice(q.shape[0], min_samples, replace=False)]

    m = 0.5 * (p_sampled + q_sampled)
    kl_pm = scipy.stats.entropy(p_sampled.T, m.T)
    kl_qm = scipy.stats.entropy(q_sampled.T, m.T)
    return 0.5 * (kl_pm + kl_qm).mean()


# 4. 主流程
if __name__ == "__main__":
    # 生成预测（确保总样本数一致）
    num_samples = 2048  # 调整为能被ensemble_size整除的值
    probas_single = get_predictions(generator, teacher, num_samples=num_samples)
    probas_ensemble = get_predictions(generatorEnsemble, teacher, num_samples=num_samples // ensemble_size,
                                      is_ensemble=True)  # 注意调整数量

    # 计算指标
    entropy_single, var_single = calculate_metrics(probas_single)
    entropy_ensemble, var_ensemble = calculate_metrics(probas_ensemble)
    js = js_divergence(probas_ensemble, probas_single)

    # 打印结果
    print("===== 分布多样性比较 =====")
    print(f"单一生成器 - 平均类别熵: {entropy_single:.4f}")
    print(f"生成器集合 - 平均类别熵: {entropy_ensemble:.4f} (差异: {entropy_ensemble - entropy_single:+.4f})")
    print(f"单一生成器 - 预测方差: {var_single:.6f}")
    print(f"生成器集合 - 预测方差: {var_ensemble:.6f} (差异: {var_ensemble - var_single:+.6f})")
    print(f"JS散度 (集合 vs 单一): {js:.4f}")

    # # 可视化
    # plt.figure(figsize=(12, 4))
    # plt.subplot(121)
    # plt.bar(range(10), probas_single.mean(axis=0), alpha=0.6, label='Single')
    # plt.bar(range(10), probas_ensemble.mean(axis=0), alpha=0.6, label='Ensemble')
    # plt.title("Class-wise Mean Probability")
    # plt.legend()

    # plt.subplot(122)
    # plt.bar(range(10), probas_single.var(axis=0), alpha=0.6, label='Single')
    # plt.bar(range(10), probas_ensemble.var(axis=0), alpha=0.6, label='Ensemble')
    # plt.title("Class-wise Variance")
    # plt.legend()
    # plt.show()

    plt.figure(figsize=(12, 6))
    classes = np.arange(10)  # 类别 0-9
    bar_width = 0.35  # 柱状图宽度

    # 调整 x 轴位置，使得 Single 和 Ensemble 的柱状图并排
    plt.bar(classes - bar_width / 2, probas_single.mean(axis=0), width=bar_width, alpha=0.6, label='Single')
    plt.bar(classes + bar_width / 2, probas_ensemble.mean(axis=0), width=bar_width, alpha=0.6, label='Ensemble')

    plt.title("Class-wise Mean Probability (Single vs Ensemble)")
    plt.xlabel("Class")
    plt.ylabel("Mean Probability")
    plt.xticks(classes)  # 设置 x 轴刻度为类别
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.tight_layout()  # 调整布局
    plt.show()

    # 计算每个类别的平均概率
    single_means = probas_single.mean(axis=0)
    ensemble_means = probas_ensemble.mean(axis=0)

    # 输出为表格格式（可直接复制到Excel或其他软件）
    print("Class | Single (Mean Prob) | Ensemble (Mean Prob)")
    print("------|--------------------|---------------------")
    for i in range(10):
        print(f"{i:5d} | {single_means[i]:18.4f} | {ensemble_means[i]:18.4f}")