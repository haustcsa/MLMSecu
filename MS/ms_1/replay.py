from collections import deque

from my_utils import perm
import torch
import torch.nn.functional as F
import numpy as np


class ReplayMemory(object):
    """Experience replay class interface"""
    def __init__(self):
        """Initialize experience replay"""
        pass

    def update(self, fake, t_logit):
        """Update experience replay with batch of features and labels"""
        pass

    def sample(self):
        """Return batch of features and labels"""
        pass

    def new_epoch(self):
        """Update epoch. This is only relevant for experience replay that has some concept of aging"""
        pass


class NoReplayMemory(ReplayMemory):
    def __init__(self):
        pass


class ClassicalMemory(ReplayMemory):
    """Circular FIFO buffer based experiment replay. Returns uniform random samples when queried."""
    def __init__(self, device, length, batch_size):
        """Initialize replay memory"""
        self.device = device
        self.max_length = length
        self.fakes = None
        self.logits = None
        self.batch_size = batch_size
        self.size = 0
        self.head = 0

    def update(self, fake, t_logit):
        """Update memory with batch of features and labels"""
        fake = fake.cpu()
        t_logit = t_logit.cpu()
        if self.fakes is None:  # Initialize circular buffer if it hasn't already been.
            self.fakes = torch.zeros((self.max_length, fake.shape[1], fake.shape[2], fake.shape[3]))
            self.logits = torch.zeros((self.max_length, t_logit.shape[1]))
        tail = self.head + fake.shape[0]
        if tail <= self.max_length:  # Buffer is not full
            self.fakes[self.head:tail] = fake
            self.logits[self.head:tail] = t_logit
            if self.size < self.max_length:
                self.size = tail
        else:  # Buffer is full
            n = self.max_length - self.head
            self.fakes[self.head:tail] = fake[:n]
            self.logits[self.head:tail] = t_logit[:n]
            tail = tail % self.max_length
            self.fakes[:tail] = fake[n:]
            self.logits[:tail] = t_logit[n:]
            self.size = self.max_length
        self.head = tail % self.max_length

    def sample(self):
        """Return samples uniformly at random from memory"""
        assert self.fakes is not None  # Only sample after having stored samples
        assert self.size >= self.batch_size  # Only sample if we have stored a full batch of samples
        idx = perm(self.size, self.batch_size, self.device).cpu()
        return self.fakes[idx].to(self.device), self.logits[idx].to(self.device)

    def new_epoch(self):
        """Does nothing for this type of experience replay"""
        pass

    def __len__(self):
        """Returns number of stored samples"""
        return self.size

class GeneratorMemory(ReplayMemory):
    """Circular FIFO buffer for experience replay with generated data. Includes class-balance and diversity."""
    def __init__(self, device, length, batch_size):
        """Initialize replay memory"""
        self.device = device
        self.max_length = length
        self.fakes = None
        self.logits = None
        self.batch_size = batch_size
        self.size = 0
        self.head = 0

    def update(self, fake, t_logit):
        """Update memory with batch of generated features (fakes) and logits"""
        fake = fake.cpu()
        t_logit = t_logit.cpu()
        if self.fakes is None:  # Initialize circular buffer if it hasn't already been.
            self.fakes = torch.zeros((self.max_length, fake.shape[1], fake.shape[2], fake.shape[3]))
            self.logits = torch.zeros((self.max_length, t_logit.shape[1]))
        tail = self.head + fake.shape[0]
        if tail <= self.max_length:  # Buffer is not full
            self.fakes[self.head:tail] = fake
            self.logits[self.head:tail] = t_logit
            if self.size < self.max_length:
                self.size = tail
        else:  # Buffer is full
            n = self.max_length - self.head
            self.fakes[self.head:tail] = fake[:n]
            self.logits[self.head:tail] = t_logit[:n]
            tail = tail % self.max_length
            self.fakes[:tail] = fake[n:]
            self.logits[:tail] = t_logit[n:]
            self.size = self.max_length
        self.head = tail % self.max_length

    def sample(self):
        """Return samples that are diverse and class-balanced"""
        assert self.fakes is not None  # Only sample after having stored samples
        assert self.size >= self.batch_size  # Only sample if we have stored a full batch of samples
        idx = self._select_diverse_and_balanced_samples()
        return self.fakes[idx].to(self.device), self.logits[idx].to(self.device)

    def _select_diverse_and_balanced_samples(self):
        """Select diverse and class-balanced samples from memory"""
        idx = perm(self.size, self.batch_size, self.device).cpu()
        # Placeholder: Implement diversity and class balance logic here
        return idx

    def new_epoch(self):
        """Does nothing for this type of experience replay"""
        pass

    def __len__(self):
        """Returns number of stored samples"""
        return self.size


class GeneratorMemory1(ReplayMemory):
    """Circular FIFO buffer for experience replay with generated data. Includes class-balance and difficulty weighting."""

    def __init__(self, device, length, batch_size):
        """Initialize replay memory"""
        self.device = device
        self.max_length = length
        self.fakes = None
        self.logits = None
        self.batch_size = batch_size
        self.size = 0
        self.head = 0

    def update(self, fake, t_logit):
        """Update memory with batch of generated features (fakes) and logits"""
        fake = fake.cpu()
        t_logit = t_logit.cpu()
        if self.fakes is None:  # Initialize circular buffer if it hasn't already been.
            self.fakes = torch.zeros((self.max_length, fake.shape[1], fake.shape[2], fake.shape[3]))
            self.logits = torch.zeros((self.max_length, t_logit.shape[1]))
        tail = self.head + fake.shape[0]
        if tail <= self.max_length:  # Buffer is not full
            self.fakes[self.head:tail] = fake
            self.logits[self.head:tail] = t_logit
            if self.size < self.max_length:
                self.size = tail
        else:  # Buffer is full
            # Step 1: Convert logits to probabilities
            probs = F.softmax(self.logits, dim=-1)  # Convert logits to probabilities
            confidence = torch.max(probs, dim=-1).values  # Get max probability (confidence)

            # Step 2: Compute predicted labels
            predicted_labels = torch.argmax(probs, dim=-1)

            # Step 3: Ensure class balance by selecting from each class
            unique_classes, class_counts = torch.unique(predicted_labels, return_counts=True)  # Get class distribution

            balanced_indices = []
            for cls in unique_classes:
                class_indices = torch.where(predicted_labels == cls)[0]
                # Ensure each class gets a fair share of samples
                samples_per_class = self.batch_size // len(unique_classes)
                if len(class_indices) > samples_per_class:
                    # Select high-confidence samples first
                    class_confidence = confidence[class_indices]
                    top_indices = class_indices[torch.argsort(class_confidence, descending=True)[:samples_per_class]]
                    balanced_indices.extend(top_indices)
                else:
                    balanced_indices.extend(class_indices)

            balanced_indices = torch.tensor(balanced_indices)

            # Step 4: If more than 256, select the top 256 based on confidence (high confidence prioritized)
            if len(balanced_indices) > 256:
                class_confidence = confidence[balanced_indices]
                top_indices = balanced_indices[torch.argsort(class_confidence, descending=True)[:256]]
                balanced_indices = top_indices

            # Step 5: If less than 256, we randomly sample to reach 256
            if len(balanced_indices) < 256:
                remaining_indices = torch.tensor([i for i in range(self.size) if i not in balanced_indices])
                remaining_indices = remaining_indices[
                    torch.randperm(len(remaining_indices))[:256 - len(balanced_indices)]]
                balanced_indices = torch.cat([balanced_indices, remaining_indices])

            # Step 6: Replace old samples with new samples
            new_indices = torch.arange(fake.shape[0]).cpu()

            for i in range(256):
                old_index = balanced_indices[i]
                new_index = new_indices[i]

                # Replace old sample with new sample
                self.fakes[old_index] = fake[new_index]
                self.logits[old_index] = t_logit[new_index]

            # Update memory size
            self.size = min(self.size + fake.shape[0], self.max_length)

        self.head = tail % self.max_length

    def sample(self):
        """Return samples that are diverse, class-balanced, and challenging"""
        assert self.fakes is not None  # Only sample after having stored samples
        assert self.size >= self.batch_size  # Only sample if we have stored a full batch of samples
        idx = self._select_low_confidence_and_balanced_samples()
        return self.fakes[idx].to(self.device), self.logits[idx].to(self.device)

    # def _select_diverse_and_balanced_samples(self):
    #     """Select diverse, class-balanced, and challenging samples from memory"""
    #     # Step 1: Compute class labels from logits
    #     probs = F.softmax(self.logits, dim=-1)  # Convert logits to probabilities
    #     predicted_labels = torch.argmax(probs, dim=-1)  # Determine the class label of each sample
    #
    #     # Step 2: Compute sample difficulty (e.g., entropy as difficulty metric)
    #     entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1)  # [self.size]
    #
    #     # Step 3: Ensure class balance by selecting samples from each class
    #     selected_indices = []
    #     unique_classes = torch.unique(predicted_labels)
    #     samples_per_class = self.batch_size // len(unique_classes)  # Number of samples per class
    #
    #     for cls in unique_classes:
    #         # Get indices of samples belonging to the current class
    #         class_indices = torch.where(predicted_labels == cls)[0]
    #         class_entropies = entropy[class_indices]
    #
    #         # Sort by difficulty (entropy) and select top `samples_per_class` samples
    #         top_indices = class_indices[torch.argsort(class_entropies, descending=True)[:samples_per_class]]
    #         selected_indices.append(top_indices)
    #
    #     # Step 4: Flatten the list of indices and ensure diversity in selection
    #     selected_indices = torch.cat(selected_indices)  # Combine indices across all classes
    #
    #     # If selected samples are less than batch size (in case of rounding issues), sample more randomly
    #     if len(selected_indices) < self.batch_size:
    #         remaining_indices = torch.tensor([i for i in range(self.size) if i not in selected_indices])
    #         additional_indices = remaining_indices[
    #             torch.randperm(len(remaining_indices))[:self.batch_size - len(selected_indices)]]
    #         selected_indices = torch.cat([selected_indices, additional_indices])
    #
    #     return selected_indices

    def _select_low_confidence_and_balanced_samples(self):
        """从内存中选择低置信度、类别均衡的样本"""

        # Step 1: 将 logits 转换为概率并计算每个样本的预测类别标签
        probs = F.softmax(self.logits, dim=-1)  # 将 logits 转换为概率
        predicted_labels = torch.argmax(probs, dim=-1)  # 确定每个样本的预测类别

        # Step 2: 计算每个样本的置信度（即预测类别的概率值）
        confidence = torch.max(probs, dim=-1).values  # 获取每个样本的最高预测概率值（置信度）

        # Step 3: 计算低置信度样本，低置信度意味着预测类别的概率接近 0.5
        # 计算预测类别的概率与 0.5 的差距，越接近 0.5，置信度越低
        low_confidence = torch.abs(confidence - 0.5)  # 计算置信度接近 0.5 的程度，越小越不确定

        # Step 4: 确保类别均衡，从每个类别中选择低置信度样本
        selected_indices = []
        unique_classes = torch.unique(predicted_labels)  # 获取所有类别标签
        samples_per_class = self.batch_size // len(unique_classes)  # 每个类别选择的样本数量

        for cls in unique_classes:
            # 获取属于当前类别的样本索引
            class_indices = torch.where(predicted_labels == cls)[0]
            class_confidence = low_confidence[class_indices]

            # 按置信度（低置信度优先）排序，选择前 samples_per_class 个样本
            top_indices = class_indices[torch.argsort(class_confidence, descending=False)[:samples_per_class]]
            selected_indices.append(top_indices)

        # Step 5: 合并所有选择的样本索引，并确保所有张量在同一设备
        selected_indices = torch.cat(selected_indices)  # 合并所有类别的样本索引
        # selected_indices = selected_indices.to(self.device)

        # Step 6: 如果所选样本数量不足 batch_size，则从剩余样本中随机补充
        if len(selected_indices) < self.batch_size:
            remaining_indices = torch.tensor([i for i in range(self.size) if i not in selected_indices.cpu()])
            # remaining_indices = remaining_indices.to(self.device)  # 确保 remaining_indices 在同一设备
            additional_indices = remaining_indices[
                torch.randperm(len(remaining_indices))[:self.batch_size - len(selected_indices)]]
            # additional_indices = additional_indices.to(self.device)
            selected_indices = torch.cat([selected_indices, additional_indices])

        return selected_indices

    def new_epoch(self):
        """Does nothing for this type of experience replay"""
        pass

    def __len__(self):
        """Returns number of stored samples"""
        return self.size


class GeneratorMemory2(ReplayMemory):
    """用于生成数据的经验回放的循环FIFO缓冲区，包含类别平衡和难度加权。"""

    def __init__(self, device, length, batch_size):
        """初始化经验回放内存"""
        self.device = device
        self.max_length = length
        self.fakes = None  # 用于存储生成的假样本
        self.logits = None  # 用于存储对应的logits
        self.batch_size = batch_size
        self.size = 0  # 当前内存中的样本数量
        self.head = 0  # 内存的头部指针
        self.sample_history = torch.zeros(self.max_length, dtype=torch.bool)  # 记录哪些样本已被采样
        self.deleted_indices = []  # 用于跟踪已删除样本的索引

    def update(self, fake, t_logit):
        """使用生成的假样本和对应的logits更新内存"""
        fake = fake.cpu()
        t_logit = t_logit.cpu()
        if self.fakes is None:  # 如果内存还没有初始化
            self.fakes = torch.zeros((self.max_length, fake.shape[1], fake.shape[2], fake.shape[3]))
            self.logits = torch.zeros((self.max_length, t_logit.shape[1]))

        tail = self.head + fake.shape[0]
        if tail <= self.max_length:  # 内存未满
            self.fakes[self.head:tail] = fake
            self.logits[self.head:tail] = t_logit
            if self.size < self.max_length:
                self.size = tail
        else:  # 内存已满
            # 替换最近被采样的旧样本
            for i in range(fake.shape[0]):
                if self.deleted_indices:
                    # 如果有已删除的样本，用已删除的位置替换
                    replace_index = self.deleted_indices.pop(0)
                    self.fakes[replace_index] = fake[i]
                    self.logits[replace_index] = t_logit[i]
                else:
                    # 如果没有已删除的样本，就覆盖内存中最旧的位置
                    self.fakes[self.head] = fake[i]
                    self.logits[self.head] = t_logit[i]
                self.head = (self.head + 1) % self.max_length  # 更新头部指针

        self.head = tail % self.max_length  # 确保头部指针在内存大小范围内

    def sample(self):
        """返回多样性高、类别均衡、具有挑战性的样本"""
        assert self.fakes is not None  # 确保在存储了样本后才可以采样
        assert self.size >= self.batch_size  # 确保存储的样本数大于等于批量大小

        # 如果内存未满，随机选择样本
        if self.size < self.max_length:
            idx = torch.randperm(self.size)[:self.batch_size]  # 随机选择 batch_size 个样本
        else:
            # 内存已满时使用低置信度和类别均衡的采样方法
            idx = self._select_low_confidence_and_balanced_samples()  # 按照低置信度和类别均衡选择样本

        # 采样后，删除高置信度样本
        self._remove_high_confidence_samples(idx)

        # 返回采样的假图像和 logits
        return self.fakes[idx].to(self.device), self.logits[idx].to(self.device)

    def _select_low_confidence_and_balanced_samples(self):
        """从内存中选择低置信度、类别均衡的样本"""

        # Step 1: 将 logits 转换为概率并计算每个样本的预测类别标签
        probs = F.softmax(self.logits, dim=-1)  # 将 logits 转换为概率
        predicted_labels = torch.argmax(probs, dim=-1)  # 确定每个样本的预测类别

        # Step 2: 计算每个样本的置信度（即预测类别的概率值）
        confidence = torch.max(probs, dim=-1).values  # 获取每个样本的最高预测概率值（置信度）

        # Step 3: 计算低置信度样本，低置信度意味着预测类别的概率接近 0.5
        low_confidence = torch.abs(confidence - 0.5)  # 计算置信度接近 0.5 的程度，越小越不确定

        # Step 4: 确保类别均衡，从每个类别中选择低置信度样本
        selected_indices = []
        unique_classes = torch.unique(predicted_labels)  # 获取所有类别标签
        samples_per_class = self.batch_size // len(unique_classes)  # 每个类别选择的样本数量

        for cls in unique_classes:
            # 获取属于当前类别的样本索引
            class_indices = torch.where(predicted_labels == cls)[0]
            class_confidence = low_confidence[class_indices]

            # 按置信度（低置信度优先）排序，选择前 samples_per_class 个样本
            top_indices = class_indices[torch.argsort(class_confidence, descending=False)[:samples_per_class]]
            selected_indices.append(top_indices)

        # Step 5: 合并所有选择的样本索引
        selected_indices = torch.cat(selected_indices)  # 合并所有类别的样本索引

        # Step 6: 如果所选样本数量不足 batch_size，则从剩余样本中随机补充
        if len(selected_indices) < self.batch_size:
            remaining_indices = torch.tensor([i for i in range(self.size) if i not in selected_indices.cpu()])
            additional_indices = remaining_indices[
                torch.randperm(len(remaining_indices))[:self.batch_size - len(selected_indices)]]
            selected_indices = torch.cat([selected_indices, additional_indices])

        return selected_indices

    def _remove_high_confidence_samples(self, sampled_indices):
        """删除高置信度的样本"""

        # Step 1: 将 logits 转换为概率
        probs = F.softmax(self.logits, dim=-1)  # 将 logits 转换为概率
        confidence = torch.max(probs, dim=-1).values  # 获取每个样本的最高预测概率值（置信度）

        # Step 2: 定义高置信度样本的标准（比如置信度大于某个阈值的样本）
        high_confidence_threshold = 0.9  # 高置信度的阈值，可以根据需要调整
        high_confidence_indices = torch.where(confidence > high_confidence_threshold)[0]  # 获取高置信度样本的索引

        # Step 3: 从已采样的索引中去除高置信度样本
        remaining_indices = [idx for idx in sampled_indices if idx not in high_confidence_indices]

        # Step 4: 更新删除样本列表（记录已删除的样本索引）
        self.deleted_indices.extend(high_confidence_indices.cpu().tolist())  # 将删除的样本索引加入删除列表

        # Step 5: 返回去除高置信度样本后的采样索引
        return torch.tensor(remaining_indices).to(self.device)

    def new_epoch(self):
        """对于这种类型的经验回放，没有特定的操作"""
        pass

    def __len__(self):
        """返回存储的样本数量"""
        return self.size


class GeneratorMemory3(ReplayMemory):
    """Circular FIFO buffer based experiment replay. Returns uniform random samples when queried."""
    def __init__(self, device, length, batch_size):
        """Initialize replay memory"""
        self.device = device
        self.max_length = length
        self.fakes = None
        self.logits = None
        self.batch_size = batch_size
        self.size = 0
        self.head = 0

    def update(self, fake, t_logit):
        """Update memory with batch of features and labels"""
        fake = fake.cpu()
        t_logit = t_logit.cpu()
        if self.fakes is None:  # Initialize circular buffer if it hasn't already been.
            self.fakes = torch.zeros((self.max_length, fake.shape[1], fake.shape[2], fake.shape[3]))
            self.logits = torch.zeros((self.max_length, t_logit.shape[1]))
        tail = self.head + fake.shape[0]
        if tail <= self.max_length:  # Buffer is not full
            self.fakes[self.head:tail] = fake
            self.logits[self.head:tail] = t_logit
            if self.size < self.max_length:
                self.size = tail
        else:  # Buffer is full
            n = self.max_length - self.head
            self.fakes[self.head:tail] = fake[:n]
            self.logits[self.head:tail] = t_logit[:n]
            tail = tail % self.max_length
            self.fakes[:tail] = fake[n:]
            self.logits[:tail] = t_logit[n:]
            self.size = self.max_length
        self.head = tail % self.max_length

    def sample(self):
        """返回半混合采样：一半随机采样，一半低置信度和类别均衡采样"""
        assert self.fakes is not None  # 确保在存储了样本后才可以采样
        assert self.size >= self.batch_size  # 确保存储的样本数大于等于批量大小

        # 一半样本进行随机采样
        random_samples = self.size // 2  # 随机采样的一半样本数量
        random_indices = perm(self.size, random_samples, self.device).cpu()

        # 另一半样本考虑低置信度和类别均衡采样
        remaining_samples = self.batch_size - random_samples  # 需要从低置信度和类别均衡采样的样本数量
        low_confidence_indices = self._select_low_confidence_and_balanced_samples(remaining_samples)

        # 合并随机采样和低置信度采样的索引
        selected_indices = torch.cat([random_indices, low_confidence_indices])

        # 返回选中的样本
        return self.fakes[selected_indices].to(self.device), self.logits[selected_indices].to(self.device)

    def _select_low_confidence_and_balanced_samples(self, num_samples):
        """从内存中选择低置信度、类别均衡的样本"""
        # Step 1: 将 logits 转换为概率并计算每个样本的预测类别标签
        probs = F.softmax(self.logits, dim=-1)  # 将 logits 转换为概率
        predicted_labels = torch.argmax(probs, dim=-1)  # 确定每个样本的预测类别

        # Step 2: 计算每个样本的置信度（即预测类别的概率值）
        confidence = torch.max(probs, dim=-1).values  # 获取每个样本的最高预测概率值（置信度）

        # Step 3: 计算低置信度样本，低置信度意味着预测类别的概率接近 0.5
        low_confidence = torch.abs(confidence - 0.5)  # 计算置信度接近 0.5 的程度，越小越不确定

        # Step 4: 确保类别均衡，从每个类别中选择低置信度样本
        selected_indices = []
        unique_classes = torch.unique(predicted_labels)  # 获取所有类别标签
        samples_per_class = num_samples // len(unique_classes)  # 每个类别选择的样本数量

        for cls in unique_classes:
            # 获取属于当前类别的样本索引
            class_indices = torch.where(predicted_labels == cls)[0]
            class_confidence = low_confidence[class_indices]

            # 按置信度（低置信度优先）排序，选择前 samples_per_class 个样本
            top_indices = class_indices[torch.argsort(class_confidence, descending=False)[:samples_per_class]]

            # 将该类别的样本索引加入 selected_indices
            selected_indices.extend(top_indices.cpu().tolist())

        # Step 5: 将 selected_indices 转换为张量并确保在正确的设备上
        selected_indices = torch.tensor(selected_indices).to(self.device)

        # Step 6: 如果所选样本数量不足 num_samples，则从剩余样本中随机补充
        if len(selected_indices) < num_samples:
            remaining_indices = torch.tensor([i for i in range(self.size) if i not in selected_indices.cpu()])
            additional_indices = remaining_indices[
                torch.randperm(len(remaining_indices))[:num_samples - len(selected_indices)]]

            # 确保 additional_indices 在正确的设备上
            additional_indices = additional_indices.to(self.device)

            selected_indices = torch.cat([selected_indices, additional_indices])

        return selected_indices

    def new_epoch(self):
        """Does nothing for this type of experience replay"""
        pass

    def __len__(self):
        """Returns number of stored samples"""
        return self.size


# class HybridMemory(ReplayMemory):
#     """Circular FIFO buffer based experiment replay. Returns a mix of random and low-confidence, class-balanced samples when queried."""
#
#     def __init__(self, device, length, batch_size):
#         """Initialize replay memory"""
#         self.device = device
#         self.max_length = length
#         self.fakes = None
#         self.logits = None
#         self.batch_size = batch_size
#         self.size = 0
#         self.head = 0
#
#     def update(self, fake, t_logit):
#         """Update memory with batch of features and labels"""
#         fake = fake.cpu()
#         t_logit = t_logit.cpu()
#         if self.fakes is None:  # Initialize circular buffer if it hasn't already been.
#             self.fakes = torch.zeros((self.max_length, fake.shape[1], fake.shape[2], fake.shape[3]))
#             self.logits = torch.zeros((self.max_length, t_logit.shape[1]))
#         tail = self.head + fake.shape[0]
#         if tail <= self.max_length:  # Buffer is not full
#             self.fakes[self.head:tail] = fake
#             self.logits[self.head:tail] = t_logit
#             if self.size < self.max_length:
#                 self.size = tail
#         else:  # Buffer is full
#             n = self.max_length - self.head
#             self.fakes[self.head:tail] = fake[:n]
#             self.logits[self.head:tail] = t_logit[:n]
#             tail = tail % self.max_length
#             self.fakes[:tail] = fake[n:]
#             self.logits[:tail] = t_logit[n:]
#             self.size = self.max_length
#         self.head = tail % self.max_length
#
#     def sample(self):
#         """Return half random, half low-confidence and class-balanced samples"""
#         assert self.fakes is not None  # Ensure there are samples stored before sampling
#         assert self.size >= self.batch_size  # Ensure enough samples are stored for the batch size
#
#         # Half samples: Random selection
#         random_samples = self.batch_size // 2  # Number of samples to select randomly
#         random_indices = torch.randperm(self.size)[:random_samples].cpu()
#
#         # Half samples: Low confidence and class-balanced selection
#         remaining_samples = self.batch_size - random_samples
#         low_confidence_indices = self._select_low_confidence_and_balanced_samples(remaining_samples)
#
#         # Combine random and low-confidence/class-balanced samples
#         selected_indices = torch.cat([random_indices, low_confidence_indices])
#
#         # Return the selected samples
#         return self.fakes[selected_indices].to(self.device), self.logits[selected_indices].to(self.device)
#
#     def _select_low_confidence_and_balanced_samples(self, num_samples):
#         """Select low-confidence and class-balanced samples from memory"""
#         # Step 1: Convert logits to probabilities and compute the predicted labels
#         probs = F.softmax(self.logits, dim=-1)  # Convert logits to probabilities
#         predicted_labels = torch.argmax(probs, dim=-1)  # Get predicted labels
#
#         # Step 2: Compute confidence (the probability of the predicted class)
#         confidence = torch.max(probs, dim=-1).values  # Get the highest predicted probability (confidence)
#
#         # Step 3: Compute low-confidence samples, where confidence is close to 0.5
#         low_confidence = torch.abs(confidence - 0.5)  # Measure how close the confidence is to 0.5 (low confidence)
#
#         # Step 4: Ensure class balance by selecting low-confidence samples from each class
#         selected_indices = []
#         unique_classes = torch.unique(predicted_labels)  # Get all unique classes
#         samples_per_class = num_samples // len(unique_classes)  # Samples per class
#
#         for cls in unique_classes:
#             # Get indices of samples belonging to the current class
#             class_indices = torch.where(predicted_labels == cls)[0]
#             class_confidence = low_confidence[class_indices]
#
#             # Sort by low confidence (ascending) and select top samples_per_class samples
#             top_indices = class_indices[torch.argsort(class_confidence, descending=False)[:samples_per_class]]
#             selected_indices.append(top_indices)
#
#         # Step 5: Combine all selected indices
#         selected_indices = torch.cat(selected_indices)  # Combine indices across all classes
#
#         # Step 6: If the number of selected indices is less than the required, add random samples
#         if len(selected_indices) < num_samples:
#             remaining_indices = torch.tensor([i for i in range(self.size) if i not in selected_indices.cpu()])
#             additional_indices = remaining_indices[torch.randperm(len(remaining_indices))[:num_samples - len(selected_indices)]]
#             selected_indices = torch.cat([selected_indices, additional_indices])
#
#         return selected_indices
#
#     def new_epoch(self):
#         """Does nothing for this type of experience replay"""
#         pass
#
#     def __len__(self):
#         """Returns the number of stored samples"""
#         return self.size



# 222
class HybridMemory(ReplayMemory):
    """基于循环FIFO缓冲区的经验回放。返回混合的随机采样和低置信度、类别均衡的样本。"""

    def __init__(self, device, length, batch_size):
        """初始化回放内存"""
        self.device = device
        self.max_length = length
        self.fakes = None  # 存储生成的假图像
        self.logits = None  # 存储每个假图像的logits（预测结果）
        self.batch_size = batch_size
        self.size = 0  # 当前内存中有效样本数量
        self.head = 0  # 当前插入数据的头部位置

    def update(self, fake, t_logit):
        """更新内存，加入一批生成的特征和标签"""
        fake = fake.cpu()
        t_logit = t_logit.cpu()

        if self.fakes is None:  # 如果内存为空，初始化
            self.fakes = torch.zeros((self.max_length, fake.shape[1], fake.shape[2], fake.shape[3]))
            self.logits = torch.zeros((self.max_length, t_logit.shape[1]))

        tail = self.head + fake.shape[0]

        if tail <= self.max_length:  # 如果内存没有满
            self.fakes[self.head:tail] = fake
            self.logits[self.head:tail] = t_logit
            if self.size < self.max_length:
                self.size = tail
        else:  # 如果内存已满
            n = self.max_length - self.head
            self.fakes[self.head:tail] = fake[:n]
            self.logits[self.head:tail] = t_logit[:n]
            tail = tail % self.max_length
            self.fakes[:tail] = fake[n:]
            self.logits[:tail] = t_logit[n:]
            self.size = self.max_length

        self.head = tail % self.max_length

    def sample(self):
        """返回一半随机采样，一半低置信度和类别均衡采样"""
        assert self.fakes is not None  # 确保在存储了样本后才可以采样
        assert self.size >= self.batch_size  # 确保存储的样本数大于等于批量大小

        # 一半样本：随机选择
        random_samples = self.batch_size // 2  # 随机选择的样本数量
        random_indices = torch.randperm(self.size)[:random_samples].cpu()

        # 另一半样本：低置信度和类别均衡选择
        remaining_samples = self.batch_size - random_samples
        class_balanced_indices = self._select_class_balanced_samples(remaining_samples)

        # 合并随机选择和低置信度/类别均衡选择的样本
        selected_indices = torch.cat([random_indices, class_balanced_indices])

        # 返回选择的样本
        return self.fakes[selected_indices].to(self.device), self.logits[selected_indices].to(self.device)

    def _select_class_balanced_samples(self, num_samples):
        """从内存中选择类别均衡的样本"""
        # Step 1: 将 logits 转换为概率并计算预测标签
        probs = F.softmax(self.logits, dim=-1)  # 将 logits 转换为概率
        predicted_labels = torch.argmax(probs, dim=-1)  # 获取预测标签

        # Step 2: 按类别均衡选择样本
        selected_indices = []
        unique_classes = torch.unique(predicted_labels)  # 获取所有类别标签
        samples_per_class = num_samples // len(unique_classes)  # 每个类别选择的样本数量

        for cls in unique_classes:
            # 获取属于当前类别的样本索引
            class_indices = torch.where(predicted_labels == cls)[0]

            # 如果该类别样本数量不足，选择所有样本；否则随机选择 samples_per_class 个样本
            if len(class_indices) <= samples_per_class:
                selected_indices.append(class_indices)
            else:
                selected_indices.append(class_indices[torch.randperm(len(class_indices))[:samples_per_class]])

        # Step 3: 合并所有选择的样本索引
        selected_indices = torch.cat(selected_indices)

        # Step 4: 如果所选样本数少于所需数量，从剩余的样本中随机选择补足
        if len(selected_indices) < num_samples:
            remaining_indices = torch.tensor([i for i in range(self.size) if i not in selected_indices.cpu()])
            additional_indices = remaining_indices[
                torch.randperm(len(remaining_indices))[:num_samples - len(selected_indices)]]
            selected_indices = torch.cat([selected_indices, additional_indices])

        return selected_indices

    # def _select_class_balanced_and_low_confidence_samples(self, num_samples):
    #     """从内存中选择低置信度和类别均衡的样本"""
    #     # Step 1: 将 logits 转换为概率并计算预测标签
    #     probs = F.softmax(self.logits, dim=-1)  # 将 logits 转换为概率
    #     predicted_labels = torch.argmax(probs, dim=-1)  # 获取预测标签
    #
    #     # Step 2: 计算每个样本的置信度（即预测类别的概率值）
    #     confidence = torch.max(probs, dim=-1).values  # 获取最高的预测概率值（置信度）
    #
    #     # Step 3: 计算低置信度样本，低置信度意味着预测类别的概率接近 0.5
    #     low_confidence = torch.abs(confidence - 0.5)  # 测量置信度接近 0.5 的程度（低置信度）
    #
    #     # Step 4: 按类别均衡选择低置信度样本
    #     selected_indices = []
    #     unique_classes = torch.unique(predicted_labels)  # 获取所有类别标签
    #     samples_per_class = num_samples // len(unique_classes)  # 每个类别选择的样本数量
    #
    #     for cls in unique_classes:
    #         # 获取属于当前类别的样本索引
    #         class_indices = torch.where(predicted_labels == cls)[0]
    #         class_confidence = low_confidence[class_indices]
    #
    #         # 按照低置信度（从小到大）排序，选择前 samples_per_class 个样本
    #         top_indices = class_indices[torch.argsort(class_confidence, descending=False)[:samples_per_class]]
    #         selected_indices.append(top_indices)
    #
    #     # Step 5: 合并所有选择的样本索引
    #     selected_indices = torch.cat(selected_indices)
    #
    #     # Step 6: 如果所选样本数少于所需数量，从剩余的样本中随机选择
    #     if len(selected_indices) < num_samples:
    #         remaining_indices = torch.tensor([i for i in range(self.size) if i not in selected_indices.cpu()])
    #         additional_indices = remaining_indices[
    #             torch.randperm(len(remaining_indices))[:num_samples - len(selected_indices)]]
    #         selected_indices = torch.cat([selected_indices, additional_indices])
    #
    #     return selected_indices

    def new_epoch(self):
        """对于这种类型的经验回放，不做任何操作"""
        pass

    def __len__(self):
        """返回当前存储的样本数量"""
        return self.size


# class HybridMemory(ReplayMemory):
#     """混合采样的循环 FIFO 缓冲区，支持随机采样和类别均衡采样，并在内存满时删除多次调用的样本。"""
#
#     def __init__(self, device, length, batch_size):
#         """初始化回放内存"""
#         self.device = device
#         self.max_length = length  # 内存的最大容量
#         self.fakes = None  # 存储样本数据
#         self.logits = None  # 存储对应的logits
#         self.call_counts = None  # 记录每个样本被调用的次数
#         self.batch_size = batch_size  # 每次采样的批量大小
#         self.size = 0  # 当前内存中存储的样本数量
#         self.head = 0  # 循环缓冲区的起始位置
#
#     def update(self, fake, t_logit):
#         """更新内存，将新生成的样本存入"""
#         fake = fake.cpu()
#         t_logit = t_logit.cpu()
#
#         # 如果内存尚未初始化，则进行初始化
#         if self.fakes is None:
#             self.fakes = torch.zeros((self.max_length, fake.shape[1], fake.shape[2], fake.shape[3]))
#             self.logits = torch.zeros((self.max_length, t_logit.shape[1]))
#             self.call_counts = torch.zeros(self.max_length, dtype=torch.int32)
#
#         # 计算当前批次样本存储的结束位置
#         tail = self.head + fake.shape[0]
#         if tail <= self.max_length:  # 如果内存未满
#             self.fakes[self.head:tail] = fake
#             self.logits[self.head:tail] = t_logit
#             self.call_counts[self.head:tail] = 0  # 新样本调用次数置 0
#             if self.size < self.max_length:
#                 self.size = tail
#         else:  # 如果内存已满
#             n = self.max_length - self.head  # 计算头部剩余空间的大小
#             self.fakes[self.head:tail] = fake[:n]
#             self.logits[self.head:tail] = t_logit[:n]
#             self.call_counts[self.head:tail] = 0  # 新样本调用次数置 0
#             tail = tail % self.max_length
#             self.fakes[:tail] = fake[n:]
#             self.logits[:tail] = t_logit[n:]
#             self.call_counts[:tail] = 0  # 新样本调用次数置 0
#
#             # 删除多次调用的样本，腾出空间
#             self._remove_overused_samples()
#             self.size = self.max_length  # 更新当前内存大小
#
#         self.head = tail % self.max_length  # 更新头部指针位置
#
#     def sample(self):
#         """返回一半随机采样，一半类别均衡采样的样本"""
#         assert self.fakes is not None  # 确保内存中有样本
#         assert self.size >= self.batch_size  # 确保内存中样本数量不小于批量大小
#
#         # 一半样本随机采样
#         random_samples = self.batch_size // 2  # 随机采样的样本数量
#         random_indices = torch.randperm(self.size)[:random_samples].cpu()
#
#         # 另一半样本进行类别均衡采样
#         remaining_samples = self.batch_size - random_samples
#         class_balanced_indices = self._select_class_balanced_samples(remaining_samples)
#
#         # 合并随机采样和类别均衡采样的索引
#         selected_indices = torch.cat([random_indices, class_balanced_indices])
#
#         # 更新被调用样本的调用次数
#         self.call_counts[selected_indices] += 1
#
#         # 返回采样的样本和对应的 logits
#         return self.fakes[selected_indices].to(self.device), self.logits[selected_indices].to(self.device)
#
#     def _remove_overused_samples(self):
#         """当内存满时，删除多次调用的样本"""
#         # 按调用次数从大到小排序
#         sorted_indices = torch.argsort(self.call_counts, descending=True)
#
#         # 计算需要删除的样本数量
#         samples_to_remove = len(sorted_indices) - self.max_length
#
#         # 如果需要删除多次调用的样本
#         if samples_to_remove > 0:
#             indices_to_remove = sorted_indices[:samples_to_remove]
#
#             # 将多次调用的样本移动到末尾，释放空间
#             self.fakes = torch.cat([self.fakes[~indices_to_remove], self.fakes[indices_to_remove]])
#             self.logits = torch.cat([self.logits[~indices_to_remove], self.logits[indices_to_remove]])
#             self.call_counts = torch.cat([self.call_counts[~indices_to_remove], self.call_counts[indices_to_remove]])
#
#     def _select_class_balanced_samples(self, num_samples):
#         """从内存中按类别均衡采样"""
#         # 将 logits 转换为概率，并计算预测类别
#         probs = F.softmax(self.logits, dim=-1)
#         predicted_labels = torch.argmax(probs, dim=-1)
#
#         # 按类别均衡采样
#         selected_indices = []
#         unique_classes = torch.unique(predicted_labels)
#         samples_per_class = num_samples // len(unique_classes)
#
#         for cls in unique_classes:
#             class_indices = torch.where(predicted_labels == cls)[0]
#             selected_indices.append(class_indices[:samples_per_class])
#
#         selected_indices = torch.cat(selected_indices)
#         return selected_indices
#
#     def new_epoch(self):
#         """新训练周期时重置调用次数"""
#         self.call_counts[:] = 0
#
#     def __len__(self):
#         """返回当前内存中存储的样本数量"""
#         return self.size



def init_replay_memory(args):
    if args.replay == "Off":
        return NoReplayMemory()
    if args.replay == "Classic":
        return ClassicalMemory(args.device, args.replay_size, args.batch_size)
    if args.replay == "Hybrid":
        return HybridMemory(args.device, args.replay_size, args.batch_size)
    raise ValueError(f"Unknown replay parameter {args.replay}")