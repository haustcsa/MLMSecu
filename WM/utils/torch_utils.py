import os
import torch
import random
import numpy as np
from config import training_config as cfg
from pytorch_wavelets import DTCWTForward, DTCWTInverse
from torchvision import transforms



def seed_torch(seed=42):
    seed = int(seed)
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.enabled = True


def images_U_dtcwt_with_low(images_U):
    xfm = DTCWTForward(J=2, biort="near_sym_b", qshift="qshift_b").to(cfg.device)
    low_pass, high_pass = xfm(images_U)
    return low_pass, high_pass


def dtcwt_images_U(low_pass, high_pass):
    ifm = DTCWTInverse(biort="near_sym_b", qshift="qshift_b").to(cfg.device)
    return ifm((low_pass, high_pass))


def images_U_dtcwt_without_low(images_U):
    xfm = DTCWTForward(J=2, biort="near_sym_b", qshift="qshift_b").to(cfg.device)
    _, high_pass = xfm(images_U)
    return high_pass


def decoded_message_error_rate(message, decoded_message):
    length = message.shape[0]

    message = message.gt(0)
    decoded_message = decoded_message.gt(0)
    error_rate = float((message != decoded_message).sum().item()) / length
    return error_rate


def decoded_message_error_rate_batch(messages, decoded_messages):
    error_rate = 0.0
    batch_size = len(messages)
    for i in range(batch_size):
        error_rate += decoded_message_error_rate(messages[i], decoded_messages[i])
    error_rate /= batch_size
    return torch.tensor(error_rate).to(cfg.device)



def decoded_message_error_rate_batch(messages, decoded_messages):
    """
    计算批量水印的平均错误率。
    Args:
        messages: 2D 张量，形状为 [batch_size, length]
        decoded_messages: 2D 张量，形状为 [batch_size, length]
    Returns:
        error_rate: 张量，平均错误率
    """
    # assert messages.dim() == 2, "Expected 2D tensor for messages"
    # assert decoded_messages.dim() == 2, "Expected 2D tensor for decoded_messages"
    # assert messages.shape == decoded_messages.shape, "Shape mismatch"
    # assert messages.device == cfg.device, "Messages must be on cfg.device"
    # assert decoded_messages.device == cfg.device, "Decoded messages must be on cfg.device"

    messages = messages.gt(0)
    decoded_messages = decoded_messages.gt(0)
    error_rate = (messages != decoded_messages).float().mean(dim=1).mean()
    return error_rate.to(cfg.device)