import torch
from sklearn.metrics import accuracy_score, classification_report
from tqdm import tqdm
import torchvision.transforms as transforms
from PIL import Image


def optimizer_picker(optimization, param, lr):
    if optimization == 'adam':
        optimizer = torch.optim.Adam(param, lr=lr)
    elif optimization == 'sgd':
        optimizer = torch.optim.SGD(param, lr=lr)
    else:
        print("automatically assign adam optimization function to you...")
        optimizer = torch.optim.Adam(param, lr=lr)
    return optimizer


def train_one_epoch(data_loader, model, criterion, optimizer, loss_mode, device):
    running_loss = 0
    model.train()
    for step, (batch_x, batch_y) in enumerate(tqdm(data_loader)):

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        optimizer.zero_grad()
        output = model(batch_x) # get predict label of batch_x
        
        loss = criterion(output, batch_y)

        loss.backward()
        optimizer.step()
        running_loss += loss
    return {
            "loss": running_loss.item() / len(data_loader),
            }

def evaluate_badnets(data_loader_val_clean, data_loader_val_poisoned, model, device):
    ta = eval(data_loader_val_clean, model, device, print_perform=True)
    asr = eval(data_loader_val_poisoned, model, device, print_perform=False)
    return {
            'clean_acc': ta['acc'], 'clean_loss': ta['loss'],
            'asr': asr['acc'], 'asr_loss': asr['loss'],
            }

def eval(data_loader, model, device, batch_size=64, print_perform=False):
    criterion = torch.nn.CrossEntropyLoss()
    model.eval() # switch to eval status
    y_true = []
    y_predict = []
    loss_sum = []
    for (batch_x, batch_y) in tqdm(data_loader):

        batch_x = batch_x.to(device, non_blocking=True)
        batch_y = batch_y.to(device, non_blocking=True)

        batch_y_predict = model(batch_x)
        loss = criterion(batch_y_predict, batch_y)
        batch_y_predict = torch.argmax(batch_y_predict, dim=1)
        y_true.append(batch_y)
        y_predict.append(batch_y_predict)
        loss_sum.append(loss.item())
    
    y_true = torch.cat(y_true,0)
    y_predict = torch.cat(y_predict,0)
    loss = sum(loss_sum) / len(loss_sum)

    if print_perform:
        print(classification_report(y_true.cpu(), y_predict.cpu(), target_names=data_loader.dataset.classes))

    return {
            "acc": accuracy_score(y_true.cpu(), y_predict.cpu()),
            "loss": loss,
            }

def predict_single_image(image_path, model, device, print_perform=False, class_names=None):
    """
    读取单张 PNG 图像，输入模型进行预测，并返回预测结果。

    参数:
    - image_path (str): PNG 图像的路径
    - model (torch.nn.Module): 已加载的 PyTorch 模型
    - device (str): 计算设备 ("cuda" 或 "cpu")
    - print_perform (bool): 是否打印预测类别
    - class_names (list, optional): 预测类别名称列表 (如 ["cat", "dog", "car", ...])

    返回:
    - result (dict): 包含预测类别索引、类别名称（如果提供 class_names）和置信度
    """
# # CIFAR数据集
#     # **1. 图像预处理**
#     transform = transforms.Compose([
#         transforms.Resize((32, 32)),  # 调整大小 (适应 CIFAR-10)
#         transforms.ToTensor(),  # 转换为 Tensor
#         transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])  # 归一化
#     ])

#     # **2. 加载图像**
#     image = Image.open(image_path).convert("RGB")  # 确保是 RGB 图像
#     image = transform(image).unsqueeze(0)  # 添加 batch 维度 (1, C, H, W)
#     image = image.to(device)  # 发送到 GPU 或 CPU

#     # **3. 进行模型预测**
#     model.eval()  # 进入评估模式
#     with torch.no_grad():  # 关闭梯度计算，提高推理速度
#         output = model(image)  # 前向传播
#         probabilities = torch.nn.functional.softmax(output, dim=1)  # 计算概率
#         pred_idx = torch.argmax(output, dim=1).item()  # 获取预测类别索引
#         confidence = probabilities[0, pred_idx].item()  # 获取置信度

#     # **4. 解析预测结果**
#     pred_class = class_names[pred_idx] if class_names else str(pred_idx)

#     # **5. 打印预测结果**
#     if print_perform:
#         print(f"图像: {image_path}")
#         print(f"预测类别: {pred_class} (索引: {pred_idx})")
#         print(f"置信度: {confidence:.4f}")

#     return {
#         "index": pred_idx,
#         "class": pred_class,
#         "confidence": confidence
#     }
    #MNIT数据集
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 转为单通道
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,))  # MNIST 常用的归一化
    ])
     # **2. 加载图像**
    image = Image.open(image_path).convert("RGB")  # 确保是 RGB 图像
    image = transform(image).unsqueeze(0)  # 添加 batch 维度 (1, C, H, W)
    image = image.to(device)  # 发送到 GPU 或 CPU

    # **3. 进行模型预测**
    model.eval()  # 进入评估模式
    with torch.no_grad():  # 关闭梯度计算，提高推理速度
        output = model(image)  # 前向传播
        probabilities = torch.nn.functional.softmax(output, dim=1)  # 计算概率
        pred_idx = torch.argmax(output, dim=1).item()  # 获取预测类别索引
        confidence = probabilities[0, pred_idx].item()  # 获取置信度

    # **4. 解析预测结果**
    pred_class = class_names[pred_idx] if class_names else str(pred_idx)

    # **5. 打印预测结果**
    if print_perform:
        print(f"图像: {image_path}")
        print(f"预测类别: {pred_class} (索引: {pred_idx})")
        print(f"置信度: {confidence:.4f}")

    return {
        "index": pred_idx,
        "class": pred_class,
        "confidence": confidence
    }
   