# Code to train a new classifier to use as teacher/victim from scratch.

import argparse
import torch
import torch.optim as optim
import torch.nn as nn
import network
import time
import os
from torch.cuda import amp

from dataloader import get_dataloader

# 创建检查点目录
os.makedirs("checkpoint/teacher", exist_ok=True)


def test(model, test_loader, device):
    """Test model on dataset. Returns mean top 1 accuracy"""
    model.eval()
    correct = 0
    count = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            count += data.shape[0]
    return correct / count


def train_epoch(model, dataloader, optimizer, device, loss_fn, scaler=None, grad_clip=1.0):
    """Train model for one epoch on training dataset"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for img, labels in dataloader:
        optimizer.zero_grad()
        data, target = img.to(device), labels.to(device)

        # 混合精度训练
        with amp.autocast(enabled=scaler is not None):
            preds = model(data)
            loss = loss_fn(preds, target)

        # 反向传播
        if scaler:
            scaler.scale(loss).backward()
            # 梯度裁剪
            if grad_clip > 0:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            # 梯度裁剪
            if grad_clip > 0:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        # 计算准确率
        _, predicted = preds.max(1)
        total += target.size(0)
        correct += predicted.eq(target).sum().item()

        total_loss += loss.item()

    epoch_loss = total_loss / len(dataloader)
    epoch_acc = correct / total
    return epoch_loss, epoch_acc


def train(model, train_loader, test_loader, optimizer, scheduler, args):
    """Outer training loop with improved monitoring and early stopping"""
    best_acc = 0
    epochs_no_improve = 0
    patience = 15  # 连续15个epoch无提升则早停

    # 使用标签平滑损失函数
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.15)

    # 混合精度训练
    scaler = amp.GradScaler(enabled=args.use_amp)

    start_time = time.time()

    for epoch in range(args.epochs):
        # 训练一个epoch
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, args.device, loss_fn, scaler, args.grad_clip
        )

        # 更新学习率
        scheduler.step()

        # 验证集评估
        val_acc = test(model, test_loader, args.device)

        # 记录最佳模型
        if val_acc > best_acc:
            best_acc = val_acc
            epochs_no_improve = 0
            torch.save(model.state_dict(), f"checkpoint/teacher/{args.dataset}-vit-best.pt")
            print(f"New best model saved with acc: {best_acc:.4f}")
        else:
            epochs_no_improve += 1

        # 打印训练信息
        current_lr = optimizer.param_groups[0]['lr']
        epoch_time = time.time() - start_time
        print(f"Epoch {epoch + 1}/{args.epochs} | "
              f"Time: {epoch_time:.1f}s | LR: {current_lr:.2e} | "
              f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | "
              f"Val Acc: {val_acc:.4f} | Best: {best_acc:.4f}")

        # 检查早停条件
        if epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch + 1} as no improvement in {patience} epochs")
            break

        # 检查目标准确率
        if val_acc >= args.max_accuracy:
            print(f"Reached target accuracy {args.max_accuracy} at epoch {epoch + 1}")
            break

    # 保存最终模型
    torch.save(model.state_dict(), f"checkpoint/teacher/{args.dataset}-vit-final.pt")
    print(f"Training completed. Best accuracy: {best_acc:.4f}")


def main():
    # 命令行参数解析
    parser = argparse.ArgumentParser(description="Optimized ViT Training Script")
    parser.add_argument('--device', type=int, default=0)
    parser.add_argument('--epochs', type=int, default=300)
    parser.add_argument('--batch-size', type=int, default=128)
    parser.add_argument('--max-accuracy', type=float, default=0.65)  # TinyImageNet合理目标
    parser.add_argument('--lr', type=float, default=0.0015)
    parser.add_argument('--weight-decay', type=float, default=0.01)
    parser.add_argument('--grad-clip', type=float, default=0.5, help="Gradient clipping value")
    parser.add_argument('--use-amp', action='store_true', default=True, help="Use mixed precision training")
    parser.add_argument('--suffix', type=str, default="")
    parser.add_argument('--data-root', type=str, default='data')
    parser.add_argument('--no-cuda', action='store_true', default=False, help='disables CUDA training')
    parser.add_argument('--dataset', type=str, default='tinyimagenet',
                        choices=['svhn', 'cifar10', 'mnist', 'cifar100', 'tinyimagenet'],
                        help='dataset name (default: tinyimagenet)')

    args = parser.parse_args()
    print("Running optimized ViT training script")

    # 设置设备
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    args.device = torch.device(f"cuda:{args.device}" if use_cuda else "cpu")
    print(f"Using device: {args.device}")

    # 确定类别数
    args.num_classes = {
        'svhn': 10,
        'cifar10': 10,
        'mnist': 10,
        'cifar100': 100,
        'tinyimagenet': 200
    }[args.dataset]

    # 创建模型
    model = network.T2T_ViT_7.T2TViT(
        img_size=32 if args.dataset != 'tinyimagenet' else 64,
        num_classes=args.num_classes,
        embed_dim=144,
        depth=7,
        num_heads=4,
        mlp_ratio=2.0,
        drop_rate=0.2
    )
    model = model.to(args.device)

    # 打印模型信息
    num_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Model created with {num_params / 1e6:.2f}M parameters")

    # 优化器设置 - 使用AdamW
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
        betas=(0.9, 0.999)
    )

    # 学习率调度 - 预热 + 余弦退火
    warmup_epochs = 10
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=warmup_epochs
            ),
            optim.lr_scheduler.CosineAnnealingLR(
                optimizer,
                T_max=args.epochs - warmup_epochs,
                eta_min=1e-6
            )
        ],
        milestones=[warmup_epochs]
    )

    args.input_space = "post-transform"
    # 获取数据加载器
    train_loader, test_loader, identity = get_dataloader(args)
    assert isinstance(identity, torch.nn.Identity)

    # 打印数据集信息
    print(f"Train dataset size: {len(train_loader.dataset)}")
    print(f"Test dataset size: {len(test_loader.dataset)}")
    print(f"Number of classes: {args.num_classes}")

    # 开始训练
    train(model, train_loader, test_loader, optimizer, scheduler, args)

if __name__ == '__main__':
    print("torch version", torch.__version__)
    main()