from __future__ import print_function

import json
import os
import random
import time

import numpy as np
from pprint import pprint

import pandas as pd
import torch
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter

from dataloader import get_dataloader
from cli_parser import parse_args
from BAM_Code.Config import Config
from BAM_Code.Utility import prepare_config_and_log, BAM_main_algorithm, generate_random_data_cifar_old
from metrics import eval_and_log_metrics, eval_and_log_validation_metrics, log_hparams, log_generator_distribution
from my_utils import *
from approximate_gradients import estimate_gradient_objective

from teacher import TeacherModel
from studentensemble import StudentEnsemble
from generatorensemble import GeneratorEnsemble
from replay import init_replay_memory
import config


def train_generator(args, generator, student_ensemble, teacher, device, optimizer):
    """Train generator model. Methodology is based on cli input args, especially the experiment-type parameter."""
    assert not teacher.training
    generator.train()
    student_ensemble.eval()

    g_loss_sum = 0
    for i in range(args.g_iter):
        optimizer.zero_grad()
        z = torch.randn((args.batch_size, args.nz)).to(device)
        if args.experiment_type == 'dfme':
            g_loss = dfme_gen_loss(args, z, generator, student_ensemble, teacher, device)
        elif args.experiment_type == 'mgct':
            g_loss = mgct_gen_loss(z, generator, student_ensemble, args)
        optimizer.step()
        g_loss_sum += g_loss
    return g_loss_sum / args.g_iter


def dfme_gen_loss(args, z, generator, student_ensemble, teacher, device):
    """Compute the generator loss for DFME method. Uses forward differences method. Update weights based on loss.
    See Also: https://github.com/cake-lab/datafree-model-extraction"""
    fake = generator(z, pre_x=args.approx_grad)  # pre_x returns the output of G before applying the activation

    # Estimate gradient for black box teacher
    approx_grad_wrt_x, loss_G = estimate_gradient_objective(args, teacher, student_ensemble, fake,
                                                            epsilon=args.grad_epsilon, m=args.grad_m,
                                                            device=device, pre_x=True)

    fake.backward(approx_grad_wrt_x)
    return loss_G.item()


def mgct_gen_loss(z, generator, student_ensemble, args):
    """Compute generator loss for MGCT method. Update weights based on loss.
    Calculates weighted average between disagreement loss and diversity loss."""
    fake = generator(z)

    preds = []
    for idx in range(student_ensemble.size()):
        preds.append(student_ensemble(fake, idx=idx))  # 2x [batch_size, K] Last dim is logits
    preds = torch.stack(preds, dim=1)  # [batch_size, 2, K]
    preds = F.softmax(preds, dim=2)  # [batch_size, 2, K] Last dim is confidence values.
    std = torch.std(preds, dim=1)  # std has shape [batch_size, K]. standard deviation over models
    loss_G = -torch.mean(std)  # Disagreement Loss
    if args.lambda_div != 0:
        soft_vote_mean = torch.mean(torch.mean(preds + 0.000001, dim=1),
                                    dim=0)  # [batch_size, 2, K] -> [batch_size, K] -> [K]
        loss_G += args.lambda_div * (torch.sum(soft_vote_mean * torch.log(soft_vote_mean)))  # Diversity Loss
    loss_G.backward()
    return loss_G.item()


def supervised_student_training(student_ensemble, fake, t_logit, optimizer, args):
    """Calculate loss and update weights for students in a supervised fashion"""
    student_iter_preds = []
    student_iter_loss = 0
    for i in range(student_ensemble.size()):
        s_logit = student_ensemble(fake, idx=i)
        with torch.no_grad():
            student_iter_preds.append(F.softmax(s_logit, dim=-1).detach())
        loss_s = student_loss(s_logit, t_logit, args)  # Helper function which handles soft- and hard-label settings

        if args.lambda_bd != 0:
            # 计算边界支持损失
            # 1. 获取学生模型和目标模型的预测类别
            T_pred = torch.argmax(t_logit, dim=1)
            S_pred = torch.argmax(s_logit, dim=1)
            # 2. 判断目标模型和学生模型的预测是否不一致
            disagree_mask = (T_pred != S_pred).float()
            # 3. 计算学生模型和目标模型输出的欧氏距离
            dist = F.pairwise_distance(t_logit, s_logit, p=2)
            # 4. 计算边界支持损失，并加到学生模型损失中
            loss_bd = (dist * disagree_mask).mean()
            # 总损失是学生损失和边界支持损失的和
            loss_s = loss_s + args.lambda_bd * loss_bd
        loss_s.backward()
        student_iter_loss += loss_s.item()
    optimizer.step()
    return torch.stack(student_iter_preds, dim=1), student_iter_loss


def train_student_ensemble(args, student_ensemble, teacher, device, optimizer, replay_memory, gen_dataloader):
    """Train student ensemble with a fixed generator"""
    assert not teacher.training
    student_ensemble.train()

    s_loss_sum = 0
    student_preds = []
    teacher_preds = []
    for d_iter in range(args.d_iter):  # Generate and train for d_iter batches. Store batches to experience replay
        optimizer.zero_grad()
        real_data = real_data.to(device, non_blocking=True)  # Move data to device
        labels = labels.to(device, non_blocking=True)  # Move labels to device
        replay_memory.update(real_data.cpu(), labels.cpu())  # Store data and labels to experience replay
        student_iter_preds, student_iter_loss = supervised_student_training(student_ensemble, real_data, labels,
                                                                            optimizer, args)  # Train students

        teacher_preds.append(labels.detach())  # Store true labels for logging purposes
        student_preds.append(student_iter_preds)  # Store student predictions for logging purposes
        s_loss_sum += student_iter_loss  # Store student loss for logging purposes

    # 不使用经验回放
    # for _ in range(args.rep_iter):  # Train for rep_iter batches on samples from experience replay.
    #     optimizer.zero_grad()
    #     fake, t_logit = replay_memory.sample()  # Sample features and labels from experience replay.
    #     fake.to(device)                         # Load features to device
    #     t_logit.to(device)                      # Load labels/teacher predictions to device
    #     student_iter_preds, student_iter_loss = supervised_student_training(student_ensemble, fake, t_logit,
    #                                                                             optimizer, args)  # Train students
    #     teacher_preds.append(F.softmax(t_logit, dim=-1).detach())  # Store teacher predictions for logging purposes
    #     student_preds.append(student_iter_preds)                   # Store student predictions for logging purposes
    #     s_loss_sum += student_iter_loss                            # Store student loss for logging purposes

    # Prep student and teacher preds for logging purposes
    teacher_preds = torch.cat(teacher_preds, dim=0)
    student_preds = torch.cat(student_preds, dim=0)

    return student_preds, teacher_preds, s_loss_sum / (args.d_iter * student_ensemble.size())


# def log_training_epoch(args, g_loss, s_loss, i, epoch):
#     """Somewhat dated function for logging training data to command line and logfile"""
#     if i % args.log_interval != 0:
#         return
#     print_and_log(f'Train Epoch: {epoch} [{i} ({100*float(i)/float(args.epoch_itrs):.0f}%)] '
#                   f'Generator Loss:{g_loss} Student Loss:{s_loss}')

def log_training_epoch(args, s_loss, epoch):
    """Somewhat dated function for logging training data to command line and logfile"""
    print_and_log(f'Train Epoch: {epoch} '
                  f'Student Loss:{s_loss}')


def train_epoch_ensemble(args, generator, student_ensemble, teacher, device,
                         optimizer_student, optimizer_generator, epoch, replay_memory, gen_dataloader):
    """Runs alternating generator and student training iterations.
    Also verifies queries counts match up with theoretically expected values."""
    student_preds = None
    teacher_preds = None
    # 默认训练50轮
    for i in range(args.epoch_itrs):
        # 训练生成器第一阶段
        # g_loss_pre = train_generator_pre(args, generator, device, optimizer_generator)
        # 训练生成器,
        # 暂时注释
        # g_loss = train_generator(args, generator, student_ensemble, teacher, device, optimizer_generator)
        # 训练学生模型（n个）
        student_preds, teacher_preds, s_loss = train_student_ensemble(args, student_ensemble, teacher,
                                                                      device, optimizer_student, replay_memory,
                                                                      gen_dataloader)

        # Update query budget based on theoretically expected values. Then verify it matches up with actual value.
        args.query_budget -= args.cost_per_iteration
        assert (
                args.query_budget + args.current_query_count == args.total_query_budget), f"{args.query_budget} + {args.current_query_count}"
        # 暂时修改g_loss
        # log_training_epoch(args, s_loss, i, epoch)  # Command line logging of training state.
        if args.query_budget < args.cost_per_iteration:  # End training if we cannot complete full iteration
            break
    return student_preds, teacher_preds


def log_test_metrics(model, test_loader, teacher_test_preds, device, args):
    if not isinstance(model, StudentEnsemble):
        print_and_log(f"log_test_metrics currently only supports Ensemble. Detected class:{model.__class__}")
        raise NotImplementedError

    preds, labels = get_model_preds_and_true_labels(model, test_loader, device)

    stats = eval_and_log_validation_metrics(config.tboard_writer, preds, teacher_test_preds, args, tag="Fidelity")
    print_and_log(
        'Fidelity=> Soft Vote:{:.4f}%, Hard Vote:{:.4f}%, Es Median/Min/Max:{:.4f}%/{:.4f}%/{:.4f}%\n'.format(
            100 * stats["Soft Vote"]["Accuracy"],
            100 * stats["Hard Vote"]["Accuracy"],
            100 * stats["Ensemble"]["Accuracy"]["Median"],
            100 * stats["Ensemble"]["Accuracy"]["Min"],
            100 * stats["Ensemble"]["Accuracy"]["Max"]))
    stats = eval_and_log_metrics(config.tboard_writer, preds, labels, args)
    print_and_log(
        'Accuracies=> Soft Vote:{:.4f}%, Hard Vote:{:.4f}%, Es Median/Min/Max:{:.4f}%/{:.4f}%/{:.4f}%\n'.format(
            100 * stats["Soft Vote"]["Accuracy"],
            100 * stats["Hard Vote"]["Accuracy"],
            100 * stats["Ensemble"]["Accuracy"]["Median"],
            100 * stats["Ensemble"]["Accuracy"]["Min"],
            100 * stats["Ensemble"]["Accuracy"]["Max"]))

    return stats["Soft Vote"]["Accuracy"]


def get_model_accuracy_and_loss(args, model, loader, device="cuda"):
    """Get model accuracy and loss. Simple function intended for simply CLI printing. Prefer using metrics.py"""
    model.eval()
    correct, loss = 0, 0
    with torch.no_grad():
        for i, (data, target) in enumerate(loader):
            data, target = data.to(device), target.to(device)
            output = model(data)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()
            loss += F.cross_entropy(output, target, reduction='sum').item()  # sum up batch
    loss /= len(loader.dataset)
    accuracy = correct / len(loader.dataset)
    return accuracy, loss


def init_logs(args):
    """Init log files. Mostly legacy behavior from DFME codebase."""
    os.makedirs(args.log_dir, exist_ok=True)

    # Save JSON with parameters
    with open(args.log_dir + "/parameters.json", "w") as f:
        json.dump(vars(args), f)

    init_or_append_to_log_file(args.log_dir, "loss.csv", "epoch,loss_G,loss_S")
    init_or_append_to_log_file(args.log_dir, "accuracy.csv", "epoch,accuracy")
    init_or_append_to_log_file(os.getcwd(), "latest_experiments.txt",
                               args.experiment_name + ":" + args.log_dir, mode="a")

    if args.rec_grad_norm:
        init_or_append_to_log_file(args.log_dir, "norm_grad.csv", "epoch,G_grad_norm,S_grad_norm,grad_wrt_X")


def main():
    # Parse command line arguments and set certain variables based on those
    args = parse_args(set_total_query_budget=True)
    # Print log directory
    print(args.log_dir)

    # Prepare the environment
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.backends.cudnn.deterministic = args.cudnn_deterministic
    torch.backends.cudnn.benchmark = False

    # Load test_loader and transform functions based on CLI args. Ignore train_loader.
    _, test_loader, normalization = get_dataloader(args)
    print(f"\nLoaded {args.dataset} successfully.")
    # Display distribution information of dataset
    # histogram表示区间内的数据量，histogram_bin_edges表示区间的边界
    print_dataset_feature_distribution(data_loader=test_loader)

    # Initialize log files and directories
    init_logs(args)
    args.model_dir = f"{args.experiment_dir}/student_{args.model_id}"
    if not os.path.exists(args.model_dir):
        os.makedirs(args.model_dir)
    with open(f"{args.model_dir}/model_info.txt", "w") as f:
        json.dump(args.__dict__, f, indent=2)
    config.log_file = open(f"{args.model_dir}/logs.txt", "w")

    # Set compute device
    use_cuda = torch.cuda.is_available() and not args.no_cuda
    args.device = torch.device("cuda:%d" % args.device if use_cuda else "cpu")
    print(f"Device is {args.device}")

    # Initialize tensorboard logger. Metrics are handled in metrics.py
    config.tboard_writer = SummaryWriter(f"tboard/general/{args.experiment_name}")

    args.normalization_coefs = None
    args.G_activation = torch.tanh

    # Set default number of classes. This will be moved to the CLI parser, eventually
    num_classes = 10 if args.dataset in ['cifar10', 'svhn', 'mnist'] else 200 if args.dataset == 'tinyimagenet' else 100
    num_channels = 1 if args.dataset in ['mnist'] else 3
    args.num_classes = num_classes

    pprint(args, width=80)

    # Init teacher
    if args.model == 'resnet34_8x':
        teacher = network.resnet_8x.ResNet34_8x(num_classes=num_classes)

        args.ckpt = 'checkpoint/teacher/' + args.dataset + '-resnet34_8x.pt'
        teacher.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    elif args.model == 'resnet18_8x':
        teacher = network.resnet_8x.ResNet18_8x(num_classes=num_classes)

        args.ckpt = 'checkpoint/teacher/' + args.dataset + '-resnet18_8x.pt'
        teacher.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    elif args.model == 'alexnet32':
        teacher = network.alexnet.AlexNet32(num_classes=num_classes)
        args.ckpt = 'checkpoint/teacher/' + args.dataset + '-alexnet.pt'
        teacher.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    elif args.model == 'vgg16':
        teacher = network.VGG.vgg16(num_classes=num_classes)
        args.ckpt = 'checkpoint/teacher/' + args.dataset + '-vgg16.pt'
        teacher.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    elif args.model == 'lenet':
        teacher = network.lenet.LeNet5(num_classes=num_classes)
        args.ckpt = 'checkpoint/teacher/' + args.dataset + '-lenet.pt'
        teacher.load_state_dict(torch.load(args.ckpt, map_location=args.device))

    elif args.model == 'lenet5':
        teacher = network.lenet.LeNet5()

        args.ckpt = 'checkpoint/teacher/' + args.dataset + '-lenet5.pt'
        teacher.load_state_dict(torch.load(args.ckpt, map_location=args.device))
    else:
        teacher = get_classifier(args.model, pretrained=True, num_classes=args.num_classes)

    # Wrap teacher model in a handler class together with the data transform
    teacher = TeacherModel(teacher, transform=normalization)
    teacher.eval()
    config.tboard_writer.add_graph(teacher, torch.rand((32, num_channels, 32, 32)))
    teacher = teacher.to(args.device)

    # Evaluate teacher on test dataset to verify accuracy is in expected range
    print_and_log("Teacher restored from %s" % (args.ckpt))
    print(f"\n\t\tTraining with {args.model} as a Target\n")
    accuracy, _ = get_model_accuracy_and_loss(args, teacher, test_loader, args.device)
    print('\nTeacher - Test set: Accuracy: {}/{} ({:.4f}%)\n'.format(np.round(accuracy * len(test_loader.dataset)),
                                                                     len(test_loader.dataset), accuracy))

    # Initialize a fresh student ensemble. Ensemble may be of size 1.
    student = get_classifier(args.student_model, pretrained=False, num_classes=args.num_classes,
                             ensemble_size=args.ensemble_size)
    for i in range(args.ensemble_size):
        config.tboard_writer.add_graph(student.get_model_by_idx(i), torch.rand((32, num_channels, 32, 32)))
    student = student.to(args.device)

    # Initialize generator
    # 原来一个生成器，现在是一个生成器集合
    generator = network.gan.GeneratorA(nz=args.nz, nc=num_channels, img_size=32, activation=args.G_activation,
                                       grayscale=args.grayscale)
    # generator = GeneratorEnsemble(nz=args.nz, num_channels=num_channels, img_size=32, G_activation=args.G_activation,
    #                               grayscale=args.grayscale, ensemble_size=2)
    # config.tboard_writer.add_graph(generator, torch.rand((32, args.nz)))
    generator = generator.to(args.device)

    args.generator = generator
    args.student = student
    args.teacher = teacher

    # Compute theoretical query cost per training iteration. This will be compared with true value to verify correctness
    args.cost_per_iteration = compute_cost_per_iteration(args)
    number_epochs = args.query_budget // (args.cost_per_iteration * args.epoch_itrs) + 1

    print(f"\nTotal budget: {args.query_budget // 1000}k")
    print("Cost per iterations: ", args.cost_per_iteration)
    print("Total number of epochs: ", number_epochs)

    # Init optimizers for student and generator
    optimizer_S = optim.SGD(student.parameters(), lr=args.lr_S, weight_decay=args.weight_decay, momentum=0.9)
    optimizer_G = optim.Adam(generator.parameters(), lr=args.lr_G)

    # Compute learning rate drop iterations based on input percentages and iteration count.
    steps = sorted([int(step * number_epochs) for step in args.steps])
    print("Learning rate scheduling at steps: {}\n".format(steps))

    if args.scheduler == "multistep":
        scheduler_S = optim.lr_scheduler.MultiStepLR(optimizer_S, steps, args.scale)
        scheduler_G = optim.lr_scheduler.MultiStepLR(optimizer_G, steps, args.scale)
    elif args.scheduler == "cosine":
        scheduler_S = optim.lr_scheduler.CosineAnnealingLR(optimizer_S, number_epochs)
        scheduler_G = optim.lr_scheduler.CosineAnnealingLR(optimizer_G, number_epochs)

    best_acc = 0
    acc_list = []
    # 回放池暂时注销
    # replay_memory = init_replay_memory(args)
    teacher_test_preds, _ = get_model_preds_and_true_labels(teacher, test_loader, args.device)

    # Accuracy milestones to log
    accuracy_goals = {0.75: 0, 0.8: 0, 0.85: 0, 0.9: 0, 0.95: 0}

    # 新增 BAM算法
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    prepare_config_and_log()
    config1 = Config.instance
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_of_classes = config1["num_of_classes"]
    name = "teacher_resnet18_for_cifar10"
    ckpt_path = "teacher_resnet18_for_cifar10_state_dict"
    # alex = Alexnet(name, num_of_classes)
    # alex.load_state_dict(torch.load(ckpt_path, map_location=device))
    # alex.to(device)
    # print(f"Accuracy of target model is:")
    # alex.test_model()
    k = config1["k"]
    epsilon = config1["epsilon"]
    population_size = config1["population_size"]
    generations = config1["generations"]
    search_spread = config1["search_spread"]
    epochs = config1["epochs"]
    dont_get_from_dist = config1["dont_get_from_disk"]
    criterion = nn.MSELoss().to(device)
    optimizer_name = config1["optimizer_name"]
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    # 使用遗传算法生成数据加载器
    gen_dataloader = BAM_main_algorithm(
        teacher,
        student,
        criterion,
        generate_random_data_cifar_old,
        num_of_classes,
        k,
        epsilon,
        population_size,
        generations,
        search_spread,
        epochs,
        optimizer_name,
        name=name,
    )

    # # Outer training loop.
    # for epoch in range(1, number_epochs + 1):
    #     print_and_log(f"{args.experiment_name} epoch {epoch}")
    #
    #     config.tboard_writer.add_scalar('Param/student_learning_rate', scheduler_S.get_last_lr()[0], args.current_query_count)
    #     # config.tboard_writer.add_scalar('Param/generator_learning_rate', scheduler_G.get_last_lr()[0],
    #     #                          args.current_query_count)
    #
    #     # Inner training loop call
    #     student_preds, teacher_preds = train_epoch_ensemble(args, generator, student, teacher, args.device,
    #                                                         optimizer_S, optimizer_G, epoch, replay_memory, gen_dataloader)
    #     if args.scheduler != "none":
    #         scheduler_S.step()
    #         scheduler_G.step()
    #     replay_memory.new_epoch()

    # Outer training loop.

    # for epoch in range(1, number_epochs + 1):
    for epoch, (real_data, labels) in enumerate(gen_dataloader):  # Iterate over gen_dataloader for up to d_iter batches
        # 把real_data的shape改为[batch_size, 3, 32, 32]
        real_data = real_data.squeeze(1)  # 移除单例维度
        print_and_log(f"{args.experiment_name} epoch {epoch}")

        config.tboard_writer.add_scalar('Param/student_learning_rate', scheduler_S.get_last_lr()[0],
                                        args.current_query_count)
        # config.tboard_writer.add_scalar('Param/generator_learning_rate', scheduler_G.get_last_lr()[0],
        #                          args.current_query_count)

        # Inner training loop
        student_preds = []
        teacher_preds = []
        s_loss_sum = 0

        optimizer_S.zero_grad()
        real_data = real_data.to(device, non_blocking=True)  # Move data to device
        labels = labels.to(device, non_blocking=True)  # Move labels to device

        student_iter_preds, student_iter_loss = supervised_student_training(student, real_data, labels,
                                                                            optimizer_S, args)  # Train students

        teacher_preds.append(labels.detach())  # Store true labels for logging purposes
        student_preds.append(student_iter_preds)  # Store student predictions for logging purposes
        s_loss_sum += student_iter_loss  # Store student loss for logging purposes
        s_loss = s_loss_sum / student.size()

        log_training_epoch(args, s_loss, epoch)  # Command line logging of training state.

        if args.scheduler != "none":
            scheduler_S.step()
            scheduler_G.step()

        # Test and log
        acc = log_test_metrics(student, test_loader, teacher_test_preds, args.device, args)
        for goal in accuracy_goals:
            if accuracy_goals[goal] == 0 and acc > goal:
                accuracy_goals[goal] = args.current_query_count / 1000000.0
                print_and_log(f"Reached {goal} accuracy goal in {accuracy_goals[goal]}m queries")

        # Prep student and teacher preds for logging purposes
        teacher_preds = torch.cat(teacher_preds, dim=0)
        # 减去第二个维度
        teacher_preds = teacher_preds.squeeze(1)  # Remove singleton dimension
        student_preds = torch.cat(student_preds, dim=0)

        eval_and_log_validation_metrics(config.tboard_writer, student_preds, teacher_preds, args)
        log_generator_distribution(config.tboard_writer, teacher_preds, args)

        acc_list.append(acc)
        # Store models
        if acc > best_acc:
            if not os.path.exists(f"{args.experiment_dir}/model_checkpoints"):
                os.makedirs(f"{args.experiment_dir}/model_checkpoints")
            best_acc = acc
            torch.save(student.state_dict(), f"{args.experiment_dir}/model_checkpoints/student_best.pt")
            torch.save(generator.state_dict(), f"{args.experiment_dir}/model_checkpoints/generator_best.pt")
        if epoch % args.generator_store_frequency == 0:
            if not os.path.exists(f"{args.experiment_dir}/model_checkpoints/generators"):
                os.makedirs(f"{args.experiment_dir}/model_checkpoints/generators")
            torch.save(generator.state_dict(),
                       f"{args.experiment_dir}/model_checkpoints/generators/generator_{args.current_query_count}.pt")
    log_hparams(config.tboard_writer, args)
    for goal in sorted(accuracy_goals):
        print_and_log(f"Goal {goal}: {accuracy_goals[goal]}")
    print_and_log("Best Acc=%.6f" % best_acc)


if __name__ == '__main__':
    # 打印当前时间
    print(f"Start time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
    main()
    print(f"End time: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime())}")
