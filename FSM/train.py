import os
import torch
import time
import platform
import argparse
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import confusion_matrix, roc_auc_score
from sklearn.preprocessing import label_binarize
from torch.optim.lr_scheduler import ReduceLROnPlateau, LambdaLR
from torch.utils.data import DataLoader
from torch.optim import lr_scheduler
from network.MainNet import MainNet
from network.data import *
from network.transform import Data_Transforms
from datetime import datetime
import torch.nn.functional as F
from network.log_record import *
from network.pipeline import *
from network.utils import setup_seed, cal_metrics, plot_ROC



def main():
    args = parse.parse_args()

    name = args.name
    train_txt_path = args.train_txt_path
    valid_txt_path = args.valid_txt_path
    continue_train = args.continue_train
    epoches = args.epoches
    batch_size = args.batch_size
    model_name = args.model_name
    model_path = args.model_path
    num_classes = args.num_classes

    output_path = os.path.join('./output', name)
    if not os.path.exists(output_path):
        os.mkdir(output_path)

    log_path = os.path.join(output_path)

    now_time = datetime.now()
    time_str = datetime.strftime(now_time, '%m-%d_%H-%M-%S')
    print('Training datetime: ', time_str)

    # -----create train&val data----- #
    train_data = SingleInputDataset(txt_path=train_txt_path, train_transform=Data_Transforms['train'])
    valid_data = SingleInputDataset(txt_path=valid_txt_path, valid_transform=Data_Transforms['val'])
    train_loader = DataLoader(dataset=train_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)
    valid_loader = DataLoader(dataset=valid_data, batch_size=batch_size, shuffle=True, num_workers=8, pin_memory=True)

    # -----create the model----- #
    setup_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = MainNet(num_classes).to(device)

    # -----calculate FLOPs and Params----- #
    flops, params = cal_params_ptflops(model, (3, 256, 256))
    print('{:<30}  {:<8}'.format('Computational complexity (FLOPs): ', flops))
    print('{:<30}  {:<8}'.format('Number of parameters (Params): ', params))

    if continue_train:
        print('继续训练====================')
        model.load_state_dict(torch.load(model_path, map_location='cpu'))

    # -----define the loss----- #
    criterion = nn.CrossEntropyLoss()

    # -----define the optimizer----- #
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

    def lr_lambda(epoch):
        if epoch < 15:
            return 1.0
        elif 15 <= epoch < 21:
            return 0.5
        elif 21 <= epoch < 25:
            return 0.25
        elif 25 <= epoch < 28:
            return 0.125
        else:
            return 0.0625

    scheduler = LambdaLR(optimizer, lr_lambda=lr_lambda)

    # -----------------------------------------define the train & val----------------------------------------- #
    best_acc = 0.0
    best_auc = 0.0
    time_open = time.time()

    for epoch in range(epoches):
        total_train_samples = 0.0
        correct_tra = 0.0
        sum_loss_tra = 0.0

        print('\nEpoch {}/{}'.format(epoch + 1, epoches))
        print('-' * 10)
        current_lr = optimizer.param_groups[0]['lr']
        print(f"Current learning rate: {current_lr}")

        # -----Training----- #
        for i, data in enumerate(train_loader):
            img_train, labels_train = data

            # input images & labels
            img_train = img_train.cuda(0)
            labels_train = labels_train.cuda(0)

            optimizer.zero_grad()
            model = model.train()

            # feed data to model
            pre_tra = model(img_train)

            # the average loss of a batch
            loss_tra = criterion(pre_tra, labels_train)
            sum_loss_tra += loss_tra.item() * labels_train.size(0)

            # prediction
            _, pred = torch.max(pre_tra.data, 1)
            loss_tra.backward()
            optimizer.step()

            # the correct number of prediction
            correct_tra += (pred == labels_train).squeeze().sum().cpu().numpy()

            # the number of all training samples
            total_train_samples += labels_train.size(0)

            # training information is printed every 100 iterations
            if i % 100 == 99:
                print("Training: Epoch[{:0>1}/{:0>1}] Iteration[{:0>1}/{:0>1}] Loss:{:.2f} Acc:{:.2%}".format(
                    epoch + 1, epoches, i + 1, len(train_loader), sum_loss_tra / total_train_samples,
                    correct_tra / total_train_samples))

        # -----Validating----- #
        if epoch % 1 == 0:
            sum_loss_val = 0.0
            correct_val = 0.0
            total_valid_samples = 0.0
            label_val_list = []
            predict_val_list = []
            y_true = []
            y_pred = []
            correct_class_count = np.zeros(num_classes)
            total_class_count = np.zeros(num_classes)

            confusion_matrix_class = np.zeros((num_classes, num_classes))
            model.eval()

            with torch.no_grad():
                for i, data in enumerate(valid_loader):
                    img_valid, labels_valid = data

                    img_valid = img_valid.cuda(0)
                    labels_valid = labels_valid.cuda(0)

                    pre_val = model(img_valid)

                    loss_val = criterion(pre_val, labels_valid)
                    sum_loss_val += loss_val.item() * labels_valid.size(0)

                    _, pred = torch.max(pre_val.data, 1)

                    total_valid_samples += labels_valid.size(0)

                    correct_val += (pred == labels_valid).squeeze().sum().cpu().numpy()
                    y_true.extend(labels_valid.cpu().numpy())
                    y_pred.extend(pred.cpu().numpy())
                    pre_val_probs = torch.nn.functional.softmax(pre_val, dim=1)
                    label_val_list.extend(pre_val_probs.cpu().numpy())

                val_loss = sum_loss_val / total_valid_samples

                epoch_acc = correct_val / total_valid_samples
                y_true_binarized = label_binarize(y_true, classes=np.arange(num_classes))
                y_pred_probs = np.array(label_val_list)
                epoch_auc = roc_auc_score(y_true_binarized, y_pred_probs, average="macro", multi_class="ovr")
                if num_classes == 5:
                    if epoch_auc > best_auc:
                        best_auc = epoch_auc
                    ap_score = 0
                    #epoch_auc = 0
                    epoch_eer = 0
                    TPR_2 = 0
                    TPR_3 = 0
                    TPR_4 = 0
                    # save the results
                    save_acc(epoch_acc, ap_score, epoch_auc, epoch_eer, TPR_2, TPR_3, TPR_4, epoch, log_path)

                print("Validating: Epoch[{:0>1}/{:0>1}] Acc:{:.2%} Auc:{:.2%} Val Loss:{:.4f}".format(
                    epoch + 1, epoches, epoch_acc, epoch_auc, val_loss))

                # select the best accuracy and save the best pretrained model
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_model_wts = model.state_dict()
                    torch.save(best_model_wts, os.path.join(output_path, "best.pkl"))

            scheduler.step()

        # -----save the pretrained model----- #
        if epoch + 1 == epoches:
            torch.save(model.state_dict(), os.path.join(output_path, str(epoch + 1) + '_epoches_' + model_name))

    # ----------------------------------------------------end----------------------------------------------------#

    # -----print the results----- #
    print('-' * 20)
    print('Best_accuracy:', best_acc)
    print('Best_AUC:', best_auc)
    # print time
    time_end = time.time() - time_open
    print('All time: ', time_end)

    # -----save final results----- #
    if num_classes == 2:
        plot_ROC(label_val_list, predict_val_list)
        save_final_results(flops, params, time_end, best_acc, best_auc, log_path)
    else:
        best_auc = best_auc
        save_final_results(flops, params, time_end, best_acc, best_auc, log_path)


if __name__ == '__main__':
    parse = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parse.add_argument('--name', '-n', type=str, default='FSMc-resnet')
    parse.add_argument('--train_txt_path', '-tp', type=str, default='')
    parse.add_argument('--valid_txt_path', '-vp', type=str, default='')
    parse.add_argument('--batch_size', '-bz', type=int, default=32)
    parse.add_argument('--epoches', '-e', type=int, default=30)
    parse.add_argument('--model_name', '-mn', type=str, default='FSMc-resnet.pkl')
    parse.add_argument('--continue_train', type=bool, default=False)
    parse.add_argument('--model_path', '-mp', type=str, default='./output/FSMc-resnet/best.pkl')
    parse.add_argument('--num_classes', '-nc', type=int, default=5)
    parse.add_argument('--seed', default=7, type=int)
    print('-' * 20)
    print("PyTorch version: {}".format(torch.__version__))
    print("Python version: {}".format(platform.python_version()))
    print("cudnn version: {}".format(torch.backends.cudnn.version()))
    print('-' * 20)
    main()