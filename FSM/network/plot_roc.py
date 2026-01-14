
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import matplotlib.pyplot as plt

# plot ROC, compute AUC and EER for multi-class classification
def plot_ROC(y, y_p, num_classes):
    # Binarize the true labels for multi-class classification
    y_true_binarized = label_binarize(y, classes=np.arange(num_classes))

    # Calculate ROC and AUC for each class
    fpr_all, tpr_all, auc_all = {}, {}, {}
    for i in range(num_classes):
        fpr, tpr, thresholds = roc_curve(y_true_binarized[:, i], y_p[:, i])
        roc_auc = auc(fpr, tpr)
        fpr_all[i] = fpr
        tpr_all[i] = tpr
        auc_all[i] = roc_auc

    # Calculate Macro AUC
    macro_auc = np.mean(list(auc_all.values()))

    # Calculate EER for each class
    eer_all = {}
    for i in range(num_classes):
        fpr = fpr_all[i]
        tpr = tpr_all[i]
        eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer_all[i] = eer

    # Print individual class AUC and EER
    for i in range(num_classes):
        print(f"AUC for class {i}: {auc_all[i]:.4f}")
        print(f"EER for class {i}: {eer_all[i]:.4f}")

    # Plot the ROC curve for each class
    plt.figure()
    ax = plt.gca()
    for i in range(num_classes):
        plt.plot(fpr_all[i], tpr_all[i], label=f'Class {i} AUC={auc_all[i]:.4f}')
    plt.plot([0, 1], [0, 1], 'r--', label='Random')
    plt.title("Receiver Operating Characteristic (ROC)")
    plt.xlabel('False Positive Rate (FPR)')
    plt.ylabel('True Positive Rate (TPR)')
    plt.legend(loc='lower right')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.0])
    plt.savefig('./output/roc.png', bbox_inches='tight')
    plt.pause(3)

    # Print the macro AUC
    print(f"Macro AUC: {macro_auc:.4f}")

    return eer_all, auc_all, macro_auc
