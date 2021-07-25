import numpy as np
import matplotlib.pyplot as plt

# TF imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics

def make_summary(path, train_log, test_labels, test_predictions):

    fig, ax = plt.subplots(1,3)
    fig.set_size_inches(10, 3)

    # TRAINING AND VALIDATION
    loss = train_log['loss']
    acc = train_log['accuracy']
    acc_final = acc[-1]
    tp = train_log['tp'][-1]
    tn = train_log['tn'][-1]
    fp = train_log['fp'][-1]
    fn = train_log['fn'][-1]
    precision = train_log['precision'][-1]
    recall = train_log['recall'][-1]
    total = tp + tn + fp + fn

    val_loss = train_log['val_loss']
    val_acc = train_log['val_accuracy']
    val_acc_final = val_acc[-1]
    val_tp = train_log['val_tp'][-1]
    val_tn = train_log['val_tn'][-1]
    val_fp = train_log['val_fp'][-1]
    val_fn = train_log['val_fn'][-1]
    val_precision = train_log['val_precision'][-1]
    val_recall = train_log['val_recall'][-1]
    val_total = val_tp + val_tn + val_fp + val_fn

    e = np.linspace(1,len(loss),len(loss)) # epochs

    # plot loss
    ax[0].plot(e, loss, 'k', label='training')
    ax[0].plot(e, val_loss, 'c', label='validation')
    ax[0].set_xlim([1,e[-1]])
    ax[0].set_yticks(np.arange(0.0, 1.1, 0.1)) 
    ax[0].set_title('training and validation loss', fontweight='bold')
    ax[0].set_xlabel('epochs')
    ax[0].set_ylabel('loss')
    ax[0].legend(loc='upper right')

    # plot accuracy
    ax[1].plot(e, acc, 'k', label='training')
    ax[1].plot(e, val_acc, 'c', label='validation')
    ax[1].set_xlim([1,e[-1]])
    ax[1].set_yticks(np.arange(0.0, 1.1, 0.1)) 
    ax[1].set_title('training and validation accuracy', fontweight='bold')
    ax[1].set_xlabel('epochs')
    ax[1].set_ylabel('accuracy')
    ax[1].legend(loc='lower right')

    # TESTING
    # accuracy
    test_acc = keras.metrics.Accuracy()
    test_acc.update_state(test_labels, test_predictions)
    test_acc = test_acc.result().numpy()

    # tp
    test_tp = keras.metrics.TruePositives()
    test_tp.update_state(test_labels, test_predictions)
    test_tp = test_tp.result().numpy()

    # tn
    test_tn = keras.metrics.TrueNegatives()
    test_tn.update_state(test_labels, test_predictions)
    test_tn = test_tn.result().numpy()

    # fp
    test_fp = keras.metrics.FalsePositives()
    test_fp.update_state(test_labels, test_predictions)
    test_fp = test_fp.result().numpy()

    # fn
    test_fn = keras.metrics.FalseNegatives()
    test_fn.update_state(test_labels, test_predictions)
    test_fn = test_fn.result().numpy()

    # precision
    test_precision = keras.metrics.Precision()
    test_precision.update_state(test_labels, test_predictions)
    test_precision = test_precision.result().numpy()

    # recall
    test_recall = keras.metrics.Recall()
    test_recall.update_state(test_labels, test_predictions)
    test_recall = test_recall.result().numpy()

    test_total = test_tp + test_tn + test_fp + test_fn

    # SUMMARY TABLE
    data = [
        [total, val_total, test_total],
        [acc_final, val_acc_final, test_acc],
        [tp, val_tp, test_tp],
        [tn, val_tn, test_tn],
        [fp, val_fp, test_fp],
        [fn, val_fn, test_fn],
        [precision, val_precision, test_precision],
        [recall, val_recall, test_recall]
    ]

    data = np.round(data, 3)
    columns = ('$\\bf{training}$', '$\\bf{validation}$', '$\\bf{testing}$')
    rows = ['$\\bf{num samples}$', '$\\bf{accuracy}$', '$\\bf{TP}$', '$\\bf{TN}$', '$\\bf{FP}$', '$\\bf{FN}$', '$\\bf{precision}$', '$\\bf{recall}$']

    ax[2].axis('tight')
    ax[2].axis('off')
    table = ax[2].table(
        cellText=data,
        rowLabels=rows,
        colLabels=columns,
        #colWidths=[0.2 for x in columns],
        #cellLoc='center',
        loc='center'
    )
    ax[2].set_title('summary', fontweight='bold')

    fig.tight_layout()
    plt.show()
    fig.savefig(path + '/train_and_test_results.png', dpi=100)

