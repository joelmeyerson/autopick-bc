import numpy as np
import matplotlib.pyplot as plt
import json

# TF imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import metrics

#def make_summary(path, train_res, test_labels, test_predictions):

###################
# TESTING
path = "/Users/joel/Desktop/autopick-bc"

with open("../testing_results.txt", "r") as text_file:
    data = text_file.read()
    res = json.loads(data)

with open("../training_log.txt", "r") as text_file:
    data = text_file.read()

###################

# fig = plt.figure()
# fig.set_size_inches(8, 3)
#
# # TRAINING AND VALIDATION
# loss = train_res['loss']
# acc = train_res['accuracy']
# val_loss = train_res['val_loss']
# val_acc = train_res['val_accuracy']
# # tp
# #     fp
# #     tn
# #     fn
# #     precision
# #     recall
# #     auc
#
# e = np.linspace(1,len(loss),len(loss)) # epochs
#
# # plot loss
# plt.subplot(121)
# plt.plot(e, loss, 'k', label='training')
# plt.plot(e, val_loss, 'c', label='validation')
# plt.xlim([1,e[-1]])
# #plt.xticks(e)
# plt.ylim([0.0,1.0])
# #plt.yticks([0.0,1.0])
# #plt.title('loss')
# plt.xlabel('epochs')
# plt.ylabel('loss')
# plt.legend(loc='upper right')
# plt.tight_layout()
#
# # plot accuracy
# plt.subplot(122)
# plt.plot(e, acc, 'k', label='training')
# plt.plot(e, val_acc, 'c', label='validation')
# plt.xlim([1,e[-1]])
# #plt.ylim([min(min(acc), min(val_acc)),1.0])
# plt.ylim([0.0,1.0])
# #plt.title('accuracy')
# plt.xlabel('epochs')
# plt.ylabel('accuracy')
# plt.legend(loc='lower right')
# plt.tight_layout()
#
# # TESTING
#
# conf_mat = tf.math.confusion_matrix(test_labels, test_predictions, num_classes=2).numpy()
#
# tp = keras.metrics.TruePositives()
# tp.update_state(test_labels, test_predictions)
# print(tp.result().numpy())
#
# #plt.show()
# fig.savefig(path + '/train_and_test_results.png', dpi=100)
