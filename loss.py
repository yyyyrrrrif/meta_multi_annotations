import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

def one_hot(target, n_classes):
    targets = np.array([target]).reshape(-1)
    one_hot_targets = np.eye(n_classes)[targets]
    return one_hot_targets

def multi_annot_one_hot(answers, N_CLASSES=8):
    row = []
    for i in range(len(answers)):
        if answers[i] == -1:
            row.append(-1 * np.ones(N_CLASSES))
        else:
            row.append(one_hot(answers[i], N_CLASSES)[0,:])
    row = np.array(row)
    return row


class MaskedMultiCrossEntropy(object):

    def loss(self, y_true, y_pred):
        one_hot_y_true = multi_annot_one_hot(y_true)

        return 
        # softmax = torch.exp(y_pred)/torch.sum(torch.exp(y_pred), dim = 1).reshape(-1, 1)
        # logsoftmax = torch.log(softmax)
        # vec = one_hot_y_true * logsoftmax / .shape[0]
        # mask = y_true.eq(torch.tensor([-1] * len(y_true)))
        # zer = torch.zeros_like(vec)
        # loss = torch.where(mask, zer, vec)
        # return loss
        

