import torch
import torch.nn as nn
import numpy as np

class CrossEntropy:

    def __init__(self, eps: float = 1e-12):
        self.eps = eps

    def sigmoid(self, x):
        # x: [n,]
        return 1 / (1 + np.exp(-x))

    def softmax(self, x):
        # x: [n, d]
        exp = np.exp(x)
        return exp / np.sum(exp, axis=1, keepdims=True)

    def binary_cross_entropy_loss(self, y_true, logits):
        # y_true: [n,], 取值: 0 / 1
        # logits: [n,]
        y_true = np.asarray(y_true)
        logits = np.asarray(logits)

        p = self.sigmoid(logits)
        p = np.clip(p, self.eps, 1 - self.eps)  # 数值稳定
        loss = -(y_true * np.log(p) + (1 - y_true) * np.log((1 - p)))
        return loss.mean()

    def cross_entropy_loss(self, y_true, logits):
        # y_true: [n,], 取值: 0 ~ C - 1
        # logits: [n, d]
        y_true = np.asarray(y_true)
        logits = np.asarray(logits)
        n = len(logits)  # y_true.shape[0]

        # softmax
        p = self.softmax(logits)  # [n, d]
        p = np.clip(p, self.eps, 1 - self.eps)  # 数值稳定
        # CE
        p = p[np.arange(n), y_true]  # [n,]
        loss = -np.log(p)
        return loss.mean()

if __name__ == "__main__":

    ce = CrossEntropy()

    # 1.二分类
    x = [6, -2, 2, 3]
    y_true = [1, 0, 1, 0]
    loss1 = ce.binary_cross_entropy_loss(y_true, x)
    print(loss1)

    # 2.多分类
    x = [[0, 1, 2, 3, 4],
         [1, 2, 3, 4, 5],
         [2, 3, 4, 5, 6],
         [3, 4, 5, 6, 7]]
    y_true = [3, 2, 0, 4]
    loss2 = ce.cross_entropy_loss(y_true, x)
    print(loss2)