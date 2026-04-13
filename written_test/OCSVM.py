import sys
import json
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.svm import OneClassSVM
from sklearn.metrics import roc_auc_score, f1_score

"""
2026/3/28
美团 2026 年春招第三场笔试【算法策略方向】（转正实习）

小美是一位算法工程师，她的任务是识别平台上的异常交易订单，以保障用户和商家的资金安全。
她计划使用 One-Class SVM 模型来解决这个问题，并设计了一套标准的模型训练、验证和调优流程。
请你遵循小美的设计，仅用 numpy / pandas / scikit-learn，完成整个流程。
"""

def load_data():
    # data = json.loads(sys.stdin.read())

    data = {
        "train": [[0, 0, 0], [-0.1, 0.1, 0], [0.2, -0.2, 0], [5, 5, 1]],
        "test": [[0.1, 0.0], [5, 5]],
        "gamma_list": [0.05, 0.2],
        "nu_list": [0.05, 0.1]
    }

    train_data = np.array(data["train"], dtype=np.float64)
    test_data = np.array(data["test"], dtype=np.float64)
    gamma_list = np.array(data["gamma_list"], dtype=np.float64)
    nu_list = np.array(data["nu_list"], dtype=np.float64)

    X = train_data[:, :-1]
    y = train_data[:, -1]
    X_test = test_data

    X_norm = X[y == 0]  # train + val
    X_val_anom = X[y == 1]  # val

    X_train, X_val_norm = train_test_split(X_norm, test_size=0.25, random_state=42, shuffle=True)

    mean = np.mean(X_train, axis=0)
    std = np.std(X_train, axis=0)
    std[std == 0] = 1

    X_train = (X_train - mean) / std
    X_val_norm = (X_val_norm - mean) / std
    X_val_anom = (X_val_anom - mean) / std
    X_test = (X_test - mean) / std


    return X_train, X_val_norm, X_val_anom, X_test, gamma_list, nu_list

def solve(X_train, X_val_norm, X_val_anom, X_test, gamma_list, nu_list):

    y_train = np.zeros(X_train.shape[0])

    X_val = np.vstack([X_val_norm, X_val_anom], axis=0)
    y_val_norm = np.zeros(X_val_norm.shape[0])
    y_val_anom = np.ones(X_val_anom.shape[0])
    y_val = np.concatenate([y_val_norm, y_val_anom], axis=0)

    res = []

    # 训练
    for gamma in gamma_list:
        for nu in nu_list:
            model = OneClassSVM(kernel='rbf',
                                gamma=gamma,
                                nu=nu,
                                shrinking=False,
                                tol=1e-4,
                                cache_size=200,
                                max_iter=-1)
            model.fit(X_train)

            score = model.decision_function(X_val)
            auc = roc_auc_score(y_val, -score)

            pred = model.predict(X_val)
            pred = (pred == -1).astype(int)  # 1 -> 0, -1 -> 1
            f1 = f1_score(y_val, pred)

            res.append((-auc, -f1, nu, gamma))

    res.sort()
    _, _, best_nu, best_gamma = res[0]

    # 重训
    X = np.concatenate([X_train, X_val_norm], axis=0)

    model = OneClassSVM(kernel='rbf',
                        gamma=best_gamma,
                        nu=best_nu,
                        shrinking=False,
                        tol=1e-4,
                        cache_size=200,
                        max_iter=-1)
    model.fit(X)

    # 预测
    test_pred = model.predict(X_test)
    return [1 if p == -1 else 0 for p in test_pred]


if __name__ == "__main__":

    X_train, X_val_norm, X_val_anom, X_test, gamma_list, nu_list = load_data()

    labels = solve(X_train, X_val_norm, X_val_anom, X_test, gamma_list, nu_list)
    print(labels)