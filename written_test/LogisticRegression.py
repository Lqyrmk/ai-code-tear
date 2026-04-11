import sys
import numpy as np
import json

"""
2026/4/11
美团 2026 年春招第四场笔试【算法策略方向】（转正实习）

小美正在为美团的优惠券推荐业务开发一个预测模型，
她需要使用对数几率回归（Logistic Regression）来预测用户是否会购买某个优惠券。
请你帮助她，在仅使用numpy、pandas 的前提下，手写实现该模型并对测试样本给出类别预测。
"""

def load_data():
    # data = json.loads(sys.stdin.read())

    data = {"train": [[1,2,0],[2,1.8,0],[5,5,1],[4.5,5.2,1]], "test": [[1.5,1.9],[5,5.1]]}

    train_data = np.array(data['train'], dtype=np.float64)
    test_data = np.array(data['test'], dtype=np.float64)

    return train_data, test_data

def concat_bias(X):
    b = np.ones((X.shape[0], 1))  # [N, 1]
    # X_bias = np.concat([b, X], axis=1)  # [N, D + 1]
    X_bias = np.concatenate([b, X], axis=1)  # [N, D + 1]
    return X_bias

def sigmoid(z):
    return 1.0 / (1 + np.exp(-z))

def fit(X, y, max_iter=30, tol=1e-6):

    D = X.shape[1]

    w = np.random.rand(D, 1)  # [D, 1]

    eps = 1e-6
    I = np.eye(D)

    for _ in range(max_iter):

        p = sigmoid(X @ w)  # [N, D] @ [D, 1] = [N, 1]
        p_flat = p.flatten()  # [N,]


        W = np.diag(p_flat * (1 - p_flat))  # [N, N]

        Z = X.T @ W @ X + eps * I  # [D, D]

        new_w = w - np.linalg.inv(Z) @ X.T @ (p - y)  # [D, 1]

        if np.linalg.norm(new_w - w) < tol:
            w = new_w
            break

        w = new_w

    return w

def predict(X, w):
    p_hat = sigmoid(X @ w)  # [N, 1]
    y_hat = (p_hat >= 0.5).flatten().astype(np.int64)  # [N,]
    return y_hat


if __name__ == "__main__":

    train_data, test_data = load_data()

    X_train = train_data[:, :-1]  # [N, D]
    y_train = train_data[:, -1]  # 行向量
    y_train = y_train.reshape(-1, 1)  # [N, 1]
    X_test = test_data

    N_train = X_train.shape[0]
    N_test = X_test.shape[0]

    # Method 1: concatenation
    # X_train_bias = concat_bias(X_train)  # [N, D + 1]
    # X_test_bias = concat_bias(X_test)  # [N', D + 1]

    # Method 2: hstack
    X_train_bias = np.hstack([np.ones((N_train, 1)), X_train])
    X_test_bias = np.hstack([np.ones((N_test, 1)), X_test])

    w = fit(X_train_bias, y_train)

    y_pred = predict(X_test_bias, w)

    ans = ' '.join(map(str, y_pred))

    print(ans)
