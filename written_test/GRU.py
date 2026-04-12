import sys
import json
import numpy as np

"""
2026/4/11
美团 2026 年春招第二场笔试【算法策略方向】（转正实习）

小美是一位算法工程师，她正在研究一个用于预测用户下单行为的序列模型。为了更好地理解模型内部的运作机制，她决定亲手实现模型核心组件——单层 GRU 的前向传播过程。请你帮助小美完成这个任务，注意，请仅使用 numpy / pandas / scikit-learn 进行实现。
"""

def load_data():

    # data = json.load(sys.stdin)
    # print(data)

    data = {"Wx":[[0.5,0,0.5,0,1,0],[0,0.5,0,0.5,0,1]],"Wh":[[0,0,0,0,0,0],[0,0,0,0,0,0]],"b":[0,0,0,0,0,0],"h0":[0,0],"X":[[0,0]]}

    Wx = np.array(data['Wx'], dtype=np.float64)
    Wh = np.array(data['Wh'], dtype=np.float64)
    b = np.array(data['b'], dtype=np.float64)
    h0 = np.array(data['h0'], dtype=np.float64)
    X = np.array(data['X'], dtype=np.float64)

    return Wx, Wh, b, h0, X

def sigmoid(x):
    return 1.0 / (1 + np.exp(-x))

def tanh(x):
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def GRU_forward(Wx, Wh, b, h0, X):

    T = X.shape[0]
    H = h0.shape[0]

    # [d, 3H] -> [d, H]
    Wxr, Wxz, Wxh = Wx[:, :H], Wx[:, H: 2 * H], Wx[:, 2 * H:]
    # [H, 3H] -> [H, H]
    Whr, Whz, Whh = Wh[:, :H], Wh[:, H: 2 * H], Wh[:, 2 * H:]
    # [3H,] -> [H,]
    br, bz, bh = b[H], b[H: 2 * H], b[2 * H:]

    h = h0  # [H,]
    for t in range(T):
        xt = X[t]
        r = sigmoid(xt @ Wxr + h @ Whr + br)  # [H,]
        z = sigmoid(xt @ Wxz + h @ Whz + bz)  # [H,]
        # h_tilde = tanh(xt @ Wxh + (r * h) @ Whh + bh)  # [H,]
        h_tilde = np.tanh(xt @ Wxh + (r * h) @ Whh + bh)  # [H,]
        h = (1 - z) * h + z * h_tilde  # [H,]

    return [round(x, 6) for x in h]


if __name__ == '__main__':

    Wx, Wh, b, h0, X = load_data()
    ans = GRU_forward(Wx, Wh, b, h0, X)
    print(ans)