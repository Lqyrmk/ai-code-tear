import torch


def mse_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def relu(x):
    return torch.maximum(x, torch.tensor(0.0))  # maximum!!

def train(x, y, W1, b1, W2, b2, lr=0.01, num_epochs=1000):

    for epoch in range(num_epochs):

        # two-layer
        H = relu(x @ W1 + b1)
        y_pred = H @ W2 + b2

        loss = mse_loss(y, y_pred)
        loss.backward()

        with torch.no_grad():
            W1 -= lr * W1.grad
            b1 -= lr * b1.grad
            W2 -= lr * W2.grad
            b2 -= lr * b2.grad

            W1.grad.zero_()
            b1.grad.zero_()
            W2.grad.zero_()
            b2.grad.zero_()

        if epoch % 50 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item():.4f}')

    print(f"W1: {W1}")
    print(f"b1: {b1}")
    print(f"W2: {W2}")
    print(f"b2: {b2}")


if __name__ == "__main__":

    N = 64
    d = 4
    lr = 0.01
    num_epochs = 1000

    x = torch.randn(N, d)
    y = torch.randn(N, 1)

    in_dim = d
    hid_dim = d
    out_dim = 1

    W1 = torch.randn(in_dim, hid_dim, requires_grad=True)
    b1 = torch.zeros(hid_dim, requires_grad=True)
    W2 = torch.randn(hid_dim, out_dim, requires_grad=True)
    b2 = torch.zeros(out_dim, requires_grad=True)

    # print(f"W1: {W1}")
    # print(f"b1: {b1}")
    # print(f"W2: {W2}")
    # print(f"b2: {b2}")

    train(x, y, W1, b1, W2, b2, lr, num_epochs)
