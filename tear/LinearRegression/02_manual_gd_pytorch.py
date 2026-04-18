import torch
import torch.nn as nn

def mse_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def train(x, y, w, b, criterion, lr=0.01, num_epochs=1000):

    for epoch in range(num_epochs):
        y_pred = w * x + b

        loss = criterion(y, y_pred)
        loss.backward()

        # manually implement the gradient descent method to update parameters
        with torch.no_grad():
            # update
            w -= lr * w.grad
            b -= lr * b.grad

            # gradient zeroing
            w.grad.zero_()
            b.grad.zero_()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, w: {w.item():.2f}, b: {b.item():.2f}')


if __name__ == "__main__":

    N = 64
    lr = 0.01
    num_epochs = 1000

    x = torch.randn(N, 1)
    noise = 0.3 * torch.randn(N, 1)
    y = 3 * x + 2 + noise

    # method 1: torch.tensor
    # w = torch.randn(1, requires_grad=True)
    # b = torch.randn(1, requires_grad=True)

    # method 2: nn.Parameter
    w = nn.Parameter(torch.randn(1))
    b = nn.Parameter(torch.randn(1))

    print(f"w: {w.item()}")
    print(f"b: {b.item()}")

    train(x, y, w, b, mse_loss, lr, num_epochs)