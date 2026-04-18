import torch

N = 64
lr = 0.01
num_epochs = 1000

x = torch.randn(N, 1)
noise = 0.3 * torch.randn(N, 1)
y = 3 * x + 2 + noise

w = torch.randn(1, requires_grad=True)
b = torch.randn(1, requires_grad=True)

def mse_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

for epoch in range(num_epochs):
    y_pred = w * x + b

    loss = mse_loss(y, y_pred)
    loss.backward()

    with torch.no_grad():
        w -= lr * w.grad
        b -= lr * b.grad

        w.grad.zero_()
        b.grad.zero_()

    if epoch % 10 == 0:
        print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, w: {w.item():.2f}, b: {b.item():.2f}')
