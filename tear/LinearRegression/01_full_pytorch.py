import torch
import torch.nn as nn


class LinearRegression(nn.Module):

    def __init__(self):
        super().__init__()
        self.w = nn.Parameter(torch.randn(1))
        self.b = nn.Parameter(torch.randn(1))

    def forward(self, x):
        return self.w * x + self.b

def mse_loss(y_true, y_pred):
    return torch.mean((y_true - y_pred) ** 2)

def train(x, y, model, criterion, optimizer, num_epochs=1000):

    for epoch in range(num_epochs):
        y_pred = model(x)
        loss = criterion(y, y_pred)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 10 == 0:
            print(f'Epoch: {epoch}, Loss: {loss.item():.4f}, w: {model.w.data.item():.2f}, b: {model.b.data.item():.2f}')


if __name__ == "__main__":

    N = 64
    num_epochs = 1000

    x = torch.randn(N, 1)
    noise = 0.3 * torch.randn(N, 1)
    y = 3 * x + 2 + noise

    model = LinearRegression()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)

    for name, param in model.named_parameters():
        print(f"{name}: {param.data}")

    train(x, y, model, mse_loss, optimizer, num_epochs)