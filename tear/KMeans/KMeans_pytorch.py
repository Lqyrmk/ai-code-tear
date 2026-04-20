import torch

class KMeans:

    def __init__(self, k=3, max_iters=100, tol=1e-4, device=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.centroids = None
        self.labels = None

    def L2_distance(self, x, c):
        """
        x: [N, D]
        c: [K, D]
        """
        # (x - c) ^ 2 = x ^ 2 + c ^ 2 - 2 * x * c
        x2 = torch.sum(x ** 2, dim=1, keepdim=True)  # [N, 1]
        c2 = torch.sum(c ** 2, dim=1, keepdim=False)  # [K,]
        xc = x @ c.T
        dis = torch.clamp(x2 + c2 - 2 * xc, min=0)  # 防止负数，数值精度问题
        return torch.sqrt(dis)

    def fit(self, x):
        """
        x: [N, D] tensor
        """
        x = x.to(self.device)

        N, _ = x.shape

        random_centroids_indices = torch.randperm(N)[:self.k]  # [0 ~ N - 1] -> K
        self.centroids = x[random_centroids_indices].clone()  # [K, D]

        for _ in range(self.max_iters):

            distances = self.L2_distance(x, self.centroids)  # [N, K]
            self.labels = torch.argmin(distances, dim=1)  # [N,]

            old_centroids = self.centroids.clone()
            for i in range(self.k):
                cluster_x = x[self.labels == i]
                if len(cluster_x) > 0:
                    self.centroids[i] = torch.mean(cluster_x, dim=0)
                # len == 0 保持不变

            if torch.linalg.norm(old_centroids - self.centroids) < self.tol:
                break

    def predict(self, x):
        x = x.to(self.device)
        distances = self.L2_distance(x, self.centroids)  # [N, K]
        pred_label = torch.argmin(distances, dim=1)  # [N,]
        return pred_label

if __name__ == "__main__":

    k = 4
    max_iters = 100
    tol = 1e-4
    N = 20
    D = 10
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(N, D)

    model = KMeans(k, max_iters, tol, device)
    model.fit(x)

    print(f"centroids: {model.centroids}")

    N_test = 5
    test_x = torch.randn(N_test, D)
    pred = model.predict(test_x)

    print(f"predict: {pred}")