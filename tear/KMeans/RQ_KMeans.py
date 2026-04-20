import torch
import torch.nn.functional as F

class RQKMeans:

    def __init__(self, num_subvectors=4, num_clusters=256, max_iters=20, tol=1e-4, device=None):
        self.M = num_subvectors
        self.K = num_clusters
        self.tol = tol
        self.max_iters = max_iters
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.codebooks = None  # [M, K, D]
        self.D = None

    def L2_distance(self, x, c):
        """
        x: [N, D] tensor
        c: [K, D] tensor
        return L2 distance: [N, K] tensor
        """
        # (x - c) ^ 2 = x ^ 2 + c ^ 2 - 2 * x * c
        x2 = torch.sum(x ** 2, dim=1, keepdim=True)  # [N, 1]
        c2 = torch.sum(c ** 2, dim=1, keepdim=False)  # [K,]
        xc = x @ c.T
        dis = torch.clamp(x2 + c2 - 2 * xc, min=0)  # 防止负数，数值精度问题
        return torch.sqrt(dis)

    def k_means(self, x):
        """
        x: [N, D] tensor
        return centroids: [K, D] tensor
        """
        N, _ = x.shape

        random_centroids_indices = torch.randperm(N)[:self.K]  # [0 ~ N - 1] -> K
        centroids = x[random_centroids_indices].clone()  # [K, D]

        for _ in range(self.max_iters):

            # distances = self.L2_distance(x, centroids)  # [N, K]
            distances = torch.cdist(x, centroids)
            labels = torch.argmin(distances, dim=1)  # [N,]

            old_centroids = centroids.clone()
            for i in range(self.K):
                cluster_x = x[labels == i]
                if len(cluster_x) > 0:
                    centroids[i] = torch.mean(cluster_x, dim=0)

            if torch.linalg.norm(old_centroids - centroids) < self.tol:
                break

        return centroids

    def predict_centroid(self, x, centroids):
        """
        x: [N, D] tensor
        centroids: [K, D] tensor
        return pred_label: [N,] tensor
        """
        # distances = self.L2_distance(x, centroids)  # [N, K]
        distances = torch.cdist(x, centroids)
        pred_label = torch.argmin(distances, dim=1)  # [N,]
        return pred_label

    def fit(self, x):
        """
        x: [N, D] tensor
        return model
        """
        self.D = x.size(1)

        codebooks = []
        # 初始化残差为 x，第一个码本是由 x 训练得到的
        r = x.clone()
        for _ in range(self.M):
            codebook = self.k_means(r)  # [K, D]
            codebooks.append(codebook)

            labels = self.predict_centroid(r, codebook)  # [N,]
            cx = codebook[labels]  # [N, D] 中心向量
            r = r - cx

        self.codebooks = torch.stack(codebooks, dim=0)  # [M, K, D]

        return self

    def encode(self, x):
        """
        x: [N, D] tensor
        return discrete token sequence indices: [N, M] tensor
        """
        indices = []
        r = x.clone()
        for i in range(self.M):
            codebook = self.codebooks[i]  # [K, D]
            labels = self.predict_centroid(r, codebook)  # [N,]
            indices.append(labels)

            cx = codebook[labels]  # [N, D]
            r = r - cx

        indices = torch.stack(indices, dim=1)  # [N, M]
        return indices

    def decode(self, indices):
        """
        indices: [N, M] tensor, discrete token sequence
        return quantized x: [N, D] tensor
        """
        N, M = indices.shape
        x_hat = torch.zeros(N, self.D, device=self.device)
        for i in range(M):
            codebook = self.codebooks[i]  # [K, D]
            labels = indices[:, i]  # [N,]
            cx = codebook[labels]  # [N, D]
            x_hat = x_hat + cx
        return x_hat  # [N, D]


if __name__ == "__main__":

    N = 10000
    D = 512
    M = 4
    K = 256
    max_iters = 20
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    x = torch.randn(N, D, dtype=torch.float32).to(device)
    model = RQKMeans(num_subvectors=M, num_clusters=K, device=device)
    model.fit(x)

    indices = model.encode(x)
    x_hat = model.decode(indices)
    recon_loss = F.mse_loss(x, x_hat)

    print(f"Codebooks: {model.codebooks.shape}")
    print(f"Indices: {indices.shape}")
    print(f"Reconstruction loss: {recon_loss.item():.4f}")

