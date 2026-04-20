import numpy as np

class KMeans:

    def __init__(self, k=3, max_iters=100, tol=1e-4, random_state=None):
        self.k = k
        self.max_iters = max_iters
        self.tol = tol  # tolerance
        self.random_state = random_state

    def fit(self, x):
        """
        x: [N, D]
        return KMeans object
        """

        N, _ = x.shape
        np.random.seed(self.random_state)

        # 初始化 K 个随机质心
        random_centroid_indices = np.random.choice(N, self.k, replace=False)  # 不放回
        self.centroids = x[random_centroid_indices]  # [K, D]

        for _ in range(self.max_iters):
            # 计算每个点与质心的距离
            distances = self.L2_distance(x, self.centroids)  # [N, K]

            # 根据距离分配簇
            labels = np.argmin(distances, axis=1)  # [N,]

            # 更新中心
            new_centroids = np.zeros_like(self.centroids)  # [K, D]
            for i in range(self.k):
                cluster_x = x[labels == i]  # [Mi, D]
                if len(cluster_x) > 0:
                    new_centroids[i] = cluster_x.mean(axis=0)
                else:  # 空簇
                    # 1.继承之前的簇中心
                    # new_centroids[i] = self.centroids[i]
                    # 2.随机选取
                    new_centroids[i] = x[np.random.randint(0, N)]

            if np.linalg.norm(self.centroids - new_centroids) < self.tol:
                break

            self.centroids = new_centroids

        self.labels = labels
        return self


    def L2_distance(self, x, c):
        """
        x: [N, D]
        c: [K, D]
        return distance: [N, K]
        """
        # (x - c) ^ 2 = x ^ 2 + c ^ 2 - 2 * x * c
        x2 = np.sum(x ** 2, axis=1, keepdims=True)   # [N, 1]
        c2 = np.sum(c ** 2, axis=1, keepdims=False)  # [K,]
        xc = x @ c.T  # [N, K]
        dis = x2 + c2 -2 * xc  # [N, K]
        return np.sqrt(dis)

    def predict(self, x):
        """
        x: [N, K]
        return label: [N,]
        """
        dis = self.L2_distance(x, self.centroids)  # [N, K]
        pred_label = np.argmin(dis, axis=1)  # [N,]
        return pred_label

if __name__ == "__main__":
    # 超参
    seed = 42
    k = 4
    max_iters = 100
    tol = 1e-4
    N = 20
    D = 10
    x = np.random.rand(N, D)  # 均匀分布

    # 训练
    model = KMeans(k, max_iters, tol, seed)
    model.fit(x)

    print(f"centroids: {model.centroids}")

    # 预测
    test_N = 5
    test_x = np.random.rand(test_N, D)
    pred = model.predict(test_x)

    print(f"predict: {pred}")
