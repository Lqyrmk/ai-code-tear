import numpy as np

def auc_bruteforce(labels, preds):
    """
    Bruteforce !!!
    input: List or NumPy Array
    time complexity: O(MN)
    """

    n = len(labels)
    pos = [i for i in range(n) if labels[i] == 1]
    neg = [i for i in range(n) if labels[i] == 0]

    cnt = 0
    for i in pos:
        for j in neg:
            if preds[i] > preds[j]:
                cnt += 1
            elif preds[i] == preds[j]:
                cnt += 0.5  # tie

    total = len(pos) * len(neg)
    return cnt / total

def auc_sort(labels, preds):
    """
    sort solution !!!
    input: List
    time complexity: O(nlogn), n = M + N
    """
    data = list(zip(preds, labels))
    data.sort(key=lambda x: x[0])  # 按照分数 pred 从小到大排序

    pos_rank_sum = pos_cnt = neg_cnt = 0
    for rank, (_, label) in enumerate(data, start=1):  # rank 从 1 开始
        if label == 1:
            pos_cnt += 1
            pos_rank_sum += rank
        else:
            neg_cnt += 1
    return (pos_rank_sum - pos_cnt * (pos_cnt + 1) / 2) / (pos_cnt * neg_cnt)

def auc_sort_tie(labels, preds):
    """
    sort solution with tie !!!
    input: List
    time complexity: O(nlogn), n = M + N
    """
    data = list(zip(preds, labels))
    data.sort(key=lambda x: x[0])

    n = len(preds)
    pos_rank_sum = M = N = 0
    i = 0
    while i < n:  # O(n)
        j = i
        pos_cnt = neg_cnt = rank_sum = 0  # 统计区间情况
        while j < n and data[i][0] == data[j][0]:
            rank_sum += j + 1  # rank 从 1 开始
            if data[j][1] == 1:
                pos_cnt += 1
            else:
                neg_cnt += 1
            j += 1
        # tie: 对 rank 进行平均，区间是 [i, j)
        mean_rank = rank_sum / (pos_cnt + neg_cnt)
        pos_rank_sum += mean_rank * pos_cnt
        M += pos_cnt
        N += neg_cnt
        i = j
    return (pos_rank_sum - M * (M + 1) / 2) / (M * N)


if __name__ == '__main__':
    labels = [1, 0, 1, 0, 1]
    preds = [0.6, 0.7, 0.4, 0.5, 0.5]

    auc1 = auc_bruteforce(labels, preds)
    print(f"AUC 1 = {auc1}")

    auc2 = auc_sort(labels, preds)
    print(f"AUC 2 = {auc2}")

    auc3 = auc_sort_tie(labels, preds)
    print(f"AUC 3 = {auc3}")