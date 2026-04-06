import numpy as np

def auc_bruteforce(labels, preds):
    """
    Bruteforce !!!
    input: List or NumPy Array
    time complexity: O(mn)
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
                cnt += 0.5

    total = len(pos) * len(neg)
    return cnt / total

def auc_sort(labels, preds):
    """
    sort solution !!!
    input: List or NumPy Array
    time complexity: O(nlogn)
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

if __name__ == '__main__':
    labels = [1, 0, 1, 0, 1]
    preds = [0.6, 0.7, 0.4, 0.5, 0.5]

    auc1 = auc_bruteforce(labels, preds)
    print(f"AUC 1 = {auc1}")
    auc2 = auc_sort(labels, preds)
    print(f"AUC 2 = {auc2}")