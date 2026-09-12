import sys
import json

from math import inf

"""
2026/9/12
京东 2026 年秋招笔试

题干是实现一个搜推的 MMR 算法，重排算法。
要求用 NumPy，实际上用不上 NumPy。。甚至空值也不需要处理
直接使用常见手法模拟就能过
"""

def sim(x, cat_set):
    return int(x["cat"] in cat_set)

if __name__ == "__main__":

    # s = sys.stdin.readline().strip()
    s = """
    {
        "K": 3,
        "lambda": 0.7,
        "queries": [
            {
                "items": [
                    {"id": 1, "score": 0.95, "cat": "A"},
                    {"id": 2, "score": 0.85, "cat": "A"},
                    {"id": 3, "score": 0.83, "cat": "B"},
                    {"id": 4, "score": 0.75, "cat": "C"}
                ]
            },
            {
                "items": [
                    {"id": 1, "score": 0.95, "cat": "A"},
                    {"id": 4, "score": 0.65, "cat": "D"},
                    {"id": 7, "score": 0.33, "cat": "B"}
                ]
            }
        ]
    }
    """

    obj = json.loads(s)
    K = obj["K"]
    lambda_ = obj["lambda"]
    ans = []
    for query in obj["queries"]:
        items = [{"id": item["id"], "score": item["score"], "cat": "__MISSING__" if item["cat"] is None else item["cat"], "vis": False} for item in query["items"]]
        items.sort(key=lambda x: (-x["score"], x["id"]))  # 排序

        seq = []
        cat_set = set()
        n = min(K, len(items))
        for i in range(n):
            if i == 0:
                item = items[0]
                seq.append(item["id"])
                item["vis"] = True
                cat_set.add(item["cat"])
            else:
                res_item = None
                mx = -inf
                for item in items:
                    if item["vis"]:
                        continue
                    J = lambda_ * item["score"] - (1 - lambda_) * sim(item, cat_set)
                    if J > mx:
                        mx = max(mx, J)
                        res_item = item
                if res_item is not None:
                    seq.append(res_item["id"])
                    res_item["vis"] = True
                    cat_set.add(res_item["cat"])
        ans.append(seq)

    print(json.dumps({"ranked": ans}))