# hello_ml.py
# Task 0 Python 热身：用列表/字典记录成绩 + NumPy 矩阵乘法

import numpy as np


def compute_stats(records):
    """返回平均分、最高分对应的姓名与成绩。records 为 [{"name": str, "score": float}]"""
    scores = [r["score"] for r in records]
    avg = sum(scores) / len(scores)
    best = max(records, key=lambda r: r["score"])
    return avg, best


def main():
    # 1. 使用列表/字典记录几组姓名与成绩
    records = [
        {"name": "Alice", "score": 88},
        {"name": "Bob", "score": 92},
        {"name": "Carol", "score": 76},
        {"name": "Dave", "score": 85},
        {"name": "Eve", "score": 95},
    ]

    print("姓名与成绩：")
    for r in records:
        print(f"  {r['name']}: {r['score']}")

    # 2. 编写函数计算平均分，并找出最高分
    avg, best = compute_stats(records)
    print(f"\n平均分: {avg:.2f}")
    print(f"最高分: {best['name']} ({best['score']})")

    # 3. 使用 NumPy 创建两个形状合适的矩阵，完成矩阵乘法
    A = np.arange(1, 7).reshape(2, 3)    # 形状 (2, 3)
    B = np.arange(1, 13).reshape(3, 4)   # 形状 (3, 4)
    C = A @ B                            # 矩阵乘法，形状 (2, 4)

    # 4. 输出计算结果以及输入、输出矩阵的形状
    print("\n矩阵 A (2x3):")
    print(A)
    print("矩阵 B (3x4):")
    print(B)
    print("矩阵乘法结果 C = A @ B (2x4):")
    print(C)
    print(f"\n输入形状: A={A.shape}, B={B.shape}")
    print(f"输出形状: C={C.shape}")


if __name__ == "__main__":
    main()
