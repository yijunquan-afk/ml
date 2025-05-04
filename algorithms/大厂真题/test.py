import sys
import numpy as np


def main():
    # 读取矩阵的行数 m 和列数 n
    m, n = map(int, sys.stdin.readline().split())

    # 读取 m 行每行 n 个整数，构造灰度矩阵 A
    A = []
    for _ in range(m):
        row = list(map(int, sys.stdin.readline().split()))
        A.append(row)
    A = np.array(A, dtype=float)

    # 读取使用的奇异值个数 k
    k = int(sys.stdin.readline())

    # 对 A 进行奇异值分解，得到 U, Sigma, Vt
    U, S, Vt = np.linalg.svd(A, full_matrices=False)
    # full_matrices=False 时，U 为 m×min(m,n)，Vt 为 min(m,n)×n

    # 取前 k 个奇异值与向量
    U_k = U[:, :k]  # U 的前 k 列，形状 m×k
    Sigma_k = np.diag(S[:k])  # Sigma 对角矩阵，形状 k×k
    Vt_k = Vt[:k, :]  # Vt 的前 k 行，形状 k×n

    # 重构矩阵 A_k = U_k * Sigma_k * Vt_k
    A_k = U_k.dot(Sigma_k).dot(Vt_k)

    # 输出结果，去掉末尾的零
    for i in range(m):
        # 对每个元素进行 round(x, 2)，并去掉末尾的零
        row_str = ' '.join(f"{round(val, 2):.2f}".rstrip('0').rstrip('.') for val in A_k[i])
        print(row_str)


if __name__ == "__main__":
    main()