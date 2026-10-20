#!/usr/bin/env python3
"""bigann-10M groundtruth:暴力精确 L2 top-100(分块 matmul,内存受控)。

输出 bench/data/bigann/gt10m_1k.ivecs(前 1000 个查询, ivecs 格式)。
用法:.venv/bin/python bench/gt_bigann.py [n_queries]
"""
import os
import sys
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bigann")
K = 100          # 多算到 top-100,recall@10/50/100 都能用
CHUNK = 100_000  # base 分块行数;1000 查询时每块峰值 ~1.7GB(dots/d2/cand 各 400MB)
                 # 500k 分块峰值 6-8GB,曾在 11GB WSL2 上把整机跑崩,勿调大


def main():
    n_queries = int(sys.argv[1]) if len(sys.argv) > 1 else 1000

    print("loading base (uint8 10M x 128) ...")
    raw8 = np.fromfile(os.path.join(DATA, "base10m.bvecs"), dtype=np.uint8)
    base_u8 = raw8.reshape(-1, 132)[:, 4:]
    nb = base_u8.shape[0]
    print(f"  {nb} rows")

    qraw = np.fromfile(os.path.join(DATA, "query.bvecs"), dtype=np.uint8)
    queries_u8 = qraw.reshape(-1, 132)[:, 4:][:n_queries].astype(np.float32)
    nq = queries_u8.shape[0]
    qnorm = (queries_u8 ** 2).sum(axis=1)                     # (nq,)

    best_d = np.full((nq, K), np.inf, dtype=np.float32)
    best_i = np.full((nq, K), -1, dtype=np.int32)

    for start in range(0, nb, CHUNK):
        end = min(start + CHUNK, nb)
        b = base_u8[start:end].astype(np.float32)             # (m,128)
        # d^2 = qn + bn - 2 q·b
        bn = (b ** 2).sum(axis=1)                             # (m,)
        dots = queries_u8 @ b.T                               # (nq,m) BLAS
        d2 = qnorm[:, None] + bn[None, :] - 2.0 * dots
        # 与已有最优合并取 top-K
        cand_d = np.concatenate([best_d, d2], axis=1)
        cand_i = np.concatenate([best_i,
                                 (np.arange(start, end, dtype=np.int32))[None, :]
                                 .repeat(nq, axis=0)], axis=1)
        sel = np.argpartition(cand_d, K - 1, axis=1)[:, :K]
        rows = np.arange(nq)[:, None]
        best_d = cand_d[rows, sel]
        best_i = cand_i[rows, sel]
        # 排序使输出升序
        order = np.argsort(best_d, axis=1)
        best_d = np.take_along_axis(best_d, order, axis=1)
        best_i = np.take_along_axis(best_i, order, axis=1)
        print(f"  [{start}:{end}] done", flush=True)

    out = os.path.join(DATA, f"gt10m_{nq}q.ivecs")
    with open(out, "wb") as f:
        for i in range(nq):
            np.array([K], dtype=np.int32).tofile(f)
            best_i[i].astype(np.int32).tofile(f)
    print(f"wrote {out} ({nq} queries x top-{K})")


if __name__ == "__main__":
    main()
