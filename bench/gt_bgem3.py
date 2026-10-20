#!/usr/bin/env python3
"""bge-m3 100k groundtruth:cosine 暴力精确 top-100。

查询 = 底库每 200 行取 1 条(500 条,确定性;查询行在库内,top-1 为自身,
各引擎同待遇)。cosine:双侧 L2 归一化后点积,取最大。
输出 bench/data/bgem3/gt_500q.ivecs(top-100,recall@10/50/100 通用)。
用法:.venv/bin/python bench/gt_bgem3.py
"""
import os
import numpy as np

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bgem3")
K = 100
QSTEP = 200


def read_fvecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    arr = raw.reshape(-1, dim + 1)
    assert (arr[:, 0] == dim).all()
    return arr[:, 1:].view(np.float32)


def main():
    base = read_fvecs(os.path.join(DATA, "base.fvecs")).astype(np.float32)
    nb = base.shape[0]
    qidx = np.arange(0, nb, QSTEP)
    base /= np.linalg.norm(base, axis=1, keepdims=True)          # cosine
    queries = base[qidx].copy()                                  # (nq, 1024) 已归一
    nq = queries.shape[0]
    print(f"base {nb}, queries {nq} (every {QSTEP}th row), cosine top-{K}")

    best_s = np.full((nq, K), -np.inf, dtype=np.float32)
    best_i = np.full((nq, K), -1, dtype=np.int32)
    CHUNK = 50_000  # 100k 底库就两块,峰值 ~几百 MB
    for start in range(0, nb, CHUNK):
        end = min(start + CHUNK, nb)
        sims = queries @ base[start:end].T                       # (nq, m)
        cand_s = np.concatenate([best_s, sims], axis=1)
        cand_i = np.concatenate([best_i,
                                 (np.arange(start, end, dtype=np.int32))[None, :]
                                 .repeat(nq, axis=0)], axis=1)
        sel = np.argpartition(-cand_s, K - 1, axis=1)[:, :K]     # 取相似度最大
        rows = np.arange(nq)[:, None]
        best_s, best_i = cand_s[rows, sel], cand_i[rows, sel]
        order = np.argsort(-best_s, axis=1)                      # 降序
        best_s = np.take_along_axis(best_s, order, axis=1)
        best_i = np.take_along_axis(best_i, order, axis=1)
        print(f"  [{start}:{end}] done", flush=True)

    out = os.path.join(DATA, "gt_500q.ivecs")
    with open(out, "wb") as f:
        for i in range(nq):
            np.array([K], dtype=np.int32).tofile(f)
            best_i[i].astype(np.int32).tofile(f)
    print(f"wrote {out} ({nq} queries x top-{K})")


if __name__ == "__main__":
    main()
