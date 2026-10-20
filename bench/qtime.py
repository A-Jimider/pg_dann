#!/usr/bin/env python3
"""计时小工具:对某引擎某参数点跑 n 条 SIFT 查询,打印 avg/p50/p99/recall。

用法:.venv/bin/python bench/qtime.py <table> <n> "<set;sql>" [query_offset]
"""
import os
import sys
import time
import numpy as np
import psycopg2

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sift")
DSN = os.environ.get("PG_DANN_DSN", "host=/tmp port=5432 dbname=postgres")
K = 10


def read_vecs(path, dtype):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    arr = raw.reshape(-1, dim + 1)
    assert (arr[:, 0] == dim).all()
    return arr[:, 1:].view(dtype)


def main():
    table = sys.argv[1]
    n = int(sys.argv[2])
    set_sql = sys.argv[3]
    off = int(sys.argv[4]) if len(sys.argv) > 4 else 0

    queries = read_vecs(os.path.join(DATA, "sift_query.fvecs"), np.float32)[off:off + n]
    truth = read_vecs(os.path.join(DATA, "sift_groundtruth.ivecs"), np.int32)[off:off + n, :K]

    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET enable_seqscan = off")
    for stmt in set_sql.split(";"):
        if stmt.strip():
            cur.execute(stmt)

    lat = []
    hits = 0
    for i in range(n):
        q = ",".join(str(x) for x in queries[i])
        t0 = time.perf_counter()
        cur.execute(f"SELECT id FROM {table} ORDER BY embedding <-> '[{q}]' LIMIT {K}")
        rows = [r[0] for r in cur.fetchall()]
        lat.append((time.perf_counter() - t0) * 1000.0)
        hits += len(set(rows) & set(truth[i].tolist()))
    lat = np.array(lat)
    print(f"{lat.mean():.2f} {np.percentile(lat, 50):.2f} "
          f"{np.percentile(lat, 99):.2f} {hits / (n * K):.4f}")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
