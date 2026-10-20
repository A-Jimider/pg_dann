#!/usr/bin/env python3
"""SIFT1M 延迟预算截断验证(W6 验收):recall 平滑降级 + 截断计数。

固定 probes=50,扫 max_latency_ms 网格,报 recall@10 / 平均与 p99 延迟 /
截断的扫描占比。用法:.venv/bin/python bench/budget.py [n_queries]
"""
import os
import sys
import time
import numpy as np
import psycopg2

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sift")
DSN = os.environ.get("PG_DANN_DSN", "host=/tmp port=5432 dbname=postgres")
K = 10
PROBES = 50
BUDGETS_MS = [0, 1, 2, 4, 8]        # 0 = 不限


def read_vecs(path, dtype):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    arr = raw.reshape(-1, dim + 1)
    assert (arr[:, 0] == dim).all()
    return arr[:, 1:].view(dtype)


def main():
    n_queries = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    queries = read_vecs(os.path.join(DATA, "sift_query.fvecs"), np.float32)[:n_queries]
    truth = read_vecs(os.path.join(DATA, "sift_groundtruth.ivecs"), np.int32)[:n_queries, :K]

    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET enable_seqscan = off")
    cur.execute(f"SET dann.probes = {PROBES}")

    print(f"== dann latency budget (probes={PROBES}, {n_queries} queries, recall@{K}) ==")
    print(f"{'budget_ms':>9} {'recall':>8} {'avg_ms':>9} {'p99_ms':>9} "
          f"{'scans':>6} {'truncated':>9} {'lists_skipped':>13}")
    for budget in BUDGETS_MS:
        cur.execute(f"SET dann.max_latency_ms = {budget}")
        cur.execute("SELECT dann_scan_stats_reset()")
        lat = []
        hits = 0
        for i in range(n_queries):
            q = ",".join(str(x) for x in queries[i])
            t0 = time.perf_counter()
            cur.execute(f"SELECT id FROM sift_base ORDER BY embedding <-> '[{q}]' LIMIT {K}")
            rows = [r[0] for r in cur.fetchall()]
            lat.append((time.perf_counter() - t0) * 1000.0)
            hits += len(set(rows) & set(truth[i].tolist()))
        cur.execute("SELECT * FROM dann_scan_stats()")
        scans, truncated, skipped = cur.fetchone()
        recall = hits / (n_queries * K)
        lat = np.array(lat)
        print(f"{budget:>9} {recall:>8.4f} {lat.mean():>9.2f} "
              f"{np.percentile(lat, 99):>9.2f} {scans:>6} {truncated:>9} {skipped:>13}")

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
