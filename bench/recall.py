#!/usr/bin/env python3
"""SIFT1M recall/QPS 对比:pg_dann(SQ8) vs pgvector IVFFlat。

对同一组查询、同一 probes 网格,报 recall@10 与平均/分位延迟。
通过 pg_index.indisusable 切换隔离两个索引,保证各自独立测量。
用法:.venv/bin/python bench/recall.py [n_queries]
"""
import os
import sys
import time
import numpy as np
import psycopg2

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sift")
DSN = os.environ.get("PG_DANN_DSN", "host=/tmp port=5432 dbname=postgres")
K = 10
PROBES = [1, 5, 10, 20, 50]
WARMUP = 3


def read_vecs(path, dtype):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    arr = raw.reshape(-1, dim + 1)
    assert (arr[:, 0] == dim).all()
    return arr[:, 1:].view(dtype)


def set_index_usable(cur, name, usable):
    pass  # indisusable 在 PG12+ 已移除;改为每引擎独立表(sift_base / sift_pgv)


def main():
    n_queries = int(sys.argv[1]) if len(sys.argv) > 1 else 200
    queries = read_vecs(os.path.join(DATA, "sift_query.fvecs"), np.float32)[:n_queries]
    truth = read_vecs(os.path.join(DATA, "sift_groundtruth.ivecs"), np.int32)[:n_queries, :K]

    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET enable_seqscan = off")   # session 级(autocommit 下 SET LOCAL 无效)

    gucs = {"dann": "dann.probes", "dann_rq": "dann.probes", "ivfflat": "ivfflat.probes"}
    tables = {"dann": "sift_base", "dann_rq": "sift_rq", "ivfflat": "sift_pgv"}

    for engine in ("dann", "dann_rq", "ivfflat"):
        print(f"\n== {engine} (nlist=1000, {n_queries} queries, recall@{K}) ==")
        print(f"{'probes':>6} {'recall':>8} {'avg_ms':>9} {'p50_ms':>9} {'p99_ms':>9}")
        for probes in PROBES:
            cur.execute(f"SET {gucs[engine]} = {probes}")
            for i in range(WARMUP):
                run_one(cur, queries[i], tables[engine])
            lat = []
            hits = 0
            for i in range(n_queries):
                rows, dt = run_one(cur, queries[i], tables[engine])
                lat.append(dt)
                hits += len(set(rows) & set(truth[i].tolist()))
            recall = hits / (n_queries * K)
            lat = np.array(lat)
            print(f"{probes:>6} {recall:>8.4f} {lat.mean():>9.2f} "
                  f"{np.percentile(lat, 50):>9.2f} {np.percentile(lat, 99):>9.2f}")

    cur.execute("SELECT pg_size_pretty(pg_relation_size('sift_dann')), "
                "pg_size_pretty(pg_relation_size('sift_rq_idx')), "
                "pg_size_pretty(pg_relation_size('sift_ivfflat')), "
                "pg_size_pretty(pg_relation_size('sift_base'))")
    dann_sz, rq_sz, ivfflat_sz, table_sz = cur.fetchone()
    print(f"\nindex size: dann_sq8={dann_sz}, dann_rabitq={rq_sz}, "
          f"ivfflat={ivfflat_sz}, table={table_sz}")

    cur.close()
    conn.close()


def run_one(cur, qvec, table):
    q = ",".join(str(x) for x in qvec)
    t0 = time.perf_counter()
    cur.execute(f"SELECT id FROM {table} ORDER BY embedding <-> '[{q}]' LIMIT {K}")
    rows = [r[0] for r in cur.fetchall()]
    dt = (time.perf_counter() - t0) * 1000.0
    return rows, dt


if __name__ == "__main__":
    main()
