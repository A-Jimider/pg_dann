#!/usr/bin/env python3
"""bge-m3 100k recall/延迟对比:cosine(`<=>`),dann(dann_cosine_ops) vs ivfflat。

查询 = 底库每 200 行一条(与 gt_bgem3.py 同规则),gt = gt_500q.ivecs 前 10。
表:bgem3_base(dann)/ bgem3_pgv(ivfflat),缺谁跳谁。
用法:.venv/bin/python bench/recall_bgem3.py [n_queries] [engine 逗号列表]
"""
import os
import sys
import time
import numpy as np
import psycopg2

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bgem3")
DSN = os.environ.get("PG_DANN_DSN", "host=/tmp port=5432 dbname=postgres")
K = 10
PROBES = [1, 5, 10, 20, 50]
WARMUP = 3
QSTEP = 200

ENGINES = {
    "dann_rq": ("bgem3_base", "dann.probes", "SET dann.rerank_factor = 40"),
    "ivfflat": ("bgem3_pgv", "ivfflat.probes", None),
}


def read_fvecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    arr = raw.reshape(-1, dim + 1)
    return arr[:, 1:].view(np.float32)


def read_ivecs(path, n=None):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    arr = raw.reshape(-1, dim + 1)
    return arr[:n, 1:] if n else arr[:, 1:]


def run_one(cur, qvec, table):
    q = ",".join(str(x) for x in qvec)
    t0 = time.perf_counter()
    cur.execute(f"SELECT id FROM {table} ORDER BY embedding <=> '[{q}]' LIMIT {K}")
    rows = [r[0] for r in cur.fetchall()]
    return rows, (time.perf_counter() - t0) * 1000.0


def main():
    n_queries = int(sys.argv[1]) if len(sys.argv) > 1 else 500
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    base = read_fvecs(os.path.join(DATA, "base.fvecs"))
    queries = base[::QSTEP][:n_queries]
    truth = read_ivecs(os.path.join(DATA, "gt_500q.ivecs"), n_queries)[:, :K]
    nq = queries.shape[0]

    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET enable_seqscan = off")

    for engine, (table, guc, extra) in ENGINES.items():
        if only and engine not in only:
            continue
        cur.execute(f"SELECT to_regclass('{table}')")
        if cur.fetchone()[0] is None:
            print(f"\n== {engine}: 表 {table} 不存在,跳过 ==")
            continue
        print(f"\n== {engine} ({table}, {nq} queries, recall@{K}, cosine) ==", flush=True)
        print(f"{'probes':>6} {'recall':>8} {'avg_ms':>9} {'p50_ms':>9} {'p99_ms':>9}")
        for probes in PROBES:
            cur.execute(f"SET {guc} = {probes}")
            if extra:
                cur.execute(extra)
            for i in range(WARMUP):
                run_one(cur, queries[i], table)
            lat = []
            hits = 0
            for i in range(nq):
                rows, dt = run_one(cur, queries[i], table)
                lat.append(dt)
                hits += len(set(rows) & set(truth[i].tolist()))
            recall = hits / (nq * K)
            lat = np.array(lat)
            print(f"{probes:>6} {recall:>8.4f} {lat.mean():>9.2f} "
                  f"{np.percentile(lat, 50):>9.2f} {np.percentile(lat, 99):>9.2f}",
                  flush=True)

    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
