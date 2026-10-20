#!/usr/bin/env python3
"""bigann-10M recall/延迟对比:pg_dann(RaBitQ/SQ8) vs pgvector IVFFlat。

协议与 recall.py(SIFT1M)一致:同查询集、同 probes 网格、recall@10 +
avg/p50/p99;每引擎独立表。查询/底库为 uint8 bvecs,groundtruth 为
gt_bigann.py 产出的 ivecs(top-100,本脚本取前 10)。
只跑库中实际存在的表(10M 是降级对比,HNSW 视内存决定)。
用法:.venv/bin/python bench/recall_bigann.py [n_queries]
"""
import os
import sys
import time
import numpy as np
import psycopg2

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bigann")
DSN = os.environ.get("PG_DANN_DSN", "host=/tmp port=5432 dbname=postgres")
K = 10
PROBES = [1, 5, 10, 20, 50, 100]
WARMUP = 3

# engine -> (table, probes GUC, 附加 SET)
ENGINES = {
    "dann_rq":  ("bigann_base", "dann.probes", "SET dann.rerank_factor = 40"),
    "dann_sq8": ("bigann_sq8",  "dann.probes", "SET dann.rerank_factor = 20"),
    "ivfflat":  ("bigann_pgv",  "ivfflat.probes", None),
}


def read_bvecs(path, n=None):
    raw = np.fromfile(path, dtype=np.uint8, count=None if n is None else 132 * n)
    arr = raw.reshape(-1, 132)
    assert (arr[:, :4].view(np.int32) == 128).all()
    return arr[:, 4:].astype(np.float32)


def read_ivecs(path, n=None):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    arr = raw.reshape(-1, dim + 1)
    assert (arr[:, 0] == dim).all()
    return arr[:n, 1:] if n else arr[:, 1:]


def run_one(cur, qvec, table):
    q = ",".join(str(int(x)) for x in qvec)
    t0 = time.perf_counter()
    cur.execute(f"SELECT id FROM {table} ORDER BY embedding <-> '[{q}]' LIMIT {K}")
    rows = [r[0] for r in cur.fetchall()]
    return rows, (time.perf_counter() - t0) * 1000.0


def main():
    n_queries = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None  # 只跑指定引擎
    queries = read_bvecs(os.path.join(DATA, "query.bvecs"), n_queries)
    truth = read_ivecs(os.path.join(DATA, "gt10m_1000q.ivecs"), n_queries)[:, :K]
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
        print(f"\n== {engine} ({table}, {nq} queries, recall@{K}) ==", flush=True)
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
