#!/usr/bin/env python3
"""W12 完整 benchmark:recall–QPS 帕累托网格(单线程)+ 32 并发代表点。

引擎与参数轴:
  dann_sq8   sift_base/sift_dann      probes × γ=20
  dann_rq    sift_rq/sift_rq_idx      probes × γ=40
  ivfflat    sift_pgv/sift_ivfflat    probes
  hnsw       sift_hnsw/sift_hnsw_idx  ef_search

用法:
  .venv/bin/python bench/pareto.py [n_queries]          # 单线程网格
  .venv/bin/python bench/pareto.py conc [seconds]       # 32 并发代表点
"""
import csv
import os
import sys
import threading
import time
import numpy as np
import psycopg2

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sift")
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
DSN = os.environ.get("PG_DANN_DSN", "host=/tmp port=5432 dbname=postgres")
K = 10
WARMUP = 3

GRID = {
    # name: (table, [(set_sql, label), ...])
    "dann_sq8": ("sift_base", [(f"SET dann.probes={p};SET dann.rerank_factor=20", f"probes={p}")
                               for p in (1, 5, 10, 20, 50)]),
    "dann_rq": ("sift_rq", [(f"SET dann.probes={p};SET dann.rerank_factor=40", f"probes={p}")
                            for p in (1, 5, 10, 20, 50)]),
    "ivfflat": ("sift_pgv", [(f"SET ivfflat.probes={p}", f"probes={p}")
                             for p in (1, 5, 10, 20, 50)]),
    "hnsw": ("sift_hnsw", [(f"SET hnsw.ef_search={e}", f"ef={e}")
                           for e in (10, 20, 40, 80, 160, 320)]),
}


def read_vecs(path, dtype):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    arr = raw.reshape(-1, dim + 1)
    assert (arr[:, 0] == dim).all()
    return arr[:, 1:].view(dtype)


def run_one(cur, qvec, table):
    q = ",".join(str(x) for x in qvec)
    t0 = time.perf_counter()
    cur.execute(f"SELECT id FROM {table} ORDER BY embedding <-> '[{q}]' LIMIT {K}")
    rows = [r[0] for r in cur.fetchall()]
    return rows, (time.perf_counter() - t0) * 1000.0


def single_thread(n_queries):
    queries = read_vecs(os.path.join(DATA, "sift_query.fvecs"), np.float32)[:n_queries]
    truth = read_vecs(os.path.join(DATA, "sift_groundtruth.ivecs"), np.int32)[:n_queries, :K]

    os.makedirs(OUT, exist_ok=True)
    csv_path = os.path.join(OUT, f"pareto_sift1m_{n_queries}q.csv")
    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET enable_seqscan = off")

    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["engine", "param", "recall@10", "avg_ms", "p50_ms", "p99_ms"])
        for engine, (table, points) in GRID.items():
            print(f"\n== {engine} ==")
            print(f"{'param':>10} {'recall':>8} {'avg_ms':>9} {'p50_ms':>9} {'p99_ms':>9}")
            for set_sql, label in points:
                for stmt in set_sql.split(";"):
                    cur.execute(stmt)
                for i in range(WARMUP):
                    run_one(cur, queries[i], table)
                lat = []
                hits = 0
                for i in range(n_queries):
                    rows, dt = run_one(cur, queries[i], table)
                    lat.append(dt)
                    hits += len(set(rows) & set(truth[i].tolist()))
                recall = hits / (n_queries * K)
                lat = np.array(lat)
                row = (recall, lat.mean(), np.percentile(lat, 50),
                       np.percentile(lat, 99))
                print(f"{label:>10} {row[0]:>8.4f} {row[1]:>9.2f} {row[2]:>9.2f} {row[3]:>9.2f}")
                w.writerow([engine, label, f"{row[0]:.4f}", f"{row[1]:.3f}",
                            f"{row[2]:.3f}", f"{row[3]:.3f}"])
    print(f"\nwrote {csv_path}")
    cur.close()
    conn.close()


# ---- 32 并发代表点 ----

CONC_POINTS = [
    ("dann_sq8", "sift_base", "SET dann.probes=10;SET dann.rerank_factor=20", "probes=10,g=20"),
    ("dann_rq", "sift_rq", "SET dann.probes=10;SET dann.rerank_factor=40", "probes=10,g=40"),
    ("ivfflat", "sift_pgv", "SET ivfflat.probes=10", "probes=10"),
    ("hnsw", "sift_hnsw", "SET hnsw.ef_search=40", "ef=40"),
    # 高 recall 对位:recall≈0.99 档
    ("dann_rq", "sift_rq", "SET dann.probes=50;SET dann.rerank_factor=40", "probes=50,g=40"),
    ("ivfflat", "sift_pgv", "SET ivfflat.probes=50", "probes=50"),
    ("hnsw", "sift_hnsw", "SET hnsw.ef_search=160", "ef=160"),
]
NTHREADS = 32


def concurrent(seconds):
    queries = read_vecs(os.path.join(DATA, "sift_query.fvecs"), np.float32)[:1000]
    truth = read_vecs(os.path.join(DATA, "sift_groundtruth.ivecs"), np.int32)[:1000, :K]
    stop = threading.Event()

    for engine, table, set_sql, label in CONC_POINTS:
        lat_all = []
        lat_lock = threading.Lock()
        hits = [0]
        errors = []

        def worker(tid):
            try:
                conn = psycopg2.connect(DSN)
                conn.autocommit = True
                cur = conn.cursor()
                cur.execute("SET enable_seqscan = off")
                for stmt in set_sql.split(";"):
                    cur.execute(stmt)
                n = 0
                while not stop.is_set():
                    i = (tid * 131 + n) % 1000
                    rows, dt = run_one(cur, queries[i], table)
                    h = len(set(rows) & set(truth[i].tolist()))
                    with lat_lock:
                        lat_all.append(dt)
                        hits[0] += h
                    n += 1
                cur.close()
                conn.close()
            except Exception as e:
                errors.append(str(e))

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(NTHREADS)]
        hits[0] = 0
        lat_all.clear()
        stop.clear()
        t0 = time.time()
        for th in threads:
            th.start()
        time.sleep(seconds)
        stop.set()
        for th in threads:
            th.join()
        dt = time.time() - t0
        lat = np.array(lat_all)
        nq = len(lat)
        if nq == 0:
            print(f"{engine:>9} {label:>14}: 0 queries; errors: {errors[:2]}")
            continue
        qps = nq / dt
        # recall 以实际命中的查询数折算
        rec = hits[0] / (nq * K) if nq else 0.0
        print(f"{engine:>9} {label:>14}: qps={qps:8.1f} avg={lat.mean():7.2f}ms "
              f"p99={np.percentile(lat, 99):7.2f}ms recall={rec:.4f} "
              f"n={nq} err={len(errors)}")
        if errors:
            print(f"    first error: {errors[0][:150]}")


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "conc":
        seconds = int(sys.argv[2]) if len(sys.argv) > 2 else 20
        print(f"== 32 并发代表点({seconds}s/点)==")
        concurrent(seconds)
    else:
        single_thread(int(sys.argv[1]) if len(sys.argv) > 1 else 200)


if __name__ == "__main__":
    main()
