#!/usr/bin/env python3
"""bigann-10M 子集加载:bvecs(uint8,128d)→ COPY 进 PG。

用法:.venv/bin/python bench/load_bigann.py [n_rows] [start_row]
  start_row > 0 时续载:不 DROP 表,从 start_row 行接着插(要求已载 0..start_row-1)。
数据:bench/data/bigann/base10m.bvecs(hf-mirror SkyTbac/ANNS_BigANN100M 前 10M 行)
内存:按批从磁盘读(50k 行 ≈ 6.6MB/批),不整文件常驻。
"""
import io
import os
import sys
import numpy as np
import psycopg2

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bigann")
DSN = os.environ.get("PG_DANN_DSN", "host=/tmp port=5432 dbname=postgres")
BATCH = 50000
ROW_BYTES = 132  # bvecs: [dim(int32) | dim×uint8]


def main():
    n_rows = int(sys.argv[1]) if len(sys.argv) > 1 else 10_000_000
    start_row = int(sys.argv[2]) if len(sys.argv) > 2 else 0
    path = os.path.join(DATA, "base10m.bvecs")

    conn = psycopg2.connect(DSN)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    if start_row == 0:
        cur.execute("DROP TABLE IF EXISTS bigann_base")
        cur.execute("CREATE TABLE bigann_base (id int PRIMARY KEY, embedding vector(128))")
        conn.commit()

    for start in range(start_row, n_rows, BATCH):
        end = min(start + BATCH, n_rows)
        raw8 = np.fromfile(path, dtype=np.uint8,
                           count=ROW_BYTES * (end - start),
                           offset=ROW_BYTES * start)
        raw8 = raw8.reshape(-1, ROW_BYTES)
        dims = raw8[:, :4].view(np.int32)
        assert (dims == 128).all(), f"bvecs dim mismatch at row {start}"
        base = raw8[:, 4:]     # (m, 128) uint8

        buf = io.StringIO()
        for i in range(end - start):
            buf.write(f"{start + i}\t[{','.join(str(int(x)) for x in base[i])}]\n")
        buf.seek(0)
        cur.copy_from(buf, "bigann_base", columns=("id", "embedding"))
        conn.commit()
        print(f"  {end} / {n_rows}", flush=True)

    cur.execute("SELECT count(*) FROM bigann_base")
    print(f"loaded: {cur.fetchone()[0]} rows")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
