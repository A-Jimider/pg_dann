#!/usr/bin/env python3
"""bge-m3 100k 加载:base.fvecs(float32 1024d)→ COPY 进 PG。

用法:.venv/bin/python bench/load_bgem3.py
表:bgem3_base(id int PK, embedding vector(1024))。
对照表 bgem3_pgv 由 SQL `CREATE TABLE ... AS SELECT` 复制(见 RESULTS.md)。
"""
import io
import os
import numpy as np
import psycopg2

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bgem3")
DSN = os.environ.get("PG_DANN_DSN", "host=/tmp port=5432 dbname=postgres")
DIM = 1024


def read_fvecs(path):
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    assert dim == DIM, f"unexpected dim {dim}"
    arr = raw.reshape(-1, dim + 1)
    assert (arr[:, 0] == dim).all(), "fvecs format mismatch"
    return arr[:, 1:].view(np.float32)


def main():
    print("reading base vectors...")
    base = read_fvecs(os.path.join(DATA, "base.fvecs"))
    print(f"  {base.shape[0]} x {base.shape[1]}")

    conn = psycopg2.connect(DSN)
    conn.autocommit = False
    cur = conn.cursor()
    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("DROP TABLE IF EXISTS bgem3_base")
    cur.execute(f"CREATE TABLE bgem3_base (id int PRIMARY KEY, embedding vector({DIM}))")

    batch = 20000
    for start in range(0, base.shape[0], batch):
        buf = io.StringIO()
        for i in range(start, min(start + batch, base.shape[0])):
            buf.write(f"{i}\t[{','.join(str(x) for x in base[i])}]\n")
        buf.seek(0)
        cur.copy_from(buf, "bgem3_base", columns=("id", "embedding"))
        conn.commit()
        print(f"  {min(start + batch, base.shape[0])} / {base.shape[0]}", flush=True)

    cur.execute("SELECT count(*) FROM bgem3_base")
    print(f"loaded: {cur.fetchone()[0]} rows")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
