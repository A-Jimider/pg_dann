#!/usr/bin/env python3
"""SIFT1M 加载脚本:解析 fvecs → COPY 进 PG(psycopg2 copy_from)。

用法:.venv/bin/python bench/load_sift.py
"""
import io
import os
import numpy as np
import psycopg2

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "sift")
DSN = os.environ.get("PG_DANN_DSN", "host=/tmp port=5432 dbname=postgres")


def read_fvecs(path):
    """解析 .fvecs:每行 = [dim(int32) | dim×float32]"""
    raw = np.fromfile(path, dtype=np.int32)
    dim = raw[0]
    assert dim == 128, f"unexpected dim {dim}"
    arr = raw.reshape(-1, dim + 1)
    assert (arr[:, 0] == dim).all(), "fvecs format mismatch"
    return arr[:, 1:].view(np.float32)  # 字节级重解释,不是数值转换


def main():
    print("reading base vectors...")
    base = read_fvecs(os.path.join(DATA, "sift_base.fvecs"))
    print(f"  {base.shape[0]} x {base.shape[1]}")

    conn = psycopg2.connect(DSN)
    conn.autocommit = False
    cur = conn.cursor()

    cur.execute("CREATE EXTENSION IF NOT EXISTS vector")
    cur.execute("DROP TABLE IF EXISTS sift_base")
    cur.execute("CREATE TABLE sift_base (id int PRIMARY KEY, embedding vector(128))")

    print("loading via COPY...")
    batch = 50000
    for start in range(0, base.shape[0], batch):
        buf = io.StringIO()
        for i in range(start, min(start + batch, base.shape[0])):
            buf.write(f"{i}\t[{','.join(str(x) for x in base[i])}]\n")
        buf.seek(0)
        cur.copy_from(buf, "sift_base", columns=("id", "embedding"))
        conn.commit()
        print(f"  {min(start + batch, base.shape[0])} / {base.shape[0]}")

    cur.execute("SELECT count(*) FROM sift_base")
    print(f"loaded: {cur.fetchone()[0]} rows")
    cur.close()
    conn.close()


if __name__ == "__main__":
    main()
