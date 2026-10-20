#!/usr/bin/env python3
"""W10 并发压测(DESIGN.md §4.7 验收):32 并发 insert+KNN 混合,无错无死锁;
随后 DELETE + VACUUM,验证 recall 指纹与精确序一致。

用法:.venv/bin/python bench/conc_test.py [seconds]
"""
import os
import sys
import threading
import time
import numpy as np
import psycopg2

DSN = os.environ.get("PG_DANN_DSN", "host=/tmp port=5432 dbname=postgres")
DIM = 128
BASE_ROWS = 20000
N_INSERTERS = 16
N_QUERIERS = 16
TABLE = "conc_t"

stop_flag = threading.Event()
errors = []
errors_lock = threading.Lock()
inserted = threading.Semaphore(0)   # 计数器(成功插入数)
inserted_ids = []                   # (tid, k) 供最终行数校验
ids_lock = threading.Lock()


def vec_lit(i):
    """确定性向量:与 installcheck 的 sin 数据同族"""
    j = np.arange(1, DIM + 1)
    arr = np.sin(i * 0.371 + j * 0.917) * 100.0
    return "[" + ",".join(str(float(x)) for x in arr) + "]"


def note_err(msg):
    with errors_lock:
        errors.append(msg)
        if len(errors) < 10:
            print("ERROR:", msg[:200])


def inserter(tid, per_thread):
    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    cur = conn.cursor()
    for k in range(per_thread):
        if stop_flag.is_set():
            break
        rowid = 1000000 + tid * per_thread + k
        try:
            cur.execute(f"INSERT INTO {TABLE} VALUES ({rowid}, '{vec_lit(rowid)}')")
            with ids_lock:
                inserted_ids.append(rowid)
        except Exception as e:
            note_err(f"inserter {tid}: {e}")
            break
    cur.close()
    conn.close()


def querier(tid):
    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    cur = conn.cursor()
    cur.execute("SET enable_seqscan = off")
    cur.execute("SET dann.probes = 10")
    n = 0
    while not stop_flag.is_set():
        qid = 1 + (tid * 7919 + n * 131) % BASE_ROWS
        try:
            cur.execute(f"SELECT id FROM {TABLE} ORDER BY embedding <-> '{vec_lit(qid)}' LIMIT 10")
            cur.fetchall()
            n += 1
        except Exception as e:
            note_err(f"querier {tid}: {e}")
            break
    print(f"  querier {tid}: {n} queries")
    cur.close()
    conn.close()


def fingerprint(cur, note):
    cur.execute(f"SET enable_seqscan = off")
    cur.execute(f"SET dann.probes = 100")
    cur.execute(f"SELECT count(*), sum(id), min(id), max(id) FROM (SELECT id FROM {TABLE} ORDER BY embedding <-> '{vec_lit(42)}' LIMIT 10) s")
    idx = cur.fetchone()
    cur.execute("SET enable_seqscan = on")
    cur.execute("SET enable_indexscan = off")
    cur.execute(f"SELECT count(*), sum(id), min(id), max(id) FROM (SELECT id FROM {TABLE} ORDER BY embedding <-> '{vec_lit(42)}' LIMIT 10) s")
    exact = cur.fetchone()
    cur.execute("RESET enable_seqscan")
    cur.execute("RESET enable_indexscan")
    print(f"  fingerprint[{note}]: idx={idx} exact={exact} {'MATCH' if idx == exact else 'MISMATCH'}")
    return idx == exact


def main():
    seconds = int(sys.argv[1]) if len(sys.argv) > 1 else 45
    conn = psycopg2.connect(DSN)
    conn.autocommit = True
    cur = conn.cursor()

    print(f"setup: {BASE_ROWS} base rows + rabitq index")
    cur.execute(f"DROP TABLE IF EXISTS {TABLE}")
    cur.execute(f"CREATE TABLE {TABLE} (id int, embedding vector({DIM}))")
    cur.execute("SELECT 1 FROM generate_series(1,1)")  # keep session warm
    rows = ",".join(f"({i}, '{vec_lit(i)}')" for i in range(1, 51))
    cur.execute(f"INSERT INTO {TABLE} VALUES {rows}")
    # 批量灌基础数据
    for start in range(51, BASE_ROWS + 1, 1000):
        vals = ",".join(f"({i}, '{vec_lit(i)}')" for i in range(start, min(start + 1000, BASE_ROWS + 1)))
        cur.execute(f"INSERT INTO {TABLE} VALUES {vals}")
    cur.execute(f"CREATE INDEX {TABLE}_idx ON {TABLE} USING dann (embedding dann_l2_ops) WITH (nlist = 100)")

    print(f"stress: {N_INSERTERS} inserters + {N_QUERIERS} queriers, {seconds}s")
    t0 = time.time()
    threads = []
    per_thread = 5000   # 上限;跑满时间即止
    for t in range(N_INSERTERS):
        threads.append(threading.Thread(target=inserter, args=(t, per_thread)))
    for t in range(N_QUERIERS):
        threads.append(threading.Thread(target=querier, args=(t,)))
    for th in threads:
        th.start()
    time.sleep(seconds)
    stop_flag.set()
    for th in threads:
        th.join()
    dt = time.time() - t0

    n_ins = len(inserted_ids)
    print(f"  inserts done: {n_ins} ({n_ins/dt:.0f}/s)")
    cur.execute(f"SELECT count(*) FROM {TABLE}")
    cnt = cur.fetchone()[0]
    expect = BASE_ROWS + n_ins
    print(f"  table count: {cnt} (expect {expect})")
    assert cnt == expect, f"row count mismatch: {cnt} != {expect}"
    assert not errors, f"{len(errors)} errors during stress"

    assert fingerprint(cur, "post-stress"), "fingerprint mismatch after stress"

    print("DELETE 25% + VACUUM ...")
    cur.execute(f"DELETE FROM {TABLE} WHERE id % 4 = 0")
    cur.execute(f"VACUUM {TABLE}")
    cur.execute(f"SELECT sum(entries) FROM dann_list_stats('{TABLE}_idx')")
    entries = cur.fetchone()[0]
    cur.execute(f"SELECT count(*) FROM {TABLE}")
    live = cur.fetchone()[0]
    print(f"  entries={entries} live={live}")
    assert entries == live, f"index entries {entries} != live rows {live}"
    assert fingerprint(cur, "post-vacuum"), "fingerprint mismatch after vacuum"

    cur.execute(f"DROP TABLE {TABLE}")
    cur.close()
    conn.close()
    print("W10 concurrency: ALL PASS")


if __name__ == "__main__":
    main()
