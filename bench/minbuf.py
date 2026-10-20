#!/usr/bin/env python3
"""W12 最小 shared_buffers 二分(DESIGN.md §6 内存指标)。

协议:高 recall 档代表点(recall≈0.99 ≥ 0.98);每档 shared_buffers:
restart + drop_caches → 第一遍预热(buffer pool 从磁盘装) → 第二遍计时。
通过判据:第二遍 avg ≤ 1.5 × 该引擎基线(shared_buffers=2048MB 同协议)。
报每引擎"不衰减"所需最小 shared_buffers。

用法:.venv/bin/python bench/minbuf.py [n_queries_per_pass]
注意:会反复重启测试实例!连接/安装路径从环境变量读:PG_DANN_PG / PGHOST / PGPORT。
"""
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PY = os.path.join(ROOT, ".venv", "bin", "python")
QTIME = os.path.join(ROOT, "bench", "qtime.py")
PG = os.environ.get("PG_DANN_PG", "/usr/local/pgsql")  # PG 安装前缀(含 bin/ 与 data/)
DATA = os.path.join(PG, "data")
PSQL_HOST = os.environ.get("PGHOST", "/tmp")
PSQL_PORT = os.environ.get("PGPORT", "5432")

ENGINES = [
    ("dann_rq", "sift_rq", "SET dann.probes=50;SET dann.rerank_factor=40"),
    ("dann_sq8", "sift_base", "SET dann.probes=50;SET dann.rerank_factor=20"),
    ("ivfflat", "sift_pgv", "SET ivfflat.probes=50"),
    ("hnsw", "sift_hnsw", "SET hnsw.ef_search=160"),
]
SIZES_MB = [32, 48, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536, 2048]
SLOWDOWN = 1.5


def sh(cmd):
    subprocess.run(cmd, shell=True, check=False,
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


def restart_with_buffers(mb):
    sh(f"sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'")
    sh(f'"{PG}/bin/pg_ctl" -D "{DATA}" -o "-c shared_buffers={mb}MB" '
       f'-l /tmp/pgdann_minbuf.log restart')
    for _ in range(90):
        r = subprocess.run([f"{PG}/bin/pg_isready", "-h", PSQL_HOST, "-p", PSQL_PORT],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        if r.returncode == 0:
            return True
        time.sleep(1)
    return False


def measure(table, set_sql, n):
    """两遍:第一遍预热,返回第二遍 avg_ms。"""
    subprocess.run([PY, QTIME, table, str(n), set_sql],
                   stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    out = subprocess.run([PY, QTIME, table, str(n), set_sql],
                         capture_output=True, text=True)
    avg_ms = float(out.stdout.split()[0])
    return avg_ms


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 100

    print(f"== 最小 shared_buffers(第二遍 avg ≤ {SLOWDOWN}×基线;{n} 查询/遍,两遍) ==")
    print(f"{'engine':>9} {'baseline_ms':>11} {'min_MB':>7} {'at_min_ms':>10}")
    for name, table, guc in ENGINES:
        assert restart_with_buffers(2048), "restart failed at 2048MB"
        base = measure(table, guc, n)
        answer = None
        for mb in SIZES_MB:
            assert restart_with_buffers(mb), f"restart failed at {mb}MB"
            avg = measure(table, guc, n)
            ok = avg <= SLOWDOWN * base
            print(f"  {name:>9} {mb:>5}MB -> avg {avg:8.2f}ms "
                  f"(base {base:.2f}) {'ok' if ok else 'SLOW'}")
            if ok:
                answer = (mb, avg)
                break
        if answer is None:
            print(f"{name:>9} {base:>10.2f}ms {'>2048':>7} {'':>10}")
        else:
            print(f"{name:>9} {base:>10.2f}ms {answer[0]:>6}MB {answer[1]:>9.2f}ms")

    restart_with_buffers(2048)   # 收尾:恢复大缓冲(后续测试用)
    print("done (instance left at shared_buffers=2048MB)")


if __name__ == "__main__":
    main()
