#!/usr/bin/env bash
# bench/coldhot.sh — W12 冷热分组(§6):每组 = 冷(restart + drop_caches)
# 后第一遍 vs 紧接着第二遍(热)。高 recall 档代表点(同 32 并发组)。
set -u
cd "$(dirname "$0")/.."
PG="${PG_DANN_PG:-/usr/local/pgsql}"   # PG 安装前缀(含 bin/ 与 data/)
PSQL="$PG/bin/psql -h ${PGHOST:-/tmp} -p ${PGPORT:-5432} -d postgres"
PY=.venv/bin/python
NQ=100   # 冷态每条查询都是真实磁盘读,100 条足够稳定

restart_cold() {
    "$PG/bin/pg_ctl" -D "$PG/data" -l /tmp/pgdann_ch.log restart >/dev/null 2>&1
    for i in $(seq 1 60); do
        "$PG/bin/pg_isready" -h "${PGHOST:-/tmp}" -p "${PGPORT:-5432}" >/dev/null 2>&1 && break
        sleep 1
    done
    sync
    sudo -n sh -c 'echo 3 > /proc/sys/vm/drop_caches'
    sleep 1
}

run_point() {  # $1=table $2=set_sql $3=label
    local cold hot
    cold=$($PY bench/qtime.py "$1" $NQ "$2")
    hot=$($PY bench/qtime.py "$1" $NQ "$2")
    printf "%-18s 冷: avg=%7.2f p50=%7.2f p99=%7.2f r=%.4f | 热: avg=%6.2f p50=%6.2f p99=%6.2f r=%.4f\n" \
        "$3" $cold $hot
}

echo "== 冷/热分组(SIFT1M,${NQ} 查询/遍;avg/p99 ms,高 recall 档) =="
for spec in \
    "sift_rq|SET dann.probes=50;SET dann.rerank_factor=40|dann_rq p50/g40" \
    "sift_base|SET dann.probes=50;SET dann.rerank_factor=20|dann_sq8 p50/g20" \
    "sift_pgv|SET ivfflat.probes=50|ivfflat p50" \
    "sift_hnsw|SET hnsw.ef_search=160|hnsw ef160"
do
    IFS='|' read -r tbl guc label <<< "$spec"
    restart_cold
    run_point "$tbl" "$guc" "$label"
done
