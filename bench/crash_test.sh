#!/usr/bin/env bash
# bench/crash_test.sh — W9 崩溃恢复测试矩阵(DESIGN.md §4.6)
#
# 注入点(kill -9 postmaster 进程组,模拟整机掉电):
#   (a) 构建中途:CREATE INDEX 进行中杀 → 恢复后事务回滚干净、可重建
#   (b) 批量插入中途:多批已提交 + 一批在飞 → 恢复后已提交数据(堆与索引)
#       一致、索引可继续插入
#   (c) VACUUM 中途:恢复后索引一致可用
#
# 一致性判据:同一张表,probes=nlist + rerank 的索引扫描聚合指纹 ==
# 强制 seqscan 的精确指纹(结构相等,不依赖 recall 噪声)。
#
# 用法:bench/crash_test.sh   (需 sudo 用于 kill -9;会重启测试实例!
# 连接/安装路径从环境变量读:PG_DANN_PG / PGHOST / PGPORT)
set -u
cd "$(dirname "$0")/.."

PG="${PG_DANN_PG:-/usr/local/pgsql}"   # PG 安装前缀(含 bin/ 与 data/)
DATA=$PG/data
PSQL="$PG/bin/psql -h ${PGHOST:-/tmp} -p ${PGPORT:-5432}"
DB=pgdann_crash
# 查询向量用真实数据行(id=42):均匀向量对本数据集是病态近平局
# (所有点几乎等距,任何估计序都会打乱),不产生有效一致性信号
QVEC="(SELECT embedding FROM t WHERE id = 42)"
QVEC_SIFT="(SELECT embedding FROM sift_base WHERE id = 42)"
FAIL=0

note() { echo; echo "== $* =="; }
fail() { echo "FAIL: $*"; FAIL=1; }

# kill -9 不清理 SysV 共享内存段;孤儿段会挡重启。等 nattch 归零后删除。
# postmaster.pid 第 7 行 = "shmem_key shmem_id"
cleanup_stale_shmem() {
    local shmid nattch i
    [ -f "$DATA/postmaster.pid" ] || return 0
    shmid=$(sed -n 7p "$DATA/postmaster.pid" | awk '{print $2}')
    [ -z "${shmid:-}" ] && return 0
    for i in $(seq 1 20); do
        nattch=$(ipcs -m | awk -v id="$shmid" '$2==id {print $6}')
        [ -z "$nattch" ] && return 0                      # 段已消失
        if [ "$nattch" = "0" ]; then
            ipcrm -m "$shmid" 2>/dev/null
            echo "  removed orphan shmem id=$shmid"
            return 0
        fi
        sleep 0.5
    done
    echo "  WARN: shmem $shmid nattch=$nattch 未归零"
}

kill_cluster() {
    local pm_pid p
    pm_pid=$(head -1 "$DATA/postmaster.pid")
    echo "  kill -9 postmaster $pm_pid + 全部本集群进程"
    # 本机的 backend 是独立进程组(pgid=pid),杀 postmaster 组不够,
    # 按 /proc/exe 路径逐个点名(严格限定本 install 目录,不误伤别的集群)
    sudo -n kill -9 "$pm_pid" 2>/dev/null
    for p in $(pgrep -f postgres); do
        case "$(readlink /proc/$p/exe 2>/dev/null)" in
            "$PG"/bin/*) sudo -n kill -9 "$p" 2>/dev/null ;;
        esac
    done
    sleep 2
    cleanup_stale_shmem
}

start_cluster() {
    local round i
    for round in 1 2 3; do
        "$PG/bin/pg_ctl" -D "$DATA" -l /tmp/pgdann_crash.log start >/dev/null 2>&1
        for i in $(seq 1 15); do
            "$PG/bin/pg_isready" -h "${PGHOST:-/tmp}" -p "${PGPORT:-5432}" >/dev/null 2>&1 && return 0
            sleep 1
        done
        cleanup_stale_shmem
    done
    echo "cluster did not come back"; exit 1
}

# 索引指纹 vs 精确指纹(两个 GUC 切换),输出 "idx=... exact=..." 供比较
fingerprint() {  # $1=table $2=probes
    $PSQL -d $DB -tA <<EOF
SET enable_seqscan = off;
SET dann.probes = $2;
SELECT 'idx=' || count(*) || '/' || sum(id) || '/' || min(id) || '/' || max(id)
FROM (SELECT id FROM $1 ORDER BY embedding <-> $QVEC LIMIT 10) s;
SET enable_seqscan = on; SET enable_indexscan = off;
SELECT 'exact=' || count(*) || '/' || sum(id) || '/' || min(id) || '/' || max(id)
FROM (SELECT id FROM $1 ORDER BY embedding <-> $QVEC LIMIT 10) s;
EOF
}

note "setup: fresh db + 50k 行确定性数据 + dann 索引"
$PSQL -d postgres -c "DROP DATABASE IF EXISTS $DB" >/dev/null
$PSQL -d postgres -c "CREATE DATABASE $DB" >/dev/null
$PSQL -d $DB >/dev/null <<'EOF'
CREATE EXTENSION vector;
CREATE EXTENSION pg_dann;
CREATE TABLE t (id int, embedding vector(128));
INSERT INTO t
SELECT i, (ARRAY(SELECT (sin(i*0.371 + j*0.917)*100)::float8
                 FROM generate_series(1,128) j))::vector
FROM generate_series(1, 50000) i;
CREATE INDEX t_idx ON t USING dann (embedding dann_l2_ops)
  WITH (nlist = 50, quantizer = 'rabitq');
CREATE TABLE progress (batch int);
EOF
echo "  base fingerprint:"; fingerprint t 50

# ---------- (a) 构建中途 ----------
note "(a) 构建中途 kill:sift_base 100 万行,构建活跃 ≥2s 后断电(至多 3 次尝试)"
AOK=0
for attempt in 1 2 3; do
    $PSQL -d postgres -c "DROP INDEX IF EXISTS sift_crash_idx" >/dev/null 2>&1
    ( $PSQL -d postgres -c "CREATE INDEX sift_crash_idx ON sift_base USING dann (embedding dann_l2_ops) WITH (nlist=1000, quantizer='rabitq')" >/dev/null 2>&1 ) &
    # 等构建活跃满 2s(落在 Pass B/C 窗口,而非启动/收尾)
    for i in $(seq 1 100); do
        ACTIVE=$($PSQL -d postgres -tA -c "SELECT coalesce(extract(epoch from now()-query_start)::int,0) FROM pg_stat_activity WHERE query LIKE 'CREATE INDEX sift_crash_idx%' AND state='active'" 2>/dev/null)
        [ "${ACTIVE:-0}" -ge 2 ] && break
        sleep 0.1
    done
    kill_cluster
    start_cluster
    STATE=$($PSQL -d postgres -tA -c "SELECT coalesce((SELECT indisready::text || '/' || indisvalid::text FROM pg_index i JOIN pg_class c ON c.oid = i.indexrelid WHERE c.relname = 'sift_crash_idx'), 'gone')")
    echo "  attempt $attempt: 恢复后索引状态 $STATE"
    if [ "$STATE" = "true/true" ]; then
        # 构建在断电前已提交(也是合法崩溃场景):验证 WAL 化的索引完整可用,然后重试
        FP=$($PSQL -d postgres -tA -c "SET enable_seqscan=off; SET dann.probes=50; SELECT count(*) FROM (SELECT id FROM sift_base ORDER BY embedding <-> $QVEC_SIFT LIMIT 10) s" | tail -1)
        echo "  已提交索引可查询: top10 count=$FP"
        [ "$FP" = "10" ] || { fail "(a) committed index unusable"; break; }
        continue
    fi
    # 事务回滚:索引不存在或 invalid;表完好;可重建
    [ "$STATE" = "gone" ] || [ "$STATE" = "f/f" ] || { fail "(a) leftover index state $STATE"; break; }
    CNT=$($PSQL -d postgres -tA -c "SELECT count(*) FROM sift_base")
    echo "  sift_base count: $CNT (应 1000000)"
    [ "$CNT" = "1000000" ] || fail "(a) sift_base count $CNT"
    $PSQL -d postgres -c "DROP INDEX IF EXISTS sift_crash_idx" >/dev/null
    echo "  重建索引..."
    $PSQL -d postgres -c "CREATE INDEX sift_crash_idx ON sift_base USING dann (embedding dann_l2_ops) WITH (nlist=1000, quantizer='rabitq')" >/dev/null || fail "(a) rebuild failed"
    FP=$($PSQL -d postgres -tA -c "SET enable_seqscan=off; SET dann.probes=50; SELECT count(*) FROM (SELECT id FROM sift_base ORDER BY embedding <-> $QVEC_SIFT LIMIT 10) s" | tail -1)
    echo "  重建后可查询: top10 count=$FP"
    [ "$FP" = "10" ] || fail "(a) query after rebuild"
    AOK=1
    break
done
if [ "$AOK" = "1" ]; then
    echo "  (a) 构建中途分支:回滚干净 + 重建可用 ✓"
elif [ $FAIL -eq 0 ]; then
    echo "  注意:未命中'构建中途'分支(全为已提交分支)"
fi
$PSQL -d postgres -c "DROP INDEX IF EXISTS sift_crash_idx" >/dev/null

# ---------- (b) 批量插入中途 ----------
note "(b) 批量插入中途 kill:60 批 × 1000 行(逐批提交),~3s 时断电"
( for b in $(seq 1 60); do
    $PSQL -d $DB >/dev/null 2>&1 <<EOF
INSERT INTO t
SELECT 100000 + $b * 1000 + j, (ARRAY(SELECT (sin((100000+$b*1000+j)*0.371 + k*0.917)*100)::float8 FROM generate_series(1,128) k))::vector
FROM generate_series(1, 1000) j;
INSERT INTO progress VALUES ($b);
EOF
  done ) &
LOAD_PID=$!
sleep 3          # 大概率落在中间某批
kill_cluster
wait $LOAD_PID 2>/dev/null
start_cluster
DONE=$($PSQL -d $DB -tA -c "SELECT coalesce(max(batch),0) FROM progress")
CNT=$($PSQL -d $DB -tA -c "SELECT count(*) FROM t")
echo "  已提交批次: $DONE,表行数: $CNT (应 = 50000 + 1000*$DONE)"
[ "$CNT" = "$((50000 + 1000 * DONE))" ] || fail "(b) count mismatch: $CNT vs $DONE committed"
FP=$(fingerprint t 50)
echo "  $FP"
[ "$(echo "$FP" | sed -n 1p | sed 's/idx=//')" = "$(echo "$FP" | sed -n 2p | sed 's/exact=//')" ] \
    || fail "(b) index fingerprint diverges from exact"
# 可继续插入
$PSQL -d $DB -c "INSERT INTO t VALUES (999999, (SELECT embedding FROM t WHERE id=1))" >/dev/null \
    && echo "  恢复后可继续插入 ✓" || fail "(b) insert after recovery"
FP=$(fingerprint t 50)
echo "  $FP"
[ "$(echo "$FP" | sed -n 1p | sed 's/idx=//')" = "$(echo "$FP" | sed -n 2p | sed 's/exact=//')" ] \
    || fail "(b) fingerprint diverges after post-recovery insert"

# ---------- (c) VACUUM 中途 ----------
note "(c) VACUUM 中途 kill"
$PSQL -d $DB -c "DELETE FROM t WHERE id > 100000 AND id % 2 = 0" >/dev/null  # 造死元组
( for i in $(seq 1 50); do $PSQL -d $DB -c "VACUUM t" >/dev/null 2>&1; done ) &
sleep 2
kill_cluster
start_cluster
CNT=$($PSQL -d $DB -tA -c "SELECT count(*) FROM t")
echo "  count after vacuum crash: $CNT"
FP=$(fingerprint t 50)
echo "  $FP"
[ "$(echo "$FP" | sed -n 1p | sed 's/idx=//')" = "$(echo "$FP" | sed -n 2p | sed 's/exact=//')" ] \
    || fail "(c) fingerprint diverges after vacuum crash"

note "汇总"
if [ $FAIL -eq 0 ]; then echo "W9 crash matrix: ALL PASS"; else echo "W9 crash matrix: FAILURES"; exit 1; fi
