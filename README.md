# pg_dann — 磁盘优先的向量索引 PostgreSQL 扩展

IVF(倒排聚类)+ 量化(SQ8 / RaBitQ)的 ANN 索引访问方法,全磁盘驻留、
全 WAL 化、崩溃可恢复。文档:[DESIGN.md](DESIGN.md)(计划稿)|
[ARCHITECTURE.md](ARCHITECTURE.md)(竣工架构,以代码为准)|
[ARTICLE.md](ARTICLE.md)(实现笔记与数据)|
[bench/RESULTS.md](bench/RESULTS.md)(逐里程碑验收)|
[DEVLOG.md](DEVLOG.md)(开发记录与现状)。

依赖 pgvector 的 `vector` 类型(生态兼容、对比零摩擦)。

## 快速开始

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pg_dann;

CREATE TABLE docs (id int, embedding vector(768));

-- quantizer 缺省 auto:dim < 128 → SQ8,dim ≥ 128 → RaBitQ
CREATE INDEX docs_dann ON docs USING dann (embedding)          -- 默认 L2
  WITH (nlist = 4096);
-- 或显式:USING dann (embedding dann_cosine_ops) WITH (quantizer='sq8', probes=64)

SELECT * FROM docs ORDER BY embedding <=> $1 LIMIT 10;   -- cosine
-- L2: <->   内积: <#>(降序相似度 = 升序负内积)
```

构建内存与表大小无关(采样封顶);SIFT1M(1M×128)构建 ≈ 17s(RaBitQ)
/ 20s(SQ8),索引 39MB / 146MB(ivfflat 525MB)。

## 参数

GUC(查询期,均可会话级 SET):

| GUC | 默认 | 说明 |
|---|---|---|
| `dann.probes` | 10 | 探查 list 数;越大 recall 越高越慢 |
| `dann.rerank_factor` | 20 | γ:回表重排候选 = γ×10 起跳;RaBitQ 建议 40 |
| `dann.enable_rerank` | on | off 时返回量化估计序(极致延迟,recall 略降) |
| `dann.max_latency_ms` | 0 | 延迟预算,0=不限;耗尽即截断尾部最远 list,recall 平滑降级 |
| `dann.io_method` | auto | `auto`/`fadvise` = 批量预取(PrefetchBuffer);`sync` = 同步读 |

reloption(CREATE INDEX ... WITH(...),`probes`/`rerank_factor` 为按索引覆盖):

| reloption | 默认 | 说明 |
|---|---|---|
| `nlist` | 1000 | IVF list 数(空表/小样本自动夹子) |
| `quantizer` | auto | `auto`/`rabitq`/`sq8` |
| `probes` | 0 | >0 时覆盖 `dann.probes` |
| `rerank_factor` | 0 | >0 时覆盖 `dann.rerank_factor` |

## 可观测性

```sql
SELECT * FROM dann_scan_stats();          -- (scans, truncated, lists_skipped) 会话级
SELECT dann_scan_stats_reset();
SELECT * FROM dann_list_stats('docs_dann');   -- 每 list:(listno, entries, pages, span)
-- span > pages 表示增量插入导致布局退化,退化严重建议 REINDEX
```

## 语义说明

- **relaxed order**:开 rerank 时返回按全精度精确距离排序的结果
  (与 seqscan+sort 一致,前提是候选集覆盖了真近邻);关 rerank 返回
  量化估计序。索引不提供精确保证时 planner 语义见 DESIGN.md §4.8。
- **NULL 查询向量**(`ORDER BY embedding <-> NULL`):退化为全索引扫描,
  行集合正确、序无定义(与 seqscan+sort 的未定序一致)。
- NaN/Inf 向量在写入与查询时报错拒绝(pgvector 类型层已先行拦截)。
- 死元组:删除/更新后索引中的旧条目由 VACUUM 清理;清理前查询自动跳过。

## 已知限制(v1)

- **页内紧凑删除,不做空页回收**:大量删除后索引体积不回缩,REINDEX 重建
  (与 IVFFlat 行为一致)。`dann_list_stats()` 的 span/entries 供运维参考。
- **插入串行化**:并发插入按 metapage 排他锁串行(≈IVFFlat;实测 ~1758
  行/s 稳态)。DESIGN.md §4.7 的 128 把 list LWLock 需
  `shared_preload_libraries` 支持,留作后续升级。
- **并行构建/并行扫描未接**:构建单线程(两遍扫表);`amcanparallel=false`。
- **崩溃一致性依赖 WAL**:wal_level=minimal 下构建结束 smgrimmedsync 兜底;
  崩溃恢复矩阵见 bench/RESULTS.md W9。
- PG18 read stream 预取为编译期 TODO(当前 PG16 统一 PrefetchBuffer);
  ARM NEON 未实现(scalar 兜底可用)。

## 开发与测试

```bash
make && make install                    # PGXS,PG_CONFIG 指向 PG16+
make installcheck PGPORT=6543 PGHOST=/tmp   # 回归:basic/ivf/rerank/rabitq/vacuum/edge
make quant-test                         # SQ8/RaBitQ golden model 对拍(纯 C)
.venv/bin/python bench/load_sift.py     # SIFT1M 装载(hf-mirror 数据见 bench/data)
.venv/bin/python bench/recall.py 200    # recall/延迟网格(dann SQ8 vs RaBitQ vs ivfflat)
.venv/bin/python bench/budget.py 200    # 延迟预算降级曲线
.venv/bin/python bench/conc_test.py     # 32 并发压测
bench/crash_test.sh                     # 崩溃恢复矩阵(kill -9;会重启实例!)
```

目录:`src/`(AM 回调)、`quant/`(量化器,纯 C)、`simd/`(AVX2/AVX-512
分派)、`golden/`(numpy golden model + 对拍夹具 + 误差曲线)、
`test/`(pg_regress)、`bench/`(SIFT1M 基准与验收脚本)。

## 许可证

[AGPL-3.0](LICENSE)。允许商业使用;但修改后再分发、或作为网络服务
对外提供(含云托管/SaaS),均须以同一许可证公开源码。
