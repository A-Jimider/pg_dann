# pg_dann 开发记录(DEVLOG)

> 一句话现状:**W1–W12 全部里程碑收官(2026-09-16),版本 0.1.0,全部
> 验收数据落盘**。本文是开发过程与当前状态的记录:里程碑时间线、环境
> 现状、构建/测试/复现入口、决策与踩坑索引、遗留事项。
> 文档分工:计划稿 DESIGN.md|竣工架构 ARCHITECTURE.md|文章
> ARTICLE.md|逐里程碑验收数据 bench/RESULTS.md|使用 README.md。

## 1. 里程碑时间线

| 里程碑 | 完成日 | 内容 | 验收(详见 RESULTS.md) |
|---|---|---|---|
| W1 | 09-08 | PGXS 骨架、AM handler stub、opclass×3 | 扩展可建 |
| W2-3 | 09-08/09 | SQ8 量化器 + k-means + SIMD(scalar/AVX2/AVX-512);golden model 对拍纪律确立 | quant-test 全绿 |
| W4-5 | 09-10 | IVF 构建/扫描落地,SIFT1M 对标 ivfflat | recall 同水位,快 1.4–2.8×;构建 19.8s vs 36.4s;mwm 64MB vs 222MB |
| W6 | 09-15 | 批量预取(io_method)+ heap rerank(γ 懒加载块)+ 延迟预算截断 | recall@10 0.9935;1ms 预算砍 45% 延迟、recall 掉 1.2pp |
| W7-8 | 09-15 | RaBitQ(旋转=置换∘FWHT∘符号翻转,种子存 metapage)+ golden 对拍 27328 项 + SIFT1M 终验 | 误差 σ·√d≈1.30–1.45 恒定;索引 39MB(vs SQ8 146MB);recall≥0.98 ✓ |
| W9 | 09-15 | 全部页写入 generic_xlog 化;崩溃恢复矩阵(kill -9×3 注入点) | 恢复后事务/行数/指纹全部精确一致 |
| W10 | 09-15 | 并发插入(metapage 锁协议)+ VACUUM bulkdelete | 32 并发 45s 零错误无死锁;VACUUM 精确删除;顺手修 NULL ORDER BY 段错误 |
| W11 | 09-15 | 加固:amcostestimate、reloption/GUC 整理、dann_scan_stats/dann_list_stats、边界用例全进 installcheck、README | 回归全绿(results/*.out,20:37) |
| W12 | 09-15/16 | 完整 benchmark(pareto/并发/冷热/minbuf/崩溃)+ bigann-10M + bge-m3 + 技术文章 | ARTICLE.md 成稿;全部 §6 指标产出 |

## 2. W12 次日(09-16)事故与恢复(重要教训)

- **事故**:minbuf.py 收尾把实例留在 `shared_buffers=2048MB`,紧接着
  load_bigann.py(当时有 1.3GB 未使用的 numpy 废读,常驻 ~2.6GB)叠加
  10M 行 COPY 写入压力,在 11GB 内存的 WSL2 上把整台 VM 跑崩。
  崩溃时 bigann_base 只提交了 650k/10M 行,minbuf 结果(只在 stdout)丢失。
- **恢复**:实例崩溃恢复正常(WAL 化有效性的顺带实证);修脚本
  (废读删除+按批读盘+断点续载;gt 分块 500k→100k,峰值 ~2GB),
  从 650k 续载到 10M;重跑 minbuf 并 tee 落盘。
- **沉淀的纪律**(已记入长期记忆):重活串行、轻操作并行 ≤3;长任务
  输出必须落盘;启动重活前查 `free -h` 与 shared_buffers;
  `pg_ctl restart` 会继承 postmaster.opts 的旧 `-c` 参数,恢复默认须
  stop+start。

## 3. 当前环境现状(2026-09-16)

**测试实例**:PG 16.13 衍生发行版(厂商定制,日志有定制 sql_limit 警告),
端口 6543,socket /tmp,shared_buffers=128MB(默认)。
i7-11800H(AVX-512),WSL2 11GB+3GB swap。
注意 gdb 无法 attach(yama ptrace_scope);崩溃调试用 `postgres --single`+gdb。

**库内表(postgres)**:

| 表 | 规模 | 索引 | 用途 |
|---|---|---|---|
| sift_base / sift_rq / sift_pgv / sift_hnsw | 各 1M×128d | dann_sq8 / dann_rabitq / ivfflat / hnsw | SIFT1M 四引擎 |
| bigann_base / bigann_pgv | 各 10M×128d(5581MB) | dann_rabitq 364MB / ivfflat 5225MB | bigann-10M(nlist/lists=4096) |
| bgem3_base / bgem3_sq8 / bgem3_pgv | 各 100k×1024d(535MB 含 TOAST) | dann_rabitq 18MB / dann_sq8 104MB / ivfflat 784MB | bge-m3 cosine(nlist/lists=320) |
| w9dbg*, w9ins, w9rq, vac2-4, dann_dbg, dann_w45, pgv_probe, rq_smoke, ts_data, tar_regression_test* | 小 | 杂 | 各里程碑现场(可清理) |

**数据文件**:bench/data/sift(SIFT1M)、bigann(base10m.bvecs 1.3GB +
query.bvecs + gt10m_1000q.ivecs)、bgem3(parquet 1.5GB + base.fvecs
410MB + gt_500q.ivecs)。下载源均为 hf-mirror.com(huggingface 被墙)。

**日志**:bench/out/(load_bigann/gt_bigann/recall_bigann_*/build_ivfflat_10m/
recall_bgem3/minbuf/pareto_sift1m_200q.csv);回归结果 results/*.out。

## 4. 构建 / 测试 / 复现入口

```bash
# 构建安装(显式指向测试实例的 pg_config)
make PG_CONFIG=/path/to/pg/install/bin/pg_config
make PG_CONFIG=/path/to/pg/install/bin/pg_config install

# 回归(串行,含边界用例;需测试实例在跑)
make installcheck PG_CONFIG=/path/to/pg/install/bin/pg_config \
     PGHOST=/tmp PGPORT=6543

# 量化器对拍 + micro-bench(不依赖 PG)
make quant-test && bench/quant_harness

# benchmark 复现(ARTICLE §7;注意 coldhot.sh/minbuf.py/crash_test.sh
# 会重启/杀实例;长跑务必 tee 落盘)。
# 脚本连接/安装路径从环境变量读:PG_DANN_DSN(Python 脚本)、
# PG_DANN_PG / PGHOST / PGPORT(shell 脚本与 minbuf.py)
.venv/bin/python bench/recall.py 200          # SIFT1M recall 网格
.venv/bin/python bench/budget.py 200          # 延迟预算
.venv/bin/python bench/conc_test.py           # 32 并发(重启实例)
.venv/bin/python bench/pareto.py              # 帕累托网格
# bigann-10M:load_bigann.py → gt_bigann.py → recall_bigann.py 1000
# bge-m3:extract_bgem3.py → load_bgem3.py → gt_bgem3.py → recall_bgem3.py
```

## 5. 关键决策与踩坑索引

按主题(细节都在 ARTICLE.md 对应章节):

| 主题 | 结论 | 出处 |
|---|---|---|
| 存储赌注 | 索引只存量化码+ctid,精确序靠回表 rerank | ARTICLE §0/§1 |
| 物理聚簇构建 | 三遍扫表、按计数连续分配、O(采样) 内存 | ARTICLE §1.2,ARCH §4 |
| 量化器双轨 | SQ8 低维 / RaBitQ 高维;auto dim<128 | ARTICLE §1.3 + §5.7 修正 |
| RaBitQ 对拍 | numpy golden 先行,码逐字节一致;fancy-indexing 与 1ulp 两个坑 | ARTICLE §2 |
| WAL 化 | generic_xlog 三坑(Finish 不解锁/hole 清零/pd_lower) | ARTICLE §3 |
| 并发协议 | named LWLock 需 preload → v1 用 metapage 锁 | ARTICLE §4 |
| minbuf 方法论 | OS page cache 兜底会稀释分辨力,hnsw 768MB vs 其余 32MB | ARTICLE §5.5 |
| 高维页效率 | ivfflat 1024d 每页只装一条 tuple,体积差 43× | ARTICLE §5.7 |
| RaBitQ γ | 语义分布需 γ≥200 或回 SQ8(估计序噪声,非聚簇) | ARTICLE §5.7 |
| 工程环境 | 定制发行版 backend 独立进程组;kill -9 后 SysV 孤儿段要 ipcrm | RESULTS W9 |

## 6. 遗留事项 / 下一步候选

**代码(v2 候选,按价值排序)**:
1. γ 自适应或构建期分布探测(bge-m3 教训,auto 判据目前只看维度)
2. list LWLock preload 版(页格式/WAL 兼容,可无感升级)
3. 定制 rmgr 降 WAL 体积(当前 WAL≈索引体积)
4. PG parallel build;PG18 read stream;NEON 实测(无 ARM 硬件)
5. CI 矩阵 PG17/18 + ASAN

**工程**:
- ~~仓库尚非 git~~ 已 git 化(2026-10-20 首次提交并推送 GitHub)
- 清理 w9*/vac*/dbg 等里程碑现场表(确认无参考价值后)
- GIST1M(960d 非 2 幂,补零路径)installcheck 已覆盖合成用例,
  真实数据可选

**文章发布前**:ARTICLE.md 数字均出自 bench/out/*.log 与
bench/RESULTS.md,可直接引用;§5.4/5.5 的冷热与 minbuf 结论已按
OS 缓存兜底做了诚实注记。
