# pg_dann 设计文档：磁盘优先的向量索引 PostgreSQL 扩展

> 版本 v3.0 ｜ 2026-09-08 ｜ 面向实现的工程最终版（含 PG 内核深度集成差异点）
> 许可证：PostgreSQL License ｜ 语言：纯 C（PGXS 扩展）｜ 目标 PG 版本：16/17/18
> 一句话：**实现一个 "IVF + RaBitQ 量化、posting list 磁盘驻留" 的索引访问方法（AM），把系统真正做出来、测出来。**

---

## 1. 目标与非目标

### 核心目标（验收线）

| 目标 | 指标 |
|---|---|
| 召回 | recall@10 ≥ 0.98（rerank 后，SIFT1M / bigann-10M） |
| 内存 | 达到同等 recall + QPS 目标所需 shared_buffers ≤ pgvector HNSW 的 1/8（目标 1/16），测量方法见 §6 |
| 构建 | 构建内存与表大小无关（采样 + 流式两遍）；构建时间显著低于同机 HNSW |
| 可靠性 | 索引页全 WAL 化，构建中/插入中/查询中 kill -9 后索引一致可用 |
| 磁盘路径 | bigann-10M（数据量超内存）下可用，报告 p50/p99 |

### v1 明确不做（写死，防范围蔓延）

- 不做属性过滤（pre/post-filtering）——接口上预留 scan key 位置
- 不做图索引（HNSW/DiskANN）
- 不做多向量 / 稀疏向量 / 二进制向量类型
- SQ8 只是脚手架与 benchmark 对照组，不作为独立卖点
- 不支持 PG15 及以下（WAL custom rmgr API 需 15+；AIO 路径需 18）

---

## 2. 与现有实现的关系：差异化与避嫌

### 2.1 对照表

| | pg_dann | pgvector | VectorChord | pgvectorscale |
|---|---|---|---|---|
| 语言 / 框架 | 纯 C / PGXS | 纯 C | Rust / pgrx | Rust / pgrx |
| 许可证 | PostgreSQL License | PostgreSQL License | AGPLv3 或 ELv2 双授权 | 其自有许可 |
| 算法 | IVF + RaBitQ（量化器可插拔） | HNSW / IVFFlat | IVF + RaBitQ | StreamingDiskANN（图） |
| 向量类型 | 复用 pgvector `vector` | 自有 `vector` | 兼容 pgvector 类型 | 兼容 pgvector 类型 |
| posting 布局 | 两遍构建，list 物理聚簇 | — | 未公开此设计 | — |
| 构建内存 | O(采样集)，与表大小无关 | O(全表) | — | — |
| IO 模型 | PG18 read stream / AIO 原生预取 | 常规 buffer 读 | 自有 IO 层 | 自有 IO 层 |

注：VectorChord 内部实现细节以其公开文档与文章为准，本表不推测其未公开的设计。

### 2.2 差异化的技术决策（实现路径层面的真实差异）

1. **两遍式构建 → list 物理聚簇**：第一遍扫表只按质心归属计数，第二遍按计数结果为每个 list **连续预分配** posting 页。查询期单个 list 的读取是纯顺序 IO，批量预取可以整段发起。这是"磁盘优先"的核心设计，不是附加优化。
2. **量化器可插拔抽象**：统一 quantizer 接口（encode / estimate / factors），SQ8 与 RaBitQ 实现同一接口、reloption 可切换。SQ8 先行打通全管线，RaBitQ 作为第二个实现插入——同一 AM、两套量化器，benchmark 对照零额外成本。
3. **rerank 走 heap + ctid 排序批量回表**：v1 不在索引内冗余存储全精度向量（决策记录见 §4.1-D1），候选 ctid 先排序再批量回表，把随机 heap 读压成近似顺序读。
4. **构建内存与表大小无关**：质心训练用采样（64–256 条/质心，封顶内存预算），assignment 两遍流式扫表。对照：pgvector IVFFlat 构建需全表进内存。
5. **PG18 read stream / AIO 原生集成**：posting 批量预取优先走 PG18 读路径（`io_method=worker/io_uring` 下生效），降级 `posix_fadvise`。注意 PG18 AIO 只覆盖读路径，对本项目足够。
6. **维度自适应量化器选择（`quantizer = 'auto'`，默认）**：RaBitQ 估计误差 ∝ 1/√d，低维时 1-bit 码噪声大、SQ8 反而更准。接口统一后按维度自动选择是顺手的事——别家没有第二个量化器可切，自然没人做这个决策。
7. **延迟预算驱动的扫描截断（`dann.max_latency_ms`）**：预算耗尽即停，list 按质心距离有序扫描，尾部截断只损失最远的 list，recall 平滑降级而非超时报错。生产上"宁可少召回也不能超 p99"是硬诉求。
8. **ANN 专项可观测性**：各家向量索引对 DBA 都是黑盒。我们做 per-query 统计视图（各阶段耗时/预取命中率/rerank 放大比/截断事件）+ 预取 IO 自定义 wait event，把"recall–延迟–IO"三角变成可观测对象（§4.10）。

这一组差异点的共同逻辑：**深度 PG 内核集成（并行框架/统计视图/wait event/代价模型）是 pgrx 封装层结构性难碰的区域，纯 C 路线在这里能打出真实差异。**

### 2.3 参考材料与 clean-room 原则

| 来源 | 许可证/性质 | 使用方式 |
|---|---|---|
| RaBitQ 论文（SIGMOD'24, Gao & Long） | 公开学术成果 | **只依据论文自行推导实现**；不复制作者参考实现的 C++ 代码 |
| pgvector | PostgreSQL License（与本项目兼容） | 参考 IVFFlat 的 AM 回调写法、页面管理、VACUUM 协议；借鉴处在代码注释注明出处 |
| Faiss | MIT | 参考 k-means 采样惯例、IndexRaBitQ 公开文档中的工程参数 |
| VectorChord | AGPLv3 / ELv2 | **不读源码**。仅使用其公开 benchmark 数字做竞品对照。规避 AGPL 传染性与演绎作品争议 |
| Milvus / Elasticsearch 等 | — | 生态背景：Milvus 2.6（Knowhere）与 Faiss 均已支持 RaBitQ；Elastic 用的是自研 BBQ 而非 RaBitQ 本尊；pgvector 上游未集成 RaBitQ（阿里云 RDS 的 pgvector 分支自带，属厂商补丁）。只作背景，不涉及代码 |

---

## 3. 总体架构

### 查询管线

```
SELECT ... ORDER BY embedding <=> $1 LIMIT k
        │
  ┌─────▼─────┐  amcostestimate：让 planner 在 ANN / 全表间做出数量级正确的选择
  │  AM 入口   │
  └─────┬─────┘
  ① 质心扫描    nlist 个全精度质心（热数据，常驻 buffer pool），SIMD 选 probes 个最近 list
  ② 批量预取    按 list 目录取连续页区间，read stream / posix_fadvise 批量发起
  ③ 压缩码打分  量化器估计距离（RaBitQ 1-bit 码 / SQ8），维护 size-γk 候选堆
  ④ 精确重排    候选 ctid 排序 → 批量回表取全精度向量 → 精确距离 → 按序返回 top-k
```

### 磁盘布局（全部走 buffer pool，全部 WAL 化）

| 页面类型 | 内容 | 访问模式 |
|---|---|---|
| metapage（page 0） | dim / 补齐后 dim / nlist / 度量 / 量化器类型 / 旋转种子 / 目录与质心起始块号 / ntuples | 每查询读一次 |
| list directory | 每 list：{首块号, 尾块号, 条目数}（16B/条） | 每查询读 probes 条 |
| centroid pages | nlist × dim 全精度质心，数组紧密排列 | 每查询全扫（热） |
| posting pages | 定长 entry 紧密排列，同 list 页面物理连续（构建期保证） | 批量预取、近似顺序 |
| ~~vector pages~~ | **v1 不存在**：全精度向量只存 heap，rerank 回表取（见 §4.1-D1） | — |

---

## 4. 详细设计

### 4.1 页面格式与存储布局

**posting entry（定长，同索引内 dim 恒定）：**

```c
typedef struct DannPostingEntry {
    ItemPointerData ctid;   /* 6B：回表取全精度向量用 */
    float4 dist_c;          /* ||o - c||，估计器校正因子 */
    float4 g;               /* <x̄, y>，RaBitQ 估计器校正因子（SQ8 时复用语义） */
    /* pad 至 16B 对齐 */
    uint8  code[FLEX];      /* ceil(dim/8)B（RaBitQ）或 dim B（SQ8） */
} DannPostingEntry;
```

entry 总长 = 16 + 码长：128d RaBitQ → 32B（约 256 条/8KB 页）；1024d RaBitQ → 144B；1024d SQ8 → 1040B，仍在索引元组惯例上限（约 2704B）之内。

**决策记录 D1：全精度向量存哪里（v1 的关键决策）**

问题：1024d float32 = 4096B、960d = 3840B，超过 8KB 页下单个索引元组的惯例上限（约 1/3 页）。索引内自存向量有三条路，都有明显代价：

| 方案 | 代价 |
|---|---|
| (a) 自定义跨页页格式 | 自己管页面、WAL redo、空间回收，工作量显著上升 |
| (b) 大 BLCKSZ（16KB+）编译 PG | 4096B 满足 1/3 规则，但要求定制编译，限制普适性 |
| (c) 索引内只存 halfvec/PQ 近似 | 逼近上限且 rerank 精度受损，违背 rerank 初衷 |

**决策：v1 不存，rerank 按 ctid 回 heap 取。** 理由：正确性优先；heap 取数有 pgvector iterative scan 的先例；配合"ctid 排序后批量回表"，随机读被压成近似顺序读；若用户按 list 顺序 CLUSTER 过表，效果更佳。代价：死元组/HOT 更新需按协议处理（见 §4.5）；索引不自包含。v2 是否做方案 (a) 见 §9 开放问题。

**决策记录 D2：两遍式构建与物理聚簇**

第一遍扫表只计数（每 list 条目数），第二遍前按计数连续分配各 list 的 posting 页段。效果：单 list = 一段连续块号区间，预取整段发起；构建内存只需 O(nlist) 计数器 + 每 list 一个写缓冲页。增量插入的追加页不再连续（布局退化），由 §4.5 的离散度指标监控。

**维度边界**：FWHT 要求 2 幂维度，非 2 幂（如 GIST1M 960d）构建期补零到下一 2 幂（960→1024），metapage 同时记录原始 dim 与补齐 dim。维度上限随 pgvector 对齐为 16000（码长 2000B + 头 16B 仍在上限内）。

### 4.2 RaBitQ 量化器

**数学概要（以论文为准，此处为实现锚点）：**

- 旋转：P = 随机置换 ∘ FWHT ∘ 随机符号翻转，O(d log d)，种子存 metapage。质心 c 为 list 质心。
- 编码：y = P·(o−c)/||o−c‖，x̄_b[i] = (y_i ≥ 0)，每维 1 bit。存储 g = ⟨x̄, y⟩（x̄ = (2·x̄_b − 1)/√d）与 dist_c = ||o−c‖。
- 估计（非对称，查询全精度）：q' = P·(q−c)/||q−c‖，coŝ = ⟨x̄, q'⟩ / g；
  L2：d̂²(o,q) = dist_c² + ||q−c||² − 2·dist_c·||q−c||·coŝ。IP/cosine 直接用 coŝ（cosine 经构建期与查询期归一化后等价）。
- 误差界：估计标准差 ≈ κ·dist_c·||q−c‖/√d（论文给出常数），d 越大越准——高维是本方法的甜区。

**实现要点（⟨x̄, q'⟩ 的快速形式）：**

⟨x̄, q'⟩ = (1/√d)·( 2·Σ_{bit=1} q'_i − Σ_all q'_i )。Σ_all 每查询只算一次；每条码只需一次"按码位掩码求和"。

- AVX-512：码的 64bit chunk 直接作 `__mmask64`，`_mm512_mask_add_ps` 掩码累加——天然契合。
- AVX2：掩码扩展 + blendv。
- NEON：码位扩展成掩码向量 + `vand` 累加。
- 平台矩阵写对：**x86（Intel/AMD/海光）→ AVX2 兜底、有 AVX-512 用 AVX-512；ARM（鲲鹏/飞腾）→ NEON**。运行时 cpuid 分派，编译期 `pg_attribute_target` 多版本。
- 可选对称路径（1-bit 查询 × 1-bit 码，popcount：AVX-512 VPOPCNTDQ / NEON CNT）留作预筛实验，不进 v1 主路径。

**工程纪律：先写 numpy golden model 再写 C。** golden model 与论文公式逐式对拍（含误差界：实测误差分布 vs 理论 1/√d 曲线），C 实现与 golden model 随机向量对拍（容差 1e-5）。

**量化器自动选择（`quantizer = 'auto'`，默认）**：规则 = dim < 128 → SQ8，否则 RaBitQ。依据：RaBitQ 估计标准差 ≈ κ/√d，低维时 1-bit 码噪声超过 SQ8 的 8-bit 分辨率收益。阈值初值 128，按 golden model 阶段的误差分布实测修正。两量化器同接口，切换零成本。

**rerank 因子 γ**：v1 固定 GUC（`dann.rerank_factor`，初值 20，按 recall 曲线调）；v2 候选：基于误差界的自适应阈值——取估计距离第 k 名 d̂_k，纳入所有 d̂ ≤ d̂_k + z·σ 的候选。这是 RaBitQ 相对传统二值量化的理论优势（有界可做原则性决策），值得做成文章素材。

### 4.3 构建流程（ambuild）

1. **采样**：表级采样，样本量 = clamp(64×nlist, 256×nlist, 内存预算上限)，构建内存与表大小无关。
2. **k-means**：采样集上 Elkan 算法，10–20 轮，SIMD 距离。
3. **第一遍扫表**：每向量找最近质心，只累计 per-list 计数（O(nlist) 内存）。
4. **连续分配**：按计数为各 list 分配连续 posting 页段，写 list directory。
5. **第二遍扫表**：编码（旋转 + 量化 + 因子），追加到对应 list 的页段；每 list 一个写缓冲页，满则刷。
6. **落盘**：centroid pages、directory、metapage。全程 WAL。

v1 单线程构建；PG 并行索引构建框架（parallel build）留作后续。空表/行数 < nlist：nlist 自动夹子（clamp）并在 WARNING 中说明。

### 4.4 查询流程（amrescan / amgettuple）

1. 读 metapage → 旋转查询向量（cosine 先归一化）→ SIMD 扫全部质心，取距离最小的 probes 个 list。质心扫描成本 nlist×d 次乘加（4096×1024 ≈ 4M，亚毫秒）。
2. 查 directory 得各 list 连续页区间，合并相邻区间后批量预取：PG18 优先 read stream，降级 `posix_fadvise(WILLNEED)`，再降级同步 ReadBuffer。
3. 流式扫 entry 打分，维护 size-γk 大顶堆。
4. 候选 ctid **排序** → 批量回表取全精度向量 → 精确距离重排 → 按序返回。
5. "relaxed order"语义写进 README：开 rerank 返回精确序；关 rerank（`dann.enable_rerank=off`）返回估计序，供极致延迟场景。

**延迟预算截断**：`dann.max_latency_ms`（默认 0 = 不限）。posting 扫描循环每处理完一个 list 检查一次 deadline，耗尽即停——list 按质心距离升序处理，截断只丢尾部最远的 list，recall 平滑降级而非查询失败。截断事件计入 `dann_scan_stats()`（§4.10）。

参数：probes / γ / 预取开关 / 延迟预算均暴露为 GUC（probes、γ 同时可作 reloption）。

### 4.5 插入 / 删除 / VACUUM

- **插入**：归一化（cosine）→ 最近质心 → 编码 → 追加到 list 尾页（无分裂、无指针修复，比 HNSW 插入简单一个量级）。尾页满则分配新页挂到链尾、更新 directory。
- **删除**：PG 不回调 AM；死 entry 由 VACUUM 清理。
- **ambulkdelete**：全索引扫描 + 回调逐 entry 判定删除；**amvacuumcleanup**：汇总统计。v1 不做页内紧凑化与空页回收（与 IVFFlat 行为一致，靠 REINDEX 重建）——写进 README 已知限制。
- **布局退化监控**：提供 `dann_list_stats()`（每 list 条目数/页数/块号离散度），文档给出"离散度超阈值建议 REINDEX"的运维指引。

### 4.6 WAL 与崩溃恢复

- **v1 用 generic_xlog**：页变更走 GenericXLogRegister/Finish，新页 REGBUF_WILL_INIT，full-page write 交给内核机制。正确性优先，逻辑简单。
- **v2 可选优化**：若实测 WAL 体积不可接受，`RegisterCustomRmgr`（PG15+）写定制 redo（插入只记录 entry 而非整页 delta）。先测量再决定，不预设。
- **崩溃测试矩阵**：kill -9 注入点 = {构建第二遍中途, 批量插入中途, VACUUM 中途}；恢复后校验：行数一致 + 固定查询集 recall 与崩溃前一致 + 可继续插入。

### 4.7 并发模型

- 查询：页级 share buffer lock，无高层锁，只读路径天然多并发安全。
- 插入：按 list id 哈希到 LWLock 数组（128 把）保护"尾页追加 + directory 更新"；页内用 exclusive buffer lock。
- 锁序：metapage → directory → list LWLock → buffer lock，禁止反向。
- VACUUM 与插入并发：沿用 IVFFlat 的协议（ambulkdelete 与插入的 buffer 级互斥）。
- 验收：32 并发插入 + 查询混合压测无错、无死锁；VACUUM 后 recall 不下降。

### 4.8 代价估算（amcostestimate）

模型：cost ≈ 质心扫描（cpu）+ probes × list 平均页数（seq_page_cost，连续布局按顺序读计）+ γk × 回表（random_page_cost + cpu）。

目标务实：**数量级正确 + 关键参数走 GUC 可调**。PG planner 不理解 ANN 的 recall 语义，"该走索引没走"类问题（pgvector #846 那类）没有完美解，不承诺解决，只承诺不添乱。

### 4.9 SQL 接口与代码结构

```sql
CREATE INDEX ON docs USING dann (embedding dann_l2_ops)
  WITH (nlist = 4096);   -- quantizer 缺省 'auto'（按维度自动选 rabitq/sq8），可显式指定
-- opclass: dann_l2_ops / dann_ip_ops / dann_cosine_ops
-- GUC: dann.probes / dann.rerank_factor / dann.enable_rerank / dann.io_method / dann.max_latency_ms
SET dann.probes = 64;
SELECT * FROM docs ORDER BY embedding <=> $1 LIMIT 10;
```

向量类型：**v1 依赖 pgvector 的 `vector` 类型**（生态兼容、用户迁移与 benchmark 对比零摩擦；VectorChord 同样如此）。是否自研类型见 §9。

```
pg_dann/
  src/  dann.c(AM handler) build.c kmeans.c scan.c insert.c vacuum.c
        page.c xlog.c cost.c prefetch.c
        quant/ quant.h(接口) sq8.c rabitq.c fwht.c
        simd/  dispatch.c est_avx2.c est_avx512.c est_neon.c
  sql/  test/(installcheck) golden/(numpy 模型+对拍) bench/(sift1m.sh bigann.sh recall.py)
```

### 4.10 可观测性

- **`dann_scan_stats()`**：共享内存哈希（pg_stat_statements 的极简版），每查询记录：质心扫描耗时、预取页数、posting 页缓存命中/未命中、候选数（γk）、回表页数、rerank 放大比（γk/k）、延迟截断事件数。
- **自定义 wait event**：`DannPrefetchIO`（PG15+ 支持扩展注册 wait event），预取等待不再是笼统的 IO.DataFileRead。
- **`dann_list_stats()`**：每 list 条目数/页数/块号离散度（§4.5 布局退化监控）。
- **已知限制**：index AM 无法向 EXPLAIN ANALYZE 注入自定义行（PG 未留此钩子），ANN 专项统计走统计视图路线，不硬做 EXPLAIN 增强。

---

## 5. 里程碑（12 周 × 每周 10–12h，按风险排序）

| 周 | 里程碑 | 验收标准 |
|---|---|---|
| W1 | PGXS 骨架 + AM 全回调 stub：能 CREATE INDEX 空索引、能插入；默认代价模型不会选中它，强制走索引时返回空且不崩溃 | 本地 PG16 `make installcheck` 绿；CI 矩阵 PG16/17/18 后续补 |
| W2–3 | quantizer 接口 + SQ8 实现 + numpy golden model 对拍 + SIMD L2 距离（AVX2 起步）+ micro-bench | SIMD ≥ 编译器自动向量化基线 4×；golden model 一致 |
| W4–5 | IVF 构建（采样 k-means + 两遍布局）+ 页面格式 + 查询 v0（同步读、SQ8 打分、无 rerank） | SIFT1M 首组 recall/QPS；recall@10 与 IVFFlat 同水位（同 probes 下） |
| W6 | 批量预取 + heap rerank（ctid 排序回表）+ 延迟预算截断（`dann.max_latency_ms`） | recall@10 ≥ 0.98（SQ8 版）；报告 p99；预算截断生效且 recall 平滑降级 |
| W7–8 | RaBitQ：numpy golden model（含误差界实测）→ C 实现 → 与 SQ8 同机对照 + `quantizer='auto'` 维度自适应 | recall@10 ≥ 0.98 且同 recall 下 QPS/内存优于 SQ8 版；误差分布图产出；低维（<128d）自动落 SQ8 |
| W9 | WAL 化（generic_xlog）+ 崩溃恢复测试矩阵 | 三个注入点恢复后索引一致可用 |
| W10 | 并发插入（LWLock）+ ambulkdelete/amvacuumcleanup | 32 并发混合压测无错；VACUUM 后 recall 不掉 |
| W11 | 加固收尾：amcostestimate、reloption/GUC 整理、可观测性（`dann_list_stats()`/`dann_scan_stats()`/wait event）、边界情况（空表/单行/重复向量/非 2 幂维/NaN 校验）、README | 边界用例全进 installcheck |
| W12 | 完整 benchmark + 技术文章（实现向：设计决策、坑、数据） | §6 全部指标产出；README 含复现脚本 |

**降级路径**：W9/W10 可整体后置 → 先发布"只读 bulkload + 查询"版本，WAL/并发作为已知 TODO 写进 README；W7–8 卡住 → SQ8 版完整发布，RaBitQ 留 golden model + 设计文档。管线（布局/预取/rerank/WAL）全部是量化器无关的，RaBitQ 失败不拖垮项目。

---

## 6. Benchmark 设计

**数据集**：SIFT1M（128d，标准对标）｜ **bigann-10M 子集（主战场，数据超内存，磁盘路径真实受力）** ｜ bge-m3 真实 embedding 10 万条 1024d（真实 RAG 负载；生成成本高，优先用 HF 上预计算 embedding 数据集或 GPU 离线批编码，规模封顶 10 万）｜ GIST1M（960d，可选，验证补零路径）。

**基线**：pgvector 0.8.x HNSW 与 IVFFlat（同机同参数调优）；有余力加 VectorChord 同机对照（只比公开接口，不碰其代码）。

**指标定义（方法论重点）：**

- **内存——"最小 shared_buffers 法"**：固定 recall@10 ≥ 0.98 与 QPS 下限，二分搜索各方案所需最小 shared_buffers，报比值。backend RSS 受 buffer pool 污染，不作为指标。同时报索引磁盘体积（SIFT1M 估算：pg_dann ≈ 34MB vs HNSW ≈ 700MB–1GB）。
- **recall–QPS 帕累托曲线**：扫 probes × γ 网格；单线程与 32 并发两组；p50/p99 同报。
- **构建**：构建时间 + 构建内存峰值（论证 O(采样) 卖点）；插入吞吐。
- **冷热分组**：每组实验前 restart + drop_caches 为冷组，pg_prewarm 后为暖组，分开报告。
- **崩溃恢复**：kill -9 → 恢复 → 行数/recall 一致性。文案注意：现行 pgvector HNSW 已有 WAL，对照的是其 0.5 之前无 WAL 的历史，不得暗示现行版本会丢索引。

**测试机**：固定 NVMe SSD，记录 CPU（核数/是否支持 AVX-512）/内存/内核版本/io_method。无 ARM 硬件时 NEON 路径只保证编译与正确性，性能数据明确标注"未实测"。

---

## 7. 测试与工程质量

- installcheck 回归（小合成数据集，覆盖全部边界情况）。
- 暴力对拍：Python + numpy 全量精确距离算 ground truth，校验 recall（bench/recall.py）。
- golden model 对拍：SQ8 与 RaBitQ 各一套，C 实现随机向量回归（容差 1e-5）。
- ASAN/UBSAN 构建进 CI；valgrind 抽查。
- CI 矩阵：PG16/17/18 × x86_64；ARM 仅在有硬件后补。

---

## 8. 风险与降级

| 风险 | 影响 | 对策 |
|---|---|---|
| RaBitQ 数学卡住 | 失去最大亮点，但管线完整 | SQ8 版兜底发布；RaBitQ 留 golden model + 设计文档 |
| WAL/并发超期（W9–10 是重活） | 只读索引 | 先发布 bulkload + 查询版，WAL/并发作为 TODO 写进 README |
| heap rerank 的 p99 超 HNSW 3 倍以上 | 磁盘路径卖点受损 | 调 probes/γ/预取深度；检查 CLUSTER；确认 NVMe 随机读基线；最终叙事锚定内存收益 |
| PG18 read stream 对扩展的 API 适配超预期 | 预取退回 fadvise | 三级降级已设计（read stream → fadvise → 同步读） |
| 无人力维护全量指标 | benchmark 缩水 | 砍 GIST1M；bge-m3 用预计算 embedding |

---

## 9. 开放问题（实现中决策）

1. **向量类型**：v1 依赖 pgvector `vector` 类型（推荐：生态兼容、对标零摩擦）。自研类型的成本（IO/cast/二进制兼容）在 W1 评估后最终定。
2. **v2 索引内向量存储**：是否做自定义跨页页格式（§4.1-D1 方案 a）？取决于 heap rerank 的实测 p99 与 CLUSTER 依赖度。
3. **γ 自适应**：误差界阈值法（d̂_k + z·σ）是否替代固定 GUC？需要 golden model 阶段的误差分布数据支撑。
4. **custom WAL rmgr**：generic_xlog 实测 WAL 体积后再定，不预设。
5. **并行构建**：PG parallel index build 框架接入的性价比，W12 后评估。
