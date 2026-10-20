# pg_dann 架构文档(竣工稿,以代码为准)

> 本文描述 **实际实现** 的架构,与 DESIGN.md(计划稿)的关系:DESIGN.md 是
> 开工时的设计与风险排序,本文以 src/ 代码为唯一事实来源;两处不一致处以
> 本文为准,偏离汇总见 §13。数据结论见 ARTICLE.md,逐里程碑验收见
> bench/RESULTS.md。

## 1. 定位

pg_dann 是 PostgreSQL 16+ 的索引访问方法(index AM,PGXS 扩展,无 fork):
IVF 倒排 + 量化(SQ8 / RaBitQ)的磁盘优先向量检索。核心赌注:**索引只存
量化码 + ctid,全精度向量只存 heap,最终序靠回表重排(rerank)保证**。
依赖 pgvector 的 `vector` 类型与其三个距离操作符(不链接 pgvector 代码,
运行时解析其内存布局,dann.h:DannVector)。

- AM 能力面(dann.c:dann_handler):`amcanorderbyop=true`(KNN 语义),
  无 bitmap/并行/唯一约束/表达式列 rerank;
  `amusemaintenanceworkmem=true`;vacuum 支持并行 bulkdel。
- 三类度量:L2(默认 opclass `dann_l2_ops`)、内积(`dann_ip_ops`,
  支持函数返回负内积使升序=最近邻)、余弦(`dann_cosine_ops`)。
- 度量在构建期由 opclass 支持函数 1 的 proname 解析(page.c:
  dann_metric_of),写入 metapage;cosine 在构建/插入/查询三端均先归一化。

## 2. 存储布局

所有页带 12B special area `DannPageOpaqueData{page_type,next,count}`;
定长内容从 `PageGetContents()` 起紧密排列,**内容末端必须并进 pd_lower**
(page.c:dann_page_set_lower)——generic_xlog 在线应用与 redo 都会把
`[pd_lower, pd_upper)` 清零,内容在 hole 区不抬 pd_lower 会被抹掉(W9 踩坑,
ARTICLE §3)。旧格式页(pd_lower=24)在插入/vacuum 写入路径统一抬升,无损兼容。

| 区域 | 块号 | 内容 |
|---|---|---|
| metapage | 0 | magic/version、dim、padded_dim、nlist、量化器(构建期已解析)、metric、rotation_seed、三个区域起始块、entry_size、ntuples |
| directory | 1 .. dir_npages | 每 list 16B:`{first, last, count, reserved}`(DannListEntry),每页 DANN_DIR_PER_PAGE 条 |
| centroid | dir 之后 | nlist×dim 全精度 float4,紧密排列、不跨页;热数据常驻 buffer pool |
| posting | 其后 | 每 list 一段**连续块号**(构建期按计数精确分配,物理聚簇 D2);页间以 opaque->next 链式连接 |

posting entry 定长(同索引恒定,4B 对齐):
`ctid(6B) + pad(2B) + 3×float4 量化因子 + code`,头 20B。

| 量化器 | 因子 | 码长 | 128d entry | 1024d entry |
|---|---|---|---|---|
| SQ8 | vmin, delta, ‖deq‖² | dim B | 148B | 1044B |
| RaBitQ | dist_c, g, dist_c² | ceil(padded_dim/8) | 36B | 148B |

`padded_dim` = dim 补零到 2 的幂(FWHT 要求);RaBitQ 码在补齐空间中。
`dann_padded_dim / dann_code_len / dann_entry_size / dann_entries_per_page`
(page.c)是全部尺寸计算的唯一入口。

## 3. 量化器(quant/,纯 C 不依赖 PG 头文件)

- **SQ8**:per-vector 仿射 `deq[i]=vmin+delta·code[i]`,舍入约定
  `floor(x+0.5)`(与 golden model 一致);估计不反量化查询,展开平方后
  热路径只剩 `<q,deq>`(‖deq‖² 编码期落盘)。
- **RaBitQ**:相对本 list 质心编码;旋转 P = 随机置换 ∘ FWHT ∘ 随机符号
  翻转,参数由 SplitMix32(seed) 派生,**seed 存 metapage**、由
  (dim,nlist) 确定性生成——构建可复现,插入/查询按种子重建同一旋转。
  L2 估计式 `d̂² = dist_c² + ‖q−c‖² − 2·dist_c·‖q−c‖·coŝ`(g≤1e-6 保护);
  cosine/IP 直接用 coŝ(作距离取负)。dist_c==0 时码全零、g=0。
- **auto 阈值**:dim < 128 → SQ8,否则 RaBitQ(构建期解析后写入
  metapage,查询/插入端不再分支)。**bge-m3 1024d 实测修正**:高维语义
  分布上 RaBitQ 估计序噪声需 γ≥200 或回 SQ8(见 §14 与 ARTICLE §5.7)。
- **golden model 纪律**:先有 numpy golden model(golden/),C 端对拍
  容差 1e-5,RaBitQ 码要求逐字节一致;`make quant-test` 27328 项全绿。
- **SIMD 运行时分派**(simd/dispatch.c,cpuid):scalar / AVX2 / AVX-512
  三档。L2、dot、SQ8 估计、RaBitQ ipqr(码 16bit 组作 `__mmask16` +
  `_mm512_mask_add_ps`)。128d 条目打分 AVX-512 7.5ns(标量 481ns,64×)。

## 4. 构建(build.c:三遍扫表,O(采样) 内存)

```
Pass A  table_index_build_scan + 蓄水池采样(算法 R,LCG 确定性)
        → sample_cap = min(64×nlist, 65536, 64MB/dim/4B)
k-means 采样集上 Lloyd 15 轮(kmeans.c)
Pass B  再次全扫:逐元组分配最近质心,per-list 计数;
        assignment(每行 list id,uint16)留存
布局    按计数为每 list 连续分配 posting 页段(物理聚簇)
Pass C  第三次全扫:重放 assignment,编码,经每 list 一个 8KB
        私有页缓冲写入;写完 directory;counts 倒计数校验两遍一致
```

- 内存:O(采样集 ≤64MB + nlist 页缓冲 + N×2B assignment),全精度向量
  从不全量驻留——与 ivfflat 对比的实测门槛:SIFT1M 64MB vs 222MB,
  bigann-10M/nlist=4096 512MB vs 3374MB。
- nlist 夹子:空表只写 nlist=0 的 metapage(后续插入报错提示 REINDEX);
  nlist > 样本数时 WARNING 并夹到样本数。
- 页写入全部 generic_xlog 全页镜像(GENERIC_XLOG_FULL_IMAGE);
  `!RelationNeedsWAL` 且永久表时构建结束 `smgrimmedsync`(nbtsort 约定)。
- `dann_extend_relation` 预扩展:P_NEW 返回的 buffer 只 pin 不 lock,
  只能 ReleaseBuffer。

## 5. 查询管线(scan.c)

amgettuple 是拉取式接口(不知 LIMIT k),首次 gettuple 完成全管线:

```
① 质心扫描   全扫 centroid 页,SIMD 距离,nlist×dim 乘加,选 probes 个
             最近 list 并按质心距离升序(供延迟截断)
② 批量预取   directory 给出各 list 连续页区间 [first, first+npages),
             排序合并后 PrefetchBuffer(fadvise WILLNEED);io_method=sync
             关闭;PG18 read stream 留 #if TODO
③ 估计打分   逐 list 沿链扫 posting 页,量化码估计距离,候选收进动态数组
④ 排序+rerank 估计序全排;rerank 以 γ×10 为块懒加载:块内 ctid 排序 →
             table_index_fetch 批量回表(MVCC 快照,死元组跳过)→ 精确
             距离块内重排 → 按精确序发射;块耗尽补下一块
```

- **延迟预算截断**:`dann.max_latency_ms>0` 时每处理完一个 list 检查
  deadline,耗尽即停(只丢尾部最远 list,recall 平滑降级);截断事件计入
  `dann_scan_stats()`(scans/truncated/lists_skipped,会话级)。
- **退化路径**:无 ORDER BY(如 count(*))或 ORDER BY 参数为 NULL
  (SK_ISNULL)时退化为全索引扫描,行集合正确、序无定义(W10 修复的
  段错误);`xs_recheckorderby` 恒 false,不设置 `xs_orderbyvals`
  (reorder queue 恒空,与 pgvector 同款约定)。
- NaN/Inf 在构建/插入/查询三端拒绝(dann_check_finite);cosine 零向量
  归一化时报错(与 pgvector 行为一致)。
- RaBitQ 查询侧:每 list 一次 `P(q−c)/‖q−c‖`(O(d·log d)),条目热循环
  只剩 ipqr 一次。

## 6. 插入(insert.c)与并发协议

单条 aminsert:读 metapage → 归一化(cosine)→ SIMD 最近质心 → 编码 →
追加到 list 尾页;尾页满则 P_NEW 扩新页挂链尾(段不再连续,布局退化由
`dann_list_stats()` 的 span>pages 观测);构建期空 list 末尾开新页。

**并发协议(v1,W10)**:插入的"读 directory → 尾页追加 → 写 directory"
全程持 metapage `BUFFER_LOCK_EXCLUSIVE` 串行化(与 IVFFlat 同协议);
持锁下重读 directory(编码期间尾页可能已被并发推进)。锁序
metapage → directory → posting,无反向。DESIGN §4.7 的 128 把 list
LWLock 需 shared_preload_libraries(named tranche 只能启动时注册),留作
preload 版升级——页格式/WAL 布局完全兼容,可无感升级。

写入纳入**单个 GenericXLog 原子提交**(posting 页 + directory 条目);
`GenericXLogFinish` 不解锁不释放 buffer,调用方逐个 UnlockReleaseBuffer。

## 7. VACUUM(vacuum.c)

- **bulkdelete**:逐 list 逐页全索引扫描,回调判死,页内
  swap-with-last 紧凑删除(条目无序,交换合法;换入者下轮再判);有删除
  才 GenericXLogFinish,否则 Abort。directory 的 count 在 metapage 排他
  锁下按 dir 页分批更新。空页不回收(与 IVFFlat 一致,靠 REINDEX)。
- **vacuumcleanup**:bulkdelete 未跑时按 directory 现值汇总精确条目数;
  analyze_only 直接返回。
- W10 验收:10 万 entries VACUUM 精确删到 75,000(=存活行数),删后
  索引指纹与精确序逐字节一致。

## 8. WAL 与崩溃恢复

所有页写入走 generic_xlog(构建全页镜像;插入/vacuum 增量登记 +
pd_lower 抬升)。WAL 体积 ≈ 索引体积(正确性优先;定制 rmgr 是 v2 候选)。
unlogged 表由 init fork(dann_buildempty 写 nlist=0 最小 metapage)重建。
崩溃恢复矩阵(bench/crash_test.sh,kill -9 整机模拟):构建中途/批量插入
中途/VACUUM 中途三注入点全部回滚干净、指纹精确一致(RESULTS W9)。

## 9. 代价模型(cost.c)

无 KNN ORDER BY 时代价 1e12 永不入选(不添乱原则);空索引给低价。
常规模型:startup = 常数 + 质心扫描 CPU(nlist×dim,SIMD 折算);IO =
(目录+质心页+probes×list 平均页数)× seq_page_cost(物理连续按顺序扫描
计价,表空间成本经 get_tablespace_page_costs);CPU = probes×平均条目×dim
×折算;rerank = γ×100 回表 ×(random_page_cost + tuple 成本)。
元数据经 NoLock 打开索引读 metapage(pgvector 同款做法),拿到构建期
实际 nlist/entry_size/ntuples(夹子后仍准)。selectivity 恒 1(KNN 截断
由 LIMIT 表达)。

## 10. 参数

reloptions(CREATE INDEX ... WITH(...)):

| 项 | 默认 | 说明 |
|---|---|---|
| `nlist` | 1000 | 1..65535(uint16 assignment 上限);空表/小样本自动夹子+WARNING |
| `quantizer` | `auto` | auto/rabitq/sq8;auto: dim<128→SQ8 否则 RaBitQ(§3 修正注记) |
| `probes` / `rerank_factor` | 0 | >0 时覆盖同名 GUC(按索引生效) |

GUC(查询期,均 USERSET):`dann.probes`(10)、`dann.rerank_factor`(γ=20,
候选=γk)、`dann.enable_rerank`(on)、`dann.max_latency_ms`(0=不限)、
`dann.io_method`(auto/read_stream/fadvise/sync;read_stream 需 PG17+,
PG16 降级 fadvise)。

## 11. SQL 接口

- opclass:`dann_l2_ops`(默认,`<->`)/ `dann_ip_ops`(`<#>`)/
  `dann_cosine_ops`(`<=>`),支持函数 1 = pgvector 距离函数。
- `dann_scan_stats()` → (scans, truncated, lists_skipped),会话级;
  `dann_scan_stats_reset()`。
- `dann_list_stats(regclass)` → (listno, entries, pages, span),布局退化
  监控(span > pages 即有离散追加页)。

## 12. 测试与对拍资产

- installcheck:`make installcheck`(test/: basic/ivf/rerank/rabitq/vacuum/
  edge,边界用例全进回归:空表/单行/重复向量/非 2 幂维/NaN/NULL
  ORDER BY 等;`--max-connections=1` 串行)。
- `make quant-test`:C 量化器 vs numpy golden model 对拍(27328 项);
  `bench/quant_harness` micro-bench。
- golden/:rabitq_model.py(旋转/编码/估计逐位规范)、rabitq_error.py
  (误差曲线,rabitq_error_curve.csv:σ·√d 恒定 1.30–1.45)。

## 13. 对 DESIGN.md 的偏离汇总(以代码为准)

| 项 | DESIGN 计划 | 实际落地 |
|---|---|---|
| 并发协议(§4.7) | 128 把 list LWLock | metapage BUFFER_LOCK_EXCLUSIVE 串行化(named tranche 需 preload);页格式兼容,可无感升级 |
| 构建遍数 | "两遍式"表述 | 三遍堆扫描(采样/计数/写入);O(采样) 内存卖点不变 |
| 页内容布局 | 未涉 generic_xlog 细节 | 内容必须并入 pd_lower(W9 踩坑);旧页插入路径兼容 |
| NULL ORDER BY | 未考虑 | 退化为全索引扫描(W10 修复段错误) |
| RaBitQ 高维甜区 | dim≥128 → RaBitQ | bge-m3 实测:语义分布需 γ≥200 或回 SQ8(ARTICLE §5.7) |
| io_method | read stream | PG16 统一 PrefetchBuffer(fadvise);PG18 read stream 留 TODO |

## 14. 已知限制(v1)与 v2 候选

限制:空页不回收靠 REINDEX;插入经 metapage 锁串行化;无并行构建/
并行扫描(amcanparallel=false);无表达式列 rerank;`quantizer=auto`
不感知数据分布;PG18 read stream 与 NEON 仅编译期 TODO;CI 矩阵
PG17/18、ASAN 待补。

v2 候选(按价值排序):γ 自适应(按估计序分歧度动态调,bge-m3 教训);
list LWLock preload 版;定制 rmgr 降 WAL 体积;PG parallel build 框架;
read stream(PG18);NEON 实测(无 ARM 硬件)。
