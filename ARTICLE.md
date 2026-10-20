# pg_dann:一个磁盘优先、崩溃可恢复的 PostgreSQL 向量索引(实现笔记)

> 一句话:IVF + 量化(SQ8/RaBitQ)的索引访问方法,全走 buffer pool、
> 全 WAL 化、崩溃可恢复;SIFT1M 上索引 39MB(HNSW 的 1/20)、构建 17s
> (HNSW 的 1/20),recall@10 0.99 档 32 并发 QPS 是 HNSW 的 2.2 倍。

代码与复现:见 README.md。本文是 W12 的实现向总结:设计决策、实测数据、
以及一路踩过的坑。

## 0. 为什么要再做一次

pgvector IVFFlat 的 posting entry 带全精度向量(128d → 520B/条),磁盘页
胖、内存驻留压力大;HNSW 快但索引体积/构建时间在高基数下失控,且图结构
的随机读对磁盘路径不友好。pg_dann 的赌注:**索引只存量化码 + ctid,
全精度向量只存 heap,最终序靠回表重排(rerank)保证**。

## 1. 架构与关键决策

### 1.1 查询管线(全部按 list 连续页段组织)

```
① 质心扫描   nlist×d 次乘加(SIMD),选 probes 个最近 list
② 批量预取   directory 给出连续页区间,合并后 PrefetchBuffer(fadvise)
③ 量化打分   SQ8 / RaBitQ 估计距离,按估计序收集候选
④ heap rerank 候选 ctid 排序 → 批量回表 → 精确距离 → 按序发射
```

amgettuple 是拉取式接口,不知道 LIMIT k。rerank 以 **γ×10 为块懒加载**
(γ=`dann.rerank_factor`):返回 k 行大约回表 γk 个候选;执行器停拉即停,
不浪费回表。

### 1.2 两遍式构建与物理聚簇(D2)

第一遍只计数(per-list 条目数,O(nlist) 内存),然后按计数**连续分配**
各 list 的 posting 页段。效果:单 list = 一段连续块号,预取整段发起;
构建内存 O(采样) 封顶(SIFT1M 实测:ivfflat 在 maintenance_work_mem=64MB
直接报错,pg_dann 采样 64MB 封顶照建)。

### 1.3 量化器双轨与 auto

| | SQ8 | RaBitQ |
|---|---|---|
| 码长 | dim B | dim/8 B |
| 128d entry | 148B | 36B |
| 估计精度 | 分辨率有界(0.05% 级) | σ_cos ≈ 1.4/√d(实测 κ 恒定) |
| 适用 | 低维 | 高维甜区 |

`quantizer='auto'`(默认):dim < 128 → SQ8,否则 RaBitQ。阈值经 golden
model 误差曲线复核(`golden/rabitq_error_curve.csv`),installcheck 用
"同数据同参数索引体积逐字节相等"验证选择确定性。(后续实测修正:高维
**语义** embedding 上 RaBitQ 需更大 γ,见 §5.7/§6。)

### 1.4 延迟预算截断

`dann.max_latency_ms`:posting 扫描每处理完一个 list 检查 deadline,耗尽
即停——list 按质心距离升序处理,截断只丢尾部最远 list。SIFT1M probes=50
实测:1ms 预算砍 45% 平均延迟(9.9→5.4ms),recall 只掉 1.2pp
(0.9935→0.9815),平滑降级而非报错。截断事件计入 `dann_scan_stats()`。

## 2. RaBitQ 落地:golden model 先行

纪律是"先写 numpy golden model 再写 C",两端对拍容差 1e-5:

- 旋转 P = 随机置换 ∘ FWHT ∘ 随机符号翻转;参数由 SplitMix32(seed) 派生,
  种子存 metapage,构建可复现,C/numpy 逐位同规范。
- 实测验证论文误差界形状:σ_cos·√d ≈ 1.30–1.45 恒定(d=16..1024),
  g 均值 0.798 ≈ √(2/π) ✓。
- AVX-512 条目打分:码的 16bit 组直接作 `__mmask16` + `_mm512_mask_add_ps`,
  128d 每条 7.5ns(-O2 标量基线 481ns,64×;SQ8-AVX2 14.4ns)。

**坑 1(numpy)**:打包码位时 `code[idx>>3] |= bits` 的 fancy-indexing 对
重复下标"只保留最后一写",码字节静悄悄地错。要用 `np.bitwise_or.at`。
这个 bug 在 numpy 侧,但表现是 C 端"对不上"——对拍纪律把它逼出来了。

**坑 2(浮点逐位)**:归一化用 `x *= 1/√n`(倒数乘法)与 numpy 的
`x /= √n` 差 1ulp;符号量化对近零坐标敏感,码位随机翻。改成同式除法后
两端逐位一致。对拍"码逐字节相等"是值得的严格度。

## 3. WAL 化:generic_xlog 的三个坑

W9 把构建/插入所有页写入切换到 generic_xlog。坑都在 API 语义细节里:

1. **`GenericXLogFinish()` 不解锁不释放 buffer**。调用方必须自己
   `UnlockReleaseBuffer`。漏掉的后果:锁泄漏到事务结束,同会话后续访问
   自死锁(不可中断的 BufferContent LWLock 等待,pg_terminate 都杀不掉)。
2. **hole 清零**:generic_xlog 在线应用与 redo 都把 `[pd_lower, pd_upper)`
   清零(防脏页分歧)。本 AM 的定长内容恰好排在 hole 区(`PageGetContents`
   起,无 line pointer)——必须写完内容后把内容末端并进 `pd_lower`。
   对 W9 前构建的旧页,插入时统一抬升 pd_lower 即无损兼容。
3. **崩溃注入的工程细节**(PG 16.13 衍生发行版):backend 是独立进程组
   (pgid=pid),kill -9 postmaster 进程组杀不死 backend;按 /proc/exe
   路径逐个点名。kill -9 后 SysV 共享内存段成孤儿(nattch=0),挡重启,
   需 ipcrm 清理。

崩溃恢复矩阵(kill -9 整机模拟,`bench/crash_test.sh`):

| 注入点 | 结果 |
|---|---|
| 构建中途(1M 行,活跃 2s 断电) | 事务回滚干净,表完好,重建可查 ✓ |
| 批量插入中途(60 批,断在第 41 批) | 行数=已提交批精确一致,索引指纹==精确序,可续插 ✓ |
| VACUUM 中途 | 一致可用 ✓ |

## 4. 并发插入:metapage 串行化(v1)

设计稿 §4.7 写的是 128 把 list LWLock。落地时发现 named LWLock tranche
只能在集群启动注册(需 shared_preload_libraries)——v1 选择 pgvector
IVFFlat 同款 metapage 排他锁协议:插入的"读 directory→尾页追加→写
directory"全程持 metapage EXCLUSIVE,锁序 metapage → directory →
posting 无反向。实测 32 并发混合压测(16 插入+16 查询,45s):插入
1758 行/s 稳态、零错误、无死锁;VACUUM 后 recall 指纹与精确序一致。
LWLock 版是 preload 化升级,页格式/WAL 布局不需要变。

顺带抓到一个越界 bug:ORDER BY 参数为 NULL(查询行刚被删)时
`sk_argument=NULL` 直接段错误——现在退化为全索引扫描(行集合正确,
序无定义,与 seqscan+sort 一致)。

## 5. SIFT1M 实测(PG 16.13,i7-11800H,AVX-512,11GB,WSL2)

环境:SIFT1M 1M×128,查询集前 200,recall@10 对 groundtruth top-10;
nlist=1000;rerank 开;`dann_sq8` γ=20,`dann_rq` γ=40。

### 5.1 构建与体积

| | dann RaBitQ | dann SQ8 | IVFFlat | HNSW |
|---|---|---|---|---|
| 构建时间 | **16.6s** | 19.8s | 36.4s | 334s |
| 索引体积 | **39MB** | 146MB | 525MB | 782MB |

### 5.2 单线程 recall–延迟网格

| 点 | recall@10 | avg | p99 |
|---|---|---|---|
| dann_rq p=10 | 0.887 | 3.68ms | 5.70ms |
| dann_rq p=50 | 0.9935 | 8.52ms | 12.30ms |
| dann_sq8 p=50 | 0.9935 | 9.10ms | 12.67ms |
| ivfflat p=10 | 0.883 | 9.49ms | 17.73ms |
| ivfflat p=50 | 0.9950 | 21.91ms | 36.78ms |
| hnsw ef=40 | 0.946 | 2.79ms | 4.35ms |
| hnsw ef=160 | 0.9945 | 7.11ms | 11.51ms |

(全网格见 bench/out/pareto_sift1m_200q.csv)

### 5.3 32 并发代表点(20s/点)

| 点 | recall | QPS | avg | p99 |
|---|---|---|---|---|
| dann_sq8 p10/γ20 | 0.864 | 2262 | 13.7ms | 35.5ms |
| dann_rq p10/γ40 | 0.865 | 2114 | 15.0ms | 39.7ms |
| ivfflat p10 | 0.865 | 1429 | 22.2ms | 62.6ms |
| hnsw ef40 | 0.928 | 1279 | 24.8ms | 64.1ms |
| **dann_rq p50/γ40** | **0.989** | **932** | 34.2ms | 73.9ms |
| ivfflat p50 | 0.991 | 278 | 112.2ms | 245.2ms |
| hnsw ef160 | 0.993 | 415 | 74.9ms | 146.7ms |

**recall≈0.99 档 32 并发:pg_dann RaBitQ 的 QPS = 2.2× HNSW、3.4×
IVFFlat。** 小索引全缓存 + 连续页段预读在高并发下兑现了磁盘布局红利。

### 5.4 冷/热分组(restart+drop_caches vs 重跑,100 查询/遍)

| 引擎 | 冷 avg | 冷 p99 | 热 avg | 热 p99 |
|---|---|---|---|---|
| dann_rq p50/γ40 | 61.7ms | 87.8ms | 9.33ms | 12.3ms |
| dann_sq8 p50/γ20 | 44.5ms | 76.5ms | 10.55ms | 18.4ms |
| ivfflat p50 | 42.4ms | 143.6ms | 29.8ms | 70.1ms |
| hnsw ef160 | 153.6ms | 693.3ms | 7.92ms | 14.6ms |

HNSW 冷启动随机图游走对磁盘路径最不友好(冷 p99 693ms);
rerank 的回表在冷态可见(γk 个堆页),但 p99 仍 2× 好于 IVFFlat。

### 5.5 最小 shared_buffers

协议:recall≈0.99 代表点,每档 shared_buffers 做 restart+drop_caches →
第一遍预热 → 第二遍计时;通过线 = 第二遍 avg ≤ 1.5× 该引擎 2048MB 基线
(100 查询/遍)。

| 引擎(代表点) | 基线@2048MB | 最小 shared_buffers |
|---|---|---|
| dann_rq p50/γ40 | 8.55ms | **32MB**(扫描下界,1.33× 通过) |
| dann_sq8 p50/γ20 | 9.33ms | **32MB**(1.12× 通过) |
| ivfflat p50 | 23.58ms | **32MB**(0.99× 通过) |
| hnsw ef160 | 3.35ms | **768MB**(32MB 时 2.5× 慢) |

诚实注记:这台 11GB 机器的 OS page cache 会在预热遍兜底(drop_caches 只
清一次),所以该法分辨的是"PG buffer 缩小 + OS 缓存仍在"的敏感度。
ivfflat 与 dann 的 posting 都是连续段读(dann 另有批量预取),OS 缓存命中
代价低,32MB 均不衰减;hnsw 随机图游走每跳一个独立页,buffer 小于索引
(782MB)后每跳退化为系统调用读,768MB 才回到基线 1.5× 内。驻留差异的
净证据在 §5.4 真冷组:hnsw 冷 p99 693ms vs dann_rq 88ms。

### 5.6 bigann-10M 子集(数据超内存,降级对比)

SIFT1B 前 10M 行切片(uint8 128d → `vector(128)`),表 5581MB——超过本机
内存的一半,而 shared_buffers 只有 128MB,磁盘路径真实受力。查询集 1000
条,groundtruth 为分块暴力精确 top-100;nlist/lists 均 4096(≈√10M)。

**构建**:dann RaBitQ 8m58s 建出 **364MB** 索引(36B/entry),
maintenance_work_mem=512MB 照建(两遍式,O(采样)封顶);ivfflat 同参数
lists=4096 直接要求 **mwm ≥ 3374MB**(不够就报错不建),给到 3584MB 后
9m41s 建出 **5225MB**(14.3×)。构建内存的差异化随规模拉大:SIFT1M 时是
222MB vs 64MB,10M 时是 3374MB vs 512MB。

**recall@10 网格**(1000 查询,自然 OS 缓存态):

| probes | dann_rq(γ=40) recall | avg | ivfflat recall | avg |
|---|---|---|---|---|
| 1  | 0.3247 | 2.99ms  | 0.3417 | 5.29ms  |
| 5  | 0.6780 | 4.62ms  | 0.6882 | 9.67ms  |
| 10 | 0.8073 | 6.10ms  | 0.8127 | 10.68ms |
| 20 | 0.8957 | 10.35ms | 0.9016 | 20.45ms |
| 50 | 0.9643 | 23.12ms | 0.9697 | 49.25ms |
| 100 | **0.9847** | **43.80ms** | **0.9907** | **89.74ms** |

同档位(probes=100):recall 低 0.6pp 换 avg **2.05×**、p99 **2.4×**
(74.3ms vs 177.8ms);索引小 14.3×。数据超内存后 ivfflat 每查询要扫
520B/条的全精度 posting,而 dann 扫 36B/条的量化码、只对 γk 个候选回表
——1M 全缓存档的"布局红利"在磁盘受力场景兑现得更干脆。

**降级说明**:HNSW@10M 本轮不测——pgvector HNSW 构建需整图常驻
maintenance_work_mem(10M×(520B+链接) ≈ 6.5GB+),叠加 5.6GB 表的构建
I/O 超出这台 11GB 测试机的安全边界(测试机限制,非引擎结论;1M 对比见
§5.1–5.4)。dann SQ8@10M 亦略去:RaBitQ vs SQ8 的体积/精度故事在 1M 已
闭环(§1.3/§5.1),10M 的论证点是"数据超内存后对全精度基线的对比"。

### 5.7 bge-m3 真实 embedding(100k×1024d,cosine):一次被数据打脸的对拍

SIFT/bigann 是合成分布,最后一组上真实 RAG 负载:土耳其语维基 2023 的
bge-m3 dense embedding(hf-mirror 预计算数据集,取前 10 万条,单位归一,
cosine 度量,nlist/lists=320,查询为底库等距抽样 500 条)。

体积与构建(与 SIFT1M 同向,高维放大):dann_rq **18MB** / dann_sq8 104MB /
ivfflat **784MB**——ivfflat 在 1024d 每页 8KB 只装得下一条 4.1KB tuple,
近半页空转,体积差从 13.5× 拉到 **43×**。

但 recall 网格出了意外:

| probes | dann_rq γ=40 | dann_sq8 γ=20 | ivfflat |
|---|---|---|---|
| 20 | 0.9238 / 4.1ms | 0.9454(γ=40) / 6.2ms | 0.9686 / 13.8ms |
| 50 | 0.9350 / 6.1ms | **0.9848 / 8.4ms** | **0.9940 / 31.6ms** |

dann_rq 的 recall 在 probes=50 平台于 0.935。逐级排查:probes 翻倍到 100
而 γ 不变,recall 反略降至 0.9208(候选变多、重排序名额不变)——不是
probes 不够;γ 从 40 提到 100/200,recall 升到 0.9654/0.9768——是 **γ
不够**:真邻居在 RaBitQ 估计序里排在 γk 名之外,根本轮不到精确重排。
对照 dann SQ8(估计序近精确)同 probes 下 γ=20 即 0.9848:同一套聚簇
分配,recall 差 5pp,瓶颈被钉死在 RaBitQ 估计噪声上。

机理:W7-8 验证过 σ·√d 恒定(1/√d 律),SIFT 上 γ=40 足够。但 bge-m3
这类语义 embedding 的近邻"挤"——top-10 内外向量的 cos 差距常常小于估计
噪声,同样的 σ 在均匀分布里撼动不了顺序,在语义空间里就会重排。**"RaBitQ
高维甜区"要按分布修正:语义负载下要么 γ≥200,要么回 SQ8**(1KB 码仍比
ivfflat 全精度条目小 4 倍)。

最终同档位:dann_sq8 p50/γ20 0.9848@8.4ms vs ivfflat p50 0.9940@31.6ms
——**3.8× 快、体积小 7.5×,recall 低 0.9pp**。结论站住了,但站住的方式
和预想不同:高维语义负载下的默认推荐从 RaBitQ 滑向 SQ8(见 §6 已知限制)。

## 6. 已知限制

见 README.md"已知限制(v1)":空页不回收靠 REINDEX;插入串行化;
无并行构建;PG18 read stream 与 NEON 为编译期 TODO;CI 矩阵 PG17/18
与 ASAN 纳入 CI 待补。另加一条实测结论(§5.7):**RaBitQ 在高维语义
embedding 上估计序噪声会吃掉 recall 上限**,需 γ≥200 或改用 SQ8;
`quantizer=auto` 的维度阈值(dim<128 → SQ8)不感知数据分布,v2 候选
是 γ 自适应(按估计序分歧度动态调)或构建期分布探测。

## 7. 数据与复现

`bench/RESULTS.md` 为逐里程碑验收记录;复现脚本:recall.py /
budget.py / conc_test.py / pareto.py / coldhot.sh / minbuf.py /
crash_test.sh(注意:后三个会重启/杀实例);bigann-10M 复现:
load_bigann.py → gt_bigann.py → 建索引(nlist/lists=4096)→
recall_bigann.py;bge-m3 复现:extract_bgem3.py → load_bgem3.py →
gt_bgem3.py → 建索引(nlist/lists=320,cosine ops)→ recall_bgem3.py。
