# pg_dann benchmark 记录

## W12 bge-m3 真实 embedding(2026-09-16,100k × 1024d,cosine)

数据:hcsolakoglu/turkish_wikipedia_with_bge_m3_embeddings(hf-mirror,
土耳其语维基 2023,bge-m3 dense,float16,源 533k 行取前 100k,
`extract_bgem3.py` 抽成 fvecs)。向量已单位归一(norm 1.0000±0.0001)。
度量 cosine(`<=>`,dann_cosine_ops / vector_cosine_ops)。查询 = 底库每
200 行一条(500 条,top-1 为自身,各引擎同待遇),gt 暴力余弦 top-100
(`gt_bgem3.py`)。nlist/lists=320(≈√100k)。

**构建与体积**(mwm=512MB):

| | dann RaBitQ | dann SQ8 | ivfflat |
|---|---|---|---|
| 构建时间 | 9.5s | 12.2s | 12.9s |
| 索引体积 | **18 MB**(148B/entry) | 104 MB | 784 MB(**43×**) |

ivfflat 在 1024d 每页 8KB 只装得下一条 4.1KB tuple,近半页空转——
全精度 posting 的页效率问题在低维被掩盖、在高维放大。

**recall@10 网格**(500 查询,自然缓存态):

| probes | dann_rq γ40 | dann_sq8 γ20 | ivfflat |
|---|---|---|---|
| 1  | 0.5718 / 2.36ms | — | 0.5922 / 1.67ms |
| 5  | 0.8342 / 3.02ms | — | 0.8584 / 4.57ms |
| 10 | 0.8916 / 3.33ms | — | 0.9230 / 7.78ms |
| 20 | 0.9238 / 4.08ms | 0.9454(γ40) / 6.16ms | 0.9686 / 13.75ms |
| 50 | 0.9350 / 6.13ms | **0.9848 / 8.40ms** | **0.9940 / 31.59ms** |

**RaBitQ 高维语义负载下的 γ 诊断**(本数据集最重要的发现):
dann_rq 在 probes=50 后 recall 平台在 0.935——不是 probes 不够
(probes 翻倍到 100、γ 不变反而 0.9208 略降),而是 γ 不够:真邻居在
RaBitQ 估计序里排在 γk=400 名外。γ=100 → 0.9654,γ=200 → 0.9768。
对照 dann SQ8(估计近精确)同 probes=50、γ=20 即 0.9848——同一聚簇
分配下 recall 差 5pp,证明瓶颈是 RaBitQ 估计序噪声,不是 IVF 聚簇。
机理:SIFT 类均匀分布里近邻间距大,σ·√d 恒定噪声不撼动顺序;bge-m3
语义空间里近邻挤(语义近邻 cos 差距小),同样 σ 就会重排,需更大 γ
或更高保真量化码兜底。

**同档位对比**:recall≈0.98 档 dann_sq8 p50/γ20 0.9848@8.40ms vs
ivfflat p50 0.9940@31.59ms——**3.8× 快,recall 低 0.9pp,体积小 7.5×**。
dann_rq γ200 档 0.9768@14.35ms——2.2× 快,低 1.7pp,体积小 43×。

对设计的修正:`quantizer=auto` 的 dim<128 阈值只看维度不看分布,
高维语义负载下 RaBitQ 需要 γ≥200 或回 SQ8——v2 候选:γ 自适应
(按估计序分歧度)或构建期分布探测。已写入 ARTICLE §5.7/§6。

复现:`extract_bgem3.py` → `load_bgem3.py` → `gt_bgem3.py` →
建索引(nlist/lists=320,cosine ops)→ `recall_bgem3.py`。

## W12 内存:最小 shared_buffers(2026-09-16,SIFT1M)

协议(DESIGN §6):recall≈0.99 代表点,每档 shared_buffers 做
restart+drop_caches → 第一遍预热 → 第二遍计时;通过线 = 第二遍 avg ≤
1.5× 该引擎 2048MB 基线。100 查询/遍,`minbuf.py`,日志 bench/out/minbuf.log。

| 引擎(代表点) | 基线@2048MB | 最小 shared_buffers | 该档 avg |
|---|---|---|---|
| dann_rq p50/γ40 | 8.55ms | **32MB**(下界) | 11.38ms |
| dann_sq8 p50/γ20 | 9.33ms | **32MB**(下界) | 10.41ms |
| ivfflat p50 | 23.58ms | **32MB**(下界) | 23.23ms |
| hnsw ef160 | 3.35ms | **768MB** | 3.21ms |

(32MB 是扫描档下界,三引擎在最小档即通过;hnsw 从 32MB 的 8.43ms
(2.5× 基线)逐档收窄,768MB 才回到 1.5× 内。)

**解读与方法论注记**:本机 11GB,drop_caches 后第一遍预热会把数据再装进
OS page cache(数 GB 空间),所以该法测的是"PG buffer pool 缩小、OS 缓存
兜底"下的敏感度。ivfflat/dann 的 posting 是连续段读(+dann 有批量预取),
OS 缓存命中代价低,32MB 也不衰减;hnsw 的随机图游走每跳都是独立页,PG
buffer 小于索引体积(782MB)时每跳退化为 OS 缓存读(syscall+拷贝),
768MB 以下明显变慢。结论按机注明:在此测试机上,dann/ivfflat 对 PG
buffer 不敏感(32MB 通过),hnsw 需 ≥768MB;"dann 与 ivfflat 的驻留差异
要靠真实冷磁盘(drop_caches 后第一遍)区分"——见 W12-SIFT1M 冷热组
(hnsw 冷 p99 693ms vs dann_rq 88ms)。

## W12 bigann-10M 子集(2026-09-16,数据超内存主场)

环境:PG 16.13 衍生发行版(6543),i7-11800H AVX-512,WSL2 11GB,shared_buffers=128MB。
数据:SIFT1B 前 10M 行切片(bvecs uint8 128d,COPY 文本加载,`load_bigann.py`),
表 5581MB(超内存一半,buffer pool 只有 128MB,磁盘路径真实受力);
查询集 1000 条,groundtruth 暴力精确 top-100 分块算(`gt_bigann.py`,
gt10m_1000q.ivecs)。nlist/lists 均 4096(≈√10M)。

**构建**:

| | dann RaBitQ | ivfflat |
|---|---|---|
| 构建时间 | **8m58s** | 9m41s |
| 索引体积 | **364 MB**(36B/entry) | 5225 MB(**14.3×**) |
| 构建内存门槛 | mwm=512MB 照建(采样封顶) | **mwm < 3374MB 直接报错** |

构建内存对比随规模拉大:SIFT1M 时 ivfflat 需 222MB(64MB 报错),
10M/nlist=4096 时需 3374MB;pg_dann 两遍式构建内存 O(采样) 与表大小无关,
512MB 不变。

**dann_rq recall@10 网格**(γ=40,自然 OS 缓存态,1000 查询):

| probes | recall@10 | avg_ms | p50_ms | p99_ms |
|---|---|---|---|---|
| 1  | 0.3247 | 2.99  | 2.65  | 8.26  |
| 5  | 0.6780 | 4.62  | 4.23  | 9.54  |
| 10 | 0.8073 | 6.10  | 5.74  | 11.65 |
| 20 | 0.8957 | 10.35 | 9.59  | 20.16 |
| 50 | 0.9643 | 23.12 | 21.85 | 40.71 |
| 100 | **0.9847** | 43.80 | 41.25 | 74.28 |

recall≥0.98 档位在 probes=100(扫 2.4% list)。

**HNSW@10M 不测的说明**:pgvector HNSW 构建需整图常驻 maintenance_work_mem
(10M×(520B 向量+链接) ≈ 6.5GB+),叠加 5.6GB 表的构建 I/O 超出这台 11GB
测试机的安全边界,本轮降级不测(测试机限制,非引擎结论;1M 对比见 §W7-8/W12-SIFT1M)。

**ivfflat recall@10 网格**(同查询同 gt,自然 OS 缓存态):

| probes | recall@10 | avg_ms | p50_ms | p99_ms |
|---|---|---|---|---|
| 1  | 0.3417 | 5.29  | 4.94  | 14.97  |
| 5  | 0.6882 | 9.67  | 8.28  | 30.88  |
| 10 | 0.8127 | 10.68 | 9.05  | 37.93  |
| 20 | 0.9016 | 20.45 | 16.66 | 59.64  |
| 50 | 0.9697 | 49.25 | 42.51 | 122.28 |
| 100 | **0.9907** | 89.74 | 80.81 | 177.78 |

**同档位对比**:probes=100 档 dann_rq 0.9847@43.8ms vs ivfflat 0.9907@89.7ms
——recall 低 0.6pp,avg 快 **2.05×**、p99 好 **2.4×**;probes=50 档
0.9643@23.1ms vs 0.9697@49.3ms,avg 快 **2.1×**。索引体积小 14.3×、
构建内存门槛低 6.6 倍。数据超内存后,"小索引+回表重排"对"全精度胖索引"
的布局红利兑现(与 SIFT1M 全缓存档结论同向,差距主要来自 ivfflat 每查询
扫描 520B/条的全精度 posting)。

复现:`load_bigann.py` → `gt_bigann.py` → 建索引(nlist/lists=4096)→
`recall_bigann.py 1000 [engine]`;日志在 bench/out/{load,gt,recall}_bigann*.log。

## W10 验收(2026-09-15,并发插入 + VACUUM)

**并发协议**:插入的"读 directory → 尾页追加 → 写 directory"由 metapage
BUFFER_LOCK_EXCLUSIVE 串行化(与 IVFFlat 同协议)。**偏离 §4.7 的说明**:
128 把 list LWLock 需 shared_preload_libraries(named tranche 只能在集群
启动时注册);v1 正确性优先选 metapage 锁,页格式/WAL 布局与 LWLock 版
完全兼容,preload 化后可无感升级。

**bulkdelete**:全索引扫描 + 回调判死 + 页内 swap-with-last 紧凑删除
(走 generic_xlog,pd_lower 随内容收缩);directory count 在 metapage 排他
锁下按 dir 页批量更新;空页不回收(与 IVFFlat 一致,靠 REINDEX)。

**压测**(`bench/conc_test.py`,16 插入 + 16 查询 × 45s,rabitq 索引
nlist=100,dim=128):

| 指标 | 结果 |
|---|---|
| 插入 | 80000 行 @ 1758/s,零错误,无死锁 |
| 查询 | 16 线程 × ~7600 次,零错误 |
| 行数一致性 | 表 count = 20000 + 插入数,精确相等 |
| 索引↔表一致 | 压测后/VACUUM 后指纹均 == 精确序(probes=nlist+rerank) |
| VACUUM 精确删除 | entries: 100000 → 75000 == 存活行数 |

**顺手抓的越界 bug**:ORDER BY 参数为 NULL(如查询行刚被删)时
`sk_argument=NULL` 导致 detoast 段错误——`dann_exec_scan` 现对 SK_ISNULL
退化为全索引扫描(行集合正确,序无定义,与 seqscan+sort 一致);
installcheck 已含 NULL 用例(vacuum.sql)。

## W9 验收(2026-09-15,WAL 化 + 崩溃恢复)

**WAL 化**(generic_xlog,全部页写入):构建(metapage/directory/centroid/
posting 全量页镜像)+ 插入(尾页追加/新页/directory 更新单 GenericXLog
原子提交);`!RelationNeedsWAL` 时构建结束 smgrimmedsync 兜底(nbtsort
约定)。WAL 体积 ≈ 索引体积(正确性优先;定制 rmgr 是 v2 候选,§4.6)。

**踩坑记录**(文章素材):
1. `GenericXLogFinish()` 不解锁不释放 buffer——调用方必须
   `UnlockReleaseBuffer`,否则锁泄漏到事务结束,同会话后续访问自死锁。
2. generic_xlog 在线应用与 redo 都把"hole"`[pd_lower, pd_upper)` 清零。
   本 AM 的定长紧密内容恰好排在 hole 区(PageGetContents 起,无 line
   pointer)——必须把内容末端并进 `pd_lower`(`dann_page_set_lower`),
   否则内容被抹掉。旧格式页插入时统一抬升 pd_lower 即兼容。
3. 该发行版的 backend 是独立进程组(pgid=pid):kill -9 postmaster 进程组
   杀不死 backend;崩溃注入要按 /proc/exe 路径逐个点名。
4. kill -9 后 SysV 共享内存段成孤儿(nattch=0),挡重启,需 ipcrm 清理。

**崩溃恢复矩阵**(`bench/crash_test.sh`,kill -9 整机模拟):

| 注入点 | 结果 |
|---|---|
| (a) 构建中途(1M 行,活跃 2s 断电) | 恢复后事务回滚干净(索引 gone),表完好,重建可查 ✓ |
| (b) 批量插入中途(60 批×1000 行,断在第 41 批) | 行数精确 = 已提交批(91000),索引指纹 == seqscan 精确指纹,可继续插入 ✓ |
| (c) VACUUM 中途 | 索引一致可用 ✓ |

一致性判据:probes=nlist + rerank 的索引扫描聚合指纹(count/sum/min/max)
与强制 seqscan 逐字节相等。另验证:已提交构建在断电后索引完整可查
(attempt 命中已提交分支 3 次均通过);W9 前格式索引(pd_lower=24)插入
新行不丢旧内容。

## W7-8 验收(SIFT1M,2026-09-15)

环境同前(PG 16.13 衍生发行版,AVX-512)。RaBitQ:旋转 P = 置换∘FWHT∘符号翻转
(SplitMix32 种子存 metapage,构建可复现);估计式 §4.2;C 端 scalar/AVX2/
AVX-512 三档,golden model 对拍 27328 项全绿(`make quant-test`);
误差界实测 σ_cos·√N ≈ 1.30–1.45 恒定(1/√d 律成立,
`golden/rabitq_error_curve.csv`)。micro-bench:条目打分 AVX-512 比
-O2 标量基线快 64×(128d:7.5ns vs 481ns;SQ8-AVX2 14.4ns)。

### recall@10(rerank 均开,γ=20)

| probes | dann SQ8 | dann RaBitQ | ivfflat | RaBitQ avg/p99 | SQ8 avg/p99 |
|---|---|---|---|---|---|
| 1  | 0.3645 | 0.3645 | 0.3665 | 0.88/1.58ms | 1.55/14.74ms |
| 5  | 0.7450 | 0.7450 | 0.7585 | 1.38/2.12ms | 1.47/2.82ms |
| 10 | 0.8870 | 0.8865 | 0.8830 | 1.91/3.11ms | 2.41/3.64ms |
| 20 | 0.9580 | 0.9550 | 0.9570 | 3.27/4.63ms | 4.01/5.62ms |
| 50 | 0.9935 | 0.9870 | 0.9950 | 7.43/10.58ms | 8.58/11.57ms |

### γ 曲线(probes=50,RaBitQ):γ=40 即追平 SQ8 且仍更快

| γ | recall@10 | avg_ms | p99_ms |
|---|---|---|---|
| 20 | 0.9870 | 7.89 | 11.08 |
| 40 | **0.9935** | **8.33** | 11.93 |
| 80 | 0.9940 | 9.19 | 12.55 |

**验收线:recall@10 ≥ 0.98 ✓(γ=20 时 0.987;γ=40 时 0.9935 追平 SQ8);
同 recall 下延迟/体积优于 SQ8 ✓**(0.9935 水位:8.33ms vs 8.58ms,39MB vs 146MB)。

### 索引体积与构建(nlist=1000,maintenance_work_mem=512MB)

| 指标 | RaBitQ | SQ8 | ivfflat |
|---|---|---|---|
| 索引体积 | **39 MB** | 146 MB(3.7×) | 525 MB(13.5×) |
| 构建时间 | **16.6s** | 19.8s | 36.4s |

entry:RaBitQ 36B(20B 头 + 128bit 码)vs SQ8 148B vs ivfflat 520B。
auto 阈值:dim<128 → SQ8(installcheck 体积相等性验证,rabitq.sql §4)。

复现:`CREATE TABLE sift_rq AS SELECT * FROM sift_base` + rabitq 索引
(16.6s)→ `bench/recall.py 200`(三引擎网格);γ 曲线见 RESULTS.md 记录。

## W6 验收(SIFT1M,2026-09-15)

环境同 W4-5。W6 内容:批量预取(`dann.io_method`)+ heap rerank(ctid 排序回表,
`dann.rerank_factor`=γ,默认 20)+ 延迟预算截断(`dann.max_latency_ms`)。
rerank 为懒加载分块:每块 γ×10 个估计序最优候选(amgettuple 拉取式不知 LIMIT k,
按典型 k=10 起跳,耗尽再补),块内按精确距离排序后发射。

### recall@10 vs IVFFlat(同 probes;rerank 后应贴近"候选内精确"水位)

| probes | dann W6(rerank) | W4-5(估计序,无 rerank) | ivfflat | dann avg/p99 | ivfflat avg/p99 |
|---|---|---|---|---|---|
| 1  | 0.3645 | 0.3645 | 0.3665 | 1.25/3.08ms | 4.49/8.37ms |
| 5  | 0.7450 | 0.7420 | 0.7585 | 1.99/4.83ms | 8.35/44.39ms |
| 10 | 0.8870 | 0.8820 | 0.8830 | 2.81/5.32ms | 11.05/41.43ms |
| 20 | 0.9580 | 0.9485 | 0.9570 | 4.59/6.77ms | 10.72/31.80ms |
| 50 | **0.9935** | 0.9835 | 0.9950 | 9.88/13.68ms | 23.85/49.56ms |

rerank 把 SQ8 量化误差抹平:recall 从"估计序"提升到候选集精确水位
(probes=10/20 甚至略超 ivfflat——k-means 聚簇不同的噪声,±0.4pp 级)。
**验收线 recall@10 ≥ 0.98:probes=50 时 0.9935 ✓**(同 recall 下仍比 ivfflat 快 2.4×,
p99 13.68ms vs 49.56ms)。rerank 代价:probes=50 avg 8.65→9.88ms(200 个
ctid 排序后的回表抓取)。

### 延迟预算截断(probes=50,200 查询,验收:截断生效且 recall 平滑降级)

| budget | recall@10 | avg_ms | p99_ms | truncated/scans | lists_skipped |
|---|---|---|---|---|---|
| 0(不限) | 0.9935 | 9.87 | 15.18 | 0/200 | 0 |
| 1ms | 0.9815 | 5.43 | 6.57 | 200/200 | 4514 |
| 2ms | 0.9915 | 8.97 | 12.10 | 64/200 | 538 |
| 4ms | 0.9935 | 9.35 | 14.62 | 1/200 | 1 |
| 8ms | 0.9935 | 9.24 | 12.29 | 0/200 | 0 |

1ms 预算砍掉 45% 平均延迟、57% p99,recall 只掉 1.2pp——list 按质心距离
升序处理,截断丢的是尾部最远 list,降级平滑 ✓。截断事件计入
`dann_scan_stats()`(会话级;W11 扩展)。

### 预取

`dann.io_method=auto` 在 PG16 上走 `PrefetchBuffer`(posix_fadvise WILLNEED),
按 directory 连续页区间合并后发起;PG18 read stream 留 `#if` TODO。
热缓存下 sync vs auto 无差异(9.41 vs 9.47ms,146MB 索引全在 buffer pool,
PrefetchBuffer 对已缓存块直接跳过)——价值在冷启动/磁盘读场景,未单独测量。

复现:`bench/recall.py 200`(recall/p99 网格);`bench/budget.py 200`(预算扫描)。

## W4-5 验收(SIFT1M,2026-09-10)

环境:PG 16.13 衍生发行版(本地实例,端口 6543),x86_64,AVX-512;shared_buffers 默认。
数据:SIFT1M base 1M×128(hf-mirror 下载),查询集前 200 条,groundtruth top-10。
索引:nlist=1000;pg_dann 用 SQ8 无 rerank(W4-5 形态),ivfflat 精确距离。

### recall@10 vs IVFFlat(同 probes,验收线"同水位")

| probes | pg_dann(SQ8) | ivfflat | dann 平均延迟 | ivfflat 平均延迟 | dann 加速比 |
|---|---|---|---|---|---|
| 1  | 0.3645 | 0.3665 | 0.49ms | 0.69ms | 1.4× |
| 5  | 0.7420 | 0.7585 | 1.66ms | 2.34ms | 1.4× |
| 10 | 0.8820 | 0.8830 | 2.17ms | 4.68ms | 2.2× |
| 20 | 0.9485 | 0.9570 | 3.86ms | 8.73ms | 2.3× |
| 50 | 0.9835 | 0.9950 | 8.65ms | 24.09ms | 2.8× |

recall 同水位(差距 ≤1.2pp,即 SQ8 量化误差,W6 rerank 的目标);
同 probes 下查询快 1.4–2.8×(posting entry 148B vs IVFFlat 全精度 520B)。

### 索引体积与构建

| 指标 | pg_dann | ivfflat |
|---|---|---|
| 索引体积 | **146 MB** | 525 MB(3.6×) |
| 构建时间(maintenance_work_mem=512MB) | **19.8s** | 36.4s(1.8×) |
| 构建内存 | 采样封顶 64MB 即可构建 | 低于 ~222MB 直接报错 |

ivfflat 在 maintenance_work_mem=64MB 下构建 SIFT1M 直接报
"memory required is 222 MB"——O(采样) 构建的差异化已被实测验证。

复现:`bench/load_sift.py` → 建两表两索引 → `bench/recall.py 200`。
