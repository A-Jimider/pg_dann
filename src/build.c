/*-------------------------------------------------------------------------
 * build.c — ambuild / ambuildempty:三遍式 IVF 构建(DESIGN.md §4.3)
 *
 *   Pass A:堆扫描 + 蓄水池采样(确定性 LCG,结果可复现)
 *   k-means:采样集上 Lloyd(见 kmeans.c)
 *   Pass B:堆扫描,逐元组分配最近质心,per-list 计数;
 *          assignment(每行 list id)留存供 Pass C 重放
 *   布局:directory / centroid / posting 按计数连续分配(物理聚簇)
 *   Pass C:堆扫描,重放 assignment,编码写入 posting 页段
 *
 * 内存占用:O(采样集 + nlist 页缓冲 + N×2B assignment)——全精度向量
 * 从不全量驻留。
 *-------------------------------------------------------------------------*/
#include "postgres.h"

#include "access/generic_xlog.h"
#include "access/tableam.h"
#include "access/tupdesc.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "storage/smgr.h"
#include "utils/rel.h"

#include "dann.h"
#include "quant.h"

#define DANN_KMEANS_ITERS	15

typedef struct DannBuildState
{
	Relation	index;
	int			dim;
	uint32		padded_dim;		/* 补零到 2 的幂(RaBitQ 旋转空间) */
	uint32		rotation_seed;	/* 由 dim/nlist 派生(构建可复现) */
	uint32		metric;
	uint32		quantizer;		/* 已解析(auto → 具体) */
	int			nlist;
	int			entry_size;
	int			entries_per_page;
	int			centroids_per_page;
	DannRqCtx	rq;				/* RaBitQ 旋转上下文(quantizer=rabitq 时初始化) */

	/* Pass A:蓄水池采样 */
	float	   *samples;		/* sample_cap × dim */
	int			nsamples;
	int			sample_cap;
	uint64		rng;
	double		reltuples;		/* 存活行数(各遍扫描一致) */

	/* k-means 输出 */
	float	   *centroids;		/* nlist × dim */

	/* Pass B:分配与计数 */
	uint16	   *assignments;	/* 每行序号的 list id */
	uint32	   *counts;			/* 每 list 条目数(pass C 倒计数校验) */
	uint64		rowno;

	/* 布局 */
	uint32		dir_npages;
	uint32		centroid_npages;
	BlockNumber posting_start;
	BlockNumber total_pages;
	BlockNumber *list_first;	/* 每 list 段首块(空 list 为 Invalid) */
	uint32	   *list_nentries;	/* 每 list 原始计数(pass C 不清零) */

	/* Pass C:每 list 一个 8KB 私有页缓冲 */
	char	   *listbufs;		/* nlist × BLCKSZ */
	BlockNumber *list_cur;		/* 当前目标块 */
} DannBuildState;

/* 把向量拷贝进工作区并按度量归一化(cosine);NaN/Inf 拒绝 */
static void
dann_prepare_vec(DannBuildState *bs, DannVector *v, float *ws)
{
	memcpy(ws, v->x, sizeof(float) * bs->dim);
	dann_check_finite(ws, bs->dim);
	if (bs->metric == DANN_METRIC_COSINE)
		dann_normalize(ws, bs->dim);
}

/* ---- Pass A:蓄水池采样 ---- */
/*
 * wal_level=minimal 时 generic_xlog 不写 WAL(RelationNeedsWAL=false),
 * 构建页直接落盘兜底(对齐 nbtsort 的 smgrimmedsync 约定);
 * unlogged 表崩溃后由 init fork 重建,无需 sync。
 */
static void
dann_build_sync_if_needed(Relation index)
{
	if (!RelationNeedsWAL(index) &&
		index->rd_rel->relpersistence == RELPERSISTENCE_PERMANENT)
		smgrimmedsync(RelationGetSmgr(index), MAIN_FORKNUM);
}

static void
dann_sample_callback(Relation index, ItemPointer tid, Datum *values,
					 bool *isnull, bool tupleIsAlive, void *state)
{
	DannBuildState *bs = (DannBuildState *) state;
	float	   *ws;
	DannVector *v;

	if (!tupleIsAlive || isnull[0])
		return;

	bs->reltuples += 1.0;

	ws = (float *) palloc(sizeof(float) * bs->dim);
	v = dann_get_vector(values[0], bs->dim);
	dann_prepare_vec(bs, v, ws);
	dann_free_vector(values[0], v);

	if (bs->nsamples < bs->sample_cap)
	{
		memcpy(bs->samples + (size_t) bs->nsamples * bs->dim, ws,
			   sizeof(float) * bs->dim);
		bs->nsamples++;
	}
	else
	{
		/* 算法 R:第 i 行(0 起)以 cap/(i+1) 概率入样 */
		uint64		i = (uint64) bs->reltuples - 1;
		uint64		j;

		bs->rng = bs->rng * 6364136223846793005ULL + 1442695040888963407ULL;
		j = (bs->rng >> 33) % (i + 1);
		if (j < (uint64) bs->sample_cap)
			memcpy(bs->samples + (size_t) j * bs->dim, ws,
				   sizeof(float) * bs->dim);
	}
	pfree(ws);
}

/* ---- Pass B:分配 + 计数 ---- */
static void
dann_count_callback(Relation index, ItemPointer tid, Datum *values,
					bool *isnull, bool tupleIsAlive, void *state)
{
	DannBuildState *bs = (DannBuildState *) state;
	float	   *ws;
	DannVector *v;
	uint32		listno;

	if (!tupleIsAlive || isnull[0])
		return;

	ws = (float *) palloc(sizeof(float) * bs->dim);
	v = dann_get_vector(values[0], bs->dim);
	dann_prepare_vec(bs, v, ws);
	dann_free_vector(values[0], v);

	listno = dann_assign(ws, bs->centroids, bs->nlist, bs->dim, bs->metric);
	bs->assignments[bs->rowno++] = (uint16) listno;
	bs->counts[listno]++;

	pfree(ws);
}

/* entry 编码:header(ctid + 3×float4 因子)+ 量化码 */
static void
dann_encode_entry(DannBuildState *bs, ItemPointer tid, const float *vec,
				  uint32 listno, char *out)
{
	uint16	   *pad = (uint16 *) (out + 6);
	uint8_t    *code = (uint8_t *) (out + DANN_ENTRY_HEADER_SIZE);

	memcpy(out, tid, sizeof(ItemPointerData));
	*pad = 0;

	if (bs->quantizer == DANN_QUANTIZER_SQ8)
	{
		DannSQ8Factors *f = (DannSQ8Factors *) (out + 8);

		dann_sq8_encode(vec, bs->dim, code, f);
	}
	else
	{
		/* RaBitQ:相对本 list 质心编码 */
		DannRabitqFactors *f = (DannRabitqFactors *) (out + 8);
		const float *cen = bs->centroids + (size_t) listno * bs->dim;

		dann_rabitq_encode(vec, cen, bs->dim, (int) bs->padded_dim,
						   bs->rq.sign, bs->rq.perm, bs->rq.scratch,
						   code, &f->dist_c, &f->g);
		f->dist_c2 = f->dist_c * f->dist_c;
	}
}

/* 把 list 的当前页缓冲刷到 list_cur 指向的磁盘页(W9:generic_xlog 全页镜像) */
static void
dann_flush_list_page(DannBuildState *bs, uint32 listno)
{
	Page		image = (Page) (bs->listbufs + (size_t) listno * BLCKSZ);
	Buffer		buf;
	Page		page;
	GenericXLogState *gxlog;

	buf = ReadBuffer(bs->index, bs->list_cur[listno]);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	gxlog = GenericXLogStart(bs->index);
	page = GenericXLogRegisterBuffer(gxlog, buf, GENERIC_XLOG_FULL_IMAGE);
	memcpy(page, image, BLCKSZ);
	dann_page_set_lower(page, ((DannPageOpaque) PageGetSpecialPointer(image))->count
					  * (uint32) bs->entry_size);
	GenericXLogFinish(gxlog);
	UnlockReleaseBuffer(buf);	/* Finish 只应用+WAL,不解锁不释放 */
}

static void
dann_start_list_page(DannBuildState *bs, uint32 listno, BlockNumber blkno)
{
	Page		image = (Page) (bs->listbufs + (size_t) listno * BLCKSZ);
	DannPageOpaque opaque;

	PageInit(image, BLCKSZ, sizeof(DannPageOpaqueData));
	opaque = (DannPageOpaque) PageGetSpecialPointer(image);
	opaque->page_type = DANN_PAGE_POSTING;
	opaque->next = InvalidBlockNumber;
	opaque->count = 0;
	bs->list_cur[listno] = blkno;
}

/* Pass C:重放 assignment,编码并写入 list 页段 */
static void
dann_write_callback(Relation index, ItemPointer tid, Datum *values,
					bool *isnull, bool tupleIsAlive, void *state)
{
	DannBuildState *bs = (DannBuildState *) state;
	uint32		listno;
	Page		image;
	DannPageOpaque opaque;
	char	   *entry;
	float	   *ws;
	DannVector *v;

	if (!tupleIsAlive || isnull[0])
		return;

	listno = bs->assignments[bs->rowno++];

	ws = (float *) palloc(sizeof(float) * bs->dim);
	v = dann_get_vector(values[0], bs->dim);
	dann_prepare_vec(bs, v, ws);
	dann_free_vector(values[0], v);

	if (bs->list_cur[listno] == InvalidBlockNumber)
		dann_start_list_page(bs, listno, bs->list_first[listno]);

	image = (Page) (bs->listbufs + (size_t) listno * BLCKSZ);
	opaque = (DannPageOpaque) PageGetSpecialPointer(image);

	if (opaque->count == (uint32) bs->entries_per_page)
	{
		/* 段内连续推进(段大小按计数精确分配);先链接再冲刷 */
		BlockNumber next = bs->list_cur[listno] + 1;

		opaque->next = next;
		dann_flush_list_page(bs, listno);
		dann_start_list_page(bs, listno, next);
		image = (Page) (bs->listbufs + (size_t) listno * BLCKSZ);
		opaque = (DannPageOpaque) PageGetSpecialPointer(image);
	}

	entry = (char *) image + SizeOfPageHeaderData
		+ (size_t) opaque->count * bs->entry_size;
	dann_encode_entry(bs, tid, ws, listno, entry);
	opaque->count++;
	bs->counts[listno]--;		/* 倒计数,校验两遍一致性 */

	pfree(ws);
}

/* 预扩展关系到 total_pages(P_NEW 循环;内容由各写入方随后填充)。
 * 注意:P_NEW 返回的 buffer 只 pin 不 lock,只能 ReleaseBuffer——
 * UnlockReleaseBuffer 会释放未持有的内容锁(assert 构建下报错)。 */
static void
dann_extend_relation(Relation index, BlockNumber total)
{
	while (RelationGetNumberOfBlocksInFork(index, MAIN_FORKNUM) < total)
	{
		Buffer		buf = ReadBufferExtended(index, MAIN_FORKNUM, P_NEW,
											 RBM_NORMAL, NULL);

		ReleaseBuffer(buf);
	}
}

/* 写 centroid 页(质心不跨页:每页只放整数个) */
static void
dann_write_centroids(DannBuildState *bs)
{
	int			cpp = bs->centroids_per_page;
	uint32		p;

	for (p = 0; p < bs->centroid_npages; p++)
	{
		Buffer		buf;
		Page		page;
		DannPageOpaque opaque;
		GenericXLogState *gxlog;
		char	   *base;
		int			cnt;
		int			i;

		buf = ReadBuffer(bs->index, 1 + bs->dir_npages + p);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		gxlog = GenericXLogStart(bs->index);
		page = GenericXLogRegisterBuffer(gxlog, buf, GENERIC_XLOG_FULL_IMAGE);
		dann_init_page_image(page, DANN_PAGE_CENTROID);
		opaque = (DannPageOpaque) PageGetSpecialPointer(page);

		cnt = Min(cpp, bs->nlist - (int) p * cpp);
		opaque->count = cnt;
		base = (char *) PageGetContents(page);
		for (i = 0; i < cnt; i++)
			memcpy(base + (size_t) i * bs->dim * sizeof(float),
				   bs->centroids + (size_t) (p * cpp + i) * bs->dim,
				   sizeof(float) * bs->dim);
		dann_page_set_lower(page, (uint32) cnt * (uint32) bs->dim
						  * (uint32) sizeof(float));

		GenericXLogFinish(gxlog);
		UnlockReleaseBuffer(buf);
	}
}

/* 写 directory 页(最后写,需要各 list 的最终 last/count) */
static void
dann_write_directory(DannBuildState *bs)
{
	uint32		p;

	for (p = 0; p < bs->dir_npages; p++)
	{
		Buffer		buf;
		Page		page;
		DannPageOpaque opaque;
		GenericXLogState *gxlog;
		char	   *base;
		int			cnt;
		int			i;

		buf = ReadBuffer(bs->index, 1 + p);
		LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
		gxlog = GenericXLogStart(bs->index);
		page = GenericXLogRegisterBuffer(gxlog, buf, GENERIC_XLOG_FULL_IMAGE);
		dann_init_page_image(page, DANN_PAGE_DIRECTORY);
		opaque = (DannPageOpaque) PageGetSpecialPointer(page);

		cnt = Min(DANN_DIR_PER_PAGE, bs->nlist - (int) p * DANN_DIR_PER_PAGE);
		opaque->count = cnt;
		base = (char *) PageGetContents(page);

		for (i = 0; i < cnt; i++)
		{
			uint32		listno = p * DANN_DIR_PER_PAGE + i;
			DannListEntry le;

			le.first = (bs->list_nentries[listno] == 0)
				? InvalidBlockNumber : bs->list_first[listno];
			le.last = bs->list_cur[listno];		/* 空 list:Invalid */
			le.count = bs->list_nentries[listno];
			le.reserved = 0;

			memcpy(base + (size_t) i * sizeof(DannListEntry), &le,
				   sizeof(DannListEntry));
		}
		dann_page_set_lower(page, (uint32) cnt * sizeof(DannListEntry));

		GenericXLogFinish(gxlog);
		UnlockReleaseBuffer(buf);
	}
}

/* 写 metapage(块 0);W9:generic_xlog 全页镜像 */
static void
dann_write_metapage_build(DannBuildState *bs)
{
	Buffer		buf;
	Page		page;
	GenericXLogState *gxlog;
	DannMetaPageData meta;

	buf = ReadBufferExtended(bs->index, MAIN_FORKNUM, P_NEW, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	gxlog = GenericXLogStart(bs->index);
	page = GenericXLogRegisterBuffer(gxlog, buf, GENERIC_XLOG_FULL_IMAGE);
	dann_init_page_image(page, DANN_PAGE_META);

	meta.magic = DANN_MAGIC;
	meta.version = DANN_VERSION;
	meta.dim = bs->dim;
	meta.padded_dim = bs->padded_dim;
	meta.nlist = bs->nlist;
	meta.quantizer = bs->quantizer;
	meta.metric = bs->metric;
	meta.rotation_seed = bs->rotation_seed;
	meta.dir_start = 1;
	meta.dir_npages = bs->dir_npages;
	meta.centroid_start = 1 + bs->dir_npages;
	meta.centroid_npages = bs->centroid_npages;
	meta.posting_start = bs->posting_start;
	meta.entry_size = bs->entry_size;
	meta.ntuples = bs->reltuples;

	memcpy(PageGetContents(page), &meta, sizeof(DannMetaPageData));
	dann_page_set_lower(page, sizeof(DannMetaPageData));
	GenericXLogFinish(gxlog);
	UnlockReleaseBuffer(buf);
}

IndexBuildResult *
dann_build(Relation heap, Relation index, IndexInfo *indexInfo)
{
	IndexBuildResult *result;
	DannBuildState bs;
	DannOptions *opts = (DannOptions *) index->rd_options;
	int			i;

	memset(&bs, 0, sizeof(bs));
	bs.index = index;
	bs.rng = 0x243F6A8885A308D3ULL;

	/* dim 与度量 */
	bs.dim = TupleDescAttr(index->rd_att, 0)->atttypmod;
	bs.metric = dann_metric_of(index);

	/* 量化器解析:auto 按维度自适应(DESIGN.md §4.2:dim<128 → SQ8) */
	bs.quantizer = dann_quantizer_from_options(index);
	if (bs.quantizer == DANN_QUANTIZER_AUTO)
		bs.quantizer = (bs.dim < DANN_RABITQ_MIN_DIM)
			? DANN_QUANTIZER_SQ8 : DANN_QUANTIZER_RABITQ;

	bs.nlist = (opts != NULL) ? opts->nlist : DANN_DEFAULT_NLIST;
	bs.padded_dim = dann_padded_dim((uint32) bs.dim);

	/* 旋转种子由索引几何派生:相同 (dim, nlist) 构建可复现;
	 * 种子存 metapage,查询/插入端据此重建同一旋转 */
	{
		uint32		st = 0xDA11DA11u ^ (uint32) bs.dim ^ ((uint32) bs.nlist << 16);

		bs.rotation_seed = dann_splitmix32(&st);
	}

	bs.entry_size = dann_entry_size(bs.quantizer, (uint32) bs.dim, bs.padded_dim);
	bs.entries_per_page = dann_entries_per_page(bs.entry_size);
	bs.centroids_per_page = (int) (DANN_PAGE_PAYLOAD / ((uint32) bs.dim * sizeof(float)));

	/* ---- Pass A:采样 ---- */
	bs.sample_cap = 64 * bs.nlist;
	if (bs.sample_cap > 65536)
		bs.sample_cap = 65536;
	{
		/* 内存封顶 64MB */
		uint64		memcap = (uint64) (64 * 1024 * 1024)
			/ ((uint32) bs.dim * sizeof(float));

		if ((uint64) bs.sample_cap > memcap)
			bs.sample_cap = (int) memcap;
	}
	bs.samples = (float *) palloc(sizeof(float) * (size_t) bs.sample_cap * bs.dim);
	bs.nsamples = 0;
	bs.reltuples = 0;

	bs.reltuples = table_index_build_scan(heap, index, indexInfo,
										  false, false,
										  dann_sample_callback,
										  (void *) &bs, NULL);

	if (bs.reltuples < 1.0)
	{
		/* 空表:只写 nlist=0 的 metapage;后续插入报错提示 REINDEX */
		bs.nlist = 0;
		bs.posting_start = 1;
		dann_write_metapage_build(&bs);
		dann_build_sync_if_needed(index);
		result = (IndexBuildResult *) palloc0(sizeof(IndexBuildResult));
		result->heap_tuples = 0;
		result->index_tuples = 0;
		return result;
	}

	/* nlist 夹子:不超过样本数 */
	if (bs.nlist > bs.nsamples)
	{
		ereport(WARNING,
				(errmsg("pg_dann: nlist %d clamped to %d (sample size)",
						bs.nlist, bs.nsamples)));
		bs.nlist = bs.nsamples;
	}

	/* ---- k-means ---- */
	bs.centroids = (float *) palloc(sizeof(float) * (size_t) bs.nlist * bs.dim);
	dann_kmeans(bs.samples, bs.nsamples, bs.dim, bs.nlist,
				DANN_KMEANS_ITERS, bs.centroids);

	/* ---- Pass B:分配 + 计数 ---- */
	bs.assignments = (uint16 *) palloc(sizeof(uint16) * (size_t) bs.reltuples);
	bs.counts = (uint32 *) palloc0(sizeof(uint32) * bs.nlist);
	bs.rowno = 0;

	table_index_build_scan(heap, index, indexInfo, false, false,
						   dann_count_callback, (void *) &bs, NULL);

	/* ---- 布局:meta(1) + dir + centroid + posting 连续段 ---- */
	bs.dir_npages = ((uint32) bs.nlist + DANN_DIR_PER_PAGE - 1) / DANN_DIR_PER_PAGE;
	bs.centroid_npages = ((uint32) bs.nlist + bs.centroids_per_page - 1)
		/ bs.centroids_per_page;
	bs.posting_start = 1 + bs.dir_npages + bs.centroid_npages;
	{
		BlockNumber cur = bs.posting_start;

		bs.list_first = palloc0(sizeof(BlockNumber) * bs.nlist);
		bs.list_nentries = palloc0(sizeof(uint32) * bs.nlist);
		for (i = 0; i < bs.nlist; i++)
		{
			uint32		npages;

			bs.list_nentries[i] = bs.counts[i];
			npages = (bs.counts[i] + bs.entries_per_page - 1)
				/ bs.entries_per_page;
			if (npages == 0)
				bs.list_first[i] = InvalidBlockNumber;
			else
			{
				bs.list_first[i] = cur;
				cur += npages;
			}
		}
		bs.total_pages = cur;
	}

	/* metapage(块 0)先行,再预扩展其余 */
	dann_write_metapage_build(&bs);
	dann_extend_relation(index, bs.total_pages);

	/* 质心页 */
	dann_write_centroids(&bs);

	/* ---- Pass C:编码写入 ---- */
	if (bs.quantizer == DANN_QUANTIZER_RABITQ)
	{
		DannMetaPageData mtmp;	/* rqctx 只需 padded_dim 与 rotation_seed */

		mtmp.padded_dim = bs.padded_dim;
		mtmp.rotation_seed = bs.rotation_seed;
		dann_rqctx_init(&bs.rq, &mtmp);
	}

	bs.listbufs = (char *) palloc0((size_t) bs.nlist * BLCKSZ);
	bs.list_cur = palloc(sizeof(BlockNumber) * bs.nlist);
	for (i = 0; i < bs.nlist; i++)
		bs.list_cur[i] = InvalidBlockNumber;
	bs.rowno = 0;

	table_index_build_scan(heap, index, indexInfo, false, false,
						   dann_write_callback, (void *) &bs, NULL);

	/* 冲刷各 list 尾部驻留页 */
	for (i = 0; i < bs.nlist; i++)
		if (bs.list_cur[i] != InvalidBlockNumber)
			dann_flush_list_page(&bs, i);

	/* 两遍一致性校验(counts 已倒计数至 0) */
	for (i = 0; i < bs.nlist; i++)
		if (bs.counts[i] != 0)
			elog(ERROR, "pg_dann: build pass mismatch for list %d", i);

	/* directory 页 */
	dann_write_directory(&bs);

	dann_build_sync_if_needed(index);

	result = (IndexBuildResult *) palloc0(sizeof(IndexBuildResult));
	result->heap_tuples = bs.reltuples;
	result->index_tuples = bs.reltuples;
	return result;
}

void
dann_buildempty(Relation index)
{
	/* 未记录表的 INIT fork:nlist=0 的最小 metapage */
	Buffer		buf;
	Page		page;
	DannMetaPageData *meta;

	buf = ReadBufferExtended(index, INIT_FORKNUM, P_NEW, RBM_NORMAL, NULL);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	page = BufferGetPage(buf);
	dann_init_page(buf, DANN_PAGE_META);
	meta = (DannMetaPageData *) PageGetContents(page);
	meta->magic = DANN_MAGIC;
	meta->version = DANN_VERSION;
	meta->dim = 0;
	meta->padded_dim = 0;
	meta->nlist = 0;
	meta->quantizer = DANN_QUANTIZER_SQ8;
	meta->metric = DANN_METRIC_L2;
	meta->rotation_seed = 0;
	meta->dir_start = 1;
	meta->dir_npages = 0;
	meta->centroid_start = 1;
	meta->centroid_npages = 0;
	meta->posting_start = 1;
	meta->entry_size = 0;
	meta->ntuples = 0;
	MarkBufferDirty(buf);
	LockBuffer(buf, BUFFER_LOCK_UNLOCK);
	ReleaseBuffer(buf);
}
