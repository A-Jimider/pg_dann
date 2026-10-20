/*-------------------------------------------------------------------------
 * scan.c — 查询路径(DESIGN.md §4.4)
 *
 *   ① 质心扫描:全扫 centroid 页,SIMD 选 dann.probes 个最近 list
 *   ② 批量预取(W6):按 directory 连续页区间合并后 PrefetchBuffer
 *   ③ 压缩码打分:SQ8 估计距离,收集全部候选后按估计距离排序
 *   ④ heap rerank(W6):估计序候选分块(γ×10)、ctid 排序批量回表、
 *      精确距离块内重排;dann.enable_rerank=off 时直接按估计序返回
 *
 * amgettuple 是拉取式、不知道 LIMIT k,故收集 probes 个 list 的
 * 全部候选排序后顺序返回(与 pgvector IVFFlat 同构);rerank 以
 * γ×10 为块懒加载,返回 k 条时回表约 γk 个候选。
 *-------------------------------------------------------------------------*/
#include "postgres.h"

#include <math.h>

#include "access/htup_details.h"
#include "access/relscan.h"
#include "access/tableam.h"
#include "executor/tuptable.h"
#include "funcapi.h"
#include "portability/instr_time.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "utils/rel.h"

#include "dann.h"
#include "quant.h"

/* ---- W6 会话级扫描统计(dann_scan_stats();W11 扩展更多维度) ---- */
static uint64 dann_stats_scans = 0;		/* orderby 扫描次数 */
static uint64 dann_stats_truncated = 0; /* 发生预算截断的扫描次数 */
static uint64 dann_stats_lists_skipped = 0; /* 因截断跳过的 list 总数 */

typedef struct DannCandidate
{
	float		est;
	ItemPointerData ctid;
} DannCandidate;

typedef struct DannListScore
{
	float		dist;
	uint32		listno;
} DannListScore;

/* W6 批量预取:待预取的连续块号区间 */
typedef struct DannPrefetchRange
{
	BlockNumber first;
	BlockNumber end;			/* 开区间 */
} DannPrefetchRange;

typedef struct DannScanOpaqueData
{
	Relation	index;
	DannMetaPageData meta;		/* rescan 时快照 */
	Datum		query_datum;	/* 原始查询向量 Datum */
	bool		scanned;		/*  posting 扫描是否已执行 */
	DannCandidate *cands;
	int			ncands;
	int			candcap;
	int			pos;

	/* ---- W6 heap rerank 状态 ---- */
	bool		rerank;			/* 本扫描是否启用精确重排 */
	float	   *q;				/* 归一化后的查询向量(精确距离用) */
	int			done;			/* cands 中已送入 rerank 的条数 */
	DannCandidate *rbuf;		/* 精确序缓冲(est 复用为精确距离) */
	int			rcap;
	int			rcount;
	int			rpos;
	TupleTableSlot *rslot;		/* 回表槽(scan 生命周期) */
	IndexFetchTableData *hfetch;
	int			heap_attno;

	/* ---- W7 RaBitQ:旋转上下文(rq_init 时才有效) ---- */
	bool		rq_init;
	DannRqCtx	rq;
} DannScanOpaqueData;

/* RaBitQ 每个 list 的查询侧预处理结果(qr 在 opaque 的 ctx 里) */
typedef struct DannListQuery
{
	float		qc_norm;		/* ||q - c|| */
	float		sum_q;			/* Σ qr[i] */
} DannListQuery;

typedef DannScanOpaqueData *DannScanOpaque;

IndexScanDesc
dann_beginscan(Relation index, int nkeys, int norderbys)
{
	IndexScanDesc scan;
	DannScanOpaque so;

	scan = RelationGetIndexScan(index, nkeys, norderbys);
	so = (DannScanOpaque) palloc0(sizeof(DannScanOpaqueData));
	so->index = index;
	scan->opaque = so;

	return scan;
}

void
dann_rescan(IndexScanDesc scan, ScanKey keys, int nkeys,
			ScanKey orderbys, int norderbys)
{
	DannScanOpaque so = (DannScanOpaque) scan->opaque;

	/* 保持 scan key 同步 */
	if (keys != NULL && scan->numberOfKeys > 0)
		memmove(scan->keyData, keys,
				scan->numberOfKeys * sizeof(ScanKeyData));
	if (orderbys != NULL && scan->numberOfOrderBys > 0)
		memmove(scan->orderByData, orderbys,
				scan->numberOfOrderBys * sizeof(ScanKeyData));

	/* 快照 metapage;标记待扫描(查询向量推迟到首次执行再解析) */
	dann_get_metapage(so->index, &so->meta);
	so->scanned = false;
	so->pos = 0;
	so->ncands = 0;
	if (so->cands != NULL)
	{
		pfree(so->cands);
		so->cands = NULL;
		so->candcap = 0;
	}

	/* W6 rerank 状态复位(rslot/hfetch 随 scan 生命周期保留复用) */
	so->rerank = false;
	so->done = 0;
	so->rcount = 0;
	so->rpos = 0;
	if (so->q != NULL)
	{
		pfree(so->q);
		so->q = NULL;
	}
	if (so->rbuf != NULL)
	{
		pfree(so->rbuf);
		so->rbuf = NULL;
		so->rcap = 0;
	}
	if (so->rslot != NULL)
		ExecClearTuple(so->rslot);
	if (so->rq_init)
	{
		dann_rqctx_free(&so->rq);
		so->rq_init = false;
	}
}

static int
dann_cmp_listscore(const void *a, const void *b)
{
	float		da = ((const DannListScore *) a)->dist;
	float		db = ((const DannListScore *) b)->dist;

	if (da < db)
		return -1;
	if (da > db)
		return 1;
	return 0;
}

static int
dann_cmp_candidate(const void *a, const void *b)
{
	float		da = ((const DannCandidate *) a)->est;
	float		db = ((const DannCandidate *) b)->est;

	if (da < db)
		return -1;
	if (da > db)
		return 1;
	return 0;
}

static int
dann_cmp_prefetch_range(const void *a, const void *b)
{
	BlockNumber fa = ((const DannPrefetchRange *) a)->first;
	BlockNumber fb = ((const DannPrefetchRange *) b)->first;

	if (fa < fb)
		return -1;
	if (fa > fb)
		return 1;
	return 0;
}

/*
 * W6 批量预取(DESIGN.md §4.4-2):把选中 list 的页区间合并后批量发起。
 *
 * 构建期各 list 页段物理连续(§4.1-D2),故 [first, first+npages) 必属
 * 本 list;npages 由 count/epp 下界估计,增量插入的离散追加页不在区间内
 * 也无妨——读取仍沿链走,预取只是提示。PG16 上统一走 PrefetchBuffer
 * (底层 posix_fadvise(WILLNEED));PG18 的 read stream 留作后续(#if
 * PG_VERSION_NUM >= 180000 时替换本函数实现)。
 */
static void
dann_prefetch_lists(Relation index, const DannMetaPageData *meta,
					const DannListEntry *entries, int nprobe)
{
	DannPrefetchRange *ranges;
	int			nranges = 0;
	int			epp;
	BlockNumber nblocks;
	int			i;
	BlockNumber cur_first;
	BlockNumber cur_end;
	BlockNumber b;

	if (dann_io_method == DANN_IO_SYNC)
		return;

	epp = dann_entries_per_page(meta->entry_size);
	ranges = (DannPrefetchRange *) palloc(sizeof(DannPrefetchRange) * nprobe);

	for (i = 0; i < nprobe; i++)
	{
		uint32		npages;

		if (entries[i].first == InvalidBlockNumber || entries[i].count == 0)
			continue;
		npages = (entries[i].count + (uint32) epp - 1) / (uint32) epp;
		ranges[nranges].first = entries[i].first;
		ranges[nranges].end = entries[i].first + npages;
		nranges++;
	}
	if (nranges == 0)
	{
		pfree(ranges);
		return;
	}

	/* 按块号排序后合并重叠/相邻区间,预取也按块号序(对 fadvise 更友好) */
	qsort(ranges, nranges, sizeof(DannPrefetchRange), dann_cmp_prefetch_range);
	nblocks = RelationGetNumberOfBlocksInFork(index, MAIN_FORKNUM);

	cur_first = ranges[0].first;
	cur_end = ranges[0].end;
	for (i = 1; i <= nranges; i++)	/* <=:最后一次迭代冲刷收尾区间 */
	{
		if (i < nranges && ranges[i].first <= cur_end)
		{
			if (ranges[i].end > cur_end)
				cur_end = ranges[i].end;
			continue;
		}
		if (cur_end > nblocks)		/* 防御:count 估计不许越出关系末尾 */
			cur_end = nblocks;
		for (b = cur_first; b < cur_end; b++)
			PrefetchBuffer(index, MAIN_FORKNUM, b);
		if (i < nranges)
		{
			cur_first = ranges[i].first;
			cur_end = ranges[i].end;
		}
	}
	pfree(ranges);
}

static void
dann_push_candidate(DannScanOpaque so, float est, ItemPointer ctid)
{
	if (so->ncands >= so->candcap)
	{
		int			newcap = so->candcap == 0 ? 1024 : so->candcap * 2;

		if (so->cands == NULL)
			so->cands = palloc(sizeof(DannCandidate) * newcap);
		else
			so->cands = repalloc(so->cands, sizeof(DannCandidate) * newcap);
		so->candcap = newcap;
	}
	so->cands[so->ncands].est = est;
	so->cands[so->ncands].ctid = *ctid;
	so->ncands++;
}

/* 单个 posting 页的条目扫描打分 */
static void
dann_score_page(DannScanOpaque so, Page page, const float *q, float qnorm,
				const DannListQuery *lq)
{
	DannPageOpaque opaque = (DannPageOpaque) PageGetSpecialPointer(page);
	char	   *base = (char *) PageGetContents(page);
	uint32		cnt = opaque->count;
	uint32		esize = so->meta.entry_size;
	uint32		dim = so->meta.dim;
	uint32		i;

	for (i = 0; i < cnt; i++)
	{
		char	   *entry = base + (size_t) i * esize;
		const uint8_t *code = (const uint8_t *) (entry + DANN_ENTRY_HEADER_SIZE);
		float		est;

		if (q == NULL)
			est = 0.0f;		/* 无 ORDER BY 的全扫描:距离无意义 */
		else if (so->meta.quantizer == DANN_QUANTIZER_RABITQ)
		{
			const DannRabitqFactors *f = (const DannRabitqFactors *) (entry + 8);

			if (so->meta.metric == DANN_METRIC_L2)
				est = dann_rabitq_estimate_l2(code, f, so->rq.qr, lq->sum_q,
											  lq->qc_norm,
											  (int) so->meta.padded_dim);
			else
				/* IP/cosine:直接用 coŝ(DESIGN.md §4.2),作距离取负 */
				est = -dann_rabitq_estimate_cos(code, f, so->rq.qr, lq->sum_q,
												(int) so->meta.padded_dim);
		}
		else
		{
			const DannSQ8Factors *f = (const DannSQ8Factors *) (entry + 8);

			if (so->meta.metric == DANN_METRIC_IP)
				est = -dann_sq8_dot(code, f, q, dim);
			else
				/* L2 与 cosine(向量已归一化)共用 L2 估计式 */
				est = dann_sq8_estimate_l2(code, f, q, qnorm, dim);
		}

		dann_push_candidate(so, est, (ItemPointer) entry);
	}
}

/* 扫描一个 list 的整条 posting 链 */
static void
dann_scan_list(DannScanOpaque so, const DannListEntry *le,
			   const float *q, float qnorm, const DannListQuery *lq)
{
	BlockNumber blkno = le->first;

	while (blkno != InvalidBlockNumber)
	{
		Buffer		buf = ReadBuffer(so->index, blkno);
		Page		page;
		DannPageOpaque opaque;
		BlockNumber next;

		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = (DannPageOpaque) PageGetSpecialPointer(page);
		next = opaque->next;
		dann_score_page(so, page, q, qnorm, lq);
		LockBuffer(buf, BUFFER_LOCK_UNLOCK);
		ReleaseBuffer(buf);

		blkno = next;
	}
}

static int
dann_cmp_itemptr(const void *a, const void *b)
{
	return ItemPointerCompare((ItemPointer) a, (ItemPointer) b);
}

/* 全精度精确距离(W6 rerank);heap 中存原始向量 */
static float
dann_exact_distance(DannScanOpaque so, const float *x)
{
	const DannMetaPageData *meta = &so->meta;

	if (meta->metric == DANN_METRIC_IP)
		return -dann_dot_product(so->q, x, (int) meta->dim);
	if (meta->metric == DANN_METRIC_COSINE)
	{
		/* q 已归一化:cosine = 1 - <q,x̂>,与 pgvector cosine_distance 同式 */
		double		dot = dann_dot_product(so->q, x, (int) meta->dim);
		double		xx = dann_dot_product(x, x, (int) meta->dim);

		if (xx <= 0.0)
			return 1.0f;		/* 零向量进不了 cosine 索引(构建/插入报错) */
		return (float) (1.0 - dot / sqrt(xx));
	}
	return dann_l2_distance(so->q, x, (int) meta->dim);
}

/*
 * rerank 缓冲补货(DESIGN.md §4.4-4):
 *   取估计序下一块候选(γ×10 条——amgettuple 拉取式不知道 LIMIT k,
 *   按典型 k=10 起跳,耗尽再补),ctid 排序后批量回表(随机读压成近似
 *   顺序读),可见元组算精确距离,块内按精确序排列。死元组跳过。
 */
static void
dann_rerank_refill(IndexScanDesc scan)
{
	DannScanOpaque so = (DannScanOpaque) scan->opaque;
	const DannMetaPageData *meta = &so->meta;
	int			n;
	int			i;
	ItemPointer tids;

	n = Min(dann_rerank_factor_of(so->index) * 10, so->ncands - so->done);
	Assert(n > 0);

	if (so->rslot == NULL)
	{
		/* 懒初始化:heapRelation 由 executor 的 index_beginscan 设置 */
		so->heap_attno = so->index->rd_index->indkey.values[0];
		if (so->heap_attno <= 0)
			ereport(ERROR,
					(errcode(ERRCODE_FEATURE_NOT_SUPPORTED),
					 errmsg("pg_dann: cannot rerank an index on expressions")));
		so->rslot = MakeSingleTupleTableSlot(RelationGetDescr(scan->heapRelation),
											 &TTSOpsBufferHeapTuple);
		so->hfetch = table_index_fetch_begin(scan->heapRelation);
	}

	if (n > so->rcap)
	{
		if (so->rbuf == NULL)
			so->rbuf = (DannCandidate *) palloc(sizeof(DannCandidate) * n);
		else
			so->rbuf = (DannCandidate *) repalloc(so->rbuf, sizeof(DannCandidate) * n);
		so->rcap = n;
	}

	tids = (ItemPointer) palloc(sizeof(ItemPointerData) * n);
	for (i = 0; i < n; i++)
		tids[i] = so->cands[so->done + i].ctid;
	so->done += n;
	qsort(tids, n, sizeof(ItemPointerData), dann_cmp_itemptr);

	so->rcount = 0;
	so->rpos = 0;
	for (i = 0; i < n; i++)
	{
		bool		call_again = false;
		bool		all_dead = false;
		bool		found = false;
		bool		isnull;
		Datum		d;
		DannVector *v;
		float		dist;

		ExecClearTuple(so->rslot);
		do
		{
			if (table_index_fetch_tuple(so->hfetch, &tids[i],
										scan->xs_snapshot, so->rslot,
										&call_again, &all_dead))
			{
				found = true;	/* 第一个可见版本即所求(MVCC 快照唯一) */
				break;
			}
		} while (call_again);
		if (!found)
			continue;			/* 死元组:跳过 */

		d = slot_getattr(so->rslot, so->heap_attno, &isnull);
		if (isnull)
			continue;
		v = dann_get_vector(d, meta->dim);
		dist = dann_exact_distance(so, v->x);
		dann_free_vector(d, v);

		so->rbuf[so->rcount].est = dist;
		so->rbuf[so->rcount].ctid = tids[i];
		so->rcount++;
	}
	pfree(tids);

	if (so->rcount > 1)
		qsort(so->rbuf, so->rcount, sizeof(DannCandidate), dann_cmp_candidate);
}

/* 首次 gettuple:完成质心选择 + posting 扫描 + 排序 */
static void
dann_exec_scan(IndexScanDesc scan)
{
	DannScanOpaque so = (DannScanOpaque) scan->opaque;
	DannMetaPageData *meta = &so->meta;
	DannVector *qv;
	float	   *q;
	float		qnorm = 0.0f;
	int			nprobe;
	uint32		i;
	uint32		p;
	DannListScore *scores;
	DannListEntry *entries;
	float	   *prob_c = NULL;	/* RaBitQ:probed list 的质心拷贝 */
	instr_time	scan_start;

	if (meta->nlist == 0)
	{
		so->scanned = true;
		so->rerank = false;
		return;
	}

	/* 解析查询向量;无 ORDER BY 时退化为全索引扫描(如 count(*));
	 * ORDER BY 参数为 NULL 时同样退化:距离恒 NULL,序无定义,
	 * 行集合正确即可(与 seqscan+sort 的未定序语义一致) */
	if (scan->numberOfOrderBys == 0 ||
		(scan->orderByData[0].sk_flags & SK_ISNULL) != 0)
	{
		uint32		listno;

		for (listno = 0; listno < meta->nlist; listno++)
		{
			DannListEntry le;

			dann_get_list_entry(so->index, meta, listno, &le);
			if (le.first != InvalidBlockNumber)
				dann_scan_list(so, &le, NULL, 0.0f, NULL);
		}
		so->scanned = true;
		so->pos = 0;
		so->rerank = false;
		return;
	}
	qv = dann_get_vector(scan->orderByData[0].sk_argument, meta->dim);
	q = (float *) palloc(sizeof(float) * meta->dim);
	memcpy(q, qv->x, sizeof(float) * meta->dim);
	dann_check_finite(q, (int) meta->dim);
	if (meta->metric == DANN_METRIC_COSINE)
		dann_normalize(q, meta->dim);
	dann_free_vector(scan->orderByData[0].sk_argument, qv);
	/* rerank 跨 gettuple 调用存活;q 挂到 opaque,rescan/endscan 释放 */
	so->q = q;

	if (meta->metric != DANN_METRIC_IP)
	{
		for (i = 0; i < meta->dim; i++)
			qnorm += q[i] * q[i];
	}

	/* ① 质心扫描:全量打分 */
	scores = (DannListScore *) palloc(sizeof(DannListScore) * meta->nlist);
	{
		uint32		listno = 0;

		for (p = 0; p < meta->centroid_npages; p++)
		{
			Buffer		buf = ReadBuffer(so->index, meta->centroid_start + p);
			Page		page;
			DannPageOpaque opaque;
			uint32		cnt;
			char	   *base;

			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			opaque = (DannPageOpaque) PageGetSpecialPointer(page);
			cnt = opaque->count;
			base = (char *) PageGetContents(page);

			for (i = 0; i < cnt && listno < meta->nlist; i++)
			{
				const float *cen = (const float *)
					(base + (size_t) i * meta->dim * sizeof(float));
				float		d;

				if (meta->metric == DANN_METRIC_IP)
					d = -dann_dot_product(q, cen, meta->dim);
				else
					d = dann_l2_distance(q, cen, meta->dim);

				scores[listno].dist = d;
				scores[listno].listno = listno;
				listno++;
			}

			LockBuffer(buf, BUFFER_LOCK_UNLOCK);
			ReleaseBuffer(buf);
		}
	}

	/* 选 probes 个最近 list,按质心距离升序(供延迟预算截断) */
	nprobe = Min(dann_probes_of(so->index), (int) meta->nlist);
	qsort(scores, meta->nlist, sizeof(DannListScore), dann_cmp_listscore);

	/* directory 条目一次取齐,预取与扫描共用 */
	entries = (DannListEntry *) palloc(sizeof(DannListEntry) * nprobe);
	for (i = 0; i < (uint32) nprobe; i++)
		dann_get_list_entry(so->index, meta, scores[i].listno, &entries[i]);

	/* RaBitQ:旋转上下文(按种子重建)+ probed list 的质心拷贝 */
	if (meta->quantizer == DANN_QUANTIZER_RABITQ)
	{
		dann_rqctx_init(&so->rq, meta);
		so->rq_init = true;
		prob_c = (float *) palloc(sizeof(float) * (size_t) nprobe * meta->dim);
		for (i = 0; i < (uint32) nprobe; i++)
			dann_get_centroid(so->index, meta, scores[i].listno,
							  prob_c + (size_t) i * meta->dim);
	}

	/* ② 批量预取(W6):io_method=sync 时函数内直接返回 */
	dann_prefetch_lists(so->index, meta, entries, nprobe);

	/*
	 * ③ 逐 list 扫描打分(W6 延迟预算截断:每处理完一个 list 检查
	 * deadline,耗尽即停——list 按质心距离升序,截断只丢尾部最远的
	 * list,recall 平滑降级而非查询失败;截断计入 dann_scan_stats)。
	 */
	if (dann_max_latency_ms > 0)
		INSTR_TIME_SET_CURRENT(scan_start);
	for (i = 0; i < (uint32) nprobe; i++)
	{
		if (entries[i].first != InvalidBlockNumber)
		{
			DannListQuery lqbuf;
			const DannListQuery *lq = NULL;

			if (so->rq_init)
			{
				/* 每 list 一次:P(q−c)/||q−c‖,O(d log d) */
				lqbuf.qc_norm = dann_rabitq_prepare_query(q,
											prob_c + (size_t) i * meta->dim,
											(int) meta->dim,
											(int) meta->padded_dim,
											so->rq.sign, so->rq.perm,
											so->rq.scratch, so->rq.qr,
											&lqbuf.sum_q);
				lq = &lqbuf;
			}
			dann_scan_list(so, &entries[i], q, qnorm, lq);
		}
		if (dann_max_latency_ms > 0 && i + 1 < (uint32) nprobe)
		{
			instr_time now;
			double		elapsed_ms;

			INSTR_TIME_SET_CURRENT(now);
			INSTR_TIME_SUBTRACT(now, scan_start);
			elapsed_ms = INSTR_TIME_GET_DOUBLE(now) * 1000.0;
			if (elapsed_ms >= (double) dann_max_latency_ms)
			{
				dann_stats_truncated++;
				dann_stats_lists_skipped += (uint64) (nprobe - 1 - i);
				elog(DEBUG1, "pg_dann: latency budget exhausted after "
					 "%d/%d lists", i + 1, nprobe);
				break;
			}
		}
	}
	dann_stats_scans++;

	if (prob_c != NULL)
		pfree(prob_c);
	pfree(entries);

	/* 估计距离升序 */
	if (so->ncands > 1)
		qsort(so->cands, so->ncands, sizeof(DannCandidate), dann_cmp_candidate);

	/* W6:rerank 状态初始化(缓冲在首次补货时按块大小分配) */
	so->rerank = dann_enable_rerank;
	so->done = 0;
	so->rcount = 0;
	so->rpos = 0;

	pfree(scores);
	so->scanned = true;
	so->pos = 0;
}

bool
dann_gettuple(IndexScanDesc scan, ScanDirection dir)
{
	DannScanOpaque so = (DannScanOpaque) scan->opaque;
	DannCandidate *cand;

	if (!so->scanned)
		dann_exec_scan(scan);

	if (so->rerank)
	{
		/* W6:按精确序发射;缓冲耗尽则补下一块(整块死元组则继续补) */
		while (so->rpos >= so->rcount)
		{
			if (so->done >= so->ncands)
				return false;
			dann_rerank_refill(scan);
		}
		cand = &so->rbuf[so->rpos++];
	}
	else
	{
		if (so->pos >= so->ncands)
			return false;
		cand = &so->cands[so->pos++];
	}

	scan->xs_heaptid = cand->ctid;
	scan->xs_recheck = false;
	/*
	 * xs_recheckorderby 恒为 false:我们内部已完成估计→(W6 精确)排序,
	 * 不需要执行器重算;且按 PG 惯例 xs_orderbyvals 由 AM 自分配
	 * (gist/spgist 在 amrescan 里 palloc)——不设置它,reorder queue
	 * 恒空,执行器不会解引用(pgvector 同样不设置)。
	 */
	scan->xs_recheckorderby = false;
	return true;
}

void
dann_endscan(IndexScanDesc scan)
{
	DannScanOpaque so = (DannScanOpaque) scan->opaque;

	if (so != NULL)
	{
		if (so->rq_init)
			dann_rqctx_free(&so->rq);
		if (so->rslot != NULL)
			ExecDropSingleTupleTableSlot(so->rslot);
		if (so->hfetch != NULL)
			table_index_fetch_end(so->hfetch);
		if (so->q != NULL)
			pfree(so->q);
		if (so->rbuf != NULL)
			pfree(so->rbuf);
		if (so->cands != NULL)
			pfree(so->cands);
		pfree(so);
	}
}

/* ---- 会话级统计 SQL 接口 ---- */

PG_FUNCTION_INFO_V1(dann_scan_stats);
PG_FUNCTION_INFO_V1(dann_scan_stats_reset);

Datum
dann_scan_stats(PG_FUNCTION_ARGS)
{
	TupleDesc	tupdesc;
	Datum		values[3];
	bool		nulls[3] = {false, false, false};
	HeapTuple	tuple;

	if (get_call_result_type(fcinfo, NULL, &tupdesc) != TYPEFUNC_COMPOSITE)
		elog(ERROR, "pg_dann: dann_scan_stats return type must be composite");

	values[0] = Int64GetDatum((int64) dann_stats_scans);
	values[1] = Int64GetDatum((int64) dann_stats_truncated);
	values[2] = Int64GetDatum((int64) dann_stats_lists_skipped);
	tuple = heap_form_tuple(tupdesc, values, nulls);
	PG_RETURN_DATUM(HeapTupleGetDatum(tuple));
}

Datum
dann_scan_stats_reset(PG_FUNCTION_ARGS)
{
	dann_stats_scans = 0;
	dann_stats_truncated = 0;
	dann_stats_lists_skipped = 0;
	PG_RETURN_VOID();
}
