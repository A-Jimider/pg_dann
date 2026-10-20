/*-------------------------------------------------------------------------
 * insert.c — aminsert:单条插入(list 尾页追加)
 *
 * 流程:读 metapage → 归一化(cosine)→ 最近质心 → 编码 → 追加到
 * list 尾页;尾页满则扩新页并链接。无页面分裂/指针修复,比 HNSW
 * 插入简单一个量级(DESIGN.md §4.5)。
 *
 * W4-5 版本:无 WAL(W9);并发安全依赖 LWLock(W10),当前并发
 * 插入同 list 可能串页——W10 前仅单并发插入为正确使用方式。
 *-------------------------------------------------------------------------*/
#include "postgres.h"

#include <float.h>

#include "access/tupdesc.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "utils/rel.h"

#include "dann.h"
#include "quant.h"

uint32
dann_quantizer_from_options(Relation index)
{
	DannOptions *opts = (DannOptions *) index->rd_options;
	const char *s = "auto";

	if (opts != NULL && opts->quantizer > 0)
		s = (char *) opts + opts->quantizer;

	if (strcmp(s, "rabitq") == 0)
		return DANN_QUANTIZER_RABITQ;
	if (strcmp(s, "sq8") == 0)
		return DANN_QUANTIZER_SQ8;
	return DANN_QUANTIZER_AUTO;
}

/*
 * 最近质心:全扫 centroid 页(SIMD),返回 list 号。
 * 质心页热数据常驻 buffer pool,nlist×dim 次乘加,亚毫秒级。
 */
uint32
dann_nearest_centroid(Relation index, const DannMetaPageData *meta,
					  const float *v)
{
	uint32		best = 0;
	float		bestscore;
	uint32		listno = 0;
	uint32		p;

	if (meta->metric == DANN_METRIC_IP)
		bestscore = -FLT_MAX;
	else
		bestscore = FLT_MAX;

	for (p = 0; p < meta->centroid_npages; p++)
	{
		Buffer		buf = ReadBuffer(index, meta->centroid_start + p);
		Page		page;
		DannPageOpaque opaque;
		uint32		cnt;
		uint32		i;
		char	   *base;

		LockBuffer(buf, BUFFER_LOCK_SHARE);
		page = BufferGetPage(buf);
		opaque = (DannPageOpaque) PageGetSpecialPointer(page);
		cnt = opaque->count;
		base = (char *) PageGetContents(page);

		for (i = 0; i < cnt; i++)
		{
			const float *cen = (const float *) (base + (size_t) i * meta->dim * sizeof(float));
			float		score;
			bool		better;

			if (meta->metric == DANN_METRIC_IP)
			{
				score = dann_dot_product(v, cen, meta->dim);
				better = score > bestscore;
			}
			else
			{
				score = dann_l2_distance(v, cen, meta->dim);
				better = score < bestscore;
			}

			if (better)
			{
				bestscore = score;
				best = listno;
			}
			listno++;
		}

		LockBuffer(buf, BUFFER_LOCK_UNLOCK);
		ReleaseBuffer(buf);
	}

	return best;
}

/*
 * 追加主体:调用方须已持 metapage EXCLUSIVE 锁并读入最新 le;
 * posting 变更 + directory 更新纳入单个 GenericXLog 原子提交;
 * 所有登记的既有页都把 pd_lower 抬到全量内容末端(兼容 W9 前构建
 * 的页,防在线应用的 hole 清零误伤),新页用 FULL_IMAGE。
 */
static void
dann_insert_locked(Relation index, const DannMetaPageData *meta,
				   uint32 listno, DannListEntry *le, const char *entry,
				   int entry_size, int epp)
{
	GenericXLogState *state = GenericXLogStart(index);
	Buffer		buf = InvalidBuffer;
	Buffer		nbuf = InvalidBuffer;
	Buffer		dbuf;
	Page		page;
	DannPageOpaque opaque;

	if (le->first == InvalidBlockNumber)
	{
		/* 构建期为空的 list:末尾开新页(新页 + directory 原子) */
		Page		npage;
		DannPageOpaque nopaque;

		nbuf = ReadBufferExtended(index, MAIN_FORKNUM, P_NEW,
								  RBM_NORMAL, NULL);
		LockBuffer(nbuf, BUFFER_LOCK_EXCLUSIVE);
		npage = GenericXLogRegisterBuffer(state, nbuf,
										  GENERIC_XLOG_FULL_IMAGE);
		dann_init_page_image(npage, DANN_PAGE_POSTING);
		nopaque = (DannPageOpaque) PageGetSpecialPointer(npage);
		memcpy((char *) PageGetContents(npage), entry, entry_size);
		nopaque->count = 1;
		dann_page_set_lower(npage, (uint32) entry_size);

		le->first = le->last = BufferGetBlockNumber(nbuf);
		le->count = 1;
		dbuf = dann_set_list_entry_xlog(state, index, meta, listno, le);

		GenericXLogFinish(state);
		UnlockReleaseBuffer(nbuf);
		UnlockReleaseBuffer(dbuf);
		return;
	}

	/* 常规路径:尾页(必有) */
	buf = ReadBuffer(index, le->last);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	page = GenericXLogRegisterBuffer(state, buf, 0);
	opaque = (DannPageOpaque) PageGetSpecialPointer(page);

	if (opaque->count < (uint32) epp)
	{
		/* 尾页有空间:原地追加 */
		memcpy((char *) PageGetContents(page)
			   + (size_t) opaque->count * entry_size, entry, entry_size);
		opaque->count++;
	}
	else
	{
		/* 尾页满:扩新页挂链尾(段不再连续——布局退化,见 §4.5) */
		Page		npage;
		DannPageOpaque nopaque;

		nbuf = ReadBufferExtended(index, MAIN_FORKNUM, P_NEW,
								  RBM_NORMAL, NULL);
		LockBuffer(nbuf, BUFFER_LOCK_EXCLUSIVE);
		npage = GenericXLogRegisterBuffer(state, nbuf,
										  GENERIC_XLOG_FULL_IMAGE);
		dann_init_page_image(npage, DANN_PAGE_POSTING);
		nopaque = (DannPageOpaque) PageGetSpecialPointer(npage);
		memcpy((char *) PageGetContents(npage), entry, entry_size);
		nopaque->count = 1;
		dann_page_set_lower(npage, (uint32) entry_size);

		opaque->next = BufferGetBlockNumber(nbuf);
		le->last = BufferGetBlockNumber(nbuf);
	}
	/* 无论追加还是挂新页,旧页内容都要被 pd_lower 完整覆盖 */
	dann_page_set_lower(page, opaque->count * (uint32) entry_size);

	le->count++;
	dbuf = dann_set_list_entry_xlog(state, index, meta, listno, le);

	GenericXLogFinish(state);
	UnlockReleaseBuffer(buf);
	if (nbuf != InvalidBuffer)
		UnlockReleaseBuffer(nbuf);
	UnlockReleaseBuffer(dbuf);
}

bool
dann_insert(Relation index, Datum *values, bool *isnull, ItemPointer heap_tid,
			Relation heap, IndexUniqueCheck checkUnique, bool indexUnchanged,
			IndexInfo *indexInfo)
{
	DannMetaPageData meta;
	DannVector *v;
	float	   *ws;
	uint32		listno;
	DannListEntry le;
	int			entry_size;
	int			epp;
	char	   *entry;

	if (isnull[0])
		return false;

	dann_get_metapage(index, &meta);

	if (meta.nlist == 0)
		ereport(ERROR,
				(errcode(ERRCODE_OBJECT_NOT_IN_PREREQUISITE_STATE),
				 errmsg("pg_dann: index was built on an empty table; "
						"REINDEX after loading data")));

	v = dann_get_vector(values[0], meta.dim);
	ws = (float *) palloc(sizeof(float) * meta.dim);
	memcpy(ws, v->x, sizeof(float) * meta.dim);
	dann_check_finite(ws, (int) meta.dim);
	if (meta.metric == DANN_METRIC_COSINE)
		dann_normalize(ws, meta.dim);
	dann_free_vector(values[0], v);

	listno = dann_nearest_centroid(index, &meta, ws);

	entry_size = (int) meta.entry_size;
	epp = dann_entries_per_page(meta.entry_size);

	entry = (char *) palloc(entry_size);
	{
		uint16	   *pad = (uint16 *) (entry + 6);
		uint8_t    *code = (uint8_t *) (entry + DANN_ENTRY_HEADER_SIZE);

		memcpy(entry, heap_tid, sizeof(ItemPointerData));
		*pad = 0;
		if (meta.quantizer == DANN_QUANTIZER_SQ8)
		{
			DannSQ8Factors *f = (DannSQ8Factors *) (entry + 8);

			dann_sq8_encode(ws, meta.dim, code, f);
		}
		else
		{
			/* RaBitQ:取本 list 质心,相对其编码;旋转参数按种子重建
			 * (逐次插入重建参数为 O(padded_dim),量小;热了再缓存) */
			DannRabitqFactors *f = (DannRabitqFactors *) (entry + 8);
			DannRqCtx	rq;
			float	   *cen = (float *) palloc(sizeof(float) * meta.dim);

			dann_get_centroid(index, &meta, listno, cen);
			dann_rqctx_init(&rq, &meta);
			dann_rabitq_encode(ws, cen, (int) meta.dim, (int) meta.padded_dim,
							   rq.sign, rq.perm, rq.scratch,
							   code, &f->dist_c, &f->g);
			f->dist_c2 = f->dist_c * f->dist_c;
			dann_rqctx_free(&rq);
			pfree(cen);
		}
	}

	/* W10:metapage BUFFER_LOCK_EXCLUSIVE 串行化"读 directory → 尾页追加
	 * → 写 directory"(named LWLock tranche 需 shared_preload_libraries;
	 * v1 用 metapage 锁,与 IVFFlat 同协议;§4.7 的 128 把 list LWLock
	 * 留作 preload 版升级,页格式/WAL 布局不受影响)。
	 * 锁序:metapage → directory → posting,无反向。 */
	{
		Buffer		mbuf = ReadBuffer(index, DANN_METAPAGE_BLKNO);

		LockBuffer(mbuf, BUFFER_LOCK_EXCLUSIVE);
		/* 持锁下重读 directory(编码期间尾页可能已被并发插入推进) */
		dann_get_list_entry(index, &meta, listno, &le);
		dann_insert_locked(index, &meta, listno, &le, entry, entry_size, epp);
		LockBuffer(mbuf, BUFFER_LOCK_UNLOCK);
		ReleaseBuffer(mbuf);
	}

	pfree(entry);
	pfree(ws);
	return true;
}

