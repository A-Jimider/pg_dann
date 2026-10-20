/*-------------------------------------------------------------------------
 * vacuum.c — ambulkdelete / amvacuumcleanup + dann_list_stats()
 *
 * bulkdelete(W10,DESIGN.md §4.5):全索引扫描,回调逐 entry 判定删除;
 * 页内 swap-with-last 紧凑化(条目无序,交换合法);删除走 generic_xlog。
 * v1 不做页内以外的空页回收(与 IVFFlat 行为一致,靠 REINDEX 重建)。
 *
 * 并发协议(W10):directory 更新持 metapage BUFFER_LOCK_EXCLUSIVE 串行化
 * (与 IVFFlat 的 metapage 协议同构;§4.7 的 128 把 list LWLock 需
 * shared_preload_libraries,留作 preload 版升级,页格式无需变)。
 *-------------------------------------------------------------------------*/
#include "postgres.h"

#include "access/genam.h"
#include "access/generic_xlog.h"
#include "access/tableam.h"
#include "catalog/pg_am.h"
#include "commands/vacuum.h"
#include "funcapi.h"
#include "miscadmin.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "utils/rel.h"
#include "utils/syscache.h"

#include "dann.h"

static IndexBulkDeleteResult *
dann_default_stats(Relation index)
{
	IndexBulkDeleteResult *stats;

	stats = (IndexBulkDeleteResult *) palloc0(sizeof(IndexBulkDeleteResult));
	stats->num_pages = RelationGetNumberOfBlocksInFork(index, MAIN_FORKNUM);
	stats->estimated_count = true;
	stats->num_index_tuples = 0;

	return stats;
}

/*
 * 单个 posting 页:回调判死 + swap-with-last 紧凑删除。
 * 返回删除数;next 经指针输出(链条在 image 上读,删除不影响链)。
 */
static uint32
dann_vacuum_page(Relation index, BlockNumber blkno, uint32 entry_size,
				 IndexBulkDeleteCallback callback, void *callback_state,
				 BlockNumber *next)
{
	Buffer		buf = ReadBuffer(index, blkno);
	Page		page;
	DannPageOpaque opaque;
	GenericXLogState *gxlog;
	uint32		i;
	uint32		cnt;
	uint32		removed = 0;
	char	   *base;

	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	gxlog = GenericXLogStart(index);
	page = GenericXLogRegisterBuffer(gxlog, buf, 0);
	opaque = (DannPageOpaque) PageGetSpecialPointer(page);
	base = (char *) PageGetContents(page);
	cnt = opaque->count;
	*next = opaque->next;

	i = 0;
	while (i < cnt)
	{
		ItemPointer ctid = (ItemPointer) (base + (size_t) i * entry_size);

		if (callback(ctid, callback_state))
		{
			/* 与最后一个有效条目交换后收缩;换入者下轮再判 */
			cnt--;
			if (i != cnt)
				memcpy(base + (size_t) i * entry_size,
					   base + (size_t) cnt * entry_size, entry_size);
			removed++;
		}
		else
			i++;
	}

	if (removed > 0)
	{
		opaque->count = cnt;
		/* 内容收缩:pd_lower 跟着缩(收缩区在 hole 清零语义内) */
		dann_page_set_lower(page, cnt * entry_size);
		GenericXLogFinish(gxlog);
	}
	else
		GenericXLogAbort(gxlog);

	UnlockReleaseBuffer(buf);
	return removed;
}

IndexBulkDeleteResult *
dann_bulkdelete(IndexVacuumInfo *info, IndexBulkDeleteResult *stats,
				IndexBulkDeleteCallback callback, void *callback_state)
{
	Relation	index = info->index;
	DannMetaPageData meta;
	uint32	   *removed_per_list;
	uint32		listno;
	uint64		removed_total = 0;
	uint64		remaining = 0;

	if (stats == NULL)
		stats = dann_default_stats(index);
	stats->num_pages = RelationGetNumberOfBlocksInFork(index, MAIN_FORKNUM);

	dann_get_metapage(index, &meta);
	if (meta.nlist == 0)
	{
		stats->num_index_tuples = 0;
		stats->estimated_count = false;
		return stats;
	}

	removed_per_list = (uint32 *) palloc0(sizeof(uint32) * meta.nlist);

	/* 逐 list 逐页判死删除 */
	for (listno = 0; listno < meta.nlist; listno++)
	{
		DannListEntry le;
		BlockNumber blkno;

		dann_get_list_entry(index, &meta, listno, &le);
		blkno = le.first;
		while (blkno != InvalidBlockNumber)
		{
			BlockNumber next;

			removed_per_list[listno] += dann_vacuum_page(index, blkno,
														 meta.entry_size,
														 callback,
														 callback_state,
														 &next);
			blkno = next;
		}
		removed_total += removed_per_list[listno];
		remaining += le.count - removed_per_list[listno];	/* 现值口径 */
	}

	/* directory 更新:metapage EXCLUSIVE 串行化,按 dir 页分批(每页一批) */
	{
		Buffer		mbuf = ReadBuffer(index, DANN_METAPAGE_BLKNO);

		LockBuffer(mbuf, BUFFER_LOCK_EXCLUSIVE);

		listno = 0;
		while (listno < meta.nlist)
		{
			uint32		dir_page_idx = listno / DANN_DIR_PER_PAGE;
			uint32		end = Min(meta.nlist,
								  (dir_page_idx + 1) * DANN_DIR_PER_PAGE);
			uint32		ln;
			bool		dirty = false;

			/* 本批恰好覆盖一个 dir 页 */
			for (ln = listno; ln < end; ln++)
				if (removed_per_list[ln] > 0)
					dirty = true;

			if (dirty)
			{
				GenericXLogState *gxlog;
				Buffer		dbuf;
				Page		dpage;
				DannPageOpaque opaque;
				char	   *base;

				gxlog = GenericXLogStart(index);
				dbuf = ReadBuffer(index, meta.dir_start + dir_page_idx);
				LockBuffer(dbuf, BUFFER_LOCK_EXCLUSIVE);
				dpage = GenericXLogRegisterBuffer(gxlog, dbuf, 0);
				opaque = (DannPageOpaque) PageGetSpecialPointer(dpage);
				base = (char *) PageGetContents(dpage);

				for (ln = listno; ln < end; ln++)
				{
					DannListEntry le;

					if (removed_per_list[ln] == 0)
						continue;
					memcpy(&le, base + (size_t) (ln % DANN_DIR_PER_PAGE)
						   * sizeof(DannListEntry), sizeof(DannListEntry));
					le.count -= removed_per_list[ln];
					memcpy(base + (size_t) (ln % DANN_DIR_PER_PAGE)
						   * sizeof(DannListEntry), &le, sizeof(DannListEntry));
				}
				dann_page_set_lower(dpage,
									opaque->count * (uint32) sizeof(DannListEntry));
				GenericXLogFinish(gxlog);
				UnlockReleaseBuffer(dbuf);
			}
			listno = end;
		}

		LockBuffer(mbuf, BUFFER_LOCK_UNLOCK);
		ReleaseBuffer(mbuf);
	}

	pfree(removed_per_list);

	stats->num_index_tuples = (double) remaining;
	stats->estimated_count = false;
	return stats;
}

IndexBulkDeleteResult *
dann_vacuumcleanup(IndexVacuumInfo *info, IndexBulkDeleteResult *stats)
{
	Relation	index = info->index;
	DannMetaPageData meta;

	if (info->analyze_only)
		return stats;

	if (stats == NULL)
		stats = dann_default_stats(index);
	stats->num_pages = RelationGetNumberOfBlocksInFork(index, MAIN_FORKNUM);

	/* bulkdelete 未跑(无死元组)时,按 directory 现值汇总精确条目数 */
	dann_get_metapage(index, &meta);
	if (meta.nlist > 0)
	{
		uint64		total = 0;
		uint32		listno;

		for (listno = 0; listno < meta.nlist; listno++)
		{
			DannListEntry le;

			dann_get_list_entry(index, &meta, listno, &le);
			total += le.count;
		}
		stats->num_index_tuples = (double) total;
		stats->estimated_count = false;
	}

	return stats;
}

/* ---- 可观测性(§4.5 离散度监控 / §4.10) ----
 * dann_list_stats(rel regclass) → (listno, entries, pages, span)
 * span = last - first + 1(构建期连续布局时 == pages;离散追加后 > pages)
 */
PG_FUNCTION_INFO_V1(dann_list_stats);

Datum
dann_list_stats(PG_FUNCTION_ARGS)
{
	Oid			relid = PG_GETARG_OID(0);
	Relation	index;
	IndexAmRoutine *amroutine;
	DannMetaPageData meta;
	ReturnSetInfo *rsinfo;
	uint32		listno;

	InitMaterializedSRF(fcinfo, 0);
	rsinfo = (ReturnSetInfo *) fcinfo->resultinfo;

	index = index_open(relid, AccessShareLock);
	if (index->rd_rel->relkind != RELKIND_INDEX)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("pg_dann: %s is not an index",
						RelationGetRelationName(index))));
	amroutine = GetIndexAmRoutine(index->rd_amhandler);
	if (amroutine->ambuild != dann_build)
		ereport(ERROR,
				(errcode(ERRCODE_WRONG_OBJECT_TYPE),
				 errmsg("pg_dann: %s is not a dann index",
						RelationGetRelationName(index))));

	dann_get_metapage(index, &meta);

	for (listno = 0; listno < meta.nlist; listno++)
	{
		DannListEntry le;
		BlockNumber blkno;
		uint64		pages = 0;
		BlockNumber	last = InvalidBlockNumber;
		Datum		values[4];
		bool		nulls[4] = {false, false, false, false};

		dann_get_list_entry(index, &meta, listno, &le);
		blkno = le.first;
		while (blkno != InvalidBlockNumber)
		{
			Buffer		buf = ReadBuffer(index, blkno);
			Page		page;
			DannPageOpaque opaque;

			LockBuffer(buf, BUFFER_LOCK_SHARE);
			page = BufferGetPage(buf);
			opaque = (DannPageOpaque) PageGetSpecialPointer(page);
			last = blkno;
			blkno = opaque->next;
			LockBuffer(buf, BUFFER_LOCK_UNLOCK);
			ReleaseBuffer(buf);
			pages++;
		}

		values[0] = Int32GetDatum((int32) listno);
		values[1] = Int64GetDatum((int64) le.count);
		values[2] = Int64GetDatum((int64) pages);
		values[3] = Int64GetDatum(pages == 0 ? 0 :
								  (int64) last - (int64) le.first + 1);
		tuplestore_putvalues(rsinfo->setResult, rsinfo->setDesc,
							 values, nulls);
	}

	index_close(index, AccessShareLock);
	return (Datum) 0;
}
