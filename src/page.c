/*-------------------------------------------------------------------------
 * page.c — 页面格式与访问助手
 *
 * 页面类型与布局见 DESIGN.md §3/§4.1。所有页面带 DannPageOpaqueData
 * special area;posting 页通过 opaque->next 链式连接。
 *-------------------------------------------------------------------------*/
#include "postgres.h"

#include <math.h>

#include "access/generic_xlog.h"
#include "access/htup_details.h"
#include "access/tupdesc.h"
#include "catalog/pg_proc.h"
#include "commands/defrem.h"
#include "storage/bufmgr.h"
#include "storage/bufpage.h"
#include "utils/rel.h"
#include "utils/syscache.h"

#include "dann.h"
#include "quant.h"

/* 初始化页镜像(PageInit + special area);generic_xlog 注册页与裸 buffer 共用 */
void
dann_init_page_image(Page page, DannPageType type)
{
	DannPageOpaque opaque;

	PageInit(page, BLCKSZ, sizeof(DannPageOpaqueData));
	opaque = (DannPageOpaque) PageGetSpecialPointer(page);
	opaque->page_type = type;
	opaque->next = InvalidBlockNumber;
	opaque->count = 0;
}

/*
 * dann 页把定长内容从 PageGetContents 起紧密排列(占用标准页的"hole"
 * 区)。generic_xlog 的在线应用与 redo 都会把 [pd_lower, pd_upper) 清零
 * ——写完内容必须把内容末端并进 pd_lower,否则内容被抹掉(W9 踩坑)。
 */
void
dann_page_set_lower(Page page, uint32 content_bytes)
{
	((PageHeader) page)->pd_lower =
		(OffsetNumber) (SizeOfPageHeaderData + content_bytes);
}

void
dann_init_page(Buffer buf, DannPageType type)
{
	dann_init_page_image(BufferGetPage(buf), type);
}

/* 读 metapage(share lock,拷出后释放) */
void
dann_get_metapage(Relation index, DannMetaPageData *meta)
{
	Buffer		buf;
	Page		page;

	buf = ReadBuffer(index, DANN_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	memcpy(meta, PageGetContents(page), sizeof(DannMetaPageData));
	LockBuffer(buf, BUFFER_LOCK_UNLOCK);
	ReleaseBuffer(buf);

	if (meta->magic != DANN_MAGIC || meta->version != DANN_VERSION)
		ereport(ERROR,
				(errcode(ERRCODE_INDEX_CORRUPTED),
				 errmsg("pg_dann: index metapage has bad magic/version "
						"(index created by an incompatible pg_dann version?)")));
}

/* 原地重写 metapage(仅构建期使用;W9 才 WAL 化) */
void
dann_write_metapage(Relation index, DannMetaPageData *meta)
{
	Buffer		buf;
	Page		page;

	buf = ReadBuffer(index, DANN_METAPAGE_BLKNO);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	page = BufferGetPage(buf);
	memcpy(PageGetContents(page), meta, sizeof(DannMetaPageData));
	MarkBufferDirty(buf);
	LockBuffer(buf, BUFFER_LOCK_UNLOCK);
	ReleaseBuffer(buf);
}

/* directory 页 slot 定位 */
static void
dann_dir_locate(const DannMetaPageData *meta, uint32 listno,
				BlockNumber *blkno, uint32 *offset)
{
	*blkno = meta->dir_start + listno / DANN_DIR_PER_PAGE;
	*offset = listno % DANN_DIR_PER_PAGE;
}

void
dann_get_list_entry(Relation index, const DannMetaPageData *meta,
					uint32 listno, DannListEntry *entry)
{
	BlockNumber blkno;
	uint32		offset;
	Buffer		buf;
	Page		page;
	char	   *base;

	dann_dir_locate(meta, listno, &blkno, &offset);
	buf = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	base = (char *) PageGetContents(page);
	memcpy(entry, base + offset * sizeof(DannListEntry), sizeof(DannListEntry));
	LockBuffer(buf, BUFFER_LOCK_UNLOCK);
	ReleaseBuffer(buf);
}

void
dann_set_list_entry(Relation index, const DannMetaPageData *meta,
					uint32 listno, const DannListEntry *entry)
{
	BlockNumber blkno;
	uint32		offset;
	Buffer		buf;
	Page		page;
	char	   *base;

	dann_dir_locate(meta, listno, &blkno, &offset);
	buf = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	page = BufferGetPage(buf);
	base = (char *) PageGetContents(page);
	memcpy(base + offset * sizeof(DannListEntry), entry, sizeof(DannListEntry));
	MarkBufferDirty(buf);
	LockBuffer(buf, BUFFER_LOCK_UNLOCK);
	ReleaseBuffer(buf);
}

/*
 * directory 条目的 WAL 化更新:登记进调用方的 GenericXLog(同一事务
 * 原子),不在此解锁/释放(交 Finish 后由调用方 UnlockReleaseBuffer)。
 * pd_lower 统一抬到全量内容末端——兼容 W9 前构建的页(pd_lower=24,
 * 内容在 hole 里,否则在线应用的 hole 清零会抹掉旧内容)。
 */
Buffer
dann_set_list_entry_xlog(GenericXLogState *state, Relation index,
						 const DannMetaPageData *meta, uint32 listno,
						 const DannListEntry *entry)
{
	BlockNumber blkno;
	uint32		offset;
	Buffer		buf;
	Page		page;
	DannPageOpaque opaque;
	char	   *base;

	dann_dir_locate(meta, listno, &blkno, &offset);
	buf = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_EXCLUSIVE);
	page = GenericXLogRegisterBuffer(state, buf, 0);
	opaque = (DannPageOpaque) PageGetSpecialPointer(page);
	base = (char *) PageGetContents(page);
	memcpy(base + offset * sizeof(DannListEntry), entry, sizeof(DannListEntry));
	dann_page_set_lower(page, opaque->count * (uint32) sizeof(DannListEntry));
	return buf;
}

/* 按 listno 随机访问质心(质心不跨页,构建期保证) */
void
dann_get_centroid(Relation index, const DannMetaPageData *meta,
				  uint32 listno, float *out)
{
	uint32		cpp = (uint32) (DANN_PAGE_PAYLOAD / (meta->dim * sizeof(float)));
	BlockNumber blkno = meta->centroid_start + listno / cpp;
	uint32		offset = listno % cpp;
	Buffer		buf;
	Page		page;
	char	   *base;

	buf = ReadBuffer(index, blkno);
	LockBuffer(buf, BUFFER_LOCK_SHARE);
	page = BufferGetPage(buf);
	base = (char *) PageGetContents(page);
	memcpy(out, base + (size_t) offset * meta->dim * sizeof(float),
		   sizeof(float) * meta->dim);
	LockBuffer(buf, BUFFER_LOCK_UNLOCK);
	ReleaseBuffer(buf);
}

uint32
dann_padded_dim(uint32 dim)
{
	uint32		n = 1;

	while (n < dim)
		n <<= 1;
	return n;
}

int
dann_code_len(uint32 quantizer, uint32 dim, uint32 padded_dim)
{
	if (quantizer == DANN_QUANTIZER_SQ8)
		return DANN_ENTRY_CODE_LEN_SQ8(dim);
	/* RaBitQ:码在补零后的 2 幂空间中 */
	return DANN_ENTRY_CODE_LEN_RABITQ(padded_dim);
}

int
dann_entry_size(uint32 quantizer, uint32 dim, uint32 padded_dim)
{
	int			raw = DANN_ENTRY_HEADER_SIZE + dann_code_len(quantizer, dim, padded_dim);

	return (raw + 3) & ~3;		/* 4B 对齐 */
}

int
dann_entries_per_page(uint32 entry_size)
{
	return (int) (DANN_PAGE_PAYLOAD / (uint32) entry_size);
}

void
dann_rqctx_init(DannRqCtx *ctx, const DannMetaPageData *meta)
{
	ctx->padded_dim = meta->padded_dim;
	ctx->sign = (float *) palloc(sizeof(float) * meta->padded_dim);
	ctx->perm = (uint32 *) palloc(sizeof(uint32) * meta->padded_dim);
	ctx->scratch = (float *) palloc(sizeof(float) * meta->padded_dim * 2);
	ctx->qr = (float *) palloc(sizeof(float) * meta->padded_dim);
	dann_rotation_params(meta->rotation_seed, (int) meta->padded_dim,
						 ctx->sign, ctx->perm);
}

void
dann_rqctx_free(DannRqCtx *ctx)
{
	pfree(ctx->sign);
	pfree(ctx->perm);
	pfree(ctx->scratch);
	pfree(ctx->qr);
}

/*
 * 解析 pgvector vector Datum。校验维度与索引一致。
 * 返回可能需要调用方释放(dann_free_vector)的向量。
 */
DannVector *
dann_get_vector(Datum d, uint32 expected_dim)
{
	DannVector *v = (DannVector *) PG_DETOAST_DATUM(d);

	if ((uint32) v->dim != expected_dim)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("pg_dann: vector dimension %d does not match index dimension %u",
						(int) v->dim, expected_dim)));
	return v;
}

void
dann_free_vector(Datum d, DannVector *v)
{
	if ((Pointer) v != DatumGetPointer(d))
		pfree(v);
}

/* NaN/Inf 校验:进入构建/插入/查询前必须拒绝(kmeans 会被 NaN 毒化,
 * SQ8 的 min/max 也会传染整段编码) */
void
dann_check_finite(const float *x, int dim)
{
	int			i;

	for (i = 0; i < dim; i++)
		if (!isfinite(x[i]))
			ereport(ERROR,
					(errcode(ERRCODE_DATA_EXCEPTION),
					 errmsg("pg_dann: vector contains NaN or Infinity")));
}

/* 原地归一化(cosine 度量用);零向量报错,与 pgvector 行为一致 */
void
dann_normalize(float *x, int dim)
{
	double		norm = 0.0;
	int			i;

	for (i = 0; i < dim; i++)
		norm += (double) x[i] * x[i];
	norm = sqrt(norm);

	if (norm == 0.0)
		ereport(ERROR,
				(errcode(ERRCODE_DATA_EXCEPTION),
				 errmsg("pg_dann: cannot normalize zero vector for cosine metric")));

	{
		float		inv = (float) (1.0 / norm);

		for (i = 0; i < dim; i++)
			x[i] = x[i] * inv;
	}
}

/*
 * 从索引支持函数 1(距离函数)解析度量。
 * 我们的 opclass 把 pgvector 的距离函数注册为支持函数:
 * l2_distance / vector_negative_inner_product / cosine_distance。
 * 以 proname 匹配,与 pgvector 的做法(比较支持函数 OID)等价。
 */
uint32
dann_metric_of(Relation index)
{
	FmgrInfo   *procinfo = index_getprocinfo(index, 1, 1);
	HeapTuple	tuple;
	Form_pg_proc procform;
	const char *fname;
	uint32		metric;

	tuple = SearchSysCache1(PROCOID, ObjectIdGetDatum(procinfo->fn_oid));
	if (!HeapTupleIsValid(tuple))
		elog(ERROR, "pg_dann: cache lookup failed for function %u",
			 procinfo->fn_oid);
	procform = (Form_pg_proc) GETSTRUCT(tuple);
	fname = NameStr(procform->proname);

	if (strcmp(fname, "l2_distance") == 0)
		metric = DANN_METRIC_L2;
	else if (strcmp(fname, "cosine_distance") == 0)
		metric = DANN_METRIC_COSINE;
	else if (strcmp(fname, "vector_negative_inner_product") == 0 ||
			 strcmp(fname, "negative_inner_product") == 0 ||
			 strcmp(fname, "inner_product") == 0)
		metric = DANN_METRIC_IP;
	else
		ereport(ERROR,
				(errcode(ERRCODE_UNDEFINED_OBJECT),
				 errmsg("pg_dann: unsupported distance function \"%s\"", fname)));

	ReleaseSysCache(tuple);
	return metric;
}
