/*-------------------------------------------------------------------------
 * dann.h — pg_dann 共享类型与接口
 *
 * 页面布局与决策依据见 DESIGN.md §3/§4。
 *-------------------------------------------------------------------------*/
#ifndef PG_DANN_H
#define PG_DANN_H

#include "postgres.h"

#include "access/amapi.h"
#include "access/genam.h"
#include "access/generic_xlog.h"
#include "access/relscan.h"
#include "fmgr.h"
#include "nodes/execnodes.h"
#include "nodes/pathnodes.h"
#include "storage/block.h"
#include "storage/itemptr.h"
#include "utils/rel.h"

/* ---- 页面布局(DESIGN.md §4.1) ---- */

#define DANN_MAGIC			0xDA11DA11
#define DANN_VERSION		2
#define DANN_METAPAGE_BLKNO	0

typedef enum DannPageType
{
	DANN_PAGE_META = 1,
	DANN_PAGE_DIRECTORY,		/* list directory */
	DANN_PAGE_CENTROID,			/* 质心页 */
	DANN_PAGE_POSTING			/* posting 页 */
} DannPageType;

/* 每个 dann 页的 special area(12B) */
typedef struct DannPageOpaqueData
{
	uint32		page_type;		/* DannPageType */
	uint32		next;			/* 链式下一页(posting);InvalidBlockNumber 结尾 */
	uint32		count;			/* 页内条目数(posting) */
} DannPageOpaqueData;

typedef DannPageOpaqueData *DannPageOpaque;

/* 页内可用净载空间 */
#define DANN_PAGE_PAYLOAD (BLCKSZ - SizeOfPageHeaderData - sizeof(DannPageOpaqueData))

/* ---- metapage(page 0,PageGetContents() 起始处) ---- */

typedef enum DannMetric
{
	DANN_METRIC_L2 = 1,
	DANN_METRIC_IP = 2,
	DANN_METRIC_COSINE = 3
} DannMetric;

typedef struct DannMetaPageData
{
	uint32		magic;
	uint32		version;
	uint32		dim;			/* 来自 vector typmod */
	uint32		padded_dim;		/* 补齐到 2 的幂,FWHT 用(W7) */
	uint32		nlist;
	uint32		quantizer;		/* 构建期已解析为具体量化器 */
	uint32		metric;			/* DannMetric,来自 opclass */
	uint32		rotation_seed;	/* RaBitQ 旋转种子(W7) */
	BlockNumber dir_start;		/* list directory 起始块 */
	uint32		dir_npages;
	BlockNumber centroid_start; /* 质心页起始块 */
	uint32		centroid_npages;
	BlockNumber posting_start;	/* posting 区起始块 */
	uint32		entry_size;		/* posting entry 定长(4B 对齐) */
	double		ntuples;
} DannMetaPageData;

/* ---- list directory(DESIGN.md §3) ---- */

typedef struct DannListEntry
{
	BlockNumber first;			/* 首个 posting 页 */
	BlockNumber last;			/* 尾页(追加目标) */
	uint32		count;			/* 本 list 条目总数 */
	uint32		reserved;
} DannListEntry;				/* 16B */

#define DANN_DIR_PER_PAGE ((int) (DANN_PAGE_PAYLOAD / sizeof(DannListEntry)))

/* ---- posting entry(定长,同索引内恒定) ----
 * 头部 20B:ctid(6) + pad(2) + 3×float4 量化因子;
 * 随后 code_len 字节量化码;整体 4B 对齐。
 * 因子区:SQ8 = DannSQ8Factors;RaBitQ = DannRabitqFactors(quant.h)。
 * RaBitQ 码在 padded_dim(补零到 2 的幂)空间中,码长 ceil(padded_dim/8)。
 */
#define DANN_ENTRY_HEADER_SIZE	20
#define DANN_ENTRY_CODE_LEN_SQ8(dim)	(dim)
#define DANN_ENTRY_CODE_LEN_RABITQ(padded_dim) (((padded_dim) + 7) / 8)

typedef enum DannQuantizer
{
	DANN_QUANTIZER_AUTO = 0,
	DANN_QUANTIZER_RABITQ = 1,
	DANN_QUANTIZER_SQ8 = 2
} DannQuantizer;

/* quantizer='auto' 的分界:dim < 128 → SQ8,否则 RaBitQ(DESIGN.md §4.2,
 * 阈值经 golden/rabitq_error.py 实测复核;W8 在 SIFT1M 上终验) */
#define DANN_RABITQ_MIN_DIM		128

/* pgvector vector 类型的内存镜像(pgvector 未导出其头文件,布局自
 * pgvector 0.x 起稳定:vl_len_ + dim + unused + x[])。运行时以
 * dim 与 typmod 交叉校验。 */
typedef struct DannVector
{
	int32		vl_len_;
	int16		dim;
	int16		unused;
	float		x[FLEXIBLE_ARRAY_MEMBER];
} DannVector;

/* ---- reloptions ---- */

typedef struct DannOptions
{
	int32		vl_len_;		/* varlena 头(勿动) */
	int			nlist;
	int			quantizer;		/* 字符串偏移量,未设为 0 */
	int			probes;			/* 0 = 未设,用 GUC dann.probes */
	int			rerank_factor;	/* 0 = 未设,用 GUC dann.rerank_factor */
} DannOptions;

#define DANN_DEFAULT_NLIST	1000
#define DANN_MIN_NLIST		1
#define DANN_MAX_NLIST		65535	/* list id 存 uint16(构建期 assignment 数组) */

/* dann.io_method 取值(预取实现见 W6) */
enum
{
	DANN_IO_AUTO,
	DANN_IO_READ_STREAM,		/* 需 PG17+,PG16 上降级 fadvise */
	DANN_IO_FADVISE,
	DANN_IO_SYNC
};

/* ---- GUC(dann.c 定义) ---- */
extern int	dann_probes;
extern int	dann_rerank_factor;
extern bool dann_enable_rerank;
extern int	dann_max_latency_ms;
extern int	dann_io_method;

/* ---- dann.c ---- */
bytea	   *dann_options(Datum reloptions, bool validate);
bool		dann_validate(Oid opclassoid);
int			dann_probes_of(Relation index);
int			dann_rerank_factor_of(Relation index);

/* ---- page.c ---- */
void		dann_init_page(Buffer buf, DannPageType type);
void		dann_init_page_image(Page page, DannPageType type);
void		dann_page_set_lower(Page page, uint32 content_bytes);
void		dann_get_metapage(Relation index, DannMetaPageData *meta);
void		dann_write_metapage(Relation index, DannMetaPageData *meta);
void		dann_get_list_entry(Relation index, const DannMetaPageData *meta,
								uint32 listno, DannListEntry *entry);
void		dann_set_list_entry(Relation index, const DannMetaPageData *meta,
								uint32 listno, const DannListEntry *entry);
Buffer		dann_set_list_entry_xlog(GenericXLogState *state, Relation index,
									 const DannMetaPageData *meta,
									 uint32 listno, const DannListEntry *entry);
void		dann_get_centroid(Relation index, const DannMetaPageData *meta,
							  uint32 listno, float *out);
uint32		dann_padded_dim(uint32 dim);
int			dann_code_len(uint32 quantizer, uint32 dim, uint32 padded_dim);
int			dann_entry_size(uint32 quantizer, uint32 dim, uint32 padded_dim);
int			dann_entries_per_page(uint32 entry_size);
DannVector *dann_get_vector(Datum d, uint32 expected_dim);
void		dann_free_vector(Datum d, DannVector *v);
void		dann_normalize(float *x, int dim);
void		dann_check_finite(const float *x, int dim);
uint32		dann_metric_of(Relation index);

/* RaBitQ 旋转上下文:sign/perm 由 meta->rotation_seed 派生,scratch/qr
 * 为工作缓冲;构建一次、扫描一次、插入每次(量小,后续可按需缓存) */
typedef struct DannRqCtx
{
	uint32		padded_dim;
	float	   *sign;			/* padded_dim */
	uint32	   *perm;			/* padded_dim */
	float	   *scratch;		/* 2 * padded_dim(rotate 工作区) */
	float	   *qr;				/* padded_dim(查询侧旋转结果) */
} DannRqCtx;

void		dann_rqctx_init(DannRqCtx *ctx, const DannMetaPageData *meta);
void		dann_rqctx_free(DannRqCtx *ctx);

/* ---- build.c ---- */
IndexBuildResult *dann_build(Relation heap, Relation index, IndexInfo *indexInfo);
void		dann_buildempty(Relation index);

/* ---- insert.c ---- */
bool		dann_insert(Relation index, Datum *values, bool *isnull,
						ItemPointer heap_tid, Relation heap,
						IndexUniqueCheck checkUnique, bool indexUnchanged,
						IndexInfo *indexInfo);
uint32		dann_quantizer_from_options(Relation index);
uint32		dann_nearest_centroid(Relation index, const DannMetaPageData *meta,
								  const float *v);

/* ---- kmeans.c ---- */
void		dann_kmeans(const float *samples, int nsamples, int dim, int k,
						int niter, float *centroids /* out: k*dim */ );
uint32		dann_assign(const float *v, const float *centroids, int k,
						int dim, uint32 metric);

/* ---- scan.c ---- */
IndexScanDesc dann_beginscan(Relation index, int nkeys, int norderbys);
void		dann_rescan(IndexScanDesc scan, ScanKey keys, int nkeys,
						ScanKey orderbys, int norderbys);
bool		dann_gettuple(IndexScanDesc scan, ScanDirection dir);
void		dann_endscan(IndexScanDesc scan);
Datum		dann_scan_stats(PG_FUNCTION_ARGS);
Datum		dann_scan_stats_reset(PG_FUNCTION_ARGS);

/* ---- vacuum.c ---- */
IndexBulkDeleteResult *dann_bulkdelete(IndexVacuumInfo *info,
									   IndexBulkDeleteResult *stats,
									   IndexBulkDeleteCallback callback,
									   void *callback_state);
IndexBulkDeleteResult *dann_vacuumcleanup(IndexVacuumInfo *info,
										  IndexBulkDeleteResult *stats);

/* ---- cost.c ---- */
void		dann_costestimate(PlannerInfo *root, IndexPath *path,
							  double loop_count,
							  Cost *indexStartupCost, Cost *indexTotalCost,
							  Selectivity *indexSelectivity,
							  double *indexCorrelation, double *indexPages);

#endif							/* PG_DANN_H */
