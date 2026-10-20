/*-------------------------------------------------------------------------
 * cost.c — amcostestimate(DESIGN.md §4.8)
 *
 * 模型(单次扫描):
 *   startup  = 常数 + 质心扫描 cpu(nlist × dim 次乘加,SIMD 折算)
 *   io       = (目录+质心页,热)× seq_page_cost
 *            + probes × list 平均页数 × seq_page_cost(构建期物理连续)
 *   cpu      = probes × list 平均条目数 × dim × cpu_operator_cost × SIMD 折算
 *   rerank   = γ × 100 回表 × (random_page_cost + cpu_tuple_cost)
 *              (amgettuple 不知 LIMIT k,按 k=100 估;γ 尊重 GUC/reloption)
 *
 * 目标务实:数量级正确 + 关键参数走 GUC/reloption 可调。ANN 的 recall
 * 语义 planner 不理解,"该走索引没走"类问题不承诺解决,只承诺不添乱:
 * 无 KNN ORDER BY 时代价压过 disable_cost,永不入选。
 *
 * 元数据(pgvector 同款做法):NoLock 打开索引读 metapage——拿到构建期
 * 实际 nlist/entry_size/ntuples,比 reloptions 意图值准(夹子后仍对)。
 *-------------------------------------------------------------------------*/
#include "postgres.h"

#include <math.h>

#include "access/amapi.h"
#include "access/generic_xlog.h"
#include "fmgr.h"
#include "nodes/pathnodes.h"
#include "optimizer/cost.h"
#include "optimizer/optimizer.h"
#include "storage/bufmgr.h"
#include "utils/rel.h"
#include "utils/selfuncs.h"
#include "utils/spccache.h"

#include "dann.h"

void
dann_costestimate(PlannerInfo *root, IndexPath *path, double loop_count,
				  Cost *indexStartupCost, Cost *indexTotalCost,
				  Selectivity *indexSelectivity, double *indexCorrelation,
				  double *indexPages)
{
	Relation	index;
	DannMetaPageData meta;
	double		startup;
	double		io;
	double		cpu;
	double		total;
	int			probes;
	int			gamma;

	if (path->indexorderbys == NIL)
	{
		/* 无 ORDER BY:dann 是全表扫描的拙劣替代品,永不入选 */
		*indexStartupCost = 1.0e12;
		*indexTotalCost = 1.0e12 + 100;
		*indexSelectivity = 1.0;
		*indexCorrelation = 0.0;
		*indexPages = 1;
		return;
	}

	index = index_open(path->indexinfo->indexoid, NoLock);
	dann_get_metapage(index, &meta);
	*indexPages = (double) RelationGetNumberOfBlocksInFork(index, MAIN_FORKNUM);
	probes = dann_probes_of(index);
	gamma = dann_rerank_factor_of(index);
	index_close(index, NoLock);

	if (meta.nlist == 0 || meta.ntuples < 1.0)
	{
		/* 空索引:内容为空但仍是合法扫描;给个低价让 planner 无负担 */
		*indexStartupCost = 100.0;
		*indexTotalCost = 200.0;
		*indexSelectivity = 1.0;
		*indexCorrelation = 0.0;
		return;
	}

	{
		double		avg_entries = meta.ntuples / meta.nlist;
		double		list_pages;		/* 每 list 平均页数 */
		double		dim = (double) meta.dim;
		double		spc_seq_cost;
		double		spc_random_cost;
		double		kheap;

		list_pages = ceil(avg_entries * (double) meta.entry_size
						  / (double) DANN_PAGE_PAYLOAD);
		if (list_pages < 1.0)
			list_pages = 1.0;

		get_tablespace_page_costs(index->rd_rel->reltablespace,
								  &spc_random_cost, &spc_seq_cost);

		/* 质心扫描:nlist×dim 乘加,AVX-512 每周期 16 维 → 折算 0.05 */
		startup = 50.0 + (double) meta.nlist * dim * cpu_operator_cost * 0.05;

		/* 目录+质心页(常驻热数据)+ posting 页段(连续 ⇒ seq) */
		io = (1.0 + (double) meta.dir_npages + (double) meta.centroid_npages)
			* spc_seq_cost
			+ (double) probes * list_pages * spc_seq_cost;

		/* 条目打分:每条约 dim 次掩码累加(AVX-512 16 车道) */
		cpu = (double) probes * avg_entries * dim * cpu_operator_cost * 0.1;

		/* rerank 回表:γk 随机页 + 逐条精确距离;k 按 100 估 */
		kheap = dann_enable_rerank ? (double) gamma * 100.0 : 0.0;
		cpu += kheap * (cpu_tuple_cost + dim * cpu_operator_cost * 0.05);
		io += kheap * spc_random_cost;

		total = startup + io + cpu;
	}

	*indexStartupCost = startup;
	*indexTotalCost = startup + (total - startup) * loop_count;
	*indexSelectivity = 1.0;	/* KNN 截断由 LIMIT 表达,不走 selectivity */
	*indexCorrelation = 0.0;
}
