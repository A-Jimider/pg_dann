/*-------------------------------------------------------------------------
 * kmeans.c — 采样集上的 k-means(Lloyd)与最近质心分配
 *
 * v1 用确定性种子的朴素 Lloyd(W4);Elkan 三角不等式剪枝是后续优化。
 * cosine 度量由调用方先归一化,这里只做 L2;IP 用负内积分配。
 *-------------------------------------------------------------------------*/
#include "postgres.h"

#include <float.h>

#include "dann.h"
#include "quant.h"

/* 确定性 PRNG(LCG)——构建结果可复现,回归测试依赖 */
static uint64
dann_rand(uint64 *state)
{
	*state = *state * 6364136223846793005ULL + 1442695040888963407ULL;
	return *state >> 33;
}

/*
 * dann_assign — 返回 v 的最近质心编号。
 * L2/COSINE:最小平方距离;IP:最大内积。
 */
uint32
dann_assign(const float *v, const float *centroids, int k, int dim,
			uint32 metric)
{
	uint32		best = 0;
	int			i;

	if (metric == DANN_METRIC_IP)
	{
		float		bestscore = -FLT_MAX;

		for (i = 0; i < k; i++)
		{
			float		score = dann_dot_product(v, centroids + (size_t) i * dim, dim);

			if (score > bestscore)
			{
				bestscore = score;
				best = i;
			}
		}
	}
	else
	{
		float		bestscore = FLT_MAX;

		for (i = 0; i < k; i++)
		{
			float		score = dann_l2_distance(v, centroids + (size_t) i * dim, dim);

			if (score < bestscore)
			{
				bestscore = score;
				best = i;
			}
		}
	}

	return best;
}

/*
 * dann_kmeans — samples[nsamples × dim] 上跑 Lloyd,输出 centroids[k × dim]。
 * 空簇重新随机选样本点;固定迭代 niter 轮。
 */
void
dann_kmeans(const float *samples, int nsamples, int dim, int k, int niter,
			float *centroids)
{
	uint64		seed = 42;
	double	   *sums;			/* k × dim 双精度累加器 */
	int		   *counts;
	uint16	   *assign;			/* 每样本当前簇号 */
	int			iter;
	int			i,
				j,
				c;

	Assert(nsamples >= k && k > 0 && k <= 65535);

	sums = palloc0(sizeof(double) * (size_t) k * dim);
	counts = palloc0(sizeof(int) * k);
	assign = palloc(sizeof(uint16) * nsamples);

	/* 初始化:确定性无放回抽 k 个样本点 */
	for (i = 0; i < k; i++)
	{
		uint32		pick = (uint32) (dann_rand(&seed) % (uint64) nsamples);

		memcpy(centroids + (size_t) i * dim, samples + (size_t) pick * dim,
			   sizeof(float) * dim);
	}

	for (iter = 0; iter < niter; iter++)
	{
		/* 分配 */
		for (i = 0; i < nsamples; i++)
			assign[i] = (uint16) dann_assign(samples + (size_t) i * dim,
											 centroids, k, dim,
											 DANN_METRIC_L2);

		/* 重算 */
		memset(sums, 0, sizeof(double) * (size_t) k * dim);
		memset(counts, 0, sizeof(int) * k);
		for (i = 0; i < nsamples; i++)
		{
			const float *v = samples + (size_t) i * dim;
			double	   *s = sums + (size_t) assign[i] * dim;

			counts[assign[i]]++;
			for (j = 0; j < dim; j++)
				s[j] += v[j];
		}
		for (c = 0; c < k; c++)
		{
			float	   *cen = centroids + (size_t) c * dim;
			double	   *s = sums + (size_t) c * dim;

			if (counts[c] == 0)
			{
				/* 空簇:随机样本点重启 */
				uint32		pick = (uint32) (dann_rand(&seed) % (uint64) nsamples);

				memcpy(cen, samples + (size_t) pick * dim, sizeof(float) * dim);
				continue;
			}
			for (j = 0; j < dim; j++)
				cen[j] = (float) (s[j] / counts[c]);
		}
	}

	pfree(sums);
	pfree(counts);
	pfree(assign);
}
