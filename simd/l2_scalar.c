/*-------------------------------------------------------------------------
 * simd/l2_scalar.c — L2 平方距离标量参考实现
 *
 * 返回值是平方距离(||a-b||^2,省去开方——最近邻排序不受影响)。
 *-------------------------------------------------------------------------*/
#include "quant.h"

float
dann_l2_scalar(const float *a, const float *b, int dim)
{
	float		sum = 0.0f;
	int			i;

	for (i = 0; i < dim; i++)
	{
		float		d = a[i] - b[i];

		sum += d * d;
	}

	return sum;
}

float
dann_dot_scalar(const float *a, const float *b, int dim)
{
	float		sum = 0.0f;
	int			i;

	for (i = 0; i < dim; i++)
		sum += a[i] * b[i];

	return sum;
}
