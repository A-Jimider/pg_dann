/*-------------------------------------------------------------------------
 * quant/sq8.c — SQ8 标量量化:编码 + 标量参考估计
 *
 * 数学约定见 quant.h 注释。此文件是 golden model 的直接对应物,
 * 保持简单直白——性能路径在 simd/ 里,这里优先"显然正确"。
 *-------------------------------------------------------------------------*/
#include "quant.h"

#include <math.h>
#include <stddef.h>

void
dann_sq8_encode(const float *v, int dim, uint8_t *code, DannSQ8Factors *f)
{
	float		vmin = v[0];
	float		vmax = v[0];
	float		delta;
	float		deq_norm_sq = 0.0f;
	int			i;

	for (i = 1; i < dim; i++)
	{
		if (v[i] < vmin)
			vmin = v[i];
		if (v[i] > vmax)
			vmax = v[i];
	}

	delta = (vmax > vmin) ? (vmax - vmin) / 255.0f : 0.0f;

	if (delta > 0.0f)
	{
		float		inv_delta = 1.0f / delta;

		for (i = 0; i < dim; i++)
		{
			/* 与 golden model 约定的舍入:floor(x + 0.5) */
			float		t = floorf((v[i] - vmin) * inv_delta + 0.5f);
			int			c = (int) t;

			if (c < 0)
				c = 0;
			else if (c > 255)
				c = 255;
			code[i] = (uint8_t) c;
		}
	}
	else
	{
		/* 常数向量:全部量化为 0 */
		for (i = 0; i < dim; i++)
			code[i] = 0;
	}

	for (i = 0; i < dim; i++)
	{
		float		d = vmin + delta * (float) code[i];

		deq_norm_sq += d * d;
	}

	f->vmin = vmin;
	f->delta = delta;
	f->deq_norm_sq = deq_norm_sq;
}

float
dann_sq8_dot_scalar(const uint8_t *code, const DannSQ8Factors *f,
					const float *q, int dim)
{
	float		dot = 0.0f;		/* <q, deq> */
	int			i;

	for (i = 0; i < dim; i++)
	{
		float		d = f->vmin + f->delta * (float) code[i];

		dot += q[i] * d;
	}

	return dot;
}

float
dann_sq8_estimate_l2_scalar(const uint8_t *code, const DannSQ8Factors *f,
							const float *q, float q_norm_sq, int dim)
{
	return q_norm_sq - 2.0f * dann_sq8_dot_scalar(code, f, q, dim)
		+ f->deq_norm_sq;
}
