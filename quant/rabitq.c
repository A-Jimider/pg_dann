/*-------------------------------------------------------------------------
 * quant/rabitq.c — RaBitQ 编码 / 查询预处理 / 标量估计(纯 C)
 *
 * 此文件是 golden/rabitq_model.py 的直接对应物,保持"显然正确";
 * 条目热循环 ⟨x̄,q'⟩ 的 SIMD 版在 simd/rabitq_ipqr_*.c,经 dispatch。
 *-------------------------------------------------------------------------*/
#include "quant.h"

#include <math.h>
#include <string.h>

void
dann_rabitq_encode(const float *v, const float *c, int dim, int padded_dim,
				   const float *sign, const uint32_t *perm, float *scratch,
				   uint8_t *code, float *dist_c_out, float *g_out)
{
	float	   *oc = scratch;				/* N:v-c 补零,rotate 消耗 */
	float	   *rot = scratch + padded_dim;	/* N:P·(v-c) */
	int			nbytes = (padded_dim + 7) / 8;
	float		dist_c = 0.0f;
	float		s = 0.0f;
	float		sum = 0.0f;
	int			i;

	memset(code, 0, (size_t) nbytes);
	for (i = 0; i < dim; i++)
	{
		float		d = v[i] - c[i];

		oc[i] = d;
		dist_c += d * d;
	}
	for (i = dim; i < padded_dim; i++)
		oc[i] = 0.0f;
	dist_c = sqrtf(dist_c);

	if (dist_c <= 0.0f)				/* o == c:code 全零、g = 0 */
	{
		*dist_c_out = 0.0f;
		*g_out = 0.0f;
		return;
	}

	dann_rotate(oc, rot, sign, perm, padded_dim);

	/* y = rot/dist_c;dist_c > 0 不变号,bits 直接取 rot 的符号;
	 * g = ⟨x̄,y⟩ = (2·Σ_{bit} rot_i - Σ rot_i) / (dist_c·√N) */
	for (i = 0; i < padded_dim; i++)
	{
		sum += rot[i];
		if (rot[i] >= 0.0f)
		{
			code[i >> 3] |= (uint8_t) (1u << (i & 7));
			s += rot[i];
		}
	}

	*dist_c_out = dist_c;
	*g_out = (2.0f * s - sum) / (dist_c * sqrtf((float) padded_dim));
}

float
dann_rabitq_prepare_query(const float *q, const float *c,
						  int dim, int padded_dim,
						  const float *sign, const uint32_t *perm,
						  float *scratch, float *qr, float *sum_q_out)
{
	float	   *qc = scratch;
	float	   *rot = scratch + padded_dim;
	float		nqc = 0.0f;
	float		sum = 0.0f;
	int			i;

	for (i = 0; i < dim; i++)
	{
		float		d = q[i] - c[i];

		qc[i] = d;
		nqc += d * d;
	}
	for (i = dim; i < padded_dim; i++)
		qc[i] = 0.0f;
	nqc = sqrtf(nqc);

	if (nqc <= 0.0f)				/* q == c:qr 全零,估计退化为 dist_c² */
	{
		memset(qr, 0, sizeof(float) * (size_t) padded_dim);
		*sum_q_out = 0.0f;
		return 0.0f;
	}

	dann_rotate(qc, rot, sign, perm, padded_dim);

	/* 与 golden 模型同为除法(reciprocal 乘法差 1ulp,见 fwht.c) */
	for (i = 0; i < padded_dim; i++)
	{
		qr[i] = rot[i] / nqc;
		sum += qr[i];
	}
	*sum_q_out = sum;
	return nqc;
}

float
dann_rabitq_ipqr_scalar(const uint8_t *code, const float *qr,
						float sum_q, int padded_dim)
{
	float		s = 0.0f;
	int			i;

	for (i = 0; i < padded_dim; i++)
		if ((code[i >> 3] >> (i & 7)) & 1)
			s += qr[i];

	return (2.0f * s - sum_q) / sqrtf((float) padded_dim);
}

float
dann_rabitq_estimate_l2(const uint8_t *code, const DannRabitqFactors *f,
						const float *qr, float sum_q, float qc_norm,
						int padded_dim)
{
	float		ipqr = dann_rabitq_ipqr(code, qr, sum_q, padded_dim);
	float		cos = (f->g > 1e-6f) ? ipqr / f->g : 0.0f;

	return f->dist_c2 + qc_norm * qc_norm - 2.0f * f->dist_c * qc_norm * cos;
}

float
dann_rabitq_estimate_cos(const uint8_t *code, const DannRabitqFactors *f,
						 const float *qr, float sum_q, int padded_dim)
{
	float		ipqr = dann_rabitq_ipqr(code, qr, sum_q, padded_dim);

	return (f->g > 1e-6f) ? ipqr / f->g : 0.0f;
}
