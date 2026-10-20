/*-------------------------------------------------------------------------
 * simd/sq8_est_avx2.c — SQ8 点积/距离估计 AVX2+FMA 实现
 *
 * 热循环:8 个 uint8 码 → 加宽为 int32 → 转 float → FMA 反量化并与
 * 查询向量点积累加。数学与 quant/sq8.c 标量版严格同式。
 *-------------------------------------------------------------------------*/
#include "quant.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

__attribute__((target("avx2,fma")))
float
dann_sq8_dot_avx2(const uint8_t *code, const DannSQ8Factors *f,
				  const float *q, int dim)
{
	__m256		acc = _mm256_setzero_ps();
	__m256		vvmin = _mm256_set1_ps(f->vmin);
	__m256		vdelta = _mm256_set1_ps(f->delta);
	int			i = 0;

	for (; i + 8 <= dim; i += 8)
	{
		__m128i		c8 = _mm_loadl_epi64((const __m128i *) (code + i));
		__m256i		c32 = _mm256_cvtepu8_epi32(c8);
		__m256		cf = _mm256_cvtepi32_ps(c32);
		__m256		deq = _mm256_fmadd_ps(vdelta, cf, vvmin);
		__m256		vq = _mm256_loadu_ps(q + i);

		acc = _mm256_fmadd_ps(vq, deq, acc);
	}

	{
		__m128		lo = _mm256_castps256_ps128(acc);
		__m128		hi = _mm256_extractf128_ps(acc, 1);
		__m128		s = _mm_add_ps(lo, hi);
		float		dot;

		s = _mm_add_ps(s, _mm_movehl_ps(s, s));
		s = _mm_add_ss(s, _mm_movehdup_ps(s));
		dot = _mm_cvtss_f32(s);

		for (; i < dim; i++)
		{
			float		d = f->vmin + f->delta * (float) code[i];

			dot += q[i] * d;
		}

		return dot;
	}
}

__attribute__((target("avx2,fma")))
float
dann_sq8_estimate_l2_avx2(const uint8_t *code, const DannSQ8Factors *f,
						  const float *q, float q_norm_sq, int dim)
{
	return q_norm_sq - 2.0f * dann_sq8_dot_avx2(code, f, q, dim)
		+ f->deq_norm_sq;
}

#else
float
dann_sq8_dot_avx2(const uint8_t *code, const DannSQ8Factors *f,
				  const float *q, int dim)
{
	return dann_sq8_dot_scalar(code, f, q, dim);
}

float
dann_sq8_estimate_l2_avx2(const uint8_t *code, const DannSQ8Factors *f,
						  const float *q, float q_norm_sq, int dim)
{
	return dann_sq8_estimate_l2_scalar(code, f, q, q_norm_sq, dim);
}
#endif
