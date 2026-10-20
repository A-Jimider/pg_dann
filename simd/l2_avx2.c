/*-------------------------------------------------------------------------
 * simd/l2_avx2.c — L2 平方距离 AVX2+FMA 实现
 *
 * 用 __attribute__((target)) 而非 -mavx2 编译开关(PG pg_crc32c_sse42.c
 * 同款),单 TU 混合编译,运行时由 dispatch.c 按 CPUID 选择。
 *-------------------------------------------------------------------------*/
#include "quant.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

__attribute__((target("avx2,fma")))
float
dann_l2_avx2(const float *a, const float *b, int dim)
{
	__m256		acc = _mm256_setzero_ps();
	int			i = 0;

	for (; i + 8 <= dim; i += 8)
	{
		__m256		va = _mm256_loadu_ps(a + i);
		__m256		vb = _mm256_loadu_ps(b + i);
		__m256		d = _mm256_sub_ps(va, vb);

		acc = _mm256_fmadd_ps(d, d, acc);
	}

	/* 水平求和 */
	{
		__m128		lo = _mm256_castps256_ps128(acc);
		__m128		hi = _mm256_extractf128_ps(acc, 1);
		__m128		s = _mm_add_ps(lo, hi);
		float		sum;

		s = _mm_add_ps(s, _mm_movehl_ps(s, s));
		s = _mm_add_ss(s, _mm_movehdup_ps(s));
		sum = _mm_cvtss_f32(s);

		for (; i < dim; i++)
		{
			float		d = a[i] - b[i];

			sum += d * d;
		}
		return sum;
	}
}

__attribute__((target("avx2,fma")))
float
dann_dot_avx2(const float *a, const float *b, int dim)
{
	__m256		acc = _mm256_setzero_ps();
	int			i = 0;

	for (; i + 8 <= dim; i += 8)
	{
		__m256		va = _mm256_loadu_ps(a + i);
		__m256		vb = _mm256_loadu_ps(b + i);

		acc = _mm256_fmadd_ps(va, vb, acc);
	}

	{
		__m128		lo = _mm256_castps256_ps128(acc);
		__m128		hi = _mm256_extractf128_ps(acc, 1);
		__m128		s = _mm_add_ps(lo, hi);
		float		sum;

		s = _mm_add_ps(s, _mm_movehl_ps(s, s));
		s = _mm_add_ss(s, _mm_movehdup_ps(s));
		sum = _mm_cvtss_f32(s);

		for (; i < dim; i++)
			sum += a[i] * b[i];
		return sum;
	}
}

#else
float
dann_l2_avx2(const float *a, const float *b, int dim)
{
	return dann_l2_scalar(a, b, dim);
}

float
dann_dot_avx2(const float *a, const float *b, int dim)
{
	return dann_dot_scalar(a, b, dim);
}
#endif