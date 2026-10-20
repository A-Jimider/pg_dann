/*-------------------------------------------------------------------------
 * simd/l2_avx512.c — L2 平方距离 AVX-512F/VL 实现
 *
 * 512 位主循环 + 掩码尾部处理(AVX-512VL),无标量尾巴。
 *-------------------------------------------------------------------------*/
#include "quant.h"

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

__attribute__((target("avx512f,avx512vl")))
float
dann_l2_avx512(const float *a, const float *b, int dim)
{
	__m512		acc = _mm512_setzero_ps();
	int			i = 0;

	for (; i + 16 <= dim; i += 16)
	{
		__m512		va = _mm512_loadu_ps(a + i);
		__m512		vb = _mm512_loadu_ps(b + i);
		__m512		d = _mm512_sub_ps(va, vb);

		acc = _mm512_fmadd_ps(d, d, acc);
	}

	/* 尾部:不足 16 维用掩码一次处理 */
	if (i < dim)
	{
		__mmask16	m = (__mmask16) ((1u << (dim - i)) - 1);
		__m512		va = _mm512_maskz_loadu_ps(m, a + i);
		__m512		vb = _mm512_maskz_loadu_ps(m, b + i);
		__m512		d = _mm512_sub_ps(va, vb);

		acc = _mm512_fmadd_ps(d, d, acc);
	}

	return _mm512_reduce_add_ps(acc);
}

__attribute__((target("avx512f,avx512vl")))
float
dann_dot_avx512(const float *a, const float *b, int dim)
{
	__m512		acc = _mm512_setzero_ps();
	int			i = 0;

	for (; i + 16 <= dim; i += 16)
	{
		__m512		va = _mm512_loadu_ps(a + i);
		__m512		vb = _mm512_loadu_ps(b + i);

		acc = _mm512_fmadd_ps(va, vb, acc);
	}

	if (i < dim)
	{
		__mmask16	m = (__mmask16) ((1u << (dim - i)) - 1);
		__m512		va = _mm512_maskz_loadu_ps(m, a + i);
		__m512		vb = _mm512_maskz_loadu_ps(m, b + i);

		acc = _mm512_fmadd_ps(va, vb, acc);
	}

	return _mm512_reduce_add_ps(acc);
}

#else
float
dann_l2_avx512(const float *a, const float *b, int dim)
{
	return dann_l2_scalar(a, b, dim);
}

float
dann_dot_avx512(const float *a, const float *b, int dim)
{
	return dann_dot_scalar(a, b, dim);
}
#endif
