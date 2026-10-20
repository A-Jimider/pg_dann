/*-------------------------------------------------------------------------
 * simd/rabitq_ipqr_avx2.c — RaBitQ ⟨x̄,q'⟩ AVX2 实现
 *
 * 每字节码 → 8 个 32bit 整数通道的位测试(cmpeq 得全 1/全 0 掩码)
 * → and_ps 掩码累加。padded_dim 为 2 的幂:≥8 时全字节循环,
 * <8 时全走尾部标量。数学与 dann_rabitq_ipqr_scalar 严格同式。
 *-------------------------------------------------------------------------*/
#include "quant.h"

#include <math.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>

__attribute__((target("avx2")))
float
dann_rabitq_ipqr_avx2(const uint8_t *code, const float *qr,
					  float sum_q, int padded_dim)
{
	__m256		acc = _mm256_setzero_ps();
	const __m256i bitpat = _mm256_setr_epi32(1, 2, 4, 8, 16, 32, 64, 128);
	float		s;
	int			i = 0;

	for (; i + 8 <= padded_dim; i += 8)
	{
		__m256i		cb = _mm256_set1_epi32((int) code[i >> 3]);
		__m256i		t = _mm256_and_si256(cb, bitpat);
		__m256i		m = _mm256_cmpeq_epi32(t, bitpat);
		__m256		qv = _mm256_loadu_ps(qr + i);

		acc = _mm256_add_ps(acc, _mm256_and_ps(qv, _mm256_castsi256_ps(m)));
	}

	{
		__m128		lo = _mm256_castps256_ps128(acc);
		__m128		hi = _mm256_extractf128_ps(acc, 1);
		__m128		s4 = _mm_add_ps(lo, hi);

		s4 = _mm_add_ps(s4, _mm_movehl_ps(s4, s4));
		s4 = _mm_add_ss(s4, _mm_movehdup_ps(s4));
		s = _mm_cvtss_f32(s4);
	}

	for (; i < padded_dim; i++)
		if ((code[i >> 3] >> (i & 7)) & 1)
			s += qr[i];

	return (2.0f * s - sum_q) / sqrtf((float) padded_dim);
}

#else
float
dann_rabitq_ipqr_avx2(const uint8_t *code, const float *qr,
					  float sum_q, int padded_dim)
{
	return dann_rabitq_ipqr_scalar(code, qr, sum_q, padded_dim);
}
#endif
