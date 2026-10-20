/*-------------------------------------------------------------------------
 * simd/rabitq_ipqr_avx512.c — RaBitQ ⟨x̄,q'⟩ AVX-512 实现
 *
 * 码的 16bit 组直接作 __mmask16,_mm512_mask_add_ps 掩码累加——天然契合
 * (DESIGN.md §4.2 实现要点)。padded_dim 为 2 的幂:≥16 时全组循环,
 * <16 时全走尾部标量。
 *-------------------------------------------------------------------------*/
#include "quant.h"

#include <math.h>

#if defined(__x86_64__) || defined(_M_X64)
#include <immintrin.h>
#include <string.h>

__attribute__((target("avx512f")))
float
dann_rabitq_ipqr_avx512(const uint8_t *code, const float *qr,
						float sum_q, int padded_dim)
{
	__m512		acc = _mm512_setzero_ps();
	float		s;
	int			i = 0;

	for (; i + 16 <= padded_dim; i += 16)
	{
		uint16_t	m;				/* 16 位码,LSB 优先对应 lane 序号 */
		__m512		qv;

		memcpy(&m, code + (i >> 3), sizeof(m));		/* 免别名争议,编译为单次加载 */
		qv = _mm512_loadu_ps(qr + i);
		acc = _mm512_mask_add_ps(acc, (__mmask16) m, acc, qv);
	}

	s = _mm512_reduce_add_ps(acc);

	for (; i < padded_dim; i++)
		if ((code[i >> 3] >> (i & 7)) & 1)
			s += qr[i];

	return (2.0f * s - sum_q) / sqrtf((float) padded_dim);
}

#else
float
dann_rabitq_ipqr_avx512(const uint8_t *code, const float *qr,
						float sum_q, int padded_dim)
{
	return dann_rabitq_ipqr_scalar(code, qr, sum_q, padded_dim);
}
#endif
