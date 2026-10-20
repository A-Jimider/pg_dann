/*-------------------------------------------------------------------------
 * simd/dispatch.c — 运行时 SIMD 分派
 *
 * 依据 CPUID 把全局函数指针升级到当前机器支持的最快实现。
 * 幂等;扩展 _PG_init 与独立 harness 的 main 都会调用。
 * x86:AVX2/FMA 兜底,有 AVX-512F+VL 用 AVX-512;ARM(后续)走 NEON。
 *-------------------------------------------------------------------------*/
#include "quant.h"

DannL2Fn	dann_l2_distance = dann_l2_scalar;
DannL2Fn	dann_dot_product = dann_dot_scalar;
DannSQ8EstimateFn dann_sq8_estimate_l2 = dann_sq8_estimate_l2_scalar;
DannSQ8DotFn dann_sq8_dot = dann_sq8_dot_scalar;
DannRabitqIpqrFn dann_rabitq_ipqr = dann_rabitq_ipqr_scalar;

void
dann_simd_init(void)
{
#if defined(__x86_64__) || defined(_M_X64)
	if (__builtin_cpu_supports("avx512f") &&
		__builtin_cpu_supports("avx512vl"))
	{
		dann_l2_distance = dann_l2_avx512;
		dann_dot_product = dann_dot_avx512;
	}
	else if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
	{
		dann_l2_distance = dann_l2_avx2;
		dann_dot_product = dann_dot_avx2;
	}

	/* SQ8 点积/估计的 SIMD 目前只有 AVX2 实现(加宽 + FMA) */
	if (__builtin_cpu_supports("avx2") && __builtin_cpu_supports("fma"))
	{
		dann_sq8_estimate_l2 = dann_sq8_estimate_l2_avx2;
		dann_sq8_dot = dann_sq8_dot_avx2;
	}

	/* RaBitQ ⟨x̄,q'⟩:AVX-512F 掩码累加优先,AVX2 兜底 */
	if (__builtin_cpu_supports("avx512f"))
		dann_rabitq_ipqr = dann_rabitq_ipqr_avx512;
	else if (__builtin_cpu_supports("avx2"))
		dann_rabitq_ipqr = dann_rabitq_ipqr_avx2;
#endif
}
