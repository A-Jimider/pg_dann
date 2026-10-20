/*-------------------------------------------------------------------------
 * quant/quant.h — 量化器接口(纯 C,不依赖 PostgreSQL 头文件)
 *
 * 设计要点(DESIGN.md §2.2-2):SQ8 与 RaBitQ 实现同一组语义操作,
 * 构建期 encode,查询期 estimate。量化器代码保持 PG 无关(pure C),
 * 以便独立对拍(golden model)与 micro-bench;W4 才被链入扩展。
 *
 * SQ8(标量量化,8 bit/维):per-vector 仿射映射
 *   deq[i] = vmin + delta * code[i],  code[i] ∈ [0,255]
 * 估计式(不反量化查询,展开平方):
 *   ||q - deq||^2 = ||q||^2 - 2*<q,deq> + ||deq||^2
 * 其中 ||deq||^2 在编码期算好存入 factors,查询期热路径只剩 <q,deq>。
 *-------------------------------------------------------------------------*/
#ifndef PG_DANN_QUANT_H
#define PG_DANN_QUANT_H

#include <stdint.h>

/* ---- SQ8 ---- */

/* 每向量校正因子(编码期产生,随 posting entry 落盘) */
typedef struct DannSQ8Factors
{
	float		vmin;
	float		delta;			/* (vmax-vmin)/255;vmax==vmin 时为 0 */
	float		deq_norm_sq;	/* ||dequant(code)||^2 */
} DannSQ8Factors;

/*
 * dann_sq8_encode — 把 v[dim] 编码为 code[dim](uint8)+ factors。
 * 舍入约定:floor(x + 0.5)(与 golden/sq8_model.py 一致,对拍依赖此约定)。
 */
void		dann_sq8_encode(const float *v, int dim, uint8_t *code,
							DannSQ8Factors *factors);

/*
 * dann_sq8_estimate_l2_* — 估计 ||q - v||^2。
 * q_norm_sq = ||q||^2 由调用方预算(每查询一次);标量/SIMD 双实现,
 * 数学上严格同式,允许浮点结合序差异(对拍容差 1e-5)。
 *
 * dann_sq8_dot_* — 同一热循环,只返回 <q, deq>(IP/cosine 度量用):
 *   L2:     q_norm_sq - 2*dot + deq_norm_sq
 *   IP:     -dot
 *   cosine: 归一化后同 L2 式
 */
float		dann_sq8_dot_scalar(const uint8_t *code, const DannSQ8Factors *f,
								const float *q, int dim);
float		dann_sq8_dot_avx2(const uint8_t *code, const DannSQ8Factors *f,
							  const float *q, int dim);
float		dann_sq8_estimate_l2_scalar(const uint8_t *code,
										const DannSQ8Factors *f,
										const float *q, float q_norm_sq,
										int dim);
float		dann_sq8_estimate_l2_avx2(const uint8_t *code,
									  const DannSQ8Factors *f,
									  const float *q, float q_norm_sq,
									  int dim);

/* ---- L2 距离(k-means / 质心扫描 / rerank 用) ---- */
float		dann_l2_scalar(const float *a, const float *b, int dim);
float		dann_l2_avx2(const float *a, const float *b, int dim);
float		dann_l2_avx512(const float *a, const float *b, int dim);

/* ---- 内积(IP 度量的质心分配用) ---- */
float		dann_dot_scalar(const float *a, const float *b, int dim);
float		dann_dot_avx2(const float *a, const float *b, int dim);
float		dann_dot_avx512(const float *a, const float *b, int dim);

/* ---- RaBitQ(W7;数学约定见 golden/rabitq_model.py 头注) ----
 *
 * 旋转 P = 置换 ∘ FWHT ∘ 符号翻转,由 SplitMix32(seed) 确定性派生,
 * C 与 golden model 逐位同规范:
 *   sign:先取 n 次,bit0=1 → +1 否则 -1;
 *   perm:恒等数组后 Fisher-Yates,i 从 n-1 降到 1,j = next() % (i+1)。
 * 码位:第 i 维 = code[i>>3] 的第 (i&7) 位(LSB 优先);码长 ceil(n/8)。
 */

/* 每向量校正因子(编码期产生,随 posting entry 落盘) */
typedef struct DannRabitqFactors
{
	float		dist_c;			/* ||o - c|| */
	float		g;				/* <x̄, y> */
	float		dist_c2;		/* dist_c^2,查询期免重算 */
} DannRabitqFactors;

/* SplitMix32:golden 模型同规范,C/numpy 两端逐位一致 */
uint32_t	dann_splitmix32(uint32_t *state);

/* 旋转参数生成;n = padded_dim(2 的幂);调用方出内存 */
void		dann_rotation_params(uint32_t seed, int n, float *sign,
								 uint32_t *perm);

/* 归一化(1/√n)FWHT,n 为 2 的幂 */
void		dann_fwht(float *x, int n);

/*
 * P·x:in 已补零到 n;计算消耗 in(内容被改写),结果写入 out(in != out)。
 */
void		dann_rotate(float *in, float *out,
						const float *sign, const uint32_t *perm, int n);

/*
 * RaBitQ 编码:v、c 为 dim 维;scratch ≥ 2*padded_dim;
 * code 长 ceil(padded_dim/8);输出 dist_c 与 g。
 * dist_c == 0(o == c)时 code 全零、g = 0(估计端走 g 保护)。
 */
void		dann_rabitq_encode(const float *v, const float *c,
							   int dim, int padded_dim,
							   const float *sign, const uint32_t *perm,
							   float *scratch,
							   uint8_t *code, float *dist_c_out, float *g_out);

/*
 * 查询侧每 list 预处理:qr = P(q-c)/||q-c||(padded_dim 维),
 * 返回 ||q-c||,sum_q = Σqr[i] 经指针输出;scratch ≥ 2*padded_dim。
 * ||q-c|| == 0 时 qr 全零、sum_q = 0(估计退化为 dist_c²,恰为精确值)。
 */
float		dann_rabitq_prepare_query(const float *q, const float *c,
									  int dim, int padded_dim,
									  const float *sign, const uint32_t *perm,
									  float *scratch, float *qr,
									  float *sum_q_out);

/*
 * ⟨x̄, q'⟩ = (2·Σ_{bit=1} q'_i - sum_q) / √padded_dim —— 条目热循环。
 * 标量 + AVX2(字节掩码扩展)+ AVX-512(mask_add);运行时 dispatch。
 */
float		dann_rabitq_ipqr_scalar(const uint8_t *code, const float *qr,
									float sum_q, int padded_dim);
float		dann_rabitq_ipqr_avx2(const uint8_t *code, const float *qr,
								  float sum_q, int padded_dim);
float		dann_rabitq_ipqr_avx512(const uint8_t *code, const float *qr,
									float sum_q, int padded_dim);

/* L2 估计:d̂² = dist_c² + ||q-c||² - 2·dist_c·||q-c||·coŝ(g 保护:g≤1e-6 → coŝ=0) */
float		dann_rabitq_estimate_l2(const uint8_t *code, const DannRabitqFactors *f,
									const float *qr, float sum_q, float qc_norm,
									int padded_dim);
/* IP/cosine:返回 coŝ(直接用,§4.2);作距离用取负 */
float		dann_rabitq_estimate_cos(const uint8_t *code, const DannRabitqFactors *f,
									 const float *qr, float sum_q, int padded_dim);


/* ---- 运行时 SIMD 分派(simd/dispatch.c) ---- */
typedef float (*DannL2Fn) (const float *, const float *, int);
typedef float (*DannSQ8EstimateFn) (const uint8_t *, const DannSQ8Factors *,
									const float *, float, int);
typedef float (*DannSQ8DotFn) (const uint8_t *, const DannSQ8Factors *,
							   const float *, int);
typedef float (*DannRabitqIpqrFn) (const uint8_t *, const float *, float, int);

extern DannL2Fn dann_l2_distance;			/* 初始为 scalar,dann_simd_init 升级 */
extern DannL2Fn dann_dot_product;
extern DannSQ8EstimateFn dann_sq8_estimate_l2;
extern DannSQ8DotFn dann_sq8_dot;
extern DannRabitqIpqrFn dann_rabitq_ipqr;

void		dann_simd_init(void);			/* cpuid 分派;幂等 */

#endif							/* PG_DANN_QUANT_H */
