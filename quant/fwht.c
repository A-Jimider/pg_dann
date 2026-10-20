/*-------------------------------------------------------------------------
 * quant/fwht.c — 旋转 P 的构造与施加(纯 C,不依赖 PostgreSQL)
 *
 * P = 随机置换 ∘ FWHT ∘ 随机符号翻转(正交,1/√n 归一化)。
 * sign/perm 由 SplitMix32(seed) 派生,规范与 golden/rabitq_model.py
 * 逐位一致——夹具只携带 seed,两端各自重新生成参数。
 *-------------------------------------------------------------------------*/
#include "quant.h"

#include <math.h>

uint32_t
dann_splitmix32(uint32_t *state)
{
	uint32_t	z;

	*state += 0x9E3779B9u;
	z = *state;
	z = (z ^ (z >> 16)) * 0x21F0AAADu;
	z = (z ^ (z >> 15)) * 0x735A2D97u;
	z = z ^ (z >> 15);
	return z;
}

void
dann_rotation_params(uint32_t seed, int n, float *sign, uint32_t *perm)
{
	uint32_t	state = seed;
	int			i;

	/* 先 sign:n 次抽取,bit0 = 1 → +1 否则 -1 */
	for (i = 0; i < n; i++)
		sign[i] = (dann_splitmix32(&state) & 1u) ? 1.0f : -1.0f;

	/* 后 perm:恒等数组 + Fisher-Yates(i 从 n-1 降到 1) */
	for (i = 0; i < n; i++)
		perm[i] = (uint32_t) i;
	for (i = n - 1; i > 0; i--)
	{
		uint32_t	j = dann_splitmix32(&state) % (uint32_t) (i + 1);
		uint32_t	tmp = perm[i];

		perm[i] = perm[j];
		perm[j] = tmp;
	}
}

/* 归一化 FWHT:迭代蝶形,末了整体 1/√n */
void
dann_fwht(float *x, int n)
{
	int			h;

	for (h = 1; h < n; h *= 2)
	{
		int			i;

		for (i = 0; i < n; i += 2 * h)
		{
			int			j;

			for (j = i; j < i + h; j++)
			{
				float		a = x[j];
				float		b = x[j + h];

				x[j] = a + b;
				x[j + h] = a - b;
			}
		}
	}

	{
		float		rn = sqrtf((float) n);
		int			i;

		/* 与 golden 模型逐位对齐:除法而非倒数乘法(1/√n  reciprocal 会差 1ulp,
		 * 符号量化对近零坐标的位判定敏感) */
		for (i = 0; i < n; i++)
			x[i] /= rn;
	}
}

void
dann_rotate(float *in, float *out, const float *sign, const uint32_t *perm,
			int n)
{
	int			i;

	for (i = 0; i < n; i++)
		in[i] *= sign[i];
	dann_fwht(in, n);
	for (i = 0; i < n; i++)
		out[i] = in[perm[i]];
}
