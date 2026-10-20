/*-------------------------------------------------------------------------
 * bench/quant_harness.c — 量化器独立测试与基准(纯 C,不依赖 PG)
 *
 * 用法:quant_harness fixture1.txt [fixture2.txt ...]
 *
 * 对拍(golden model,golden/sq8_model.py 生成):
 *   - encode:码逐字节相等;因子 |Δ| ≤ 1e-3 + 1e-5|want|
 *   - estimate / L2:标量与 SIMD 分派实现均对拍,同上容差
 * micro-bench:标量(-O2 自动向量化基线)vs AVX2 vs AVX-512。
 *-------------------------------------------------------------------------*/
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

#include "quant.h"

#define REL_TOL 1e-5
#define ABS_TOL 1e-3

static int	n_pass = 0;
static int	n_fail = 0;

typedef struct Fixture
{
	int			dim;
	int			nv;
	int			nq;
	float	  *vecs;		/* nv * dim */
	uint8_t	   *codes;		/* nv * dim */
	DannSQ8Factors *facts;	/* nv */
	float	  *queries;		/* nq * dim */
	float	  *qnorms;		/* nq */
	float	  *ests;		/* nv * nq */
	float	  *l2s;			/* nv * nq */
} Fixture;

static int
close_enough(double got, double want)
{
	double		d = fabs(got - want);

	return d <= ABS_TOL + REL_TOL * fabs(want);
}

static void
check(int cond, const char *what, int idx)
{
	if (cond)
		n_pass++;
	else
	{
		if (n_fail < 10)
			printf("  FAIL: %s (index %d)\n", what, idx);
		n_fail++;
	}
}

static int
load_fixture(const char *path, Fixture *fx)
{
	FILE	   *fp = fopen(path, "r");
	char		tag[16];
	char		line[256];
	int			vi = 0, ci = 0, fi = 0, qi = 0, qni = 0, ei = 0, li = 0;

	if (fp == NULL)
	{
		printf("cannot open %s\n", path);
		return -1;
	}

	if (fgets(line, sizeof(line), fp) == NULL || line[0] != '#')
	{
		printf("%s: bad header\n", path);
		fclose(fp);
		return -1;
	}
	if (fscanf(fp, "%d %d %d", &fx->dim, &fx->nv, &fx->nq) != 3)
	{
		printf("%s: bad sizes\n", path);
		fclose(fp);
		return -1;
	}

	fx->vecs = malloc(sizeof(float) * fx->nv * fx->dim);
	fx->codes = malloc(sizeof(uint8_t) * fx->nv * fx->dim);
	fx->facts = malloc(sizeof(DannSQ8Factors) * fx->nv);
	fx->queries = malloc(sizeof(float) * fx->nq * fx->dim);
	fx->qnorms = malloc(sizeof(float) * fx->nq);
	fx->ests = malloc(sizeof(float) * fx->nv * fx->nq);
	fx->l2s = malloc(sizeof(float) * fx->nv * fx->nq);
	if (!fx->vecs || !fx->codes || !fx->facts || !fx->queries ||
		!fx->qnorms || !fx->ests || !fx->l2s)
	{
		printf("%s: out of memory\n", path);
		fclose(fp);
		return -1;
	}

	while (fscanf(fp, "%15s", tag) == 1)
	{
		int			i;

		if (strcmp(tag, "V") == 0 && vi < fx->nv)
		{
			for (i = 0; i < fx->dim; i++)
				if (fscanf(fp, "%f", &fx->vecs[vi * fx->dim + i]) != 1)
					goto bad;
			vi++;
		}
		else if (strcmp(tag, "C") == 0 && ci < fx->nv)
		{
			for (i = 0; i < fx->dim; i++)
			{
				int			c;

				if (fscanf(fp, "%d", &c) != 1)
					goto bad;
				fx->codes[ci * fx->dim + i] = (uint8_t) c;
			}
			ci++;
		}
		else if (strcmp(tag, "F") == 0 && fi < fx->nv)
		{
			if (fscanf(fp, "%f %f %f", &fx->facts[fi].vmin,
					   &fx->facts[fi].delta,
					   &fx->facts[fi].deq_norm_sq) != 3)
				goto bad;
			fi++;
		}
		else if (strcmp(tag, "Q") == 0 && qi < fx->nq)
		{
			for (i = 0; i < fx->dim; i++)
				if (fscanf(fp, "%f", &fx->queries[qi * fx->dim + i]) != 1)
					goto bad;
			qi++;
		}
		else if (strcmp(tag, "QN") == 0 && qni < fx->nq)
		{
			if (fscanf(fp, "%f", &fx->qnorms[qni]) != 1)
				goto bad;
			qni++;
		}
		else if (strcmp(tag, "E") == 0 && ei < fx->nv)
		{
			for (i = 0; i < fx->nq; i++)
				if (fscanf(fp, "%f", &fx->ests[ei * fx->nq + i]) != 1)
					goto bad;
			ei++;
		}
		else if (strcmp(tag, "L2") == 0 && li < fx->nv)
		{
			for (i = 0; i < fx->nq; i++)
				if (fscanf(fp, "%f", &fx->l2s[li * fx->nq + i]) != 1)
					goto bad;
			li++;
		}
		else
			goto bad;
	}
	fclose(fp);

	if (vi != fx->nv || ci != fx->nv || fi != fx->nv || qi != fx->nq ||
		qni != fx->nq || ei != fx->nv || li != fx->nv)
	{
		printf("%s: count mismatch V%d C%d F%d Q%d QN%d E%d L2d\n",
			   path, vi, ci, fi, qi, qni, ei);
		return -1;
	}
	return 0;

bad:
	printf("%s: parse error near tag %s\n", path, tag);
	fclose(fp);
	return -1;
}

static void
free_fixture(Fixture *fx)
{
	free(fx->vecs);
	free(fx->codes);
	free(fx->facts);
	free(fx->queries);
	free(fx->qnorms);
	free(fx->ests);
	free(fx->l2s);
}

static void
check_fixture(const char *path, const Fixture *fx)
{
	uint8_t    *codebuf = malloc(fx->dim);
	int			i,
				j;

	printf("== %s (dim=%d, %d vec x %d query)\n", path, fx->dim, fx->nv, fx->nq);

	for (i = 0; i < fx->nv; i++)
	{
		const float *v = fx->vecs + (size_t) i * fx->dim;
		DannSQ8Factors f;

		/* 1) encode 对拍:码逐字节相等,因子容差 */
		dann_sq8_encode(v, fx->dim, codebuf, &f);
		check(memcmp(codebuf, fx->codes + (size_t) i * fx->dim, fx->dim) == 0,
			  "code mismatch", i);
		check(close_enough(f.vmin, fx->facts[i].vmin), "vmin mismatch", i);
		check(close_enough(f.delta, fx->facts[i].delta), "delta mismatch", i);
		check(close_enough(f.deq_norm_sq, fx->facts[i].deq_norm_sq),
			  "deq_norm_sq mismatch", i);

		for (j = 0; j < fx->nq; j++)
		{
			const float *q = fx->queries + (size_t) j * fx->dim;
			float		want_e = fx->ests[i * fx->nq + j];
			float		want_l = fx->l2s[i * fx->nq + j];
			float		got;

			/* 2) estimate 对拍:标量 + 分派(SIMD) */
			got = dann_sq8_estimate_l2_scalar(fx->codes + (size_t) i * fx->dim,
											  &fx->facts[i], q,
											  fx->qnorms[j], fx->dim);
			check(close_enough(got, want_e), "est scalar mismatch",
				  i * fx->nq + j);
			got = dann_sq8_estimate_l2(fx->codes + (size_t) i * fx->dim,
									   &fx->facts[i], q,
									   fx->qnorms[j], fx->dim);
			check(close_enough(got, want_e), "est simd mismatch",
				  i * fx->nq + j);

			/* 3) L2 对拍:标量 + 各 SIMD 实现 */
			got = dann_l2_scalar(q, v, fx->dim);
			check(close_enough(got, want_l), "l2 scalar mismatch",
				  i * fx->nq + j);
			got = dann_l2_avx2(q, v, fx->dim);
			check(close_enough(got, want_l), "l2 avx2 mismatch",
				  i * fx->nq + j);
			got = dann_l2_avx512(q, v, fx->dim);
			check(close_enough(got, want_l), "l2 avx512 mismatch",
				  i * fx->nq + j);
		}
	}

	printf("   cumulative: %d pass, %d fail\n", n_pass, n_fail);
	free(codebuf);
}

/* ---- micro-bench ---- */

static double
now_ns(void)
{
	struct timespec ts;

	clock_gettime(CLOCK_MONOTONIC, &ts);
	return (double) ts.tv_sec * 1e9 + (double) ts.tv_nsec;
}

static volatile float sink;

static uint32_t lcg_state = 0x12345678;

static float
frand(void)
{
	lcg_state = lcg_state * 1664525u + 1013904223u;
	return (float) (lcg_state >> 8) / (float) (1u << 24);
}

static void
bench_l2(int dim, int nvec, int rounds)
{
	float	   *vecs = malloc(sizeof(float) * nvec * dim);
	float	   *q = malloc(sizeof(float) * dim);
	size_t		k;
	int			r;
	double		t0;
	double		ns_per;
	struct
	{
		const char *name;
		DannL2Fn	fn;
	}			variants[3];
	int			vi;

	variants[0].name = "scalar(-O2 autovec)";
	variants[0].fn = dann_l2_scalar;
	variants[1].name = "avx2";
	variants[1].fn = dann_l2_avx2;
	variants[2].name = "avx512";
	variants[2].fn = dann_l2_avx512;

	for (k = 0; k < (size_t) nvec * dim; k++)
		vecs[k] = frand();
	for (k = 0; k < (size_t) dim; k++)
		q[k] = frand();

	printf("-- l2 squared distance, dim=%d, %d vecs x %d rounds\n",
		   dim, nvec, rounds);
	for (vi = 0; vi < 3; vi++)
	{
		float		acc = 0.0f;

		/* 热身(同时防止页面缺页计入) */
		for (k = 0; k < (size_t) nvec; k++)
			acc += variants[vi].fn(q, vecs + k * dim, dim);

		t0 = now_ns();
		for (r = 0; r < rounds; r++)
			for (k = 0; k < (size_t) nvec; k++)
				acc += variants[vi].fn(q, vecs + k * dim, dim);
		ns_per = (now_ns() - t0) / ((double) nvec * rounds);
		sink = acc;
		printf("   %-20s %8.2f ns/call\n", variants[vi].name, ns_per);
	}

	free(vecs);
	free(q);
}

static void
bench_sq8(int dim, int nvec, int rounds)
{
	float	   *vecs = malloc(sizeof(float) * nvec * dim);
	uint8_t    *codes = malloc(sizeof(uint8_t) * nvec * dim);
	DannSQ8Factors *facts = malloc(sizeof(DannSQ8Factors) * nvec);
	float	   *q = malloc(sizeof(float) * dim);
	float		qnorm = 0.0f;
	size_t		k;
	int			r,
				vi;
	double		t0;
	struct
	{
		const char *name;
		DannSQ8EstimateFn fn;
	}			variants[2];

	variants[0].name = "scalar(-O2 autovec)";
	variants[0].fn = dann_sq8_estimate_l2_scalar;
	variants[1].name = "avx2";
	variants[1].fn = dann_sq8_estimate_l2_avx2;

	for (k = 0; k < (size_t) nvec * dim; k++)
		vecs[k] = frand();
	for (k = 0; k < (size_t) dim; k++)
	{
		q[k] = frand();
		qnorm += q[k] * q[k];
	}
	for (k = 0; k < (size_t) nvec; k++)
		dann_sq8_encode(vecs + k * dim, dim, codes + k * dim, &facts[k]);

	printf("-- sq8 estimate, dim=%d, %d vecs x %d rounds\n", dim, nvec, rounds);
	for (vi = 0; vi < 2; vi++)
	{
		float		acc = 0.0f;

		for (k = 0; k < (size_t) nvec; k++)
			acc += variants[vi].fn(codes + k * dim, &facts[k], q, qnorm, dim);

		t0 = now_ns();
		for (r = 0; r < rounds; r++)
			for (k = 0; k < (size_t) nvec; k++)
				acc += variants[vi].fn(codes + k * dim, &facts[k], q,
									   qnorm, dim);
		sink = acc;
		printf("   %-20s %8.2f ns/call\n",
			   variants[vi].name,
			   (now_ns() - t0) / ((double) nvec * rounds));
	}

	free(vecs);
	free(codes);
	free(facts);
	free(q);
}

/* ---- RaBitQ fixture(W7;golden/rabitq_model.py 生成) ---- */

typedef struct RqFixture
{
	int			dim;
	int			padded;
	int			nv;
	int			nq;
	uint32_t	seed;
	float	  *vecs;			/* nv * dim */
	float	  *cens;			/* nv * dim */
	uint8_t    *codes;			/* nv * ceil(padded/8) */
	float	  *factors;			/* nv * 2:dist_c, g */
	float	  *queries;			/* nq * dim */
	float	  *ests;			/* nv * nq(L2 估计) */
	float	  *ips;				/* nv * nq(IP 分数) */
	float	  *l2s;				/* nv * nq(精确 L2) */
} RqFixture;

static int
load_rq_fixture(const char *path, RqFixture *fx)
{
	FILE	   *fp = fopen(path, "r");
	char		tag[16];
	char		line[256];
	int			nbytes;
	int			vi = 0, ci = 0, bi = 0, gi = 0, qi = 0, ei = 0, ii = 0,
				li = 0;

	if (fp == NULL)
	{
		printf("cannot open %s\n", path);
		return -1;
	}
	if (fgets(line, sizeof(line), fp) == NULL || strstr(line, "rabitq") == NULL)
	{
		printf("%s: not a rabitq fixture\n", path);
		fclose(fp);
		return -1;
	}
	if (fscanf(fp, "%d %d %d %d %u", &fx->dim, &fx->padded, &fx->nv,
			   &fx->nq, &fx->seed) != 5)
	{
		printf("%s: bad sizes\n", path);
		fclose(fp);
		return -1;
	}
	nbytes = (fx->padded + 7) / 8;

	fx->vecs = malloc(sizeof(float) * fx->nv * fx->dim);
	fx->cens = malloc(sizeof(float) * fx->nv * fx->dim);
	fx->codes = malloc((size_t) fx->nv * nbytes);
	fx->factors = malloc(sizeof(float) * fx->nv * 2);
	fx->queries = malloc(sizeof(float) * fx->nq * fx->dim);
	fx->ests = malloc(sizeof(float) * fx->nv * fx->nq);
	fx->ips = malloc(sizeof(float) * fx->nv * fx->nq);
	fx->l2s = malloc(sizeof(float) * fx->nv * fx->nq);
	if (!fx->vecs || !fx->cens || !fx->codes || !fx->factors ||
		!fx->queries || !fx->ests || !fx->ips || !fx->l2s)
	{
		printf("%s: out of memory\n", path);
		fclose(fp);
		return -1;
	}

	while (fscanf(fp, "%15s", tag) == 1)
	{
		int			i;

		if (strcmp(tag, "V") == 0 && vi < fx->nv)
		{
			for (i = 0; i < fx->dim; i++)
				if (fscanf(fp, "%f", &fx->vecs[vi * fx->dim + i]) != 1)
					goto bad;
			vi++;
		}
		else if (strcmp(tag, "CEN") == 0 && ci < fx->nv)
		{
			for (i = 0; i < fx->dim; i++)
				if (fscanf(fp, "%f", &fx->cens[ci * fx->dim + i]) != 1)
					goto bad;
			ci++;
		}
		else if (strcmp(tag, "B") == 0 && bi < fx->nv)
		{
			for (i = 0; i < nbytes; i++)
			{
				int			b;

				if (fscanf(fp, "%d", &b) != 1)
					goto bad;
				fx->codes[(size_t) bi * nbytes + i] = (uint8_t) b;
			}
			bi++;
		}
		else if (strcmp(tag, "G") == 0 && gi < fx->nv)
		{
			if (fscanf(fp, "%f %f", &fx->factors[gi * 2],
					   &fx->factors[gi * 2 + 1]) != 2)
				goto bad;
			gi++;
		}
		else if (strcmp(tag, "Q") == 0 && qi < fx->nq)
		{
			for (i = 0; i < fx->dim; i++)
				if (fscanf(fp, "%f", &fx->queries[qi * fx->dim + i]) != 1)
					goto bad;
			qi++;
		}
		else if (strcmp(tag, "E") == 0 && ei < fx->nv)
		{
			for (i = 0; i < fx->nq; i++)
				if (fscanf(fp, "%f", &fx->ests[ei * fx->nq + i]) != 1)
					goto bad;
			ei++;
		}
		else if (strcmp(tag, "EI") == 0 && ii < fx->nv)
		{
			for (i = 0; i < fx->nq; i++)
				if (fscanf(fp, "%f", &fx->ips[ii * fx->nq + i]) != 1)
					goto bad;
			ii++;
		}
		else if (strcmp(tag, "L2") == 0 && li < fx->nv)
		{
			for (i = 0; i < fx->nq; i++)
				if (fscanf(fp, "%f", &fx->l2s[li * fx->nq + i]) != 1)
					goto bad;
			li++;
		}
		else
			goto bad;
	}
	fclose(fp);

	if (vi != fx->nv || ci != fx->nv || bi != fx->nv || gi != fx->nv ||
		qi != fx->nq || ei != fx->nv || ii != fx->nv || li != fx->nv)
	{
		printf("%s: count mismatch V%d CEN%d B%d G%d Q%d E%d EI%d L2%d\n",
			   path, vi, ci, bi, gi, qi, ei, ii, li);
		return -1;
	}
	return 0;

bad:
	printf("%s: parse error near tag %s\n", path, tag);
	fclose(fp);
	return -1;
}

static void
free_rq_fixture(RqFixture *fx)
{
	free(fx->vecs);
	free(fx->cens);
	free(fx->codes);
	free(fx->factors);
	free(fx->queries);
	free(fx->ests);
	free(fx->ips);
	free(fx->l2s);
}

/* 与 quant/rabitq.c 组合式同步(用于分实现单独对拍 ipqr) */
static float
rq_l2_combine(float ipqr, float dist_c, float g, float qc_norm)
{
	float		cos = (g > 1e-6f) ? ipqr / g : 0.0f;

	return dist_c * dist_c + qc_norm * qc_norm
		- 2.0f * dist_c * qc_norm * cos;
}

static void
check_rq_fixture(const char *path, const RqFixture *fx)
{
	int			nbytes = (fx->padded + 7) / 8;
	uint8_t    *codebuf = malloc(nbytes);
	float	   *sign = malloc(sizeof(float) * fx->padded);
	uint32_t   *perm = malloc(sizeof(uint32_t) * fx->padded);
	float	   *scratch = malloc(sizeof(float) * fx->padded * 2);
	float	   *qr = malloc(sizeof(float) * fx->padded);
	int			i,
				j;

	printf("== %s (dim=%d, padded=%d, %d vec x %d query, seed=%u)\n",
		   path, fx->dim, fx->padded, fx->nv, fx->nq, fx->seed);

	dann_rotation_params(fx->seed, fx->padded, sign, perm);

	for (i = 0; i < fx->nv; i++)
	{
		const float *v = fx->vecs + (size_t) i * fx->dim;
		const float *c = fx->cens + (size_t) i * fx->dim;
		float		dc,
				g;
		const uint8_t *want_code = fx->codes + (size_t) i * nbytes;
		float		want_dc = fx->factors[i * 2];
		float		want_g = fx->factors[i * 2 + 1];

		/* 1) encode 对拍:码逐字节相等,因子容差 */
		dann_rabitq_encode(v, c, fx->dim, fx->padded, sign, perm, scratch,
						   codebuf, &dc, &g);
		check(memcmp(codebuf, want_code, nbytes) == 0, "rq code mismatch", i);
		check(close_enough(dc, want_dc), "rq dist_c mismatch", i);
		check(close_enough(g, want_g), "rq g mismatch", i);

		for (j = 0; j < fx->nq; j++)
		{
			const float *q = fx->queries + (size_t) j * fx->dim;
			float		sum_q;
			float		nqc;
			float		ipqr_s,
						ipqr_d,
						ipqr_a2,
						ipqr_a5;
			float		want_e = fx->ests[i * fx->nq + j];
			float		want_i = fx->ips[i * fx->nq + j];
			float		want_l = fx->l2s[i * fx->nq + j];
			float		cos_d;
			float		got;

			nqc = dann_rabitq_prepare_query(q, c, fx->dim, fx->padded,
											sign, perm, scratch, qr, &sum_q);

			/* 2) ipqr 各实现 → 同一组合式,分别对拍 E */
			ipqr_s = dann_rabitq_ipqr_scalar(want_code, qr, sum_q, fx->padded);
			ipqr_d = dann_rabitq_ipqr(want_code, qr, sum_q, fx->padded);
			ipqr_a2 = dann_rabitq_ipqr_avx2(want_code, qr, sum_q, fx->padded);
			ipqr_a5 = dann_rabitq_ipqr_avx512(want_code, qr, sum_q, fx->padded);
			check(close_enough(rq_l2_combine(ipqr_s, dc, g, nqc), want_e),
				  "rq est scalar mismatch", i * fx->nq + j);
			check(close_enough(rq_l2_combine(ipqr_d, dc, g, nqc), want_e),
				  "rq est dispatch mismatch", i * fx->nq + j);
			check(close_enough(rq_l2_combine(ipqr_a2, dc, g, nqc), want_e),
				  "rq est avx2 mismatch", i * fx->nq + j);
			check(close_enough(rq_l2_combine(ipqr_a5, dc, g, nqc), want_e),
				  "rq est avx512 mismatch", i * fx->nq + j);

			/* 3) EI(IP 分数):dist_c·coŝ + <c,q>,coŝ 走分派实现 */
			cos_d = (g > 1e-6f) ? ipqr_d / g : 0.0f;
			got = dc * cos_d + dann_dot_scalar(c, q, fx->dim);
			check(close_enough(got, want_i), "rq ip mismatch", i * fx->nq + j);

			/* 4) L2 精确距离对拍(与 SQ8 路径共用实现) */
			got = dann_l2_scalar(q, v, fx->dim);
			check(close_enough(got, want_l), "l2 scalar mismatch",
				  i * fx->nq + j);
			got = dann_l2_avx512(q, v, fx->dim);
			check(close_enough(got, want_l), "l2 avx512 mismatch",
				  i * fx->nq + j);
		}
	}

	printf("   cumulative: %d pass, %d fail\n", n_pass, n_fail);
	free(codebuf);
	free(sign);
	free(perm);
	free(scratch);
	free(qr);
}

static void
bench_rabitq(int padded_dim, int nvec, int rounds)
{
	int			nbytes = (padded_dim + 7) / 8;
	uint8_t    *codes = malloc((size_t) nvec * nbytes);
	float	   *qr = malloc(sizeof(float) * padded_dim);
	float		sum_q = 0.0f;
	size_t		k;
	int			i,
				r,
				vi;
	double		t0;
	struct
	{
		const char *name;
		DannRabitqIpqrFn fn;
	}			variants[3];

	variants[0].name = "scalar(-O2 autovec)";
	variants[0].fn = dann_rabitq_ipqr_scalar;
	variants[1].name = "avx2";
	variants[1].fn = dann_rabitq_ipqr_avx2;
	variants[2].name = "avx512";
	variants[2].fn = dann_rabitq_ipqr_avx512;

	for (k = 0; k < (size_t) nvec * nbytes; k++)
		codes[k] = (uint8_t) (frand() * 256.0f);
	for (i = 0; i < padded_dim; i++)
	{
		qr[i] = frand() - 0.5f;
		sum_q += qr[i];
	}

	printf("-- rabitq ipqr, padded_dim=%d, %d vecs x %d rounds\n",
		   padded_dim, nvec, rounds);
	for (vi = 0; vi < 3; vi++)
	{
		float		acc = 0.0f;

		for (k = 0; k < (size_t) nvec; k++)
			acc += variants[vi].fn(codes + k * nbytes, qr, sum_q, padded_dim);

		t0 = now_ns();
		for (r = 0; r < rounds; r++)
			for (k = 0; k < (size_t) nvec; k++)
				acc += variants[vi].fn(codes + k * nbytes, qr, sum_q,
									   padded_dim);
		sink = acc;
		printf("   %-20s %8.2f ns/call\n",
			   variants[vi].name,
			   (now_ns() - t0) / ((double) nvec * rounds));
	}

	free(codes);
	free(qr);
}

int
main(int argc, char **argv)
{
	int			i;

	dann_simd_init();

	printf("dispatch: l2=%s, sq8_est=%s, rabitq_ipqr=%s\n",
		   dann_l2_distance == dann_l2_avx512 ? "avx512" :
		   dann_l2_distance == dann_l2_avx2 ? "avx2" : "scalar",
		   dann_sq8_estimate_l2 == dann_sq8_estimate_l2_avx2 ? "avx2" : "scalar",
		   dann_rabitq_ipqr == dann_rabitq_ipqr_avx512 ? "avx512" :
		   dann_rabitq_ipqr == dann_rabitq_ipqr_avx2 ? "avx2" : "scalar");

	if (argc < 2)
	{
		printf("usage: %s fixture.txt [...]\n", argv[0]);
		return 2;
	}

	for (i = 1; i < argc; i++)
	{
		if (strstr(argv[i], "rabitq") != NULL)
		{
			RqFixture	fx;

			if (load_rq_fixture(argv[i], &fx) != 0)
				return 2;
			check_rq_fixture(argv[i], &fx);
			free_rq_fixture(&fx);
		}
		else
		{
			Fixture		fx;

			if (load_fixture(argv[i], &fx) != 0)
				return 2;
			check_fixture(argv[i], &fx);
			free_fixture(&fx);
		}
	}

	printf("\ncorrectness: %d pass, %d fail\n", n_pass, n_fail);

	if (n_fail == 0)
	{
		bench_l2(128, 4096, 2000);
		bench_l2(1024, 2048, 1000);
		bench_sq8(128, 4096, 2000);
		bench_sq8(1024, 2048, 1000);
		bench_rabitq(128, 8192, 2000);
		bench_rabitq(1024, 2048, 1000);
	}

	return n_fail == 0 ? 0 : 1;
}
