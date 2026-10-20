#!/usr/bin/env python3
"""pg_dann SQ8 golden model —— 生成对拍夹具(golden fixtures)。

与 C 实现的约定(quant/quant.h):
  code[i] = floor((v[i]-vmin) * (1/delta) + 0.5), clip [0,255]   (float32)
  est     = ||q||^2 - 2*<q,deq> + ||deq||^2                       (float32)

所有算术用 float32,与 C 端逐步同构;夹具对拍的容差(1e-5)只用于
吸收浮点结合序差异(标量顺序求和 vs SIMD 并行归约)。

用法:.venv/bin/python golden/sq8_model.py
输出:golden/fixtures/sq8_d{dim}.txt
"""
import os
import numpy as np

DIMS = [4, 128, 1001]       # 极小维 / 16 整除 / 奇数尾巴(测 SIMD 尾部路径)
N_VEC = 64
N_QUERY = 8
SEED = 42
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

F32 = np.float32


def gen(dim, rng):
    """混合分布:一半均匀(0~10),一半标准正态;末尾一个常数向量(delta=0 边界)。"""
    v = rng.standard_normal((N_VEC, dim)).astype(F32)
    v[: N_VEC // 2] = rng.random((N_VEC // 2, dim), dtype=F32) * F32(10.0)
    v[N_VEC - 1] = F32(3.25)
    q = rng.standard_normal((N_QUERY, dim)).astype(F32)
    return v, q


def sq8_encode(v):
    """与 quant/sq8.c 逐步同构的 float32 编码。"""
    vmin = F32(v.min())
    vmax = F32(v.max())
    delta = (vmax - vmin) / F32(255.0) if vmax > vmin else F32(0.0)
    if delta > 0:
        inv = F32(1.0) / delta
        code = np.floor((v - vmin) * inv + F32(0.5)).clip(0, 255).astype(np.uint8)
    else:
        code = np.zeros(v.shape, dtype=np.uint8)
    deq = vmin + delta * code.astype(F32)
    deq_norm_sq = F32(np.sum(deq * deq, dtype=F32))
    return code, vmin, delta, deq_norm_sq


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    for dim in DIMS:
        vecs, qs = gen(dim, rng)
        path = os.path.join(OUTDIR, f"sq8_d{dim}.txt")
        with open(path, "w") as f:
            f.write("# dann sq8 golden fixture v1\n")
            f.write(f"{dim} {N_VEC} {N_QUERY}\n")

            codes, facts = [], []
            for v in vecs:
                code, vmin, delta, dns = sq8_encode(v)
                codes.append(code)
                facts.append((vmin, delta, dns))
                f.write("V " + " ".join("%.9e" % x for x in v) + "\n")
            for code in codes:
                f.write("C " + " ".join(str(int(c)) for c in code) + "\n")
            for vmin, delta, dns in facts:
                f.write("F %.9e %.9e %.9e\n" % (vmin, delta, dns))
            for q in qs:
                f.write("Q " + " ".join("%.9e" % x for x in q) + "\n")
            for q in qs:
                qn = F32(np.sum(q * q, dtype=F32))
                f.write("QN %.9e\n" % qn)
            for i, v in enumerate(vecs):
                vmin, delta, dns = facts[i]
                deq = vmin + delta * codes[i].astype(F32)
                ests = []
                for q in qs:
                    qn = F32(np.sum(q * q, dtype=F32))
                    dot = F32(np.dot(q, deq))
                    ests.append(qn - F32(2.0) * dot + dns)
                f.write("E " + " ".join("%.9e" % e for e in ests) + "\n")
            for v in vecs:
                l2s = []
                for q in qs:
                    d = q - v
                    l2s.append(F32(np.sum(d * d, dtype=F32)))
                f.write("L2 " + " ".join("%.9e" % e for e in l2s) + "\n")

        print(f"wrote {path} (dim={dim}, {N_VEC} vec x {N_QUERY} query)")


if __name__ == "__main__":
    main()
