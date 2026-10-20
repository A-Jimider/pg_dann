#!/usr/bin/env python3
"""pg_dann RaBitQ golden model —— 公式锚点 + 对拍夹具(golden fixtures)。

数学约定(DESIGN.md §4.2,与 C 端 quant/rabitq.c 逐步同构):
  N = padded_dim(补零到 2 的幂)
  旋转 P = 随机置换 ∘ FWHT ∘ 随机符号翻转(正交,1/√N 归一化)
  编码:o_c = o - c;dist_c = ||o_c||;y = P·o_c / dist_c
       x̄_b[i] = (y_i >= 0);x̄ = (2·x̄_b - 1)/√N
       g = ⟨x̄, y⟩            (随 entry 落盘:dist_c, g, dist_c^2)
  估计(非对称,查询全精度):
       q_c = q - c;nqc = ||q_c||;q' = P·q_c / nqc
       ⟨x̄, q'⟩ = (2·Σ_{bit=1} q'_i - Σ_all q'_i) / √N
       coŝ = ⟨x̄, q'⟩ / g     (g ≤ 1e-6 时取 0:退化为只用范数项)
       L2: d̂² = dist_c² + nqc² - 2·dist_c·nqc·coŝ
       IP: score = dist_c·coŝ + ⟨c, q⟩(cosine 归一化后同式)

确定性来源只有一个:metapage 的 rotation_seed。sign/perm 由 SplitMix32
派生,C 端(quant/fwht.c)按同一规范重新生成,夹具只携带 seed:
  sign:先取 N 次,bit0 = 1 → +1 否则 -1
  perm:恒等数组后 Fisher-Yates,i 从 N-1 降到 1,j = next() % (i+1)

SplitMix32(spec,与 C 完全一致):
  state += 0x9E3779B9
  z = state; z = (z ^ (z>>16)) * 0x21f0aaad
  z = (z ^ (z>>15)) * 0x735a2d97; z ^= z >> 15; 返回 z

码位约定:第 i 维 = code[i>>3] 的第 (i&7) 位(LSB 优先)。
所有算术 float32;夹具容差 1e-5 只吸收浮点结合序差异。

用法:.venv/bin/python golden/rabitq_model.py
输出:golden/fixtures/rabitq_d{dim}.txt
"""
import os
import numpy as np

DIMS = [4, 64, 100, 128, 1001]    # 极小维(标量路径)/ 2 幂 / 非 2 幂补零
N_VEC = 64
N_QUERY = 8
SEED = 42
ROTATION_SEED = 0xDAB007
OUTDIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "fixtures")

F32 = np.float32
U32 = np.uint32


class SplitMix32:
    """与 C 端 dann_splitmix32 逐位一致的 PRNG。"""

    def __init__(self, seed):
        self.state = U32(seed)

    def next(self):
        with np.errstate(over='ignore'):    # uint32 回绕是 spec 的一部分
            self.state = U32(self.state + U32(0x9E3779B9))
            z = self.state
            z = U32((z ^ (z >> U32(16))) * U32(0x21F0AAAD))
            z = U32((z ^ (z >> U32(15))) * U32(0x735A2D97))
            z = z ^ (z >> U32(15))
        return z


def rotation_params(seed, n):
    """sign[n] ∈ {±1}(float32),perm[n](uint32)。生成顺序固定:先 sign 后 perm。"""
    rng = SplitMix32(seed)
    sign = np.where(np.array([rng.next() & U32(1) for _ in range(n)]) != 0,
                    F32(1.0), F32(-1.0)).astype(F32)
    perm = np.arange(n, dtype=np.uint32)
    for i in range(n - 1, 0, -1):
        j = int(rng.next() % U32(i + 1))
        perm[i], perm[j] = perm[j], perm[i]
    return sign, perm


def fwht(x):
    """原位归一化 FWHT(n 须为 2 的幂),float32。"""
    n = len(x)
    h = 1
    while h < n:
        for i in range(0, n, 2 * h):
            for j in range(i, i + h):
                a = x[j]
                b = x[j + h]
                x[j] = a + b
                x[j + h] = a - b
        h *= 2
    x /= F32(np.sqrt(F32(n)))
    return x


def rotate(x, sign, perm):
    """P·x:补零到 N(调用方保证)→ 符号翻转 → FWHT → 置换。"""
    t = (x * sign).astype(F32)
    fwht(t)
    return t[perm].astype(F32)


def next_pow2(d):
    n = 1
    while n < d:
        n *= 2
    return n


def pad(x, n):
    if len(x) == n:
        return x.astype(F32)
    out = np.zeros(n, dtype=F32)
    out[: len(x)] = x
    return out


def encode(v, c, sign, perm):
    """返回 (code uint8[ceil(N/8)], dist_c, g)。"""
    n = len(sign)
    nbytes = (n + 7) // 8
    oc = pad((v - c).astype(F32), n)
    dist_c = F32(np.sqrt(np.sum(oc * oc, dtype=F32)))
    if dist_c <= F32(0.0):
        return np.zeros(nbytes, dtype=np.uint8), dist_c, F32(0.0)
    y = rotate(oc, sign, perm) / dist_c
    bits = (y >= F32(0.0))
    idx = np.flatnonzero(bits)
    code = np.zeros(nbytes, dtype=np.uint8)
    if len(idx) > 0:
        # 注意:重复下标的 |= 必须用 .at,否则 fancy indexing 只有最后一写生效
        np.bitwise_or.at(code, idx >> 3,
                         (np.uint8(1) << (idx & 7).astype(np.uint8)))
    g = F32((F32(2.0) * np.sum(y[bits], dtype=F32) - np.sum(y, dtype=F32))
            / np.sqrt(F32(n)))
    return code, dist_c, g


def estimate(code, g, dist_c, q, c, sign, perm):
    """返回 (l2_est, ip_score)。"""
    n = len(sign)
    qc = pad((q - c).astype(F32), n)
    nqc = F32(np.sqrt(np.sum(qc * qc, dtype=F32)))
    if nqc <= F32(0.0):
        qr = np.zeros(n, dtype=F32)
    else:
        qr = rotate(qc, sign, perm) / nqc
    bits = (code.repeat(8) >> np.tile(np.arange(8, dtype=np.uint8), len(code))) & 1
    bits = bits.astype(bool)[:n]            # 末尾字节的高位补零不计
    s = F32(np.sum(qr[bits], dtype=F32))
    ipqr = F32((F32(2.0) * s - np.sum(qr, dtype=F32)) / np.sqrt(F32(n)))
    cos = F32(ipqr / g) if g > F32(1e-6) else F32(0.0)
    l2 = F32(dist_c * dist_c + nqc * nqc
             - F32(2.0) * dist_c * nqc * cos)
    ip = F32(dist_c * cos + np.sum(c * q, dtype=F32))
    return l2, ip


def gen(dim, rng):
    """混合分布向量 + 每向量一个随机质心(靠近向量,模拟聚类内) + 查询。"""
    v = rng.standard_normal((N_VEC, dim)).astype(F32)
    v[: N_VEC // 2] = rng.random((N_VEC // 2, dim), dtype=F32) * F32(10.0)
    v[N_VEC - 1] = F32(3.25)          # 常数向量边界
    cen = v + rng.standard_normal((N_VEC, dim)).astype(F32) * F32(0.5)
    cen[N_VEC - 1] = v[N_VEC - 1]     # dist_c = 0 边界(o == c)
    q = rng.standard_normal((N_QUERY, dim)).astype(F32)
    return v, cen, q


def main():
    os.makedirs(OUTDIR, exist_ok=True)
    rng = np.random.default_rng(SEED)

    for dim in DIMS:
        n = next_pow2(dim)
        sign, perm = rotation_params(ROTATION_SEED, n)
        vecs, cens, qs = gen(dim, rng)
        path = os.path.join(OUTDIR, f"rabitq_d{dim}.txt")
        with open(path, "w") as f:
            f.write("# dann rabitq golden fixture v1\n")
            f.write(f"{dim} {n} {N_VEC} {N_QUERY} {ROTATION_SEED}\n")
            for v in vecs:
                f.write("V " + " ".join("%.9e" % x for x in v) + "\n")
            for c in cens:
                f.write("CEN " + " ".join("%.9e" % x for x in c) + "\n")
            codes = []
            for i, v in enumerate(vecs):
                code, dist_c, g = encode(v, cens[i], sign, perm)
                codes.append((code, dist_c, g))
                f.write("B " + " ".join(str(int(b)) for b in code) + "\n")
                f.write("G %.9e %.9e\n" % (dist_c, g))
            for q in qs:
                f.write("Q " + " ".join("%.9e" % x for x in q) + "\n")
            for i in range(N_VEC):
                ests = []
                for q in qs:
                    l2, _ = estimate(codes[i][0], codes[i][2], codes[i][1],
                                     q, cens[i], sign, perm)
                    ests.append(l2)
                f.write("E " + " ".join("%.9e" % e for e in ests) + "\n")
            for i in range(N_VEC):
                ips = []
                for q in qs:
                    _, ip = estimate(codes[i][0], codes[i][2], codes[i][1],
                                     q, cens[i], sign, perm)
                    ips.append(ip)
                f.write("EI " + " ".join("%.9e" % e for e in ips) + "\n")
            for v in vecs:
                l2s = []
                for q in qs:
                    d = (q - v).astype(F32)
                    l2s.append(F32(np.sum(d * d, dtype=F32)))
                f.write("L2 " + " ".join("%.9e" % e for e in l2s) + "\n")
        print(f"wrote {path} (dim={dim}, padded={n}, {N_VEC} vec x {N_QUERY} query)")


if __name__ == "__main__":
    main()
