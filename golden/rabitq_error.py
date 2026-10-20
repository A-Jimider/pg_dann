#!/usr/bin/env python3
"""RaBitQ 误差界实测(DESIGN.md §4.2 工程纪律:golden model 先行)。

逐 dim 实测:
  (a) cos 估计子的实测标准差 σ_cos,验证 σ_cos ∝ 1/√N(论文误差界形状),
      拟合常数 κ = σ_cos·√N;
  (b) 同批数据上 SQ8 的 L2 相对误差(分辨率有界,不随 dim 衰减)作对照。
产出 golden/rabitq_error_curve.csv,供 auto 阈值(dim<128 → SQ8)复核与
文章绘图用。

用法:.venv/bin/python golden/rabitq_error.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rabitq_model import (SplitMix32, rotation_params, rotate, encode,
                          next_pow2, pad, F32, U32)

DIMS = [16, 32, 64, 128, 256, 512, 1024]
N_PAIRS = 400
SEED = 11
ROTATION_SEED = 0xDAB007
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                   "rabitq_error_curve.csv")


def sq8_rel_err(v, c, q):
    """SQ8 估计 ||q-deq||^2 的相对误差(golden/sq8_model.py 同式)。"""
    vmin = F32(v.min())
    vmax = F32(v.max())
    delta = (vmax - vmin) / F32(255.0) if vmax > vmin else F32(0.0)
    if delta > 0:
        inv = F32(1.0) / delta
        code = np.floor((v - vmin) * inv + F32(0.5)).clip(0, 255).astype(np.uint8)
    else:
        code = np.zeros(v.shape, dtype=np.uint8)
    deq = vmin + delta * code.astype(F32)
    true = F32(np.sum((q - v).astype(F32) ** 2, dtype=F32))
    est = F32(np.sum((q - deq).astype(F32) ** 2, dtype=F32))
    if true <= F32(1e-9):
        return 0.0
    return abs(float(est) - float(true)) / float(true)


def main():
    rng = np.random.default_rng(SEED)
    rows = []
    print(f"{'dim':>5} {'sigma_cos':>10} {'kappa=s*sqrt(N)':>15} "
          f"{'rabitq_rel':>10} {'sq8_rel':>9}")
    for dim in DIMS:
        n = next_pow2(dim)
        sign, perm = rotation_params(ROTATION_SEED, n)
        cos_errs, rq_rels, sq_rels = [], [], []
        for _ in range(N_PAIRS):
            v = rng.standard_normal(dim).astype(F32)
            c = v + rng.standard_normal(dim).astype(F32) * F32(0.5)
            q = rng.standard_normal(dim).astype(F32)

            code, dc, g = encode(v, c, sign, perm)
            qc = pad((q - c).astype(F32), n)
            nqc = F32(np.sqrt(np.sum(qc * qc, dtype=F32)))
            qr = rotate(qc, sign, perm) / nqc
            bits = (code.repeat(8)
                    >> np.tile(np.arange(8, dtype=np.uint8), len(code))) & 1
            bits = bits.astype(bool)[:n]
            ipqr = F32((F32(2.0) * np.sum(qr[bits], dtype=F32)
                        - np.sum(qr, dtype=F32)) / np.sqrt(F32(n)))
            cos_est = float(ipqr) / float(g)
            oc = pad((v - c).astype(F32), n)
            cos_true = float(np.sum(oc * qc)) / (float(dc) * float(nqc))
            cos_errs.append(cos_est - cos_true)

            true_l2 = float(np.sum((q - v).astype(F32) ** 2, dtype=F32))
            est_l2 = float(dc * dc + nqc * nqc
                           - F32(2.0) * dc * nqc * F32(cos_est))
            if true_l2 > 1e-9:
                rq_rels.append(abs(est_l2 - true_l2) / true_l2)
            sq_rels.append(sq8_rel_err(v, c, q))

        s = float(np.std(cos_errs))
        row = (dim, n, s, s * np.sqrt(n),
               float(np.median(rq_rels)), float(np.median(sq_rels)))
        rows.append(row)
        print(f"{dim:>5} {row[2]:>10.5f} {row[3]:>15.4f} "
              f"{row[4]:>10.5f} {row[5]:>9.5f}")

    with open(OUT, "w") as f:
        f.write("dim,padded_dim,sigma_cos,kappa,rabitq_rel_median,sq8_rel_median\n")
        for r in rows:
            f.write(f"{r[0]},{r[1]},{r[2]:.6f},{r[3]:.6f},{r[4]:.6f},{r[5]:.6f}\n")
    print(f"\nwrote {OUT}")
    print("结论预期:kappa 近似恒定 ⇒ σ_cos ∝ 1/√N(论文误差界成立);"
          "SQ8 相对误差不随 dim 变化(分辨率有界)。dim≥128 时 RaBitQ "
          "估计精度足以支撑 γ=20 的候选生成,且码长仅为 SQ8 的 1/8。")


if __name__ == "__main__":
    main()
