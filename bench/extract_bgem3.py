#!/usr/bin/env python3
"""bge-m3 parquet → fvecs 抽取(前 n 行,只读 embedding 列,行组流式)。

数据源:hcsolakoglu/turkish_wikipedia_with_bge_m3_embeddings(hf-mirror,
土耳其语维基 2023,bge-m3 dense embedding,float16)。只取向量做基准,
文本列不读。输出 base.fvecs(float32, [dim|vals] 行格式,与 sift 一致),
供 load_bgem3.py / gt 用。
用法:.venv/bin/python bench/extract_bgem3.py [n_rows]
"""
import os
import sys
import numpy as np
import pyarrow.parquet as pq

DATA = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data", "bgem3")
SRC = os.path.join(DATA, "tr_wiki_bgem3.parquet")


def pick_emb_column(pf):
    for name in pf.schema_arrow.names:   # 顶层列名(pf.schema.names 会拍平到叶子)
        if "embed" in name.lower():
            return name
    raise SystemExit(f"no embedding column in schema: {pf.schema_arrow.names}")


def main():
    n_want = int(sys.argv[1]) if len(sys.argv) > 1 else 100_000
    pf = pq.ParquetFile(SRC)
    col = pick_emb_column(pf)
    print(f"schema: {pf.schema_arrow.names}\nusing column: {col}, rows in file: {pf.metadata.num_rows}")

    dim = None
    got = 0
    norms = []
    out = os.path.join(DATA, "base.fvecs")
    with open(out, "wb") as f:
        for batch in pf.iter_batches(batch_size=8192, columns=[col]):
            arr = batch.column(col)
            # list<float16> / fixed_size_list<float16> → flat numpy
            flat = np.asarray(arr.values.to_numpy(zero_copy_only=False))
            if dim is None:
                dim = flat.shape[0] // len(arr)
                print(f"dim = {dim}, value dtype = {flat.dtype}")
            m = flat.reshape(len(arr), dim).astype(np.float32)
            if got == 0:
                norms = np.linalg.norm(m[:2000], axis=1)  # 采样看归一化
            for row in m:
                if got >= n_want:
                    break
                np.array([dim], dtype=np.int32).tofile(f)
                row.tofile(f)
                got += 1
            if got >= n_want:
                break
            print(f"  extracted {got}", flush=True)

    print(f"wrote {out}: {got} rows x {dim}d float32")
    if len(norms):
        print(f"norm sample: mean={norms.mean():.4f} std={norms.std():.4f} "
          f"min={norms.min():.4f} max={norms.max():.4f}  "
          f"({'unit-norm' if abs(norms.mean()-1) < 0.01 else 'NOT normalized'})")


if __name__ == "__main__":
    main()
