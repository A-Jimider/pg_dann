# pg_dann — Disk-First Vector Index PostgreSQL Extension

[中文](README.md) | **English**

An ANN index access method combining IVF (inverted clustering) with
quantization (SQ8 / RaBitQ): fully disk-resident, fully WAL-logged,
crash-recoverable. Docs (in Chinese): [DESIGN.md](DESIGN.md) (design plan) |
[ARCHITECTURE.md](ARCHITECTURE.md) (as-built architecture; code is authoritative) |
[ARTICLE.md](ARTICLE.md) (implementation notes & data) |
[bench/RESULTS.md](bench/RESULTS.md) (per-milestone acceptance) |
[DEVLOG.md](DEVLOG.md) (dev log & current status).

Built on pgvector's `vector` type (ecosystem-compatible, zero-friction comparison).

## Quick Start

```sql
CREATE EXTENSION vector;
CREATE EXTENSION pg_dann;

CREATE TABLE docs (id int, embedding vector(768));

-- quantizer defaults to auto: dim < 128 → SQ8, dim ≥ 128 → RaBitQ
CREATE INDEX docs_dann ON docs USING dann (embedding)          -- default L2
  WITH (nlist = 4096);
-- or explicit: USING dann (embedding dann_cosine_ops) WITH (quantizer='sq8', probes=64)

SELECT * FROM docs ORDER BY embedding <=> $1 LIMIT 10;   -- cosine
-- L2: <->   inner product: <#> (descending similarity = ascending negative IP)
```

Build memory is independent of table size (sampling is capped). SIFT1M (1M×128)
builds in ≈ 17s (RaBitQ) / 20s (SQ8); index size 39MB / 146MB (ivfflat 525MB).

## Parameters

GUCs (query-time, all can be SET per session):

| GUC | Default | Description |
|---|---|---|
| `dann.probes` | 10 | lists probed; higher = better recall, slower |
| `dann.rerank_factor` | 20 | γ: heap-rerank candidates start at γ×10; 40 recommended for RaBitQ |
| `dann.enable_rerank` | on | off returns the quantized estimated order (lowest latency, slightly lower recall) |
| `dann.max_latency_ms` | 0 | latency budget, 0 = unlimited; on exhaustion the farthest tail lists are truncated and recall degrades smoothly |
| `dann.io_method` | auto | `auto`/`fadvise` = batched prefetch (PrefetchBuffer); `sync` = synchronous reads |

reloptions (`CREATE INDEX ... WITH(...)`; `probes`/`rerank_factor` act as per-index overrides):

| reloption | Default | Description |
|---|---|---|
| `nlist` | 1000 | IVF list count (auto-clamped for empty tables / small samples) |
| `quantizer` | auto | `auto`/`rabitq`/`sq8` |
| `probes` | 0 | >0 overrides `dann.probes` |
| `rerank_factor` | 0 | >0 overrides `dann.rerank_factor` |

## Observability

```sql
SELECT * FROM dann_scan_stats();          -- (scans, truncated, lists_skipped), session-scoped
SELECT dann_scan_stats_reset();
SELECT * FROM dann_list_stats('docs_dann');   -- per list: (listno, entries, pages, span)
-- span > pages means incremental inserts degraded the layout; REINDEX if severe
```

## Semantics

- **relaxed order**: with rerank on, results are ordered by full-precision exact
  distance (identical to seqscan+sort, provided the candidate set covers the true
  neighbors); with rerank off, the quantized estimated order is returned.
  Planner semantics when the index provides no exact guarantee: DESIGN.md §4.8.
- **NULL query vector** (`ORDER BY embedding <-> NULL`): degrades to a full index
  scan — row set correct, order undefined (same undefined order as seqscan+sort).
- NaN/Inf vectors are rejected on write and on query (the pgvector type layer
  rejects them first).
- Dead tuples: stale index entries left by DELETE/UPDATE are removed by VACUUM;
  queries skip them until then.

## Known Limitations (v1)

- **In-page compact delete, no empty-page recycling**: the index does not shrink
  after mass deletes; REINDEX to rebuild (same as IVFFlat). Use span/entries in
  `dann_list_stats()` for operational guidance.
- **Serialized inserts**: concurrent inserts serialize on the metapage exclusive
  lock (≈IVFFlat; ~1758 rows/s steady-state measured). The 128 list LWLocks of
  DESIGN.md §4.7 require `shared_preload_libraries` support — left for a future upgrade.
- **No parallel build/scan**: single-threaded build (two table passes);
  `amcanparallel=false`.
- **Crash consistency relies on WAL**: under wal_level=minimal, smgrimmedsync at
  end of build is the backstop; crash-recovery matrix in bench/RESULTS.md W9.
- PG18 read stream prefetch is a compile-time TODO (PG16 uses PrefetchBuffer
  uniformly); ARM NEON not implemented (scalar fallback works).

## Development & Testing

```bash
make && make install                    # PGXS, PG_CONFIG pointing to PG16+
make installcheck PGPORT=6543 PGHOST=/tmp   # regression: basic/ivf/rerank/rabitq/vacuum/edge
make quant-test                         # SQ8/RaBitQ golden-model cross-check (pure C)
.venv/bin/python bench/load_sift.py     # load SIFT1M (data via hf-mirror, see bench/data)
.venv/bin/python bench/recall.py 200    # recall/latency grid (dann SQ8 vs RaBitQ vs ivfflat)
.venv/bin/python bench/budget.py 200    # latency-budget degradation curve
.venv/bin/python bench/conc_test.py     # 32-way concurrency stress
bench/crash_test.sh                     # crash-recovery matrix (kill -9; restarts the instance!)
```

Bench scripts read connection settings from environment variables:
`PG_DANN_DSN` (Python scripts) and `PG_DANN_PG` / `PGHOST` / `PGPORT`
(shell scripts and minbuf.py).

Layout: `src/` (AM callbacks), `quant/` (quantizers, pure C), `simd/`
(AVX2/AVX-512 dispatch), `golden/` (numpy golden models + fixtures + error
curves), `test/` (pg_regress), `bench/` (SIFT1M benchmarks & acceptance scripts).

## License

[AGPL-3.0](LICENSE). Commercial use is allowed; however, distributing modified
versions or offering them as a network service (including cloud hosting/SaaS)
requires disclosing the source under the same license.
