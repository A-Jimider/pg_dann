# pg_dann — disk-first ANN index access method for PostgreSQL
# 设计文档：DESIGN.md（W1 = PGXS 骨架 + AM 回调 stub）

EXTENSION   = pg_dann
EXTVERSION  = 0.1.0

MODULE_big  = pg_dann
OBJS        = src/dann.o src/build.o src/insert.o src/scan.o src/vacuum.o \
              src/cost.o src/page.o src/kmeans.o \
              quant/sq8.o quant/fwht.o quant/rabitq.o \
              simd/dispatch.o simd/l2_scalar.o \
              simd/l2_avx2.o simd/l2_avx512.o simd/sq8_est_avx2.o \
              simd/rabitq_ipqr_avx2.o simd/rabitq_ipqr_avx512.o

DATA        = sql/pg_dann--0.1.0.sql

# 各用例都 CREATE/DROP EXTENSION,必须串行(--max-connections=1)
REGRESS      = basic ivf rerank rabitq vacuum edge
REGRESS_OPTS = --inputdir=test --max-connections=1

PG_CPPFLAGS = -Iquant

PG_CONFIG ?= pg_config
PGXS := $(shell $(PG_CONFIG) --pgxs)
include $(PGXS)

# ---- 独立量化器对拍与基准(纯 C,不依赖 PG;W2-3 验收入口) ----
VENV_PYTHON ?= $(CURDIR)/.venv/bin/python

QUANT_OBJS = quant/sq8.o quant/fwht.o quant/rabitq.o \
             simd/dispatch.o simd/l2_scalar.o \
             simd/l2_avx2.o simd/l2_avx512.o simd/sq8_est_avx2.o \
             simd/rabitq_ipqr_avx2.o simd/rabitq_ipqr_avx512.o

bench/quant_harness: bench/quant_harness.c $(QUANT_OBJS)
	$(CC) $(CFLAGS) -Iquant -o $@ bench/quant_harness.c $(QUANT_OBJS) -lm

.PHONY: quant-test
quant-test: bench/quant_harness
	$(VENV_PYTHON) golden/sq8_model.py
	$(VENV_PYTHON) golden/rabitq_model.py
	./bench/quant_harness golden/fixtures/sq8_d4.txt \
		golden/fixtures/sq8_d128.txt golden/fixtures/sq8_d1001.txt \
		golden/fixtures/rabitq_d4.txt golden/fixtures/rabitq_d64.txt \
		golden/fixtures/rabitq_d100.txt golden/fixtures/rabitq_d128.txt \
		golden/fixtures/rabitq_d1001.txt
