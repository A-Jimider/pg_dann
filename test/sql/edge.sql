-- W11 边界用例 + 计划选择测试
CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION pg_dann;

-- ===== 1. 单行表(nlist 夹子到 1) =====
CREATE TABLE edge_one (id int, embedding vector(8));
INSERT INTO edge_one VALUES (1, '[1,2,3,4,5,6,7,8]');
CREATE INDEX edge_one_idx ON edge_one USING dann (embedding dann_l2_ops) WITH (nlist = 4);
SET enable_seqscan = off;
SELECT id FROM edge_one ORDER BY embedding <-> '[1,2,3,4,5,6,7,9]' LIMIT 1;
RESET enable_seqscan;

-- ===== 2. 重复向量(全表同一向量,全 tie) =====
CREATE TABLE edge_dup (id int, embedding vector(8));
INSERT INTO edge_dup SELECT i, '[1,2,3,4,5,6,7,8]'::vector FROM generate_series(1, 200) i;
CREATE INDEX edge_dup_idx ON edge_dup USING dann (embedding dann_l2_ops) WITH (nlist = 4);
SET enable_seqscan = off;
SELECT count(*) FROM (SELECT id FROM edge_dup ORDER BY embedding <-> '[1,2,3,4,5,6,7,8]' LIMIT 10) s;
RESET enable_seqscan;

-- ===== 3. NaN / Inf 拒绝(插入与查询) =====
INSERT INTO edge_dup VALUES (9999, '[1,2,3,4,nan,6,7,8]');
INSERT INTO edge_dup VALUES (9998, '[1,2,3,4,inf,6,7,8]');
SET enable_seqscan = off;
SELECT id FROM edge_dup ORDER BY embedding <-> '[1,2,3,4,nan,6,7,8]' LIMIT 1;
RESET enable_seqscan;

-- 构建期 NaN 同样拒绝
CREATE TABLE edge_nan (id int, embedding vector(8));
INSERT INTO edge_nan VALUES (1, '[1,2,3,4,5,6,7,8]');
INSERT INTO edge_nan VALUES (2, '[1,2,3,4,nan,6,7,8]');
CREATE INDEX edge_nan_idx ON edge_nan USING dann (embedding dann_l2_ops) WITH (nlist = 1);
DROP TABLE edge_nan;

-- ===== 4. NULL 向量插入:跳过索引,不报错 =====
INSERT INTO edge_dup VALUES (null, NULL);
SELECT count(*) FROM edge_dup;

-- ===== 5. reloption 覆盖(probes/rerank_factor)可用 =====
CREATE INDEX edge_dup_idx2 ON edge_dup USING dann (embedding dann_l2_ops)
  WITH (nlist = 4, probes = 2, rerank_factor = 5);
SET enable_seqscan = off;
SELECT count(*) FROM (SELECT id FROM edge_dup ORDER BY embedding <-> '[1,2,3,4,5,6,7,8]' LIMIT 5) s;
RESET enable_seqscan;
DROP INDEX edge_dup_idx2;

-- ===== 6. 计划选择:大表(5 万)选 dann,小表(200)选 seqscan =====
CREATE TABLE edge_big (id int, embedding vector(32));
INSERT INTO edge_big
SELECT i, (ARRAY(SELECT (sin(i*0.371 + j*0.917)*100)::float8
                 FROM generate_series(1,32) j))::vector
FROM generate_series(1, 50000) i;
CREATE INDEX edge_big_idx ON edge_big USING dann (embedding dann_l2_ops) WITH (nlist = 100);
ANALYZE edge_big;
EXPLAIN (COSTS OFF)
SELECT id FROM edge_big ORDER BY embedding <-> (SELECT embedding FROM edge_big WHERE id = 42) LIMIT 10;

CREATE TABLE edge_small (id int, embedding vector(32));
INSERT INTO edge_small
SELECT i, (ARRAY(SELECT (sin(i*0.371 + j*0.917)*100)::float8
                 FROM generate_series(1,32) j))::vector
FROM generate_series(1, 200) i;
CREATE INDEX edge_small_idx ON edge_small USING dann (embedding dann_l2_ops) WITH (nlist = 10);
ANALYZE edge_small;
EXPLAIN (COSTS OFF)
SELECT id FROM edge_small ORDER BY embedding <-> (SELECT embedding FROM edge_small WHERE id = 42) LIMIT 10;

DROP TABLE edge_one;
DROP TABLE edge_dup;
DROP TABLE edge_big;
DROP TABLE edge_small;
DROP EXTENSION pg_dann;
DROP EXTENSION vector;
