-- W1 骨架 + W4-5 构建路径的基础设施测试
-- (KNN 功能测试在 ivf.sql)

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION pg_dann;

-- AM 与 opclass 注册
SELECT amname FROM pg_am WHERE amname = 'dann';
SELECT opcname FROM pg_opclass WHERE opcmethod = (SELECT oid FROM pg_am WHERE amname = 'dann') ORDER BY 1;

CREATE TABLE dann_test (id int, embedding vector(4));
INSERT INTO dann_test
SELECT i, ARRAY[i::float4, (i + 1)::float4, (i + 2)::float4, (i + 3)::float4]::vector
FROM generate_series(1, 100) AS g(i);

-- 建索引(触发三遍构建)
CREATE INDEX dann_test_l2 ON dann_test USING dann (embedding dann_l2_ops)
  WITH (nlist = 4);

-- 缺省 opclass 应为 dann_l2_ops(验证后即删,避免同列多索引导致 EXPLAIN 不稳定)
CREATE INDEX dann_test_default ON dann_test USING dann (embedding);
DROP INDEX dann_test_default;

-- 非法 reloption 必须被拒绝
CREATE INDEX dann_test_bad ON dann_test USING dann (embedding dann_l2_ops)
  WITH (quantizer = 'nope');

-- W7 起 rabitq 可用:dim<128 时 auto 落 SQ8;显式 rabitq 亦接受(功能测试见 rabitq.sql)
CREATE INDEX dann_test_rb ON dann_test USING dann (embedding dann_l2_ops)
  WITH (quantizer = 'rabitq', nlist = 4);
SET enable_seqscan = off;
SELECT count(*) FROM (
  SELECT id FROM dann_test ORDER BY embedding <-> '[1,2,3,4]' LIMIT 5
) s;
RESET enable_seqscan;
DROP INDEX dann_test_rb;

-- 默认计划不能选 dann(代价 stub 保证),结果走 seqscan
SELECT count(*) FROM dann_test;

-- 强制走索引:返回真实候选(W4 起非空)
SET enable_seqscan = off;
EXPLAIN (COSTS OFF)
SELECT id FROM dann_test ORDER BY embedding <-> '[1,2,3,4]' LIMIT 5;
SELECT count(*) FROM (
  SELECT id FROM dann_test ORDER BY embedding <-> '[1,2,3,4]' LIMIT 5
) s;
RESET enable_seqscan;

-- VACUUM 走 ambulkdelete / amvacuumcleanup
VACUUM dann_test;

-- 空表构建:nlist=0,插入必须报错
CREATE TABLE dann_empty (id int, embedding vector(4));
CREATE INDEX dann_empty_idx ON dann_empty USING dann (embedding) WITH (nlist = 4);
INSERT INTO dann_empty VALUES (1, '[1,2,3,4]');

DROP TABLE dann_test;
DROP TABLE dann_empty;
DROP EXTENSION pg_dann;
DROP EXTENSION vector;
