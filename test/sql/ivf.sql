-- W4-5 IVF 功能测试:真实构建 + KNN 查询 + 建后插入 + 三度量
-- 数据为确定性构造(间距 >> SQ8 量化误差),结果逐字节可复现。

CREATE EXTENSION IF NOT EXISTS vector;
CREATE EXTENSION pg_dann;

-- ===== L2:线性数据,间距 1000 =====
CREATE TABLE ivf_l2 (id int, embedding vector(8));
INSERT INTO ivf_l2 SELECT i, (ARRAY[1000.0*i, 0,0,0,0,0,0,0])::vector
FROM generate_series(0, 999) i;
CREATE INDEX ivf_l2_idx ON ivf_l2 USING dann (embedding dann_l2_ops)
  WITH (nlist = 8, quantizer = 'sq8');

SET enable_seqscan = off;

-- 查询点 [42200,...]:真值序 42(d=200), 43(800), 41(1200),间隔 >> 量化误差
SELECT id FROM ivf_l2 ORDER BY embedding <-> '[42200,0,0,0,0,0,0,0]' LIMIT 3;

-- 建后插入:42300 距查询点 100,比 42(200)更近,应排第一
INSERT INTO ivf_l2 VALUES (9999, '[42300,0,0,0,0,0,0,0]');
SELECT id FROM ivf_l2 ORDER BY embedding <-> '[42200,0,0,0,0,0,0,0]' LIMIT 3;

-- 再插入几行验证追加路径(页容量 291,不满页)
INSERT INTO ivf_l2 SELECT 10000+i, (ARRAY[42400.0+i, 0,0,0,0,0,0,0])::vector
FROM generate_series(0, 9) i;
-- 无 ORDER BY 的全索引扫描(count(*)):返回全部条目
SELECT count(*) FROM ivf_l2;

-- ===== IP:点积最大者 =====
CREATE TABLE ivf_ip (id int, embedding vector(8));
INSERT INTO ivf_ip SELECT i, (ARRAY[1000.0*i, 0,0,0,0,0,0,0])::vector
FROM generate_series(0, 999) i;
CREATE INDEX ivf_ip_idx ON ivf_ip USING dann (embedding dann_ip_ops)
  WITH (nlist = 8, quantizer = 'sq8');

-- 查询 [2,0...]:<q,v_i> = 2000i,最大者 i=999
SELECT id FROM ivf_ip ORDER BY embedding <#> '[2,0,0,0,0,0,0,0]' LIMIT 1;

-- ===== cosine:one-hot 向量,角度间隔大 =====
CREATE TABLE ivf_cos (id int, embedding vector(8));
-- v_i = 第 i 维 one-hot(i=1..8)
INSERT INTO ivf_cos SELECT i, (SELECT array_agg(CASE WHEN j = i THEN 1.0 ELSE 0.0 END)::vector FROM generate_series(1,8) j) FROM generate_series(1, 8) i;
CREATE INDEX ivf_cos_idx ON ivf_cos USING dann (embedding dann_cosine_ops)
  WITH (nlist = 2, quantizer = 'sq8');

-- 查询 ≈ e_3:cos_dist(e_3)≈0,其余 ≥ 0.5,间隔 >> 量化误差
SELECT id FROM ivf_cos ORDER BY embedding <=> '[0,0,1,0.01,0,0,0,0]' LIMIT 1;

RESET enable_seqscan;

DROP TABLE ivf_l2;
DROP TABLE ivf_ip;
DROP TABLE ivf_cos;
DROP EXTENSION pg_dann;
DROP EXTENSION vector;
