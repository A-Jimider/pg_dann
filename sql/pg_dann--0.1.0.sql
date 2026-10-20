-- pg_dann 0.1.0：磁盘优先 ANN 索引访问方法（IVF + RaBitQ）
-- W1 骨架：注册 AM 与 opclass；索引结构从 W4 开始写入。
-- 依赖 pgvector 扩展（vector 类型），见 DESIGN.md §4.9。

CREATE FUNCTION dann_handler(internal)
RETURNS index_am_handler
AS 'MODULE_PATHNAME'
LANGUAGE C;

CREATE ACCESS METHOD dann
TYPE INDEX
HANDLER dann_handler;

COMMENT ON ACCESS METHOD dann IS
'pg_dann: disk-first ANN index (IVF + quantization), W1 skeleton';

-- L2（默认 opclass：USING dann (embedding) 时缺省使用）
CREATE OPERATOR CLASS dann_l2_ops
DEFAULT FOR TYPE vector USING dann AS
	OPERATOR 1 <-> (vector, vector) FOR ORDER BY float_ops,
	FUNCTION 1 l2_distance(vector, vector);

-- 内积（排序支持函数返回负内积，使升序 = 最近邻）
CREATE OPERATOR CLASS dann_ip_ops
FOR TYPE vector USING dann AS
	OPERATOR 1 <#> (vector, vector) FOR ORDER BY float_ops,
	FUNCTION 1 vector_negative_inner_product(vector, vector);

-- 余弦
CREATE OPERATOR CLASS dann_cosine_ops
FOR TYPE vector USING dann AS
	OPERATOR 1 <=> (vector, vector) FOR ORDER BY float_ops,
	FUNCTION 1 cosine_distance(vector, vector);

-- W6 可观测性:会话级扫描统计(W11 扩展更多维度)
CREATE FUNCTION dann_scan_stats_reset()
RETURNS void
AS 'MODULE_PATHNAME'
LANGUAGE C VOLATILE;

CREATE FUNCTION dann_scan_stats(
	OUT scans bigint,
	OUT truncated bigint,
	OUT lists_skipped bigint
)
RETURNS record
AS 'MODULE_PATHNAME'
LANGUAGE C VOLATILE;

-- W10 可观测性:每 list 条目数/页数/块号跨度(§4.5 布局退化监控)
CREATE FUNCTION dann_list_stats(
	rel regclass,
	OUT listno int,
	OUT entries bigint,
	OUT pages bigint,
	OUT span bigint
)
RETURNS SETOF record
AS 'MODULE_PATHNAME'
LANGUAGE C VOLATILE;
