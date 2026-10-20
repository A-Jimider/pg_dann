/*-------------------------------------------------------------------------
 * dann.c — 扩展入口：GUC、reloptions、AM handler
 *
 * 参数语义见 DESIGN.md §4.4/§4.9。
 *-------------------------------------------------------------------------*/
#include "postgres.h"

#include "access/amapi.h"
#include "access/reloptions.h"
#include "commands/vacuum.h"
#include "fmgr.h"
#include "storage/lock.h"
#include "utils/guc.h"
#include "utils/guc_tables.h"

#include "dann.h"
#include "quant.h"

PG_MODULE_MAGIC;

/* GUC 变量（缺省值见 DESIGN.md §4.2/§4.4） */
int			dann_probes = 10;
int			dann_rerank_factor = 20;
bool		dann_enable_rerank = true;
int			dann_max_latency_ms = 0;
int			dann_io_method = DANN_IO_AUTO;

static relopt_kind dann_relopt_kind;

static const struct config_enum_entry dann_io_method_options[] = {
	{"auto", DANN_IO_AUTO, false},
	{"read_stream", DANN_IO_READ_STREAM, false},
	{"fadvise", DANN_IO_FADVISE, false},
	{"sync", DANN_IO_SYNC, false},
	{NULL, 0, false}
};

static void
dann_validate_quantizer(const char *value)
{
	if (value == NULL ||
		(strcmp(value, "auto") != 0 &&
		 strcmp(value, "rabitq") != 0 &&
		 strcmp(value, "sq8") != 0))
		ereport(ERROR,
				(errcode(ERRCODE_INVALID_PARAMETER_VALUE),
				 errmsg("invalid value for reloption \"quantizer\": \"%s\"",
						value != NULL ? value : ""),
				 errhint("Valid values are \"auto\", \"rabitq\", and \"sq8\".")));
}

void
_PG_init(void)
{
	/* SIMD 运行时分派 */
	dann_simd_init();

	/* reloptions：CREATE INDEX ... WITH (...) */
	dann_relopt_kind = add_reloption_kind();
	add_int_reloption(dann_relopt_kind, "nlist",
					  "Number of IVF lists (clusters).",
					  DANN_DEFAULT_NLIST, DANN_MIN_NLIST, DANN_MAX_NLIST,
					  AccessExclusiveLock);
	add_string_reloption(dann_relopt_kind, "quantizer",
						 "Quantizer: auto | rabitq | sq8; "
						 "\"auto\" picks by dimension (DESIGN.md §4.2).",
						 "auto", dann_validate_quantizer,
						 AccessExclusiveLock);
	add_int_reloption(dann_relopt_kind, "probes",
					  "Per-index override of dann.probes (0 = use GUC).",
					  0, 0, DANN_MAX_NLIST,
					  AccessExclusiveLock);
	add_int_reloption(dann_relopt_kind, "rerank_factor",
					  "Per-index override of dann.rerank_factor (0 = use GUC).",
					  0, 0, 10000,
					  AccessExclusiveLock);

	/* GUC：查询期参数 */
	DefineCustomIntVariable("dann.probes",
							"Sets the number of lists to probe.",
							NULL,
							&dann_probes,
							10, 1, DANN_MAX_NLIST,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("dann.rerank_factor",
							"Rerank candidate multiplier gamma "
							"(candidates = gamma * k).",
							NULL,
							&dann_rerank_factor,
							20, 1, 10000,
							PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomBoolVariable("dann.enable_rerank",
							 "Rerank candidates with exact distances "
							 "from the heap.",
							 NULL,
							 &dann_enable_rerank,
							 true,
							 PGC_USERSET, 0, NULL, NULL, NULL);
	DefineCustomIntVariable("dann.max_latency_ms",
							"Latency budget for the posting scan; 0 = unlimited.",
							"Scanning stops when the budget is exhausted and "
							"the best candidates so far are returned "
							"(graceful recall degradation, DESIGN.md §4.4).",
							&dann_max_latency_ms,
							0, 0, INT_MAX,
							PGC_USERSET, GUC_UNIT_MS, NULL, NULL, NULL);
	DefineCustomEnumVariable("dann.io_method",
							 "Prefetch method for posting pages.",
							 "read_stream needs PG17+; falls back to "
							 "fadvise/sync on older versions.",
							 &dann_io_method,
							 DANN_IO_AUTO,
							 dann_io_method_options,
							 PGC_USERSET, 0, NULL, NULL, NULL);
}

bytea *
dann_options(Datum reloptions, bool validate)
{
	static const relopt_parse_elt tab[] = {
		{"nlist", RELOPT_TYPE_INT, offsetof(DannOptions, nlist)},
		{"quantizer", RELOPT_TYPE_STRING, offsetof(DannOptions, quantizer)},
		{"probes", RELOPT_TYPE_INT, offsetof(DannOptions, probes)},
		{"rerank_factor", RELOPT_TYPE_INT, offsetof(DannOptions, rerank_factor)}
	};

	return (bytea *) build_reloptions(reloptions, validate,
									  dann_relopt_kind,
									  sizeof(DannOptions),
									  tab, lengthof(tab));
}

/* 每索引生效 probes:reloption 优先,否则 GUC */
int
dann_probes_of(Relation index)
{
	DannOptions *opts = (DannOptions *) index->rd_options;

	if (opts != NULL && opts->probes > 0)
		return opts->probes;
	return dann_probes;
}

int
dann_rerank_factor_of(Relation index)
{
	DannOptions *opts = (DannOptions *) index->rd_options;

	if (opts != NULL && opts->rerank_factor > 0)
		return opts->rerank_factor;
	return dann_rerank_factor;
}

bool
dann_validate(Oid opclassoid)
{
	/* W1 stub：接受一切 opclass。支持函数校验随量化器（W2+）落地。 */
	return true;
}

PG_FUNCTION_INFO_V1(dann_handler);

Datum
dann_handler(PG_FUNCTION_ARGS)
{
	IndexAmRoutine *amroutine = makeNode(IndexAmRoutine);

	amroutine->amstrategies = 0;
	amroutine->amsupport = 1;	/* 支持函数 1 = 距离函数 */
	amroutine->amoptsprocnum = 0;
	amroutine->amcanorder = false;
	amroutine->amcanorderbyop = true;	/* KNN：ORDER BY ... LIMIT */
	amroutine->amcanbackward = false;
	amroutine->amcanunique = false;
	amroutine->amcanmulticol = false;
	amroutine->amoptionalkey = true;
	amroutine->amsearcharray = false;
	amroutine->amsearchnulls = false;
	amroutine->amstorage = false;
	amroutine->amclusterable = false;
	amroutine->ampredlocks = false;
	amroutine->amcanparallel = false;	/* W1：无并行扫描状态 */
	amroutine->amcaninclude = false;
	amroutine->amusemaintenanceworkmem = true;
	amroutine->amsummarizing = false;
	amroutine->amparallelvacuumoptions = VACUUM_OPTION_PARALLEL_BULKDEL;
	amroutine->amkeytype = InvalidOid;

	amroutine->ambuild = dann_build;
	amroutine->ambuildempty = dann_buildempty;
	amroutine->aminsert = dann_insert;
	amroutine->ambulkdelete = dann_bulkdelete;
	amroutine->amvacuumcleanup = dann_vacuumcleanup;
	amroutine->amcanreturn = NULL;
	amroutine->amcostestimate = dann_costestimate;
	amroutine->amoptions = dann_options;
	amroutine->amproperty = NULL;
	amroutine->ambuildphasename = NULL;
	amroutine->amvalidate = dann_validate;
	/* amadjustmembers（PG14+）：可选，makeNode 已置 NULL */
	amroutine->ambeginscan = dann_beginscan;
	amroutine->amrescan = dann_rescan;
	amroutine->amgettuple = dann_gettuple;
	amroutine->amgetbitmap = NULL;
	amroutine->amendscan = dann_endscan;
	amroutine->ammarkpos = NULL;
	amroutine->amrestrpos = NULL;

	PG_RETURN_POINTER(amroutine);
}
