import click
from dataclasses import replace
from pathlib import Path
from src.pipeline import Pipeline, configs, preprocess_configs
from src.route_config import apply_route_to_run_config

@click.group()
def cli():
    """Pipeline command line interface for processing PDF reports and questions."""
    pass

@cli.command()
def download_models():
    """Download required docling models."""
    click.echo("Downloading docling models...")
    Pipeline.download_docling_models()

@cli.command()
@click.option('--parallel/--sequential', default=True, help='Run parsing in parallel or sequential mode')
@click.option('--chunk-size', default=2, help='Number of PDFs to process in each worker')
@click.option('--max-workers', default=10, help='Number of parallel worker processes')
def parse_pdfs(parallel, chunk_size, max_workers):
    """Parse PDF reports with optional parallel processing."""
    root_path = Path.cwd()
    pipeline = Pipeline(root_path)

    click.echo(f"Parsing PDFs (parallel={parallel}, chunk_size={chunk_size}, max_workers={max_workers})")
    pipeline.parse_pdf_reports(parallel=parallel, chunk_size=chunk_size, max_workers=max_workers)

@cli.command()
@click.option('--max-workers', default=10, help='Number of workers for table serialization')
def serialize_tables(max_workers):
    """Serialize tables in parsed reports using parallel threading."""
    root_path = Path.cwd()
    pipeline = Pipeline(root_path)

    click.echo(f"Serializing tables (max_workers={max_workers})...")
    pipeline.serialize_tables(max_workers=max_workers)

@cli.command()
@click.option('--config', type=click.Choice(['ser_tab', 'no_ser_tab']), default='no_ser_tab', help='Configuration preset to use')
@click.option(
    '--embedding-provider',
    type=click.Choice(['openai', 'bge_api']),
    default='openai',
    help='Embedding API for .faiss files: openai -> stem.faiss; bge_api -> stem_bge.faiss (must match process-questions).',
)
@click.option('--embedding-model', default='', help='Optional model id; empty uses provider default from env.')
@click.option(
    '--build-bm25/--no-build-bm25',
    default=False,
    help='After vector DBs, rebuild bm25_dbs from chunked_reports (needed for hybrid LLM reranking).',
)
def process_reports(config, embedding_provider, embedding_model, build_bm25):
    """Process parsed reports through the pipeline stages."""
    root_path = Path.cwd()
    run_config = replace(
        preprocess_configs[config],
        embedding_provider=embedding_provider,
        embedding_model=(embedding_model or "").strip(),
    )
    pipeline = Pipeline(root_path, run_config=run_config)

    click.echo(
        f"Processing parsed reports (config={config}, embedding_provider={embedding_provider}, "
        f"build_bm25={build_bm25})..."
    )
    pipeline.process_parsed_reports()
    if build_bm25:
        click.echo("Building BM25 indices...")
        pipeline.create_bm25_db()


@cli.command('rebuild-vector-dbs')
@click.option(
    '--embedding-provider',
    type=click.Choice(['openai', 'bge_api']),
    default='openai',
    help='Must match the provider you use in process-questions.',
)
@click.option('--embedding-model', default='', help='Optional model id; empty uses provider default from env.')
@click.option(
    '--build-bm25/--no-build-bm25',
    default=False,
    help='Also rebuild bm25_dbs (hybrid reranking needs this if missing).',
)
def rebuild_vector_dbs(embedding_provider, embedding_model, build_bm25):
    """Rebuild vector indexes from existing databases/chunked_reports (no merge/chunk steps)."""
    root_path = Path.cwd()
    run_config = replace(
        preprocess_configs['no_ser_tab'],
        embedding_provider=embedding_provider,
        embedding_model=(embedding_model or "").strip(),
    )
    pipeline = Pipeline(root_path, run_config=run_config)
    click.echo(f"Rebuilding vector DBs (embedding_provider={embedding_provider})...")
    pipeline.create_vector_dbs()
    if build_bm25:
        click.echo("Building BM25 indices...")
        pipeline.create_bm25_db()

@cli.command()
@click.option('--config', type=click.Choice(['base', 'pdr', 'max', 'max_no_ser_tab', 'max_nst_o3m', 'max_st_o3m', 'ibm_llama70b', 'ibm_llama8b', 'gemini_thinking', 'finance_vertical']), default='base', help='Configuration preset to use')
@click.option('--profile', type=click.Choice(['enhanced', 'legacy']), default='enhanced', help='Enhanced (new logic) or legacy (original-like logic).')
@click.option('--embedding-provider', 'embedding_provider_opt', type=click.Choice(['auto', 'openai', 'bge_api']), default='auto', help='Embedding provider. auto follows profile defaults.')
@click.option('--reranker-type', 'reranker_type_opt', type=click.Choice(['auto', 'llm', 'bge']), default='auto', help='Reranker type. auto follows profile defaults.')
@click.option('--route', type=click.Choice(['balanced', 'economy', 'quality']), default='balanced', help='Cost/quality preset for LLM models, verification rounds, and similarity mode.')
@click.option('--hybrid-llm-weight', type=float, default=None, help='Hybrid rerank: weight for LLM/BGE score vs vector similarity (0-1).')
@click.option('--hybrid-bm25-weight', type=float, default=None, help='Hybrid merge: weight for BM25 vs vector (0-1).')
@click.option('--llm-rerank-batch-size', type=int, default=None, help='Documents per batch when using LLM reranker.')
@click.option('--top-n-retrieval', type=int, default=None, help='Final top-k chunks after rerank (default: preset from --config).')
@click.option('--llm-reranking-sample-size', type=int, default=None, help='Hybrid rerank pool size (default: preset from --config).')
@click.option('--enable-abstention-gate/--no-abstention-gate', default=False, help='Enable multi-signal abstention (default off).')
@click.option('--abstain-min-retrieval-strength', type=float, default=None, help='Abstain if best retrieval signal is below this (0-1).')
@click.option('--abstain-min-confidence', type=float, default=None, help='Abstain if model confidence is below this.')
@click.option(
    '--abstain-on-validation-fail/--no-abstain-on-validation-fail',
    default=False,
    help='Abstain when answer validation did not pass.',
)
def process_questions(
    config,
    profile,
    embedding_provider_opt,
    reranker_type_opt,
    route,
    hybrid_llm_weight,
    hybrid_bm25_weight,
    llm_rerank_batch_size,
    top_n_retrieval,
    llm_reranking_sample_size,
    enable_abstention_gate,
    abstain_min_retrieval_strength,
    abstain_min_confidence,
    abstain_on_validation_fail,
):
    """Process questions using the pipeline."""
    root_path = Path.cwd()
    run_config = replace(configs[config])

    if profile == 'legacy':

        run_config.domain = "general"
        run_config.finance_metric_expansion = False
        run_config.finance_normalize_numeric = False
        run_config.finance_currency_consistency_check = False
        run_config.finance_unit_conversion = False
        run_config.enable_query_rewrite = False
        run_config.enable_similarity_check = False
        run_config.enable_multi_turn = False
        default_embedding_provider = "openai"
        default_reranker_type = "llm"
    else:

        run_config.domain = "finance"
        run_config.finance_metric_expansion = True
        run_config.finance_normalize_numeric = True
        run_config.finance_currency_consistency_check = True
        run_config.finance_unit_conversion = True
        run_config.enable_query_rewrite = True
        run_config.enable_similarity_check = True
        default_embedding_provider = "bge_api"
        default_reranker_type = "bge"

    run_config.embedding_provider = default_embedding_provider if embedding_provider_opt == "auto" else embedding_provider_opt
    run_config.reranker_type = default_reranker_type if reranker_type_opt == "auto" else reranker_type_opt
    run_config = replace(run_config, route=route)
    run_config = apply_route_to_run_config(run_config)
    if hybrid_llm_weight is not None:
        run_config = replace(run_config, hybrid_rerank_llm_weight=hybrid_llm_weight)
    if hybrid_bm25_weight is not None:
        run_config = replace(run_config, hybrid_merge_bm25_weight=hybrid_bm25_weight)
    if llm_rerank_batch_size is not None:
        run_config = replace(run_config, llm_rerank_documents_batch_size=llm_rerank_batch_size)
    if top_n_retrieval is not None:
        run_config = replace(run_config, top_n_retrieval=max(1, int(top_n_retrieval)))
    if llm_reranking_sample_size is not None:
        run_config = replace(run_config, llm_reranking_sample_size=max(1, int(llm_reranking_sample_size)))
    run_config = replace(
        run_config,
        enable_abstention_gate=enable_abstention_gate,
        abstain_min_retrieval_strength=abstain_min_retrieval_strength,
        abstain_min_confidence=abstain_min_confidence,
        abstain_on_validation_fail=abstain_on_validation_fail,
    )
    pipeline = Pipeline(root_path, run_config=run_config)

    click.echo(
        f"Processing questions (config={config}, profile={profile}, route={run_config.route}, "
        f"embedding_provider={run_config.embedding_provider}, reranker_type={run_config.reranker_type})..."
    )
    pipeline.process_questions()

if __name__ == '__main__':
    cli()