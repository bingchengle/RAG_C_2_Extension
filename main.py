import click
from dataclasses import replace
from pathlib import Path
from src.pipeline import Pipeline, configs, preprocess_configs

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
def process_reports(config):
    """Process parsed reports through the pipeline stages."""
    root_path = Path.cwd()
    run_config = preprocess_configs[config]
    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(f"Processing parsed reports (config={config})...")
    pipeline.process_parsed_reports()

@cli.command()
@click.option('--config', type=click.Choice(['base', 'pdr', 'max', 'max_no_ser_tab', 'max_nst_o3m', 'max_st_o3m', 'ibm_llama70b', 'ibm_llama8b', 'gemini_thinking', 'finance_vertical']), default='base', help='Configuration preset to use')
@click.option('--profile', type=click.Choice(['enhanced', 'legacy']), default='enhanced', help='Enhanced (new logic) or legacy (original-like logic).')
@click.option('--embedding-provider', 'embedding_provider_opt', type=click.Choice(['auto', 'openai', 'bge_api']), default='auto', help='Embedding provider. auto follows profile defaults.')
@click.option('--reranker-type', 'reranker_type_opt', type=click.Choice(['auto', 'llm', 'bge']), default='auto', help='Reranker type. auto follows profile defaults.')
def process_questions(config, profile, embedding_provider_opt, reranker_type_opt):
    """Process questions using the pipeline."""
    root_path = Path.cwd()
    run_config = replace(configs[config])

    if profile == 'legacy':
        # Approximate original project behavior.
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
        # Enhanced profile: BGE-first retrieval + finance-enhanced controls.
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
    pipeline = Pipeline(root_path, run_config=run_config)
    
    click.echo(
        f"Processing questions (config={config}, profile={profile}, "
        f"embedding_provider={run_config.embedding_provider}, reranker_type={run_config.reranker_type})..."
    )
    pipeline.process_questions()

if __name__ == '__main__':
    cli()