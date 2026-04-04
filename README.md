# RAG Challenge Winner Solution

**Read more about this project:**
- Russian: https://habr.com/ru/articles/893356/
- English: https://abdullin.com/ilya/how-to-build-best-rag/

This repository contains the winning solution for both prize nominations in the RAG Challenge competition. The system achieved state-of-the-art results in answering questions about company annual reports using a combination of:

- Custom PDF parsing with Docling
- Vector search with parent document retrieval
- LLM reranking for improved context relevance
- Structured output prompting with chain-of-thought reasoning
- Query routing for multi-company comparisons

## Disclaimer

This is competition code - it's scrappy but it works. Some notes before you dive in:

- IBM Watson integration won't work (it was competition-specific)
- The code might have rough edges and weird workarounds
- No tests, minimal error handling - you've been warned
- You'll need your own API keys for OpenAI/Gemini
- GPU helps a lot with PDF parsing (I used 4090)

If you're looking for production-ready code, this isn't it. But if you want to explore different RAG techniques and their implementations - check it out!

## Quick Start

Clone and setup:
```bash
git clone https://github.com/IlyaRice/RAG-Challenge-2.git
cd RAG-Challenge-2
python -m venv venv
venv\Scripts\Activate.ps1  # Windows (PowerShell)
pip install -e . -r requirements.txt
```

Rename `env` to `.env` and add your API keys.

## Test Dataset

The repository includes two datasets:

1. A small test set (in `data/test_set/`) with 5 annual reports and questions
2. The full ERC2 competition dataset (in `data/erc2_set/`) with all competition questions and reports

Each dataset directory contains its own README with specific setup instructions and available files. You can use either dataset to:

- Study example questions, reports, and system outputs
- Run the pipeline from scratch using provided PDFs
- Use pre-processed data to skip directly to specific pipeline stages

See the respective README files for detailed dataset contents and setup instructions:
- `data/test_set/README.md` - For the small test dataset
- `data/erc2_set/README.md` - For the full competition dataset

## Usage

You can run any part of pipeline by uncommenting the method you want to run in `src/pipeline.py` and executing:
```bash
python .\src\pipeline.py
```

You can also run any pipeline stage using `main.py`, but you need to run it from the directory containing your data:
```bash
cd .\data\test_set\
python ..\..\main.py process-questions --config max_nst_o3m
```

### CLI Commands

Get help on available commands:
```bash
python main.py --help
```

Available commands:
- `download-models` - Download required docling models
- `parse-pdfs` - Parse PDF reports with parallel processing options
- `serialize-tables` - Process tables in parsed reports
- `process-reports` - Run the full pipeline on parsed reports
- `process-questions` - Process questions using specified config

Each command has its own options. For example:
```bash
python main.py parse-pdfs --help
# Shows options like --parallel/--sequential, --chunk-size, --max-workers

python main.py process-reports --config ser_tab
# Process reports with serialized tables config
```

## Some configs

- `max_nst_o3m` - Best performing config using OpenAI's o3-mini model
- `ibm_llama70b` - Alternative using IBM's Llama 70B model
- `gemini_thinking` - Full context answering with using enormous context window of Gemini. It is not RAG, actually

Check `pipeline.py` for more configs and detils on them.

## License

MIT