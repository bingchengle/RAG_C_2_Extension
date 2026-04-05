from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.ingestion import VectorDBIngestor


all_reports_dir = ROOT / "databases" / "chunked_reports"
output_dir = ROOT / "databases" / "vector_dbs"

print("Generating BGE embeddings...")
ingestor = VectorDBIngestor(embedding_provider="bge_api")
ingestor.process_reports(all_reports_dir, output_dir, use_bge=True)
print("BGE embeddings generation completed!")
