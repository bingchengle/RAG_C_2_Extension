from pathlib import Path
from src.ingestion import VectorDBIngestor

# 配置路径
all_reports_dir = Path("./databases/chunked_reports")
output_dir = Path("./databases/vector_dbs")

# 生成 BGE 嵌入
print("Generating BGE embeddings...")
ingestor = VectorDBIngestor()
ingestor.process_reports(all_reports_dir, output_dir, use_bge=True)
print("BGE embeddings generation completed!")
