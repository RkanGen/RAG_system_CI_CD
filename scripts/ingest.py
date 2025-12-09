# scripts/ingest.py
import asyncio
import argparse
import sys
import os
from pathlib import Path

sys.path.append(os.getcwd())

from src.pipeline.ingestion import get_ingestion_pipeline
from src.api.schemas import IngestRequest

async def ingest_file(file_path: str, source_tag: str):
    pipeline = get_ingestion_pipeline()
    path_obj = Path(file_path)
    
    if not path_obj.exists():
        print(f"Error: File {file_path} not found")
        return

    # Simple loader logic (production should use the full DocumentProcessor)
    with open(path_obj, "r", encoding="utf-8", errors="ignore") as f:
        content = f.read()

    req = IngestRequest(
        content=content,
        source=str(path_obj.name),
        title=path_obj.stem,
        category=source_tag,
        metadata={"local_path": str(path_obj)}
    )

    print(f"Ingesting {path_obj.name}...")
    try:
        result = await pipeline.ingest_document(req)
        print(f"Success: {result['document_id']} ({result['chunks_created']} chunks)")
    except Exception as e:
        print(f"Failed: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Manual Ingestion Script")
    parser.add_argument("file", help="Path to file to ingest")
    parser.add_argument("--tag", default="manual", help="Category tag")
    
    args = parser.parse_args()
    asyncio.run(ingest_file(args.file, args.tag))