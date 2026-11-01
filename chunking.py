import os
import re
from dotenv import load_dotenv

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", 500))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", 50))

MIN_CHUNK_LENGTH = 10  # skip tiny leftover fragments from the sliding window


def chunk_text(text: str, chunk_size: int = None, overlap: int = None) -> list[str]:
    chunk_size = chunk_size or CHUNK_SIZE
    overlap = overlap or CHUNK_OVERLAP

    # Split the text into alternating "normal text" and "whole table" segments
    parts = re.split(r"(\[\[TABLE_START\]\].*?\[\[TABLE_END\]\])", text, flags=re.DOTALL)

    chunks = []
    for part in parts:
        if not part.strip():
            continue

        if part.startswith("[[TABLE_START]]"):
            # Tables are never split — the whole table becomes exactly one chunk
            clean_table = part.replace("[[TABLE_START]]", "").replace("[[TABLE_END]]", "").strip()
            if len(clean_table) > MIN_CHUNK_LENGTH:
                chunks.append(clean_table)
        else:
            # Normal sliding-window chunking, applied only to non-table text
            start = 0
            while start < len(part):
                end = start + chunk_size
                chunk = part[start:end].strip()
                if len(chunk) > MIN_CHUNK_LENGTH:
                    chunks.append(chunk)
                start += chunk_size - overlap

    return chunks


if __name__ == "__main__":
    from extraction import extract_text
    text = extract_text("data/documents/vacation_policy.docx")
    chunks = chunk_text(text)

    print(f"Total chunks: {len(chunks)}")
    for i, c in enumerate(chunks):
        print(f"\n--- Chunk {i} ({len(c)} chars) ---")
        print(c)