#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
4o Vibe Benchmark – Embedding Generation
================================================================================

This script processes all JSON conversation files from the data collection phase
and generates embeddings for every assistant response using Qwen3-Embedding-8B
via the OpenRouter API.

The output mirrors the input directory structure: each input file produces a
corresponding output file containing the same messages plus their embeddings.

================================================================================
HOW TO RUN
================================================================================

1.  INSTALL DEPENDENCIES
    --------------------
    pip install aiohttp httpx

2.  GET AN OPENROUTER API KEY
    -------------------------
    - Sign up at https://openrouter.ai
    - Generate an API key at https://openrouter.ai/keys
    - It will look like: sk-or-v1-...

3.  CONFIGURE THE SCRIPT
    --------------------
    Edit the CONFIGURATION section below and set:
        - OPENROUTER_API_KEY   : your OpenRouter API key
        - SOURCE_DIR           : where your raw JSON files are (step 1 output)
        - OUTPUT_DIR           : where to save the embedded files

4.  RUN THE SCRIPT
    ---------------
    python embed.py

5.  OUTPUT
    -------
    The script will recursively process all JSON files under SOURCE_DIR,
    mirror the folder structure under OUTPUT_DIR, and save embedded versions.

================================================================================
WHAT THE SCRIPT DOES
================================================================================

For each JSON file:
    1. Loads the file
    2. Extracts all assistant messages (skipping errors)
    3. Splits messages into batches (BATCH_SIZE = 50)
    4. Sends each batch to OpenRouter for embedding
    5. Saves the original messages + embeddings to the output path

If an output file already exists, it is skipped (resume capability).
If a batch fails, it retries up to RETRY_ATTEMPTS times.
"""

import os
import json
import asyncio
from pathlib import Path
from typing import List

# Import the OpenRouter client
from openrouter_client import OpenRouterClient

# ============================================================================
#  CONFIGURATION  – ⚠️ EDIT THESE BEFORE RUNNING
# ============================================================================

# ---- OpenRouter API ----
# Get your key from https://openrouter.ai/keys
OPENROUTER_API_KEY = "sk-or-v1-..."  # 👈 REPLACE WITH YOUR ACTUAL KEY

# OpenRouter embedding endpoint
OPENROUTER_EMBEDDING_URL = "https://openrouter.ai/api/v1/embeddings"

# The embedding model to use
# Qwen3-Embedding-8B is the recommended choice for this benchmark:
# - Excellent quality-to-price ratio ($0.01 per 1M tokens)
# - 33K token context window
# - Strong multilingual and reasoning capabilities
EMBEDDING_MODEL = "qwen/qwen3-embedding-8b"

# ---- Paths ----
# Source: where your raw conversation JSON files are stored
SOURCE_DIR = r"path/to/your/1-Raw"          # e.g. D:\Benchmark\1- Raw

# Output: where to save the embedded JSON files
OUTPUT_DIR = r"path/to/your/2-Embedded"      # e.g. D:\Benchmark\2- Embedded

# ---- Performance Settings ----
# Max number of texts to send in a single API call (OpenRouter supports up to 512)
BATCH_SIZE = 50

# Max concurrent API requests (avoid rate limits)
MAX_CONCURRENT = 10

# Retry settings
RETRY_ATTEMPTS = 3
RETRY_DELAY = 5  # seconds between retries


# ============================================================================
#  EMBEDDING SESSION
# ============================================================================

class EmbeddingSession:
    """
    Manages embedding requests using the OpenRouterClient with concurrency control.
    """

    def __init__(self, api_key: str):
        self.client = OpenRouterClient(api_key=api_key)
        self.semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    async def embed_texts(self, texts: List[str]) -> List[List[float]]:
        """
        Embed a batch of texts using the OpenRouter embeddings endpoint.

        Args:
            texts: List of text strings to embed.

        Returns:
            List of embedding vectors (each vector is a list of floats).

        Raises:
            Exception: On API failure after all retry attempts.
        """
        if not texts:
            return []

        async with self.semaphore:
            payload = {
                "model": EMBEDDING_MODEL,
                "input": texts,
            }

            for attempt in range(RETRY_ATTEMPTS):
                try:
                    # Use the underlying httpx client from OpenRouterClient
                    resp = await self.client.client.post(
                        OPENROUTER_EMBEDDING_URL,
                        json=payload,
                        headers=self.client.headers,
                    )

                    if resp.status_code != 200:
                        error_text = resp.text
                        raise Exception(f"HTTP {resp.status_code}: {error_text}")

                    data = resp.json()
                    # OpenRouter returns: {"data": [{"embedding": [...]}, ...]}
                    embeddings = [item["embedding"] for item in data["data"]]
                    return embeddings

                except Exception as e:
                    if attempt < RETRY_ATTEMPTS - 1:
                        await asyncio.sleep(RETRY_DELAY * (attempt + 1))
                    else:
                        raise e

    async def close(self):
        """Close the underlying client."""
        await self.client.close()


# ============================================================================
#  FILE PROCESSING
# ============================================================================

async def process_file(
    file_path: Path,
    session: EmbeddingSession,
):
    """
    Process a single JSON file: extract assistant messages, embed them, save.

    Args:
        file_path: Path to the input JSON file.
        session: The embedding session.
    """
    # Determine output path (mirroring the source structure)
    relative_path = file_path.relative_to(SOURCE_DIR)
    output_file = OUTPUT_DIR / relative_path

    # Skip if already processed (resume capability)
    if output_file.exists():
        print(f"⏭️ Skipping {file_path.name} - output already exists")
        return

    # ----- Load the file -----
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception as e:
        print(f"❌ Failed to read {file_path}: {e}")
        return

    # ----- Handle both dict and list formats -----
    if isinstance(data, dict):
        messages = data.get("messages", [])
        model_name = data.get("model")
        dataset_name = data.get("dataset")
    elif isinstance(data, list):
        # Some files were saved as a raw messages array
        messages = data
        model_name = file_path.parent.name
        dataset_name = file_path.stem
    else:
        print(f"⚠️ Unexpected data type in {file_path}: {type(data)}")
        return

    # ----- Extract assistant messages (skip errors) -----
    assistant_messages = []
    for msg in messages:
        if isinstance(msg, dict) and msg.get("role") == "assistant":
            content = msg.get("content", "").strip()
            # Skip placeholders from failed turns
            if content and not content.startswith("[ERROR:"):
                assistant_messages.append(content)

    if not assistant_messages:
        print(f"⚠️ No valid assistant messages in {file_path}")
        return

    # ----- Embed in batches -----
    all_embeddings = []
    for i in range(0, len(assistant_messages), BATCH_SIZE):
        batch = assistant_messages[i : i + BATCH_SIZE]
        try:
            embeddings = await session.embed_texts(batch)
            all_embeddings.extend(embeddings)
        except Exception as e:
            print(f"❌ Failed to embed batch for {file_path}: {e}")
            return

    # ----- Build output structure -----
    output_data = {
        "source_file": str(file_path),
        "model": model_name,
        "dataset": dataset_name,
        "assistant_responses": [
            {"text": text, "embedding": emb}
            for text, emb in zip(assistant_messages, all_embeddings)
        ],
    }

    # ----- Save output -----
    output_file.parent.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)

    print(f"✅ Embedded {len(assistant_messages)} responses from {file_path.name}")


# ============================================================================
#  MAIN
# ============================================================================

async def main():
    """Main entry point: scans SOURCE_DIR and processes all JSON files."""
    source_root = Path(SOURCE_DIR)

    if not source_root.exists():
        print(f"❌ Source directory not found: {source_root}")
        return

    # Gather all JSON files (recursive)
    json_files = list(source_root.rglob("*.json"))
    print(f"📂 Found {len(json_files)} JSON files to process.")

    if not json_files:
        print("No files to process.")
        return

    # Create embedding session
    session = EmbeddingSession(OPENROUTER_API_KEY)

    try:
        for idx, file_path in enumerate(json_files, 1):
            print(f"\n[{idx}/{len(json_files)}] Checking {file_path.name} ...")
            await process_file(file_path, session)
    finally:
        await session.close()

    print("\n🎉 All files processed!")


if __name__ == "__main__":
    asyncio.run(main())