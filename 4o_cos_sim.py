#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
4o Vibe Benchmark – Embedding Similarity Scoring
================================================================================

This script compares every model's embedded responses against a gold standard
(GPT-4o) using cosine similarity on the embedding vectors.

It processes each file individually and generates per-file scores, then
aggregates them into a final leaderboard.

================================================================================
HOW TO RUN
================================================================================

1.  INSTALL DEPENDENCIES
    --------------------
    pip install numpy

2.  CONFIGURE THE SCRIPT
    --------------------
    Edit the CONFIGURATION section below and set:
        - EMBEDDED_DIR  : path to the embedded JSON files (step 2 output)
        - OUTPUT_DIR    : path where scored results should be saved
        - GOLD_MODEL    : the exact folder name of the gold standard model

3.  RUN THE SCRIPT
    ---------------
    python similarity_scorer.py

4.  OUTPUT
    -------
    - A folder for each model (except the gold model) inside OUTPUT_DIR
    - Per-file JSON: {model, gold_model, dataset, average_similarity, turn_similarities, assistant_responses}
    - A global leaderboard: leaderboard.json and leaderboard.txt

================================================================================
WHAT THE SCRIPT DOES
================================================================================

For each model folder (excluding the gold model):
    1. For each dataset file in the gold folder:
       - Finds the corresponding file in the model folder
       - Aligns assistant responses by turn index
       - Computes cosine similarity between each pair of embeddings
       - Averages the similarities for that file
    2. Averages all file scores to get a model-level score
    3. Saves per-file results and a final leaderboard
"""

import os
import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple


# ============================================================================
#  CONFIGURATION  – ⚠️ EDIT THESE BEFORE RUNNING
# ============================================================================

# ---- Paths ----
# Source: where the embedded JSON files are stored (step 2 output)
EMBEDDED_DIR = r"path/to/your/2-Embedded"   # e.g. D:\Benchmark\2- Embedded

# Output: where to save the similarity scores
OUTPUT_DIR = r"path/to/your/3-Scored"       # e.g. D:\Benchmark\3- Scored

# ---- Gold Standard ----
# The folder name of the model you want to compare everything against.
# This must match the folder name exactly (case-sensitive).
GOLD_MODEL = "gpt-4o-2024-11-20"            # e.g. "gpt-4o-2024-11-20"


# ============================================================================
#  COSINE SIMILARITY
# ============================================================================

def cosine_similarity(vec_a: List[float], vec_b: List[float]) -> float:
    """
    Compute cosine similarity between two vectors.

    Args:
        vec_a: First embedding vector.
        vec_b: Second embedding vector.

    Returns:
        Cosine similarity (0.0 to 1.0, or 0.0 if either vector is zero).
    """
    a = np.array(vec_a)
    b = np.array(vec_b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return float(np.dot(a, b) / (norm_a * norm_b))


# ============================================================================
#  LOAD & COMPARE
# ============================================================================

def load_embeddings(file_path: Path) -> Dict:
    """Load embeddings from a JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def compare_files(
    gold_file: Path,
    model_file: Path,
) -> Tuple[float, List[float], List[str]]:
    """
    Compare embeddings between gold and model files.

    Args:
        gold_file: Path to the gold standard file.
        model_file: Path to the candidate model file.

    Returns:
        Tuple: (average_similarity, list_of_turn_similarities, list_of_texts)
    """
    gold_data = load_embeddings(gold_file)
    model_data = load_embeddings(model_file)

    gold_responses = gold_data.get("assistant_responses", [])
    model_responses = model_data.get("assistant_responses", [])

    # Align by turn index (assumes both files have the same number of turns)
    min_len = min(len(gold_responses), len(model_responses))
    similarities = []
    texts = []

    for i in range(min_len):
        gold_emb = gold_responses[i].get("embedding")
        model_emb = model_responses[i].get("embedding")
        text = model_responses[i].get("text", "")

        if gold_emb is not None and model_emb is not None:
            sim = cosine_similarity(gold_emb, model_emb)
            similarities.append(sim)
            texts.append(text)

    if not similarities:
        return 0.0, [], []

    avg_similarity = sum(similarities) / len(similarities)
    return avg_similarity, similarities, texts


# ============================================================================
#  MAIN PROCESSING LOOP
# ============================================================================

def process_all_models():
    """Process all models and compare to the gold standard."""
    embedded_root = Path(EMBEDDED_DIR)
    output_root = Path(OUTPUT_DIR)

    # ------------------------------------------------------------------------
    # 1. Locate gold model folder
    # ------------------------------------------------------------------------
    gold_folder = embedded_root / GOLD_MODEL
    if not gold_folder.exists():
        print(f"❌ Gold model folder not found: {gold_folder}")
        print(f"   Make sure GOLD_MODEL matches the folder name in {EMBEDDED_DIR}")
        return

    # Get all gold files
    gold_files = list(gold_folder.glob("*.json"))
    print(f"📄 Found {len(gold_files)} gold files in {gold_folder.name}")

    # ------------------------------------------------------------------------
    # 2. Get all model folders (excluding gold)
    # ------------------------------------------------------------------------
    model_folders = [
        f for f in embedded_root.iterdir()
        if f.is_dir() and f.name != GOLD_MODEL
    ]
    print(f"📂 Found {len(model_folders)} model folders to evaluate")

    # ------------------------------------------------------------------------
    # 3. Process each model
    # ------------------------------------------------------------------------
    all_results = {}
    total_skipped = 0
    total_processed = 0

    for model_folder in model_folders:
        model_name = model_folder.name
        print(f"\n{'='*60}")
        print(f"📊 Processing model: {model_name}")
        print(f"{'='*60}")

        output_folder = output_root / model_name
        output_folder.mkdir(parents=True, exist_ok=True)

        model_results = []

        for gold_file in gold_files:
            dataset_name = gold_file.name
            model_file = model_folder / dataset_name
            output_file = output_folder / dataset_name

            # Skip if output already exists (resume capability)
            if output_file.exists():
                print(f"⏭️ Skipping {dataset_name} – output already exists")
                total_skipped += 1
                continue

            # Check if model has this file
            if not model_file.exists():
                print(f"⚠️ Missing {dataset_name}")
                continue

            # Compare embeddings
            avg_sim, similarities, texts = compare_files(gold_file, model_file)

            if not similarities:
                print(f"⚠️ No comparable responses for {dataset_name}")
                continue

            # Save per-file results
            result_data = {
                "model": model_name,
                "gold_model": GOLD_MODEL,
                "dataset": dataset_name,
                "average_similarity": avg_sim,
                "turn_similarities": similarities,
                "assistant_responses": texts,
            }

            with open(output_file, "w", encoding="utf-8") as f:
                json.dump(result_data, f, indent=2, ensure_ascii=False)

            print(f"✅ {dataset_name}: avg similarity = {avg_sim:.4f} ({len(similarities)} turns)")
            total_processed += 1

            model_results.append({
                "dataset": dataset_name,
                "avg_sim": avg_sim,
                "turn_count": len(similarities),
            })

        all_results[model_name] = model_results

    # ------------------------------------------------------------------------
    # 4. Generate leaderboard
    # ------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📊 GENERATING LEADERBOARD")
    print("=" * 60)

    # Compute average per model
    model_averages = {}
    for model_name, results in all_results.items():
        if results:
            avg_score = sum(r["avg_sim"] for r in results) / len(results)
            model_averages[model_name] = avg_score

    if not model_averages:
        print("❌ No results to generate leaderboard. Check your data.")
        return

    sorted_models = sorted(model_averages.items(), key=lambda x: x[1], reverse=True)

    # Print leaderboard
    print("\n🏆 LEADERBOARD (Average Similarity to GPT-4o):")
    print("-" * 60)
    for rank, (model_name, score) in enumerate(sorted_models, 1):
        bar = "█" * int(score * 50)
        print(f"{rank:2d}. {model_name[:40]:40s} {score:.4f}  {bar}")

    # Save JSON leaderboard
    leaderboard_data = {
        "gold_model": GOLD_MODEL,
        "total_models": len(sorted_models),
        "total_files_processed": total_processed,
        "total_files_skipped": total_skipped,
        "leaderboard": [
            {"rank": rank, "model": model_name, "similarity": score}
            for rank, (model_name, score) in enumerate(sorted_models, 1)
        ],
    }

    leaderboard_json = output_root / "leaderboard.json"
    with open(leaderboard_json, "w", encoding="utf-8") as f:
        json.dump(leaderboard_data, f, indent=2, ensure_ascii=False)
    print(f"\n✅ Leaderboard saved to: {leaderboard_json}")

    # Save text summary
    leaderboard_txt = output_root / "leaderboard.txt"
    with open(leaderboard_txt, "w", encoding="utf-8") as f:
        f.write("=" * 60 + "\n")
        f.write("LEADERBOARD: All Models vs GPT-4o\n")
        f.write("=" * 60 + "\n\n")
        for rank, (model_name, score) in enumerate(sorted_models, 1):
            f.write(f"{rank:2d}. {model_name:40s} {score:.4f}\n")
    print(f"✅ Summary saved to: {leaderboard_txt}")

    # ------------------------------------------------------------------------
    # 5. Final summary
    # ------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("🎉 ALL SIMILARITY SCORING COMPLETE!")
    print("=" * 60)
    print(f"   ✅ Processed: {total_processed} files")
    print(f"   ⏭️ Skipped:   {total_skipped} files (already existed)")
    print(f"   📊 Models:    {len(sorted_models)}")
    print("=" * 60)


# ============================================================================
#  MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    process_all_models()