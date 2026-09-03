#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
4o Vibe Benchmark – Final Leaderboard Generator
================================================================================

This script combines the LLM judge scores (behavioral vibe) and embedding
similarity scores (semantic content) into a single weighted final leaderboard.

It reads all per-file scores from the LLM judge and embedding scorer outputs,
averages them per model, normalizes the embedding scores (since they are tightly
clustered), and produces a combined leaderboard with a configurable weight ratio.

================================================================================
HOW TO RUN
================================================================================

1.  INSTALL DEPENDENCIES
    --------------------
    No external dependencies – uses only Python standard library.

2.  CONFIGURE THE SCRIPT
    --------------------
    Edit the CONFIGURATION section below and set:
        - LLM_SCORE_DIR        : path to the LLM judge outputs (step 4)
        - EMBEDDING_SCORE_DIR  : path to the embedding similarity outputs (step 3)
        - OUTPUT_DIR           : where to save the final leaderboard
        - GOLD_MODEL           : the exact folder name of the gold standard model
        - LLM_WEIGHT           : weight for LLM judge (vibe) vs embedding (content)

3.  RUN THE SCRIPT
    ---------------
    python finalize_leaderboard.py

4.  OUTPUT
    -------
    - leaderboard_final.json : full JSON with all scores and metadata
    - leaderboard_final.txt  : clean text summary for Reddit/posting

================================================================================
WEIGHTING EXPLANATION
================================================================================

The final score is a weighted average of two metrics:

    Final = (Embedding_Norm × EMBED_WEIGHT) + (LLM_Raw × LLM_WEIGHT)

- Embedding scores (cosine similarity) are tightly clustered (~0.74–0.83).
  Min-Max normalization stretches them to 0–100 so they contribute meaningfully.

- LLM judge scores are already 0–100, so they are used raw.

- LLM_WEIGHT should be higher since this benchmark focuses on "vibe."
  Default: 85% Vibe (LLM) + 15% Content (Embedding).

================================================================================
WHAT THE SCRIPT DOES
================================================================================

1. Loads all LLM judge scores from LLM_SCORE_DIR
2. Loads all embedding similarity scores from EMBEDDING_SCORE_DIR
3. Finds models that have BOTH scores (intersection)
4. Min-Max normalizes the embedding scores (0–100)
5. Keeps LLM scores raw (already 0–100)
6. Computes weighted final score for each model
7. Sorts and saves the leaderboard
8. Excludes the gold model from the printed leaderboard (but keeps it in JSON)
"""

import os
import json
from pathlib import Path
from collections import defaultdict

# ============================================================================
#  CONFIGURATION  – ⚠️ EDIT THESE BEFORE RUNNING
# ============================================================================

# ---- Paths ----
# Directory containing the LLM judge output files (step 4)
LLM_SCORE_DIR = r"path/to/your/4-LLM-Score"          # e.g. D:\Benchmark\4- LLM Score

# Directory containing the embedding similarity output files (step 3)
EMBEDDING_SCORE_DIR = r"path/to/your/3-Scored"       # e.g. D:\Benchmark\3- Scored

# Directory where the final leaderboard will be saved
OUTPUT_DIR = r"path/to/your/5-Final"                 # e.g. D:\Benchmark\5- Final

# ---- Models ----
# The folder name of the gold standard model (must match exactly)
GOLD_MODEL = "gpt-4o-2024-11-20"

# ---- Weighting ----
# LLM_WEIGHT: how much the judge's "vibe" score matters (0.0 to 1.0)
# EMBED_WEIGHT: how much the semantic "content" score matters
# These should sum to 1.0.
# Default: 85% Vibe (LLM) + 15% Content (Embedding)
LLM_WEIGHT = 0.85
EMBED_WEIGHT = 1.0 - LLM_WEIGHT


# ============================================================================
#  SETUP OUTPUT DIR
# ============================================================================

OUTPUT_DIR_PATH = Path(OUTPUT_DIR)
OUTPUT_DIR_PATH.mkdir(parents=True, exist_ok=True)

print("=" * 70)
print(f"🏆 FINAL LEADERBOARD GENERATOR")
print(f"   Gold Standard: {GOLD_MODEL}")
print(f"   Weighting: {LLM_WEIGHT*100:.0f}% LLM (Vibe, raw) + {EMBED_WEIGHT*100:.0f}% Embedding (Content, min-max)")
print("=" * 70)


# ============================================================================
#  1. LOAD ALL LLM SCORES (RAW)
# ============================================================================

print("\n📊 Loading LLM scores...")
llm_model_scores = defaultdict(list)

llm_root = Path(LLM_SCORE_DIR)
if not llm_root.exists():
    print(f"❌ LLM score directory not found: {llm_root}")
    exit(1)

for model_folder in llm_root.iterdir():
    if not model_folder.is_dir():
        continue
    model_name = model_folder.name
    for file in model_folder.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            score = data.get("score")
            if score is not None:
                llm_model_scores[model_name].append(float(score))
        except Exception as e:
            print(f"⚠️ Failed to read {file}: {e}")

# Compute LLM averages (raw, unchanged)
llm_averages = {}
for model, scores in llm_model_scores.items():
    if scores:
        llm_averages[model] = sum(scores) / len(scores)

print(f"   Found {len(llm_averages)} models with LLM scores.")


# ============================================================================
#  2. LOAD ALL EMBEDDING SCORES
# ============================================================================

print("\n📊 Loading Embedding scores...")
embedding_model_scores = defaultdict(list)

embed_root = Path(EMBEDDING_SCORE_DIR)
if not embed_root.exists():
    print(f"❌ Embedding score directory not found: {embed_root}")
    exit(1)

for model_folder in embed_root.iterdir():
    if not model_folder.is_dir():
        continue
    model_name = model_folder.name
    for file in model_folder.glob("*.json"):
        try:
            with open(file, "r", encoding="utf-8") as f:
                data = json.load(f)
            avg_sim = data.get("average_similarity")
            if avg_sim is not None:
                embedding_model_scores[model_name].append(float(avg_sim))
        except Exception as e:
            print(f"⚠️ Failed to read {file}: {e}")

# Compute Embedding averages
embedding_averages = {}
for model, scores in embedding_model_scores.items():
    if scores:
        embedding_averages[model] = sum(scores) / len(scores)

print(f"   Found {len(embedding_averages)} models with Embedding scores.")


# ============================================================================
#  3. MERGE MODELS (only those with BOTH scores)
# ============================================================================

all_models = set(llm_averages.keys()) & set(embedding_averages.keys())
print(f"\n🔗 Merged {len(all_models)} models with both scores.")

if not all_models:
    print("❌ No common models found. Check your folder paths.")
    exit()


# ============================================================================
#  4. MIN-MAX NORMALIZE ONLY EMBEDDING
# ============================================================================
# Embedding raw cosine scores are clustered (e.g., 0.74–0.83).
# Stretching them to 0–100 ensures they contribute meaningfully when weighted.
embedding_values = [embedding_averages[m] for m in all_models]
emb_min, emb_max = min(embedding_values), max(embedding_values)
emb_range = emb_max - emb_min if emb_max - emb_min > 0 else 1

# LLM is NOT normalized – we keep the raw scores (already 0-100).


# ============================================================================
#  5. BUILD FINAL SCORES
# ============================================================================

final_scores = {}

for model in all_models:
    # Normalize Embedding to 0-100
    norm_emb = (embedding_averages[model] - emb_min) / emb_range * 100

    # LLM raw score (unchanged)
    llm_raw = llm_averages[model]

    # Clamp just in case
    norm_emb = max(0.0, min(100.0, norm_emb))
    llm_raw = max(0.0, min(100.0, llm_raw))

    # Weighted final score
    final_score = (norm_emb * EMBED_WEIGHT) + (llm_raw * LLM_WEIGHT)

    final_scores[model] = {
        "embedding_raw": embedding_averages[model],
        "embedding_norm": norm_emb,
        "llm_raw": llm_raw,
        "final_score": final_score,
        "embedding_samples": len(embedding_model_scores[model]),
        "llm_samples": len(llm_model_scores[model]),
    }


# ============================================================================
#  6. SORT LEADERBOARD
# ============================================================================

sorted_models = sorted(
    final_scores.items(),
    key=lambda x: x[1]["final_score"],
    reverse=True
)


# ============================================================================
#  7. FILTER OUT GOLD MODEL FOR PRINTING
# ============================================================================
# Keep gold in calculations, but remove it from the printed leaderboard
display_models = [(m, s) for m, s in sorted_models if m != GOLD_MODEL]


# ============================================================================
#  8. PRINT TO CONSOLE (Gold removed, no sample count)
# ============================================================================

print("\n" + "=" * 80)
print(f"🏆 Full GPT-4o Similarity Leaderboard")
print(f"   vs: {GOLD_MODEL}")
print(f"   Weighting: {LLM_WEIGHT*100:.0f}% Vibe (LLM Analysis) + {EMBED_WEIGHT*100:.0f}% Content (Embeddings)")
print("=" * 80)
print(f"{'Rank':<5} {'Model':<42} {'Final':<9} {'Embed':<9} {'LLM':<9}")
print("-" * 80)

for rank, (model, scores) in enumerate(display_models, 1):
    print(f"{rank:<5} {model[:41]:<42} {scores['final_score']:>6.1f}%   {scores['embedding_norm']:>6.1f}%   {scores['llm_raw']:>6.1f}%")


# ============================================================================
#  9. SAVE JSON (includes all models, including gold)
# ============================================================================

leaderboard_json = {
    "metadata": {
        "gold_standard": GOLD_MODEL,
        "embedding_model": "qwen/qwen3-embedding-8b",
        "llm_judge": "zai-org/glm-5.2:thinking",
        "embedding_normalization": "min-max (0-100)",
        "llm_normalization": "raw (0-100, unchanged)",
        "weighting": {
            "embedding": EMBED_WEIGHT,
            "llm": LLM_WEIGHT,
            "description": f"{LLM_WEIGHT*100:.0f}% Vibe (LLM raw) + {EMBED_WEIGHT*100:.0f}% Content (Embedding min-max)",
        },
        "total_models": len(sorted_models),
        "models_displayed": len(display_models),
    },
    "leaderboard": [
        {
            "rank": rank,
            "model": model,
            "final_score": round(scores["final_score"], 4),
            "embedding_score": round(scores["embedding_norm"], 4),
            "llm_score": round(scores["llm_raw"], 4),
            "embedding_raw": round(scores["embedding_raw"], 6),
            "llm_raw": round(scores["llm_raw"], 4),
            "embedding_samples": scores["embedding_samples"],
            "llm_samples": scores["llm_samples"],
        }
        for rank, (model, scores) in enumerate(sorted_models, 1)
    ],
}

output_json = OUTPUT_DIR_PATH / "leaderboard_final.json"
with open(output_json, "w", encoding="utf-8") as f:
    json.dump(leaderboard_json, f, indent=2, ensure_ascii=False)


# ============================================================================
#  10. SAVE TEXT SUMMARY (Gold removed, no sample count)
# ============================================================================

output_txt = OUTPUT_DIR_PATH / "leaderboard_final.txt"
with open(output_txt, "w", encoding="utf-8") as f:
    f.write("=" * 80 + "\n")
    f.write(f"🏆 Full GPT-4o Similarity Leaderboard\n")
    f.write(f"   vs: {GOLD_MODEL}\n")
    f.write(f"   Weighting: {LLM_WEIGHT*100:.0f}% Vibe (LLM raw) + {EMBED_WEIGHT*100:.0f}% Content (Embedding min-max)\n")
    f.write("=" * 80 + "\n")
    f.write(f"{'Rank':<5} {'Model':<42} {'Final':<9} {'Embed':<9} {'LLM':<9}\n")
    f.write("-" * 80 + "\n")
    for rank, (model, scores) in enumerate(display_models, 1):
        f.write(f"{rank:<5} {model[:41]:<42} {scores['final_score']:>6.1f}%   {scores['embedding_norm']:>6.1f}%   {scores['llm_raw']:>6.1f}%\n")


# ============================================================================
#  DONE
# ============================================================================

print("\n" + "=" * 80)
print("✅ FINAL LEADERBOARD GENERATED")
print("=" * 80)
print(f"📄 JSON: {output_json}")
print(f"📄 TXT:  {output_txt}")
print("\n📐 Scoring breakdown:")
print(f"   Embedding = min-max(cosine) → 0-100")
print(f"   LLM       = raw judge score (0-100, unchanged)")
print(f"   Final     = {EMBED_WEIGHT*100:.0f}% Embedding + {LLM_WEIGHT*100:.0f}% LLM")
print(f"\n📊 Showing {len(display_models)} models (excluded {GOLD_MODEL})")
print("=" * 80)