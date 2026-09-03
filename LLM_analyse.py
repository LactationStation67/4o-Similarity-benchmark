#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
4o Vibe Benchmark – LLM Judge Scoring
================================================================================

This script uses a frontier LLM (GLM-5.2 thinking) as a judge to evaluate
the behavioral and stylistic similarity of candidate model responses against
a gold standard (GPT-4o).

Unlike embedding similarity (which measures semantic content), the LLM judge
evaluates conversational vibe: tone, personality, empathy, naturalness, and flow.

================================================================================
HOW TO RUN
================================================================================

1.  INSTALL DEPENDENCIES
    --------------------
    pip install httpx orjson

2.  SET UP THE PROVIDER CLIENT
    ---------------------------
    This script expects `nanogpt_client.py` to be in the same directory or
    on your PYTHONPATH. It must export `NanoGPTClient`.
    - If you're using the OpenRouter client instead, replace the import.
    - Update the `JUDGE_MODEL` variable to match a model available on your provider.

3.  CONFIGURE THE SCRIPT
    --------------------
    Edit the CONFIGURATION section below and set:
        - RAW_DIR         : path to the raw conversation JSON files (step 1 output)
        - OUTPUT_DIR      : path where judge scores should be saved
        - GOLD_MODEL      : the exact folder name of the gold standard model
        - JUDGE_MODEL     : the model ID of the judge (e.g., "zai-org/glm-5.2:thinking")

4.  RUN THE SCRIPT
    ---------------
    python llm_judge.py

5.  OUTPUT
    -------
    - A folder for each model (except the gold model) inside OUTPUT_DIR
    - Per-file JSON: {gold_model, candidate_model, dataset, judge_model, attempts, score, raw_judge_output}
    - If the judge fails after MAX_JUDGE_ATTEMPTS, an error record is saved instead.

================================================================================
WHAT THE SCRIPT DOES
================================================================================

For each model folder (excluding the gold model):
    1. For each dataset file in the gold folder:
       - Loads the gold conversation and the candidate conversation
       - Formats both into a readable User/Assistant dialogue
       - Sends the prompt to the judge LLM
       - Parses the score from the response
       - Saves the result (or error) to the output folder
    2. If a file already exists with a valid score, it is skipped (resume capability).
    3. If a file fails, it retries up to MAX_JUDGE_ATTEMPTS times.
"""

import os
import json
import asyncio
import re
from pathlib import Path
from typing import Dict, List, Optional

# ============================================================================
#  PROVIDER CLIENT – Replace with your own
# ============================================================================
# This script uses the OpenRouterClient
# provider, replace this import with your own client.
from openrouter_client import OpneRouterClient


# ============================================================================
#  CONFIGURATION  – ⚠️ EDIT THESE BEFORE RUNNING
# ============================================================================

# ---- Paths ----
# Source: where the raw conversation JSON files are stored (step 1 output)
RAW_DIR = r"path/to/your/1-Raw"              # e.g. D:\Benchmark\1- Raw

# Output: where to save the judge scores
OUTPUT_DIR = r"path/to/your/4-LLM-Score"     # e.g. D:\Benchmark\4- LLM Score

# ---- Models ----
# The folder name of the gold standard model (must match exactly)
GOLD_MODEL = "gpt-4o-2024-11-20"

# The judge model ID (must be available on your provider)
JUDGE_MODEL = "zai-org/glm-5.2:thinking"

# ---- Retry Settings ----
# Maximum attempts per file (the judge may fail to output a parseable score)
MAX_JUDGE_ATTEMPTS = 30

# Delay (in seconds) between retry attempts
RETRY_DELAY = 5


# ============================================================================
#  SYSTEM PROMPT FOR THE JUDGE
# ============================================================================

JUDGE_SYSTEM_PROMPT = """You are an expert evaluator of AI conversational behavior.

Compare the Candidate Response to the Gold Standard and score **how behaviorally similar they are**, from 0–100. Do not judge whether the Candidate is better or worse overall. A response can be equally correct and informative while still being very different in conversational style.

Evaluate:

* **Tone & Personality (25):** warmth, register, humor, enthusiasm, seriousness, personality, and overall vibe.
* **Responsiveness & Adaptation (25):** how specifically it reacts to the user's message, framing, emotions, implications, and conversational thread.
* **Naturalness & Flow (25):** conversational rhythm, phrasing, structure, pacing, and whether it feels natural rather than generic or robotic.
* **Helpfulness & Strategy (25):** relevance, detail, explanation style, and whether it approaches the user's request in a similar way.

Focus on **behavioral and stylistic similarity**, not word overlap. Do not reward superficial tricks such as emojis, questions, or casual phrases unless they genuinely match the Gold Standard.

Use the full scale. 90+ means extremely close; 70–89 means clearly similar; 50–69 means moderately similar; below 50 means substantially different. Do not give 100 unless the responses are effectively indistinguishable in behavior.

Briefly justify each criterion in one sentence.

End exactly with:
Overall score: [score]/100

--- Gold Standard ---
{best_response}

--- Candidate ---
{candidate_response}

--- Evaluation ---
"""


# ============================================================================
#  HELPERS
# ============================================================================

def format_conversation(messages: List[Dict]) -> str:
    """
    Format messages into a readable conversation string, excluding system messages.

    Args:
        messages: List of message dicts with "role" and "content".

    Returns:
        A formatted string with "User:" and "Assistant:" labels.
    """
    lines = []
    for msg in messages:
        role = msg.get("role", "").lower()
        if role == "system":
            continue
        content = msg.get("content", "").strip()
        if not content:
            continue
        if role == "user":
            lines.append(f"User: {content}")
        elif role == "assistant":
            lines.append(f"Assistant: {content}")
        else:
            lines.append(f"{role.capitalize()}: {content}")
    return "\n\n".join(lines)


def extract_score(text: str) -> Optional[int]:
    """
    Extract the "Overall score: X/100" from the judge's response.

    Args:
        text: The raw judge output.

    Returns:
        An integer score, or None if not found.
    """
    pattern = r"Overall score:\s*(\d+)(?:/100)?"
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        return int(match.group(1))
    return None


# ============================================================================
#  PROCESS SINGLE FILE (with retries)
# ============================================================================

async def judge_file(
    client: NanoGPTClient,
    gold_file_path: Path,
    model_file_path: Path,
    output_path: Path,
):
    """
    Judge a single dataset for one model against the gold standard.

    Args:
        client: The API client instance.
        gold_file_path: Path to the gold conversation file.
        model_file_path: Path to the candidate conversation file.
        output_path: Path where the result should be saved.
    """
    # ----- Load gold conversation -----
    with open(gold_file_path, "r", encoding="utf-8") as f:
        gold_data = json.load(f)

    if isinstance(gold_data, dict):
        gold_messages = gold_data.get("messages", [])
    elif isinstance(gold_data, list):
        gold_messages = gold_data
    else:
        print(f"⚠️ Unexpected gold data type: {type(gold_data)}")
        return

    # ----- Load candidate conversation -----
    with open(model_file_path, "r", encoding="utf-8") as f:
        model_data = json.load(f)

    if isinstance(model_data, dict):
        model_messages = model_data.get("messages", [])
    elif isinstance(model_data, list):
        model_messages = model_data
    else:
        print(f"⚠️ Unexpected model data type: {type(model_data)}")
        return

    # Skip if either is empty
    if not gold_messages or not model_messages:
        print(f"⚠️ Skipping {model_file_path.parent.name}/{model_file_path.name} - empty conversation")
        return

    # ----- Format conversations -----
    best_response = format_conversation(gold_messages)
    candidate_response = format_conversation(model_messages)

    # ----- Build prompt -----
    prompt = JUDGE_SYSTEM_PROMPT.format(
        best_response=best_response,
        candidate_response=candidate_response,
    )

    # ----- Retry loop -----
    for attempt in range(1, MAX_JUDGE_ATTEMPTS + 1):
        try:
            response = await client.generate(
                model=JUDGE_MODEL,
                user_prompt=prompt,
                temperature=0.3,
                max_tokens=4096,
            )

            if isinstance(response, dict) and response.get("error"):
                raise Exception(response.get("error"))

            score = extract_score(response)
            if score is None:
                raise Exception(f"Could not parse score from response: {response[:200]}...")

            # ----- Success: save and return -----
            output_data = {
                "gold_model": GOLD_MODEL,
                "candidate_model": model_file_path.parent.name,
                "dataset": model_file_path.name,
                "judge_model": JUDGE_MODEL,
                "attempts": attempt,
                "score": score,
                "raw_judge_output": response,
            }

            output_path.parent.mkdir(parents=True, exist_ok=True)
            with open(output_path, "w", encoding="utf-8") as f:
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"✅ {model_file_path.parent.name}/{model_file_path.name} → Score: {score}/100 (attempt {attempt})")
            return

        except Exception as e:
            print(f"⚠️ Attempt {attempt}/{MAX_JUDGE_ATTEMPTS} failed for {model_file_path.name}: {e}")
            if attempt < MAX_JUDGE_ATTEMPTS:
                await asyncio.sleep(RETRY_DELAY)
            else:
                # ----- Final failure: save error record -----
                error_data = {
                    "gold_model": GOLD_MODEL,
                    "candidate_model": model_file_path.parent.name,
                    "dataset": model_file_path.name,
                    "judge_model": JUDGE_MODEL,
                    "attempts": attempt,
                    "error": str(e),
                }
                output_path.parent.mkdir(parents=True, exist_ok=True)
                with open(output_path, "w", encoding="utf-8") as f:
                    json.dump(error_data, f, indent=2, ensure_ascii=False)

                print(f"❌ Failed {model_file_path.parent.name}/{model_file_path.name} after {attempt} attempts.")


# ============================================================================
#  MAIN
# ============================================================================

async def main():
    """Main entry point: scans RAW_DIR and processes all models."""
    raw_root = Path(RAW_DIR)
    output_root = Path(OUTPUT_DIR)

    # ----- Locate gold folder -----
    gold_folder = raw_root / GOLD_MODEL
    if not gold_folder.exists():
        print(f"❌ Gold folder not found: {gold_folder}")
        return

    gold_files = list(gold_folder.glob("*.json"))
    if not gold_files:
        print("❌ No gold files found.")
        return
    print(f"📄 Found {len(gold_files)} gold dataset files.")

    # ----- Get all model folders (excluding gold) -----
    model_folders = [f for f in raw_root.iterdir() if f.is_dir() and f.name != GOLD_MODEL]
    print(f"📁 Found {len(model_folders)} model folders to evaluate.")

    # ----- Initialize client -----
    # If you're using a different provider, replace this with your client.
    client = NanoGPTClient()

    # ----- Process each model -----
    for model_folder in model_folders:
        model_name = model_folder.name
        print(f"\n{'='*60}")
        print(f"📊 Processing model: {model_name}")
        print(f"{'='*60}")

        output_model_dir = output_root / model_name
        output_model_dir.mkdir(parents=True, exist_ok=True)

        for gold_file in gold_files:
            dataset_name = gold_file.name
            model_file = model_folder / dataset_name

            # Skip if model doesn't have this dataset
            if not model_file.exists():
                print(f"⚠️ {model_name} missing {dataset_name} – skipping")
                continue

            output_path = output_model_dir / dataset_name

            # Skip if already scored successfully
            if output_path.exists():
                with open(output_path, "r", encoding="utf-8") as f:
                    existing = json.load(f)
                if existing.get("score") is not None:
                    print(f"⏭️ Skipping {model_name}/{dataset_name} – already scored")
                    continue
                else:
                    print(f"🔄 Re-processing {model_name}/{dataset_name} – previous was error")

            # Process the file
            await judge_file(client, gold_file, model_file, output_path)

    print("\n🎉 All judging complete!")


# ============================================================================
#  MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    asyncio.run(main())