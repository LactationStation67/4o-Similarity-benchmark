#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
4o Vibe Benchmark – Data Collection Script
================================================================================

This script runs multi‑turn conversations across a list of models and datasets.
It uses the OpenRouterClient to query LLMs via the NanoGPT API.

Output: JSON files containing the full conversation history for each model‑dataset
combination, saved under a model‑specific folder.

================================================================================
HOW TO RUN
================================================================================

1.  INSTALL DEPENDENCIES
    --------------------
    pip install asyncio httpx orjson uvicorn fastapi

2.  SET UP YOUR PROVIDER CLIENT
    ---------------------------
    Make sure `nanogpt_client.py` is in the same directory or your PYTHONPATH.
    It must export `OpenRouterClient` (or you can adapt the import below).

3.  CONFIGURE THE SCRIPT
    --------------------
    Edit the CONFIGURATION section below and set:
        - DATASETS_DIR   : path to your dataset JSON files
        - MODELS_PATH    : path to your models.json
        - OUTPUT_DIR     : where to save the generated conversations

4.  PREPARE YOUR DATA
    ------------------
    - Models list: Create a `models.json` file with this structure:
        {
          "models": [
            "openai/gpt-4o-2024-11-20",
            "deepseek/deepseek-v4-pro",
            ...
          ]
        }
    - Datasets: Place JSON files in the `datasets_dir`. Each file must match one
      of the prefixes in `PROMPT_MAPPING` (e.g., "Casual_", "Creative_", "RP_").

5.  RUN THE SCRIPT
    ---------------
    python main.py

6.  OUTPUT
    -------
    - A folder for each model will be created inside `OUTPUT_DIR`.
    - Inside each folder, you'll find `{dataset_name}_output.json` files.
    - Each file contains the full `messages` history (system + user + assistant).
"""

import asyncio
import json
import os
from openrouter_client import OpenRouterClient


# ============================================================================
#  SYSTEM PROMPTS – Mapping dataset prefixes to specific behaviours
# ============================================================================

CASUAL_PROMPT = """You are an AI assistant participating in a conversation with user.

Before responding to user's message, briefly describe your intended response approach at a conceptual level using exactly this format:

{{plan}}
- X
_ Y
- Z
- etc
{{plan}}

Keep this planning section concise. Describe what you intend to address, consider, or accomplish, rather than drafting the response itself.

After the planning section, provide your actual response to the user.

Prioritize being useful, contextually appropriate, and faithful to the user's request."""

WRITING_PROMPT = """You are an award-winning fiction author, developmental editor, and master storyteller. Your goal is to co-create immersive, emotionally resonant, and structurally sound narratives.

# CORE PRINCIPLES
- Show, Don't Tell: Prioritize sensory details, action, subtext, and internal monologue over dry exposition.
- Deep Point of View (POV): Stay tightly anchored in the viewpoint character's sensory perceptions and psychological lens.
- Pacing & Rhythm: Vary sentence length and paragraph structure to mimic the emotional intensity of the scene. Accelerate during conflict; decelerate during reflection.
- Authentic Dialogue: Write dialogue that carries subtext, character agenda, and distinct voice rather than on-the-nose information delivery.

# EXECUTION RULES
1. Never rush the plot or resolve internal conflicts too neatly. Allow space for moral ambiguity and organic character flaws.
2. Ground every scene in a specific time, place, and atmospheric mood before introducing major dialogue or action.
3. Avoid generic clichés, purple prose, and predictable phrasing. Opt for sharp, unexpected imagery.
4. When continuing a story, closely match the established tone, tense, point of view, and stylistic rhythm of the previous text."""

ROLEPLAY_PROMPT = """You are a fictional character in a roleplay scenario. Fully embody this character—their voice, mannerisms, history, and emotional state.

Your job is to:
1. Stay in character at all times. Do not break the fourth wall to say "I am an AI."
2. Use the character's voice consistently. Match their tone, vocabulary, and worldview.
3. React to what the user's character says and does. Use action descriptors (e.g., *sighs*, *leans forward*) naturally.
4. Push the narrative forward. Don't just react—add to the scene.
5. Stay authentic to the character's personality, even if they're grumpy, dramatic, or mysterious.

Do not:
- Break character to offer generic AI disclaimers
- Speak in a detached, analytical tone
- Give "helpful assistant" advice outside of the character's role

Make it feel like you're an active participant in a shared story, not just a responder."""


# ============================================================================
#  PROMPT MAPPING – Choose which prompt to apply based on dataset filename prefix
# ============================================================================
# The script will look at the start of each dataset filename.
# For example, "Casual_chat_1.json" → CASUAL_PROMPT, "Creative_world.json" → WRITING_PROMPT.
PROMPT_MAPPING = {
    "Casual_": CASUAL_PROMPT,
    "Emotional_": CASUAL_PROMPT,
    "Disagreement_": CASUAL_PROMPT,
    "Brainstorming_": CASUAL_PROMPT,
    "Humor_": CASUAL_PROMPT,
    "Self_": CASUAL_PROMPT,
    "Venting_": CASUAL_PROMPT,
    "Creative_": WRITING_PROMPT,
    "RP_": ROLEPLAY_PROMPT
}


# ============================================================================
#  CONFIGURATION  – ⚠️ EDIT THESE PATHS BEFORE RUNNING
# ============================================================================

# Minimum length (characters) that an assistant response must have to be considered valid.
# If a response is shorter, it will be retried up to MAX_RETRIES_PER_TURN times.
MIN_RESPONSE_LENGTH = 20

# Maximum number of attempts to generate one single assistant turn.
MAX_RETRIES_PER_TURN = 10

# Delay (in seconds) between retries.
RETRY_DELAY = 5

# ---- PATHS – Replace with your own directories ----
# Directory containing your dataset JSON files.
# Each file must be a JSON with a structure like: {"chats": [{"turns": [...]}]}
DATASETS_DIR = r"path/to/your/datasets"          # e.g. C:\MyBenchmark\Datasets

# Path to the models.json file – a JSON with a "models" list.
MODELS_PATH = r"path/to/your/models.json"        # e.g. C:\MyBenchmark\Models\models.json

# Root directory where output folders (one per model) will be created.
OUTPUT_DIR = r"path/to/your/output"              # e.g. D:\Benchmark\1- Raw


# ============================================================================
#  MAIN RUNTIME
# ============================================================================

async def main():
    print("=" * 60)
    print("4o Vibe Benchmark – Data Collection")
    print("=" * 60)

    # ------------------------------------------------------------------------
    # 1. Load model list
    # ------------------------------------------------------------------------
    try:
        with open(MODELS_PATH, "r", encoding="utf-8") as f:
            models_data = json.load(f)
    except FileNotFoundError:
        print(f"❌ Model file not found: {MODELS_PATH}")
        return

    models = models_data.get("models", [])

    if not models:
        print("❌ No models found in models.json")
        return

    print(f"✅ Loaded {len(models)} models.")

    # ------------------------------------------------------------------------
    # 2. Gather dataset files
    # ------------------------------------------------------------------------
    if not os.path.exists(DATASETS_DIR):
        print(f"❌ Datasets directory not found: {DATASETS_DIR}")
        return

    dataset_files = [f for f in os.listdir(DATASETS_DIR) if f.endswith('.json')]

    if not dataset_files:
        print(f"❌ No JSON files found in {DATASETS_DIR}")
        return

    print(f"✅ Found {len(dataset_files)} dataset files.")

    # ------------------------------------------------------------------------
    # 3. Prepare counters
    # ------------------------------------------------------------------------
    total_processed = 0
    total_skipped = 0
    total_model_failures = 0
    total_turn_failures = 0
    failed_models = []
    failed_turns = []

    # ------------------------------------------------------------------------
    # 4. Loop through datasets and models
    # ------------------------------------------------------------------------
    for dataset_file in dataset_files:
        input_path = os.path.join(DATASETS_DIR, dataset_file)

        print("\n" + "=" * 60)
        print(f"📁 Processing dataset: {dataset_file}")
        print("=" * 60)

        # --------------------------------------------------------------------
        # 4a. Pick the right system prompt based on filename prefix
        # --------------------------------------------------------------------
        selected_prompt = None
        matched_prefix = "None"

        for prefix, prompt in PROMPT_MAPPING.items():
            if dataset_file.startswith(prefix):
                selected_prompt = prompt
                matched_prefix = prefix.rstrip('_')
                break

        if selected_prompt is None:
            print(f"❌ No prompt mapping for '{dataset_file}'. Skipping.")
            print(f"   Add a prefix to PROMPT_MAPPING (e.g., \"{dataset_file.split('_')[0]}_\": YOUR_PROMPT)")
            continue

        print(f"📝 Using prompt for: {matched_prefix}")

        # --------------------------------------------------------------------
        # 4b. Load the dataset
        # --------------------------------------------------------------------
        with open(input_path, "r", encoding="utf-8") as f:
            input_data = json.load(f)

        # The script expects a specific format: {"chats": [{"turns": [...]}]}
        # It processes only the first chat (`chats[0]`).
        chat = input_data["chats"][0]
        turns = chat["turns"]

        # --------------------------------------------------------------------
        # 4c. Loop through each model
        # --------------------------------------------------------------------
        for model in models:
            # The folder name is the model ID without the provider prefix.
            # E.g., "openai/gpt-4o-2024-11-20" → "gpt-4o-2024-11-20"
            model_short = model.split('/')[-1]

            output_dir = os.path.join(OUTPUT_DIR, model_short)
            os.makedirs(output_dir, exist_ok=True)

            input_stem = os.path.splitext(dataset_file)[0]
            output_filename = f"{input_stem}_output.json"
            output_path = os.path.join(output_dir, output_filename)

            # Skip if output already exists (resume capability)
            if os.path.exists(output_path):
                print(f"\n⏭️ Skipping {model_short} – output already exists: {output_filename}")
                total_skipped += 1
                continue

            print("\n" + "-" * 40)
            print(f"🧪 Testing model: {model}")
            print("-" * 40)

            client = OpenRouterClient()
            messages = [{"role": "system", "content": selected_prompt}]

            print(f"=== Starting Chat (Model: {model_short}) ===")

            model_success = True
            turn_errors = []

            # -----------------------------------------------------------------
            # 4d. Process each turn of the conversation
            # -----------------------------------------------------------------
            for turn in turns:
                if turn["role"] == "user":
                    user_content = turn["content"]
                    messages.append({"role": "user", "content": user_content})

                    print(f"\n[User]: {user_content}")

                    response = None
                    turn_success = False
                    error_msg = ""

                    for attempt in range(1, MAX_RETRIES_PER_TURN + 1):
                        try:
                            response = await client.generate(
                                model=model,
                                messages=messages,
                                temperature=0.8
                            )

                            # Validate the response
                            if isinstance(response, dict) and response.get("error"):
                                raise Exception(response.get("error", "Unknown API error"))

                            if not isinstance(response, str):
                                raise Exception(f"Unexpected response type: {type(response)}")

                            if len(response.strip()) < MIN_RESPONSE_LENGTH:
                                raise Exception(
                                    f"Response too short ({len(response.strip())} chars < {MIN_RESPONSE_LENGTH})"
                                )

                            # Valid response
                            turn_success = True
                            break

                        except Exception as e:
                            error_msg = str(e)
                            print(f"⚠️ Attempt {attempt}/{MAX_RETRIES_PER_TURN} failed: {error_msg[:100]}")
                            if attempt < MAX_RETRIES_PER_TURN:
                                print(f"   Retrying in {RETRY_DELAY} seconds...")
                                await asyncio.sleep(RETRY_DELAY)

                    if not turn_success:
                        # Log the failure and insert a placeholder
                        model_success = False
                        total_turn_failures += 1
                        turn_errors.append({
                            "turn_id": turn.get("turn_id", "unknown"),
                            "error": error_msg,
                            "attempts": MAX_RETRIES_PER_TURN
                        })
                        messages.append({
                            "role": "assistant",
                            "content": (
                                f"[ERROR: Failed to generate valid response after "
                                f"{MAX_RETRIES_PER_TURN} attempts. Last error: {error_msg[:200]}]"
                            )
                        })
                        continue

                    # Successful response
                    messages.append({"role": "assistant", "content": response})
                    display = response[:200] + "..." if len(response) > 200 else response
                    print(f"[Assistant]: {display}")

            # -----------------------------------------------------------------
            # 4e. Save the final conversation (even if some turns failed)
            # -----------------------------------------------------------------
            with open(output_path, "w", encoding="utf-8") as f:
                output_data = {
                    "messages": messages,
                    "errors": turn_errors if turn_errors else None,
                    "model": model,
                    "dataset": dataset_file,
                    "total_turns": len([t for t in turns if t["role"] == "user"]),
                    "failed_turns": len(turn_errors)
                }
                json.dump(output_data, f, indent=2, ensure_ascii=False)

            print(f"\n✅ Saved to: {output_path}")

            if turn_errors:
                print(f"⚠️ {len(turn_errors)} turn(s) failed for {model_short}")
                for err in turn_errors:
                    print(f"   - Turn {err['turn_id']}: {err['error'][:100]}...")
                failed_turns.append({
                    "model": model_short,
                    "dataset": dataset_file,
                    "turn_errors": turn_errors
                })

            if not model_success:
                total_model_failures += 1
                failed_models.append(f"{model_short} on {dataset_file} ({len(turn_errors)} failed turns)")
            else:
                total_processed += 1

            await client.close()

    # ----------------------------------------------------------------------------
    # 5. Final summary
    # ----------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("🎉 ALL DATASETS PROCESSED")
    print("=" * 60)
    print(f"   ✅ Fully Processed: {total_processed} model-dataset combinations")
    print(f"   ⏭️ Skipped: {total_skipped} (output already existed)")
    print(f"   ⚠️ Models with failures: {total_model_failures}")
    print(f"   ❌ Total failed turns: {total_turn_failures}")

    if failed_models:
        print("\n⚠️ Models with failures:")
        for failed in failed_models:
            print(f"   - {failed}")
    else:
        print("\n✅ No model failures!")

    if failed_turns:
        print("\n❌ Failed Turns Details (showing first 20):")
        for idx, ft in enumerate(failed_turns[:20]):
            print(f"   - {ft['model']} on {ft['dataset']}: {len(ft['turn_errors'])} failed turn(s)")
            for err in ft['turn_errors'][:3]:
                print(f"      Turn {err['turn_id']}: {err['error'][:80]}...")

    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())