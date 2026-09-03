#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
================================================================================
4o Vibe Benchmark – Bad File Cleaner
================================================================================

This script scans your benchmark output directory for problematic JSON files
and optionally deletes them. It checks for three types of issues:

1. API errors (files containing an "error" key)
2. Too-short or empty assistant responses (below a configurable threshold)
3. Malformed JSON (files that can't be parsed)

================================================================================
WHY USE THIS SCRIPT
================================================================================

During large-scale benchmarking runs, some API calls may fail, return errors,
or produce empty/truncated responses. These files can corrupt downstream
analysis (embedding, scoring) and should be removed before further processing.

This script helps you:
- Identify which files are problematic
- Review the issues before deletion
- Safely delete them (with confirmation)

================================================================================
HOW TO RUN
================================================================================

1.  INSTALL DEPENDENCIES
    --------------------
    No external dependencies – uses only Python standard library.

2.  CONFIGURE THE SCRIPT
    --------------------
    Edit the CONFIGURATION section below and set:
        - BENCHMARK_DIR         : root directory containing model folders
        - MIN_ASSISTANT_LENGTH  : minimum characters for a valid response

3.  RUN THE SCRIPT (DRY RUN FIRST)
    --------------------------------
    python clean_bad_files.py

    The script will first list all bad files and ask for confirmation.
    Type 'DELETE' to permanently remove them, or anything else to cancel.

================================================================================
OUTPUT
================================================================================

The script prints:
- A list of all bad files found (with reasons)
- A summary by model
- A final summary of what was deleted

================================================================================
SAFETY WARNING
================================================================================

⚠️  DELETED FILES CANNOT BE RECOVERED!
    They do NOT go to the Recycle Bin.
    Use the confirmation prompt carefully.
"""

import os
import json
from pathlib import Path

# ============================================================================
#  CONFIGURATION  – ⚠️ EDIT THESE BEFORE RUNNING
# ============================================================================

# ---- Paths ----
# Root directory containing all model folders (step 1 output)
# This is typically where your raw conversation JSON files are stored.
BENCHMARK_DIR = r"path/to/your/1-Raw"          # e.g. D:\Benchmark\1- Raw

# ---- Validation ----
# Minimum character count for a valid assistant response.
# If a response is shorter than this, the file is flagged as "bad."
MIN_ASSISTANT_LENGTH = 20


# ============================================================================
#  HELPERS
# ============================================================================

def _add_to_dict(dictionary, model_name, filename, reason):
    """Helper to add to the model summary dict."""
    if model_name not in dictionary:
        dictionary[model_name] = []
    dictionary[model_name].append({"file": filename, "reason": reason})


# ============================================================================
#  SCAN FUNCTION
# ============================================================================

def find_bad_files(root_dir, min_length=MIN_ASSISTANT_LENGTH):
    """
    Scan all JSON files under root_dir and identify problematic ones.

    Returns:
        bad_files: List of dicts with keys: model, file, path, reason
        bad_by_model: Dict grouping bad files by model name
        error_count: Number of API error files
        short_count: Number of files with too-short responses
        malformed_count: Number of malformed JSON files
    """
    bad_files = []
    bad_by_model = {}
    error_count = 0
    short_count = 0
    malformed_count = 0

    root_path = Path(root_dir)

    for model_folder in root_path.iterdir():
        if not model_folder.is_dir():
            continue

        model_name = model_folder.name

        for json_file in model_folder.glob("*.json"):
            try:
                with open(json_file, "r", encoding="utf-8") as f:
                    content = f.read()
                    data = json.loads(content)

                # Check 1: API error in raw content
                if '"error":' in content or "'error':" in content:
                    bad_files.append({
                        "model": model_name,
                        "file": json_file.name,
                        "path": str(json_file),
                        "reason": "API error found"
                    })
                    error_count += 1
                    _add_to_dict(bad_by_model, model_name, json_file.name, "API error")
                    continue

                # Check 2: Empty or too-short assistant responses
                if "messages" in data:
                    messages = data["messages"]
                    assistant_responses = [
                        msg.get("content", "")
                        for msg in messages
                        if msg.get("role") == "assistant"
                    ]

                    has_short_response = False
                    for resp in assistant_responses:
                        if not resp or len(resp.strip()) < min_length:
                            has_short_response = True
                            break

                    if has_short_response:
                        bad_files.append({
                            "model": model_name,
                            "file": json_file.name,
                            "path": str(json_file),
                            "reason": f"Assistant response too short (< {min_length} chars) or empty"
                        })
                        short_count += 1
                        _add_to_dict(bad_by_model, model_name, json_file.name, "Too short/empty")
                        continue

            except json.JSONDecodeError as e:
                # Malformed JSON
                bad_files.append({
                    "model": model_name,
                    "file": json_file.name,
                    "path": str(json_file),
                    "reason": f"Malformed JSON: {e}"
                })
                malformed_count += 1
                _add_to_dict(bad_by_model, model_name, json_file.name, "Malformed JSON")
                continue

            except Exception as e:
                print(f"⚠️ Could not read: {json_file} ({e})")
                continue

    return bad_files, bad_by_model, error_count, short_count, malformed_count


# ============================================================================
#  DELETE FUNCTION
# ============================================================================

def delete_files(file_list):
    """
    Delete all files in the list.

    Returns:
        deleted_count: Number of files successfully deleted
        failed_files: List of paths that could not be deleted
    """
    deleted_count = 0
    failed_files = []

    for entry in file_list:
        try:
            os.remove(entry["path"])
            print(f"   🗑️ Deleted: {entry['model']}/{entry['file']}")
            deleted_count += 1
        except Exception as e:
            print(f"   ❌ Failed to delete {entry['path']}: {e}")
            failed_files.append(entry["path"])

    return deleted_count, failed_files


# ============================================================================
#  MAIN
# ============================================================================

def main():
    """Main entry point: scans, lists, and optionally deletes bad files."""
    print("=" * 60)
    print("🔍 Scanning for bad files...")
    print(f"📂 Directory: {BENCHMARK_DIR}")
    print(f"📏 Minimum assistant response length: {MIN_ASSISTANT_LENGTH} chars")
    print("=" * 60)

    bad_files, bad_by_model, error_count, short_count, malformed_count = find_bad_files(
        BENCHMARK_DIR, MIN_ASSISTANT_LENGTH
    )

    if not bad_files:
        print("\n✅ No bad files found! Everything looks good.")
        return

    # =========================
    # LIST ALL BAD FILES
    # =========================
    print(f"\n❌ Found {len(bad_files)} bad file(s):")
    print("-" * 60)

    for entry in bad_files:
        print(f"  📁 {entry['model']}")
        print(f"     📄 {entry['file']}")
        print(f"     🏷️  Reason: {entry['reason']}")
        print(f"     📍 {entry['path']}")
        print()

    print("=" * 60)
    print("📊 Summary:")
    print("-" * 60)
    print(f"  ❌ API errors: {error_count}")
    print(f"  📝 Too short/empty responses: {short_count}")
    print(f"  🔧 Malformed JSON: {malformed_count}")
    print(f"  📦 Total bad files: {len(bad_files)}")
    print()

    print("📊 Summary by model:")
    print("-" * 60)
    for model, files in sorted(bad_by_model.items()):
        print(f"  ❌ {model}: {len(files)} file(s)")
        for f in files:
            print(f"     - {f['file']} ({f['reason']})")
        print()

    print("=" * 60)
    print(f"Total files to delete: {len(bad_files)}")
    print("=" * 60)

    # =========================
    # CONFIRMATION PROMPT
    # =========================
    print("\n⚠️  WARNING: This will permanently delete these files.")
    print("   (They will NOT go to the Recycle Bin.)")

    confirm = input("\nType 'DELETE' to confirm deletion, or anything else to cancel: ")

    if confirm != "DELETE":
        print("\n❌ Deletion cancelled. No files were deleted.")
        return

    # =========================
    # DELETE FILES
    # =========================
    print("\n🗑️ Deleting files...")
    print("-" * 60)

    deleted_count, failed_files = delete_files(bad_files)

    # =========================
    # FINAL SUMMARY
    # =========================
    print("-" * 60)
    print(f"\n✅ Deleted: {deleted_count} file(s)")
    if failed_files:
        print(f"❌ Failed to delete: {len(failed_files)} file(s)")
        for f in failed_files:
            print(f"   - {f}")
    else:
        print("✅ All bad files successfully deleted!")

    print("\n📌 Tip: Run your benchmark script again to regenerate these files.")


# ============================================================================
#  MAIN ENTRY POINT
# ============================================================================

if __name__ == "__main__":
    main()