# GPT-4o Similarity Benchmark

**A lightweight, reproducible pipeline for measuring how closely open-source LLMs match GPT-4o's conversational style and content — not just what they say, but how they say it.**

<img width="860" height="361" alt="draw chat_dc2sjolmi9lwa4bs1cnx9h6pth9hvh_p1" src="https://github.com/user-attachments/assets/7f75ca42-3fc9-4d0f-ba79-5d8be86d14c6" />

---

## What is this?

This benchmark measures how similar other LLMs are to GPT-4o in terms of **response style (vibe)** and **content coverage**, using a weighted scoring system **(85% vibe / 15% content)** . It uses a two-pronged approach: an LLM judge (GLM-5.2-thinking) evaluates behavioral similarity (tone, empathy, naturalness, flow), while embedding cosine similarity measures semantic content overlap. The final score combines both into a single 0–100 leaderboard.

---

## How it works

**Methodology:**

1. **Dataset Collection** – 50+ multi-turn conversations across 9 categories (Casual, Emotional, Brainstorming, Roleplay, Creative Writing, etc.) are sent to each model.
2. **Embedding Similarity (15%)** – Assistant responses are embedded using Qwen3-Embedding-8B, and cosine similarity is computed against GPT-4o's responses.
3. **LLM Judge (85%)** – GLM-5.2-thinking rates each conversation's behavioral similarity to GPT-4o across four criteria: Tone & Personality, Responsiveness & Adaptation, Naturalness & Flow, and Helpfulness & Strategy.
4. **Weighted Scoring** – LLM scores (raw 0–100) and embedding scores (min-max normalized to 0–100) are combined with configurable weights (default 85/15).
5. **Leaderboard** – Final scores are averaged across all datasets per model, producing a clean ranking.

**Datasets Used:** ~50 JSON files, each containing a 4–12 turn conversation with a specific user persona and scenario.

---

## Setup

```bash
# Clone the repository
git clone https://github.com/LactationStation67/4o-Similarity-benchmark.git
cd 4o-vibe-benchmark

# Install dependencies
pip install -r requirements.txt
```

**Configuration:**

1. Open `openrouter_client.py` and replace `"sk-or-v1-..."` with your OpenRouter API key.
2. Edit the `CONFIGURATION` section in each script to point to your input/output directories.
3. (Optional) Adjust `LLM_WEIGHT` in `final_score.py` to change the vibe/content balance.

---

## Running (in order)

| Step | Script | Description |
|------|--------|-------------|
| **1** | `openrouter_client.py` | Insert your API key and do custom setup if needed. This is the core client used by all other scripts. |
| **2** | `main.py` | Calls each model on every dataset and collects the raw conversation outputs. |
| **3** | `find_bad_files.py` | Run this on the directory where the datasets were collected. Checks for API errors, empty responses, and too-short responses — and deletes them. Run `main.py` again to regenerate them. |
| **4** | `embed.py` | Calls Qwen3-Embedding-8B via OpenRouter and embeds all assistant responses from the cleaned datasets. |
| **5** | `4o_cos_sim.py` | Computes cosine similarity between each model's embeddings and the GPT-4o embeddings. |
| **6** | `LLM_analyse.py` | Calls GLM-5.2-thinking across all datasets and rates each conversation's behavioral similarity to GPT-4o (0–100). |
| **7** | `final_score.py` | If all steps completed correctly, run this to generate the final weighted leaderboard (JSON + TXT). |

---

## Results

**🏆 Leaderboard (vs gpt-4o-2024-08-06)**

```
Rank  Model                                       Final     Embed     LLM    
--------------------------------------------------------------------------------
1     deepseek-v3.2-thinking                       73.7%     56.5%     81.1%
2     llama-4-scout                               73.5%     69.1%     75.4%
3     Mistral-Small-3.2-24B-Instruct-2506         72.6%     76.7%     70.9%
4     deepseek-v3.2                               69.3%     49.0%     78.0%
5     nemotron-3-super-120b-a12b                  49.6%     60.9%     44.8%
...
```

*(Full results are available in `leaderboard_final.json` and `leaderboard_final.txt` inside the output directory.)*

---

## Limitations / Notes

- **No NSFW** – This benchmark deliberately avoids NSFW content. Some models have hard refusals on sensitive topics, which would break the dataset consistency and complicate scoring.
- **Dataset bias** – The benchmark reflects the specific datasets used. Adding more categories or altering prompts may shift results.
- **LLM judge variability** – Like all LLM-as-judge evaluations, scores can vary slightly across runs. Using temperature=0.3 and averaging across many datasets mitigates this.
- **Cost** – Running all steps with OpenRouter costs around **$15–20** total (embedding ~$0.13, LLM judge ~$7, generation ~$8–12).
- **Time** – Running all 60 models on all datasets takes approximately **4–7 days** depending on provider and rate limits.

---

### 📝 Worth noting

- **GLM 5.3 and its thinking variant** are around the bottom. You can hear the distinct *Vallonesim* in their responses pretty clearly. Well-deserved placement.

- **The 24B Mistral model** (Mistral-Small-3.2-24B-Instruct-2506) is an amazing candidate for a 4o fine-tune. It's small enough to run on 16GB VRAM, performs well overall, but needs adjustment—it's nowhere near 4o out of the box.

- **DeepSeek** scored multiple high placements. They have some genuinely impressive models, and they're my personal favorites from this benchmark.

- **The scores are partially just relevance to your prompt.** Any non-hostile LLM will have a baseline of around **30% similarity** to 4o simply by answering coherently. So the top score of **64%** isn't just about capturing 4o's "spark"—it reflects content relevance + vibe combined. The real gap is in the vibe layer.

---

## Credits

Built as a personal project to answer: *"Which open-source model actually feels like 4o?"* — using a combination of embedding similarity, LLM-as-judge, and a lot of patience.

Feel free to fork, extend, or contribute!
