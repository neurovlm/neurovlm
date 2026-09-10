# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.19.5
#   kernelspec:
#     display_name: .conda
#     language: python
#     name: python3
# ---

# %%
from pathlib import Path
import json
from tqdm.notebook import tqdm
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from neurovlm.data import load_dataset

# %% [markdown]
# # Transforming Text
#
# This notebook transforms title + abstract pairs to more general, high-level summaries or reviews, removing study specific details from text. Downstream models can be trained on this transformed dataset, specifically for brain-to-text generation. Given a brain, text can be learned to generate that is more of a wiki-style tone, rather than hallucinating studies and specific details. Here is an example transformation:
#
# #### Original
# > Impact of meditation training on the default mode network during a restful state.[SEP]Mindfulness meditation has been shown to promote emotional stability. Moreover, during the processing of aversive and self-referential stimuli, mindful awareness is associated with reduced medial prefrontal cortex (MPFC) activity, a central default mode network (DMN) component. However, it remains unclear whether mindfulness practice influences functional connectivity between DMN regions and, if so, whether such impact persists beyond a state of meditation. Consequently, this study examined the effect of extensive mindfulness training on functional connectivity within the DMN during a restful state. Resting-state data were collected from 13 experienced meditators (with over 1000 h of training) and 11 beginner meditators (with no prior experience, trained for 1 week before the study) using functional magnetic resonance imaging (fMRI). Pairwise correlations and partial correlations were computed between DMN seed regions' time courses and were compared between groups utilizing a Bayesian sampling scheme. Relative to beginners, experienced meditators had weaker functional connectivity between DMN regions involved in self-referential processing and emotional appraisal. In addition, experienced meditators had increased connectivity between certain DMN regions (e.g. dorso-medial PFC and right inferior parietal lobule), compared to beginner meditators. These findings suggest that meditation training leads to functional connectivity changes between core DMN regions possibly reflecting strengthened present-moment awareness.
#
# #### Transformed
# > Meditation has been found to influence various aspects of brain function, particularly in relation to the default mode network (DMN), a set of brain regions active during rest and self-referential thought. The DMN is composed of key regions, including the medial prefrontal cortex (MPFC), which is involved in emotional processing and self-awareness. Research has shown that meditation can alter DMN activity, with some studies indicating reduced activity in the MPFC during the processing of aversive and self-referential stimuli. Additionally, meditation has been associated with changes in functional connectivity within the DMN, which may reflect strengthened present-moment awareness. Specifically, studies have found that meditation can influence the connectivity between DMN regions involved in self-referential processing and emotional appraisal, potentially leading to improved emotional stability. Overall, the effects of meditation on the DMN suggest that this practice can have a profound impact on the brain's default mode of operation.

# %%
# Load data
df_pubs = load_dataset("pubmed_text")

# Load LLM
model_id = "meta-llama/Meta-Llama-3.1-8B-Instruct"

tokenizer = AutoTokenizer.from_pretrained(model_id, padding_side="left")

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

model = AutoModelForCausalLM.from_pretrained(
    model_id,
    dtype=torch.bfloat16,
    device_map="auto",
)

model.config.pad_token_id = tokenizer.pad_token_id

# Eval mode and disable gradients
model.eval()
for p in model.parameters():
    p.requires_grad_(False)
torch.set_grad_enabled(False)

# %%
system_prompt = """
You are writing a short, topic-focused neuroscience mini-review.

INPUT:
You will receive:
    title + " [SEP] " + abstract from an fMRI publication.

GOAL:
Write a single coherent paragraph (4–6 sentences) that:
- Describes the scientific topic implied by the paper (e.g., meditation and the default mode
  network, exercise and working memory), not the paper’s methods.
- Explains the key cognitive processes (e.g., working memory, self-referential thought,
  emotion regulation) and the main brain systems or networks involved.
- May mention typical effects or relationships in general terms (e.g., “has been associated
  with changes in connectivity” or “can influence executive control”), but should stay high-level.

ALLOWED:
- General statements about effects or relationships (e.g., “exercise has been associated
  with improved cognition”, “meditation can alter DMN connectivity”).
- Generic references to “studies” or “research” in the field.
- Qualitative descriptions of brain activity or connectivity (e.g., “increased activation
  in prefrontal regions”, “changes in functional connectivity”).

AVOID:
- Exact numerical details (sample sizes, p-values, coordinates, time points).
- Detailed task or protocol descriptions from the specific study.
- Directly quoting or closely paraphrasing long sentences from the abstract.
- Overly narrow focus on this single experiment; write as if summarizing the topic area.

STYLE:
- Neutral, scientific tone, like a short review or encyclopedia entry.
- One paragraph only. No bullet points or headings.
- Focus on giving a clear sense of “what this topic is about” in the context of the field.
"""

# %%
names = df_pubs["name"].tolist()
descs = df_pubs["description"].tolist()
dois = df_pubs["doi"].tolist()

pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

output_path = Path("neuro_summaries.jsonl")
BATCH_SIZE = 16
MAX_NEW_TOKENS = 256    # ~5–8 sentence paragraph

print(f"Writing summaries to: {output_path.resolve()}")

with output_path.open("w", encoding="utf-8") as f_out:
    # iterate in batches over the index range
    for start in tqdm(range(0, len(names), BATCH_SIZE)):
        end = min(start + BATCH_SIZE, len(names))

        batch_names = names[start:end]
        batch_descs = descs[start:end]
        batch_dois = dois[start:end]

        # Build a batch of chat-style message lists
        messages_batch = []
        for name, desc in zip(batch_names, batch_descs):
            user_prompt = f"{name}[SEP]{desc}"
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]
            messages_batch.append(messages)

        # Generate in inference mode (no gradients)
        with torch.inference_mode():
            outputs = pipe(
                messages_batch,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=True,
                temperature=0.3,
                top_p=0.9,
                top_k=50,
                return_full_text=True,
                batch_size=BATCH_SIZE,
                padding=True,
                truncation=True,
            )

        # Stream each result as a JSON line: {doi, summary}
        for doi, out in zip(batch_dois, outputs):
            # For chat-style pipelines: last message is assistant
            summary = out[0]["generated_text"][-1]["content"].strip()
            record = {
                "doi": doi,
                "summary": summary,
            }
            f_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

# %%
summaries = []

for name, desc in tqdm(zip(names, descs), total=len(names)):
    user_prompt = f"{name}[SEP]{desc}"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]
    outputs = pipeline(
        messages,
        max_new_tokens=512,
        do_sample=True,
        temperature=0.3,
        top_p=0.9,
        top_k=50,
        return_full_text=True,
    )
    out = outputs[0]["generated_text"][-1]["content"].strip()
    summaries.append(out)

df_pubs["summary"] = summaries
df_pubs.to_parquet("neuro_summaries.parquet")  # or csv

# %%
# save
import pandas as pd
df_summaries = pd.read_json("neuro_summaries.jsonl", lines=True)
df_summaries = df_summaries.merge(df_pubs[["doi", "name", "description"]], on="doi", how="left")
df_summaries.to_csv("neuro_summaries_sample.csv", index=False)
