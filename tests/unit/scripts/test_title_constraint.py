"""Title constraint of scripts/models/_common_models.py on a toy tokenizer and a toy greedy decoder (CPU, no model).

A prompt passed as input_ids stays at the start of the ids that generate hands to the logits processors, while a prompt
passed as inputs_embeds does not. The constraint must give the same titles either way.
"""

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "scripts" / "models"))
import _common_models as T  # noqa: E402

VOCAB = ["\n", "\n\n", "[", "NETWORK", "REGION", "FUNCTION", "]", "The", " map", " shows",
         "Default", " Mode", " Network", "Visual", "Foo", " Bar", " is", " active"]
NAMES = {"network": ["Default Mode Network", "Visual Network"], "region": ["Visual Network"]}


class ToyTokenizer:
    """Greedy longest-match tokenizer over VOCAB."""

    def __len__(self):
        return len(VOCAB)

    def __call__(self, text, add_special_tokens=False):
        ids, i = [], 0
        while i < len(text):
            tok = max((v for v in VOCAB if text.startswith(v, i)), key=len)
            ids.append(VOCAB.index(tok))
            i += len(tok)
        return {"input_ids": ids}

    def decode(self, ids):
        return "".join(VOCAB[i] for i in ids)


class ToyLM:
    """Greedy decoder that prefers the next token by the last one: an invented title ("Foo Bar") after the tag, then
    "Visual Network"."""

    NEXT = {"]": ["Foo", "Visual", "Default"], "Foo": [" Bar"], " Bar": ["\n"], "Visual": [" Network", " Mode"],
            " Network": ["\n", "\n\n"], "\n": ["The"]}

    def __init__(self):
        self.generation_config = SimpleNamespace(num_beams=1)
        self.embed = torch.nn.Embedding(len(VOCAB), 4)

    def get_input_embeddings(self):
        return self.embed

    def scores(self, seq):
        out = torch.zeros(seq.shape[0], len(VOCAB))
        for r, row in enumerate(seq.tolist()):
            last = VOCAB[row[-1]] if row else "]"  # inputs_embeds: the prompt ends with the tag
            for rank, tok in enumerate(self.NEXT.get(last, [])):
                out[r, VOCAB.index(tok)] = 10.0 - rank
        return out

    def generate(self, input_ids=None, inputs_embeds=None, logits_processor=None, max_new_tokens=4):
        rows = (input_ids if input_ids is not None else inputs_embeds).shape[0]
        seq = input_ids if input_ids is not None else torch.zeros((rows, 0), dtype=torch.long)
        for _ in range(max_new_tokens):
            scores = self.scores(seq)
            for proc in logits_processor or []:
                scores = proc(seq, scores)
            seq = torch.cat([seq, scores.argmax(1, keepdim=True)], 1)
        return seq


def ids(text):
    return torch.tensor([ToyTokenizer()(text)["input_ids"]])


PROMPT = "The map shows[NETWORK]"


def allowed(proc, input_ids):
    scores = proc(input_ids, torch.zeros(input_ids.shape[0], len(VOCAB)))
    return {VOCAB[i] for i in torch.nonzero(torch.isfinite(scores[0])).flatten().tolist()}


@pytest.mark.parametrize("generated, expected", [
    ("", {"Default", "Visual"}),
    ("Default", {" Mode"}),
    ("Default Mode Network", {"\n", "\n\n"}),
    ("Visual Network\n", set(VOCAB)),
])
def test_title_mask_same_for_input_ids_and_inputs_embeds(generated, expected):
    tok = ToyTokenizer()
    trie = T.title_trie(tok, NAMES["network"])
    prompt, gen = ids(PROMPT), ids(generated) if generated else torch.zeros((1, 0), dtype=torch.long)
    from_ids = T.TitleConstraint(tok, [trie], offset=prompt.shape[1])  # prompt as input_ids: prompt + generated tokens
    from_embeds = T.TitleConstraint(tok, [trie], offset=0)  # prompt as inputs_embeds: generated tokens only
    assert allowed(from_ids, torch.cat([prompt, gen], 1)) == allowed(from_embeds, gen) == expected


@pytest.mark.parametrize("as_embeds", [False, True])
def test_title_constraint_installed_on_generate(as_embeds):
    tok, lm = ToyTokenizer(), ToyLM()
    T.install_title_constraint(lm, tok, NAMES)
    prompt = ids(PROMPT)
    if as_embeds:
        out = lm.generate(inputs_embeds=lm.get_input_embeddings()(prompt))
    else:
        out = lm.generate(input_ids=prompt)[:, prompt.shape[1]:]
    assert tok.decode(out[0].tolist()) == "Visual Network\nThe"  # canonical title, then free text


def test_title_free_without_constraint():
    tok, lm = ToyTokenizer(), ToyLM()
    T.install_title_constraint(lm, tok, None)
    prompt = ids(PROMPT)
    out = lm.generate(input_ids=prompt)[:, prompt.shape[1]:]
    assert tok.decode(out[0].tolist()) == "Foo Bar\nThe"
