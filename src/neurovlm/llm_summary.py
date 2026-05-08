"""Summarize retrieved neuroscience content with a local LLM.

This module focuses on constructing prompts (system + user) and calling
either a Hugging Face model or Ollama to synthesise summaries. Retrieval of relevant
publications or NeuroWiki concepts is delegated to the specialised
`brain2text` and `text_to_brain` modules.

Supported backends:
- 'ollama': Fast, lightweight, requires Ollama installed locally (recommended)
- 'huggingface': More control, works offline, but slower and uses more memory
"""

from __future__ import annotations

import re
from typing import List, Optional, Tuple, Literal

import torch

# Global variables to cache the model and tokenizer (for HuggingFace backend)
_MODEL = None
_TOKENIZER = None
_MODEL_NAME: str | None = None


def load_huggingface_model(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    force_reload: bool = False,
):
    """Load the Hugging Face model and tokenizer.

    Uses global caching to avoid reloading the same model multiple times.
    If a different model name is requested, the cache is refreshed automatically.

    Parameters
    ----------
    model_name : str
        The Hugging Face model name to load.
        Recommended options:
        - "Qwen/Qwen2.5-1.5B-Instruct" (default, ~3GB, good balance)
        - "Qwen/Qwen2.5-0.5B-Instruct" (~1GB, fastest, lower quality)
        - "Qwen/Qwen2.5-3B-Instruct" (~6GB, best quality, slower)
    force_reload : bool
        If True, reload the model even if it's already cached.

    Returns
    -------
    Tuple[AutoModelForCausalLM, AutoTokenizer]
        The loaded model and tokenizer.

    Notes
    -----
    First-time model download may take several minutes depending on model size
    and internet speed. Subsequent loads will be faster as the model is cached.
    """
    from transformers import AutoModelForCausalLM, AutoTokenizer

    global _MODEL, _TOKENIZER, _MODEL_NAME

    model_changed = _MODEL_NAME is not None and _MODEL_NAME != model_name
    should_reload = _MODEL is None or _TOKENIZER is None or force_reload or model_changed

    if should_reload:
        # If switching models, clear references first to reduce memory pressure.
        if model_changed:
            print(f"Switching HuggingFace model: {_MODEL_NAME} -> {model_name}")
            _MODEL = None
            _TOKENIZER = None
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        print(f"\n{'='*60}")
        print(f"Loading HuggingFace model: {model_name}")
        print(f"{'='*60}")
        print("Note: First-time download may take several minutes...")
        print("Progress indicators will appear below:\n")

        # Load tokenizer first (faster)
        print("Step 1/2: Loading tokenizer...")
        _TOKENIZER = AutoTokenizer.from_pretrained(model_name)
        print("✓ Tokenizer loaded\n")

        # Load model with progress
        print("Step 2/2: Loading model (this may take a while)...")
        _MODEL = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype="auto",
            device_map="auto",
            low_cpu_mem_usage=True,  # Optimize memory usage
        )

        # Determine device
        device = "unknown"
        if hasattr(_MODEL, 'device'):
            device = str(_MODEL.device)
        elif hasattr(_MODEL, 'hf_device_map'):
            device = str(_MODEL.hf_device_map)

        print(f"✓ Model loaded successfully!")
        print(f"  Device: {device}")
        print(f"  Model size: {model_name.split('/')[-1]}")
        print(f"{'='*60}\n")
        _MODEL_NAME = model_name

    return _MODEL, _TOKENIZER


def _wants_short_output(user_prompt: str = "") -> bool:
    """Heuristic for notebook/eval prompts that request a title or one-liner."""
    text = str(user_prompt or "").lower()
    markers = (
        "title only",
        "output only",
        "single concise sentence",
        "one concise sentence",
        "reply with a single",
        "generate only a paper title",
    )
    return any(marker in text for marker in markers)


def _postprocess_short_output(output_text: str, user_prompt: str = "") -> str:
    """Trim accidental paragraphs from title/one-sentence generations."""
    text = str(output_text or "").strip()
    if not _wants_short_output(user_prompt) or not text:
        return text

    text = re.sub(r"^\s*(?:#+\s*)?", "", text)
    first_line = next((line.strip() for line in text.splitlines() if line.strip()), text)
    first_line = first_line.strip(" \"'")

    prompt = str(user_prompt or "").lower()
    if "title" in prompt:
        return first_line

    match = re.match(r"^(.+?[.!?])(?:\s|$)", first_line)
    return (match.group(1) if match else first_line).strip()


def system_prompt(for_brain_input: bool = False, short_output: bool = False) -> str:
    """Return instructions for the LLM."""
    if short_output:
        return """You are a neuroscience editor converting ranked neuroimaging retrieval evidence into a very short answer.

INPUT: ranked neuroscience evidence from a brain-to-text retrieval model. Each row may include
a rank number and cosine similarity score. Rank 1 is the strongest evidence.
OUTPUT: Follow the user's requested format exactly. Write only the requested title or one sentence.
Do not add explanations, definitions, headings, bullets, abstracts, or supporting paragraphs.

Ranking rules:
- Use rank 1 as the main clue.
- Use lower-ranked rows only to disambiguate the wording.
- Do not mention ranks, scores, retrieval, evidence, or this prompt."""

    return """You are a neuroscience editor writing a concise explanatory entry from ranked neuroimaging retrieval evidence.

INPUT: ranked neuroscience evidence from a brain-to-text retrieval model. Each row may include
a rank number and cosine similarity score. Rank 1 is the strongest evidence.
OUTPUT: ONE in-depth paragraph grounded in the ranked evidence. Do not use a title,
section headings, bullets, or numbered lists.

Ranking rules:
- Treat rank 1 as the main topic to define and explain. If ranks 1-2 are near-duplicates or a network plus its canonical function, define them together.
- Use ranks 2-5 to explain how the main topic relates to supporting functions, regions, or component concepts.
- Do not flatten all terms into an equal bag of words.
- If rank 1 is a named brain network, define that network as the subject. Do not drift into a generic article about the broad cognitive domain.
- If lower-ranked terms are regions, explain them as likely nodes or supporting anatomy of the top-ranked network.
- If lower-ranked terms are cognitive functions, explain them as functions associated with the top-ranked network.
- Do not use uncertainty framing like "hypothesis", "suggests", "may indicate", or "appears to". Write as an explanatory definition.

Rules:
1) Write one coherent paragraph:
   - Define the rank-1 topic directly.
   - Explain how 2–4 supporting terms relate to that topic as functions, regions, network nodes, or component concepts.
   - Do NOT say "the provided list", "top-ranked", "these terms appear", or anything about scoring/ranking.

2) Be concrete:
   - Prefer specific mechanisms, pathways, and canonical associations over vague statements.
   - If a term is too vague/ambiguous/unrelated, ignore it in the main text.

No references. Do not mention this prompt."""


def build_user_prompt(
    query: str | torch.Tensor,
    papers_context: str | None = None,
    wiki_context: str | None = None,
    cogatlas_context: str | None = None,
    user_prompt: str = "",
) -> str:
    """Compose the user-facing prompt from ranked neuroscience context."""
    terms: list[str] = []

    for ctx in (wiki_context, cogatlas_context, papers_context):
        if not ctx:
            continue
        for line in ctx.splitlines():
            line = line.strip()
            if line.startswith("- "):
                term = line[2:].split(":")[0].strip()
                if term:
                    terms.append(term)

    prompt = ""
    if user_prompt:
        prompt += f"Context: {user_prompt}\n\n"

    if terms:
        prompt += (
            "Ranked evidence rows, in descending model similarity. "
            "Use row 1 as the main interpretation and later rows as support:\n"
            + "\n".join(f"- {t}" for t in terms)
        )

    return prompt


def generate_response(
    query: str | torch.Tensor,
    papers_context: str | None = None,
    wiki_context: str | None = None,
    cogatlas_context: str | None = None,
    user_prompt: str = "",
    backend: Literal["ollama", "huggingface"] = "ollama",
    model_name: str | None = None,
    max_new_tokens: Optional[int] = 512,
    verbose: Optional[bool] = False,
    think: Optional[bool] = False
) -> str:
    """Summarize publications relevant to a query or brain-derived input.

    This function now accepts pre-retrieved contexts instead of doing retrieval itself.

    Parameters
    ----------
    query : str | torch.Tensor
        Natural language query or pre-computed brain embedding (used for determining brain vs text mode).
    papers_context : str | None
        Pre-formatted papers context. If None, papers won't be included.
    wiki_context : str | None
        Pre-formatted NeuroWiki context. If None, NeuroWiki won't be included.
    cogatlas_context : str | None
        Pre-formatted CogAtlas context. If None, CogAtlas won't be included.
    user_prompt : str
        Optional user query/prompt to provide additional context.
    backend : {"ollama", "huggingface"}, optional
        Which LLM backend to use. Default: "ollama" (faster, requires Ollama installed).
        "huggingface" loads models directly from HuggingFace (slower, works offline).
    model_name : str, optional
        Model name to use. If None, uses defaults:
        - Ollama: "qwen2.5:3b-instruct"
        - HuggingFace: "Qwen/Qwen2.5-1.5B-Instruct"

        For HuggingFace, options include:
        - "Qwen/Qwen2.5-1.5B-Instruct" (~3GB, good balance)
        - "Qwen/Qwen2.5-0.5B-Instruct" (~1GB, fastest)
        - "Qwen/Qwen2.5-3B-Instruct" (~6GB, best quality)

        For Ollama, any installed model works (e.g., "llama3.2:3b", "qwen2.5:3b-instruct")
    max_new_tokens : int, optional
        Maximum number of tokens to generate (HuggingFace only).
    verbose : bool, optional
        If True, prints progress messages.

    Returns
    -------
    str
        The LLM-generated summary text.

    Examples
    --------
    >>> # Use Ollama (default, fast)
    >>> output = generate_response(query="default mode network", backend="ollama")

    >>> # Use HuggingFace with small model
    >>> output = generate_response(
    ...     query="default mode network",
    ...     backend="huggingface",
    ...     model_name="Qwen/Qwen2.5-0.5B-Instruct"
    ... )
    """
    for_brain_input = not isinstance(query, str)

    if verbose:
        if for_brain_input:
            print(f"Generating LLM response from brain input using {backend}...")
            if wiki_context:
                print("- Using NeuroWiki concepts")
            if cogatlas_context:
                print("- Using CogAtlas terms")
        else:
            print(f"Generating LLM response for query: '{query}' using {backend}...")
            if papers_context:
                print("- Using papers")
            if cogatlas_context:
                print("- Using CogAtlas terms")
        print("LLM writing summary...")

    # Build the messages
    messages = [
        {"role": "system", "content": system_prompt(for_brain_input=for_brain_input, short_output=_wants_short_output(user_prompt))},
        {
            "role": "user",
            "content": build_user_prompt(query, papers_context, wiki_context, cogatlas_context, user_prompt),
        },
    ]

    # Generate response based on backend
    if backend == "ollama":
        output_text = _generate_with_ollama(messages, model_name, max_new_tokens, verbose)
    elif backend == "huggingface":
        output_text = _generate_with_huggingface(messages, model_name, max_new_tokens, think, verbose)
    else:
        raise ValueError(f"Unknown backend: {backend}. Must be 'ollama' or 'huggingface'")

    if verbose:
        print("LLM finished.")
        print(output_text)

    return _postprocess_short_output(output_text, user_prompt)


def _generate_with_ollama(
    messages: List[dict],
    model_name: Optional[str] = None,
    max_new_tokens: Optional[int] = 512,
    verbose: bool = False,
) -> str:
    """Generate response using Ollama backend.

    Parameters
    ----------
    messages : list of dict
        Chat messages in OpenAI format.
    model_name : str, optional
        Ollama model name. Default: "qwen2.5:3b-instruct"
    verbose : bool
        Print progress messages.

    Returns
    -------
    str
        Generated text.
    """
    try:
        import ollama
    except ImportError:
        raise ImportError(
            "Ollama package not found. Install with: pip install ollama\n"
            "Or use backend='huggingface' instead."
        )

    if model_name is None:
        model_name = "qwen2.5:3b-instruct"

    if verbose:
        print(f"Using Ollama model: {model_name}")

    try:
        response = ollama.chat(
            model=model_name,
            messages=messages,
            options={"num_predict": max_new_tokens} if max_new_tokens is not None else None,
        )
        return response["message"]["content"]
    except Exception as e:
        if "model" in str(e).lower() and "not found" in str(e).lower():
            raise RuntimeError(
                f"Ollama model '{model_name}' not found. "
                f"Pull it first with: ollama pull {model_name}\n"
                f"Or use a different model_name parameter."
            )
        raise


def _generate_with_huggingface(
    messages: List[dict],
    model_name: Optional[str] = None,
    max_new_tokens: Optional[int] = 512,
    think: Optional[bool] = False,
    verbose: Optional[bool] = False,
) -> str:
    """Generate response using HuggingFace backend.

    Parameters
    ----------
    messages : list of dict
        Chat messages in OpenAI format.
    model_name : str, optional
        HuggingFace model name. Default: "Qwen/Qwen2.5-1.5B-Instruct"
    max_new_tokens : int
        Maximum tokens to generate.
    verbose : bool
        Print progress messages.

    Returns
    -------
    str
        Generated text.
    """
    if model_name is None:
        model_name = "Qwen/Qwen2.5-1.5B-Instruct"

    # Load model and tokenizer
    model, tokenizer = load_huggingface_model(model_name)

    # Apply chat template
    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=think
    )

    # Tokenize and move to device
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    # Generate response
    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens
    )

    # Extract only the newly generated tokens
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    # Decode the response
    output_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    return output_text
