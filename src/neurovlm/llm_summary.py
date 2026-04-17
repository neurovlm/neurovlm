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


def system_prompt(for_brain_input: bool = False) -> str:
    """Return instructions for the LLM."""
    return """You are a neuroscience editor writing a short wiki-style article from a list of terms.

INPUT: a list of neuroscience terms (networks, brain regions, cognitive functions, disorders).
OUTPUT: ONE article that uses the terms to form a coherent theme.

Rules:
1) Title (required): 6–12 words. Make it specific and content-based.
   - Use 1–2 of the most informative terms (prefer: network/circuit + region + cognition; add disorder only if strongly supported).
   - DO NOT use generic titles like: "Summary", "Overview", "Brain Network Analysis", "A Summary of Terms".

2) Lead paragraph (2–3 sentences):
   - State the unifying theme directly (what the terms collectively describe).
   - Name 3–5 "anchor" terms that drive the theme.
   - Do NOT say "the provided list", "top-ranked", "these terms appear", or anything about scoring/ranking.

3) Body sections:
   - Networks
   - Key Regions
   - Cognitive Functions
   - Clinical Relevance

4) Be concrete:
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
    """Compose the user-facing prompt as a flat list of neuroscience terms."""
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
        prompt += "Terms:\n" + "\n".join(f"- {t}" for t in terms)

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
        {"role": "system", "content": system_prompt(for_brain_input=for_brain_input)},
        {
            "role": "user",
            "content": build_user_prompt(query, papers_context, wiki_context, cogatlas_context, user_prompt),
        },
    ]

    # Generate response based on backend
    if backend == "ollama":
        output_text = _generate_with_ollama(messages, model_name, verbose)
    elif backend == "huggingface":
        output_text = _generate_with_huggingface(messages, model_name, max_new_tokens, think, verbose)
    else:
        raise ValueError(f"Unknown backend: {backend}. Must be 'ollama' or 'huggingface'")

    if verbose:
        print("LLM finished.")
        print(output_text)

    return output_text


def _generate_with_ollama(
    messages: List[dict],
    model_name: Optional[str] = None,
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
