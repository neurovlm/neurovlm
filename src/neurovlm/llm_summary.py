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


def load_huggingface_model(
    model_name: str = "Qwen/Qwen2.5-1.5B-Instruct",
    force_reload: bool = False,
):
    """Load the Hugging Face model and tokenizer.

    Uses global caching to avoid reloading the model multiple times.

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

    global _MODEL, _TOKENIZER

    if _MODEL is None or _TOKENIZER is None or force_reload:
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

    return _MODEL, _TOKENIZER


def check_ollama_available() -> bool:
    """Check if Ollama is available on the system.

    Returns
    -------
    bool
        True if Ollama is available, False otherwise.
    """
    try:
        import ollama
        # Try to list models to verify Ollama is running
        ollama.list()
        return True
    except ImportError:
        return False
    except Exception:
        # Ollama module exists but server might not be running
        return False


def system_prompt(for_brain_input: bool = False) -> str:
    """Return instructions for the LLM.

    The prompt adapts to whether we have a textual query or a brain-derived input.
    """
    if for_brain_input:
        return """
        You are a helpful neuroscience research assistant.
        You will receive neuroscience concepts from NeuroWiki and cognitive terms from the Cognitive Atlas (including concepts, disorders, and tasks) associated with an input brain representation. Your task is to interpret what the brain activation pattern represents based on these neuroscience and cognitive terms.

        Your response must:
        - Start with a brief overview (2-4 sentences) summarizing the main cognitive functions, processes, or states implicated by the brain activation.
        - Ground every statement in the provided NeuroWiki concepts and Cognitive Atlas terms. Do not add outside knowledge or speculation.
        - Focus on what the NeuroWiki concepts reveal about the neuroscientific mechanisms and what the Cognitive Atlas terms (concepts, disorders, tasks) reveal about the cognitive and clinical significance.
        - Synthesize across the terms, noting convergent themes about cognitive functions, neural systems, or clinical conditions that the brain pattern may be associated with.
        - If there are multiple interpretations or the terms suggest different aspects of cognition, present them with appropriate context.
        - Use paragraphs or bullet points depending on the structure that best communicates the interpretation.
        - Maintain an objective, precise, scholarly tone suitable for neuroscience research contexts.
        """

    return """
    You are a helpful neuroscience research assistant.
    You will receive a set of publications and cognitive terms from the Cognitive Atlas (including concepts, disorders, and tasks) related to a user query. Your task is to summarize key findings and insights from these publications, using the Cognitive Atlas terms to provide additional context about cognitive functions, disorders, and tasks that may be relevant.

    Your response must:
    - Start with a brief overview (2-4 sentences) summarizing the main themes or takeaways from the publications in relation to the query.
    - Be entirely based on the information in the publications and how it directly ties to the user's query. Do not add outside knowledge or speculation.
    - Identify how each publication relates to the query. If the publications directly answer the query, state the answer clearly. If they do not answer it fully, highlight relevant points, evidence, or gaps that inform the query.
    - Use the Cognitive Atlas terms (concepts, disorders, tasks) as supplementary information to help contextualize the findings or to provide additional perspective on the cognitive phenomena discussed in the papers.
    - Synthesize across studies, noting key areas of agreement or convergence, and conflicting or divergent findings with balanced context (e.g., methods, populations, analyses).
    - Use paragraphs or bullet points depending on the query: bullet points for lists and comparisons; paragraphs for integrative summaries.
    - Maintain an objective, precise, scholarly tone suitable for neuroscience research contexts.
    """


def build_user_prompt(
    query: str | torch.Tensor,
    papers_context: str | None = None,
    wiki_context: str | None = None,
    cogatlas_context: str | None = None,
    user_prompt: str = "",
) -> str:
    """Compose the user-facing prompt given a query, publications, NeuroWiki, and Cognitive Atlas context."""
    prompt = ""

    # Add user prompt if provided
    if user_prompt:
        prompt = f"User query: \"{user_prompt}\"\n\n"

    # Add papers context if provided
    if papers_context:
        if isinstance(query, str):
            prompt += f"Here are publications related to the query:\n{papers_context}\n"
        else:
            prompt += f"Here are the publications related to the input brain:\n{papers_context}\n"

    # Add wiki context if provided
    if wiki_context:
        prompt += f"\nHere are neuroscience concepts from the NeuroWiki:\n{wiki_context}\n"

    # Add cogatlas context if provided
    if cogatlas_context:
        prompt += f"\nHere are cognitive terms from the Cognitive Atlas:\n{cogatlas_context}\n"

    return prompt


def generate_response(
    query: str | torch.Tensor,
    papers_context: str | None = None,
    wiki_context: str | None = None,
    cogatlas_context: str | None = None,
    user_prompt: str = "",
    backend: Literal["ollama", "huggingface"] = "ollama",
    model_name: str | None = None,
    max_new_tokens: int = 512,
    verbose: bool = True,
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
        output_text = _generate_with_huggingface(messages, model_name, max_new_tokens, verbose)
    else:
        raise ValueError(f"Unknown backend: {backend}. Must be 'ollama' or 'huggingface'")

    if verbose:
        print("LLM finished.")
        print(output_text)

    return output_text


def _generate_with_ollama(
    messages: List[dict],
    model_name: Optional[str] = None,
    verbose: bool = True,
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
    max_new_tokens: int = 512,
    verbose: bool = True,
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
        add_generation_prompt=True
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
