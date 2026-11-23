import json
import os
import sys
from datetime import datetime
from time import time

import litellm
from src.pipelines.prompt_generator import PromptGenerator
from src.pipelines.RAG_a import OntologyRAG

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


INPUT_PATH = os.path.join(BASE_DIR, "data", "input")

API_BASE = "http://localhost:11434"
NUM_CTX = 24576
# wine.rdf alone weighs ~24k tokens: it does not fit in NUM_CTX with the rest of the prompt
NUM_CTX_BY_ONTOLOGY = {"wine.rdf": 32768}
NUM_PREDICT = 2048
REQUEST_TIMEOUT_SEC = 900
# Native context window of each model (ollama show): upper bound for the
# automatic expansion of num_ctx. A model whose native window is smaller than
# the prompt cannot hold the whole ontology.
NUM_CTX_BY_MODEL = {
    "qwen2.5:7b": 32768,
    "granite3.3:8b": 131072,
    "granite4.2:8b": 131072,
    "llama3.1:8b": 131072,
    "gpt-oss:20b": 131072,
    "qwen3.8:27b": 262144,
}
# Safety cap for the automatic expansion (KV cache: unified memory is not
# enough for 27B models with very large contexts).
MAX_NUM_CTX = 98304
CHARS_PER_TOKEN = 3.5   # typical estimate for RDF/Turtle text (~3-4 chars/token)
CHARS_PER_TOKEN_UPPER = 6  # upper bound chars/token -> lower-bound token estimate
# "Thinking" models consume num_predict while reasoning: granite4.2 writes it into
# the answer, gpt-oss returns empty text. Reasoning is disabled for all models
# (homogeneous comparison); gpt-oss cannot disable it and accepts only
# low/medium/high: with "low" it answers empty when the retrieved context is
# insufficient, with "medium" it answers.
THINK_DEFAULT = False
THINK_BY_MODEL = {"gpt-oss:20b": "medium"}
# Behavior verified experimentally on Ollama 0.34.2 (direct API, without LiteLLM):
# a prompt longer than num_ctx does NOT raise an error, but is silently truncated
# to num_ctx/2 + 2 tokens, keeping the TAIL of the prompt (the beginning disappears).
# Examples: num_ctx=24576 -> 12290 evaluated tokens; num_ctx=65536 -> 32770.
# A prompt that fits in num_ctx is instead evaluated in full. Consequences:
# 1) num_ctx must be chosen so the prompt fits (automatic expansion below);
# 2) the truncation flag cannot compare prompt_tokens with num_ctx (after the cut
#    it equals num_ctx/2, not num_ctx): it compares the reported tokens with a
#    lower-bound estimate of the tokens actually sent.
TRUNCATION_MARGIN = 8

# Ontologies on which RAG_a is applicable, with the configuration of their index.
# GO uses labels (GO_xxxxxxx IDs alone cannot be matched against the question) and
# indexes only hierarchy and exact synonyms: the full 1.4M triples would take too long.
RAG_CONFIG = {
    "Rientra.ttl": {
        "ontology_path": os.path.join(BASE_DIR, "data", "ontologies", "Rientra.ttl"),
        "collection_name": "rientra_rag",
    },
    "go.owl": {
        "ontology_path": os.path.join(BASE_DIR, "data", "ontologies", "go.owl"),
        "collection_name": "go_rag",
        "label_aware": True,
        "predicates": {"subClassOf", "hasExactSynonym"},
    },
}
RAG_ONTOLOGIES = set(RAG_CONFIG)

# Ontologies too large to be pasted into the prompt (go.owl: 130 MB, ~37M tokens)
NO_FULL_PASTE_ONTOLOGIES = {"go.owl"}

_rag_cache = {}

# ==========================================


def get_rag(ontology_name):
    """Create the RAG index once per ontology (loading the embedder is expensive)."""
    if ontology_name not in _rag_cache:
        cfg = RAG_CONFIG[ontology_name]
        predicates = cfg.get("predicates")
        _rag_cache[ontology_name] = OntologyRAG(
            ontology_path=str(cfg["ontology_path"]),
            persist_dir=os.path.join(BASE_DIR, "chroma_db"),
            collection_name=str(cfg["collection_name"]),
            label_aware=bool(cfg.get("label_aware", False)),
            predicates=predicates if isinstance(predicates, set) else None,
        )
    return _rag_cache[ontology_name]


def benchmark(model_name, prompt_mode, specific_ontology, output_dir, questions_path=None):
    print(f"Running benchmark for model: {model_name} with prompt mode: {prompt_mode}")

    think = THINK_BY_MODEL.get(model_name, THINK_DEFAULT)
    original_model_name = model_name
    model_name = f"ollama/{model_name}"

    # load the questions from the JSON file and initialize the prompt generator
    questions_path = questions_path or os.path.join(INPUT_PATH, "questions.json")
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if prompt_mode == "Q+Onto+Domain" and specific_ontology in NO_FULL_PASTE_ONTOLOGIES:
        raise ValueError(f"{specific_ontology} is too large to be pasted into the prompt")
    rag_a = None
    if prompt_mode == "RAG_a":
        if specific_ontology not in RAG_ONTOLOGIES:
            raise ValueError(f"RAG_a is supported only for {sorted(RAG_ONTOLOGIES)}")
        rag_a = get_rag(specific_ontology)
    generator = PromptGenerator(rag_a=rag_a)

    # inizializza il dizionario dei risultati
    results = {
        "model": model_name,
        "prompt_mode": prompt_mode,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "responses": []
    }

    for q in questions:
        question_text = q.get("Question")
        ontology_context = q.get("Ontology")
        if specific_ontology == "Rientra.ttl" and ontology_context == "Rientra.rdf":
            # Force the ontology to Rientra.ttl for the comparison
            ontology_context = "Rientra.ttl"
        if specific_ontology and (
            not ontology_context
            or specific_ontology.strip().lower() != ontology_context.strip().lower()
        ):
            continue
        sparql_query = q.get("SPARQL")

        # --- Context sizing ---
        # num_ctx must hold prompt + generation, otherwise Ollama silently
        # truncates the prompt to num_ctx/2 (see the comment at the top of the file).
        # Start from the ontology's base value and expand until the prompt fits,
        # within the model's native window and the safety cap.
        num_ctx = max(
            NUM_CTX_BY_ONTOLOGY.get(ontology_context, NUM_CTX), NUM_CTX
        )
        model_ctx_cap = min(
            NUM_CTX_BY_MODEL.get(original_model_name, MAX_NUM_CTX), MAX_NUM_CTX
        )
        system_prompt = user_prompt = None
        extra_info = {}
        error = None
        start_time = time()
        try:
            system_prompt, user_prompt = generator.generate_prompt(
                question=question_text,
                ontology_context=ontology_context,
                mode=prompt_mode,
                sparql=sparql_query
            )
            start_time = time()
            prompt_chars = len(system_prompt) + len(user_prompt)
            est_tokens = prompt_chars / CHARS_PER_TOKEN
            needed_ctx = int(est_tokens + NUM_PREDICT + TRUNCATION_MARGIN)
            if needed_ctx > num_ctx:
                expanded = min(needed_ctx, model_ctx_cap)
                if expanded > num_ctx:
                    print(
                        f"  [ctx] prompt ~{int(est_tokens)} token: "
                        f"num_ctx {num_ctx} -> {expanded} "
                        f"(cap modello {model_ctx_cap})"
                    )
                    num_ctx = expanded
            extra_info["est_prompt_tokens"] = int(est_tokens)
            if est_tokens + NUM_PREDICT > num_ctx:
                # Not even the maximum window holds the prompt: Ollama will
                # truncate it to num_ctx/2. The request is still executed
                # (experimental "fragment" condition), but flagged in advance.
                extra_info["context_expected_truncated"] = True
                print(
                    f"  [ctx] WARNING: prompt ~{int(est_tokens)} tokens "
                    f"beyond the maximum window {num_ctx} of "
                    f"{original_model_name}: it will be truncated to ~{num_ctx // 2} tokens"
                )
            response = litellm.completion(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.0,
                api_base=API_BASE,
                timeout=REQUEST_TIMEOUT_SEC,
                # Engine-specific parameters passed directly to Ollama
                extra_body={
                    "think": think,
                    "options": {
                        "num_ctx": num_ctx,      # Extended context window to hold the ontology
                        "num_predict": NUM_PREDICT,    # Generation budget for the inference
                        "temperature": 0.0,
                        "repeat_penalty": 1.0,
                    }
                }
            )
            time_taken = round(time() - start_time, 3)
            answer = (response.choices[0].message.content or "").strip()
            usage = response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            response_cost = response._hidden_params.get(
                "response_cost", 0.0) if hasattr(response, "_hidden_params") else 0.0
            min_prompt_tokens = (len(system_prompt) + len(user_prompt)) / CHARS_PER_TOKEN_UPPER
            # Truncation: the evaluated tokens are fewer than the tokens
            # actually sent (lower-bound estimate), or the estimated prompt
            # did not fit in num_ctx anyway (Ollama cuts it to num_ctx/2).
            extra_info["context_truncated"] = bool(
                prompt_tokens < min_prompt_tokens - TRUNCATION_MARGIN
                or min_prompt_tokens > num_ctx
                or extra_info.get("context_expected_truncated", False)
            )
            if not answer:
                extra_info["empty_answer"] = True
        except Exception as e:  # noqa: BLE001 - the error is saved in the result
            error = str(e)
            answer = f"Error: {error}"
            time_taken = round(time() - start_time, 3)
            prompt_tokens = completion_tokens = total_tokens = 0
            response_cost = 0.0
            print(f"Error: {e}")

        tokens_per_second = round(
            completion_tokens / time_taken, 2) if time_taken and time_taken > 0 else 0
        results["responses"].append({
            "QID": q.get("QID"),
            "question": question_text,
            "ontology_context": ontology_context,
            "system_prompt": system_prompt,
            "user_prompt": user_prompt,
            "SPARQL": sparql_query,
            "FC_Ans": q.get("FC_Ans"),
            "answer": answer,
            "error": error,
            "extra_info": extra_info,
            "metrics": {
                "runtime_seconds": time_taken,
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "tokens_per_second": tokens_per_second,
                "cost": response_cost
            }
        })

        print(results["responses"][-1].get("QID"),
              results["responses"][-1].get("metrics"))
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace(':', '_').replace('/', '_')
    name_file_output = f"{safe_model_name}_{prompt_mode}_{timestamp}.json"
    percorso_salvataggio = os.path.join(output_dir, name_file_output)
    with open(percorso_salvataggio, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)
