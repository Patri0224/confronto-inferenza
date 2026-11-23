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
# wine.rdf da solo pesa ~24k token: non entra in NUM_CTX con il resto del prompt
NUM_CTX_BY_ONTOLOGY = {"wine.rdf": 32768}
NUM_PREDICT = 2048
REQUEST_TIMEOUT_SEC = 900
# Finestra di contesto nativa di ogni modello (ollama show): limite superiore
# all'espansione automatica di num_ctx. Un modello la cui finestra nativa è più
# piccola del prompt non può contenere l'ontologia per intero.
NUM_CTX_BY_MODEL = {
    "qwen2.5:7b": 32768,
    "granite3.3:8b": 131072,
    "granite4.2:8b": 131072,
    "llama3.1:8b": 131072,
    "gpt-oss:20b": 131072,
    "qwen3.8:27b": 262144,
}
# Tetto di sicurezza per l'espansione automatica (KV cache: la memoria unificata
# non basta per 27B con contesti molto grandi).
MAX_NUM_CTX = 98304
CHARS_PER_TOKEN = 3.5   # stima tipica per testo RDF/Turtle (~3-4 caratteri/token)
CHARS_PER_TOKEN_UPPER = 6  # limite superiore caratteri/token -> stima per difetto dei token
# I modelli "thinking" consumano num_predict nel ragionamento: granite4.2 lo scrive nella
# risposta, gpt-oss restituisce testo vuoto. Il ragionamento è disattivato per tutti i modelli
# (confronto omogeneo); gpt-oss non può disattivarlo e accetta solo low/medium/high:
# con "low" risponde vuoto quando il contesto recuperato non basta, con "medium" risponde.
THINK_DEFAULT = False
THINK_BY_MODEL = {"gpt-oss:20b": "medium"}
# Comportamento verificato sperimentalmente su Ollama 0.34.2 (API diretta, senza LiteLLM):
# un prompt più lungo di num_ctx NON dà errore, ma viene troncato in silenzio a
# num_ctx/2 + 2 token tenendo la CODA del prompt (l'inizio sparisce).
# Esempi: num_ctx=24576 -> 12290 token valutati; num_ctx=65536 -> 32770.
# Un prompt che entra in num_ctx viene invece valutato per intero. Conseguenze:
# 1) num_ctx va scelto in modo che il prompt ci stia (espansione automatica qui sotto);
# 2) il flag di troncamento non può confrontare prompt_tokens con num_ctx (dopo il taglio
#    vale num_ctx/2, non num_ctx): confronta i token riportati con una stima per difetto
#    dei token effettivamente inviati.
TRUNCATION_MARGIN = 8

# Ontologie su cui RAG_a è applicabile, con la configurazione del relativo indice.
# GO usa le etichette (gli ID GO_xxxxxxx da soli non si confrontano con la domanda) e indicizza
# solo gerarchia e sinonimi esatti: 1,4M triple complete richiederebbero troppo tempo.
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

# Ontologie troppo grandi per essere incollate nel prompt (go.owl: 130 MB, ~37M token)
NO_FULL_PASTE_ONTOLOGIES = {"go.owl"}

_rag_cache = {}

# ==========================================


def get_rag(ontology_name):
    """Crea l'indice RAG una sola volta per ontologia (caricamento embedder costoso)."""
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

    # carica le domande dal file JSON e inizializza il generatore di prompt
    questions_path = questions_path or os.path.join(INPUT_PATH, "questions.json")
    with open(questions_path, "r", encoding="utf-8") as f:
        questions = json.load(f)

    if prompt_mode == "Q+Onto+Domain" and specific_ontology in NO_FULL_PASTE_ONTOLOGIES:
        raise ValueError(f"{specific_ontology} è troppo grande per essere incollata nel prompt")
    rag_a = None
    if prompt_mode == "RAG_a":
        if specific_ontology not in RAG_ONTOLOGIES:
            raise ValueError(f"RAG_a è supportato solo per {sorted(RAG_ONTOLOGIES)}")
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
            # Forza l'ontologia a Rientra.ttl per il confronto
            ontology_context = "Rientra.ttl"
        if specific_ontology and (
            not ontology_context
            or specific_ontology.strip().lower() != ontology_context.strip().lower()
        ):
            continue
        sparql_query = q.get("SPARQL")

        # --- Dimensionamento del contesto ---
        # num_ctx deve contenere prompt + generazione, altrimenti Ollama tronca
        # il prompt in silenzio a num_ctx/2 (vedi commento in testa al file).
        # Si parte dal valore base dell'ontologia e si espande fino a far entrare
        # il prompt, entro la finestra nativa del modello e il tetto di sicurezza.
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
                # Nemmeno la finestra massima contiene il prompt: Ollama lo
                # troncherà a num_ctx/2. La richiesta viene comunque eseguita
                # (condizione sperimentale "frammento"), ma marcata a priori.
                extra_info["context_expected_truncated"] = True
                print(
                    f"  [ctx] ATTENZIONE: prompt ~{int(est_tokens)} token "
                    f"oltre la finestra massima {num_ctx} di "
                    f"{original_model_name}: sara' troncato a ~{num_ctx // 2} token"
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
                # Parametri specifici passati direttamente al motore Ollama
                extra_body={
                    "think": think,
                    "options": {
                        "num_ctx": num_ctx,      # Finestra di contesto estesa per contenere l'ontologia
                        "num_predict": NUM_PREDICT,    # Budget per la generazione dell'inferenza
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
            # Troncamento: i token valutati sono meno dei token (stimati per
            # difetto) effettivamente inviati, oppure il prompt stimato non
            # entrava comunque in num_ctx (Ollama lo taglia a num_ctx/2).
            extra_info["context_truncated"] = bool(
                prompt_tokens < min_prompt_tokens - TRUNCATION_MARGIN
                or min_prompt_tokens > num_ctx
                or extra_info.get("context_expected_truncated", False)
            )
            if not answer:
                extra_info["empty_answer"] = True
        except Exception as e:  # noqa: BLE001 - l'errore viene salvato nel risultato
            error = str(e)
            answer = f"Error: {error}"
            time_taken = round(time() - start_time, 3)
            prompt_tokens = completion_tokens = total_tokens = 0
            response_cost = 0.0
            print(f"Errore: {e}")

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
