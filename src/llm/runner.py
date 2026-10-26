from src.pipelines.prompt_generator import PromptGenerator
import json
import os
import sys
import litellm
from time import time
from datetime import datetime
from src.pipelines.RAG_a import OntologyRAG
from src.pipelines.TextToSparql import TextToSparqlEngine

BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)


INPUT_PATH = os.path.join(BASE_DIR, "data", "input")
OUTPUT_PATH = os.path.join(BASE_DIR, "output", "responses")


MODELLI_LOCALI = [
    "qwen2.5:0.5b",
    "qwen2.5:1.5b",
    "qwen2.5:7b",
    "llama2:7b",
    "llama3.1",
    "qwen3.5:9b",
    "granite4.1:8b",
    "gpt-oss:20b"
]


def benchmark(model_name, prompt_mode="Q+Onto+Domain", specific_ontology=None, output_dir=OUTPUT_PATH):
    print(
        f"Running benchmark for model: {model_name} with prompt mode: {prompt_mode}")
    if model_name in MODELLI_LOCALI:
        api_base = "http://localhost:11434"
        model_name = f"ollama/{model_name}"
    else:
        api_base = None

    # carica le domande dal file JSON e inizializza il generatore di prompt
    with open(os.path.join(INPUT_PATH, "questions.json"), "r") as f:
        questions = json.load(f)

    initialize_rag_a = True if prompt_mode == "RAG_a" and specific_ontology == "Rientra.ttl" else False
    rag_a = None
    if initialize_rag_a:
        rag_a = OntologyRAG(
            ontology_path="./data/ontologies/Rientra.ttl",
            persist_dir="./chroma_db",
            collection_name="rientra_rag"
        )
        generator = PromptGenerator(rag_a=rag_a)

    initialize_rag_b = True if prompt_mode == "RAG_b" and specific_ontology == "Rientra.ttl" else False
    rag_b = None
    if initialize_rag_b:
        rag_b = TextToSparqlEngine(
            ontology_path="./data/ontologies/Rientra.ttl")
        generator = PromptGenerator(rag_b=rag_b)

    if not initialize_rag_a and not initialize_rag_b:
        generator = PromptGenerator()
    # inizializza il dizionario dei risultati
    results = {
        "model": model_name,
        "prompt_mode": prompt_mode,
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "responses": []
    }

    for q in questions:
        # genera il prompt per la domanda corrente

        question_text = q.get("Question")
        ontology_context = q.get("Ontology")
        if specific_ontology == "Rientra.ttl" and ontology_context == "Rientra.rdf":
            # Forza l'ontologia a Rientra.ttl per il confronto
            ontology_context = "Rientra.ttl"
        if specific_ontology:
            if not ontology_context or specific_ontology.strip().lower() != ontology_context.strip().lower():
                continue
        sparql_query = q.get("SPARQL")
        system_prompt, user_prompt = generator.generate_prompt(
            question=question_text,
            ontology_context=ontology_context,
            mode=prompt_mode,
            sparql=sparql_query
        )
        # esegue domanda su modello
        start_time = time()
        try:

            if initialize_rag_b:
                risultato = rag_b.run_text_to_sparql(
                    question=question_text,
                    model="ollama/llama3.1"
                )

                time_taken = round(time() - start_time, 3)
                answer = risultato.get("short_answer", "")
                extra_info = {}
                extra_info["generated_sparql"] = risultato.get(
                    "generated_sparql", "")
                extra_info["raw_output"] = risultato.get("raw_output", "")
                usage = risultato.get("metrics", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                tokens_per_second = round(
                    completion_tokens / time_taken, 2) if time_taken > 0 else 0
                response_cost = usage.get("cost", 0.0)
                # salva risposta
                results["responses"].append({
                    "QID": q.get("QID"),
                    "question": question_text,
                    "ontology_context": ontology_context,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "SPARQL": sparql_query,
                    "FC_Ans": q.get("FC_Ans"),
                    "answer": answer,
                    "extra_info": extra_info,
                    # "correct_answer": q.get("correct_answer"),
                    "metrics": {
                        "runtime_seconds": time_taken,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "tokens_per_second": tokens_per_second,
                        "cost": response_cost
                    }
                })
            else:
                response = litellm.completion(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.0,
                    api_base=api_base,
                    # Parametri specifici passati direttamente al motore Ollama
                    extra_body={
                        "options": {
                            "num_ctx": 24576,      # Finestra di contesto estesa per contenere l'ontologia
                            "num_predict": 2048,    # Budget per la generazione dell'inferenza
                            "temperature": 0.0,
                            "repeat_penalty": 1.0,
                        }
                    }
                )
                time_taken = round(time() - start_time, 3)
                answer = response.choices[0].message.content.strip()
                extra_info = {}
                usage = response.get("usage", {})
                prompt_tokens = usage.get("prompt_tokens", 0)
                completion_tokens = usage.get("completion_tokens", 0)
                total_tokens = usage.get("total_tokens", 0)
                tokens_per_second = round(
                    completion_tokens / time_taken, 2) if time_taken > 0 else 0
                response_cost = response._hidden_params.get(
                    "response_cost", 0.0) if hasattr(response, "_hidden_params") else 0.0
                # salva risposta
                results["responses"].append({
                    "QID": q.get("QID"),
                    "question": question_text,
                    "ontology_context": ontology_context,
                    "system_prompt": system_prompt,
                    "user_prompt": user_prompt,
                    "SPARQL": sparql_query,
                    "FC_Ans": q.get("FC_Ans"),
                    "answer": answer,
                    "extra_info": extra_info,
                    # "correct_answer": q.get("correct_answer"),
                    "metrics": {
                        "runtime_seconds": time_taken,
                        "prompt_tokens": prompt_tokens,
                        "completion_tokens": completion_tokens,
                        "total_tokens": total_tokens,
                        "tokens_per_second": tokens_per_second,
                        "cost": response_cost
                    }
                })

        except Exception as e:
            answer = f"Error: {str(e)}"
            time_taken = None
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            tokens_per_second = 0
            response_cost = 0.0
            print(f"Errore: {e}")
            results["responses"].append({
                "QID": q.get("QID"),
                "question": question_text,
                "ontology_context": ontology_context,
                "system_prompt": system_prompt,
                "user_prompt": user_prompt,
                "SPARQL": sparql_query,
                "FC_Ans": q.get("FC_Ans"),
                "answer": answer,
                "extra_info": extra_info,
                # "correct_answer": q.get("correct_answer"),
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
    with open(percorso_salvataggio, "w") as f:
        json.dump(results, f, indent=4)
