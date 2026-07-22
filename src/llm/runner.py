from src.pipelines.prompt_generator import PromptGenerator
import json
import os
import sys
import litellm
from time import time
from datetime import datetime


BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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


def benchmark(model_name, prompt_mode="Q+Onto+Domain", specific_ontology=None):
    print(f"Running benchmark for model: {model_name} with prompt mode: {prompt_mode}")
    if model_name in MODELLI_LOCALI:
        api_base = "http://localhost:11434"
        model_name = f"ollama/{model_name}"
    else:
        api_base = None

    # carica le domande dal file JSON e inizializza il generatore di prompt
    with open(os.path.join(INPUT_PATH, "questions.json"), "r") as f:
        questions = json.load(f)
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
        if specific_ontology:
            if not ontology_context or specific_ontology.strip().lower() != ontology_context.strip().lower():
                continue
        prompt = generator.generate_prompt(question_text, ontology_context, prompt_mode)

        # esegue domanda su modello
        start_time = time()
        try:
            response = litellm.completion(
                model=model_name,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                api_base=api_base
            )
            time_taken = round(time() - start_time, 3)
            answer = response.choices[0].message.content.strip()
            usage = response.get("usage", {})
            prompt_tokens = usage.get("prompt_tokens", 0)
            completion_tokens = usage.get("completion_tokens", 0)
            total_tokens = usage.get("total_tokens", 0)
            tokens_per_second = round(completion_tokens / time_taken, 2) if time_taken > 0 else 0
            response_cost = response._hidden_params.get(
                "response_cost", 0.0) if hasattr(response, "_hidden_params") else 0.0
            print(
                f"Completato in {time_taken}s | Output: {completion_tokens} tok ({tokens_per_second} tok/s)")
        except Exception as e:
            answer = f"Error: {str(e)}"
            time_taken = None
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            tokens_per_second = 0
            response_cost = 0.0
            print(f"Errore: {e}")
        # salva risposta
        results["responses"].append({
            "question": question_text,
            "ontology_context": ontology_context,
            "prompt": prompt,
            "answer": answer,
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
        print(results["responses"][-1])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_model_name = model_name.replace(':', '_').replace('/', '_')
    name_file_output = f"{safe_model_name}_{prompt_mode}_{timestamp}.json"
    percorso_salvataggio = os.path.join(OUTPUT_PATH, name_file_output)
    with open(percorso_salvataggio, "w") as f:
        json.dump(results, f, indent=4)


if __name__ == "__main__":
    print(BASE_DIR)
    benchmark(model_name="llama2:7b", prompt_mode="Q+Domain", specific_ontology="pizza.owl")
