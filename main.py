from src.evaluation.scorer import LLMEvaluator
from src.llm.runner import benchmark
import glob
import json
import os
import sys
from datetime import datetime

import pandas as pd

# Assicuriamo che BASE_DIR sia nel path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

# Import dei moduli dal tuo progetto (modifica il percorso se necessario)

MODELLI_LOCALI = [
    "qwen2.5:0.5b",
    "qwen2.5:1.5b",
    "qwen2.5:7b",
    "llama2:7b",
    "llama3.1",
    "qwen3.5:9b",
    "granite4.1:8b",
    "gpt-oss:20b",
]

PROMPT_MODES = ["Q", "Q+Domain", "Q+Onto+Domain"]
ONTOLOGY_NAME = "pizza.owl"

RESPONSES_DIR = os.path.join(BASE_DIR, "output", "responses")
COMPARISONS_DIR = os.path.join(BASE_DIR, "output", "comparisons")


# ==========================================
# 2. FASE INFERENZA (BENCHMARK CICLICO)
# ==========================================
def run_all_benchmarks(op, output_dir=RESPONSES_DIR):
    print("--- FASE 1: AVVIO ESECUZIONE BENCHMARK ---")
    models_to_run = MODELLI_LOCALI
    if op == 1:
        models_to_run = ["qwen2.5:0.5b", "llama2:7b"]
    if op == 2:
        models_to_run = ["qwen2.5:1.5b", "llama3.1",
                         "granite4.1:8b", "gpt-oss:20b"]

    for model in models_to_run:
        print(f"\nModello Corrente: {model}")
        for mode in PROMPT_MODES:
            try:
                benchmark(
                    model_name=model,
                    prompt_mode=mode,
                    specific_ontology=ONTOLOGY_NAME,
                    output_dir=output_dir
                )
            except Exception as e:
                print(
                    f"Errore durante l'esecuzione di {model} con modalità {mode}: {e}"
                )


# ==========================================
# 3. FASE VALUTAZIONE & ESTRAZIONE MATRICE
# ==========================================
def evaluate_and_generate_matrix(responses_dir=RESPONSES_DIR, comparisons_dir=COMPARISONS_DIR, timestamp=None):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("--- FASE 2: ELABORAZIONE RISPOSTE E VALUTAZIONE METRICHE ---")
    os.makedirs(comparisons_dir, exist_ok=True)

    evaluator = LLMEvaluator()

    # Prende tutti i file nella cartella responses
    json_files = glob.glob(os.path.join(responses_dir, "*.json"))

    matrix_results = []

    for file_path in json_files:
        filename = os.path.basename(file_path)

        # Salta esplicitamente il file 'a' o altri file non validi
        if filename == "a" or not filename.endswith(".json"):
            continue

        print(f"📖 Analisi file: {filename}")

        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        model_name = data.get("model", "Unknown")
        prompt_mode = data.get("prompt_mode", "Unknown")

        responses = data.get("responses", [])

        for resp in responses:
            qid = resp.get("QID") or resp.get("qid")
            onto = resp.get("ontology_context")
            question = resp.get("question")
            sparql = resp.get("SPARQL")
            llm_answer = resp.get("answer", "")
            fc_ans = resp.get("FC_Ans") or resp.get("ground_truth", "")
            metrics = resp.get("metrics", {})

            eval_metrics = evaluator.compute_metrics(
                llm_answer_text=llm_answer,
                fc_answer=fc_ans if fc_ans else "N/A",
                sparql_query=sparql if sparql else "N/A",
                ground_truth_text=None
            )

            # Costruzione riga dell'array bidimensionale
            row = {
                "QID": qid,
                "Ontology": onto,
                "Model": model_name,
                "Prompt_Mode": prompt_mode,
                "Question": question,
                "Ground_Truth_FC": fc_ans,
                "LLM_Answer": llm_answer,
                "Runtime_Sec": metrics.get("runtime_seconds", 0.0),
                "Tokens_Per_Sec": metrics.get("tokens_per_second", 0.0),
                "Completion_Tokens": metrics.get("completion_tokens", 0),
                "Cosine_Similarity": eval_metrics["cosine_similarity"],
                "BLEU_Score": eval_metrics["bleu_score"],
                "Exact_Match": eval_metrics["exact_match"],
            }

            matrix_results.append(row)

    # 1. Salvataggio Matrice JSON
    json_output_path = os.path.join(
        comparisons_dir, f"matrix_evaluation_{timestamp}.json"
    )
    with open(json_output_path, "w", encoding="utf-8") as f:
        json.dump(matrix_results, f, indent=4, ensure_ascii=False)

    # 2. Salvataggio Tabella CSV (utilissima per DataFrame/Pandas/Excel)
    df = pd.DataFrame(matrix_results)
    # csv_output_path = os.path.join(
    #    comparisons_dir, f"matrix_evaluation_{timestamp}.csv"
    # )
    # df.to_csv(csv_output_path, index=False, encoding="utf-8-sig")

    print(
        f"\nValutazione completata con successo!\n"
        f"Matrice JSON salvata in: {json_output_path}\n"
        # f"Tabella CSV salvata in: {csv_output_path}"
    )

    return df


# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == "__main__":
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(BASE_DIR, "output", "responses", timestamp)
    os.makedirs(output_dir, exist_ok=True)
    run_all_benchmarks(op=0, output_dir=output_dir)
    #output_dir = os.path.join(BASE_DIR, "output", "responses","20260831_113831")
    df_matrix = evaluate_and_generate_matrix(
        responses_dir=output_dir, comparisons_dir=COMPARISONS_DIR, timestamp=timestamp)

    print("\n📋 Anteprima della Matrice Risultati:")
    if not df_matrix.empty:
        columns_to_show = [
            "QID",
            "Model",
            "Prompt_Mode",
            "Exact_Match",
            "Cosine_Similarity",
            "Tokens_Per_Sec",
        ]
        existing_cols = [
            col for col in columns_to_show if col in df_matrix.columns]
        print(df_matrix[existing_cols].head(10))
    else:
        print("Attenzione: Il DataFrame è vuoto! Nessun file di risposta valido è stato trovato in output/responses/.")
