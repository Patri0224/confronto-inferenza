import argparse
import glob
import json
import os
import sys
from datetime import datetime

import pandas as pd

from src.evaluation.scorer import LLMEvaluator
from src.llm.runner import benchmark

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

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
ONTOLOGY_NAME = ["pizza.owl", "Rientra.rdf", "Rientra.ttl"]

RESPONSES_DIR = os.path.join(BASE_DIR, "output", "responses")
COMPARISONS_DIR = os.path.join(BASE_DIR, "output", "comparisons")


# ==========================================
# FASE INFERENZA (BENCHMARK CICLICO)
# ==========================================
def run_all_benchmarks(
    model_option,
    output_dir=RESPONSES_DIR,
    specific_ontology=ONTOLOGY_NAME[0],
    questions_path=None,
):
    print("--- FASE 1: AVVIO ESECUZIONE BENCHMARK ---")
    if model_option == 1:
        models_to_run = ["qwen2.5:0.5b", "qwen2.5:1.5b"]
    elif model_option == 2:
        models_to_run = ["qwen2.5:7b", "llama3.1", "granite4.1:8b", "gpt-oss:20b"]
    elif model_option == 3:
        models_to_run = MODELLI_LOCALI
    elif isinstance(model_option, list):
        models_to_run = model_option
    else:
        models_to_run = MODELLI_LOCALI

    if specific_ontology in ["Rientra.rdf", "Rientra.ttl"]:
        prompt_modes = ["RAG_b"]
    else:
        prompt_modes = PROMPT_MODES

    for model in models_to_run:
        print(f"\nModello Corrente: {model}")
        for mode in prompt_modes:
            try:
                benchmark(
                    model_name=model,
                    prompt_mode=mode,
                    specific_ontology=specific_ontology,
                    output_dir=output_dir,
                )
            except Exception as e:
                print(
                    f"Errore durante l'esecuzione di {model} con modalità {mode}: {e}"
                )


# ==========================================
# FASE VALUTAZIONE & ESTRAZIONE MATRICE
# ==========================================
def evaluate_and_generate_matrix(
    responses_dir=RESPONSES_DIR, comparisons_dir=COMPARISONS_DIR, timestamp=None
):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("--- FASE 2: ELABORAZIONE RISPOSTE E VALUTAZIONE METRICHE ---")
    os.makedirs(comparisons_dir, exist_ok=True)

    evaluator = LLMEvaluator()
    json_files = glob.glob(os.path.join(responses_dir, "*.json"))

    matrix_results = []
    accuracy_test_results = []

    for file_path in json_files:
        filename = os.path.basename(file_path)
        if not filename.endswith(".json"):
            continue

        print(f" Analisi file: {filename}")
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
                ground_truth_text=None,
            )

            row = {
                "QID": qid,
                "Ontology": onto,
                "Model": model_name,
                "Prompt_Mode": prompt_mode,
                "Question": question,
                "Ground_Truth_FC": fc_ans,
                "Short_Answer": eval_metrics.get("short_answer", ""),
                "LLM_Answer": llm_answer,
                "Runtime_Sec": metrics.get("runtime_seconds", 0.0),
                "Tokens_Per_Sec": metrics.get("tokens_per_second", 0.0),
                "Completion_Tokens": metrics.get("completion_tokens", 0),
                "Cosine_Similarity": eval_metrics.get("cosine_similarity", 0.0),
                "BLEU_Score": eval_metrics.get("bleu_score", 0.0),
                "Exact_Match": eval_metrics.get("exact_match", 0.0),
            }

            row_accuracy = {
                "QID": qid,
                "Model": model_name,
                "Prompt_Mode": prompt_mode,
                "Ground_Truth_FC": fc_ans,
                "Short_Answer": eval_metrics.get("short_answer", ""),
                "Exact_Match": eval_metrics.get("exact_match", 0.0),
            }

            matrix_results.append(row)
            accuracy_test_results.append(row_accuracy)

    json_matrix_path = os.path.join(
        comparisons_dir, f"matrix_evaluation_{timestamp}.json"
    )
    with open(json_matrix_path, "w", encoding="utf-8") as f:
        json.dump(matrix_results, f, indent=4, ensure_ascii=False)

    json_accuracy_path = os.path.join(
        comparisons_dir, f"accuracy_evaluation_{timestamp}.json"
    )
    with open(json_accuracy_path, "w", encoding="utf-8") as f:
        json.dump(accuracy_test_results, f, indent=4, ensure_ascii=False)

    df = pd.DataFrame(matrix_results)
    print(
        f"\nValutazione completata con successo!\n"
        f"Matrice JSON salvata in: {json_matrix_path}"
    )
    return df


# ==========================================
# CLI PARSER E MAIN EXECUTION
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Benchmark di inferenza ontologica e Text-to-SPARQL con LLM locali."
    )
    parser.add_argument(
        "--ontology",
        type=str,
        choices=ONTOLOGY_NAME,
        default=None,
        help="Ontologia di riferimento su cui eseguire il benchmark (es. pizza.owl, Rientra.ttl).",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Percorso al file JSON contenente le domande (default: data/input/questions.json).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Livello di modelli da eseguire ('1': piccoli, '2': grandi, '3': tutti) o lista separata da virgole.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Salta la fase di inferenza e valuta solo le risposte già presenti nella cartella di output più recente.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Cartella di destinazione personalizzata per le risposte.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    responses_base_dir = os.path.join(BASE_DIR, "output", "responses")

    # 1. Risoluzione ontologia target (da argomenti o interattiva)
    if args.ontology:
        chosen_ontology_name = args.ontology
    else:
        user_onto = input(
            f"Scegli l'ontologia da utilizzare [0(default): {ONTOLOGY_NAME[0]}, 1: {ONTOLOGY_NAME[1]}, 2: {ONTOLOGY_NAME[2]}]: "
        ).strip()
        if user_onto == "2":
            chosen_ontology_name = ONTOLOGY_NAME[2]
        elif user_onto == "1":
            chosen_ontology_name = ONTOLOGY_NAME[1]
        else:
            chosen_ontology_name = ONTOLOGY_NAME[0]

    # 2. Risoluzione modalità benchmark (da argomenti o interattiva)
    if args.skip_benchmark:
        rerun_option = 0
    elif args.models:
        if args.models in ["1", "2", "3"]:
            rerun_option = int(args.models)
        else:
            rerun_option = [m.strip() for m in args.models.split(",") if m.strip()]
    else:
        user_run = input(
            "Eseguire i benchmark? [0(default): No/Valuta esistenti, 1: Solo piccoli, 2: Solo grandi, 3: Tutti]: "
        ).strip()
        try:
            rerun_option = int(user_run)
            if rerun_option not in [0, 1, 2, 3]:
                rerun_option = 0
        except ValueError:
            rerun_option = 0

    # 3. Setup directory di output
    if args.output_dir:
        output_dir = args.output_dir
    else:
        output_dir = os.path.join(responses_base_dir, timestamp)

    # 4. Esecuzione o fallback su run esistente
    if rerun_option == 0:
        subdirs = [
            d
            for d in os.listdir(responses_base_dir)
            if os.path.isdir(os.path.join(responses_base_dir, d))
        ]
        if subdirs:
            latest_dir = sorted(subdirs)[-1]
            output_dir = os.path.join(responses_base_dir, latest_dir)
            print(f"Modalità sola valutazione: utilizzo directory più recente -> {output_dir}")
        else:
            output_dir = os.path.join(responses_base_dir, timestamp)
            os.makedirs(output_dir, exist_ok=True)
    else:
        os.makedirs(output_dir, exist_ok=True)
        run_all_benchmarks(
            rerun_option,
            output_dir=output_dir,
            specific_ontology=chosen_ontology_name,
            questions_path=args.questions,
        )

    # 5. Generazione della matrice
    df_matrix = evaluate_and_generate_matrix(
        responses_dir=output_dir,
        comparisons_dir=COMPARISONS_DIR,
        timestamp=timestamp,
    )

    print("\n Anteprima della Matrice Risultati:")
    if not df_matrix.empty:
        columns_to_show = [
            "QID",
            "Model",
            "Prompt_Mode",
            "Exact_Match",
            "Cosine_Similarity",
            "Tokens_Per_Sec",
        ]
        existing_cols = [c for c in columns_to_show if c in df_matrix.columns]
        print(df_matrix[existing_cols].head(10))
    else:
        print("Attenzione: Il DataFrame è vuoto! Nessun file di risposta valido trovato.")
