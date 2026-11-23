import argparse
import glob
import json
import math
import os
import sys
from datetime import datetime
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

import pandas as pd
from src.evaluation.scorer import LLMEvaluator
from src.llm.runner import NO_FULL_PASTE_ONTOLOGIES, RAG_ONTOLOGIES, benchmark

MODELLI_PICCOLI = [
    "granite3.3:8b",
    "granite4.2:8b",
    "llama3.1:8b",
    "qwen2.5:7b",
]
MODELLI_GRANDI = [
    "gpt-oss:20b",
    "qwen3.8:27b",
]
MODELLI_LOCALI = MODELLI_PICCOLI + MODELLI_GRANDI
MODEL_LEVELS = {
    "1": MODELLI_PICCOLI,
    "2": MODELLI_GRANDI,
    "3": MODELLI_LOCALI,
}

# Q+Onto+Domain incolla l'ontologia intera nel prompt; RAG_a usa solo le triple
# recuperate ed è applicabile solo a Rientra.ttl (caso speciale).
PROMPT_MODES = ["Q", "Q+Domain", "Q+Onto+Domain", "RAG_a"]
RAG_MODES = {"RAG_a"}
ONTOLOGY_NAME = ["pizza.owl", "Rientra.rdf", "Rientra.ttl", "foaf.rdf", "wine.rdf", "go.owl"]

# Risposte attese troppo lunghe per num_predict (2048) / num_ctx: punteggio dipendente
# dalla lunghezza dell'output, non dal ragionamento. Escluse dalle medie, riportate a parte.
OVERSIZED_QIDS = {
    18: "2720 righe (~33k token)",
    19: "501 righe (~4k token)",
    25: "288 righe (~5k token)",
    69: "9914 identificativi GO (~70k token)",
}

SUMMARY_METRICS = [
    "Exact_Match", "Cosine_Similarity", "BLEU_Score", "Runtime_Sec", "Tokens_Per_Sec",
]
# Risposta vuota (es. modello di ragionamento che esaurisce num_predict nel "thinking"):
# conta come risposta sbagliata, quindi punteggio 0, ed è riportata in N_Empty.
SCORE_METRICS = ["cosine_similarity", "bleu_score", "exact_match"]

RESPONSES_ROOT = Path(BASE_DIR) / "output" / "responses"


def ontology_folder_name(ontology_name):
    """'Rientra.rdf' -> 'Rientra_rdf' (evita collisioni tra .rdf e .ttl)."""
    return ontology_name.replace(".", "_")


def modes_for_ontology(ontology_name, modes):
    """RAG_a solo sulle ontologie con indice RAG; Q+Onto+Domain non su quelle troppo grandi."""
    return [
        m for m in modes
        if (m not in RAG_MODES or ontology_name in RAG_ONTOLOGIES)
        and not (m == "Q+Onto+Domain" and ontology_name in NO_FULL_PASTE_ONTOLOGIES)
    ]


def resolve_modes(modes_arg):
    if not modes_arg:
        return PROMPT_MODES
    modes = [m.strip() for m in modes_arg.split(",") if m.strip()]
    unknown = [m for m in modes if m not in PROMPT_MODES]
    if unknown:
        sys.exit(f"Modalità sconosciute: {unknown}. Valide: {PROMPT_MODES}")
    return modes


def resolve_models(models_arg):
    if not models_arg:
        return MODELLI_LOCALI
    if models_arg in MODEL_LEVELS:
        return MODEL_LEVELS[models_arg]
    return [m.strip() for m in models_arg.split(",") if m.strip()]


# ==========================================
# FASE INFERENZA (BENCHMARK CICLICO)
# ==========================================
def run_all_benchmarks(output_dir, specific_ontology, models, modes, questions_path=None):
    print("--- FASE 1: AVVIO ESECUZIONE BENCHMARK ---")
    for model in models:
        print(f"\nModello Corrente: {model}")
        for mode in modes_for_ontology(specific_ontology, modes):
            try:
                benchmark(
                    model_name=model,
                    prompt_mode=mode,
                    specific_ontology=specific_ontology,
                    output_dir=output_dir,
                    questions_path=questions_path,
                )
            except Exception as e:  # noqa: BLE001 - un modello non deve fermare gli altri
                print(
                    f"Errore durante l'esecuzione di {model} con modalità {mode}: {e}"
                )


# ==========================================
# FASE VALUTAZIONE & ESTRAZIONE MATRICE
# ==========================================
def _json_safe(value):
    """json.dump scrive NaN, che non è JSON valido: lo sostituisce con None."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _dump_json(rows, path):
    safe_rows = [{k: _json_safe(v) for k, v in row.items()} for row in rows]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_rows, f, indent=4, ensure_ascii=False)


def summarize(df, oversized=False):
    """Medie per Ontology/Model/Prompt_Mode escludendo risposte troncate o in errore.

    Le righe escluse restano nella matrice e vengono contate come statistica
    (N_Total, N_Truncated, N_Error, N_Empty, N_Valid). Le risposte vuote restano
    nelle medie con punteggio 0. Le domande con risposta attesa
    troppo lunga (OVERSIZED_QIDS) sono escluse dal riepilogo principale
    (oversized=False) e riportate in un riepilogo separato (oversized=True).
    """
    if df.empty:
        return pd.DataFrame()
    df = df[df["QID"].isin(OVERSIZED_QIDS) == oversized]
    if df.empty:
        return pd.DataFrame()
    keys = ["Ontology", "Model", "Prompt_Mode"]
    valid = df[~df["Context_Truncated"] & df["Error"].isna()]
    counts = df.groupby(keys).agg(
        N_Total=("QID", "size"),
        N_Truncated=("Context_Truncated", "sum"),
        N_Error=("Error", lambda s: int(s.notna().sum())),
        N_Empty=("Is_Empty", "sum"),
    )
    means = valid.groupby(keys)[SUMMARY_METRICS].mean()
    summary = counts.join(means, how="left")
    summary["N_Valid"] = valid.groupby(keys).size().reindex(summary.index).fillna(0).astype(int)
    return summary.reset_index()


def evaluate_and_generate_matrix(
    responses_dir, comparisons_dir, evaluator, timestamp=None
):
    if timestamp is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("--- FASE 2: ELABORAZIONE RISPOSTE E VALUTAZIONE METRICHE ---")
    os.makedirs(comparisons_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(responses_dir, "*.json")))

    matrix_results = []
    accuracy_test_results = []

    for file_path in json_files:
        filename = os.path.basename(file_path)

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
            error = resp.get("error")
            extra_info = resp.get("extra_info") or {}
            llm_answer = resp.get("answer", "")
            fc_ans = resp.get("FC_Ans") or resp.get("ground_truth", "")
            metrics = resp.get("metrics", {})

            # Le risposte in errore non vanno valutate come risposte reali
            eval_metrics = evaluator.compute_metrics(
                llm_answer_text="" if error else llm_answer,
                fc_answer=fc_ans if fc_ans else "N/A",
                sparql_query=sparql if sparql else "N/A",
                ground_truth_text=None,
            )
            # Risposta vuota senza errore: sbagliata, non mancante (punteggio 0)
            is_empty = not error and not (llm_answer or "").strip()
            if is_empty:
                eval_metrics = {**eval_metrics, **{k: 0.0 for k in SCORE_METRICS}}

            row = {
                "QID": qid,
                "Ontology": onto,
                "Model": model_name,
                "Prompt_Mode": prompt_mode,
                "Question": question,
                "Ground_Truth_FC": fc_ans,
                "Short_Answer": eval_metrics.get("short_answer", ""),
                "LLM_Answer": llm_answer,
                "Error": error,
                "Is_Empty": bool(is_empty),
                "Context_Truncated": bool(extra_info.get("context_truncated", False)),
                "Prompt_Tokens": metrics.get("prompt_tokens") or 0,
                "Runtime_Sec": metrics.get("runtime_seconds") or 0.0,
                "Tokens_Per_Sec": metrics.get("tokens_per_second") or 0.0,
                "Completion_Tokens": metrics.get("completion_tokens") or 0,
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
    _dump_json(matrix_results, json_matrix_path)

    json_accuracy_path = os.path.join(
        comparisons_dir, f"accuracy_evaluation_{timestamp}.json"
    )
    _dump_json(accuracy_test_results, json_accuracy_path)

    df = pd.DataFrame(matrix_results)
    for oversized, name in ((False, "summary"), (True, "summary_oversized")):
        summary = summarize(df, oversized=oversized)
        if summary.empty:
            continue
        json_summary_path = os.path.join(
            comparisons_dir, f"{name}_evaluation_{timestamp}.json"
        )
        _dump_json(summary.to_dict(orient="records"), json_summary_path)
        print(f"Riepilogo '{name}' (senza risposte troncate/in errore) salvato in: {json_summary_path}")
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
        help="Ontologia di riferimento su cui eseguire il benchmark (default: tutte).",
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
        "--modes",
        type=str,
        default=None,
        help=f"Lista separata da virgole di modalità di prompt tra {PROMPT_MODES} (default: tutte). "
             f"RAG_a è eseguita solo su {sorted(RAG_ONTOLOGIES)}.",
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
        help="Cartella base di destinazione per le risposte (default: output/responses/<timestamp>).",
    )
    return parser.parse_args()


def latest_responses_dir():
    """Cartella più recente che contiene sottocartelle per ontologia."""
    if not RESPONSES_ROOT.is_dir():
        return None
    candidates = [
        d for d in RESPONSES_ROOT.iterdir()
        if d.is_dir() and any((d / ontology_folder_name(o) / "output").is_dir()
                              for o in ONTOLOGY_NAME)
    ]
    return max(candidates, key=lambda d: d.name) if candidates else None


def print_preview(df_matrix):
    print("\n Anteprima della Matrice Risultati:")
    if df_matrix.empty:
        print("Attenzione: Il DataFrame è vuoto! Nessun file di risposta valido trovato.")
        return
    columns_to_show = [
        "QID",
        "Model",
        "Prompt_Mode",
        "Exact_Match",
        "Cosine_Similarity",
        "Tokens_Per_Sec",
        "Context_Truncated",
    ]
    existing_cols = [c for c in columns_to_show if c in df_matrix.columns]
    print(df_matrix[existing_cols].head(10))


def main():
    args = parse_arguments()
    models = resolve_models(args.models)
    modes = resolve_modes(args.modes)
    ontologies = [args.ontology] if args.ontology else ONTOLOGY_NAME
    questions_path = str(Path(args.questions).resolve()) if args.questions else None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    if args.skip_benchmark:
        responses_base_dir = (
            Path(args.output_dir) if args.output_dir else latest_responses_dir()
        )
        if responses_base_dir is None or not responses_base_dir.is_dir():
            sys.exit("Nessuna cartella di risposte trovata da valutare.")
    else:
        responses_base_dir = (
            Path(args.output_dir) if args.output_dir else RESPONSES_ROOT / timestamp
        )

    evaluator = LLMEvaluator()

    for chosen_ontology_name in ontologies:
        ontology_result_folder = responses_base_dir / ontology_folder_name(chosen_ontology_name)
        output_dir = ontology_result_folder / "output"
        comparison_dir = ontology_result_folder / "comparison"

        if args.skip_benchmark:
            if not output_dir.is_dir():
                print(f"Nessuna risposta per {chosen_ontology_name} in {output_dir}, salto.")
                continue
        else:
            output_dir.mkdir(exist_ok=True, parents=True)
            run_all_benchmarks(
                output_dir=output_dir,
                specific_ontology=chosen_ontology_name,
                models=models,
                modes=modes,
                questions_path=questions_path,
            )
        comparison_dir.mkdir(exist_ok=True, parents=True)

        df_matrix = evaluate_and_generate_matrix(
            responses_dir=output_dir,
            comparisons_dir=comparison_dir,
            evaluator=evaluator,
            timestamp=timestamp,
        )
        print_preview(df_matrix)


if __name__ == "__main__":
    main()
