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

# Q+Onto+Domain pastes the whole ontology into the prompt; RAG_a uses only the
# retrieved triples and is applicable only to the ontologies in RAG_CONFIG.
PROMPT_MODES = ["Q", "Q+Domain", "Q+Onto+Domain", "RAG_a"]
RAG_MODES = {"RAG_a"}
ONTOLOGY_NAME = ["pizza.owl", "Rientra.rdf", "Rientra.ttl", "foaf.rdf", "wine.rdf", "go.owl"]

# Expected answers too long for num_predict (2048) / num_ctx: the score would
# depend on output length, not on reasoning. Excluded from the averages and
# reported separately in the summary_oversized_* files.
OVERSIZED_QIDS = {
    18: "2720 righe (~33k token)",
    19: "501 righe (~4k token)",
    25: "288 righe (~5k token)",
    69: "9914 identificativi GO (~70k token)",
}

SUMMARY_METRICS = [
    "Exact_Match", "Cosine_Similarity", "BLEU_Score", "Runtime_Sec", "Tokens_Per_Sec",
]
# Empty answer (e.g. a reasoning model that exhausts num_predict while
# "thinking"): counts as a wrong answer, i.e. score 0, and is reported in N_Empty.
SCORE_METRICS = ["cosine_similarity", "bleu_score", "exact_match"]

RESPONSES_ROOT = Path(BASE_DIR) / "output" / "responses"


def ontology_folder_name(ontology_name):
    """'Rientra.rdf' -> 'Rientra_rdf' (avoids collisions between .rdf and .ttl)."""
    return ontology_name.replace(".", "_")


def modes_for_ontology(ontology_name, modes):
    """RAG_a only on ontologies with a RAG index; Q+Onto+Domain not on the oversized ones."""
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
        sys.exit(f"Unknown modes: {unknown}. Valid modes: {PROMPT_MODES}")
    return modes


def resolve_models(models_arg):
    if not models_arg:
        return MODELLI_LOCALI
    if models_arg in MODEL_LEVELS:
        return MODEL_LEVELS[models_arg]
    return [m.strip() for m in models_arg.split(",") if m.strip()]


# ==========================================
# INFERENCE PHASE (BENCHMARK LOOP)
# ==========================================
def run_all_benchmarks(output_dir, specific_ontology, models, modes, questions_path=None):
    print("--- PHASE 1: BENCHMARK EXECUTION ---")
    for model in models:
        print(f"\nCurrent model: {model}")
        for mode in modes_for_ontology(specific_ontology, modes):
            try:
                benchmark(
                    model_name=model,
                    prompt_mode=mode,
                    specific_ontology=specific_ontology,
                    output_dir=output_dir,
                    questions_path=questions_path,
                )
            except Exception as e:  # noqa: BLE001 - one model must not stop the others
                print(
                    f"Error while running {model} in mode {mode}: {e}"
                )


# ==========================================
# EVALUATION PHASE & MATRIX EXTRACTION
# ==========================================
def _json_safe(value):
    """json.dump writes NaN, which is not valid JSON: replace it with None."""
    if isinstance(value, float) and math.isnan(value):
        return None
    return value


def _dump_json(rows, path):
    safe_rows = [{k: _json_safe(v) for k, v in row.items()} for row in rows]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(safe_rows, f, indent=4, ensure_ascii=False)


def summarize(df, oversized=False):
    """Means per Ontology/Model/Prompt_Mode, excluding truncated or errored responses.

    Excluded rows stay in the matrix and are counted as statistics
    (N_Total, N_Truncated, N_Error, N_Empty, N_Valid). Empty answers stay
    in the means with score 0. Questions whose expected answer is too long
    (OVERSIZED_QIDS) are excluded from the main summary (oversized=False)
    and reported in a separate summary (oversized=True).
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
    print("--- PHASE 2: RESPONSE PROCESSING AND METRIC EVALUATION ---")
    os.makedirs(comparisons_dir, exist_ok=True)

    json_files = sorted(glob.glob(os.path.join(responses_dir, "*.json")))

    matrix_results = []
    accuracy_test_results = []

    for file_path in json_files:
        filename = os.path.basename(file_path)

        print(f" Analyzing file: {filename}")
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

            # Errored responses must not be evaluated as real answers
            eval_metrics = evaluator.compute_metrics(
                llm_answer_text="" if error else llm_answer,
                fc_answer=fc_ans if fc_ans else "N/A",
                sparql_query=sparql if sparql else "N/A",
                ground_truth_text=None,
            )
            # Empty answer without error: wrong, not missing (score 0)
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
        print(f"Summary '{name}' (without truncated/errored responses) saved to: {json_summary_path}")
    print(
        f"\nEvaluation completed successfully!\n"
        f"JSON matrix saved to: {json_matrix_path}"
    )
    return df


# ==========================================
# CLI PARSER AND MAIN EXECUTION
# ==========================================
def parse_arguments():
    parser = argparse.ArgumentParser(
        description="Ontology inference and Text-to-SPARQL benchmark with local LLMs."
    )
    parser.add_argument(
        "--ontology",
        type=str,
        choices=ONTOLOGY_NAME,
        default=None,
        help="Ontology to run the benchmark on (default: all).",
    )
    parser.add_argument(
        "--questions",
        type=str,
        default=None,
        help="Path to the JSON file with the questions (default: data/input/questions.json).",
    )
    parser.add_argument(
        "--models",
        type=str,
        default=None,
        help="Model level to run ('1': small, '2': large, '3': all) or a comma-separated list.",
    )
    parser.add_argument(
        "--modes",
        type=str,
        default=None,
        help=f"Comma-separated list of prompt modes among {PROMPT_MODES} (default: all). "
             f"RAG_a è eseguita solo su {sorted(RAG_ONTOLOGIES)}.",
    )
    parser.add_argument(
        "--skip-benchmark",
        action="store_true",
        help="Skip the inference phase and only score the responses already present in the latest output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Base output directory for the responses (default: output/responses/<timestamp>).",
    )
    return parser.parse_args()


def latest_responses_dir():
    """Most recent directory containing per-ontology subdirectories."""
    if not RESPONSES_ROOT.is_dir():
        return None
    candidates = [
        d for d in RESPONSES_ROOT.iterdir()
        if d.is_dir() and any((d / ontology_folder_name(o) / "output").is_dir()
                              for o in ONTOLOGY_NAME)
    ]
    return max(candidates, key=lambda d: d.name) if candidates else None


def print_preview(df_matrix):
    print("\n Preview of the results matrix:")
    if df_matrix.empty:
        print("Warning: the DataFrame is empty! No valid response file found.")
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
            sys.exit("No responses directory found to evaluate.")
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
                print(f"No responses for {chosen_ontology_name} in {output_dir}, skipping.")
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
