import json
import evaluate
from sentence_transformers import SentenceTransformer, util


class LLMEvaluator:
    def __init__(self):
        # Carica il modello per le similarità semantiche
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Carica la metrica BLEU di Hugging Face (equivalente robusto a sacrebleu/nltk)
        self.bleu_metric = evaluate.load("bleu")

    def compute_metrics(self, llm_answer_text, fc_answer, sparql_query, ground_truth_text=None):
        parts = llm_answer_text.split("---")
        short_part = parts[0].strip() if len(
            parts) > 0 else llm_answer_text.strip()
        explanation_part = parts[1].strip() if len(parts) > 1 else ""

        is_ask = "ASK" in sparql_query.upper() if sparql_query else "SELECT"
        ref_text = ground_truth_text if ground_truth_text else fc_answer
        text_to_compare = explanation_part if explanation_part else llm_answer_text
        # 1. Similarity con Sentence Transformers
        emb_gt = self.encoder.encode(ref_text, convert_to_tensor=True)
        emb_llm = self.encoder.encode(text_to_compare, convert_to_tensor=True)
        cosine_sim = float(util.cos_sim(emb_gt, emb_llm).item())

        # 2. BLEU Score con Hugging Face evaluate
        predictions = [text_to_compare]  # BLEU richiede una lista di predizioni
        references = [[ref_text]]  # BLEU richiede una lista di liste per le referenze

        try:
            bleu_result = self.bleu_metric.compute(
                predictions=predictions, references=references)
            bleu = float(bleu_result.get("bleu", 0.0))
        except Exception:
            bleu = 0.0
            print("Warning: BLEU computation failed, setting BLEU score to 0.0")

        # 3. Exact Match su Booleani (se presente la risposta formale)
        short_lower = short_part.lower()
        if is_ask:
            target_ans = fc_answer.strip().lower()
            if target_ans == "yes":
                exact_match = 1.0 if (
                    "yes" in short_lower and "no" not in short_lower) else 0.0
            elif target_ans == "no":
                exact_match = 1.0 if (
                    "no" in short_lower and "yes" not in short_lower) else 0.0
        else:
            gt_items = {item.strip().lower() for item in fc_answer.replace(
                ";", ",").split(",") if item.strip()}
            llm_items = {item.strip().lower() for item in short_part.replace(
                ";", ",").split(",") if item.strip()}

            n = len(gt_items)
            if n > 0:
                correct_hits = len(gt_items.intersection(llm_items))
                extra_items = len(llm_items - gt_items)

                # Formula: 1/n per ogni corretto, -1/(n*2) per ogni in più
                score = (correct_hits / n) - (extra_items / (n * 2))
                exact_match = max(0.0, min(1.0, score))

        return {
            "cosine_similarity": round(cosine_sim, 4),
            "bleu_score": round(bleu, 4),
            "exact_match": exact_match
        }
