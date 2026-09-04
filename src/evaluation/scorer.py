import evaluate
from sentence_transformers import SentenceTransformer, util
import re

SYNONYM_MAPPINGS = {
    # Per QID 13 e domande generali sui topping di pesce
    "sea food topping": [
        "anchovies", "prawns", "mixedseafood", "seafood", "fish", "squid",
        "clam", "mussel", "crab", "lobster", "scallop", "anchoviestopping",
        "mixedseafoodtopping", "prawnstopping", "sea food"
    ],

    # Per QID 9 e 15 (Frutti di mare / Seafood)
    "frutti di mare": [
        "fruttidimare", "seafood", "frutti di mare", "fruttidimarepizza", "FishPizza"
    ],

    # Per QID 3 (Basi incompatibili)
    "deep pan base": [
        "deep pan", "deeppanbase", "deeppan", "deep_pan"
    ],

    # Per categorie generali di topping formaggio
    "cheese topping": [
        "cheese", "mozzarella", "parmesan", "gorgonzola", "cheesetopping"
    ]
}


class LLMEvaluator:
    def __init__(self):
        # Carica il modello per le similarità semantiche
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Carica la metrica BLEU di Hugging Face (equivalente robusto a sacrebleu/nltk)
        self.bleu_metric = evaluate.load("bleu")

        REVERSE_SYNONYM_MAP = {}
        for key, synonyms in SYNONYM_MAPPINGS.items():
            norm_key = key.lower().strip()
            for syn in synonyms:
                REVERSE_SYNONYM_MAP[syn.lower().strip()] = norm_key
        self.reverse_synonym_map = REVERSE_SYNONYM_MAP

    def parse_llm_response(self, llm_answer_text):
        if not llm_answer_text:
            return "", ""

        if re.search(r'\-{3,}', llm_answer_text):
            parts = re.split(r'\-{3,}', llm_answer_text, 1)
            raw_short = parts[0].strip()
            explanation_part = parts[1].strip() if len(parts) > 1 else ""
        else:
            lines = llm_answer_text.strip().split("\n", 1)
            raw_short = lines[0].strip()
            explanation_part = lines[1].strip() if len(lines) > 1 else ""

        cleaned_short = re.sub(
            r'^(?:\[)?\s*(?:short\s*answer|answer)\s*[:\]\-]*\s*', '', raw_short, flags=re.IGNORECASE).strip()
        cleaned_short = cleaned_short.strip('[]').strip()

        return cleaned_short, explanation_part

    def compute_metrics(self, llm_answer_text, fc_answer, sparql_query, ground_truth_text=None):
        if not llm_answer_text:
            return {
                "cosine_similarity": float('nan'),
                "bleu_score": float('nan'),
                "exact_match": float('nan'),
                "short_answer": ""
            }

        short_part, explanation_part = self.parse_llm_response(llm_answer_text)

        is_ask = "ASK" in sparql_query.upper() if sparql_query else "SELECT"
        ref_text = ground_truth_text if ground_truth_text else fc_answer
        text_to_compare = explanation_part if explanation_part else llm_answer_text
        # 1. Similarity con Sentence Transformers
        emb_gt = self.encoder.encode(ref_text, convert_to_tensor=True)
        emb_llm = self.encoder.encode(text_to_compare, convert_to_tensor=True)
        cosine_sim = float(util.cos_sim(emb_gt, emb_llm).item())

        # 2. BLEU Score con Hugging Face evaluate
        # BLEU richiede una lista di predizioni
        predictions = [text_to_compare]
        # BLEU richiede una lista di liste per le referenze
        references = [[ref_text]]

        try:
            bleu_result = self.bleu_metric.compute(
                predictions=predictions, references=references)
            bleu = float(bleu_result.get("bleu", 0.0))
        except Exception:
            bleu = float('nan')
            print("Warning: BLEU computation failed, setting BLEU score to 0.0")

        # 3. Exact Match (accuracy) su short_answer
        exact_match = 0.0
        short_lower = short_part.lower()
        fc_lower = fc_answer.strip().lower()

        if is_ask:
            target_ans = fc_lower
            if target_ans == "yes":
                exact_match = 1.0 if (
                    "yes" in short_lower and "no" not in short_lower) else 0.0
            elif target_ans == "no":
                exact_match = 1.0 if (
                    "no" in short_lower and "yes" not in short_lower) else 0.0
        else:
            # caso None in SELECT
            if fc_lower in ["none", "no", ""]:
                empty_keywords = ["no", "none", "nothing",
                                  "n/a", "nessuna", "nessuno", "empty", "-"]
                if any(kw in short_lower for kw in empty_keywords):
                    exact_match = 1.0
                else:
                    exact_match = 0.0
            else:
                gt_items = {item.strip().lower() for item in fc_answer.replace(
                    ";", ",").split(",") if item.strip()}
                llm_items = {item.strip().lower() for item in short_part.replace(
                    ";", ",").split(",") if item.strip()}
                llm_items = {self.reverse_synonym_map.get(
                    item, item) for item in llm_items}

                n = len(gt_items)
                if n > 0:
                    correct_hits = len(gt_items.intersection(llm_items))
                    extra_items = len(llm_items - gt_items)
                    score = (correct_hits / n) - (extra_items / (n * n))
                    exact_match = max(0.0, min(1.0, score))

        return {
            "cosine_similarity": round(cosine_sim, 4),
            "bleu_score": round(bleu, 4),
            "exact_match": exact_match,
            "short_answer": short_part,
        }
