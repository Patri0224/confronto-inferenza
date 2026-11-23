import evaluate
from sentence_transformers import SentenceTransformer, util
import re

SYNONYM_MAPPINGS = {
    # For QID 13 and general sea-food-topping questions
    "sea food topping": [
        "anchovies", "prawns", "mixedseafood", "seafood", "fish", "squid",
        "clam", "mussel", "crab", "lobster", "scallop", "anchoviestopping",
        "mixedseafoodtopping", "prawnstopping", "sea food"
    ],

    # For QID 9 and 15 (Frutti di mare)
    # NB: "seafood" belongs to "sea food topping", do not duplicate it here
    "frutti di mare": [
        "fruttidimare", "frutti di mare", "fruttidimarepizza", "FishPizza"
    ],

    # For QID 3 (incompatible bases)
    "deep pan base": [
        "deep pan", "deeppanbase", "deeppan", "deep_pan"
    ],

    # For general cheese-topping categories
    "cheese topping": [
        "cheese", "mozzarella", "parmesan", "gorgonzola", "cheesetopping"
    ]
}


def normalize_item(text):
    """Lowercase, without namespace prefix and without separators.

    'Quattro Formaggi' == 'QuattroFormaggi'; 'foaf:Person' == 'Person'.
    """
    text = str(text).strip()
    # CURIE GO:0000278 and ID GO_0000278 are the same identifier: keep the prefix
    text = re.sub(r'^GO:(?=\d)', 'GO_', text, flags=re.IGNORECASE)
    text = re.sub(r'^(?!https?:)(?!GO_)[A-Za-z][\w\-]*:', '', text, flags=re.IGNORECASE)
    return re.sub(r'[\s_\-]+', '', text.lower())


class LLMEvaluator:
    def __init__(self):
        # Load the model for semantic similarities
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Load the Hugging Face BLEU metric (robust equivalent of sacrebleu/nltk)
        self.bleu_metric = evaluate.load("bleu")

        REVERSE_SYNONYM_MAP = {}
        for key, synonyms in SYNONYM_MAPPINGS.items():
            norm_key = normalize_item(key)
            for syn in synonyms:
                norm_syn = normalize_item(syn)
                if REVERSE_SYNONYM_MAP.get(norm_syn, norm_key) != norm_key:
                    raise ValueError(
                        f"Ambiguous synonym in SYNONYM_MAPPINGS: '{syn}'")
                REVERSE_SYNONYM_MAP[norm_syn] = norm_key
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

        # PREFIX/BASE declarations may also sit on the same line as the query
        query_body = re.sub(
            r'\b(?:PREFIX\s+[\w\-]*:\s*<[^>]*>|BASE\s*<[^>]*>)', '',
            sparql_query or '', flags=re.IGNORECASE)
        is_ask = bool(re.match(r'\s*ASK\b', query_body, re.IGNORECASE))
        ref_text = ground_truth_text if ground_truth_text else fc_answer
        # The ground truth is a short answer: compare it with the short answer
        text_to_compare = short_part if short_part else llm_answer_text
        # 1. Similarity with Sentence Transformers
        emb_gt = self.encoder.encode(ref_text, convert_to_tensor=True)
        emb_llm = self.encoder.encode(text_to_compare, convert_to_tensor=True)
        cosine_sim = float(util.cos_sim(emb_gt, emb_llm).item())

        # 2. BLEU score with Hugging Face evaluate
        # BLEU requires a list of predictions
        predictions = [text_to_compare]
        # BLEU requires a list of lists for the references
        references = [[ref_text]]

        try:
            bleu_result = self.bleu_metric.compute(
                predictions=predictions, references=references)
            bleu = float(bleu_result.get("bleu", 0.0))
        except Exception:
            bleu = float('nan')
            print("Warning: BLEU computation failed, setting BLEU score to 0.0")

        # 3. Exact match (accuracy) on short_answer
        exact_match = 0.0
        short_lower = short_part.lower()
        fc_lower = fc_answer.strip().lower()
        has_yes = bool(re.search(r'\byes\b', short_lower))
        has_no = bool(re.search(r'\bno\b', short_lower))

        if is_ask:
            target_ans = fc_lower
            if target_ans == "yes":
                exact_match = 1.0 if (has_yes and not has_no) else 0.0
            elif target_ans == "no":
                exact_match = 1.0 if (has_no and not has_yes) else 0.0
        else:
            # None case in SELECT
            if fc_lower in ["none", "no", ""]:
                empty_words = ["no", "none", "nothing",
                               "n/a", "nessuna", "nessuno", "empty"]
                is_empty_answer = (
                    short_lower.strip() == "-"
                    or any(re.search(rf'\b{re.escape(kw)}\b', short_lower)
                           for kw in empty_words))
                exact_match = 1.0 if is_empty_answer else 0.0
            else:
                def to_items(text):
                    return {normalize_item(item)
                            for item in text.replace(";", ",").split(",")
                            if item.strip()}

                gt_items = {self.reverse_synonym_map.get(i, i)
                            for i in to_items(fc_answer)}
                llm_items = {self.reverse_synonym_map.get(i, i)
                             for i in to_items(short_part)}

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
