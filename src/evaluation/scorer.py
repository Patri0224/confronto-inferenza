import json
from sentence_transformers import SentenceTransformer, util
from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

class LLMEvaluator:
    def __init__(self):
        # Carica il modello per le similarità semantiche
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')
        self.smooth = SmoothingFunction().method1

    def compute_metrics(self, llm_answer_text, ground_truth_text, true_boolean_ans=None):
        # 1. Similarity con Sentence Transformers
        emb_gt = self.encoder.encode(ground_truth_text, convert_to_tensor=True)
        emb_llm = self.encoder.encode(llm_answer_text, convert_to_tensor=True)
        cosine_sim = float(util.cos_sim(emb_gt, emb_llm).item())

        # 2. BLEU Score
        ref_tokens = [ground_truth_text.lower().split()]
        cand_tokens = llm_answer_text.lower().split()
        bleu = sentence_bleu(ref_tokens, cand_tokens, smoothing_function=self.smooth)

        # 3. Exact Match su Booleani (se presente la risposta formale)
        exact_match = None
        if true_boolean_ans is not None:
            llm_text_clean = llm_answer_text.lower()
            if true_boolean_ans == "Yes":
                exact_match = 1.0 if ("yes" in llm_text_clean and "no" not in llm_text_clean) else 0.0
            elif true_boolean_ans == "No":
                exact_match = 1.0 if ("no" in llm_text_clean or "not" in llm_text_clean) else 0.0

        return {
            "cosine_similarity": round(cosine_sim, 4),
            "bleu_score": round(bleu, 4),
            "exact_match": exact_match
        }

if __name__ == "__main__":
    evaluator = LLMEvaluator()
    metrics = evaluator.compute_metrics(
        llm_answer_text="Yes, a FourCheesePizza is considered a VegetarianPizza.",
        ground_truth_text="Yes",
        true_boolean_ans="Yes"
    )
    print("Metriche Calcolate:", metrics)