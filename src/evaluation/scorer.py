import json
import evaluate
from sentence_transformers import SentenceTransformer, util


class LLMEvaluator:
    def __init__(self):
        # Carica il modello per le similarità semantiche
        self.encoder = SentenceTransformer('all-MiniLM-L6-v2')

        # Carica la metrica BLEU di Hugging Face (equivalente robusto a sacrebleu/nltk)
        self.bleu_metric = evaluate.load("bleu")

    def compute_metrics(self, llm_answer_text, ground_truth_text, true_boolean_ans=None):
        # 1. Similarity con Sentence Transformers
        emb_gt = self.encoder.encode(ground_truth_text, convert_to_tensor=True)
        emb_llm = self.encoder.encode(llm_answer_text, convert_to_tensor=True)
        cosine_sim = float(util.cos_sim(emb_gt, emb_llm).item())

        # 2. BLEU Score con Hugging Face evaluate
        # La metrica vuole una lista di predizioni (stringhe) e una lista di liste di reference
        predictions = [llm_answer_text.strip()]
        references = [[ground_truth_text.strip()]]

        try:
            bleu_result = self.bleu_metric.compute(
                predictions=predictions, references=references)
            bleu = float(bleu_result.get("bleu", 0.0))
        except Exception:
            bleu = 0.0
            print("Warning: BLEU computation failed, setting BLEU score to 0.0")

        # 3. Exact Match su Booleani (se presente la risposta formale)
        exact_match = 0.0
        if true_boolean_ans is not None:
            llm_text_clean = llm_answer_text.lower()
            if true_boolean_ans == "Yes":
                exact_match = 1.0 if (
                    "yes" in llm_text_clean and "no" not in llm_text_clean) else 0.0
            elif true_boolean_ans == "No":
                exact_match = 1.0 if ("no" in llm_text_clean) else 0.0

        return {
            "cosine_similarity": round(cosine_sim, 4),
            "bleu_score": round(bleu, 4),
            "exact_match": exact_match
        }
