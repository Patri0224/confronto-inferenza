class Scorer:
    def __init__(self, scoring_function):
        self.scoring_function = scoring_function

    def score(self, reference_answer, generated_answer):
        return self.scoring_function(reference_answer, generated_answer)
    
    def scoring_function(self, reference_answer, generated_answer):
        # Implement your scoring logic here
        # For example, you can use string similarity, BLEU score, etc.
        # This is a placeholder implementation that returns a dummy score
        return 1.0 if reference_answer == generated_answer else 0.0