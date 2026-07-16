class PromptGenerator:
    def __init__(self, system_role=None, domain_constraint=None):
        self.system_role = system_role or (
            "You are a specialized model for answering questions using ontology-based knowledge."
        )
        self.domain_constraint = domain_constraint or (
            "Focus strictly on pizza-related topics."
        )

    def generate_prompt(self, question, ontology_context=None, mode="Q+Onto+Domain"):
        #Q, Q+Onto, Q+Domain, Q+Onto+Domain
        
        prompt_parts = []

        if "Onto" in mode:
            prompt_parts.append(self.system_role)
            if ontology_context:
                prompt_parts.append(f"Refer to the following Pizza Ontology when responding to questions:\n{ontology_context}")
            else:
                prompt_parts.append("Refer to your internal ontology knowledge when responding.")

        if "Domain" in mode:
            prompt_parts.append(self.domain_constraint)

        prompt_parts.append(f"Question: {question}")
        prompt_parts.append("Answer:")

        return "\n\n".join(prompt_parts)

