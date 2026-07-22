class PromptGenerator:
    def __init__(self, system_role=None, domain_constraint=None, answer_type=None):
        self.system_role = system_role or (
            "You are a specialized model for answering questions using ontology-based knowledge."
        )
        self.domain_constraint = domain_constraint or (
            "Focus strictly on pizza-related topics."
        )
        self.answer_type = answer_type or (
            "Mostra i passaggi di inferenza utilizzati per arrivare alla risposta")

    def generate_prompt(self, question, ontology_context=None, mode="Q+Onto+Domain"):
        # Q, Q+Onto, Q+Domain, Q+Onto+Domain

        prompt_parts = []
        ontology_context = self.getOntology(ontology_context, optional=True) if ontology_context else None
        if "Onto" in mode:
            prompt_parts.append(self.system_role)
            if ontology_context:
                prompt_parts.append(
                    f"Refer to the following Pizza Ontology when responding to questions:\n{ontology_context}")
            else:
                prompt_parts.append(
                    "Refer to your internal ontology knowledge when responding.")

        prompt_parts.append(
            "Provide a clear and concise answer to the following question.")
        if "Domain" in mode:
            prompt_parts.append(self.domain_constraint)

        prompt_parts.append("Avoid extra explanation or unrelated details")
        prompt_parts.append(f"Question: {question}")
        prompt_parts.append(self.answer_type)
        prompt_parts.append("Answer:")

        return "\n\n".join(prompt_parts)

    def getOntology(self, ontology_name,optional=False):
        if (optional):
            return ontology_name
        if ontology_name:
            from owlready2 import get_ontology, onto_path
            onto_path.append("./data/ontologies/")
            ontology = get_ontology(f"{ontology_name}.owl").load()
            return ontology
        else:
            return None
