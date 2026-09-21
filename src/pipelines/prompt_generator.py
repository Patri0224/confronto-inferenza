import os


class PromptGenerator:
    def __init__(self, system_role=None, domain_constraint=None, answer_type=None):
        self.system_role = system_role or (
            "You are a specialized model for answering questions using ontology-based knowledge."
        )
        self.domain_constraint_pizza = domain_constraint or (
            "Focus strictly on pizza-related topics."
        )
        self.answer_type_pizza = answer_type or (
            "Provide your response using EXACTLY this format:\n"
            "[Short Answer: yes, no, or comma-separated list]\n"
            "---\n"
            "[Detailed explanation of inference steps]"
        )
        self.domain_constraint_rientra = domain_constraint or (
            "your job is to answer questions about the Rientra ontology instead of a reasoner, the SPARQL query is only for reference."
        )
        self.answer_type_rientra = answer_type or (
            "Task: Answer the question by extracting the exact class or individual names from the ontology text.\n"
            "Rules:\n"
            "- Output only the names (short identifiers), separated by commas.\n"
            "- Do NOT output generic placeholders like 'Job1', 'Job2', or 'none'.\n"
            "- Do NOT repeat the same word multiple times.\n"
            "- Look specifically for classes defined under rdfs:subClassOf or RDF rdf:about attributes."
        )

    def generate_prompt(self, question, sparql=None, ontology_context=None, mode="Q+Onto+Domain"):
        # Q, Q+Domain, Q+Onto+Domain
        system_parts = []
        user_parts = []

        onto_name = ontology_context if ontology_context else "N/A"
        is_pizza = (ontology_context == "pizza.owl")

        raw_ontology = self.getOntology(
            ontology_context) if ontology_context else None

        # -------------------------------------------------------------
        # SYSTEM PROMPT: Contesto ontologico, ruolo e vincoli globali
        # -------------------------------------------------------------
        if "Onto" in mode:
            system_parts.append(self.system_role)
            if raw_ontology:
                system_parts.append(
                    f"Refer to the following {onto_name} Ontology when responding to questions:\n{raw_ontology}"
                )
            else:
                system_parts.append(
                    "Refer to your internal ontology knowledge when responding.")

        if "Domain" in mode:
            domain_rule = self.domain_constraint_pizza if is_pizza else self.domain_constraint_rientra
            system_parts.append(domain_rule)

        # Regole di formato dell'output e vincoli generali
        format_rule = self.answer_type_pizza if is_pizza else self.answer_type_rientra
        system_parts.append(format_rule)
        system_parts.append("Avoid extra explanation or unrelated details.")

        # -------------------------------------------------------------
        # USER PROMPT: La domanda operativa specifica e input dinamico
        # -------------------------------------------------------------
        user_parts.append(
            "Provide a clear and concise answer to the following question.")
        user_parts.append(f"Question: {question}")

        if "Onto" in mode and sparql:
            user_parts.append(f"SPARQL query: {sparql}")

        user_parts.append("Answer:")

        system_prompt = "\n\n".join(system_parts)
        user_prompt = "\n\n".join(user_parts)

        return system_prompt, user_prompt

    def getOntology(self, ontology_name):
        onto_path = os.path.join("./data/ontologies/", f"{ontology_name}")
        if os.path.exists(onto_path):
            with open(onto_path, "r", encoding="utf-8") as f:
                return f.read()
        return ontology_name
