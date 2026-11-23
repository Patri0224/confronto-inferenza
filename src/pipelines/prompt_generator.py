import os


class PromptGenerator:
    def __init__(self, system_role=None, domain_constraint=None, answer_type=None, rag_a=None,rag_b=None):
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
        self.domain_constraint_foaf = domain_constraint or (
            "Focus strictly on the FOAF (Friend of a Friend) vocabulary: its classes, properties and axioms."
        )
        self.domain_constraint_go = domain_constraint or (
            "Focus strictly on the Gene Ontology (GO): biological process classes and their subclass hierarchy."
        )
        # GO identifica le classi con ID (GO_0008150): la risposta deve usare gli ID, non le etichette
        self.answer_type_go = answer_type or (
            "Provide your response using EXACTLY this format:\n"
            "[Short Answer: yes, no, or comma-separated list of GO identifiers]\n"
            "---\n"
            "[Detailed explanation of inference steps]\n"
            "If the question asks whether something is true (Is/Are/Does...), the short answer must be only "
            "'yes' or 'no'. If the question asks for classes, the short answer must be a comma-separated "
            "list of their GO identifiers (format GO_0000000), not their labels."
        )
        self.domain_constraint_wine = domain_constraint or (
            "Focus strictly on wine-related topics, as modelled in the W3C OWL Guide wine ontology."
        )
        # Formato generico (risposta breve + spiegazione) per ontologie con domande si/no e liste
        self.answer_type_short = answer_type or (
            "Provide your response using EXACTLY this format:\n"
            "[Short Answer: yes, no, or comma-separated list]\n"
            "---\n"
            "[Detailed explanation of inference steps]\n"
            "Use local names without namespace prefix in the short answer (e.g. Person, not foaf:Person)."
        )
        self.domain_constraint_rientra = domain_constraint or (
            "your job is to answer questions about the Rientra ontology instead of a reasoner, the SPARQL query is only for reference."
        )
        self.answer_type_rientra = answer_type or (
            "Task: Answer the question by extracting the exact class or individual names from the ontology text.\n"
            "Rules:\n"
            "- Output only the names (short identifiers), separated by commas.\n"
            "- If the question asks for several values per result (e.g. a job and its code, a person and a count),"
            " write each result as its values separated by a single space (e.g. 'Name 42'),"
            " and separate results with commas.\n"
            "- Do NOT output generic placeholders like 'Job1', 'Job2', or 'none'.\n"
            "- Do NOT repeat the same word multiple times.\n"
            "- Look specifically for classes defined under rdfs:subClassOf or RDF rdf:about attributes."
        )
        self.rag_a = rag_a
        self.rag_b = rag_b

    def generate_prompt(self, question, sparql=None, ontology_context=None, mode="Q+Onto+Domain"):
        # Q, Q+Domain, Q+Onto+Domain, RAG_a, RAG_b

        if mode == "RAG_a":
            mode = "Q+Domain+Onto+RAG_a"
        system_parts = []
        user_parts = []

        onto_name = ontology_context if ontology_context else "N/A"
        is_foaf = (ontology_context == "foaf.rdf")
        is_wine = (ontology_context == "wine.rdf")
        is_go = (ontology_context == "go.owl")
        is_pizza = (ontology_context == "pizza.owl")

        # L'ontologia intera si legge solo se serve (go.owl pesa 130 MB)
        needs_full_text = "Onto" in mode and "RAG_a" not in mode
        raw_ontology = self.getOntology(
            ontology_context) if ontology_context and needs_full_text else None

        # -------------------------------------------------------------
        # SYSTEM PROMPT: Contesto ontologico, ruolo e vincoli globali
        # -------------------------------------------------------------
        if "Onto" in mode:
            system_parts.append(self.system_role)
            if "RAG_a" in mode:
                if not self.rag_a:
                    raise ValueError(
                        "RAG is not initialized. Please provide a valid RAG instance.")
                retrieved_context = self.rag_a.retrieve_as_context(
                    query=question, top_k=35)
                system_parts.append(
                    f"Refer to the following retrieved context extracted using RAG from the ontology when responding to questions:\n{retrieved_context}"
                )
            elif raw_ontology:
                system_parts.append(
                    f"Refer to the following {onto_name} Ontology when responding to questions:\n{raw_ontology}"
                )
            else:
                system_parts.append(
                    "Refer to your internal ontology knowledge when responding.")

        if "Domain" in mode:
            if is_pizza:
                domain_rule = self.domain_constraint_pizza
            elif is_foaf:
                domain_rule = self.domain_constraint_foaf
            elif is_wine:
                domain_rule = self.domain_constraint_wine
            elif is_go:
                domain_rule = self.domain_constraint_go
            else:
                domain_rule = self.domain_constraint_rientra
            system_parts.append(domain_rule)

        # Regole di formato dell'output e vincoli generali
        if is_pizza:
            format_rule = self.answer_type_pizza
        elif is_go:
            format_rule = self.answer_type_go
        elif is_foaf or is_wine:
            format_rule = self.answer_type_short
        else:
            format_rule = self.answer_type_rientra
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
        base_dir = os.path.dirname(os.path.dirname(
            os.path.dirname(os.path.abspath(__file__))))
        onto_path = os.path.join(base_dir, "data", "ontologies", ontology_name)
        if os.path.exists(onto_path):
            with open(onto_path, "r", encoding="utf-8") as f:
                return f.read()
        return None
