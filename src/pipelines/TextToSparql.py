import json
import re
from matplotlib import text
from rdflib import Graph
import litellm


class TextToSparqlEngine:
    def __init__(self, ontology_path: str = "./data/ontologies/Rientra.ttl"):
        self.ontology_path = ontology_path
        self.graph = Graph()
        print(f"Caricamento ontologia in RDFLib da: {ontology_path}...")
        file_format = "turtle" if ontology_path.endswith(
            (".ttl", ".turtle")) else "xml"
        self.graph.parse(ontology_path, format=file_format)
        print(f"Ontologia caricata con successo ({len(self.graph)} triple).")

    def generate_sparql_prompt(self, question: str) -> tuple[str, str]:
        #Rientra_TBox_Schema.ttl
        #RTSS.ttl
        TBox_Schema_path = "./data/ontologies/RTSS.ttl"
        TBox_Schema_content = ""
        try:
            with open(TBox_Schema_path, "r", encoding="utf-8") as f:
                TBox_Schema_content = f.read()
        except FileNotFoundError:
            print(
                f"[Warning]: Il file {TBox_Schema_path} non è stato trovato. Assicurati che sia presente nella directory corrente.")
            TBox_Schema_content = "Schema non disponibile."
        system_prompt = (
            "You are an expert SPARQL query generator for the Rientra OWL ontology.\n"
            "Your task is to translate the user question into a valid SPARQL query using EXCLUSIVELY the provided vocabulary and exact prefix URIs.\n\n"
            + TBox_Schema_content + "\n\n"
            "RULES:\n"
            "1. Output ONLY the query inside a ```sparql ... ``` block.\n"
            "2. Do NOT add filters, variables, or joins not strictly requested by the question.\n"
            "3. Never invent namespaces like example.org."
        )

        user_prompt = f"Question: {question}\nSPARQL Query:"
        return system_prompt, user_prompt

    def extract_sparql_query(self, raw_llm_response: str) -> str:
        """Estrae la query ripulita dai blocchi di codice markdown."""
        match = re.search(r"```(?:sparql)?(.*?)```",
                          raw_llm_response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        
        text = re.sub(r"```(?:sparql)?|```", "", text, flags=re.IGNORECASE)

        keyword_match = re.search(
            r"\b(PREFIX|SELECT|ASK|CONSTRUCT|DESCRIBE)\b", text, re.IGNORECASE)
        if keyword_match:
            text = text[keyword_match.start():]

        last_brace_idx = text.rfind("}")
        if last_brace_idx != -1:
            text = text[:last_brace_idx + 1]

        return text.strip()

    def execute_query(self, sparql_query: str) -> list[str]:
        """Esegue la query generata su RDFLib ed estrae i risultati come nomi locali."""
        try:
            qres = self.graph.query(sparql_query)
            extracted_items = []

            for row in qres:
                # Se la query restituisce una sola variabile o più variabili per riga
                row_items = []
                for val in row:
                    val_str = str(val)
                    # Pulizia degli URI per ottenere il nome locale
                    local_name = val_str.split(
                        "#")[-1].split("/")[-1].split(":")[-1]
                    row_items.append(local_name)

                extracted_items.append(" ".join(row_items))

            return extracted_items
        except Exception as e:
            print(
                f"[RDFLib Error]: Impossibile eseguire la query. Dettaglio: {e}")
            return []

    def run_text_to_sparql(self, question: str, model: str = "ollama/llama3.1") -> dict:
        """Esegue l'intero flusso: Prompt -> LLM -> Estrazione -> Esecuzione su grafo."""
        sys_p, usr_p = self.generate_sparql_prompt(question)

        # Chiamata tramite LiteLLM all'istanza locale di Ollama
        response = litellm.completion(
            model=model,
            messages=[
                {"role": "system", "content": sys_p},
                {"role": "user", "content": usr_p}
            ],
            temperature=0.0,
            max_tokens=300
        )

        raw_output = response.choices[0].message.content
        sparql_query = self.extract_sparql_query(raw_output)
        results = self.execute_query(sparql_query)
        usage = response.get("usage", {})
        prompt_tokens = usage.get("prompt_tokens", 0)
        completion_tokens = usage.get("completion_tokens", 0)
        total_tokens = usage.get("total_tokens", 0)
        response_cost = response._hidden_params.get(
            "response_cost", 0.0) if hasattr(response, "_hidden_params") else 0.0

        return {
            "generated_sparql": sparql_query,
            "raw_output": raw_output,
            "execution_results": results,
            "short_answer": ", ".join(results),
            "metrics": {
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "cost": response_cost
            }
        }


if __name__ == "__main__":
    # Esempio di utilizzo
    engine = TextToSparqlEngine(
        ontology_path="./data/ontologies/Rientra.rdf")
    with open("./data/input/questions.json", "r") as f:
        questions = json.load(f)
    results_list = []
    for q in questions:

        ontology_context = q.get("Ontology")
        if ontology_context != "Rientra.rdf":
            continue
        query = q.get("SPARQL")
        results = engine.execute_query(query)
        answer = ", ".join(results)
        results_list.append({
            "QID": q.get("QID"),
            "Answer": answer
        })

    # Salvataggio dei risultati in un file JSON
    with open("./output/results.json", "w", encoding="utf-8") as f:
        json.dump(results_list, f, ensure_ascii=False, indent=4)
