import json
import os
import sys



BASE_DIR = os.path.dirname(os.path.dirname(
    os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.reasoner.reasoners import Reasoner

INPUT_PATH = os.path.join(BASE_DIR, "data", "input")
OUTPUT_PATH = os.path.join(BASE_DIR, "output", "responses")

ontology_paths = {
    "pizza.owl": os.path.join(BASE_DIR, "data", "ontologies", "pizza.owl"),
    "rientra.rdf": os.path.join(BASE_DIR, "data", "ontologies", "Rientra.rdf")
}
reasoner_instances = {}


def get_reasoner(ontology_name):
    if ontology_name in reasoner_instances:
        return reasoner_instances[ontology_name]
    else:
        ontology_path = ontology_paths.get(ontology_name)
        if ontology_path:
            reasoner_instances[ontology_name] = Reasoner(ontology_path)
            return reasoner_instances[ontology_name]
        else:
            raise ValueError(f"Ontology path for {ontology_name} not found.")


def generate_true_answers():
    with open(os.path.join(INPUT_PATH, "questions.json"), "r") as f:
        questions = json.load(f)

    for q in questions:
        reasoner = get_reasoner(q.get("Ontology"))
        SPARQL_query = q.get("SPARQL")
        if reasoner is None or SPARQL_query is None:
            print(f"Skipping question {q.get("QID")} due to missing reasoner or SPARQL query.")
            q["TrueAnswer"] = None
            continue
        true_answer = reasoner.evaluate_question(SPARQL_query)
        q["TrueAnswer"] = true_answer

    output_file_path = os.path.join(INPUT_PATH, "questionsTrue.json")
    with open(output_file_path, "w") as f:
        json.dump(questions, f, indent=4)

    print(f"True answers generated and saved to {output_file_path}")


if __name__ == "__main__":
    generate_true_answers()