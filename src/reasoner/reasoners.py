from owlready2 import *


class Reasoner:
    def __init__(self, ontology_path):
        if not ontology_path:
            raise ValueError("Ontology path cannot be None or empty.")
        self.ontology = get_ontology(ontology_path).load()
        self.reason()
        self.graph = default_world.as_rdflib_graph()

    def reason(self):
        with self.ontology:
            sync_reasoner_hermit(infer_property_values=True)

    def _extract_class_name(self, uri_str):
        if "#" in uri_str:
            return uri_str.split("#")[-1]
        elif ":" in uri_str:
            return uri_str.split(":")[-1]
        return uri_str

    def evaluate_question(self, sparql_query):
        if "ASK" in sparql_query.upper() and "subClassOf" in sparql_query:
            lines = [line.strip() for line in sparql_query.split("\n")
                     if "subClassOf" in line]
            if lines:
                parts = lines[0].split("rdfs:subClassOf")
                if len(parts) == 2:
                    sub_name = self._extract_class_name(parts[0].strip())
                    super_name = self._extract_class_name(
                        parts[1].replace(".", "").strip())

                    sub_cls = getattr(self.ontology, sub_name, None)
                    super_cls = getattr(self.ontology, super_name, None)

                    if sub_cls and super_cls:
                        # Controlla se super_cls fa parte degli antenati inferiti di sub_cls
                        is_subclass = super_cls in sub_cls.ancestors()
                        return is_subclass

        # --- CASO 2: QUERY ASK PER DISJOINTWITH ---
        if "ASK" in sparql_query.upper() and "disjointWith" in sparql_query:
            lines = [line.strip() for line in sparql_query.split("\n")
                     if "disjointWith" in line]
            if lines:
                parts = lines[0].split("owl:disjointWith")
                if len(parts) == 2:
                    cls1_name = self._extract_class_name(parts[0].strip())
                    cls2_name = self._extract_class_name(
                        parts[1].replace(".", "").strip())

                    cls1 = getattr(self.ontology, cls1_name, None)
                    cls2 = getattr(self.ontology, cls2_name, None)

                    if cls1 and cls2:
                        # Controlla se le due classi o i loro antenati sono disgiunti
                        disjoints = list(cls1.disjoint_with())
                        is_disjoint = cls2 in disjoints or any(
                            cls2 in ancestor.disjoint_with() for ancestor in cls1.ancestors())
                        return is_disjoint

        # --- CASO 3: QUERY SPARQL STANDARD (SELECT / Altre ASK) ---
        try:
            result = self.graph.query(sparql_query)

            if result.type == "ASK":
                return bool(result.askAnswer)

            elif result.type == "SELECT":
                rows = []
                for row in result:
                    # Ripuliamo i risultati lasciando solo i nomi semplici delle risorse
                    row_dict = {
                        str(var): str(value).split("#")[-1]
                        for var, value in row.asdict().items()
                    }
                    rows.append(row_dict)
                return rows

        except Exception as e:
            print(f"❌ Errore durante l'esecuzione della query SPARQL: {e}")
            return None
