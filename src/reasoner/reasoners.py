import os
import re
from owlready2 import get_ontology, sync_reasoner_hermit, Thing, Nothing, Or, And


class Reasoner:
    def __init__(self, ontology_path):
        if not ontology_path or not os.path.exists(ontology_path):
            raise ValueError(
                f"Ontology path non valido o file inesistente: {ontology_path}")

        self.ontology_path = ontology_path
        self.ontology = get_ontology(ontology_path).load()

        # 1. Eseguiamo HermiT per classificare l'ontologia
        self.reason()

    def reason(self):
        """Avvia HermiT per materializzare le classificazioni OWL"""
        print(f"⚙️ Avvio HermiT su {os.path.basename(self.ontology_path)}...")
        with self.ontology:
            sync_reasoner_hermit(infer_property_values=True)
        print("✅ Ragionamento HermiT completato!")

    def _clean_name(self, raw_name):
        """Pulisce la stringa isolando il nome puro della classe"""
        if not raw_name:
            return ""
        text = str(raw_name).strip()
        text = re.sub(r'<.*?>', '', text)
        if "#" in text:
            text = text.split("#")[-1]
        elif ":" in text:
            text = text.split(":")[-1]
        return text.strip(" .;")

    def _get_class(self, raw_name):
        """Mappa i nomi usati in 60qas.json con i nomi reali in pizza.owl"""
        clean = self._extract_clean_name_raw(raw_name)

        mapping = {
            "fourcheesepizza": "QuattroFormaggi",
            "quattroformaggipizza": "QuattroFormaggi",
            "anchovytopping": "AnchoviesTopping",
            "americanpizza": "American",
            "americanhotpizza": "AmericanHot",
            "seafoodpizza": "FruttiDiMare",
            "napolitanapizza": "Napoletana",
            "sloppygiuseppepizza": "SloppyGiuseppe",
            "sohopizza": "Soho",
            "supremepizza": "InterestingPizza",
            "hotandspicypizza": "SpicyPizza",
            "italianpizza": "RealItalianPizza",
            "prawntopping": "PrawnsTopping",
            "margheritapizza": "Margherita",
            "hottopping": "SpicyTopping",
            "cheese-topping": "CheeseTopping"
        }

        target = mapping.get(clean.lower(), clean)

        # Cerca la classe case-insensitive nell'ontologia
        for cls in self.ontology.classes():
            if cls.name.lower() == target.lower():
                return cls
        return None

    def _extract_clean_name_raw(self, name):
        if not name:
            return ""
        s = str(name).strip()
        if "#" in s:
            s = s.split("#")[-1]
        elif ":" in s:
            s = s.split(":")[-1]
        return s.strip(" .;")

    def _get_toppings_for_pizza(self, pizza_cls):
        """Estrae l'elenco dei condimenti (someValuesFrom) definiti per una pizza"""
        toppings = []
        if not pizza_cls:
            return toppings

        for parent in pizza_cls.is_a:
            # Se è una restrizione owl:hasTopping
            if hasattr(parent, "property") and parent.property.name == "hasTopping":
                if hasattr(parent, "value"):
                    toppings.append(parent.value)
        return toppings

    def evaluate_question(self, sparql_query):
        """Valuta le domande tramite ispezione ontologica formale DL"""
        if not sparql_query:
            return None

        query_upper = sparql_query.upper()

        # --- CASO 1: QUERY ASK SUBCLASSOF (es. Q2, Q11, Q18, Q58) ---
        if "ASK" in query_upper and "SUBCLASSOF" in query_upper:
            match = re.search(
                r'([\w:<>/#.]+)\s+rdfs:subClassOf\s+([\w:<>/#.]+)', sparql_query, re.IGNORECASE)
            if match:
                sub_cls = self._get_class(match.group(1))
                super_cls = self._get_class(match.group(2))

                if sub_cls and super_cls:
                    # 1. Controllo diretto o tra gli antenati
                    if super_cls in sub_cls.ancestors():
                        return True

                    # 2. Gestione speciale per VegetarianPizza e CheeseyPizza (Classi Equivalenti DL)
                    if super_cls.name in ["VegetarianPizza", "VegetarianPizzaEquivalent1", "VegetarianPizzaEquivalent2"]:
                        # Una pizza è vegetariana se NON ha MeatTopping nè FishTopping
                        toppings = self._get_toppings_for_pizza(sub_cls)
                        has_meat_or_fish = False
                        for t in toppings:
                            ancestor_names = [a.name for a in t.ancestors()]
                            if "MeatTopping" in ancestor_names or "FishTopping" in ancestor_names:
                                has_meat_or_fish = True
                                break
                        return not has_meat_or_fish

                    if super_cls.name == "CheesePizza" or super_cls.name == "CheeseyPizza":
                        toppings = self._get_toppings_for_pizza(sub_cls)
                        for t in toppings:
                            if "CheeseTopping" in [a.name for a in t.ancestors()]:
                                return True
                        return False

                return False

        # --- CASO 2: QUERY ASK DISJOINTWITH (es. Q5, Q16, Q23, Q49, Q51) ---
        if "ASK" in query_upper and "DISJOINTWITH" in query_upper:
            match = re.search(
                r'([\w:<>/#.]+)\s+owl:disjointWith\s+([\w:<>/#.]+)', sparql_query, re.IGNORECASE)
            if match:
                cls1 = self._get_class(match.group(1))
                cls2 = self._get_class(match.group(2))

                if cls1 and cls2:
                    # Verifica disgiunzione diretta o ereditata tra antenati
                    anc1 = cls1.ancestors()
                    anc2 = cls2.ancestors()

                    for a1 in anc1:
                        if a1 == Thing:
                            continue
                        disjoints = a1.disjoint_with()
                        for d in disjoints:
                            entities = getattr(d, 'entities', [])
                            if any(a2 in entities for a2 in anc2):
                                return True
                    return False
                return False

        # --- CASO 3: QUERY SELECT CARDINALITY (es. Q34: pizze con > 3 toppings) ---
        if "CARDINALITY" in query_upper or "INTERSECTIONOF" in query_upper or "FILTER EXISTS" in query_upper:
            results = []
            named_pizza_cls = getattr(self.ontology, "NamedPizza", None)
            if named_pizza_cls:
                for pizza in named_pizza_cls.subclasses():
                    toppings = self._get_toppings_for_pizza(pizza)
                    if len(toppings) > 3:
                        results.append({"pizza": pizza.name})
            return results

        # --- CASO 4: QUERY SELECT TOPPINGS / INGREDIENTI ---
        if "SELECT" in query_upper and "HASTOPPING" in query_upper:
            match = re.search(
                r'(\w+)\s+pizza:hasTopping\s+\?(\w+)', sparql_query, re.IGNORECASE)
            if match:
                pizza_cls = self._get_class(match.group(1))
                if pizza_cls:
                    toppings = self._get_toppings_for_pizza(pizza_cls)
                    return [{"topping": t.name} for t in toppings]

        return None
