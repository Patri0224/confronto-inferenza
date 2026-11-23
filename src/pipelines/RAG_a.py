import os
import chromadb
from rdflib import BNode, Graph, Literal, RDFS
from sentence_transformers import SentenceTransformer


class OntologyRAG:
    def __init__(
        self,
        ontology_path: str = "./data/ontologies/Rientra.ttl",
        model_name: str = "all-MiniLM-L6-v2",
        persist_dir: str = "./chroma_db",
        collection_name: str = "rientra_rag",
        label_aware: bool = False,
        predicates: set[str] | None = None,
    ):
        """
        label_aware: mostra anche le rdfs:label accanto agli ID ("cell cycle (GO_0007049)"),
            necessario per ontologie come GO dove l'ID da solo non è confrontabile con la domanda.
        predicates: se impostato, indicizza solo le triple con questi nomi locali di predicato
            (utile per ontologie enormi). Con None si indicizza tutto (comportamento originale).
        """
        self.ontology_path = ontology_path
        self.label_aware = label_aware
        self.predicates = predicates
        self.embedder = SentenceTransformer(model_name)
        
        # Inizializza ChromaDB persistente su disco
        self.client = chromadb.PersistentClient(path=persist_dir)
        self.collection = self.client.get_or_create_collection(name=collection_name)
        
        # Se la collezione è vuota, indicizza l'ontologia
        if self.collection.count() == 0:
            print(f"Collezione '{collection_name}' vuota. Avvio indicizzazione di {ontology_path}...")
            self._build_index()
        else:
            print(f"Collezione '{collection_name}' caricata da disco: {self.collection.count()} asserzioni presenti.")

    def _extract_local_name(self, uri_or_literal) -> str:
        val = str(uri_or_literal)
        if "#" in val:
            val = val.split("#")[-1]
        elif "/" in val:
            val = val.split("/")[-1]
        return val.strip()

    def _load_and_linearize_triples(self) -> list[str]:
        if not os.path.exists(self.ontology_path):
            raise FileNotFoundError(f"File ontologia non trovato: {self.ontology_path}")

        g = Graph()
        file_format = "turtle" if self.ontology_path.endswith((".ttl", ".turtle")) else "xml"
        g.parse(self.ontology_path, format=file_format)

        labels = {}
        if self.label_aware:
            for s, o in g.subject_objects(RDFS.label):
                labels.setdefault(s, str(o))

        def render(term):
            if isinstance(term, Literal):
                return str(term).strip()
            name = self._extract_local_name(term)
            if self.label_aware and not isinstance(term, Literal) and term in labels:
                return f"{labels[term]} ({name})"
            return name

        triples_set = set()
        for s, p, o in g:
            if self.predicates is not None:
                if self._extract_local_name(p) not in self.predicates:
                    continue
                if isinstance(s, BNode) or isinstance(o, BNode):
                    continue
            triples_set.add(f"{render(s)} {self._extract_local_name(p)} {render(o)}")

        return list(triples_set)

    def _build_index(self, batch_size: int = 1000):
        triples = self._load_and_linearize_triples()
        print(f"Estratte {len(triples)} asserzioni uniche. Calcolo embedding...")

        for i in range(0, len(triples), batch_size):
            batch = triples[i : i + batch_size]
            batch_embeddings = self.embedder.encode(batch, show_progress_bar=False).tolist()
            batch_ids = [f"triple_{i + j}" for j in range(len(batch))]

            self.collection.add(
                documents=batch,
                embeddings=batch_embeddings,
                ids=batch_ids
            )
            print(f"Indicizzate {min(i + batch_size, len(triples))}/{len(triples)} triple...")

        print("Indicizzazione completata con successo.")

    def retrieve(self, query: str, top_k: int = 15) -> list[str]:
        query_emb = self.embedder.encode([query]).tolist()
        results = self.collection.query(query_embeddings=query_emb, n_results=top_k)
        
        if results and results.get("documents"):
            return results["documents"][0]
        return []

    def retrieve_as_context(self, query: str, top_k: int = 15) -> str:
        triples = self.retrieve(query=query, top_k=top_k)
        return "\n".join(triples)

