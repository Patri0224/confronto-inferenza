import os
import chromadb
from rdflib import Graph
from sentence_transformers import SentenceTransformer


class OntologyRAG:
    def __init__(
        self,
        ontology_path: str = "./data/ontologies/Rientra.ttl",
        model_name: str = "all-MiniLM-L6-v2",
        persist_dir: str = "./chroma_db",
        collection_name: str = "rientra_rag"
    ):
        self.ontology_path = ontology_path
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

        triples_set = set()
        for s, p, o in g:
            sub = self._extract_local_name(s)
            pred = self._extract_local_name(p)
            obj = self._extract_local_name(o)
            triples_set.add(f"{sub} {pred} {obj}")

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

