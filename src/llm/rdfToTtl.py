import os
from rdflib import Graph

input_file = os.path.join("./data/ontologies/", "Rientra.rdf")          # Percorso del file RDF/XML
output_file = os.path.join("./data/ontologies/", "Rientra.ttl")         # Destinazione Turtle

g = Graph()

print("Caricamento del grafo RDF/XML in corso...")
# Parsing del formato XML originale
g.parse(input_file, format="xml")

print(f"Salvataggio in formato Turtle ({output_file})...")
# Serializzazione in Turtle (oppure format="nt" per N-Triples)
g.serialize(destination=output_file, format="turtle")

# Confronto dimensioni
size_orig = os.path.getsize(input_file) / (1024 * 1024)
size_new = os.path.getsize(output_file) / (1024 * 1024)
print(f"Fatto! Dimensione originale: {size_orig:.2f} MB -> Dimensione Turtle: {size_new:.2f} MB")


