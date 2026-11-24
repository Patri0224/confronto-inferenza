import os
from rdflib import Graph

input_file = os.path.join("./data/ontologies/", "Rientra.rdf")          # RDF/XML source path
output_file = os.path.join("./data/ontologies/", "Rientra.ttl")         # Turtle destination

g = Graph()

print("Loading the RDF/XML graph...")
# Parse the original XML format
g.parse(input_file, format="xml")

print(f"Saving in Turtle format ({output_file})...")
# Serialize to Turtle (or format="nt" for N-Triples)
g.serialize(destination=output_file, format="turtle")

# Size comparison
size_orig = os.path.getsize(input_file) / (1024 * 1024)
size_new = os.path.getsize(output_file) / (1024 * 1024)
print(f"Done! Original size: {size_orig:.2f} MB -> Turtle size: {size_new:.2f} MB")


