from rdflib import Graph, RDF, OWL, RDFS

g = Graph()
g.parse("./data/ontologies/Rientra.ttl", format="turtle")

schema_graph = Graph()

# Copia i prefissi originali
for prefix, ns in g.namespaces():
    schema_graph.bind(prefix, ns)

# Estrai solo assiomi di classi e proprietà (TBox)
for s, p, o in g:
    if o in [OWL.Class, OWL.ObjectProperty, OWL.DatatypeProperty, RDF.Property]:
        schema_graph.add((s, p, o))
    elif p in [RDFS.subClassOf, RDFS.domain, RDFS.range, RDFS.subPropertyOf]:
        schema_graph.add((s, p, o))

schema_graph.serialize(destination="Rientra_TBox_Schema.ttl", format="turtle")
print(f"Schema estratto: {len(schema_graph)} triple (contro le {len(g)} originali).")