# Data

## `ontologies/`

The five ontologies used in the evaluation:

| File | Ontology | Serialization |
|------|----------|---------------|
| `foaf.rdf` | FOAF vocabulary | RDF/XML |
| `wine.rdf` | W3C OWL Guide Wine ontology | RDF/XML |
| `pizza.owl` | Pizza ontology | OWL/XML |
| `Rientra.ttl` / `Rientra.rdf` | Rientra (disability profiles & occupational capabilities) | Turtle / RDF/XML — same content, two serializations |
| `go.owl` | Gene Ontology | OWL/XML |
| `food.rdf` | Food ontology (auxiliary, imported by Wine) | RDF/XML |
| `RTSS.ttl`, `Rientra_TBox_Schema.ttl` | TBox extracts of Rientra used by the Text-to-SPARQL pipeline | Turtle |

## `input/questions.json`

The 71 benchmark questions. Each entry has:

```json
{
  "QID": 16,
  "Question": "…natural-language question…",
  "Ontology": "Rientra.rdf",
  "SPARQL": "…reference query…",
  "FC_Ans": "…ground-truth short answer…"
}
```

* `Ontology` selects the ontology the question belongs to; for the Turtle
  runs the runner remaps `Rientra.rdf` questions to `Rientra.ttl`
  (same content, same ground truth).
* `FC_Ans` is the ground truth computed by executing `SPARQL` against the
  ontology. For yes/no questions it is `Yes`/`No`; for list questions it is
  the comma-separated set of local names.
* `60qas.json`, `elenco_nomi_pizze.txt` and the `*.txt` files are earlier
  drafts and working notes kept for provenance.
