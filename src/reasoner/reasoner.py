from owlready2 import *

onto_path.append("./data/ontologies/")
ontology = get_ontology("pizza.owl").load()

#with ontology:
#    sync_reasoner()


named_pizza=ontology.NamedPizza

if named_pizza:
    pizze = list(named_pizza.subclasses())
    pizzearray=[]
    for pizza in pizze:
        pizzearray.append(pizza.name)
    print(pizzearray)