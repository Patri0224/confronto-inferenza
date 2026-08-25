from owlready2 import *

# Load your local pizza ontology file
onto = get_ontology("data/ontologies/pizza.owl").load()


with onto:
    sync_reasoner()
quattro_formaggi = onto.search_one(iri="*QuattroFormaggi")

# Print all classes (both asserted and inferred by HermiT) that QuattroFormaggi belongs to:
print("Inferred and asserted classes for QuattroFormaggi:")
for parent in quattro_formaggi.is_a:
    print(f"  - {parent}")
# Access classes safely via the ontology
margherita = onto.search_one(iri="*Margherita")
# Alternatively, if you know the exact name:
# margherita = onto.Margherita

if margherita:
    print(f"Found class: {margherita}")
    # Correct way to list direct subclasses in Owlready2
    print("Subclasses:")
    for sub in margherita.subclasses():
        print(f"  - {sub}")
else:
    print("Class Margherita not found.")

# Checking subclass or ancestor inheritance after reasoning
quattro_formaggi = onto.search_one(iri="*QuattroFormaggi")
vegetarian_pizza = onto.search_one(iri="*VegetarianPizza")

if quattro_formaggi and vegetarian_pizza:
    # Check if VegetarianPizza is in the ancestor tree of QuattroFormaggi
    is_vegetarian = vegetarian_pizza in quattro_formaggi.ancestors()
    print(
        f"\nQuattroFormaggi is a subclass/ancestor of VegetarianPizza: {is_vegetarian}")
