import ollama

def interroga_llm(prompt):
    print("Inviando la richiesta all'LLM locale...")
    
    # Chiamiamo il modello locale tramite l'API di Ollama
    risposta = ollama.chat(
        model='llama3.1', 
        messages=[
            {
                'role': 'user',
                'content': prompt,
            }
        ]
    )
    
    return risposta['message']['content']

if __name__ == "__main__":
    # Un semplice prompt di test logico-ontologico
    test_prompt = """
    Analizza questa affermazione:
    'Ogni cane è un mammifero. Fido è un cane.'
    Rispondi in modo sintetico: Fido è un mammifero? Spiega brevemente il perché usando la logica delle classi.
    """
    
    risultato = interroga_llm(test_prompt)
    print("\n--- Risposta dell'LLM ---")
    print(risultato)