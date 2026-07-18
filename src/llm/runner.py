import json
import os
import sys
import ollama
from time import time
from datetime import datetime



BASE_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

from src.pipelines.prompt_generator import PromptGenerator

INPUT_PATH=os.path.join(BASE_DIR,"data","input")
OUTPUT_PATH=os.path.join(BASE_DIR,"output","responses")


def benchmark(model_name,prompt_mode="Q+Onto+Domain"):
    print(f"Running benchmark for model: {model_name} with prompt mode: {prompt_mode}")
    with open(os.path.join(INPUT_PATH,"questions.json"),"r") as f:
        questions_data=json.load(f)

    questions=questions_data.get("questions",[])

    generator=PromptGenerator()
    results={
        "model":model_name,
        "prompt_mode":prompt_mode,
        "timestamp":datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "responses":[]
    }


    for q in questions:
        question_text=q.get("question")
        ontology_context=q.get("ontology_context")
        prompt=generator.generate_prompt(question_text,ontology_context,prompt_mode)


        start_time=time()
        try:
            response=ollama.chat(
                model=model_name,
                messages=[{"role":"user","content":prompt}],
                options={"temperature":0.0}
            )
            answer=response['message']['content'].strip()
            time_taken=round(time()-start_time, 2)

        except Exception as e:
            answer=f"Error: {str(e)}"
            time_taken=None

        results["responses"].append({
            "question":question_text,
            "ontology_context":ontology_context,
            "prompt":prompt,
            "response_time":time_taken,
            "answer":answer,
            #"correct_answer":q.get("correct_answer"),
            #"value":Scorer.score(reference_answer=q.get("correct_answer"), generated_answer=answer)
        })
        print(results["responses"][-1])
        
    safe_model_name = model_name.replace(':', '_').replace('/', '_')
    name_file_output = f"{safe_model_name}_{prompt_mode}.json"
    percorso_salvataggio = os.path.join(OUTPUT_PATH, name_file_output)
    with open(percorso_salvataggio,"w") as f:
        json.dump(results,f,indent=4)


if __name__ == "__main__":
    print(BASE_DIR)
    benchmark(model_name="llama2:7b", prompt_mode="Q+Domain")