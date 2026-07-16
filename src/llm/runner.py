import json
import os
import ollama
from time import time
from datetime import datetime

from src.evaluation.scorer import Scorer
from src.pipelines.prompt_generator import PromptGenerator


BASE_DIR=os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
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
            answer=response['messages']['content'].strip()
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
        
        
        name_file_output=f"{model_name.replace('/','_')}_{prompt_mode}.json"
        with open(os.path.join(OUTPUT_PATH,name_file_output),"w") as f:
            json.dump(results,f,indent=4)


if __name__ == "__main__":
    # Puoi cambiare il modello qui per testare i vari Llama o Qwen scaricati
    benchmark(model_name="llama2:7b", prompt_mode="Q+Domain")