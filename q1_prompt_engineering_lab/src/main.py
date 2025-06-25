import json
import subprocess

model = "qwen2.5:3b"
input_file = "evaluation/input_queries.json"
prompt_files = {
    "zero_shot": "prompts/zero_shot.txt",
    "few_shot": "prompts/few_shot.txt",
    "cot": "prompts/cot_prompt.txt",
    "meta": "prompts/meta_prompt.txt"
}

with open(input_file) as f:
    queries = json.load(f)

results = []

for style, prompt_path in prompt_files.items():
    with open(prompt_path) as f:
        base_prompt = f.read()

    for query in queries:
        final_prompt = base_prompt.replace("Question:", f"Question: {query}")
        print(f"\n=== {style.upper()} | Query: {query} ===\n")
        
        result = subprocess.run(
            ["ollama", "run", model],
            input=final_prompt.encode(),
            capture_output=True
        )
        
        output = result.stdout.decode()
        prompt_result={
            "prompt_type": style,
            "query": query,
            "output": output.strip()
        }
        print(prompt_result)

        results.append(prompt_result)

# Save to output logs
with open("evaluation/output_logs.json", "w") as f:
    json.dump(results, f, indent=2)
