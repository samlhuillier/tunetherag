import bitsandbytes
import json

# with open("sql-create-context-spider-intersect-validation-with-prompts.jsonl", "r") as f:
#     data = json.load(f)

# with open('sql-create-context-spider-intersect-validation-with-prompts.jsonl', 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         # print(data)

data = []
with open('sql-create-context-spider-intersect-validation-with-prompts.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))
# print(data_list[2])
# Step 2: Apply the function to each entry
# responses = map(your_function, data)
def generate_prompt(data_point):
#     full_prompt =f"""You are a powerful text-to-SQL model. Your job is to answer questions about a database. You are given a question and context regarding one or more tables. You must output the SQL query that answers the question.

# ### Input:
# {data_point["question"]}

# ### Context:
# {data_point["context"]}

# ### Response:
# """
#     print(full_prompt)
    return data_point['inference_prompt']
prompts = []
for data_point in data:
    # prompt = generate_prompt(data_point)
    # append prompt to prompts
    print("INBETWEEN POUNTS:")
    prompts.append(generate_prompt(data_point))
print(prompts)
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

base_model = "codellama/CodeLlama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
from peft import PeftModel
model = PeftModel.from_pretrained(model, "/home/sam/finetune-llm-for-rag/first-rag-codellama-7b/checkpoint-160")
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

tokenizer.pad_token = tokenizer.eos_token

batch_size = 16  # You can adjust the batch size based on your GPU capacity
outputs = []

model.eval()
with torch.no_grad(), open("first-finetune-of-rag-codellama7b.txt", "a") as f:
    for i in range(0, len(prompts), batch_size):
        print("i is: ", i)
        batch_inputs = prompts[i : i + batch_size]

        # Step 3: Loop over the batches and generate the outputs
        batch_model_inputs = tokenizer(
            batch_inputs, return_tensors="pt", padding=True, truncation=True
        )
        batch_model_inputs.to(model.device)
        input_lengths = [
            len(input_ids) for input_ids in batch_model_inputs["input_ids"]
        ]

        batch_outputs = model.generate(**batch_model_inputs, max_new_tokens=100)

        # Step 4: Collect the outputs
        batch_decoded_outputs = [
            tokenizer.decode(output[input_length:], skip_special_tokens=True)
            for output, input_length in zip(batch_outputs, input_lengths)
        ]
        outputs.extend(batch_decoded_outputs)
        # print(outputs)
        for b in batch_decoded_outputs:
            print('full b is: ', b)
            print("B IS: ", b.split('\n')[1])
            f.write(b.split('\n')[1] + "\n")