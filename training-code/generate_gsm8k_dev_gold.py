import bitsandbytes
import json

# with open("sql-create-context-spider-intersect-validation-with-prompts.jsonl", "r") as f:
#     data = json.load(f)

# with open('sql-create-context-spider-intersect-validation-with-prompts.jsonl', 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         # print(data)

data = []
with open('/home/sam/finetune-llm-for-rag/datasets/gsm8k/gsm8k-test-with-1-examples-random-False-emb_fn-text-embedding-ada-002.jsonl', 'r') as f:
    for line in f:
        data.append(json.loads(line))
# print(data_list[2])
# Step 2: Apply the function to each entry
# responses = map(your_function, data)
def generate_prompt(data_point):
    print("inference prompt is: ", data_point['inference_prompt'])
    return data_point['inference_prompt']
prompts = []
for data_point in data:
    # prompt = generate_prompt(data_point)
    # append prompt to prompts
    # print("INBETWEEN POUNTS:")
    prompts.append(generate_prompt(data_point))
# print(prompts)
import torch
from transformers import AutoModelForCausalLM, BitsAndBytesConfig, AutoTokenizer

base_model = "meta-llama/Llama-2-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
from peft import PeftModel
model = PeftModel.from_pretrained(model, "/home/sam/finetune-llm-for-rag/training-code/gsm8k-1-example-llama2-7b/checkpoint-400")
tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

tokenizer.pad_token = tokenizer.eos_token

batch_size = 28  # You can adjust the batch size based on your GPU capacity
outputs = []
print("length of prompts: ", len(prompts))
model.eval()

with torch.no_grad(), open('gsm8k-one-example-llama2-7b-yesfinetune-checkpoint-400.txt', 'a') as f:
    for i in range(0, len(prompts), batch_size):
        print("i is: ", i)
        batch_inputs = prompts[i:i+batch_size]
        
        # Step 3: Loop over the batches and generate the outputs
        batch_model_inputs = tokenizer(batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=1024)
        batch_model_inputs.to(model.device)
        input_lengths = [len(input_ids) for input_ids in batch_model_inputs["input_ids"]]
        
        batch_outputs = model.generate(**batch_model_inputs, max_new_tokens=450)
        
        # Step 4: Collect the outputs
        batch_decoded_outputs = [tokenizer.decode(output[input_length:], skip_special_tokens=True) for output, input_length in zip(batch_outputs, input_lengths)]
        outputs.extend(batch_decoded_outputs)
        print(outputs)
        for b in batch_decoded_outputs:
            print("b out is: ", b.replace("\n", " ") + "\n")
            f.write(b.replace("\n", " ") + "\n")