import bitsandbytes
import json

# with open("sql-create-context-spider-intersect-validation-with-prompts.jsonl", "r") as f:
#     data = json.load(f)

# with open('sql-create-context-spider-intersect-validation-with-prompts.jsonl', 'r') as f:
#     for line in f:
#         data = json.loads(line)
#         # print(data)

data = []
with open('/home/sam/finetune-llm-for-rag/datasets/sql/MiniLM-L6/samlhuillier-sql-create-context-spider-intersect-validation-with-1-examples-random-False-emb_fn-default-emb-fn-different-db_id.jsonl', 'r') as f:
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

base_model = "codellama/CodeLlama-7b-hf"
model = AutoModelForCausalLM.from_pretrained(
    base_model,
    load_in_8bit=True,
    torch_dtype=torch.float16,
    device_map="auto",
)
from peft import PeftModel
model = PeftModel.from_pretrained(model, "/home/sam/finetune-llm-for-rag/training-code/1-example-new-fix-chroma-bug-tune-for-rag-different-db_id/checkpoint-210")
tokenizer = AutoTokenizer.from_pretrained("codellama/CodeLlama-7b-hf")

tokenizer.pad_token = tokenizer.eos_token

batch_size = 64  # You can adjust the batch size based on your GPU capacity
outputs = []

model.eval()
with torch.no_grad(), open("new-fix-chroma-bug-one-example-rag-diff-db_id-yes-finetune-codellama7B-checkpoint-210.txt", "a") as f:
    for i in range(0, len(prompts), batch_size):
        print("i is: ", i)
        batch_inputs = prompts[i : i + batch_size]

        # Step 3: Loop over the batches and generate the outputs
        batch_model_inputs = tokenizer(
            batch_inputs, return_tensors="pt", padding=True, truncation=True, max_length=700
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
        for i in range(len(batch_decoded_outputs)):
            print("input is: ", prompts[i])
            b = batch_decoded_outputs[i]
            print('full b is: ', b)
            print("B IS: ", b.split('\n')[1])
            f.write(b.split('\n')[1] + "\n")