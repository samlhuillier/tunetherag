from datasets import load_dataset

# Load the dataset
dataset = load_dataset('gsm8k', 'main', split='test')

# Specify the output text file path
output_file_path = 'answers.txt'

# Open the output file for writing
with open(output_file_path, 'w', encoding='utf-8') as output_file:
    # Loop through each example in the dataset
    for example in dataset:
        # Get the "answer" feature and replace newline characters with spaces
        answer = example['answer'].replace('\n', ' ')
        
        # Write the modified answer to the output file
        output_file.write(answer + '\n')

print(f'Answers have been written to {output_file_path}')
