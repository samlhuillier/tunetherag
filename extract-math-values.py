def extract_values(filename):
    values = []

    with open(filename, 'r') as file:
        for line in file:
            if '####' in line:
                # Find the first occurrence of '####'
                index = line.find('####')
                # Extract the value after the space
                splitted = line[index+5:].split()
                value = splitted[0] if splitted else ""
                values.append(value)
            else:
                print("PATTERN NOT FOUND FOR LINE: ", line)
                values.append("")

    return values


def calculate_accuracy(list1, list2):
    # Determine the shorter length
    length = min(len(list1), len(list2))

    # Slice both lists to the size of the smaller list
    list1 = list1[:length]
    list2 = list2[:length]

    matching_elements = sum(1 for a, b in zip(list1, list2) if a == b)
    return matching_elements / length


ground_truth_values = extract_values('answers.txt')
predicted_values = extract_values('/home/sam/finetune-llm-for-rag/training-code/gsm8k-one-example-llama2-7b-yesfinetune-checkpoint-400.txt')
print(len(predicted_values))

accuracy = calculate_accuracy(ground_truth_values, predicted_values)
print(f'Accuracy: {accuracy:.2%}')