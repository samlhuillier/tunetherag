def calculate_line_accuracy(line1, line2):
    # Take the length of the shorter line to avoid index out-of-bounds errors
    min_length = min(len(line1), len(line2))
    correct_characters = sum(line1[i] == line2[i] for i in range(min_length))
    
    # Denominator is the length of the longer line as that's the potential maximum number of correct characters
    accuracy = correct_characters / max(len(line1), len(line2))
    return accuracy

def calculate_average_accuracy(file1_path, file2_path):
    with open(file1_path, 'r') as file1, open(file2_path, 'r') as file2:
        lines1 = file1.readlines()
        lines2 = file2.readlines()

    # Ensure both files have the same number of lines; if not, raise an error
    if len(lines1) != len(lines2):
        raise ValueError("The two files have different number of lines!")

    total_accuracy = sum(calculate_line_accuracy(line1, line2) for line1, line2 in zip(lines1, lines2))
    average_accuracy = total_accuracy / len(lines1)

    return average_accuracy

if __name__ == "__main__":
    # Specify the paths for the two files you want to compare here
    file1_path = "/home/sam/finetune-llm-for-rag/datasets/viggo/meaning_representations.txt"
    file2_path = "/home/sam/finetune-llm-for-rag/training-code/viggo-zero-example-codellama7b-yesfinetune-checkpoint-480.txt"

    avg_accuracy = calculate_average_accuracy(file1_path, file2_path)
    print(f"Average line-by-line accuracy: {avg_accuracy:.2%}")

