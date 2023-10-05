from datasets import load_dataset

def write_meaning_representations_to_file():
    # Load the viggo dataset from the gem collection
    dataset = load_dataset("gem/viggo", split="test")

    # Extract the meaning_representation feature from each sample
    meaning_representations = [sample['meaning_representation'] for sample in dataset]

    # Write to a text file
    with open("meaning_representations.txt", "w") as file:
        for representation in meaning_representations:
            file.write(representation + "\n")

if __name__ == "__main__":
    write_meaning_representations_to_file()