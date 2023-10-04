#!/bin/bash

# Specify the directory where the files are located
directory="/home/sam/finetune-llm-for-rag/training-code/viggo-1-example-openai-embedding-ada-002"

# Loop from 10 to 100 in increments of 10
for i in {10..100..10}; do
    # Construct the filename
    filename="checkpoint-$i"

    # Check if the file exists in the directory
    if [ -e "$directory/$filename" ]; then
        # Delete the file
        rm -rf "$directory/$filename"
        echo "Deleted $filename"
    else
        echo "$filename does not exist"
    fi
done
