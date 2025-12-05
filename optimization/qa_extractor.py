import json
import csv
from pathlib import Path

# --- IMPORTANT: SET THIS TO THE FOLDER CONTAINING THE TWO PQAL FILES ---
# e.g., if they are in your 'data' folder
INPUT_FOLDER = "../data/PQA-L"
OUTPUT_CSV = "rag_instruction_dataset.csv"


def extract_qa_pairs_pqal(input_dir, output_csv):
    qa_pairs = []

    # Target only the pqal files: pqal_train_dev_set.json and pqal_test_set.json
    pqal_files = list(Path(input_dir).glob("pqal_*.json"))

    if not pqal_files:
        print(
            f"🛑 Error: No 'pqal_' JSON files found in {input_dir}. Check your path.")
        return

    for file_path in pqal_files:
        print(f"Processing: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

            # Loop through each entry in the JSON dictionary
            for entry_id, entry_data in data.items():
                if 'QUESTION' in entry_data and 'LONG_ANSWER' in entry_data:
                    qa_pairs.append({
                        'question': entry_data['QUESTION'],
                        'answer': entry_data['LONG_ANSWER']
                    })

    # 3. Write to CSV
    if qa_pairs:
        # Create the optimization folder if it doesn't exist
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)

        with open(output_csv, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = ['question', 'answer']
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(qa_pairs)
        print(
            f"\n✅ Successfully extracted {len(qa_pairs)} QA pairs to {output_csv}")
    else:
        print("🛑 No QA pairs found.")

# Execute the script:
extract_qa_pairs_pqal(INPUT_FOLDER, OUTPUT_CSV)