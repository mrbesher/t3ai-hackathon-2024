import json
import glob
from pathlib import Path
import re
from typing import Dict, List
import logging

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')

def load_jsonl(file_path: str) -> List[Dict]:
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return [json.loads(line) for line in f]
    except json.JSONDecodeError as e:
        logging.error(f"Error decoding JSON in file {file_path}: {e}")
        return []

def save_jsonl(data: List[Dict], file_path: str) -> None:
    with open(file_path, 'w', encoding='utf-8') as f:
        for item in data:
            json.dump(item, f, ensure_ascii=False)
            f.write('\n')

def convert_to_openai_format(item: Dict) -> Dict:
    choices = item['choices']
    
    if len(choices) != 5:
        logging.warning(f"Skipping item with invalid number of choices: {len(choices)}")
        return None
    
    for choice in choices:
        choice = choice.strip()
        if len(choice) < 3:
            return None
        if choice and choice[0].isalpha() and choice[1] == ')':
            logging.warning(f"Skipping item with malformed choice: {choice}")
            return None

    # Remove extra whitespaces and multiple dashes from the question
    question = re.sub(r'\s{2,}', ' ', item['question'])  # Remove multiple spaces
    question = re.sub(r'-{3,}', '--', question)          # Replace more than 2 dashes with exactly 2 dashes

    # Check if the question is too short after formatting
    if len(question.strip()) < 15:  # You can adjust the length condition
        logging.warning(f"Skipping item due to short question length: {len(question.strip())}")
        return None

    question = f"{question.strip()}\n" + '\n'.join([f"{chr(65+i)}) {choice}" for i, choice in enumerate(choices)])
    answer = choices[item['answer']]
    
    return {
        "messages": [
            {"role": "user", "content": question},
            {"role": "assistant", "content": f"{chr(65+item['answer'])}) {answer}"}
        ]
    }

def main(input_folder: str = '.', output_file: str = 'aof_combined.jsonl'):
    input_files = glob.glob(str(Path(input_folder) / '*.jsonl'))
    processed_questions = set()
    combined_data = []

    for file in input_files:
        try:
            data = load_jsonl(file)
            for item in data:
                if item['question'] not in processed_questions:
                    processed_questions.add(item['question'])
                    converted_item = convert_to_openai_format(item)
                    if converted_item:
                        combined_data.append(converted_item)
        except KeyError as e:
            logging.error(f"Missing key in file {file}: {e}")
        except Exception as e:
            logging.error(f"Unexpected error processing file {file}: {e}")

    save_jsonl(combined_data, output_file)
    logging.info(f"Combined dataset saved to {output_file}")
    logging.info(f"Total unique questions processed: {len(combined_data)}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('-i', '--input_folder', default='.')
    parser.add_argument('-o', '--output_file', default='aof_combined.jsonl')
    args = parser.parse_args()
    main(args.input_folder, args.output_file)
