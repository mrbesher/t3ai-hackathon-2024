import argparse
import datetime
import json
import logging
import os
import re
from io import BytesIO
from urllib.parse import urljoin, urlparse

import google.generativeai as genai
import pytesseract
import requests
from bs4 import BeautifulSoup
from PIL import Image
from ratelimit import limits, sleep_and_retry
from tqdm import tqdm

# Set up logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

OPTION_MAPPING = {"a": 0, "b": 1, "c": 2, "d": 3, "e": 4}
CALLS = 15
RATE_LIMIT = 60


def setup_gemini():
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise ValueError("GEMINI_API_KEY environment variable is not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        "gemini-1.5-flash", generation_config={"response_mime_type": "application/json"}
    )


def get_question_links(url):
    try:
        # Send GET request to the page
        response = requests.get(url)
        response.raise_for_status()  # Raise an exception for HTTP errors
    except requests.exceptions.RequestException as err:
        logging.error(f"Request to {url} failed: {err}")
        return []

    # Parse the HTML
    soup = BeautifulSoup(response.text, "html.parser")
    # Find all links to question pages
    links = soup.find_all("a", class_="btn btn-primary")
    # Extract the href attribute (the link URL)
    return [link["href"] for link in links]


@sleep_and_retry
@limits(calls=CALLS, period=RATE_LIMIT)
def process_question(question_text):
    prompt = f"""{question_text}
---
The above text is OCRed from an image. Can you please fix the errors (choices not aligned with their letters, etc.) and rewrite the question and choices as json with question and option keys. "question" contains text as value and "choices" contain list of strings. Do not change wording. Write in Turkish. DO NOT write anything else. Remove any numbers or prefixes before the questions and choices."""

    try:
        response = MODEL.generate_content(prompt)
        response_text = response.text
        json_str = re.search(r"\{.*\}", response_text, re.DOTALL)
    except Exception as e:
        logging.warning(f"Skipping {question_text}. Cannot find valid response. {str(e)}")
        return None

    if json_str:
        try:
            data = json.loads(json_str.group())
            if (
                data.get("question")
                and data.get("choices")
                and isinstance(data["question"], str)
                and data["question"].strip()
                and isinstance(data["choices"], list)
                and len(data["choices"]) > 0
                and all(isinstance(opt, str) and opt.strip() for opt in data["choices"])
            ):
                return data
            else:
                logging.warning(f"Invalid data structure: {json_str.group()}")
        except json.JSONDecodeError:
            logging.warning(f"JSON decode error: {json_str.group()}")
    else:
        logging.warning(f"No JSON-like structure found in: {response.text}")

    return None


def remove_grey_watermark(img, threshold=150):
    """
    Removes grey watermark from an image.

    Args:
        img (PIL.Image): Input image.
        threshold (int): Threshold value for grey color.

    Returns:
        PIL.Image: Processed image.
    """
    if img.mode != "RGB":
        img = img.convert("RGB")
    pixels = img.load()
    for i in range(img.width):
        for j in range(img.height):
            r, g, b = pixels[i, j]
            if abs(r - g) < 10 and abs(g - b) < 10 and abs(b - r) < 10:
                if r > threshold and g > threshold and b > threshold:
                    pixels[i, j] = (255, 255, 255)
            elif not (r == 0 and g == 0 and b == 0):
                pixels[i, j] = (255, 255, 255)
    return img


def convert_question_json(question):
    question


def scrape_questions(url):
    """
    Scrapes questions from the given URL.

    Args:
        url (str): URL of the webpage to scrape.

    Returns:
        list: List of question and answer tuples.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.RequestException as err:
        logging.error(f"Request to {url} failed: {err}")
        return None

    soup = BeautifulSoup(response.text, "html.parser")

    questions = []
    for question_card in tqdm(
        soup.find_all("div", class_="card text-lg-center"), leave=False
    ):
        img_url = question_card.find("img")["data-src"]
        img_response = requests.get(img_url)
        if img_response.status_code == 200:
            img = Image.open(BytesIO(img_response.content))
            img_without_watermark = remove_grey_watermark(img)
            question = pytesseract.image_to_string(img_without_watermark, lang="tur")
        else:
            question = question_card.find("img")["alt"]

        answer = question_card["data-value"].strip().lower()

        answer_idx = OPTION_MAPPING.get(answer, None)

        if answer_idx is None:
            logging.warning(
                f"Skipping questionn as the option is not parsable: {answer}"
            )
            continue

        question_obj = process_question(question)

        if question_obj is None:
            continue

        question_obj["answer"] = answer_idx
        questions.append(question_obj)

    return questions


def save_to_jsonl(data, filename, mode="a"):
    with open(filename, mode, encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False)
        f.write("\n")


def main():
    parser = argparse.ArgumentParser(
        description="Scrape questions and answers from a given URL."
    )
    parser.add_argument("url", help="The URL to scrape")
    parser.add_argument(
        "--output",
        default=None,  # Let's set this later
        help="Output filename",
    )
    args = parser.parse_args()

    base_url = args.url
    endpoint = urlparse(base_url).path.strip('/')

    # Set the output filename if not provided
    default_filename = f"{endpoint}_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"
    filename = args.output or default_filename
    filename = f"{filename}.jsonl" if not filename.endswith(".jsonl") else filename

    # Validate URL
    parsed_url = urlparse(base_url)
    if not parsed_url.scheme or not parsed_url.netloc:
        logging.error("Invalid URL provided")
        return

    question_links = get_question_links(base_url)
    logging.info(f"Got {len(question_links)} question links")

    # Clear the file before starting
    open(filename, "w").close()

    for link in tqdm(question_links[23:]):
        # Create a full URL by joining the base URL with the relative link
        full_url = urljoin(base_url, link)
        questions_and_answers = scrape_questions(full_url)
        if questions_and_answers:
            logging.info(f"Scraped questions and answers from {full_url}")
            for qa in questions_and_answers:
                json_object = {
                    "question": qa["question"],
                    "choices": qa["choices"],
                    "answer": qa["answer"],
                    "source": full_url,
                }
                save_to_jsonl(json_object, filename)

    logging.info("All data saved to JSONL file")


if __name__ == "__main__":
    MODEL = setup_gemini()
    main()
