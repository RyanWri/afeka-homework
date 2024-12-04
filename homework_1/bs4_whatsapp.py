from bs4 import BeautifulSoup
import requests
from tokenize_nltk import tokenize_nltk
from tokenize_spacy import tokenize_spacy
from analysis import compare_tokenizations
from lemmatization_nltk import lemmatize_nltk
from lemmatization_spacy import lemmatize_spacy
from analysis import compare_lemmatizations
from stem_nltk import stem_nltk
from stem_spacy import stem_spacy
from analysis import compare_stemmings
from comparison import find_message_with_stem_lemma_disparity
from comparison import find_message_with_lemma_stem_disparity
import re

# Regex pattern to match the WhatsApp message format
whatsapp_pattern = r"^\d{1,2}/\d{1,2}/\d{2}, \d{2}:\d{2} - .*?: (.*)"


# A function to get the published date/time of the article either by tag <time> or from the text itself.
def get_time(soup_cp):
    date_tag = soup_cp.find("time")
    if date_tag:
        date = date_tag.text.strip()
    else:
        paragraphs = soup_cp.find_all("p")
        date = "Date not found"
        for para in paragraphs:
            if "Published:" in para.text:
                date = para.text.split("Published:")[-1].strip()
                break
    print("Publication Date:", date)


def extract_messages_from_whatsapp_chat(file_path):
    # Read the WhatsApp chat file
    with open(file_path, "r", encoding="utf-8") as file:
        chat_data = file.readlines()

    chat_messages = []
    for line in chat_data:
        if "<" in line or ">" in line:
            continue
        # Match the pattern for WhatsApp messages
        match = re.match(whatsapp_pattern, line)
        if match:
            chat_messages.append(match.group(1))  # Extract the message content
    return chat_messages


# Whatsapp file path
whatsapp_file_path = "homework_1/data/whatsapp_group_chat.txt"

# URL of the webpage to scrape
url = "https://dotesports.com/league-of-legends/news/faker-wins-yet-another-award-to-add-to-his-trophy-cabinet"

# Fetch the webpage
response = requests.get(url)

article_text = ""

# Check if the request was successful
if response.status_code == 200:
    # Parse the HTML content
    soup = BeautifulSoup(response.text, "html.parser")

    # Extract the article's title
    title = soup.find("h1").text.strip()
    print("Title:", title)

    get_time(soup)
    paragraphs = soup.find_all("p")

    # Extract all paragraphs in the article
    article_text = "\n".join([para.text.strip() for para in paragraphs])
    print(
        "Article Text (First 594 characters):", article_text[:594]
    )  # I just didn't want the paragraph to be cut in the middle of the sentence.
else:
    print(f"Failed to fetch the page. Status code: {response.status_code}")

if article_text:
    print("\n********** Task 5 **********")
    # Apply Task 5
    # Tokenize using NLTK
    nltk_tokens = tokenize_nltk(article_text)
    print(f"Total NLTK tokens: {len(nltk_tokens)}")

    # Tokenize using spaCy
    spacy_tokens = tokenize_spacy(article_text)
    print(f"Total spaCy tokens: {len(spacy_tokens)}")

    # Compare results.
    compare_token_method_result = compare_tokenizations(nltk_tokens, spacy_tokens)
    for key, value in compare_token_method_result.items():
        print(f"{key}: {value}")

    print("\n********** Task 6 **********")
    # Apply Task 6
    # Lemmatize using NLTK
    nltk_lemmas = lemmatize_nltk(article_text)
    print(f"Total NLTK lemmas: {len(nltk_lemmas)}")

    # Lemmatize using spaCy
    spacy_lemmas = lemmatize_spacy(article_text)
    print(f"Total spaCy lemmas: {len(spacy_lemmas)}")

    # Compare lemmatizations
    lemma_diff = compare_lemmatizations(nltk_lemmas, spacy_lemmas)
    print("\nComparison of Lemmatizations:")
    for key, value in lemma_diff.items():
        print(f"{key}: {value}")

    print("\n********** Task 7 **********")
    # Stem using NLTK (Porter Stemmer)
    nltk_stems = stem_nltk(article_text, method="porter")
    print(f"Total NLTK (Porter) stems: {len(nltk_stems)}")

    # Stem using spaCy (approximation)
    spacy_stems = stem_spacy(article_text)
    print(f"Total spaCy stems: {len(spacy_stems)}")

    # Compare stemmings
    stem_diff = compare_stemmings(nltk_stems, spacy_stems)
    print("\nComparison of Stemmings:")
    for key, value in stem_diff.items():
        print(f"{key}: {value}")

    print("\n********** Task 8 **********")
    messages = article_text.split("\n")  # Splits by newlines
    messages = [line.strip() for line in messages if line.strip()]  # Remove empty lines
    result_message = find_message_with_stem_lemma_disparity(
        messages=messages, nltk_stems=nltk_stems, nltk_lemmas=nltk_lemmas
    )

    if result_message:
        print("Message causing the effect:\n")
        print(
            repr(result_message)
        )  # Ensures the entire message, including multiline, is printed
    else:
        print("No message found that meets the criteria.")

    print("\n********** Task 9 **********")
    # Example usage
    result_message = find_message_with_lemma_stem_disparity(
        messages=messages, nltk_stems=nltk_stems, nltk_lemmas=nltk_lemmas
    )

    if result_message:
        print("Message causing the effect:\n")
        print(
            repr(result_message)
        )  # Ensures the entire message, including multiline, is printed
    else:
        print("No message found that meets the criteria.")

    extracted_whatsapp_messages = extract_messages_from_whatsapp_chat(
        whatsapp_file_path
    )

    if len(extracted_whatsapp_messages) != 0:
        print("\n********** Task 5 **********")
        # Apply Task 5
        # Tokenize using NLTK
        nltk_tokens = tokenize_nltk(extracted_whatsapp_messages)
        print(f"Total NLTK tokens: {len(nltk_tokens)}")

        # Tokenize using spaCy
        spacy_tokens = tokenize_spacy(extracted_whatsapp_messages)
        print(f"Total spaCy tokens: {len(spacy_tokens)}")

        # Compare results.
        compare_token_method_result = compare_tokenizations(nltk_tokens, spacy_tokens)
        for key, value in compare_token_method_result.items():
            print(f"{key}: {value}")

        print("\n********** Task 6 **********")
        # Apply Task 6
        # Lemmatize using NLTK
        nltk_lemmas = lemmatize_nltk(extracted_whatsapp_messages)
        print(f"Total NLTK lemmas: {len(nltk_lemmas)}")

        # Lemmatize using spaCy
        spacy_lemmas = lemmatize_spacy(extracted_whatsapp_messages)
        print(f"Total spaCy lemmas: {len(spacy_lemmas)}")

        # Compare lemmatizations
        lemma_diff = compare_lemmatizations(nltk_lemmas, spacy_lemmas)
        print("\nComparison of Lemmatizations:")
        for key, value in lemma_diff.items():
            print(f"{key}: {value}")

        print("\n********** Task 7 **********")
        # Stem using NLTK (Porter Stemmer)
        nltk_stems = stem_nltk(extracted_whatsapp_messages, method="porter")
        print(f"Total NLTK (Porter) stems: {len(nltk_stems)}")

        # Stem using spaCy (approximation)
        spacy_stems = stem_spacy(extracted_whatsapp_messages)
        print(f"Total spaCy stems: {len(spacy_stems)}")

        # Compare stemmings
        stem_diff = compare_stemmings(nltk_stems, spacy_stems)
        print("\nComparison of Stemmings:")
        for key, value in stem_diff.items():
            print(f"{key}: {value}")

        print("\n********** Task 8 **********")
        result_message = find_message_with_stem_lemma_disparity(
            messages=extracted_whatsapp_messages,
            nltk_stems=nltk_stems,
            nltk_lemmas=nltk_lemmas,
        )

        if result_message:
            print("Message causing the effect:\n")
            print(
                repr(result_message)
            )  # Ensures the entire message, including multiline, is printed
        else:
            print("No message found that meets the criteria.")

        print("\n********** Task 9 **********")
        # Example usage
        result_message = find_message_with_lemma_stem_disparity(
            messages=extracted_whatsapp_messages,
            nltk_stems=nltk_stems,
            nltk_lemmas=nltk_lemmas,
        )

        if result_message:
            print("Message causing the effect:\n")
            print(
                repr(result_message)
            )  # Ensures the entire message, including multiline, is printed
        else:
            print("No message found that meets the criteria.")
