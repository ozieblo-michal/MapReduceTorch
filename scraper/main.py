import os
import time
from typing import NoReturn
import ebooklib
from ebooklib import epub
import fitz  # PyMuPDF

from ebooklib import epub
from bs4 import BeautifulSoup


def read_epub(file_path: str) -> str:
    """
    Reads the text from an EPUB file using BeautifulSoup to extract text without HTML tags.
    Additionally, prints the book's title and author if available.

    Parameters:
    - file_path (str): The path to the EPUB file.

    Returns:
    - str: The extracted text content from the EPUB file.
    """
    book = epub.read_epub(file_path)
    text = ""

    title = book.get_metadata("DC", "title")
    author = book.get_metadata("DC", "creator")

    if title:
        print(f"Title: {title[0][0]}")
    if author:
        print(f"Author: {author[0][0]}")

    for item in book.get_items():
        if item.get_type() == ebooklib.ITEM_DOCUMENT:
            soup = BeautifulSoup(item.content, "html.parser")
            text += soup.get_text(separator=" ", strip=True) + "\n\n"
    return text


def read_pdf(file_path: str) -> str:
    """
    Reads the text from a PDF file.

    Parameters:
    - file_path (str): The path to the PDF file.

    Returns:
    - str: The extracted text content from the PDF file.
    """
    doc = fitz.open(file_path)
    text = ""
    for page in doc:
        text += page.get_text()
    return text


def save_text_to_file(text: str, output_file_path: str) -> None:
    """
    Saves the given text to a file, placing each sentence on a new line.

    Parameters:
    - text (str): The text to save.
    - output_file_path (str): The path to the output file where the text will be saved.
    """
    formatted_text = text.replace(". ", ".\n")
    with open(output_file_path, "w", encoding="utf-8") as file:
        file.write(formatted_text)


def convert_to_txt(input_file_path: str, output_file_path: str) -> None:
    """
    Converts an EPUB or PDF file to text and saves it to a specified file.

    Parameters:
    - input_file_path (str): The path to the input EPUB or PDF file.
    - output_file_path (str): The path to the output text file.

    Raises:
    - ValueError: If the input file format is unsupported.
    """
    start_time = time.time()
    file_extension = os.path.splitext(input_file_path)[1].lower()
    if file_extension == ".epub":
        text = read_epub(input_file_path)
    elif file_extension == ".pdf":
        text = read_pdf(input_file_path)
    else:
        raise ValueError("Unsupported file format.")
    save_text_to_file(text, output_file_path)
    end_time = time.time()
    print(f"File processing time: {end_time - start_time} seconds")


if __name__ == "__main__":
    input_file_path = input("Enter the path to the input file: ")
    output_file_path = input("Enter the path to the output file (TXT format): ")
    convert_to_txt(input_file_path, output_file_path)
