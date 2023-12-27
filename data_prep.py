# importing necessary libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

import math

import unicodedata
import string
import re

import torchtext
from torchtext.data.utils import get_tokenizer

from tqdm import tqdm

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def unicode_to_ascii(s: str) -> str:
    """
    Converts a Unicode string to its ASCII representation by removing diacritics
    and filtering out non-ASCII characters.

    Parameters:
    - s (str): The input Unicode string.

    Returns:
    - str: The ASCII representation of the input string.

    Example:
    >>> unicode_to_ascii("The Abbé Scarron.")
    'The Abbe Scarron.'
    """
    # Define a set of allowed characters (ASCII letters and common symbols)
    ALL_LETTERS = string.ascii_letters + " .,;'"

    # Normalizing the string by decomposing characters and removing diacritics
    normalized_string = unicodedata.normalize('NFD', s)

    # Filtering out characters that are not in the allowed set
    ascii_characters = [c for c in normalized_string
                        if unicodedata.category(c) != 'Mn' and c in ALL_LETTERS]

    # Join to get final string
    return ''.join(ascii_characters)


def read_file(file_dir: str) -> str:
    """
    Read the content of a file and replace newline characters with spaces.

    Args:
        file_dir (str): The directory of the file to be read.

    Returns:
        str: The content of the file with newline characters replaced by spaces.
    """
    with open(file_dir, 'r', encoding='utf-8') as file:
        data = file.read()
    data = data.replace("\n", " ")  # Replacing new line with space
    return data


def extract_sentences(ascii_data: str) -> list:
    """
    Extract sentences from a given ASCII text.

    Args:
        ascii_data (str): The input ASCII text.

    Returns:
        list: A list of cleaned sentences.
    """
    sentences = re.split(
        r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?)\s', ascii_data)
    """
    This regular expression is designed to split a text into sentences.
    It considers the following as sentence endings:
    1. Period (.)
    2. Question mark (?)
    It takes care to avoid splitting common abbreviations like "Mr." or "U.S.".
    """

    # Replacing consecutive whitespace characters with a single space
    cleaned_sentences = [
        re.sub(r'\s+', ' ', sentence).strip().lower() for sentence in sentences]

    # Excluding the sentence which does not contains any word
    cleaned_sentences = [i for i in cleaned_sentences if len(i.split()) >= 1]

    # Selecting only alphabets and numbers
    clean_data = [re.sub("[^a-z1-9' ]", "", i) for i in cleaned_sentences]

    return clean_data


def save_clean_data(save_dir: str, clean_data: list) -> None:
    """
    Save cleaned data to a file.

    Args:
        save_dir (str): The directory to save the cleaned data.
        clean_data (list): The list of cleaned data.

    Returns:
        None
    """
    with open(save_dir, "w") as file:  # Open a file in write mode
        for item in clean_data:  # Write each item in the list to a new line in the file
            file.write("%s\n" % item)


file_dir = "/content/drive/MyDrive/DL assignment/Auguste_Maquet.txt"
data = read_file(file_dir)
ascii_data = unicode_to_ascii(data)
clean_data = extract_sentences(ascii_data)
save_dir = "clean_data.txt"
save_clean_data(save_dir, clean_data)
