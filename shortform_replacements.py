"""
Hardcoded shortform to longform replacements for text preprocessing.
This module provides a fallback for catching shortforms that may be missed by LLM preprocessing.
"""

import re
from typing import Dict

# Dictionary of shortform to longform replacements
# Based on examples from the preprocessing system prompt
SHORTFORM_MAPPINGS: Dict[str, str] = {
    # Common internet/texting abbreviations
    "tbh": "to be honest",
    "idk": "I don't know",
    "rly": "really",
    "ppl": "people",
    "u": "you",
    "ur": "your",
    "r": "are",
    "n": "and",
    "w/": "with",
    "w/o": "without",
    "b/c": "because",
    "thru": "through",
    "tho": "though",
    "thx": "thanks",
    "pls": "please",
    "plz": "please",
    "btw": "by the way",
    "fyi": "for your information",
    "imo": "in my opinion",
    "imho": "in my humble opinion",
    "afaik": "as far as I know",
    "iirc": "if I recall correctly",
    "lol": "laugh out loud",
    "lmao": "laughing my a off",
    "rofl": "rolling on the floor laughing",
    "omg": "oh my god",
    "omfg": "oh my f god",
    "smh": "shaking my head",
    "nvm": "never mind",
    "jk": "just kidding",
    "brb": "be right back",
    "gtg": "got to go",
    "ttyl": "talk to you later",
    "irl": "in real life",
    "rn": "right now",
    "atm": "at the moment",
    "asap": "as soon as possible",
    "diy": "do it yourself",
    "tmi": "too much information",
    "nsfw": "not safe for work",

    # Reddit-specific abbreviations
    "aita": "am I the a",
    "AITA": "Am I The A",
    "ah": "a hole",
    "AH": "A Hole",
    "nta": "not the a",
    "NTA": "Not The A",
    "yta": "you're the a",
    "YTA": "You're The A",
    "esh": "everyone sucks here",
    "ESH": "Everyone Sucks Here",
    "nah": "no a holes here",
    "NAH": "No A Holes Here",
    "mil": "mother in law",
    "MIL": "Mother In Law",
    "fil": "father in law",
    "FIL": "Father In Law",
    "sil": "sister in law",
    "SIL": "Sister In Law",
    "bil": "brother in law",
    "BIL": "Brother In Law",
    "dh": "dear husband",
    "DH": "Dear Husband",
    "dw": "dear wife",
    "DW": "Dear Wife",
    "ds": "dear son",
    "DS": "Dear Son",
    "dd": "dear daughter",
    "DD": "Dear Daughter",
    "op": "original poster",
    "OP": "Original Poster",
    "eta": "edited to add",
    "ETA": "Edited To Add",
    "tldr": "too long didn't read",
    "TLDR": "Too Long Didn't Read",
    "tl;dr": "too long didn't read",
    "TL;DR": "Too Long Didn't Read",

    # Profanity censoring (first letter only)
    "wtf": "what the f",
    "WTF": "What The F",
    "stfu": "shut the f up",
    "STFU": "Shut The F Up",
    "af": "as f",
    "AF": "As F",

    # Common contractions that might need expansion for clarity
    "gonna": "going to",
    "wanna": "want to",
    "gotta": "got to",
    "shoulda": "should have",
    "coulda": "could have",
    "woulda": "would have",
    "kinda": "kind of",
    "sorta": "sort of",
    "lemme": "let me",
    "gimme": "give me",
    "dunno": "don't know",
    "ain't": "is not",

    # Numbers/symbols often used as words
    "2": "to",
    "4": "for",
    "b4": "before",
}


def replace_shortforms(text: str, case_sensitive: bool = False) -> str:
    """
    Replace shortforms with their longform equivalents.

    Args:
        text: The input text to process
        case_sensitive: Whether to perform case-sensitive replacements (default: False)

    Returns:
        Text with shortforms replaced by longforms
    """
    processed_text = text

    for shortform, longform in SHORTFORM_MAPPINGS.items():
        # Create a regex pattern that matches the shortform as a whole word
        # This prevents replacing parts of words (e.g., "run" in "running")
        if case_sensitive:
            pattern = r'\b' + re.escape(shortform) + r'\b'
            processed_text = re.sub(pattern, longform, processed_text)
        else:
            # Case-insensitive replacement while preserving the case of the first letter
            pattern = r'\b' + re.escape(shortform) + r'\b'

            def replace_match(match):
                original = match.group(0)
                # If the original starts with uppercase, capitalize the replacement
                if original[0].isupper():
                    return longform[0].upper() + longform[1:] if len(longform) > 1 else longform.upper()
                return longform

            processed_text = re.sub(pattern, replace_match, processed_text, flags=re.IGNORECASE)

    return processed_text


def post_process_text(text: str) -> str:
    """
    Post-process text after LLM preprocessing to catch any missed shortforms.

    This function serves as a safety net to ensure common abbreviations
    are expanded even if the LLM misses them.

    Args:
        text: The text that has already been processed by the LLM

    Returns:
        Text with additional shortform replacements applied
    """
    # Apply shortform replacements
    processed_text = replace_shortforms(text, case_sensitive=False)

    # Additional processing can be added here if needed
    # For example, ensuring proper spacing, removing double spaces, etc.
    processed_text = re.sub(r'\s+', ' ', processed_text)  # Replace multiple spaces with single space
    processed_text = processed_text.strip()  # Remove leading/trailing whitespace

    return processed_text


if __name__ == "__main__":
    # Test the function with some example text
    test_text = "tbh idk what ur talking about. AITA for thinking this? My SIL said wtf when I told her."
    print("Original text:")
    print(test_text)
    print("\nProcessed text:")
    print(post_process_text(test_text))