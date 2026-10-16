import re


def clean_text(text: str) -> str:
    """
    Cleans raw review text for sentiment classification.

    Steps:
    1. Replace HTML tags with a space (prevents glued words like 'end<br />then' -> 'end then').
    2. Lowercase text.
    3. Strip apostrophes (ASCII ' and unicode ’) so contractions stay intact as single tokens
       (e.g., "didn't" -> "didnt", "can't" -> "cant", "it's" -> "its", "won't" -> "wont").
       This prevents scikit-learn's default tokenizer from dropping single-letter tokens ('t', 's')
       and corrupting word semantics.
    4. Keep only lowercase letters (a-z) and spaces.
    5. Collapse consecutive whitespace characters into a single space and strip edges.
    """
    if not isinstance(text, str):
        return ""

    # 1. Replace HTML tags with a single space
    text = re.sub(r"<[^>]+>", " ", text)

    # 2. Lowercase
    text = text.lower()

    # 3. Strip apostrophes BEFORE removing non-letters so contractions stay united
    text = re.sub(r"['’]", "", text)

    # 4. Retain only lowercase alphabet characters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # 5. Collapse multiple spaces and strip leading/trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    # Regression and unit tests
    test_html = clean_text("end<br />then")
    test_upper = clean_text("<br />I LOVED it!!!")
    test_didnt = clean_text("I didn't like it")
    test_contractions = clean_text("it's great and can't be missed, won't disappoint")

    print(f"Test HTML: '{test_html}'")
    print(f"Test Upper: '{test_upper}'")
    print(f"Test Didn't: '{test_didnt}'")
    print(f"Test Contractions: '{test_contractions}'")

    assert test_html == "end then", f"Failed: expected 'end then', got '{test_html}'"
    assert test_upper == "i loved it", f"Failed: expected 'i loved it', got '{test_upper}'"
    assert test_didnt == "i didnt like it", f"Failed: expected 'i didnt like it', got '{test_didnt}'"
    assert test_contractions == "its great and cant be missed wont disappoint", (
        f"Failed: expected 'its great and cant be missed wont disappoint', got '{test_contractions}'"
    )

    print("All preprocessing tests passed successfully!")
