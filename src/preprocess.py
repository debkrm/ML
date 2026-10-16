import re


def clean_text(text: str) -> str:
    """
    Cleans raw review text for sentiment classification.

    Steps:
    1. Replace HTML tags with a space (prevents glued words like 'end<br />then' -> 'end then').
    2. Lowercase text.
    3. Keep only letters (a-z) and spaces.
    4. Collapse consecutive whitespace characters into a single space and strip edges.
    """
    if not isinstance(text, str):
        return ""

    # 1. Replace HTML tags with a single space
    text = re.sub(r"<[^>]+>", " ", text)

    # 2. Lowercase
    text = text.lower()

    # 3. Retain only lowercase alphabet characters and spaces
    text = re.sub(r"[^a-z\s]", " ", text)

    # 4. Collapse multiple spaces and strip leading/trailing whitespace
    text = re.sub(r"\s+", " ", text).strip()

    return text


if __name__ == "__main__":
    # Smoke tests for clean_text
    test_1 = clean_text("end<br />then")
    test_2 = clean_text("<br />I LOVED it!!!")

    print(f"Test 1 result: '{test_1}'")
    print(f"Test 2 result: '{test_2}'")

    assert test_1 == "end then", f"Test 1 failed: expected 'end then', got '{test_1}'"
    assert test_2 == "i loved it", f"Test 2 failed: expected 'i loved it', got '{test_2}'"

    print("All preprocessing tests passed!")
