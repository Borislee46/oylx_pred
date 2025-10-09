import re
import time


def stream_text(text):
    for word in text:
        yield word
        time.sleep(0.01)


def clean_user_input(text):
    if not text:
        return ""

    text = re.sub(r"\s+", " ", text)
    text = text.strip()
    text = re.sub(r"[^\u4e00-\u9fa5a-zA-Z0-9\s，。？！、的有哪些什么关于]", "", text)
    text = re.sub(r"\s+", " ", text).strip()

    return text
