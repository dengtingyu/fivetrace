import re

def normalize(text):
    return re.sub(r"[ \t\u3000]+", " ", text.replace("\r\n", "\n").replace("\r", "\n")).strip()
