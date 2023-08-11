import spacy
import os


def load_tokenizer(name='en_core_web_sm'):
    tokenizer = None
    if name in ("en", "en_core_web_sm"):
        try:
            tokenizer = spacy.load("en_core_web_sm")
        except IOError:
            os.system("python -m spacy download en_core_web_sm")
            tokenizer = spacy.load("en_core_web_sm")
    elif name in ("de", "de_core_news_sm"):
        try:
            tokenizer = spacy.load("de_core_news_sm")
        except IOError:
            os.system("python -m spacy download de_core_news_sm")
            tokenizer = spacy.load("de_core_news_sm")
    elif name in ("zh", "zh-core-web-sm"):
        try:
            tokenizer = spacy.load("zh-core-web-sm")
        except IOError:
            os.system("python -m spacy download zh-core-web-sm")
            tokenizer = spacy.load("zh-core-web-sm")

    return tokenizer


def tokenizing(text: str, tokenizer):
    try:
        tokens = [tok.text for tok in tokenizer.tokenizer(text)]
    except Exception as e:
        tokens = text.strip().split()
    return tokens
