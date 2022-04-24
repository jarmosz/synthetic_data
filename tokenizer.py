import spacy

class Tokenizer:
    def __init__(self):
        self.polish_tokenizer = spacy.load('pl_core_news_lg')

    def tokenize(self, text):
        return [tok.text for tok in self.polish_tokenizer.tokenizer(text)]
