import spacy


class Tokenizer:

    def __init__(self):
        # corpus_de, corpus_en
        self.spacy_de = spacy.load('de_core_news_sm')
        self.spacy_en = spacy.load('en_core_web_sm')

    # Tokenize German text from a string to a list of tokens
    def tokenize_de(self, text):
        return [token.text for token in self.spacy_de.tokenizer(text)]

    # Tokenize English text from a string to a list of tokens
    def tokenize_en(self, text):
        return [token.text for token in self.spacy_en.tokenizer(text)]
