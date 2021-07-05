from torchtext.legacy.data import Field, BucketIterator
from torchtext.legacy.datasets.translation import Multi30k


class DataLoader:
    source: Field = None
    target: Field = None

    def __init__(self, ext, tokenize_en, tokenize_de, sos_token, eos_token):
        self.ext = ext
        self.tokenize_en = tokenize_en
        self.tokenize_de = tokenize_de
        self.sos_token = sos_token
        self.eos_token = eos_token
        print('data initializing start')

    # generate field
    def make_dataset(self):
        if self.ext == ('.de', '.en'):
            self.source = Field(tokenize=self.tokenize_de, init_token=self.sos_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_en, init_token=self.sos_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
        elif self.ext == ('.en', '.de'):
            self.source = Field(tokenize=self.tokenize_en, init_token=self.sos_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)
            self.target = Field(tokenize=self.tokenize_de, init_token=self.sos_token, eos_token=self.eos_token,
                                lower=True, batch_first=True)

        train_data, valid_data, test_data = Multi30k.splits(exts=self.ext, fields=(self.source, self.target))
        return train_data, valid_data, test_data

    # build the vocabulary & mapping integer
    def build_vocab(self, train_data, min_freq):
        # min_freq : lower bound frequency of the word's appearance
        self.source.build_vocab(train_data, min_freq=min_freq)
        self.target.build_vocab(train_data, min_freq=min_freq)

    def make_iter(self, train, validate, test, batch_size, device):
        train_iterator, valid_iterator, test_iterator = BucketIterator.splits((train, validate, test),
                                                                              batch_size=batch_size,
                                                                              device=device)

        print('dataset initializing done')
        return train_iterator, valid_iterator, test_iterator
