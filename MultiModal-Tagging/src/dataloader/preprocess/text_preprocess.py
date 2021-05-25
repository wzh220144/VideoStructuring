import tokenization
import numpy as np

class Preprocess:

    def __init__(self, vocab, max_len, is_training=False):
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab)
        self.max_len = max_len
        self.is_training = is_training

    def __call__(self, text):
        tokens = ['[CLS]'] + self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = ids[:self.max_len]
        ids = ids + [0]*(self.max_len-len(ids))
        return np.array(ids).astype('int64')
