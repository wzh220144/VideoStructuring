import tokenization
import numpy as np
import os
import json

class Preprocess:

    def __init__(self, vocab, max_len, is_training=False):
        self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab)
        self.max_len = max_len
        self.is_training = is_training

    def __call__(self, text_path):
        text = ""
        if os.path.exists(text_path):
            with open(text_path, 'r') as f:
                obj = json.load(f)
                text = obj['video_ocr'] + obj['video_asr']
        tokens = ['[CLS]'] + self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = ids[:self.max_len]
        ids = ids + [0]*(self.max_len-len(ids))
        return np.array(ids).astype('int64')
