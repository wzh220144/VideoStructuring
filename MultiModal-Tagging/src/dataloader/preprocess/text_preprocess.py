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
        cols = text_path.split('/')[:-2]
        vid = text_path.split('/')[-1]
        asr_path = '/'.join(cols + ['asr_txt', vid])
        ocr_path = '/'.join(cols + ['ocr_txt', vid])

        if os.path.exists(asr_path):
            with open(asr_path, 'r') as f:
                for line in f:
                    text += ' ' + line.strip('\n')
        if os.path.exists(ocr_path):
            with open(ocr_path, 'r') as f:
                for line in f:
                    text += ' ' + line.strip('\n')
        tokens = ['[CLS]'] + self.tokenizer.tokenize(text)
        ids = self.tokenizer.convert_tokens_to_ids(tokens)
        ids = ids[:self.max_len]
        ids = ids + [0]*(self.max_len-len(ids))
        return np.array(ids).astype('int64')
