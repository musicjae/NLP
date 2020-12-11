# https://www.youtube.com/watch?v=sQUqQddQtB4
from data_load import train, valid, test
import pandas as pd
import unicodedata
import re


def unicode_to_ascii(s):
    """
    Normalizes latin chars with accent to their canonical decomposition
    """
    return ''.join(c for c in unicodedata.normalize('NFD', s)
                   if unicodedata.category(c) != 'Mn')


def preprocess_sentence(w):
    w = unicode_to_ascii(w.lower().strip())

    w = re.sub(r"([?.!,¿])", r" \1 ", w)
    w = re.sub(r'[" "]+', " ", w)

    w = re.sub(r"[^a-zA-Z?.!,¿]+", " ", w)

    w = w.rstrip().strip()

    w = '<sos> ' + w + ' <eos>'
    return w

train['en'] = train.en.apply(lambda w : preprocess_sentence(w))
train['de'] = train.en.apply(lambda w : preprocess_sentence(w))
train = pd.concat([train['de'],train['en']],axis = 1)

valid['en'] = train.en.apply(lambda w : preprocess_sentence(w))
valid['de'] = train.en.apply(lambda w : preprocess_sentence(w))
valid = pd.concat([valid['de'], valid['en']],axis = 1)

test['en'] = train.en.apply(lambda w : preprocess_sentence(w))
test['de'] = train.en.apply(lambda w : preprocess_sentence(w))
test = pd.concat([test['de'],test['en']],axis=1)
