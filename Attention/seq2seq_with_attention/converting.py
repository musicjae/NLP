from preprocessing import train, valid, test
from utils import LanguageIndex, pad_sequence, max_length, MyData
from torch.utils.data import DataLoader
import torch

# converting.py to get results from preprocessing.py

## To do List ##

# (1) word to index
# (2) Set max_length of each sentence in language sets.
# (3) padding sequence
# (4) DataLoader for Batching
BATCH_SIZE = 64

input_lang_train = LanguageIndex(train['de'].values.tolist())
target_lang_train = LanguageIndex(train['en'].values.tolist())

input_lang_valid = LanguageIndex(valid['de'].values.tolist())
target_lang_valid = LanguageIndex(valid['en'].values.tolist())

input_lang_test = LanguageIndex(test['de'].values.tolist())
target_lang_test = LanguageIndex(test['en'].values.tolist())
#print(torch.tensor([[target_lang_test.w2i['<sos>']]])*64)

"""
(1) 언어 집합 내의 문장들을 토큰화한 뒤, 나오는 단어들을 w2i로 인덱스화
"""
input_tensor_train = [[input_lang_train.w2i[word] for word in sen.split(' ')] for sen in train['de'].values.tolist()]
target_tensor_train = [[target_lang_train.w2i[word] for word in sen.split(' ')] for sen in train['en'].values.tolist()]

input_tensor_val = [[input_lang_valid.w2i[word] for word in sen.split(' ')] for sen in valid['de'].values.tolist()]
target_tensor_val = [[target_lang_valid.w2i[word] for word in sen.split(' ')] for sen in valid['en'].values.tolist()]

input_tensor_test = [[input_lang_test.w2i[word] for word in sen.split(' ')] for sen in test['de'].values.tolist()]
target_tensor_test = [[target_lang_test.w2i[word] for word in sen.split(' ')] for sen in test['en'].values.tolist()]

"""
(2) max_length
"""

max_len_inp_train = max_length(input_tensor_train)
max_len_inp_valid = max_length(input_tensor_val)
max_len_inp_test = max_length(input_tensor_test)

max_len_tgt_train = max_length(target_tensor_train)
max_len_tgt_valid = max_length(target_tensor_val)
max_len_tgt_test = max_length(target_tensor_test)

"""
(3) padding sequence
"""

input_tensor_train = [pad_sequence(x, max_len_inp_train) for x in input_tensor_train]
target_tensor_train = [pad_sequence(x, max_len_tgt_train) for x in target_tensor_train]

input_tensor_valid = [pad_sequence(x, max_len_inp_valid) for x in input_tensor_val]
target_tensor_valid = [pad_sequence(x, max_len_tgt_valid) for x in target_tensor_val]

input_tensor_test = [pad_sequence(x, max_len_inp_test) for x in input_tensor_test]
target_tensor_test = [pad_sequence(x, max_len_tgt_test) for x in target_tensor_test]


"""
(4) Dataset and DataLoader

  - Get final dataset for training and test_dataset
"""

train_Dataset = MyData(input_tensor_train, target_tensor_train)
valid_Dataset = MyData(input_tensor_valid, target_tensor_valid)
test_Dataset = MyData(input_tensor_test, target_tensor_test)

dataset = DataLoader(train_Dataset, batch_size=BATCH_SIZE,drop_last=True,shuffle=True)




