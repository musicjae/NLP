import numpy as np
from torch.utils.data import Dataset, DataLoader
import torch

class LanguageIndex(): # For w2i, i2w

    def __init__(self,lang):

        self.lang = lang
        self.w2i = {}
        self.i2w = {}
        self.vocab = set() # init
        self.create_index()

    def create_index(self):

        for sen in self.lang:
            self.vocab.update(sen.split(' ')) # 각 언어 집합 내의 문장을 토큰화하여 단어 집합 내에 추가

        self.vocab = sorted(self.vocab) # 업데이트된 단어집합을 정렬
        self.w2i['<pad>'] = 0 # 인덱스화 시에, 언어 집합 내의 모든 문장의 길이는 동일하지 않다. 그러한 탓에 인덱스된 벡터의 길이도 다 다를 수 있다. 이것을 방지하기 위해 패딩

        for idx, word in enumerate(self.vocab):
            self.w2i[word] = idx + 1 # pad 토큰이 0에 들어왔기 때문에 다른 모든 단어들의 인덱스를 +1 해준다

        for word, idx in self.w2i.items():
            self.i2w[idx] = word

def max_length(tensor):
    return max(len(t) for t in tensor)

def pad_sequence(x, max_len):

    padded = np.zeros((max_len), dtype=np.int64)

    if len(x) > max_len:
        padded[:] = x[:max_len] # 설정된 최대 길이보다 더 긴 문장이 들어온다면, x의 길이를 max_len까지 padded에 저장
    else:
        padded[:len(x)] = x #

    return padded

class MyData(Dataset):

    def __init__(self,X,y):
        self.data = X
        self.target = y
        self.length = [np.sum(1-np.equal(x,0)) for x in X] # can't understand

    def __getitem__(self, index):

        x = self.data[index]
        y = self.target[index]
        x_len = self.length[index]

        return x,y, x_len

    def __len__ (self):

        return len(self.data)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def sort_batch(X,y,leng):
    len, idx = leng.sort(dim = 0, descending = True)
    X = X[idx]
    y = y[idx]

    return X.transpose(0,1), y, len # transpose (batch x seq) to (seq x batch)

def loss_function(real_value, pred_value, criterion): # """ Only consider non-zero inputs in the loss; mask needed """

    mask = real_value.ge(1).type(torch.FloatTensor).to(device)
    loss_ = criterion(pred_value, real_value) * mask

    return torch.mean(loss_)
