import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from converting import input_tensor_train, input_lang_train, input_lang_valid, input_lang_test, target_lang_train, target_lang_valid, target_lang_test
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence # https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html

"""
pack_padded_sequence 를 사용하면 PackedSequence object를 얻을 수 있다. packed_input 에는 위에서 말한 합병된 데이터와 각 타임스텝의 배치사이즈들이 담겨있다
"""

BUFFER_SIZE = len(input_tensor_train)
BATCH_SIZE = 64
N_BATCH = BUFFER_SIZE//BATCH_SIZE
embedding_dim = 256
units = 1024
vocab_input_size = len(input_lang_train.w2i)+len(input_lang_valid.w2i)+len(input_lang_test.w2i)
vocab_target_size =  len(target_lang_train.w2i)+len(target_lang_valid.w2i)+len(target_lang_test.w2i)


class Encoder(nn.Module):

    def __init__(self, vocab_size, embedding_dim, enc_units, batch_size):
        super(Encoder,self).__init__()
        self.batch_size = batch_size
        self.enc_units = enc_units # 몇 개의 unit이 들어오는가? 이것은 timestep의 개수에 의존한다.
        self.embedding_dim = embedding_dim
        self.vocab_size = vocab_size
        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim,self.enc_units)

    def forward(self, x, lens, device):

        x = self.embedding(x)
        x = pack_padded_sequence(x, lens) # 상단 설명 참고

        # hidden state를 따로 return하기 위해 초기화를 해준 뒤, return에 넣어준다.
        self.hidden = self.initialize_hidden_state(device) # self.hidden: 1, batch_size, enc_units

        output, self.hidden = self.gru(x, self.hidden)

        output, _ = pad_packed_sequence(output)

        return output, self.hidden

    def initialize_hidden_state(self, device):
        return torch.zeros((1,self.batch_size, self.enc_units)).to(device)


class Decoder(nn.Module):
    def __init__(self, vocab_size, embedding_dim, dec_units, enc_units, batch_size):
        super(Decoder, self).__init__()
        self.batch_size = batch_size
        self.dec_units = dec_units
        self.enc_units = enc_units
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim

        self.embedding = nn.Embedding(self.vocab_size, self.embedding_dim)
        self.gru = nn.GRU(self.embedding_dim + self.enc_units, self.dec_units, batch_first=True)  # ???
        self.fc = nn.Linear(self.enc_units, self.vocab_size)

        # Attention
        self.W1 = nn.Linear(self.enc_units, self.dec_units)
        self.W2 = nn.Linear(self.enc_units, self.dec_units)
        self.V = nn.Linear(self.enc_units,1)

    def forward(self, x, hidden, enc_output):

        enc_output = enc_output.permute(1,0,2) #(max_length, batch_size, enc_units) -> (batch_size, max_length, hidden_size)
        hidden_with_time_axis = hidden.permute(1,0,2) # hidden state는  (length, batch_size, hidden size)형태로 입력된다. 우리는 이것을 각 timestep에 대해 고려하기 위해,  (batch_size, 1, hidden size)로 바꿔준다.
        score = torch.tanh(self.W1(enc_output)+self.W2(hidden_with_time_axis)) # Bahdanaus's score: (batch_size, max_length, hidden_size)
        attention_weights = torch.softmax(self.V(score),axis=1) # attn_weight: (batch_size, max_length, 1). 여기서 V에 attn score를 적용함으로써, (_,_,1)을 얻는다
        context_vector = torch.sum((attention_weights*enc_output), dim= 1)

        x = self.embedding(x) # (batch_size, 1, embedding_dim)
        x = torch.cat((context_vector.unsqueeze(1),x), -1) # x shape after concat == (batch_size, 1, embedding_dim + hidden_size)
        output, state = self.gru(x) # (batch_size, 1, hidden_size)
        output = output.view(-1, output.size(2))# output shape == (batch_size * 1, hidden_size)
        x = self.fc(output)# output shape == (batch_size * 1, vocab)

        return x, state, attention_weights






