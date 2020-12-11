from utils import device, sort_batch
from model import Encoder, Decoder, vocab_input_size, embedding_dim, units, BATCH_SIZE, vocab_target_size
from converting import dataset, target_lang_train
import torch

"""
인코더 테스터
"""
encoder = Encoder(vocab_input_size,embedding_dim,units,BATCH_SIZE)
encoder.to(device)

iteration = iter(dataset)
x,y,x_len =  next(iteration)
xsorted, ysorted, lensorted = sort_batch(x,y,x_len)

enc_output, enc_hidden = encoder(xsorted.to(device),lensorted,device)

#print(enc_output[0][0])
print("------------Encoder info --------------------")
print("Input: ", x.shape)
print("Output: ", y.shape)
print("Encoder Output: ", enc_output.shape) # batch_size X max_length X enc_units
print("Encoder Hidden: ", enc_hidden.shape) # batch_size X enc_units (corresponds to the last state)

"""
디코더 테스터
"""
decoder = Decoder(vocab_target_size, embedding_dim, units, units, BATCH_SIZE).to(device)

## enc 마지막 은닉층 --> dec 첫 은닉층
dec_hidden = enc_hidden
dec_input = torch.tensor([[target_lang_train.w2i['<sos>']]] * BATCH_SIZE)
print(f'Decoder input shape: {dec_input.shape}')

for t in range(1, y.size(1)): # target의 개수만큼 loop
    predictions, dec_hidden, _ = decoder(dec_input.to(device), # Remind that "def forward(self, x, hidden, enc_output)"
                                         dec_hidden.to(device),
                                         enc_output.to(device))

    print("Prediction: ", predictions.shape)
    print("Decoder Hidden: ", dec_hidden.shape)

    dec_input = y[:,t].unsqueeze(1)
    print(dec_input.shape)
    break