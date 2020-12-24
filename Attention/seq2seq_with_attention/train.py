import torch
import torch.nn as nn
import torch.autograd as autograd
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence # https://simonjisu.github.io/nlp/2018/07/05/packedsequence.html
from time import time

from converting import input_tensor_train, input_lang_train, input_lang_valid, input_lang_test, target_lang_train, target_lang_valid, target_lang_test, dataset
from utils import loss_function
from utils import device, sort_batch
from model import Encoder, Decoder, vocab_input_size, embedding_dim, units, BATCH_SIZE, vocab_target_size, N_BATCH


EPOCHS = 10

encoder = Encoder(vocab_input_size, embedding_dim, units, BATCH_SIZE).to(device)
decoder = Decoder(vocab_target_size, embedding_dim, units, units, BATCH_SIZE).to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr= 0.001)

#### Training START!!! ####
losses = []

start = time()
for epoch in range(EPOCHS):
    print('Training............................................................')

    encoder.train()
    decoder.train()

    total_loss = 0

    for (batch, (input, target, input_len)) in enumerate(dataset):

        loss = 0

        xsorted, ysorted, x_sorted_len = sort_batch(input, target, input_len)
        enc_output, enc_hidden = encoder(xsorted.to(device), x_sorted_len, device) # Remind that  def forward(self, x, lens, device):
        dec_hidden = enc_hidden
        dec_input = torch.tensor([[target_lang_train.w2i['<sos>']]] * BATCH_SIZE) # tensor([[256]])

        for t in range(1, ysorted.size(1)):

            predictions, dec_hidden, _ = decoder(dec_input.to(device),
                                                 dec_hidden.to(device),
                                                 enc_output.to(device))


            loss += loss_function(ysorted[:,t].to(device), predictions.to(device), criterion) #Remind that this: (real_value, pred_value, criterion)


            dec_input = ysorted[:,t].unsqueeze(1) # t 마다 디코더에 들어가는 입력을 갱신

        batch_loss = loss / int(ysorted.size(1))
        #print('GPU 사용 여부 확인: ', torch.cuda.is_available())
        print('소요 Batch 개수:', batch)
        total_loss += batch_loss
        losses.append(total_loss)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        if batch % 1 == 0:
            print('Epoch {} Batch {} Loss {:.4f}'.format(epoch + 1,
                                                         batch,
                                                         batch_loss.detach().item()))

    print(f'Epoch: {epoch+1}\n\t Loss: {total_loss/N_BATCH}')

end = time()
print('소요 시간: ', round(end-start,3),'sec')


