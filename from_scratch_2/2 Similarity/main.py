import numpy as np
import preprocessing
import similarity
import matplotlib.pyplot as plt
from nltk.corpus import stopwords

f = open("plato.txt", 'r')
text = f.read()

## 전처리 ##

pre_text,w2i,i2w = preprocessing.tokenize(text)

## co 행렬 ##

mat = preprocessing.create_co_matrix(pre_text,vocab_size=len(w2i))

## 단어 벡터 간 유사도 확인 ##

c1 =mat[w2i['plato']]
c2 = mat[w2i['socrates']]


print(f'유사도 값: {similarity.cos_sim(c1,c2)}')
print('\n유사도랭킹: ')
#print(similarity.most_similar('socrates',w2i,i2w,mat,top=7))

W = similarity.ppmi(mat)
np.set_printoptions(precision=3) #유효자릿수 3
#print('\nPPMI로 구한 유사도 랭킹')
#print(similarity.most_similar('socrates',w2i,i2w,W,top=7)) ########### 시간이 오래 걸려서 기다려야 한다 ###################

### SVD ###

U, S, V = np.linalg.svd(W) ## 희소벡터 W --> 밀집벡터 U로 바꾸자.
print(U[0], W[0])

for word, word_id in w2i.items():
    plt.annotate(word, (U[word_id,0],U[word_id,1]))
    plt.scatter(U[:,0],U[:,1],alpha=0.5)
    plt.show()