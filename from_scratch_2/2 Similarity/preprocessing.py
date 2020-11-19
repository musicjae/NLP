import numpy as np
import re
from nltk.corpus import stopwords

def tokenize (corpus):
    # (1) 소문자화
    corpus = corpus.lower()
    # (2) 마침표 앞에 공백 넣기
    corpus = corpus.replace('.', '')
    corpus = corpus.replace('\n', '')
    corpus = corpus.replace('?','')
    corpus = corpus.replace('--', '')
    # (3) 공백을 기준으로 말뭉치를 단어로 분할하기
    words = corpus.split(' ')


    word2id = {}  # dict 형태로 초기화
    id2word = {}

    for word in words:
        if word not in word2id:
            new_id = len(word2id)  # 우변은 0 - 12까지 늘어난다.
            word2id[word] = new_id  # 처음 0 idx 경우, word2id[word]에는 입력되는 첫 단어 'philosophy'가 0 idx와 상응하여 저장. 12까지 차례대로 단어들 들어오며 이 과정 반복
            id2word[new_id] = word  # idx 0부터 상응하는 각 단어를 dict 형태로 출력

    texts_id = np.array([word2id[w] for w in words])

    return texts_id, word2id, id2word

# 분포 가설: 단어의 의미는 주변 단어에 의해 형성된다.


#################################################
########## 단어를 벡터로 표현하기 ###################
#################################################
# Statistical based (주변 단어 카운트하여 매트릭스 만들기)

def create_co_matrix (corpus, vocab_size, window_size=1):

    corpus_size = len(corpus) # 길이 구하기
    co_matrix = np.zeros((vocab_size, vocab_size), dtype=np.int32) # 초기화

    for idx, word_id in enumerate(corpus): ## 윈도우 내에 포함된 주변 단어context words 세기

        for i in range(1, window_size+1):

            left_idx = idx - i # 맨 왼쪽 idx 조사
            right_idx = idx + i # 맨 오른쪽 idx 조사

            if left_idx >= 0: # 맨 왼쪽이거나 그것보다 크면

                left_word_id = corpus[left_idx]
                co_matrix[word_id, left_word_id] += 1

            if right_idx < corpus_size: # 코퍼스 사이즈 내에 조사 범위가 있다면

                right_word_id = corpus[right_idx]
                co_matrix[word_id, right_word_id] += 1

    return co_matrix




