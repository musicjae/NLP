import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
def cos_sim (x,y,eps=1e-8):
    nx = x/(np.sqrt(np.sum(x**2))+eps)
    ny=  y/(np.sqrt(np.sum(y**2))+eps)

    return np.dot(nx,ny)

def most_similar(query, word2id,id2word, word_matrix, top=10):

    if query not in word2id:
        print(f'{query}를 찾을 수 없습니다')
        return

    print('\n[query]'+query)
    query_id = word2id[query] # 해당 쿼리에 대한 id 가져오기
    query_vec = word_matrix[query_id] # 그 id로 행렬 만들기

    vocab_size = len(id2word)
    similarity = np.zeros(vocab_size) # 유사도를 구하기 위한 초기화

    for i in range(vocab_size):
        similarity[i] = cos_sim(word_matrix[i], query_vec) # 해당 쿼리로 전체 행렬의 item에 대해 유사도 조사

    count = 0
    for i in (-1*similarity).argsort(): # 내림차순 정렬
        if id2word[i] == query: # 해당 단어와 쿼리가 같을 시
            continue
        print(f'{id2word[i]},{similarity[i]}')

        count += 1

        if count >= top: # 상위 원소들만 출력

            return


def ppmi (mat, verbose=False, eps=1e-8):

    M = np.zeros_like(mat, dtype=np.float32) # 두 개의 단어가 동시에 mat에 나타날 확률에 대한 행렬 초기화
    N = np.sum(mat) # 말뭉치에 포함된 단어의 개수
    S = np.sum(mat, axis = 0) # 한 개의 단어가 mat에 나타날 확률에 대한 행렬 초기화
    total = mat.shape[0]*mat.shape[1]
    cnt = 0

    for i in range(mat.shape[0]):
        for j in range(mat.shape[1]):

            pmi = np.log2((mat[i,j]*N)/(S[j]*S[i])+eps)
            M[i,j] = max(0,pmi) # PPMI

            if verbose:
                cnt +=1
                if cnt % (total//100) ==0:
                    print(f'{100*cnt/total}완료')

    return M