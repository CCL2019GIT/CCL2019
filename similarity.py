import numpy as np
import gensim
import pdb
import sys,getopt
from scipy.linalg import norm
from scipy.stats import spearmanr
from pypinyin import pinyin, lazy_pinyin, Style, TONE2
from elmoformanylangs import Embedder
#
model_file = 'data/word2vec/news_12g_baidubaike_20g_novel_90g_embedding_64.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)

def judge_pure_english(keyword):
    return keyword.encode( 'UTF-8' ).isalpha()

def read_wordpair(sim_file):
    f1 = open(sim_file, 'r')
    pairs = []
    for line in f1:
        print(line)
        pair = line.split()
        pair[2] = float(pair[2])
        pairs.append(pair)
    f1.close()
    return pairs


def vector_similarity(s1, s2):
    def sentence_vector(s):
        words = jieba.lcut(s)
        v = np.zeros(64)
        for word in words:
            v += model[word]
        v /= len(words)
        return v
    v1, v2 = sentence_vector(s1), sentence_vector(s2)
    return np.dot(v1, v2) / (norm(v1) * norm(v2))

if  __name__ == '__main__':
    # w1='可能'
    # w2='科技'
    # pw1 = lazy_pinyin(w1, style=TONE2)
    # pw2 = lazy_pinyin(w2, style=TONE2)
    # p = Embedder('elmoformanylangs/pinyin.model', batch_size=10)
    # elmos=p.sents2elmo([pw1,pw2])
    # print(len(elmos))
    # print(len(elmos[0]))
    # print(len(elmos[0][0]))
    # pw1=0
    # pw2=0
    # for i in range(len(w1)):
    #     pw1+=elmos[0][i]
    # pw1/=len(w1)
    # for i in range(len(w2)):
    #     print(i)
    #     pw2+=elmos[1][i]
    # pw2/=len(w2)
    # print(type(pw1))
    # print(pw1)
    # vsim =pw1.dot(pw2.T) / (np.linalg.norm(pw1) * np.linalg.norm(pw2))

    # human_sim.append(pair[2])
    # vec_sim.append(vsim)
    # print(vsim)
    p = Embedder('elmoformanylangs/pinyin.model', batch_size=10)
    fname= 'sim-502.txt'
    pairs = read_wordpair(fname)
    human_sim = []
    vec_sim = []
    cnt = 0
    total = len(pairs)
    for pair in pairs:
        w1 = pair[0]
        w2 = pair[1]
        # # fw.write(w1+'\t'+w2+'\n')
        # print(w1,w2)
        # pw1 = lazy_pinyin(w1, style=TONE2)
        # pw2 = lazy_pinyin(w2, style=TONE2)

        # elmos = p.sents2elmo([pw1, pw2])
        # print(len(elmos))
        # print(len(elmos[0]))
        # print(len(elmos[0][0]))
        # pw1 = 0
        # pw2 = 0
        # for i in range(len(w1)):
        #     pw1 += elmos[0][i]
        # pw1 /= len(w1)
        # for i in range(len(w2)):
        #     print(i)
        #     pw2 += elmos[1][i]
        # pw2 /= len(w2)
        # print(type(pw1))
        # print(pw1)
        # # print(len(pww1))
        # vsim = pw1.dot(pw2.T) / (np.linalg.norm(pw1) * np.linalg.norm(pw2))
        # human_sim.append(pair[2])
        # vec_sim.append(vsim)
        if w1 in model and w2 in model :#and not judge_pure_english(w1) and not judge_pure_english(w2)
            we1 = model[w1]
            we2 = model[w2]
            print(type(we1))
            print(we1)
            print(len(we1))
            print(len(we2))
            vsim = we1.dot(we2.T) / (np.linalg.norm(we1) * np.linalg.norm(we2))
            #w1 = w1.decode('utf-8')
            #w2 = w2.decode('utf-8')
            pw1 = lazy_pinyin(w1, style=TONE2)
            pw2 = lazy_pinyin(w2, style=TONE2)

            elmos=p.sents2elmo([pw1,pw2])
            print(len(elmos))
            print(len(elmos[0]))
            print(len(elmos[0][0]))
            pw1=0
            pw2=0
            for i in range(len(w1)):
                pw1+=elmos[0][i]
            pw1/=len(w1)
            for i in range(len(w2)):
                print(i)
                pw2+=elmos[1][i]
            pw2/=len(w2)
            print(type(pw1))
            print(pw1)

            pww1=np.append(we1, pw1)
            pww2 = np.append(we2, pw2)
            print(len(pww1))
            vsim =pw1.dot(pw2.T) / (np.linalg.norm(pw1) * np.linalg.norm(pw2))
            human_sim.append(pair[2])
            vec_sim.append(vsim)
    print(cnt, ' word pairs appered in the training dictionary , total word pairs ', total)
    score = spearmanr(human_sim, vec_sim)
    print(score)