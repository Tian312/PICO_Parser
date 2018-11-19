from scipy import spatial
import numpy as np
from nltk.corpus import stopwords
english_stopwords = stopwords.words('english')
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
import re, operator

def mult_sim(w, target_v, context_v):
    target_similarity = w.dot(target_v)
    target_similarity[target_similarity<0] = 0.0
    context_similarity = w.dot(context_v)
    context_similarity[context_similarity<0] = 0.0
    return (target_similarity * context_similarity)

def get_ngram_vec(ngram, w,word2index):
    words = re.split("\s+",ngram.lower())
    dim = len(w[word2index["the"]])
    non_vec = np.zeros(dim)
    ngram_vec = np.zeros(dim)
    for word in words:
        if word in word2index.keys():
            ngram_vec = ngram_vec+w[word2index[word]]
    ngram_vec = ngram_vec/len(words)
    return ngram_vec

    
def get_most_similar(target_term,term_list,w,word2index):
    target_vec = get_ngram_vec(target_term,w,word2index)
    max_sim=0
    max_word=""
    sim_dic={}
    for term in term_list:
        term_vec = get_ngram_vec(term,w,word2index)
        sim = 1 - spatial.distance.cosine(target_vec, term_vec)
        sim_dic[term]=sim
        if sim > max_sim:
            max_term=term
            max_sim=sim
    sim_dic = sorted(sim_dic.items(), key=operator.itemgetter(1),reverse=True)
    return max_term,max_sim,sim_dic

''' TEST
model_param_file = "/home/tk2624/gitrepos/context2vec/models/pubmed.c2v.200d.model.params"
from model_reader import ModelReader
model_reader = ModelReader(model_param_file)
w = model_reader.w
word2index = model_reader.word2index

target = "hospital stay"
term_list = ['cost of medical care','surgical morbidity','feasible and effective in improving pulmonary functions']
max_term,max_sim,sim_dic = get_most_similar(target,term_list,w,word2index)
print (target,max_term)
print (sim_dic)
'''