import warnings,time
warnings.simplefilter(action='ignore', category=RuntimeWarning)
from scipy import spatial
import numpy as np
from nltk.corpus import stopwords
english_stopwords = stopwords.words('english')
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
import re, operator

def mult_sim(w, target_v, context_v):
    if target_v is not None and context_v is not None:
        target_similarity = w.dot(target_v)
        target_similarity[target_similarity<0] = 0.0
        context_similarity = (w.dot(context_v)+1)/2
        
        #context_similarity=w.dot(context_v)
        #context_similarity[context_similarity<0] = 0.0
        
        similarity = target_similarity * context_similarity
    else:
        if target_v is not None:
                v = target_v
        elif context_v is not None:
                v = context_v                
        else:
            raise ParseException("Can't find a target nor context.")   
        similarity = (w.dot(v)+1.0)/2 # Cosine similarity can be negative, mapping similarity to [0,1

    return similarity

def get_context_vec(sent, target_pos, model):
    if len(sent) > 1:
        context_v = model.context2vec(sent, target_pos) 
        context_v = context_v / np.sqrt((context_v * context_v).sum())
    else:
        context_v = None
    return (context_v)
#
def get_ngram_vec(ngram, w,word2index):
    words = re.split("\s+",ngram.lower())
    dim = len(w[word2index["the"]])
    non_vec = np.zeros(dim)
    ngram_vec = np.zeros(dim)
    for word in words:
        if word in word2index.keys():
            ngram_vec = ngram_vec+w[word2index[word]]
        else:
            ngram_vec = ngram_vec+non_vec
    
    ngram_vec = ngram_vec/len(words)
    
    return ngram_vec

    
def get_most_similar(target_term,term_list,w,word2index):
    target_vec = get_ngram_vec(target_term,w,word2index)
    max_sim=0
    max_term=""
    sim_dic={}
    for term in term_list:
        if term == target_term:
            sim =1
        else: 
            term_vec = get_ngram_vec(term,w,word2index)
            if any(term_vec)+ all(term_vec) == 0 or any(target_vec)+ all(target_vec) == 0:
                sim = 0
            else:
                sim = 1 - spatial.distance.cosine(target_vec, term_vec)

        sim_dic[term]=sim
        if sim > max_sim:
            max_term=term
            max_sim=sim
    sim_dic = sorted(sim_dic.items(), key=operator.itemgetter(1),reverse=True)
    return max_term,max_sim,sim_dic

# test
model_param_file = "/home/tk2624/gitrepos/context2vec/models/pubmed.c2v.200d.model.params"
from model_reader import ModelReader
model_reader = ModelReader(model_param_file)
w = model_reader.w
word2index = model_reader.word2index
model = model_reader.model

sent = "Even though both groups showed a significant rise in post-operative peak expiratory flow rate and inspiratory capacity after surgery , the post-operative peak expiratory flow rate and inspiratory capacity in group I was significantly higher than in group II ."
target="inspiratory capacity"
pos = 16
list = ["cost of medical care", "surgical morbidity","pulmonary functions"]
w_list = np.zeros((3,200))
w_list[0] = get_ngram_vec(list[0], w,word2index)
w_list[1] = get_ngram_vec(list[1], w,word2index)
w_list[2] = get_ngram_vec(list[2], w,word2index)
#print (w_list)
context_v = get_context_vec(sent.split(), pos, model)

t_vec = get_ngram_vec(target, w,word2index)

sim = mult_sim(w_list, t_vec, context_v)
count =0
print ((-sim).argsort())
for i in (-sim).argsort():
    if np.isnan(sim[i]):
        continue
    print('{0}: {1}'.format(list[i], sim[i]))
    count += 1
    if count == len(list):
        break
#target = "pindolol"
#term_list = ['pindolol', 'metoprolol', 'atenolol', 'labetalol']
#max_term,max_sim,sim_dic = get_most_similar(target,term_list,w,word2index)
#print (target,max_term)
#print (sim_dic)
