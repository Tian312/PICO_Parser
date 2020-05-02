from model.data_utils import Dataset,get_processing_word, minibatches
from model.models import HANNModel
from model.config import Config
import argparse
import codecs,re,time
import os,sys,warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES']='1,2,3'
import warnings
warnings.filterwarnings("ignore")

import text_tokenize
tokenizer = text_tokenize.mytokenizer()

parser = argparse.ArgumentParser()
config = Config(parser) 
# build model
model = HANNModel(config)
model.build()
model.restore_session("study_arms/model.weights")

def preprocess_ab(line):  
        info = re.split("\s\|\s",line)
        pmid = info[0]
        title = info[1]
        ab = info[2]
        #sents = sent_tokenize(line)
        sents =tokenizer.sent_tokenize(ab,mask = False)
        #sents.insert(0,title)
        #title = sents[0]

        return pmid,title, sents

'''
0 OBJECTIVE
1 RESULTS
2 METHODS
3 BACKGROUND
4 CONCLUSIONS
'''
    
def tag_sentence(data_dir, out_dir,model,tokenizer, lines_ready = False):
    tags= {0:"OBJECTIVE",1:"RESULTS",2:"METHODS",3:"BACKGROUND",4:"CONCLUSIONS",5:"TITLE"}
    
    
    if lines_ready == False:
    
        rawdata_dir = data_dir
        datadir = rawdata_dir+"rawtext_byline.all"
        infile = codecs.open(rawdata_dir,'r')
        docs= []
        count = 0
        old_time =time.time()

        for line in infile:
        
            if line.rstrip() == "":
                continue
            count += 1
            if count%100 == 0:
                new_time = time.time()
                cost = new_time-old_time
                cost_m,cost_s=divmod(cost, 60)
                print ("processing",count,"th abstracts... cost", cost_m," m...")
            new_sents=[]
            outfile = codecs.open(datadir,'w')
            pmid, title ,sents = preprocess_ab(line.strip())
            for sent in sents:
                sent=re.sub("^\s*OBJECTIVES\s*\:?\s*|BACKGROUND\s*\:?\s*|DESIGN\s*\:?\s*|RESULTS\s*\:?\s*|CONCLUSIONS\s*\:?\s*|RESEARCH DESIGN AND METHODS\s*\:?\s*|SETTING\s*\:?\s*|MEASUREMENTS\s*\:?\s*|PARTICIPANTS\s*\:?\s*|RATIONALE\s*\:?\s*|METHODSs*\:?\s*","",sent)
                outfile.write("CONCLUSIONS\t"+sent+"\n")
            outfile.write("\n\n")
            data = Dataset(datadir, config.processing_word, config.processing_tag)
            for words, labels in data:
                labels_pred, document_lengths = model.predict_batch([words])
                sents.insert(0,title)
                labels_pred[0].insert(0,5)
            for sent,pred in zip(sents,labels_pred[0]):

                ##=========== Only need METHODS
                #if tags[pred] == "OBJECTIVE" or tags[pred] == "CONCLUSIONS":
                new_sents.append(tags[pred]+"||"+sent)
                #new_sents.append(sent)
            docs.append(new_sents)
        rmcommand = "rm "+datadir
        os.system(rmcommand)
    else:
        print ("lines ready! \n")
        rawdata_dir= data_dir
        infile = codecs.open(data_dir,"r")
        datadir = rawdata_dir+"rawtext_byline.all"
        docs= []
        sents=[]
        new_sents=[]
        outfile = codecs.open(datadir,'w')
        for line in infile:
            print ("processing===", line.rstrip()+"\n")
            if line.rstrip() == "":
                outfile.write("\n")
                data = Dataset(datadir, config.processing_word, config.processing_tag)
                for words, labels in data:
                    labels_pred, document_lengths = model.predict_batch([words])
                for sent,pred in zip(sents,labels_pred[0]):
                    print (tags[pred],"===", sent)
                    new_sents.append(tags[pred]+"||"+sent)
                '''
                for s in new_sents:
                    print (s)
                    out_dir.write(s+"\n")
                out_dir.write("\n\n")
                '''
                docs.append(new_sents)
                outfile = codecs.open(datadir,'w')
                sents = []
                new_sents=[]
            else:
                info = re.split("\s",line.rstrip())
                tag= info[0]
                sent = " ".join(info[1:]) 
                new_sent=re.sub("^\s*OBJECTIVES\s*\:?\s*|BACKGROUND\s*\:?\s*|DESIGN\s*\:?\s*|RESULTS\s*\:?\s*|CONCLUSIONS\s*\:?\s*|RESEARCH D        ESIGN AND METHODS\s*\:?\s*|SETTING\s*\:?\s*|MEASUREMENTS\s*\:?\s*|PARTICIPANTS\s*\:?\s*|RATIONALE\s*\:?\s*|METHODSs*\:?\s*","",sent)
                outfile.write(tag+"\t"+new_sent+"\n")
                sents.append(sent)
    return docs

def main():
    data_dir = "test.txt"
    out_dir= codecs.open("test.sents","w")
    docs = tag_sentence(data_dir,out_dir, model, tokenizer, lines_ready = False)
    for i in docs:
        for s in i:
            out_dir.write(s+"\n")

        out_dir.write("\n\n") 
if __name__ == "__main__":
    main()

