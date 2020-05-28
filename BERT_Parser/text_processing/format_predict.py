import sys,re,os,codecs
import time

def raw2IOB(preds):
    new_preds=[]
    begin = 0
    lastp = "O"
    for p in preds:
        
        if p == "O":
            begin = 0
            new_preds.append("O")
            lastp = p
            continue
        else:
            if begin ==0:
                new_preds.append("B-"+p)
                lastp=p
                begin =1
            else:
                if p==lastp:
                    new_preds.append("I-"+p)
                    lastp=p
                    begin =1 
                else: 
                    new_preds.append("B-"+p)
                    lastp=p
                    begin = 1
                    
    return new_preds

def check_IOB(preds):
    
    start =0
    match = re.search("B-",str(preds))
    if not match:
        preds=raw2IOB(preds)
    s = "=".join(preds)
    new_s = re.sub("O=I-","O=B-",s)
    new_s = re.sub("^I","B",new_s)
    new_preds=new_s.split("=")
    return (new_preds)

def clean_text(text):
    words = word_tokenize(text)
    return " ".join(words)

def get_predict(abstract_text,model, tokenizer,pmid=True):
    if pmid==True:
        pmid_text,abstract_text = re.split("\|\|",abstract_text)
    sents = tokenizer.sent_tokenize(abstract_text, mask = True) #mask = True protects contents in [] () not to be splited. can set mask = False to turn off this function
    out_text = []
    out_preds= []
    for sent in sents:
        words = sent.split()
        out_text.append(words)
        preds=model.predict(words)
        out_preds.append(preds)
    if pmid == True:
        return out_text, out_preds,pmid_text
    else:
        return out_text, out_preds

def get_predict_from_bert(abstract_text,model,tokenizer,pmid=True):
    if pmid==True:
        pmid_text,abstract_text = re.split("\|\|",abstract_text)
    sents = tokenizer.sent_tokenize(abstract_text, mask = True)
    out_text = []
    out_preds= []
    for sent in sents:
        words = sent.split()

    if pmid == True:
        return out_text, out_preds,pmid_text
    else:
        return out_text, out_preds
