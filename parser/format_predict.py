import sys,re,os,codecs
import nltk
from rusenttokenize import ru_sent_tokenize


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


def get_predict(input_dir,model,tokenizer, output_dir="temp.conll"):

    infile=codecs.open(input_dir,'r').read()
    outfile=codecs.open(output_dir,'w')
    abs = infile.split("\n")
    last = 0-len(abs)
    j = -1
    while len(abs)>1:
        if abs[j].rstrip() == "":
            del abs[j]
        else:
            break
    for i in range(len(abs)):
        if abs[i] == "":
            continue
        #sent_text = tokenizer.sent_tokenize(abs[i].rstrip())
        sent_text = ru_sent_tokenize(abs[i].rstrip())
        for sent in sent_text:
            #print ("\n",sent)
            skip = re.search("background|implication|match",sent,re.IGNORECASE)
            #words = sent.split()
            words = tokenizer.word_tokenize(sent)
            print (sent)
            print (words)
            if skip:
                preds = ["O"]*len(words)
            else:
                preds=model.predict(words)
                #print (preds)
                preds=check_IOB(preds)
                #print (preds,"===")
            for (w, p) in zip(words, preds):
                outfile.write(w+"\t"+p+"\n")
            outfile.write("\n")
        if i < len(abs)-1:
            outfile.write("END\tO\n\n")

