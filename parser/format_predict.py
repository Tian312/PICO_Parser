import sys,re,os,codecs
import nltk
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


def get_predict(input_dir,model,output_dir="temp.conll"):

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
        sent_text = nltk.sent_tokenize(abs[i].rstrip())

        for sent in sent_text:
            words = sent.split()
            preds=model.predict(words)
            preds=check_IOB(preds)
            for (w, p) in zip(words, preds):
                outfile.write(w+"\t"+p+"\n")
            outfile.write("\n")
        if i < len(abs)-1:
            outfile.write("END\tO\n\n")

