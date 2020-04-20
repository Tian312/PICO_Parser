# coding: utf-8

import warnings,time,os,sys,codecs
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from model.ner_model import NERModel
from model.config import Config
from parser import txtconll,format_predict,formalization
from parser_config import Config as parser_Config
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ['CUDA_VISIBLE_DEVICES']='1,2,3'
import warnings
if not sys.warnoptions:
        warnings.simplefilter("ignore")





time0 = time.time()
config =Config()
model = NERModel(config)
model.build()
model.restore_session(config.dir_model)
time1 = time.time()
parser_config = parser_Config()



matcher = None
if parser_config.use_UMLS >0:
    #QuickUMLS matcher
    from QuickUMLS.quickumls import QuickUMLS
    matcher = QuickUMLS(parser_config.QuickUMLS_dir,threshold=0.8)

print ("\nloading model...",time1-time0)



def main():
    #predict
    input= parser_config.infile_dir
    time2 = time.time()
    from parser import text_tokenize
    tokenizer = text_tokenize.mytokenizer()
    infile = codecs.open(input,"r") # assume each line is one abstract: pmid||abstracttext
    outdir = parser_config.outjson_dir
    exception_dir = os.path.join(outdir+"/exceptionlist.txt")
    except_out = codecs.open(exception_dir,"w")
    if not os.path.exists(outdir):
        try:
            createdir= "mkdir "+outdir
            os.system(createdir)
        except:
            print("DIR ERROR! Unable to create this directory!")
            
    
    count = 0
    for line in infile:
        line= line.rstrip()
        try:
            # Named Entity Recognition
            out_text, out_preds,pmid = format_predict.get_predict(line,model, tokenizer,pmid=True)
            outfile_dir= codecs.open(os.path.join(outdir,pmid+".json"),"w")

            # format for json object
            json_out = formalization.generate_json(out_text, out_preds,matcher,pmid,sent_tags=[],entity_tags=["Participant","Intervention","Outcome"],attribute_tags=["measure","modifier","temporal"],relation_tags=[])
            outfile_dir.write(json_out)
        except:
            except_out.write(line+"\n")
        # TIME prediction 
        if count%100 ==0:
            new_time = time.time()
            cost = new_time-time2
            cost_m,cost_s=divmod(cost, 60)
            
            print ("processing",count,"th abstracts... cost", cost_m," min in total...")
        count+=1

    time3 = time.time()
    print ("formatting xml...")
    print ("saving json file in "+outdir+"\n"+ str(time3-time0)+" s in total...")


if __name__ == '__main__': main()
