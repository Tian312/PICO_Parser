# coding: utf-8

import warnings,time
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)

from model.ner_model import NERModel
from model.config import Config
from parser import txtconll,format_predict

from parser_config import Config as parser_Config



time0 = time.time()
config =Config()
model = NERModel(config)
model.build()
model.restore_session(config.dir_model)
time1 = time.time()
parser_config = parser_Config()

if parser_config.use_UMLS >0:

    #QuickUMLS matcher
    from QuickUMLS.quickumls import QuickUMLS
    matcher = QuickUMLS(parser_config.QuickUMLS_dir,threshold=0.8)
else:
    matcher = None
print ("loading model...",time1-time0)



def main():
    #predict
    input= parser_config.infile_dir
    time2 = time.time()
    format_predict.get_predict(input, model)
    print ("\nprediting..."),time2-time1   
    
    #generate xml   
    output_xml = parser_config.outfile_dir
    txtconll.generate_XML(output_xml,matcher,parser_config.use_UMLS)
    time3 = time.time()
    print ("formatting xml...")
    print ("saving xml file in "+output_xml+"\n"+ str(time3-time0)+" s in total...")

if __name__ == '__main__': main()