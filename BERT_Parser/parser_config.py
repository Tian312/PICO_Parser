import os,sys

# example command: 
#python bluebert/run_bluebert_ner_predict.py --data_dir=test/txt --output_dir=test/json

#--------------------- MODIFY -------------------------------------#
#  Please modifie the following parameters before parsing.

class Config():

    # Base BERT config
    max_seq_length=128 
    vocab_file="bert_init_models/vocab.txt"  
    bert_config_file= "bert_init_models/bert_config.json"  
    init_checkpoint= "bert_init_models/bert_model.ckpt"
     
    # NCBI blueBERT config
    bluebert_dir = "bluebert_pretrained"
    
    # UMLS config
    use_UMLS = 0 # 0 represents not using UMLS
    QuickUMLS_git_dir = "/home/tk2624/tools/QuickUMLS-master"
    QuickUMLS_dir = "/home/tk2624/tools/QuickUMLS" # where your QuickUMLS data is intalled
    if not os.path.exists("QuickUMLS"):
        command = "ln -s "+ QuickUMLS_git_dir + " QuickUMLS"
        os.system (command)
    
