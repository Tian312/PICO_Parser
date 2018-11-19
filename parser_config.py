import os,sys

class Config():
    
    # Parser basic config
    data_dir = "pretrained_model/span"
    pretrain_dir = "pretrained_model/span/result_wpre_set6_clean"
    infile_dir="test.txt"
    outxml_dir = "test.xml"
    
    # UMLS config
    use_UMLS = 1 # 0 represents not using UMLS
    QuickUMLS_git_dir = "/home/tk2624/tools/QuickUMLS-master"
    QuickUMLS_dir = "/home/tk2624/tools/QuickUMLS" # where your QuickUMLS data is intalled
    command = "ln -s "+ QuickUMLS_git_dir + " QuickUMLS"
    os.system (command)
    
    #cluster config
    c2v_model_param_file_dir = "cluster/model/pubmed.c2v.200d.model.params"
    outcluster_dir = "test.txt.cluster"
    
