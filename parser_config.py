import os,sys

class Config():
    data_dir = "pretrained_model/span"
    pretrain_dir = "pretrained_model/span/result_wpre_set6_clean"
    infile_dir="test.txt"
    outfile_dir = "test.xml"
    use_UMLS = 0
    UMLS_dir ="" # where your UMLS (ex. MRCONSO.RRF) is installed
    QuickUMLS_dir = "/home/tk2624/tools/QuickUMLS" # where your QuickUMLS data is intalled
