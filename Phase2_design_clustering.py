import os,codecs,re
import sys
from xml.etree import ElementTree as ET
from parser_config import Config as parser_Config
from cluster.get_term_similarity import get_most_similar
parser_config = parser_Config()

input_dir=parser_config.outxml_dir
output_dir=parser_config.outcluster_dir
entity_list = ["Participant","Intervention","Outcome"]
cluster_cutoff = 0 
def get_studydesign(ab_element,entity_list):
    sent_index = 0
    
    studydesign={"Primary":{},"Details":{}}
    for key in studydesign.keys():
        for entity in entity_list:
            studydesign[key][entity]=[]
    sents = ab_element.findall("sent")
    sent_total = len(sents)
    print("===",sent_total)
            
    for sent in ab_element.findall("sent"):
        sent_index=sent_index+1
        for entity in sent.findall("entity"):
            entity_text = re.sub("^\s+|\s+$","",entity.text)
            
            if sent_index <=2 or sent_index == sent_total:
                if entity_text not in studydesign["Primary"][entity.attrib['class']]:
                    studydesign["Primary"][entity.attrib['class']].append(entity_text)
            else:
                if entity_text not in studydesign["Details"][entity.attrib['class']]:
                    studydesign["Details"][entity.attrib['class']].append(entity_text)
    return studydesign   
                

def main():
    infile = parser_config.outxml_dir
    outfile = codecs.open(parser_config.outcluster_dir,"w")
    
    
    #relaod trained c2v embeddings    
    model_param_file = parser_config.c2v_model_param_file_dir
    model_param_file = "/home/tk2624/gitrepos/context2vec/models/pubmed.c2v.200d.model.params"
    from cluster.model_reader import ModelReader
    model_reader = ModelReader(model_param_file)
    w = model_reader.w
    word2index = model_reader.word2index
    
    #parse PICO tree from NER step
    tree = ET.ElementTree(file=input_dir)
    root = tree.getroot()
    for abstract in root:
        design = get_studydesign(abstract,entity_list)
        outfile.write("Abstract\n")
        cluster={"Participant":{},"Intervention":{},"Outcome":{}}
        for entity in entity_list:
            # clustering
            
            primary_list = design["Primary"][entity]
            detail_list = design["Details"][entity]
            print (entity)
            print (primary_list)
            print (detail_list)
            for p in primary_list:
                cluster[entity][p] = []

            for detail in detail_list:

                
                max_term,max_sim,sim_dic = get_most_similar(detail,primary_list,w,word2index)             
                cluster[entity][max_term].append(detail)
                
                #print (sim_dic)
                #print (design,"\n")
        print (cluster)

if __name__ == '__main__': main()