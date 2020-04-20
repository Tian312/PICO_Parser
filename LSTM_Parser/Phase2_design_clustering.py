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
        sent_text = sent.find("text").text
        print (sent_text)
        
        for entity in sent.findall("entity"):
            entity_text = re.sub("^\s+|\s+$","",entity.text).lower()
            pos = entity.attrib["start"]
            print(pos,entity_text)
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
    model = model_reader.model

    #parse PICO tree from NER step
    tree = ET.ElementTree(file=input_dir)
    root = tree.getroot()
    for abstract in root:
        design = get_studydesign(abstract,entity_list)
        print (design)
        '''
        for key1 in design.keys():
            print (key1)
            print (design[key1],"\n")

        cluster={"Participant":{},"Intervention":{},"Outcome":{}}
        cluster={}

            # clustering
        primary_list = design["Primary"]["Outcome"]
        detail_list = design["Details"]["Outcome"]
        for p in primary_list:
            cluster[p] = []
        for detail in detail_list:
            max_term,max_sim,sim_dic = get_most_similar(detail,primary_list,w,word2index)             
            cluster[max_term].append(detail)
            print (detail,sim_dic)
            #print (design,"\n")
        '''
if __name__ == '__main__': main()