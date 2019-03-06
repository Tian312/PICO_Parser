# coding: utf-8

import os, re,  string
import sys,codecs
from parser.umls_tagging import get_umls_tagging
import json

def generate_XML(NERxml_dir,matcher,use_UMLS = 0,crfresult_dir="temp.conll"):
    crfresult_input = codecs.open(crfresult_dir,'r')
    NERxml_output=codecs.open(NERxml_dir,'w')
    if use_UMLS ==0:
        sents,entities_result=conll2txt_no_umls(crfresult_input)
    else:
        sents,entities_result=conll2txt(crfresult_input,matcher)
    entity_lists=['Participant','Intervention','Outcome']
    attribute_lists=['modifier','measure']
    NERxml_output.write("<?xml version=\"1.0\"?>")
    NERxml_output.write("<root>\n\t<abstract>\n")
    j=0

    for index,(sent, entities_forsent) in enumerate(zip(sents, entities_result)):
        if sent == "":
            continue
        if sent == "END":
            NERxml_output.write("\t</abstract>\n\n\t<abstract>\n")
            continue
        clean_sent=clean_txt(sent)

        pattern='class=\'(\w+)\''
        entities=entities_forsent.split('\n\t\t')
        new_entities=[]
        for e in entities:
            if e =='':
                new_entities.append('\n')
                continue
            match=re.search(pattern,e)
           
            if match.group(1) in attribute_lists:

                p1='\<entity'
                p2='entity\>'
                new=re.sub(p1,'<attribute',e)
                new=re.sub(p2,'attribute>',new)
                new_entities.append(new)
            else:

                new_entities.append(e)
        entities="\n\t\t\t".join(new_entities)
        entities=re.sub("\t\t\t\n$","",entities)
    
        NERxml_output.write("\t\t"+"<sent>\n"+"\t\t\t<text>"+clean_sent+"</text>\n")
        NERxml_output.write("\t\t\t"+entities)
        NERxml_output.write("\t\t"+"</sent>\n")
        j+=1
    NERxml_output.write("\t</abstract>\n</root>\n")
    rm_command = "rm "+crfresult_dir
    #os.system(rm_command)


def generate_json(out_text, out_preds,matcher,pmid="",sent_tags=[],entity_tags=["Participant","Intervention","Outcome"],attribute_tags=["measure","modifier","temporal"],relation_tags=[]):
    #abstract{ pmid; sent{section}; {entity{class;UMLS;negation;Index;start};relation{class;entity1;entity2}}
    results = {}
    results["pmid"] = pmid
    results["sentences"]={}
    #json_r=json.dumps(results)
    
    sent_id = 0
    entity_id=0
    attribute_id=0
    
    for sent, pred in zip(out_text, out_preds):
        sent_id+= 1
        sent_header = "sent_"+str(sent_id)
        results["sentences"][sent_header]={"Section":"","text":" ".join(sent),"entities":{},"relations":{}}
        
        indices_B = [i for i, x in enumerate(pred) if x.split("-")[0] == "B"]
        term_index = 1
        
        for ind in indices_B:   
            
            ''' retrieve all info for Enities and Attributes:
            "entity1":{                       
                       "text":"infliximab",
                       "class":"Intervention",
                       "negation":"0",
                        "UMLS":"",
                        "index":"T1",
                        "start":"19" 
            }
            '''
            entity_class = pred[ind].split("-")[1] # class
            if entity_class in entity_tags:        # header
                entity_id+=1
                entity_header = "entity_"+str(entity_id)
            else:
                attribute_id+=1
                entity_header = "attribute_"+str(attribute_id)
            start = ind
            inds=[]
            while(pred[start] !="O" or start > len(pred)):
                inds.append(start)
                start+=1
            c = [ sent[i] for i in inds]
            term_text = " ".join(c) # text
            #============= Negation =====================
            neg = 0
            
            #==============Negation END===================
            
            
            #============== UMLS encoding ================
            taggings = get_umls_tagging(term_text, matcher)
            umls_tag=""
            if taggings:
                for t in taggings:
                    umls_tag = umls_tag +str(t["cui"])+":"+str(t["term"])+","
            #===============UMLS EDN =====================
            
            
            results["sentences"][sent_header]["entities"][entity_header]={"text":term_text,"class":entity_class,"negation":neg, "UMLS":umls_tag,"index":term_index,"start":ind }
            term_index +=1
     
        #=============Relations ======================
        '''
         "relations":{
            "rel1":{
                "class":"has_value",
                "left":"T1",
                "right":"T2"
                }
            }
        }
        '''    
        #============END =============================
        
        
    json_r=json.dumps(results)
    return json_r

''' TEST
from QuickUMLS.quickumls import QuickUMLS
matcher = QuickUMLS(parser_config.QuickUMLS_dir,threshold=0.8)      
out_text = [["a","bad","guy","is","having","heart","attack","and","hr","<","10","."]]
out_preds = [["O","O","O","O","O","B-Intervention","I-Intervention","O","B-Outcome","I-Outcome","I-Outcome","O"]]
print(generate_json(out_text, out_preds, matcher))
'''
    