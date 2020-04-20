import json,os, codecs
import re


def json2ann(json_object):
    entity_index = 0
    ori_text = ""
    output = []
    for key, value in dict.items(json_object["sentences"]):
        # key: sent_1
        # value: {section:,text: entities}
        #sents_object = json.loads(value)
        #key1 = "Section";"text"; "entities"
        
        sent_text = value["text"]
        sent_text = " ".join(re.split("\s+",sent_text))
        start_index = len(ori_text)+1
        ori_text = ori_text +" "+sent_text
        #print (sent_text)
        entities = value["entities"]
        for key in entities.keys():
            entity = entities[key]
            #print (entity)
            info = []
            entity_index += 1
            tag = "T"+str(entity_index)
            info.append(tag)
            entity_text = entity["text"]
            entity_class = entity["class"]
            start = entity["start"]
            info.append(entity_class)
            words = re.split("\s+",sent_text)
            prefix = " ".join(words[:start])
            if start  == 0:
                entity_start = start_index
            else:
                entity_start = start_index + len(prefix)+1
            entity_end = entity_start + len(entity_text)
            entity_end = entity_end -1
            entity_start = entity_start -1
            info.append(str(entity_start))
            info.append(str(entity_end))
            info.append(entity_text)
            #print (entity_text)
            #print (entity_start ,entity_end)
            #print (ori_text[entity_start:entity_end])
            #print ("\n")            
            output.append(info)
    ori_text = re.sub("^\s+","",ori_text)
    return output, ori_text

import glob
for infile in glob.glob("/home/tk2624/projects/relation_extraction/data/raw_B/json/*.json"):
  
    #infile = codecs.open(infile).read()
    file_id = re.sub("\.json","",re.split("/",infile)[-1])
    infile = codecs.open(infile).read()
    print (file_id)
    ob = json.loads(infile)
    output,ori_text = json2ann(ob)
    output_dir = "/home/tk2624/projects/relation_extraction/data/raw_B/json2brat/"
    output_ann_path = output_dir+file_id+".ann"
    output_text_path = output_dir+file_id+".txt"
    output_ann = codecs.open(output_ann_path,"w")

    for info in output:
        output_ann.write(info[0]+"\t"+info[1]+" "+info[2]+" "+info[3]+"\t"+info[4]+"\n")
    output_text = codecs.open(output_text_path,"w")
    output_text.write(ori_text)
