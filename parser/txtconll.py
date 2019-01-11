
# coding: utf-8

# In[ ]:


import os, re,  string
import sys,codecs
from parser.umls_tagging import get_umls_tagging

# In[ ]:


# Feb. 2016
# This function is to transform format between raw text and  CoNLL format

def txt2conll(text_line,label_sign):

# clean text; punctuation problems

    text_line=text_line.strip()
    if re.search("[\:|\?]\.$",text_line):
        text_line=re.sub("\.$","",text_line)
    end=re.search("\.(\s)",text_line)
    if end:
        text_line=re.sub("\.\s"," ."+end.group(1),text_line)

 # text could be raw text or labelled text from annotation
    if label_sign == 0: # is the raw text without the annotation, only need to return one term list
        term_list=text_line.split()
        return(term_list)
    if label_sign == 1: # is the labelled text, need to return two lists: term list & label list
        raw_list=text_line.split()
        term_list=list()
        label_list=list()
        index_list=list()
        for term in raw_list:
            if re.search('__[B|I]',term):
                #print term
                info=term.split("__")
                term_list.append(info[0])
                label_list.append(info[1])
                index_list.append(info[2])
            else:
                term_list.append(term)
                label_list.append("O")
                index_list.append("O")
        return term_list,label_list,index_list


# In[ ]:


def conll2txt (conll_file,matcher):
    attribute_lists=['modifier','measure']
    entity_lists=['Participant','Intervention','Outcome']
    sent_flag=0 #

    term_flag=0 #
    term_label="O"
    entitiy_count=0;
    sents=[]
    entity=[]
    raw_terms=[]
    terms=[]
    i=0

    for line in conll_file:
        line=line.strip()
        line=re.sub(">>NCT","##NCT",line)
        line=re.sub("<=","smaller_equal_than",line)
        line=re.sub(">=","larger equal_than",line)
        line=re.sub("<","smaller_than",line)
        line=re.sub(">","larger_than",line)



        if not line.strip():

            if term_flag>0:
                last_label="</entity>\n\t\t"
                terms.append(last_label)
                term_flag=0
            new_line=" ".join(raw_terms)
            concept=" ".join(terms)
            
            #####======= ADD UMLS tagging #===============
            new_concept = [] 
            for line_concept in concept.split("\n"):
                #=== temporary rule:=====
                line_concept =re.sub(r"\( group I+ \)","",line_concept)
                #==========================
                match =re.search(r">\s+(.*)\s+<",line_concept)
                if match:
                    if re.search("measure|modifier",line_concept):
                        new_concept.append(line_concept)
                    else:
                        text = match.group(1)
                        
                        #=== temporary rule:=====
                        
                        match =re.search("^\s+$|group\s+I",text)
                        if match:
                            continue
                        #==========================
                        taggings = get_umls_tagging(text, matcher)
                        umls_tag = "UMLS=\'"
                        if taggings is not None:
                            for t in taggings:
                                umls_tag = umls_tag +str(t["cui"])+":"+str(t["term"])+","
                        umls_tag = re.sub(",$","",umls_tag)
                        umls_tag = umls_tag+"\' index"                   
                        new_line_concept =re.sub("index",umls_tag,line_concept)
                        new_concept.append(new_line_concept)
                else:
                    new_concept.append(line_concept)

            new_concept = "\n".join(new_concept)
            #=======================================
            
            entity.append(new_concept)

            sents.append(new_line)
            
            terms=[]
            term_flag=0
            raw_terms=[]
            i=0
        else:

            if re.search("##NCT",line):
                line=re.sub("##","",line)

            #line=re.sub("\:","",line)
            line=re.sub("&","and",line)
            info=line.split()
            word=info[0]


            raw_terms.append(word)
            is_entity=re.search('B\-',info[-1])
            if is_entity:
                entitiy_count+=1
                index="T"+str(entitiy_count)
            else:
                index=" "
            label=info[-1]
            raw_index=info[-2]

            if term_flag > 0 and re.search("^B",label):
                
                last_label="</entity>\n\t\t"
                terms.append(last_label)
                term_flag=0


            if label=="O":
                if term_flag>0:
                    word="</entity>\n\t\t"
                    term_flag=0
                    terms.append(word)

            else:
                term_flag+=1
                            
                
                if re.search("^B\-",label):
                    match=re.search("^B\-(.*)$",label)
                    term_label=match.group(1)
                    umls=""

                    word="<entity "+ "class="+"\'"+term_label+"\'"+" index="+"\'"+index+"\'"+" start="+"\'"+str(i)+"\'"+"> "+word
                terms.append(word)
            i+=1
    if term_flag > 0:
        last_label = "</entity>\n\t\t"
        terms.append(last_label)
        term_flag = 0
    new_line = " ".join(raw_terms)
    concept = " ".join(terms)
    
    
    #####======= ADD UMLS tagging #===============
            
    new_concept = [] 
    for line_concept in concept.split("\n"):
        match =re.search(r">\s+(.*)\s+<",line_concept)
        if match:
            if re.search("measure|modifier",line_concept):
                new_concept.append(line_concept)
            else:
                text = match.group(1)
                
                #==== temp rules =====
                match =re.search("group\sI+",text)        
                if match:
                    continue
                #===============

                taggings = get_umls_tagging(text, matcher)
                umls_tag = "UMLS=\'"
                if taggings:
                    for t in taggings:
                        umls_tag = umls_tag +str(t["cui"])+":"+str(t["term"])+","
                umls_tag = re.sub(",$","",umls_tag)
                umls_tag = umls_tag+"\' index"
                        
                new_line_concept =re.sub("index",umls_tag,line_concept)
                new_concept.append(new_line_concept)
        else:
            new_concept.append(line_concept)                   
    new_concept = "\n".join(new_concept)
    #=======================================
    
    entity.append(new_concept)
    sents.append(new_line)
    terms = []
    term_flag = 0
    raw_terms = []
    i = 0
    return sents,entity

def conll2txt_no_umls(conll_file):
    attribute_lists=['modifier','measure']
    entity_lists=['Participant','Intervention','Outcome']
    sent_flag=0 #

    term_flag=0 #
    term_label="O"
    entitiy_count=0;
    sents=[]
    entity=[]
    raw_terms=[]
    terms=[]
    i=0

    for line in conll_file:
        line=line.strip()
        line=re.sub(">>NCT","##NCT",line)
        line=re.sub("<=","smaller_equal_than",line)
        line=re.sub(">=","larger equal_than",line)
        line=re.sub("<","smaller_than",line)
        line=re.sub(">","larger_than",line)



        if not line.strip():

            if term_flag>0:
                last_label="</entity>\n\t\t"
                terms.append(last_label)
                term_flag=0
            new_line=" ".join(raw_terms)
            concept=" ".join(terms)
           
            # ========== temp rules ============
            concepts=concept.split("\n")
            concepts =re.sub(r"\( group I+ \)","",concepts)

            concept_new = []
            for c in concepts:
                match = re.search("^\s+$|group\s+I+",c)
                if match:
                    continue
                else:
                    concept_new.append(c)
            concept = "\n".join(concept_new)
            # =================================

            entity.append(concept)
            
            sents.append(new_line)
            
            terms=[]
            term_flag=0
            raw_terms=[]
            i=0
        else:

            if re.search("##NCT",line):
                line=re.sub("##","",line)

            line=re.sub("\:","",line)
            line=re.sub("&","and",line)
            info=line.split()
            word=info[0]


            raw_terms.append(word)
            is_entity=re.search('B\-',info[-1])
            if is_entity:
                entitiy_count+=1
                index="T"+str(entitiy_count)
            else:
                index=" "
            label=info[-1]
            raw_index=info[-2]

            if term_flag > 0 and re.search("^B",label):
                
                last_label="</entity>\n\t\t"
                terms.append(last_label)
                term_flag=0


            if label=="O":
                if term_flag>0:
                    word="</entity>\n\t\t"
                    term_flag=0
                    terms.append(word)

            else:
                term_flag+=1
                            
                
                if re.search("^B\-",label):
                    match=re.search("^B\-(.*)$",label)
                    term_label=match.group(1)
                    umls=""

                    word="<entity "+ "class="+"\'"+term_label+"\'"+" index="+"\'"+index+"\'"+" start="+"\'"+str(i)+"\'"+"> "+word
                terms.append(word)
            i+=1
    if term_flag > 0:
        last_label = "</entity>\n\t\t"
        terms.append(last_label)
        term_flag = 0
    new_line = " ".join(raw_terms)
    concept = " ".join(terms)
    entity.append(concept)
    sents.append(new_line)
    terms = []
    term_flag = 0
    raw_terms = []
    i = 0
    return sents,entity


def clean_txt(tagged_text): # from tagged_text to raw_text
    clean_text=re.sub("\<\w+\ >","",tagged_text)
    clean_text=re.sub("\<\/\w+\>","",clean_text)
    return clean_text




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
