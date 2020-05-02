# implement NLP-Cube https://github.com/adobe/NLP-Cube for sent and word tokenization, and lemma
from cube.api import Cube       # import the Cube object
cube=Cube(verbose=False)         # initialize it
cube.load("en")    
from rusenttokenize import ru_sent_tokenize

import re

class mytokenizer:
    def bracket_mask(self,text):
        
        text_temp = re.sub("\[","<<",text)
        text_temp = re.sub("\]",">>",text_temp)
        text_temp = re.sub("\+","===",text_temp)
        parenthesis = re.findall("\([^()]*\)",text_temp)
        bracket = re.findall("\[[^\[\]]*\]",text)
        if not parenthesis and not bracket:
            return text
            
        for p in parenthesis:
            p = re.sub("[\(\)]","",p)
            parts = [re.sub("\.\s*$","",s) for s in ru_sent_tokenize(p)]
            new_p=" ; ".join(parts)
            #print ("P:", p)
            #print("new_p:",new_p,"\n")
            #print("text_temp:",text_temp)
            text_temp = re.sub(p,new_p,text_temp)

    
        text = re.sub("<<","[",text_temp)
        text = re.sub(">>","]",text)
        text = re.sub("===","+",text)
        bracket = re.findall("\[[^\[\]]*\]",text)

        for p in bracket:
            p = re.sub("[\[\]]","",p)
            parts = [re.sub("\.\s*$","",s) for s in ru_sent_tokenize(p)]
            #parts = [re.sub("\.\s*$","",re.sub("\[","\[", re.sub("\]","\]",s))) for s in ru_sent_tokenize(p)]
            new_p=" ; ".join(parts)
            #print ("OLD:",p,"\n","NEW:",new_p)
            text = re.sub(p,new_p,text)
        
        return text


    def word_tokenize(self,text):
        sentences=cube(text)
        new_words = []
        for sent in sentences:
            for entry in sent: 
                lemmas= entry.word
                
                new_words.append(lemmas)
        return new_words
    
    def sent_tokenize_old(self, text,word_tokenize = True,bracket_mask = True):
        #text = re.sub(";",".",text)
        text = " ".join(self.word_tokenize(text))
        sentences=cube(text)
        new_sentences = []
        for sent in sentences:
            lemmas = ""
            for entry in sent: 
                lemmas += entry.word
                # now, we look for a space after the lemma to add it as well
                if word_tokenize == False:
                    if not "SpaceAfter=No" in entry.space_after:
                        lemmas += " "
                else:
                    lemmas += " "
            new_sentences.append(lemmas)
        return new_sentences
    
    def sent_tokenize(self, text,word_tokenize = True, mask = True):
        #text = re.sub(";",".",text)
        
        if mask == True:
            text_after_mask = self.bracket_mask(text)
        else:
            text_after_mask = text
        text_after_word = " ".join(self.word_tokenize(text_after_mask))
        sentences=ru_sent_tokenize(text_after_word)
        return sentences

#test parentathis masking
'''
tok = mytokenizer()
text = "Interventions: Hospitals provided either usual care (control group. n = 10 066 participants [step 0: n = 2915; step 1: n = 2649; step 2: n = 2251; step 3: n = 1422; step 4; n = 829; step 5: n = 0]) or care using a quality improve    ment tool kit (intervention group; n = 11 308 participants [step 0: n = 0; step 1: n = 662; step 2: n = 1265. step 3: n = 2432. step 4: n = 3214. step 5: n = 3735]) that consisted of audit and     feedback, checklists, patient education materials, and linkage to emergency cardiovascular care and quality improvement training. Main Outcomes and Measures: The primary outcome was the compo    site of all-cause death, reinfarction, stroke, or major bleeding using standardized definitions at 30 days."


print ("ORI:",text)
sents = tok.sent_tokenize(text)
for i in sents:
    print("\n"+i+"\n")
'''
