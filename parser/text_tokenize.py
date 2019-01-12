# implement NLP-Cube https://github.com/adobe/NLP-Cube for sent and word tokenization, and lemma
from cube.api import Cube       # import the Cube object
cube=Cube(verbose=False)         # initialize it
cube.load("en")    

import re

class mytokenizer:
    def sent_tokenize(self, text,word_tokenize = True):
        text = re.sub(";",".",text)
        sentences=cube(text)
        new_sentences = []
        '''
        for sentence in sentences:
            for entry in sentence:
                print(str(entry.index)+"\t"+entry.word+"\t"+entry.lemma+"\t"+entry.upos+"\t"+entry.xpos+"\t"+entry.attrs+"\t"+str(entry.head)+"\t"+str(entry.label)+"\t"+entry.space_after)
               
            print("==")
        '''
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
    
    def word_tokenize(self,text):
        sentences=cube(text)
        new_words = []
        for sent in sentences:
            for entry in sent: 
                lemmas= entry.word
                
                new_words.append(lemmas)
        return new_words


    
    
    
    
'''       
text = "Treatment of hypertensive and hypercholesterolaemic patients in general practice. The effect of captopril, atenolol and pravastatin combined with life style intervention. The objective of the present study was prospectively and randomly to evaluate the role of L-arginine in improving uterine and follicular Doppler flow and in improving ovarian response to gonadotrophin in poor responder women. A total of 34 patients undergoing assisted reproduction was divided in two groups according to different ovarian stimulation protocols: (i) flare-up gonadotrophin-releasing hormone analogue (GnRHa) plus elevated pure follicle stimulating hormone (pFSH) (n = 17); and (ii) flare-up GnRHa plus elevated pFSH plus oral L-arginine (n = 17). During the ovarian stimulation regimen, the patients were submitted to hormonal (oestradiol and growth hormone), ultrasonographic (follicular number and diameter, endometrial thickness) and Doppler (uterine and perifollicular arteries) evaluations. Furthermore, the plasma and follicular fluid concentrations of arginine, citrulline, nitrite/nitrate (NO2-/NO3-), and insulin-like growth factor-1 (IGF-1) were assayed. All 34 patients completed the study. In the L-arginine treated group a lower cancellation rate, an increased number of oocytes collected, and embryos transferred were observed. In the same group, increased plasma and follicular fluid concentrations of arginine, citrulline, NO2-/NO3-, and IGF-1 was observed. Significant Doppler flow improvement was obtained in the L-arginine supplemented group. Three pregnancies were registered in these patients. No pregnancies were observed in the other group. It was concluded that oral L-arginine supplementation in poor responder patients may improve ovarian response, endometrial receptivity and pregnancy rate."
t = mytokenizer()
sents= t.sent_tokenize(text)
for s in sents:
    print(s,"==")
'''
