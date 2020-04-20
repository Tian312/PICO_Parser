import os, re,  string
import sys,codecs

def get_umls_tagging(text,matcher):
    info = matcher.match(text, best_match=True, ignore_syntax=False)
    taggings=[]
    if len(info) == 0:
        return None
    for one_c in info:

        one_c = one_c[0]
    
        result = {"cui":one_c["cui"],"term":one_c["term"]}
        taggings.append(result)
    return taggings

      
