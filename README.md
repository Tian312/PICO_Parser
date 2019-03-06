# PICO_Parser

**Parse PubMed abstracts following PICO framework to standarize PICO elements.**  

* Author: Tian Kang (tk2624@cumc.columbia.edu)  
* Affiliation: Department of Biomedical Informatics, Columbia Univerisity ([Dr. Chunhua Weng](http://people.dbmi.columbia.edu/~chw7007/)'s lab)  
 
#### **UPDATE March 6th, 2019: json formatted output is available now**   
* ongoing: simplification of entities; relations;  


## Usage  

#### PICO Element with attributes in XML   
1.  Install `requirements.txt`
2.  If you want to use UMLS to standardize entities, please install ['UMLS'](https://www.nlm.nih.gov/research/umls/licensedcontent/umlsknowledgesources.html) and ['QuickUMLS'](https://github.com/Georgetown-IR-Lab/QuickUMLS) locally  
3.  Edit `parser_config.py` to customize your own diretories and installation  
4.  Run `python Phase1_NER_predict.py` to start parsing  

#### Clustering parsed PICO elements to represent study design    
1. Download context vector pretrained in all pubmed abstracts from 1990-2019 (downlaod link in [cluster/model/download.txt](https://github.com/Tian312/PICO_Parser/blob/master/cluster/model/download.txt))   
2. Extract 3 files and put them under cluster/model  
3. TO BE CONTINUED    

## Exmample  
**JSON**
**Input** `example.txt` contain over 70+ abstracts with methods sections
**Parsing results**  folder `example_json_out`
```json
{
  "pmid": "11264545",
  "sentences": {
    "sent_1": {
      "Section": "",
      "text": "METHODS AND RESULTS : To determine the relative power of radiographic     heart measurements for predicting outcome in dilated cardiomyopathy , we retrospectively studied 88 adult patients with chest radiographs     obtained within 35 days of echocardiography .",
      "entities": {
        "entity_1": {
          "text": "radiographic heart measurements",
          "class": "Outcome",
          "n    egation": 0,
          "UMLS": "C0018787:heart,C1306645:radiograph,",
          "index": 1,
          "start": 10
        },
        "entity_2": {
          "text": "predicting outcome",
          "class": "    Outcome",
          "negation": 0,
          "UMLS": "",
          "index": 2,
          "start": 14
        },
        "entity_3": {
          "text": "dilated cardiomyopathy",
          "class": "Participant",
          "nega    tion": 0,
          "UMLS": "C0007193:dilated cardiomyopathy,",
          "index": 3,
          "start": 17
        },
        "entity_4": {
          "text": "chest radiographs",
          "class": "Participant",
          "negation": 0,
          "UMLS": "C1306645:radiographs,C0817096:chest,",
          "index": 4,
          "start": 27
        },
        "entity_5": {
          "text": "echocardiography",
          "c    lass": "Participant",
          "negation": 0,
          "UMLS": "C0013516:echocardiography,",
          "index": 5,
          "start": 34
        }
      },
      "relations": {}
    },
    "sent_2": {
      "Section    ": "",
      "text": "Standard radiographic variables were measured for each patient , and the cardiothoracic ( CT ) ratio , frontal cardiac area     , and volume were calculated .",
      "entities": {
        "entity_6": {
          "text": "Standard radiographic variables",
          "class": "Outcome",
          "negation": 0,
          "    UMLS": "C0038137:Standard,C1306645:radiograph,",
          "index": 1,
          "start": 0
        },
        "entity_7": {
          "text": "cardiothoracic ( CT ) ratio",
          "class": "Outcome",
          "negation": 0,
          "UMLS": "",
          "index": 2,
          "start": 11
        },
        "entity_8": {
          "text": "frontal cardiac area",
          "class": "Outcome",
          "negation": 0,
          "UMLS": "C0018787:cardiac,",
          "index": 3,
          "start": 17
        },
        "entity_9": {
          "text": "volume",
          "class": "Outcome",
          "negation": 0,
          "UMLS": "",
          "inde    x": 4,
          "start": 22
        }
      },
      "relations": {}
    }
  }
}

```

**XML**
**Input**  `test.txt`  
**Parsing results** `temp.xml`  

    A double-blind crossover comparison of pindolol , metoprolol , atenolol and labetalol in mild to moderate hypertension . 1     This study was designed to compare in a double-blind randomized crossover trial , atenolol , labetalol , metoprolol and pindolol . Considerable differences in dose ( atenolol 138 +/- 13 mg daily ; labetalol 308 +/- 34 mg daily ; metoprolol 234 +/- 22 mg daily ; and pindolol 24 +/-2 mg daily were required to produce similar antihypertensive effects . 
  
```xml
<abstract>
		<sent>
			<text>A double-blind crossover comparison of pindolol , metoprolol , atenolol and labetalol in mild to moderate hypertension .</text>
			<entity class='Intervention' UMLS='C0031937:pindolol' index='T1' start='5'> pindolol </entity>
			 <entity class='Intervention' UMLS='C0025859:metoprolol' index='T2' start='7'> metoprolol </entity>
			 <entity class='Intervention' UMLS='C0004147:atenolol' index='T3' start='9'> atenolol </entity>
			 <entity class='Intervention' UMLS='C0022860:labetalol' index='T4' start='11'> labetalol </entity>
			 <entity class='Participant' UMLS='C0020538:hypertension' index='T5' start='13'> mild to moderate hypertension </entity>
		</sent>
		<sent>
			<text>1 This study was designed to compare in a double-blind randomized crossover trial , atenolol , labetalol , metoprolol and pindolol .</text>
			<entity class='Intervention' UMLS='C0004147:atenolol' index='T6' start='14'> atenolol </entity>
			 <entity class='Intervention' UMLS='C0022860:labetalol' index='T7' start='16'> labetalol </entity>
			 <entity class='Intervention' UMLS='C0025859:metoprolol' index='T8' start='18'> metoprolol </entity>
			 <entity class='Intervention' UMLS='C0031937:pindolol' index='T9' start='20'> pindolol </entity>
		</sent>
		<sent>
			<text>Considerable differences in dose ( atenolol 138 +/- 13 mg daily ; labetalol 308 +/- 34 mg daily ; metoprolol 234 +/- 22 mg daily ; and pindolol 24 +/-2 mg daily were required to produce similar antihypertensive effects .</text>
			<attribute class='modifier' index='T10' start='1'> differences </attribute>
			 <entity class='Intervention' UMLS='C0004147:atenolol' index='T11' start='5'> atenolol </entity>
			 <attribute class='measure' index='T12' start='6'> 138 +/- 13 mg daily </attribute>
			 <entity class='Intervention' UMLS='C0022860:labetalol' index='T13' start='12'> labetalol </entity>
			 <attribute class='measure' index='T14' start='13'> 308 +/- 34 mg daily </attribute>
			 <entity class='Intervention' UMLS='C0025859:metoprolol' index='T15' start='19'> metoprolol </entity>
			 <attribute class='measure' index='T16' start='20'> 234 +/- 22 mg daily </attribute>
			 <entity class='Intervention' UMLS='C0031937:pindolol' index='T17' start='27'> pindolol </entity>
			 <attribute class='measure' index='T18' start='28'> 24 +/-2 mg daily </attribute>
			 <entity class='Outcome' UMLS='C0003364:antihypertensive' index='T19' start='37'> antihypertensive effects </entity>
		</sent>
</abstract>   

```

## Reference

- Parser achitecture is adapted from my previous project of eligibility criteria parser [`EliIE`](https://github.com/Tian312/EliIE).   
- LSTM-CRF scritps modified from [EBM-NLP](https://github.com/bepnye/EBM-NLP/tree/master/acl_scripts/lstm-crf)   
