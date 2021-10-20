import os
import json
import numpy as np
import jieba
import argparse


from textrank4zh import TextRank4Keyword
from pyhanlp import *


TR4W = TextRank4Keyword()
###################
def split_to_single_dialogs(dialog_file):
    one_diag=[]
    one_diags=[]
    with open(dialog_file) as fin:
        for i,line in enumerate(fin):
            ll=line.rstrip()
            if ll!="":
                one_diag.append(ll)
            else:
                one_diags.append(one_diag)
                one_diag=[]
    return one_diags

def nlp_parser(text,tr4w=None,n_words=5,wordAttr = ["n","v","nhd"]):
   
   word2nature = []
   for term in HanLP.segment(text):
       word2nature.append(( term.word, str(term.nature)))
   word2nature = {w:n for (w,n) in word2nature}
   if tr4w is None:
      tr4w = TextRank4Keyword()
   
   tr4w.analyze(text=text, lower=True, window=2)
   res=[]
   for item in tr4w.get_keywords(n_words, word_min_len=2):
        res.append(item.word)

   #keywords = list(HanLP.extractKeyword(text, n_words))
   #words = []
   #for k in keywords:
   #    if k in word2nature:
   #       nature = word2nature[k]
   #       if nature in wordAttr:
   #          words.append(k)
   wordss = []
   for r in res:
       if r in word2nature:
          nature = word2nature[r]
          if nature in wordAttr:
             wordss.append(r)

   return  word2nature,wordss


def single_sentence_tokens(sentence,turn_idx=0,role="doctor",diag_type=None,n_words=3,wordAttr=["n","v","nhd"]):
    """
    sentence : dialog sentence
    turn_idx : this dialog turn id
    role     : the speaker role(doctor or patient)
    diag_type: this talk attribute (chitchat or prescribe or null)
    """
    turn_dict={}
    seg_list=jieba.cut(sentence,cut_all=False)
    
    turn_dict["turn"] = turn_idx
    turn_dict["sentence"] = sentence
    turn_dict["role"] = role
    turn_dict["tokens"] = list(seg_list)
    #keywords = []
    #if role =="doctor":
    _,keywords = nlp_parser(sentence,TR4W,n_words,wordAttr) 
    turn_dict["keywords"] = keywords
    turn_dict["type"] = diag_type
    return turn_dict
    

def dialog_session(dialog,use_sum=False,n_words=3,wordAttr=["n","v","nhd"]):
    diag_session={}
    diag_session["question_description"] = None
    diag_session["question_answer"] = None
    diag_session["time"] = None
    diag_session["page_url"] = None
    diag_session["disease_grad"]=None
    diag_session["topic"] = None
    diag_session["dialogues"] =[]
    role_dict={"P":"patient","D":"doctor","SUM1":"patient","SUM2":"doctor","SUM2.0":"doctor","SUM2.1":"doctor","SUM2.2":"doctor"}
    
    for i,talk in enumerate(dialog):
        if i==0:
            s=talk.split("\t")
            department=s[3]
            diag_session["disease_grad"] = f'医院>{department}'
            diag_session["topic"]=department
        else:
            talk_list=talk.split("\t")
            if "SUM" in  talk_list[0]:
                if not use_sum:
                    continue
                else:
                    role=role_dict[talk_list[0]]
                    sentence=talk_list[1]
                    #label=talk_list[2]
                    talk_dict=single_sentence_tokens(sentence=sentence,turn_idx=i-1,role=role,diag_type=None,n_words=n_words,wordAttr=wordAttr)
                    diag_session["dialogues"].append(talk_dict)
            else:
                role=role_dict[talk_list[0]]
                sentence=talk_list[1]
                #label=talk_list[2]
                talk_dict=single_sentence_tokens(sentence=sentence,turn_idx=i-1,role=role,diag_type=None,n_words=n_words,wordAttr=wordAttr)
                diag_session["dialogues"].append(talk_dict)
    return diag_session

def write_dialog_session(session_dict,session_file):
    with open(session_file,"w",encoding="utf-8") as fout:
         fout.write(json.dumps(session_dict,ensure_ascii=False))
    fout.close()
######################
if __name__=="__main__":

   parser = argparse.ArgumentParser()
   parser.add_argument("--dialog",type=str,default=None,help="het-mc dialog file")
   parser.add_argument("--use_sum",action="store_true",help="whether use sum")
   parser.add_argument("--outdir",type=str,default="dialog",help="output path to save result")
   args = parser.parse_args()
   
   use_sum = args.use_sum
   dialog_file = args.dialog
   outdir = args.outdir

   n_words=3
   wordAttr=["n","v","nhd"]
   if not os.path.exists(outdir):
      os.makedirs(outdir)
    
   print("get dialogs..")
   dialogs=split_to_single_dialogs(dialog_file) 
   
   for i,dialog in enumerate(dialogs):
        if i%100==0:
           print("write NO.{} dialog session\n".format(str(i)))
        session_filename=outdir+"/"+"custom_session_"+str(i)+".json"
        session=dialog_session(dialog,use_sum,n_words=n_words,wordAttr=wordAttr)
        n_len = len(session["dialogues"])
        if n_len > 3 and n_len%2==0:
           write_dialog_session(session,session_filename)

   print("Done!")
