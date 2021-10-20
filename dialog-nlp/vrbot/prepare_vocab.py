import json
import logging
import zipfile
import numpy as np
from tqdm import tqdm

d_logger = logging.getLogger("main")


def read_sessions_from_zip_filename(zip_filename):
    zip_file = zipfile.ZipFile(zip_filename)
    zip_namelist = list(filter(lambda x: x.endswith("json"), list(zip_file.namelist())))
    pbar = tqdm(zip_namelist)
    name2session = {}
    d_logger.info("\nread session from {}".format(zip_filename))

    for name in pbar:
        pbar.set_description(name)
        session = json.loads(zip_file.read(name).decode("utf-8"))
        dialogues = session["dialogues"]
        dialogues = [{"turn": dia.get("turn", None) or dia.get("turn_index", None),
                      "sentence": dia["sentence"],
                      "role": dia["role"],
                      "tokens": [word.strip("\n") for word in dia["tokens"] if word != "\n"],
                      "type": dia.get("type", None),
                      "keywords": dia["keywords"] if "keywords" in dia else ""} for dia in dialogues]
        session["dialogues"] = dialogues
        name2session[name] = session

    return name2session


def make_vocab(sessions,outfile="vocab.csv"):
    """
    sessions : from list(read_sessions_from_zip_filename(*).values)
    """
    tokens_list,keywords_list = [],[]
    for session in tqdm(sessions,total=len(sessions),ncols=100,desc="make vocab"):
        dialogues = session["dialogues"]
        for dialogue in dialogues:
            tokens =  dialogue["tokens"]
            keywords = dialogue["keywords"]
            for token in tokens:
                tokens_list.append(token)
            for keyword in keywords:
                keywords_list.append(keyword)
    
    #tokens_list = reduce(lambda x, y: x+y,tokens_list) 
    #keywords_list = reduce(lambda x, y: x+y,keywords_list) 
    tokens_list = np.unique(tokens_list).tolist()
    keywords_list = np.unique(keywords_list).tolist()
    
    fout =  open(outfile,"w")
    header  = ",".join(["Word","Is_know"])
    fout.write(header+"\n")
    for token in tokens_list:
        if token in keywords_list:
            line = ",".join([token,"1"])
        else:
            line = ",".join([token,"0"])
        fout.write(line+'\n')
    fout.close()
    #return tokens_list,keywords_list

if __name__=="__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i","--json_zip_file",type=str,default=None,help="json zip file")
    parser.add_argument("-o","--outfile",type=str,default="vocab.csv",help="output vocab file")
    args = parser.parse_args()

    zip_file = args.json_zip_file
    outfile  = args.outfile
    name2session=read_sessions_from_zip_filename(zip_file)
    sessions = list(name2session.values())
    make_vocab(sessions,outfile)
