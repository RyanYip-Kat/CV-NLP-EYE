import json
import logging
import zipfile
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

def get_keyword_sentence(sessions):
    key_sens=[]
    for i,session in enumerate(sessions):
        for sess in session:
            r=sess[1]
            if r is None:
                continue
            key_sens.append(sess)
    return key_sens


def write_items(key_sessions,outfile="test.tsv"):
    column_names=["head_event","relation","tail_event"] 
    with open(outfile,"w",encoding="utf-8") as fout:
        fout.write("\t".join(column_names)+"\n")
        for sess in key_sessions:
            line=sess[0]+"\t"+sess[1]+"\t"+",".join(sess[2])
            fout.write(line+"\n")
    fout.close()



train_sessions=list(read_sessions_from_zip_filename("data/meddg_train.zip").values())
test_sessions=list(read_sessions_from_zip_filename("data/meddg_test.zip").values())
valid_sessions=list(read_sessions_from_zip_filename("data/meddg_valid.zip").values())

train_sessions_list=[[(sess["sentence"],sess.get("type", None),sess["keywords"],sess["role"]) for sess in  session["dialogues"]] for session in train_sessions]
test_sessions_list=[[(sess["sentence"],sess.get("type", None),sess["keywords"],sess["role"]) for sess in  session["dialogues"]] for session in test_sessions]
valid_sessions_list=[[(sess["sentence"],sess.get("type", None),sess["keywords"],sess["role"]) for sess in  session["dialogues"]] for session in valid_sessions]

train_sessions_key_list=get_keyword_sentence(train_sessions_list)
test_sessions_key_list=get_keyword_sentence(test_sessions_list)
valid_sessions_key_list=get_keyword_sentence(valid_sessions_list)

write_items(train_sessions_key_list,"train.tsv")
write_items(test_sessions_key_list,"test.tsv")
write_items(valid_sessions_key_list,"valid.tsv")
