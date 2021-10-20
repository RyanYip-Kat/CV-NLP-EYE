import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import GPT2LMHeadModel,BertTokenizer,BertConfig,TextGenerationPipeline
from generate_utils import calculate_rouge, use_task_specific_params, calculate_bleu_score, trim_batch

import numpy as np
from nltk.translate.bleu_score import sentence_bleu
from utils import read_jsonl, remove_prefix, write_jsonl
from evaluation.eval import QGEvalCap
from tabulate import tabulate
import json
import os
from collections import defaultdict
import random


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]
def postprocess(sentence):
    return sentence


class Comet:
    def __init__(self, model_path):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        #self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path).to(self.device)
        #self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.tokenizer = BertTokenizer.from_pretrained(model_path)
        self.model =  GPT2LMHeadModel.from_pretrained(model_path).to(self.device)

        task = "summarization"
        use_task_specific_params(self.model, task)
        self.batch_size = 1
        self.decoder_start_token_id = None

    def generate(
            self, 
            queries,
            decode_method="beam", 
            num_generate=5, 
            ):

        with torch.no_grad():
            examples = queries

            decs = []
            for batch in list(chunks(examples, self.batch_size)):

                batch = self.tokenizer(batch, return_tensors="pt", truncation=True, padding="max_length").to(self.device)
                batch.pop("token_type_ids") # BertTokenizer will more than one key : "token_type_ids",must remove
                input_ids, attention_mask = trim_batch(**batch, pad_token_id=self.tokenizer.pad_token_id)

                summaries = self.model.generate(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    decoder_start_token_id=self.decoder_start_token_id,
                    num_beams=num_generate,
                    num_return_sequences=num_generate,
                    )

                dec = self.tokenizer.batch_decode(summaries, skip_special_tokens=True, clean_up_tokenization_spaces=False)
                decs.append(dec)

            return decs

all_relations = ['Accelerator', 'Inhibitor', 'Test', 'X-Y', 'uncorrelated']


def prepare_data(filename="train.tsv"):
    heads_relations = []
    references = []

    with open(filename,encoding="utf-8") as fin:
        for i,line in enumerate(fin):
            if i==0:
                continue
            hr={}
            ll=line.rstrip().split("\t")
            head,relation,tail = ll
            hr["head"],hr["relation"]  = head,relation
            heads_relations.append(hr) 
            references.append(tail)
    return heads_relations,references

def generate(infile,model_path):
    print("model loading ...")
    comet = Comet(model_path)
    comet.model.zero_grad()
    print("model loaded")

    head_relation,reference = prepare_data(infile)
    hyps =[]
    reference_list= []
    head_relations= []
    
    for i,hr in enumerate(head_relation):
        queries = []
        head = hr["head"]
        rel = hr["relation"]
        query = "{} {} [GEN]".format(head, rel)
        queries.append(query)
        try:
            res = comet.generate(queries, decode_method="beam", num_generate=4)
            print(res)
            hyps.append([x.replace(" ","")  for x in  res[0]])
            reference_list.append(reference[i])
            head_relations.append(head_relation[i])
        except:
            print(f'{query} invalid')
    return hyps,reference_list,head_relations

def preprocess_generations(references_list,heads_relations,hypothesises,outpath="./"):
    outfile_path = outpath+"/"+"gen.jsonl"
    outfile = open(outfile_path, 'w',encoding="utf-8")

    idx = 0

    total_bleu_1 = 0
    total_bleu_2 = 0
    total_bleu_3 = 0
    total_bleu_4 = 0

    relation_bleu_1 = defaultdict(lambda: defaultdict(int))

    count = 0

    for head_relation, references, hypothesis in zip(heads_relations, references_list, hypothesises):
        bleu_1 = sentence_bleu(references, hypothesis, weights=[1.0])
        bleu_2 = sentence_bleu(references, hypothesis, weights=[0.5, 0.5])
        bleu_3 = sentence_bleu(references, hypothesis, weights=[0.34, 0.33, 0.33])
        bleu_4 = sentence_bleu(references, hypothesis)

        result = {
            'generation': postprocess(hypothesis),
            'references': [postprocess(reference) for reference in references],
            'input': head_relation
        }
        if hypothesis != 'none':
            total_bleu_1 += bleu_1
            total_bleu_2 += bleu_2
            total_bleu_3 += bleu_3
            total_bleu_4 += bleu_4

            relation_bleu_1[head_relation["relation"]]["total"] += bleu_1
            relation_bleu_1[head_relation["relation"]]["count"] += 1

            count += 1

        outfile.write(json.dumps(result,ensure_ascii=False) + "\n")
    print('gens non-none', count)
    outfile_scores = open(outpath + "/"+"gen_scores.jsonl", 'w',encoding="utf-8")

    summary = {
        'bleu1': total_bleu_1 / count,
        'bleu2': total_bleu_2 / count,
        'bleu3': total_bleu_3 / count,
        'bleu4': total_bleu_4 / count
    }

    for relation in relation_bleu_1:
        summary[relation] = relation_bleu_1[relation]["total"] / relation_bleu_1[relation]["count"]

    outfile_scores.write(json.dumps(summary) + "\n")
    excel_str = ""
    for key in summary:
        excel_str += str(key) + '\t'
    outfile_scores.write(excel_str.strip())
    outfile_scores.write("\n")
    excel_str = ""
    for key in summary:
        excel_str += str(summary[key]) + '\t'

    outfile_scores.write(excel_str.strip())

    print(f"Saved gens in {outfile_path}")

    return(os.path.abspath(outfile_path))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-i','--infile', type=str,default="demo.tsv", help='atomic 2020 datasset file (train.tsv,test.tsv...)')
    parser.add_argument('-m','--model_path',type=str,default="/home/jovyan/work/VRBot-History2/youlai_test/checkpoint_49/",help="atomic 2020 model path")
    parser.add_argument('-o','--outdir',type=str,default="generation")

    args = parser.parse_args()

    # sample usage
    #model_path = "/home/jovyan/work/VRBot-History2/youlai_test/checkpoint_49/"
    #infile = "demo.tsv"
    model_path = args.model_path
    infile = args.infile 
    outpath = args.outdir
    if not os.path.exists(outpath):
        os.makedirs(outpath)

    hyps,reference_list,head_relations = generate(infile,model_path)
    print(hyps[0])
    print(reference_list[0])
    print(head_relations[0])
    preprocess_generations(hyps,head_relations,reference_list,outpath)
