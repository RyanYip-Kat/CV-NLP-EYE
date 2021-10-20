import numpy as np
import pandas as pd
import torch
import argparse
import torch.nn.functional as F
import json
# Import os for env varibles via Beaker
import os
import logging

#from torch import cuda
from typing import List
from torch.utils.data import Dataset, DataLoader, RandomSampler, SequentialSampler
from transformers import GPT2LMHeadModel,BertTokenizer,BertConfig,TextGenerationPipeline

from comet_utils import write_items
from mosaic.infra.modeling import train, validate, beam_generations
from mosaic.datasets.KGDataset import KGDataset


def config():
    parser = argparse.ArgumentParser(description='Configure for Atomic GPT2 Chinese')
    parser.add_argument("--train_batch_size",type=int,default=2,help="train batch size")
    parser.add_argument("--valid_batch_size",type=int,default=1,help="valid batch size")
    parser.add_argument("--train_epochs",type=int,default=3,help="train epochs")
    parser.add_argument("--valid_epochs",type=int,default=1,help="valid epochs")

    parser.add_argument("--lr",type=float,default=1e-5,help="learning rate")
    parser.add_argument("--seed",type=int,default=7777,help="sample seed")
    parser.add_argument("--in_len",type=int,default=16)
    parser.add_argument("--out_len",type=int,default=34)
    parser.add_argument("--summary_len",type=int,default=0)

    parser.add_argument("--outdir",type=str,default="./models")
    parser.add_argument("--do_train",action="store_true")
    parser.add_argument("--do_pred",action="store_true")
    parser.add_argument("--debug",action="store_true")

    parser.add_argument("--pred_file",type=str,default=None)
    parser.add_argument("--pred_batch",type=int,default=64)
    parser.add_argument("--top_k",type=int,default=40)
    parser.add_argument("--model",type=str,default="uer/gpt2-chinese-cluecorpussmall",help="model name or path")
    parser.add_argument("--data",type=str,default="./data",help="data path include(train.tsv,valid.tsv,test.tsv)")
    parser.add_argument("--device",type=int,default=0,help="which gpu to use")
    args=parser.parse_args()
    return args

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def read_jsonl_lines(input_file: str) -> List[dict]:
    with open(input_file) as f:
        lines = f.readlines()
        return [json.loads(l.strip()) for l in lines]


def load_data(datapath,encoding="latin-1",sep="\t",debug=True,num_inst=100):
    columns_names=["head_event","relation","tail_event"]
    ########### train
    train_dataset = pd.read_csv(datapath+"/"+"train.tsv",encoding=encoding, sep=sep)
    train_dataset.columns = columns_names
    if debug:
        train_dataset = train_dataset.head(num_inst)

    train_dataset.head_event = train_dataset.head_event + ' ' + train_dataset.relation \
                               + "[GEN]"
    train_dataset.tail_event = train_dataset.tail_event + '[EOS]'
    #logger.info(train_dataset.head())
    #logger.info(train_dataset.tail_event)

    ############ test
    test_dataset = pd.read_csv(datapath+"/"+"test.tsv",encoding=encoding, sep=sep)
    test_dataset.columns = columns_names
    if debug:
        test_dataset = test_dataset.head(num_inst)

    test_dataset.head_event = test_dataset.head_event + ' ' + test_dataset.relation \
                               + "[GEN]"
    test_dataset.tail_event = test_dataset.tail_event + '[EOS]'
    #logger.info(test_dataset.head())
    #logger.info(test_dataset.tail_event)

    ############ valid
    valid_dataset = pd.read_csv(datapath+"/"+"valid.tsv",encoding=encoding, sep=sep)
    valid_dataset.columns = columns_names
    if debug:
        valid_dataset = valid_dataset.head(num_inst)
    valid_dataset.head_event = valid_dataset.head_event + ' ' + valid_dataset.relation \
                               + "[GEN]"
    valid_dataset.tail_event = valid_dataset.tail_event + '[EOS]'
    #logger.info(valid_dataset.head())
    #logger.info(valid_dataset.tail_event)

    ###########
    return train_dataset,test_dataset,valid_dataset

def add_additional_tokens(tokenizer,tokens:list):
    """
    tokens :  all relation  in data
    """
    tokenizer.add_special_tokens({'eos_token': '[EOS]'})
    tokenizer.add_special_tokens({'pad_token': '[PAD]'})

    if "[GEN]" not in tokens:
        tokens.append("[GEN]")
    tokenizer.add_special_tokens({'additional_special_tokens':tokens})
    return tokenizer



if __name__=="__main__":
    args = config()
    os.environ["CUDA_VISIBLE_DEVICES"] =  str(args.device)

    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    ##########
    logger = logging.getLogger("gpt2-comet")
    logging.basicConfig(level=logging.DEBUG)

    logging.info("Loading model from {}".format(args.model))
    tokenizer = BertTokenizer.from_pretrained(args.model)

    model =  GPT2LMHeadModel.from_pretrained(args.model)
    logging.info("Move model to device {}".format(device))
    model = model.to(device)
    model.resize_token_embeddings(len(tokenizer))

    #additional_tokens= ["[CHITCHAT]", "[ASK_SYMPTOMS]", "[DIAGNOSIS]", "[PRESCRIBE]","[RISKA]","[RISKB]"]
    #additional_tokens=["chitchat","ask_symptoms","diagnosis","prescribe"]
    additional_tokens =['Accelerator', 'Inhibitor', 'Test', 'X-Y', 'uncorrelated']
    tokenizer=add_additional_tokens(tokenizer,additional_tokens)

    ###########
    train_dataset,test_dataset,valid_dataset=load_data(args.data,debug=False,encoding="utf-8")
    training_set = KGDataset(train_dataset, tokenizer, args.out_len, args.summary_len, model="gpt2")
    val_set = KGDataset(valid_dataset, tokenizer, args.in_len, args.out_len - args.in_len, model="gpt2", is_eval=False)
    val_set_mini = KGDataset(valid_dataset.head(2000), tokenizer, args.in_len,  args.out_len - args.in_len, model="gpt2", is_eval=False)
    test_set = KGDataset(test_dataset, tokenizer, args.in_len,  args.out_len - args.in_len, model="gpt2", is_eval=False)

    train_params = {
        'batch_size': args.train_batch_size,
        'shuffle': True,
        'num_workers': 0
    }

    val_params = {
        'batch_size': 1,
        'shuffle': False,
        'num_workers': 0
    }

    training_loader = DataLoader(training_set, **train_params, drop_last=True)
    val_loader = DataLoader(val_set, **val_params, drop_last=True)
    test_loader = DataLoader(test_set, **val_params, drop_last=True)
    val_loader_mini = DataLoader(val_set_mini, **val_params, drop_last=True)

    model = model.to(device)
    model.resize_token_embeddings(len(tokenizer))

    optimizer = torch.optim.Adam(params=model.parameters(), lr=args.lr)

    if args.do_train:
        logger.info('Initiating Fine-Tuning for the model on our dataset')

        for epoch in range(args.train_epochs):
            train(epoch, tokenizer, model, device, training_loader, optimizer, val_loader, model_class="gpt2",save_dir=args.outdir)
            model.save_pretrained('{}/checkpoint_{}'.format(args.outdir, epoch))
            tokenizer.save_pretrained('{}/checkpoint_{}'.format(args.outdir, epoch))
        model.save_pretrained(args.outdir)

    if args.do_pred:
        columns_names=["head_event","relation","tail_event"]
        if args.pred_file.endswith("jsonl"):
            records = read_jsonl_lines(args.pred_file)
            pred_dataset = pd.DataFrame.from_records(records)
            pred_dataset.columns = columns_names
            #pred_dataset = pred_dataset.rename(columns={"head": "head_event", "tails": "tail_event"})
            pred_dataset = pred_dataset.explode('tail_event')
        else:
            #pred_dataset = pd.read_csv(args.pred_file, encoding='latin-1', sep="\t")
            pred_dataset = pd.read_csv(args.pred_file, encoding='utf-8', sep="\t")
            pred_dataset.columns = columns_names

        if args.debug:
            pred_dataset = pred_dataset.head(100)

        #pred_dataset = pred_dataset.drop_duplicates(['head_event', 'relation'], ignore_index=True)

        pred_dataset.head_event = pred_dataset.head_event + ' ' + pred_dataset.relation + "[GEN]"
        pred_dataset.tail_event = pred_dataset.tail_event +  '[EOS]'
        logger.info(pred_dataset.tail_event)
        logger.info(pred_dataset.head())

        pred_set = KGDataset(pred_dataset, tokenizer, args.in_len, args.out_len - args.in_len, model="gpt2", is_eval=True)
        pred_loader = DataLoader(pred_set, **val_params, drop_last=False)

        pred_generations = beam_generations(tokenizer, model, device, pred_loader, max_len=args.out_len,top_k=args.top_k)
        write_items(os.path.join(args.outdir, "pred_generations.jsonl"),
                    [json.dumps(r,ensure_ascii=False) for r in pred_generations])

        # Resave the model to keep generations and model associated
        model.save_pretrained(args.outdir)
        tokenizer.save_pretrained(args.outdir)
