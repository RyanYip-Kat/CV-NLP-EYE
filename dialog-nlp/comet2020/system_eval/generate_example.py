import json
import torch
import argparse
from tqdm import tqdm
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from transformers import GPT2LMHeadModel,BertTokenizer,BertConfig,TextGenerationPipeline
from generate_utils import calculate_rouge, use_task_specific_params, calculate_bleu_score, trim_batch


def chunks(lst, n):
    """Yield successive n-sized chunks from lst."""
    for i in range(0, len(lst), n):
        yield lst[i : i + n]


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

if __name__ == "__main__":

    # sample usage
    print("model loading ...")
    model_path = "/home/jovyan/work/VRBot-History2/youlai_test/checkpoint_49/"
    comet = Comet(model_path)
    comet.model.zero_grad()
    print("model loaded")

    queries = []
    head = "湿性的黄斑变性"
    rel = "X-Y"
    query = "{} {} [GEN]".format(head, rel)
    queries.append(query)
    print(queries)
    results = comet.generate(queries, decode_method="beam", num_generate=5)
    print(results)
