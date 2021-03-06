proj_root_path="/home/jovyan/work/QueryGeneration/"

import sys
import os
import argparse
sys.path.append(proj_root_path)
import tensorflow as tf
import numpy as np

from train.modeling import GroverConfig, sample
from tokenization import tokenization

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.DEBUG)
from tensorflow.python.util import deprecation
deprecation._PRINT_DEPRECATION_WARNINGS = False
try:
    from tensorflow.python.util import module_wrapper as deprecation
except ImportError:
    from tensorflow.python.util import deprecation_wrapper as deprecation
deprecation._PER_MODULE_WARNING_LIMIT = 0

#proj_root_path="/home/jovyan/work/QueryGeneration/"


def extract_generated_target(output_tokens, tokenizer):
    """
    Given some tokens that were generated, extract the target
    :param output_tokens: [num_tokens] thing that was generated
    :param encoder: how they were encoded
    :param target: the piece of metadata we wanted to generate!
    :return:
    """
    # Filter out first instance of start token
    assert output_tokens.ndim == 1

    start_ind = 0
    end_ind = output_tokens.shape[0]

    return {
        'extraction': tokenization.printable_text(''.join(tokenizer.convert_ids_to_tokens(output_tokens))),
        'start_ind': start_ind,
        'end_ind': end_ind,
    }


def Q2Qpredict(texts,
        model_config_fn,
        model_ckpt,
        batch_size=1,
        max_batch_size=None,
        top_p=5.0,
        samples=5,
        do_topk=True,
        eostoken=102,
        minlen=5):

    #proj_root_path = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
    vocab_file_path = os.path.join(proj_root_path, "tokenization/bert-base-chinese-vocab.txt")
    tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file_path , do_lower_case=True)
    news_config = GroverConfig.from_json_file(model_config_fn)

    # We might have to split the batch into multiple chunks if the batch size is too large
    default_mbs = {12: 32, 24: 16, 48: 3}
    max_batch_size = max_batch_size if max_batch_size is not None else default_mbs[news_config.num_hidden_layers]

    # factorize args.batch_size = (num_chunks * batch_size_per_chunk) s.t. batch_size_per_chunk < max_batch_size
    num_chunks = int(np.ceil(batch_size / max_batch_size))
    batch_size_per_chunk = int(np.ceil(batch_size / num_chunks))

    # This controls the top p for each generation.
    top_p = np.ones((num_chunks, batch_size_per_chunk), dtype=np.float32) * top_p

    tf_config = tf.ConfigProto(allow_soft_placement=True)

    results=[]
    with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
        initial_context = tf.placeholder(tf.int32, [batch_size_per_chunk, None])
        p_for_topp = tf.placeholder(tf.float32, [batch_size_per_chunk])
        eos_token = tf.placeholder(tf.int32, [])
        min_len = tf.placeholder(tf.int32, [])
        tokens, probs = sample(news_config=news_config, initial_context=initial_context,
                eos_token=eos_token, min_len=min_len, ignore_ids=None, p_for_topp=p_for_topp,
                do_topk=do_topk)

        saver = tf.train.Saver()
        saver.restore(sess, model_ckpt)
        for text in texts:
            res=[]
            print("INFO : [QUESTION] --- {}".format(text))
            for i in range(samples):
               print("Sample,", i + 1, " of ", samples)
               line = tokenization.convert_to_unicode(text)
               bert_tokens = tokenizer.tokenize(line)
               bert_tokens.append("[SEP]")
               encoded = tokenizer.convert_tokens_to_ids(bert_tokens)
               context_formatted = []
               context_formatted.extend(encoded)
               # Format context end
               gens = []
               gens_raw = []
               gen_probs = []
               for chunk_i in range(num_chunks):
                  tokens_out, probs_out = sess.run([tokens, probs],
                        feed_dict={initial_context: [context_formatted] * batch_size_per_chunk,
                            eos_token: eostoken, min_len: minlen,
                            p_for_topp: top_p[chunk_i]})
                  for t_i, p_i in zip(tokens_out, probs_out):
                      extraction = extract_generated_target(output_tokens=t_i, tokenizer=tokenizer)
                      gens.append(extraction['extraction'])
               l = gens[0].replace('[UNK]', '').replace('##', '').split("[SEP]")
               print("generate query:", l[1])
            results.append(l[1])
            
    return results


class query2Q(object):
    def __init__(self,model_ckpt,
            model_fn,
            proj_root_path,
            batch_size=1,
            max_batch_size=None,
            top_p=5.0,
            samples=5,
            do_topk=True,
            eostoken=102,
            minlen=5):
        
        self.model_ckpt = model_ckpt
        self.model_fn = model_fn
        self.proj_root_path = proj_root_path

        self.batch_size = batch_size
        self.max_batch_size = max_batch_size
        self.top_p = top_p
        self.samples = samples
        self.do_topk = do_topk
        self.eostoken = eostoken
        self.minlen = minlen

        self.sess=self._load_model()


    def _load_model(self):

         vocab_file_path = os.path.join(self.proj_root_path, "tokenization/bert-base-chinese-vocab.txt")
         self.tokenizer = tokenization.FullTokenizer(vocab_file=vocab_file_path , do_lower_case=True)
         self.news_config = GroverConfig.from_json_file(self.model_fn)


         # We might have to split the batch into multiple chunks if the batch size is too large
         default_mbs = {12: 32, 24: 16, 48: 3}
         max_batch_size = self.max_batch_size if self.max_batch_size is not None else default_mbs[self.news_config.num_hidden_layers]

         # factorize args.batch_size = (num_chunks * batch_size_per_chunk) s.t. batch_size_per_chunk < max_batch_size
         self.num_chunks = int(np.ceil(self.batch_size / max_batch_size))
         self.batch_size_per_chunk = int(np.ceil(self.batch_size / self.num_chunks))

         # This controls the top p for each generation.

         tf_config = tf.ConfigProto(allow_soft_placement=True)
         with tf.Session(config=tf_config, graph=tf.Graph()) as sess:
             initial_context = tf.placeholder(tf.int32, [self.batch_size_per_chunk, None])
             p_for_topp = tf.placeholder(tf.float32, [self.batch_size_per_chunk])
             eos_token = tf.placeholder(tf.int32, [])
             min_len = tf.placeholder(tf.int32, [])
             tokens, probs = sample(news_config=self.news_config, initial_context=initial_context,
                     eos_token=eos_token, min_len=min_len, ignore_ids=None, p_for_topp=p_for_topp,
                     do_topk=self.do_topk)

             saver = tf.train.Saver()
             saver.restore(sess, self.model_ckpt)
         return sess

    def _generate(self,text):

        # This controls the top p for each generation.
        top_p = np.ones((self.num_chunks, self.batch_size_per_chunk), dtype=np.float32) * self.top_p
        initial_context = tf.placeholder(tf.int32, [self.batch_size_per_chunk, None])
        p_for_topp = tf.placeholder(tf.float32, [self.batch_size_per_chunk])
        eos_token = tf.placeholder(tf.int32, [])
        min_len = tf.placeholder(tf.int32, [])
        tokens, probs = sample(news_config=self.news_config, initial_context=initial_context,
                     eos_token=eos_token, min_len=min_len, ignore_ids=None, p_for_topp=p_for_topp,
                     do_topk=self.do_topk)

        for i in range(self.samples):
            print("Sample,", i + 1, " of ", self.samples)
            line = tokenization.convert_to_unicode(text)
            bert_tokens = self.tokenizer.tokenize(line)
            bert_tokens.append("[SEP]")
            encoded = self.tokenizer.convert_tokens_to_ids(bert_tokens)
            context_formatted = []
            context_formatted.extend(encoded)
            # Format context end
            gens = []
            gens_raw = []
            gen_probs = []
            for chunk_i in range(self.num_chunks):
                tokens_out, probs_out = self.sess.run([tokens, probs],
                        feed_dict={initial_context: [context_formatted] * self.batch_size_per_chunk,
                            eos_token: self.eostoken, min_len: self.minlen,
                            p_for_topp: top_p[chunk_i]})
                for t_i, p_i in zip(tokens_out, probs_out):
                    extraction = extract_generated_target(output_tokens=t_i, tokenizer=self.tokenizer)
                    gens.append(extraction['extraction'])
            l = gens[0].replace('[UNK]', '').replace('##', '').split("[SEP]")
            print("generate query:", l[1])
        return l



if __name__=="__main__":
    model_ckpt="/path/Query2QueryModel/model.ckpt-850000"
    model_fn="/home/jovyan/work/QueryGeneration/configs/mega.json"

    texts=["??????????????????????????????","??????????????????????????????????","?????????????????????????????????"]
    generator=query2Q(model_ckpt,model_fn,proj_root_path)
    l=generator._generate(texts[0])
    print(l)
#    res=Q2Qpredict(texts=texts,model_config_fn=model_fn,model_ckpt=model_ckpt,minlen=7,samples=1)
#    print(res)
