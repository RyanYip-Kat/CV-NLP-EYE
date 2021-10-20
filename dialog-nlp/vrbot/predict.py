import argparse
import torch
import random
#from resource.option.config import config
from resource.util.get_logger import get_logger
from resource.option.train_option import TrainOption as TO
from resource.option.dataset_option import DatasetOption as DO
from resource.option.vrbot_option import VRBotOption as VO

main_logger = get_logger("main", "data/predict_log/{}.log".format(TO.task_uuid))
main_logger.info("TASK ID {}".format(TO.task_uuid))

from resource.model.vrbot import VRBot
from resource.vrbot_engine_testor import TestVRBotEngine
from resource.model.vrbot_train_state import vrbot_train_stage
from resource.input.graph_db import GraphDB, TripleLoader
from resource.input.data_processor import DataProcessor
from resource.input.session_dataset import SessionDataset
from resource.input.session_dataset import MixedSessionDataset
from resource.input.session_dataset import SessionProcessor
from resource.util.loc_glo_trans import LocGloInterpreter
from resource.tools.load_data import read_sessions_from_zip_filename
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task",type=str, help="kamed, meddialog,meddg,meddg_dev,eyeai", required=True)
    parser.add_argument("-i","--input_session_filename",type=str,help="input session zip filename to test")
    parser.add_argument("--supervised", action="store_true")
    parser.add_argument("--auto_regressive",action="store_true")
    parser.add_argument("--test_batch_size", type=int, default=32)
    parser.add_argument("--embed_dim", type=int, default=300)
    parser.add_argument("--hidden_dim", type=int, default=512)

    parser.add_argument("--hidden_units",type=int, default=512)
    parser.add_argument("--device", type=int, default=-1)
    parser.add_argument("--ckpt", type=str,help="ckpt model file")
    parser.add_argument("--worker_num", type=int, default=3)
    parser.add_argument("--hop", type=int, default=1)
    parser.add_argument("--ppn_dq", action="store_true")
    parser.add_argument("--gen_strategy", choices=["mlp", "gru"], default="gru")
    parser.add_argument("--mem_depth", type=int, default=3)
    parser.add_argument("--state_action_copy_lambda", type=float, default=1.0)
    parser.add_argument("--copy_lambda_decay_interval", type=int, default=10000)
    parser.add_argument("--tau_decay_interval", type=int, default=5000)
    parser.add_argument("--copy_lambda_decay_value", type=float, default=1.0)
    parser.add_argument("--state_action_copy_lambda_mini", type=float, default=1.0)
    parser.add_argument("--mask_state_prob", action="store_true")
    parser.add_argument("--no_classify_weighting", action="store_true")
    parser.add_argument("--gradient_stack", type=int, default=1)
    parser.add_argument("--beam_width", type=int, default=5)
    parser.add_argument("--state_num", type=int, default=10)
    parser.add_argument("--action_num", type=int, default=3)
    args = parser.parse_args()
    TO.auto_regressive = args.auto_regressive and (args.gen_strategy == "gru")
    TO.update_device(args.device)
    TO.beam_width = args.beam_width
    VO.ppn_dq = args.ppn_dq
    TO.no_classify_weighting = args.no_classify_weighting

    TO.update("mem_depth", args.mem_depth)
    TO.update("worker_num", args.worker_num)
    TO.update("test_batch_size", args.test_batch_size)
    TO.update("task", args.task)
    DO.update_joint_graph(TO.task)

    VO.state_num = args.state_num
    DO.state_num = args.state_num
    VO.action_num = args.action_num
    DO.action_num = args.action_num
    VO.s_copy_lambda = max(args.state_action_copy_lambda, args.state_action_copy_lambda_mini)
    VO.a_copy_lambda = max(args.state_action_copy_lambda, args.state_action_copy_lambda_mini)
    VO.copy_lambda_decay_interval = args.copy_lambda_decay_interval
    VO.copy_lambda_decay_value = args.copy_lambda_decay_value
    VO.state_action_copy_lambda_mini = args.state_action_copy_lambda_mini
    VO.tau_decay_interval = args.tau_decay_interval
    VO.mask_state_prob = args.mask_state_prob
    VO.embed_dim = args.embed_dim
    VO.hidden_dim = args.hidden_dim
    VO.hidden_units  = args.hidden_units
    return args

def prepare_data(args):
    main_logger.info("preparing sessions")
    data_processor = DataProcessor(args.task)
    #train_sessions, test_sessions, valid_sessions = data_processor.get_session()

    main_logger.info("preparing vocab")
    word_vocab, know_vocab, glo2loc, loc2glo, vocab_size, inner_vocab_size = data_processor.get_vocab()
    glo2loc = torch.tensor(glo2loc, device=TO.device)
    loc2glo = torch.tensor(loc2glo, device=TO.device)

    #return [train_sessions, test_sessions, valid_sessions,
    #        word_vocab, know_vocab, glo2loc, loc2glo, vocab_size, inner_vocab_size]
    return word_vocab,know_vocab, glo2loc, loc2glo, vocab_size, inner_vocab_size


def cfg2str(option):
    cfg_str = ["\n======= {} START =======".format(option.__name__)]
    for key, value in option.__dict__.items():
        if key.startswith("_"):
            continue
        cfg_str.append("{} : {}".format(key, value))
    cfg_str += ["======= {} END =======\n".format(option.__name__)]
    return "\n".join(cfg_str)

def read_session(session_zip_filename,supervised=False):
    def _extract_pure_sessions(sessions):
            if supervised:
                pure_sessions = []
                for session in sessions:
                    pure_session = []
                    state = []
                    for sentence in session["dialogues"]:
                        keywords = sentence["keywords"]
                        pure_sentence = list()
                        # [sentence, type, state / action]
                        pure_sentence.append(sentence["tokens"])
                        pure_sentence.append(sentence.get("type", None))

                        if sentence["role"] == "doctor":
                            pure_sentence.append(keywords[:DO.action_num])  # action -
                            to_add_state = [k for k in keywords if k not in state]
                            state = state + to_add_state
                            state = state[-DO.state_num:]
                        else:
                            to_add_state = [k for k in keywords if k not in state]
                            state = state + to_add_state
                            state = state[-DO.state_num:]
                            pure_sentence.append(state)  # state -

                        pure_session.append(pure_sentence)
                    pure_sessions.append(pure_session)
            else:
                # [sentence, type]
                pure_sessions = [[(sentence["tokens"], sentence.get("type", None)) for sentence in session["dialogues"]]
                                 for session in sessions]
            return pure_sessions
    sessions = list(
            read_sessions_from_zip_filename(session_zip_filename).values())
    sessions = _extract_pure_sessions(sessions)
    return sessions

def main():
    seed = 123
    random.seed(seed)
    main_logger.info("PARAMETER PARSING")

    args = config()
    vrbot_train_stage.update_relay()
    main_logger.info(cfg2str(VO))

    main_logger.info("PREPARE DATA")
    word_vocab, inner_vocab, glo2loc, loc2glo, vocab_size, inner_vocab_size = prepare_data(args)
    sp = SessionProcessor(word_vocab, inner_vocab, DO.pv_r_u_max_len, DO.r_max_len)

    sessions = read_session(args.input_session_filename,supervised=args.supervised) 
    dataset = SessionDataset(sp, "test", args.test_batch_size, sessions,supervised=args.supervised)

    lg_interpreter = LocGloInterpreter(loc2glo, glo2loc)
    triple_loader = TripleLoader(DO.joint_graph_filename, inner_vocab)
    head_relation_tail_np, head2index, tail2index = triple_loader.load_triples()
    graph_db = GraphDB(head_relation_tail_np, head2index, tail2index,
                       args.hop, VO.max_node_num1 if args.hop == 1 else VO.max_node_num2,
                       VO.single_node_max_triple1, VO.single_node_max_triple2)

    model = VRBot(loc2glo, VO.state_num, VO.action_num, VO.hidden_dim,
                  inner_vocab_size, vocab_size, VO.response_max_len, VO.embed_dim,
                  lg_interpreter, gen_strategy=args.gen_strategy,
                  with_copy=True, graph_db=graph_db, beam_width=TO.beam_width)

    if args.device >= 0:
        model = model.to(TO.device)

    engine = TestVRBotEngine(model, word_vocab, inner_vocab)
    main_logger.info("LOAD CHECKPOINT FROM {}".format(args.ckpt))
    epoch, global_step, origin_task_uuid = engine.load_model(args.ckpt)
    engine.global_step = global_step

    mode = "test" 
    model_name = model.__class__.__name__.upper()
    engine.test_with_log(dataset, epoch, model_name, mode)


if __name__=="__main__":
    main()
