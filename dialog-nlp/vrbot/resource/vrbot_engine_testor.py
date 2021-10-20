# -*- coding: utf-8 -*-

import torch
import logging
import os
from tqdm import tqdm
from resource.input.vocab import Vocab
from resource.model.vrbot import TRAIN
from resource.model.vrbot import VRBot
from resource.metric.eval_bleu import eval_bleu
from resource.input.tensor2nl import TensorNLInterpreter
from resource.input.session_dataset import SessionDataset
from resource.option.train_option import TrainOption as TO
from resource.option.vrbot_option import VRBotOption as VO
from resource.option.dataset_option import DatasetOption as DO
from resource.util.misc import mkdir_if_necessary
from resource.input.session import SessionCropper
from resource.util.misc import one_hot_scatter
from resource.base_engine import BaseEngine,TestBaseEngine
from resource.model.vrbot_train_state import vrbot_train_stage

engine_logger = logging.getLogger("main.engine")

def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)

class TestVRBotEngine(TestBaseEngine):
    def __init__(self, model: VRBot,
                 vocab: Vocab,
                 inner_vocab: Vocab,
                 lr=None):
        super(TestVRBotEngine, self).__init__(model, lr or TO.lr)
        self.iw_interpreter = TensorNLInterpreter(vocab)
        self.ii_interpreter = TensorNLInterpreter(inner_vocab)
        self.cache = {"super": {"pv_hidden": None, "pv_state": None},
                      "unsuper": {"pv_hidden": None, "pv_state": None}}
    def test(self, dataset: SessionDataset, mode="test"):
        assert mode == "test" or mode == "valid"
        print("SESSION NUM: {}".format(len(dataset.sessions)))
        dataset_bs = dataset.batch_size
        pv_hidden = torch.zeros((1, dataset_bs, self.model.hidden_dim), dtype=torch.float, device=TO.device)
        pv_state = torch.zeros((dataset_bs, self.model.state_num, self.model.inner_vocab_size),
                               dtype=torch.float, device=TO.device)
        pv_state[:, :, 0] = 1.0

        # cache
        all_targets = []
        all_outputs = []

        engine_logger.info("{} INFERENCE START ...".format(mode.upper()))
        session_cropper = SessionCropper(dataset.batch_size)

        self.model.eval()
        with torch.no_grad():
            for input_tensors, inherited, materialistic in tqdm(dataset.load_data()):
                if len(input_tensors) == 5:
                    pv_r_u, pv_r_u_len, r, r_len, gth_intention = input_tensors
                    gth_s, gth_a = None, None
                elif len(input_tensors) == 7:
                    pv_r_u, pv_r_u_len, r, r_len, gth_s, gth_intention, gth_a = input_tensors
                else:
                    raise RuntimeError

                pv_hidden, pv_state = self.hidden_state_mask(pv_hidden, pv_state, inherited, materialistic)

                gen_log_probs, state_index, action_index, hidden4post = self.model.forward(pv_state,
                                                                                           pv_hidden,
                                                                                           pv_r_u,
                                                                                           pv_r_u_len,
                                                                                           None)

                posts = self.iw_interpreter.interpret_tensor2nl(pv_r_u)
                targets = self.iw_interpreter.interpret_tensor2nl(r[:, 1:])
                outputs = self.iw_interpreter.interpret_tensor2nl(gen_log_probs)
                states = self.ii_interpreter.interpret_tensor2nl(state_index)
                actions = self.ii_interpreter.interpret_tensor2nl(action_index)

                if gth_s is not None:
                    gth_states = self.ii_interpreter.interpret_tensor2nl(gth_s)
                else:
                    gth_states = ["<pad>"] * len(posts)

                inherited = inherited.detach().cpu().numpy().tolist()
                materialistic = materialistic.detach().cpu().numpy().tolist()
                session_cropper.step_on(posts, targets, outputs, states, actions,
                                        inherited, materialistic, gth_states)
                all_targets += targets
                all_outputs += outputs

                # for next loop
                pv_hidden = hidden4post
                #print(f'pv_hidden : {pv_hidden.shape}')
                pv_state = one_hot_scatter(state_index, self.model.inner_vocab_size, dtype=torch.float)
                #print(f'pv_state : {pv_state.shape}')

        self.model.train()
        engine_logger.info("{} INFERENCE FINISHED".format(mode.upper()))
        metrics = eval_bleu([all_targets], all_outputs)

        return all_targets, all_outputs, metrics, session_cropper

    @staticmethod
    def hidden_state_mask(pv_hidden, pv_state, inherited, materialistic):
        if pv_hidden.shape[1] != inherited.unsqueeze(0).unsqueeze(2).float().shape[1]:
             n_c = inherited.unsqueeze(0).unsqueeze(2).float().shape[1]-pv_hidden.shape[1]
             pv_hidden_c=torch.randn(pv_hidden.shape[0],n_c,pv_hidden.shape[2]).to(pv_hidden.device)
             pv_hidden = torch.cat([pv_hidden,pv_hidden_c],axis=1) 
             #n_c = pv_hidden.shape[1]
             #torch.randint(0,16,(3,))

        pv_hidden = pv_hidden * inherited.unsqueeze(0).unsqueeze(2).float()

        B, S, K = pv_state.shape
        state_placeholder = torch.zeros(B, S, K, dtype=torch.float, device=pv_state.device)
        state_placeholder[:, :, 0] = 1.0

        vn = int(inherited.sum().item())
        inherited_batch_index = \
            torch.sort(torch.arange(0, B, dtype=torch.long, device=TO.device) * inherited)[0][- vn:]

        state_placeholder[inherited_batch_index] = pv_state[inherited_batch_index]
        pv_state = state_placeholder

        batch_size = pv_hidden.size(1)
        vn = int(materialistic.sum().item())
        reserved_batch_index = \
            torch.sort(torch.arange(0, batch_size, dtype=torch.long, device=TO.device) * materialistic)[0][- vn:]


        reserved_hidden = pv_hidden[:, reserved_batch_index, :]
        reserved_state = pv_state[reserved_batch_index, :]
        return reserved_hidden, reserved_state

    def tick(self, state_train=False, action_train=False):
        if self.global_step % VO.copy_lambda_decay_interval == 0:
            if VO.s_copy_lambda > VO.state_action_copy_lambda_mini and state_train:
                VO.s_copy_lambda = max(VO.s_copy_lambda - VO.copy_lambda_decay_value,
                                       VO.state_action_copy_lambda_mini)
            if VO.a_copy_lambda > VO.state_action_copy_lambda_mini and action_train:
                VO.a_copy_lambda = max(VO.a_copy_lambda - VO.copy_lambda_decay_value,
                                       VO.state_action_copy_lambda_mini)

    @staticmethod
    def balance_act(origin_value, base_num):
        assert base_num > 1.0
        return base_num ** (origin_value - 1.0)

    def test_with_log(self, dataset, epoch, model_name, mode):
        targets, outputs, metrics, session_cropper = self.test(dataset, mode=mode)
        metric_str = "(" + "-".join(["{:.4f}".format(x) for x in metrics]) + ")"
        #valid_output_filename = DO.test_filename_template.format(model=model_name,
        #                                                         uuid=TO.task_uuid,
        #                                                         epoch=epoch,
        #                                                         global_step=self.global_step,
        #                                                         mode=mode,
        #                                                         metric=metric_str)
        valid_output_filename = "data/test/"+model_name+"-"+TO.task_uuid+"-"+mode+".txt"
        mkdir_if_necessary(valid_output_filename)
        engine_logger.info("WRITE {} OUTPUT TO FILE {}".format(mode, valid_output_filename))
        self.json_writer.write2file(valid_output_filename, session_cropper.to_dict())


    def dump_model(self, epoch, step, ckpt_filename):
        dump_dict = {
            "epoch": epoch,
            "step": step,
            "ckpt": self.model.state_dict(),
            "task_uuid": TO.task_uuid,
            "vrbot_train_stage": vrbot_train_stage.dump()
        }
        engine_logger.info("DUMPING CKPT TO FILE {}".format(ckpt_filename))
        torch.save(dump_dict, ckpt_filename)
        engine_logger.info("DUMPING CKPT DONE")

    def load_model(self, ckpt_filename):
        engine_logger.info("LOAD CKPT FROM {}".format(ckpt_filename))
        dump_dict = torch.load(ckpt_filename)
        epoch = dump_dict["epoch"]
        step = dump_dict["step"]
        task_uuid = dump_dict["task_uuid"]
        ckpt = dump_dict["ckpt"]
        self.model.load_state_dict(ckpt, strict=False)
        vrbot_train_stage.self_update(dump_dict["vrbot_train_stage"])
        VO.s_copy_lambda = vrbot_train_stage.s_copy_lambda
        VO.a_copy_lambda = vrbot_train_stage.a_copy_lambda
        return epoch, step, task_uuid
