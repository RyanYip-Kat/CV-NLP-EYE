# -*- coding: utf-8 -*-


import json
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
from resource.option.train_option import TrainOption as TO
from resource.option.vrbot_option import VRBotOption as VO

from resource.input.vocab import Vocab
from resource.option.dataset_option import DatasetOption
#from resource.module.graph_policy_network import  GraphPolicyNetwork
def norm(in_tensor, mask=None, dim=-1, eps=1e-24):
    if mask is not None:
        assert in_tensor.shape == mask.shape
    in_tensor = in_tensor * mask
    sum_tensor = in_tensor.sum(dim).unsqueeze(dim)
    out_tensor = in_tensor / (sum_tensor + eps)
    return out_tensor


class Sequeeze(nn.Module):
    def __init__(self):
        super(Sequeeze, self).__init__()

    def forward(self, inp):
        out = inp.squeeze(-1)
        return out


class GraphPolicyNetwork(nn.Module):
    def __init__(self, adjacent_matrix, entity_type, alias2scientific, scientific2alias, hidden_dim=32):
        super(GraphPolicyNetwork, self).__init__()
        self.adjacent_matrix = adjacent_matrix.float()
        self.mask_matrix = self.adjacent_matrix < 0.5
        self.logits_matrix = nn.Parameter(
            torch.tensor(self.adjacent_matrix.float(), device=TO.device, dtype=torch.float), requires_grad=True)
        self.logits_matrix.data = norm(self.logits_matrix.data, self.adjacent_matrix, dim=-1)

        self.scientific_entity_num = self.logits_matrix.size(1)
        self.alias_entity_num = alias2scientific.size(0)

        self.entity_embedding = nn.Embedding(entity_type.size(0), hidden_dim)
        self.intention_embedding = nn.Embedding(4, hidden_dim)
        self.state_weight_mlp = nn.Sequential(nn.Linear(hidden_dim + hidden_dim + 4 + VO.hidden_dim, hidden_dim),
                                              nn.ReLU(),
                                              nn.Linear(hidden_dim, 1),
                                              Sequeeze(),
                                              nn.Softmax(-1))
        self.entity_type = entity_type
        self.alias2scientific = alias2scientific  # K
        self.scientific2alias = scientific2alias  # K, K_s

    def forward(self, state, intention, hidden):
        state = state.long()
        intention = intention.long()
        hidden = hidden.long()
        trans_matrix = torch.softmax(self.logits_matrix.masked_fill(self.mask_matrix, -1e24), -1)
        trans_matrix = trans_matrix * self.adjacent_matrix

        common_first_tm = trans_matrix.sum(0)
        # ask_symptom & prescribe_medicine & diagnosis
        ask_symptom_second_tm = trans_matrix[0]
        ask_symptom_first_denominator = 1.0
        prescribe_second_tm = trans_matrix[3]
        prescribe_first_denominator = 1.0
        diagnosis_second_tm = trans_matrix[1] + trans_matrix[2]
        diagnosis_first_denominator = 2.0

        # B, S
        state_type = self.entity_type.index_select(0, state.reshape(-1)).reshape(state.size(0), state.size(1), -1)
        state_emb = self.entity_embedding(state)  # B, S, 32
        intention_emb = self.intention_embedding(intention.unsqueeze(1).expand(-1, state_emb.size(1)))
        state_weight_mlp_inp = torch.cat(
            [state_type, state_emb, intention_emb, hidden.unsqueeze(1).expand(-1, state.size(1), -1)], -1)
        state_weight = self.state_weight_mlp.forward(state_weight_mlp_inp)
        # B, S
        scientific_state = self.alias2scientific.unsqueeze(0).expand(state.size(0), -1).gather(1, state)
        # B, K_s
        tmp = torch.zeros(state.size(0), self.scientific_entity_num, dtype=torch.float, device=TO.device)
        state = tmp.scatter(1, scientific_state, state_weight)

        intention_effect_tag = (intention > 0).long()
        efficient_intention_num = intention_effect_tag.sum().item()

        if efficient_intention_num <= 0:
            # B, K
            ret = torch.zeros(state.size(0), self.alias_entity_num, device=TO.device, dtype=torch.float)
            return ret

        _, efficient_indices = intention_effect_tag.sort(dim=-1, descending=True)
        efficient_indices = efficient_indices[:efficient_intention_num]  # B'
        selected_intention = intention.index_select(0, efficient_indices)  # B,
        selected_state = state.index_select(0, efficient_indices).unsqueeze(1)  # B', 1, K
        batch_first_denominator = []
        batch_second_tm = []

        for single_intention in selected_intention.cpu().numpy().tolist():
            if single_intention == 1:
                batch_first_denominator.append(ask_symptom_first_denominator)
                batch_second_tm.append(ask_symptom_second_tm)
            elif single_intention == 2:
                batch_first_denominator.append(diagnosis_first_denominator)
                batch_second_tm.append(diagnosis_second_tm)
            elif single_intention == 3:
                batch_first_denominator.append(prescribe_first_denominator)
                batch_second_tm.append(prescribe_second_tm)

        batch_first_denominator = torch.tensor(batch_first_denominator, dtype=torch.float, device=TO.device)
        batch_second_tm = torch.stack(batch_second_tm, 0)
        first_out = torch.bmm(selected_state, common_first_tm.unsqueeze(0).expand(efficient_intention_num, -1,
                                                                                  -1)) / batch_first_denominator.unsqueeze(
            -1).unsqueeze(-1)
        second_in = first_out + selected_state
        second_out = torch.bmm(second_in, batch_second_tm)  # B', 1, K_s
        ret = torch.bmm(second_out,
                        self.scientific2alias.unsqueeze(0).expand(efficient_intention_num, -1, -1))  # B', 1, K
        ret = ret.squeeze(1)  # B', K
        eps = 1e-3
        ret = ret / (ret.detach().sum(-1).unsqueeze(-1) + eps)
        # B, K
        ret_placeholder = torch.zeros(state.size(0), self.alias_entity_num, device=TO.device, dtype=torch.float)
        ret_placeholder[efficient_indices] = ret  # B, K
        ret = ret_placeholder
        return ret

class GraphLoader:
    def __init__(self, alias2scientific_filename, entity2type_filename, joint_graph_filename, know_vocab: Vocab):
        self.alias2scientific_filename = alias2scientific_filename
        self.entity2type_filename = entity2type_filename
        self.joint_graph_filename = joint_graph_filename
        self.know_vocab = know_vocab

        self.alias2scientific = json.load(open(self.alias2scientific_filename))
        self.entity2type = json.load(open(self.entity2type_filename))
        self.joint_graph = json.load(open(self.joint_graph_filename))["graph"]

        scientific2id = dict(zip([DatasetOption.PreventWord.PAD] + list(self.entity2type.keys()),
                                 list(range(len(self.entity2type) + 1))))

        # alias_index2scientific_index \in \mathcal{R}^{K_a}, np.long
        alias_index2scientific_index = []
        for know_word_index, know_word in enumerate(know_vocab.word_list):
            tmp = scientific2id.get(know_word if TO.task == "meddg" else self.alias2scientific.get(know_word, know_word))
            alias_index2scientific_index.append(tmp)
        self.alias_index2scientific_index = np.asarray(alias_index2scientific_index)

        # scientific_index2alias_index \in \mathcal{R}^{K_s \times K_a}, np.float
        scientific_index2alias_index = []
        for scientific, scientific_id in sorted(scientific2id.items(), key=lambda x: x[1]):
            signal = np.asarray(self.alias_index2scientific_index == scientific_id, dtype=np.float)
            scientific_index2alias_index.append(signal / (signal.sum() + 1e-24))
        self.scientific_index2alias_index = np.asarray(scientific_index2alias_index)

        # entity2type_index \in \mathcal{R}^{K_s}, np.float
        entity2type_list = [self.entity2type.get(x[0], "None") for x in sorted(scientific2id.items(), key=lambda x: x[1])]
        self.entity_type2type_index = {"药物": 0, "疾病": 1, "诊疗": 2, "症状": 3, "None": 4}
        self.scientific_entity2type_index = [self.entity_type2type_index[entity_type] for entity_type in
                                             entity2type_list]

        # type_matrix \in \mathcal{R}^{K_a \times T_knowledge_word_type}
        type_matrix = []
        for know_word_index, know_word in enumerate(know_vocab.word_list):
            type_str = self.entity2type.get(self.alias2scientific.get(know_word, know_word), "None")
            tmp = [0, 0, 0, 0]
            type_index = self.entity_type2type_index[type_str]
            if type_index < 4:
                tmp[type_index] = 1.0
            type_matrix.append(tmp)
        self.type_matrix = np.asarray(type_matrix, dtype=np.float)

        # relation2type
        self.relation_type2type_index = {"contains_symptom": 0, "symptom_of": 1,
                                         "caused_by": 0, "lead_to": 1,
                                         "to_treat": 2, "treated_by": 3,
                                         "checked_for": 2, "check_item": 3}

        # adjacent_matrix \in \mathcal{R}^{4 \times K_s \times K_s}
        self.adjacent_matrix = np.zeros(shape=(4, len(scientific2id), len(scientific2id)), dtype=np.long)
        for head, relation, tail in self.joint_graph:
            head_id, tail_id = scientific2id[head], scientific2id[tail]
            if relation not in self.relation_type2type_index:
                continue
            relation_id = self.relation_type2type_index[relation]
            self.adjacent_matrix[relation_id, head_id, tail_id] = 1

        print("GraphLoader load finished")


if __name__ == '__main__':
    vocab_df = pd.read_csv(DatasetOption.vocab_csv_filename.format(dataset="meddg"))
    know_vocab_items = [DatasetOption.PreventWord.PAD] + [str(x) for x in
                                                          list(vocab_df[vocab_df["Is_know"] > 0]["Word"])]
    know_vocab = Vocab(know_vocab_items, DatasetOption.know_vocab_size)

    alias2scientific_filename = "data/meddg_joint_graph/alias2scientific.json"
    entity2type_filename = "data/meddg_joint_graph/entity2type.json"
    joint_graph_filename = "data/meddg_joint_graph/graph.json"
    gl = GraphLoader(alias2scientific_filename,
                     entity2type_filename,
                     joint_graph_filename,
                     know_vocab)
    #GPN = GraphPolicyNetwork(torch.from_numpy(gl.adjacent_matrix),
    #        torch.from_numpy(gl.type_matrix),
    #        torch.from_numpy(gl.alias_index2scientific_index.astype("float32")),
    #        torch.from_numpy(gl.scientific_index2alias_index))
