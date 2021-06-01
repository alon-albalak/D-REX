import logging
import argparse
import random
import numpy as np
from models.explanation_policy_BERT import explanation_policy_BERT
from models.explanation_policy_RoBERTa import explanation_policy_RoBERTa
from models.relation_extraction_BERT import relation_extraction_BERT
from models.relation_extraction_RoBERTa import relation_extraction_RoBERTa
from transformers import BertConfig, BertTokenizer, RobertaConfig, RobertaTokenizer
import torch


logger = logging.getLogger(__name__)

MODEL_CLASSES = {
    "explanation_policy": (BertConfig, explanation_policy_BERT, BertTokenizer),
    "relation_extraction": (BertConfig, relation_extraction_BERT, BertTokenizer),
    # "joint_model": (BertConfig, joint_explanation_RE_BERT, BertTokenizer),
    "relation_extraction_roberta": (RobertaConfig, relation_extraction_RoBERTa, RobertaTokenizer),
    "explanation_policy_roberta": (RobertaConfig, explanation_policy_RoBERTa, RobertaTokenizer),
}

relations = {
    1: "per:positive_impression",
    2: "per:negative_impression",
    3: "per:acquaintance",
    4: "per:alumni",
    5: "per:boss",
    6: "per:subordinate",
    7: "per:client",
    8: "per:dates",
    9: "per:friends",
    10: "per:girl/boyfriend",
    11: "per:neighbor",
    12: "per:roommate",
    13: "per:children",
    14: "per:other_family",
    15: "per:parents",
    16: "per:siblings",
    17: "per:spouse",
    18: "per:place_of_residence",
    19: "per:place_of_birth",  # does not exist in training set
    20: "per:visited_place",
    21: "per:origin",
    22: "per:employee_or_member_of",
    23: "per:schools_attended",
    24: "per:works",
    25: "per:age",
    26: "per:date_of_birth",
    27: "per:major",
    28: "per:place_of_work",
    29: "per:title",
    30: "per:alternate_names",
    31: "per:pet",
    32: "gpe:residents_of_place",
    34: "gpe:visitors_of_place",
    33: "gpe:births_in_place",  # does not exist in training set
    35: "org:employees_or_members",
    36: "org:students",
    37: "unanswerable",
}

relation_id_to_NL = {
    1: "person positive impression",
    2: "person negative impression",
    3: "person acquaintance",
    4: "person alumni",
    5: "person boss",
    6: "person subordinate",
    7: "person client",
    8: "person dates",
    9: "person friends",
    10: "person girl or boyfriend",
    11: "person neighbor",
    12: "person roommate",
    13: "person children",
    14: "person other family",
    15: "person parents",
    16: "person siblings",
    17: "person spouse",
    18: "person place of residence",
    19: "person place of birth",  # does not exist in training set
    20: "person visited place",
    21: "person origin",
    22: "person employee or member of",
    23: "person schools attended",
    24: "person works",
    25: "person age",
    26: "person date of birth",
    27: "person major",
    28: "person place of work",
    29: "person title",
    30: "person alternate names",
    31: "person pet",
    32: "location residents of place",
    34: "location visitors of place",
    33: "location births in place",  # does not exist in training set
    35: "organization employees or members",
    36: "organization students",
    37: "unanswerable",
}


def get_relation(relation_id):
    return relations[int(relation_id) + 1]


def get_relation_in_NL(relation_id):
    return relation_id_to_NL[int(relation_id) + 1]


def set_seed(seed):
    if seed != -1:
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        logger.info(f"Setting seed to {seed}")


def batch_to_device(batch, device):
    batch_on_device = []
    for element in batch:
        if isinstance(element, dict):
            # if element does not contain tensors, do not move it to device
            if isinstance(list(element.values())[0], torch.Tensor):
                batch_on_device.append({k: v.to(device) for k, v in element.items()})
            else:
                batch_on_device.append(element)
        elif isinstance(element[0], str):
            batch_on_device.append(element)
        elif isinstance(element, list) and len(element) == 1 and isinstance(element[0], dict):
            batch_on_device.append(element[0])
        elif isinstance(element[0], dict):
            batch_on_device.append(element)
        else:
            batch_on_device.append(element.to(device))
    return tuple(batch_on_device)


def parse_args():
    parser = argparse.ArgumentParser()
    # general args
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    # model args
    parser.add_argument("--model_class", type=str, default="relation_extraction_roberta")
    parser.add_argument("--model_name_or_path", type=str, default="roberta-base")
    parser.add_argument("--base_model", type=str, default="roberta-base")
    parser.add_argument("--keep_case", action="store_true")
    parser.add_argument("--dropout_rate", type=float, default=0.3)
    parser.add_argument("--max_sequence_len", type=int, default=512)
    parser.add_argument("--pos_weight_RE_samples", type=int, default=2)
    # training args
    parser.add_argument("-e", "--num_epochs", type=int, default=20)
    parser.add_argument("--effective_batch_size", type=int, default=28)
    parser.add_argument("--gpu_batch_size", type=int, default=7)
    parser.add_argument("-lr", "--learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output_dir", type=str, default="")
    # test args
    parser.add_argument("--threshold1", type=int, default=None)
    parser.add_argument("--threshold2", type=int, default=None)
    parser.add_argument("--data_split", type=str, default=None)
    parser.add_argument("--predict_trigger_for_unlabelled", action="store_true")
    # data args
    parser.add_argument("--data_path", type=str, default="data_v2")
    parser.add_argument("--num_relations", type=int, default=37)
    # explanation args
    parser.add_argument("--explanation_policy_pretraining", action="store_true")
    parser.add_argument("--rename_entities", type=bool, default=True)
    parser.add_argument("--max_pred_trigger_dist", type=int, default=10)
    # relation extraction args
    parser.add_argument("--relation_extraction_pretraining", action="store_true")
    parser.add_argument("--relation_extraction_conditioned_on_explanations", action="store_true")
    parser.add_argument("--use_predicted_explanations", action="store_true")
    parser.add_argument("--predicted_explanation_path", type=str, default="baseline_explanation_model/F1-0.6575")
    # joint model args
    # parser.add_argument("--joint_model_training", action="store_true")
    # parser.add_argument("--alpha", type=float, default=0.5)
    # parser.add_argument("--test_explanations_only")
    args = parser.parse_args()

    if "roberta" in getattr(args, "base_model"):
        setattr(args, "keep_case", True)

    if not args.cpu_only:
        setattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    else:
        setattr(args, "device", "cpu")

    return vars(args)


def parse_drex_args():
    parser = argparse.ArgumentParser()
    # general args
    parser.add_argument("--cpu_only", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    # model args
    parser.add_argument("--relation_extraction_ranker_path", type=str, default="roberta-base")
    parser.add_argument("--relation_extraction_ranker_base_model", type=str, default="roberta-base")
    parser.add_argument("--relation_extraction_ranker_model_class", type=str, default="relation_extraction_roberta")
    parser.add_argument("--explanation_policy_path", type=str, default="roberta-base")
    parser.add_argument("--explanation_policy_base_model", type=str, default="roberta-base")
    parser.add_argument("--explanation_policy_model_class", type=str, default="explanation_policy_roberta")
    parser.add_argument("--relation_extraction_reranker_path", type=str, default="roberta-base")
    parser.add_argument("--relation_extraction_reranker_base_model", type=str, default="roberta-base")
    parser.add_argument("--relation_extraction_reranker_model_class", type=str, default="relation_extraction_roberta")
    parser.add_argument("--keep_case", action="store_true")
    parser.add_argument("--max_sequence_len", type=int, default=512)
    # training args
    parser.add_argument("-e", "--num_epochs", type=int, default=30)
    parser.add_argument("--effective_batch_size", type=int, default=30)
    parser.add_argument("--gpu_batch_size", type=int, default=1)
    parser.add_argument("--expl_learning_rate", type=float, default=3e-5)
    parser.add_argument("--reranker_learning_rate", type=float, default=3e-5)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--initial_softmax_temp", type=float, default=1.0)
    parser.add_argument("--softmax_decay_ratio", type=float, default=1.1)
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--output_dir", type=str, default="")
    # policy gradient args
    parser.add_argument("--reranking_reward", action="store_true")
    parser.add_argument("--supervised_expl_loss", action="store_true")
    parser.add_argument("--leave_one_out_loss_initial_ranker", action="store_true")
    parser.add_argument("--leave_one_out_loss_reranker", action="store_true")
    # test args
    parser.add_argument("--threshold1", type=int, default=None)
    parser.add_argument("--threshold2", type=int, default=None)
    parser.add_argument("--data_split", type=str, default=None)
    # data args
    parser.add_argument("--data_path", type=str, default="data_v2")
    parser.add_argument("--num_relations", type=int, default=37)
    # misc. args
    parser.add_argument("--topk_relations", type=int, default=5)

    args = parser.parse_args()

    if "roberta" in getattr(args, "relation_extraction_ranker_base_model"):
        setattr(args, "keep_case", True)

    if not args.cpu_only:
        setattr(args, "device", "cuda" if torch.cuda.is_available() else "cpu")
    else:
        setattr(args, "device", "cpu")

    return vars(args)
