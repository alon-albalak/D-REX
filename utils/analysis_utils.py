from utils import utils
import numpy as np


def get_f1_from_logits(logits, labels, threshold1=None, threshold2=None):
    def get_predictions(all_logits, threshold1=0.5, threshold2=0.4):
        all_preds = []
        for sample_logits in all_logits:
            sample_preds = []
            max_logit, max_idx = -1, -1
            for idx, l in enumerate(sample_logits):
                if l > threshold1:
                    sample_preds.append(idx)
                if l > max_logit:
                    max_logit = l
                    max_idx = idx
            if not sample_preds:
                if max_logit <= threshold2:
                    sample_preds = [36]
                else:
                    sample_preds.append(max_idx)
            all_preds.append(sample_preds)
        return all_preds

    def calculate_f1(preds, labels):
        total_GT, total_correct, total_pred = 0, 0, 0
        for pred, label in zip(preds, labels):
            for l in label:
                if l != 36:
                    total_GT += 1
                    if l in pred:
                        total_correct += 1
            for p in pred:
                if p != 36:
                    total_pred += 1
        precision = 1 if total_pred == 0 else total_correct / total_pred
        recall = 0 if total_GT == 0 else total_correct / total_GT
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0
        return precision, recall, f1

    logits = 1 / (1 + np.exp(-np.array(logits)))
    labels = [np.nonzero(l)[0] for l in labels]

    if threshold1 and threshold2:
        preds = get_predictions(logits, threshold1=threshold1 / 100, threshold2=threshold2 / 100)
        precision, recall, f1 = calculate_f1(preds, labels)
        return precision, recall, f1

    best_f1, best_pr, best_re, best_T1, best_T2 = 0, 0, 0, 0, 0
    for thresh1 in range(0, 100, 10):
        for thresh2 in range(0, thresh1, 5):
            preds = get_predictions(logits, threshold1=thresh1 / 100, threshold2=thresh2 / 100)
            precision, recall, f1 = calculate_f1(preds, labels)
            if f1 > best_f1:
                best_f1 = f1
                best_pr = precision
                best_re = recall
                best_T1 = thresh1 / 100
                best_T2 = thresh2 / 100
    return best_f1, best_pr, best_re, best_T1, best_T2


def get_hits_at_k(k_list, logits, labels):
    if isinstance(k_list, int) or isinstance(k_list, float):
        k_list = [k_list]
    hits = [0 for _ in k_list]

    sorted_logits_idxs = np.argsort(-np.array(logits), axis=1)
    labels = [np.nonzero(l)[0] for l in labels]
    total_samples = len(labels)

    for logit_idx, label in zip(sorted_logits_idxs, labels):
        for rank, idx in enumerate(logit_idx):
            if idx in label:
                for k_idx, k in enumerate(k_list):
                    if rank < k:
                        hits[k_idx] += 1

    hits_at_k = [h / total_samples for h in hits]
    return hits_at_k


def get_MRR(logits, labels):
    reciprocal_ranks = []

    sorted_logits_idxs = np.argsort(-np.array(logits), axis=1)
    labels = [np.nonzero(l)[0] for l in labels]

    for logit_idx, label in zip(sorted_logits_idxs, labels):
        for rank, idx in enumerate(logit_idx):
            if idx in label:
                reciprocal_ranks.append(1 / (rank + 1))

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


def load_explanation_results(result_path):
    results = []
    with open(result_path, "r") as f:
        for row in f:
            results.append(row.replace("\n", "").split("\t"))
    return results


def parse_DREX_explanations(result_path):
    res = load_explanation_results(result_path)
    guids = []
    relations = []
    explanation_idxs = []
    explanations = []
    for r in res:
        sample_guid = r[0]
        sample_relations = r[1][1:-1].replace("'", "").replace(" ", "").split(",")
        sample_explanations = [expl.strip() for expl in r[3][1:-1].replace("'", "").split(",")]
        idxs = r[2][1:-1].replace("(", "").replace(")", "").split(",")
        sample_explanation_idxs = []
        for i in range(len(idxs) // 2):
            sample_explanation_idxs.append([int(idxs[i * 2]), int(idxs[(i * 2) + 1])])
        guids.append(sample_guid)
        relations.append(sample_relations)
        explanation_idxs.append(sample_explanation_idxs)
        explanations.append(sample_explanations)
    return guids, relations, explanation_idxs, explanations


def get_explanations_per_relation(result_path):
    guids, relations, explanation_idxs, _ = parse_DREX_explanations(result_path)
    all_relations = utils.relations.values()
    predicted_explanation_count = {rel: 0 for rel in all_relations}
    relation_count = {rel: 0 for rel in all_relations}

    for rel, expl in zip(relations, explanation_idxs):
        for r, e in zip(rel, expl):
            relation_count[r] += 1
            if not (e[0] == 0 or e[1] < e[0]):
                predicted_explanation_count[r] += 1
    return predicted_explanation_count, relation_count
