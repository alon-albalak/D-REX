import logging
import os
import sys
from tqdm import tqdm
import torch
import numpy as np
from utils import utils, data_utils, analysis_utils

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def main(**kwargs):
    assert kwargs["threshold1"] and kwargs["threshold2"]
    assert kwargs["data_split"]

    if kwargs["seed"] != -1:
        utils.set_seed(kwargs["seed"])

    # load pre-trained initial relation ranker
    initial_ranker_config_class, initial_ranker_model_class, initial_ranker_tokenizer_class = utils.MODEL_CLASSES[
        kwargs["relation_extraction_ranker_model_class"]
    ]
    initial_ranker_config = initial_ranker_config_class.from_pretrained(kwargs["relation_extraction_ranker_path"])
    initial_ranker_tokenizer = initial_ranker_tokenizer_class.from_pretrained(kwargs["relation_extraction_ranker_base_model"])

    # load pre-trained explanation policy
    expl_config_class, expl_model_class, expl_tokenizer_class = utils.MODEL_CLASSES[kwargs["explanation_policy_model_class"]]
    expl_config = expl_config_class.from_pretrained(kwargs["explanation_policy_path"])
    expl_tokenizer = expl_tokenizer_class.from_pretrained(kwargs["explanation_policy_base_model"])

    # load pre-trained relation re-ranker
    reranker_config_class, reranker_model_class, reranker_tokenizer_class = utils.MODEL_CLASSES[kwargs["relation_extraction_reranker_model_class"]]
    reranker_config = reranker_config_class.from_pretrained(kwargs["relation_extraction_reranker_path"])
    reranker_tokenizer = reranker_tokenizer_class.from_pretrained(kwargs["relation_extraction_reranker_base_model"])

    kwargs["relation_extraction_pretraining"] = True
    test_dataset = data_utils.get_data(initial_ranker_tokenizer, include_samples=True, include_relations_in_sample=True, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=kwargs["gpu_batch_size"],
        shuffle=True,
        collate_fn=data_utils.default_collate,
    )

    # load models
    initial_ranker_model = initial_ranker_model_class.from_pretrained(kwargs["relation_extraction_ranker_path"])
    initial_ranker_model.to(kwargs["device"])
    initial_ranker_model.eval()
    expl_model = expl_model_class.from_pretrained(kwargs["explanation_policy_path"])
    expl_model.to(kwargs["device"])
    expl_model.eval()
    reranker_model = reranker_model_class.from_pretrained(kwargs["relation_extraction_reranker_path"])
    reranker_model.to(kwargs["device"])
    reranker_model.eval()

    logger.info(f"******** Evaluating ************")
    logger.info(f"  Num samples: {len(test_dataset)}")

    outputs, logits, labels = [], [], []
    pbar = tqdm(enumerate(test_dataloader), total=len(test_dataloader))
    for step, initial_ranker_batch in pbar:
        initial_ranker_batch = utils.batch_to_device(initial_ranker_batch, kwargs["device"])
        guids, input_ids, attention_mask, segment_ids, relations, start_trigger_ids, end_trigger_ids, samples = initial_ranker_batch
        samples = [{k: v[i] for k, v in samples.items()} for i in range(len(samples["dialogue"]))]

        if initial_ranker_model_class.__name__ == "relation_extraction_RoBERTa":
            initial_ranker_inputs = (input_ids, attention_mask)
        else:
            initial_ranker_inputs = (input_ids, attention_mask, segment_ids)

        # rank relations w/out explanations
        with torch.no_grad():
            initial_ranker_logits = initial_ranker_model(*initial_ranker_inputs)
        initial_sorted_rankings = initial_ranker_model.sort_relations_from_logits(initial_ranker_logits)

        # predict explanation for top k relations
        for i, (initial_sorted_rankings_idxs, sample, guid, sample_relations) in enumerate(
            zip(initial_sorted_rankings, samples, guids, relations.detach().cpu().numpy())
        ):
            topk_initial_ranking_idxs = initial_sorted_rankings_idxs[: kwargs["topk_relations"]]

            # create batch for explanation policy
            explanation_prediction_samples = []
            for initial_ranking_idx in topk_initial_ranking_idxs:
                explanation_prediction_samples.append(
                    data_utils.Sample(guid=guid, dialogue=sample["dialogue"], head=sample["head"], tail=sample["tail"], relations=initial_ranking_idx)
                )
            expl_pred_features = data_utils.convert_samples_to_features(
                samples=explanation_prediction_samples,
                max_sequence_len=kwargs["max_sequence_len"],
                tokenizer=expl_tokenizer,
                append_relation=True,
                logging=False,
            )
            expl_dataset = data_utils.get_data(expl_tokenizer, features=expl_pred_features, include_relation_entities_mask=True)
            expl_batch = utils.batch_to_device(expl_dataset.data[1:], kwargs["device"])
            expl_input_ids, expl_attention_mask, expl_segment_ids, _, _, _, relation_entities_mask = expl_batch

            if expl_model_class.__name__ == "explanation_policy_RoBERTa":
                expl_inputs = (expl_input_ids, expl_attention_mask)
            else:
                expl_inputs = (expl_input_ids, expl_attention_mask, expl_segment_ids)

            with torch.no_grad():
                expl_start_logits, expl_end_logits = expl_model(*expl_inputs)
            expl_start_pred_idxs = torch.argmax(expl_start_logits, dim=1)
            expl_end_pred_idxs = torch.argmax(expl_end_logits, dim=1)

            # convert predicted start/end indexes to tokens
            expl_preds = []
            for expl_start, expl_end, expl_input_id, rel_ent_mask in zip(
                expl_start_pred_idxs, expl_end_pred_idxs, expl_input_ids, relation_entities_mask
            ):
                first_real_token = (rel_ent_mask == 1).nonzero(as_tuple=True)[0][0]
                if expl_start < first_real_token:
                    expl_preds.append("")
                else:
                    expl_preds.append(
                        data_utils.decode_normalize_tokens(
                            expl_input_id.squeeze(), expl_start, torch.clamp(expl_end, max=expl_start + 10), expl_tokenizer
                        )
                    )

            # create batch for relation reranking
            relation_extraction_samples = []
            for explanation, initial_ranking_idx in zip(expl_preds, topk_initial_ranking_idxs):
                relation_extraction_samples.append(
                    data_utils.Sample(
                        guid=guid,
                        dialogue=sample["dialogue"],
                        head=sample["head"],
                        tail=sample["tail"],
                        relations=initial_ranking_idx,
                        triggers=explanation,
                    )
                )
            # Calculate re-ranked relation extraction logits
            relation_extraction_features = data_utils.convert_samples_to_features(
                samples=relation_extraction_samples,
                max_sequence_len=kwargs["max_sequence_len"],
                tokenizer=reranker_tokenizer,
                append_trigger_tokens=True,
                logging=False,
            )
            relation_extraction_dataset = data_utils.get_data(reranker_tokenizer, features=relation_extraction_features)
            relation_extraction_relations = relation_extraction_dataset.data[4].int()
            relation_extraction_batch = utils.batch_to_device(relation_extraction_dataset.data[1:4], kwargs["device"])
            relation_extraction_input_ids, relation_extraction_attention_mask, relation_extraction_segment_ids = relation_extraction_batch

            if reranker_model_class.__name__ == "relation_extraction_RoBERTa":
                relation_extraction_inputs = (relation_extraction_input_ids, relation_extraction_attention_mask)
            else:
                relation_extraction_inputs = (relation_extraction_input_ids, relation_extraction_attention_mask, relation_extraction_segment_ids)

            with torch.no_grad():
                relation_logits = reranker_model(*relation_extraction_inputs)

            explained_logits = []
            for e, l in zip(expl_preds, relation_logits.detach().cpu().numpy()):
                if e:
                    explained_logits.append(l)
            if explained_logits:
                explained_logits.append(initial_ranker_logits.detach().cpu().numpy()[i])
                logits.append(np.mean(np.stack(explained_logits, axis=0), axis=0))
            else:
                logits.append(initial_ranker_logits.detach().cpu().numpy()[i])

            labels.append(sample_relations)
            outputs.append(
                [
                    guid,
                    [utils.get_relation(i) for i in topk_initial_ranking_idxs],
                    [(start.item(), end.item()) for start, end in zip(expl_start_pred_idxs, expl_end_pred_idxs)],
                    expl_preds,
                    sample["head"],
                    sample["tail"],
                ]
            )

    topk = [1, 3, 5]
    precision, recall, f1 = analysis_utils.get_f1_from_logits(logits, labels, kwargs["threshold1"], kwargs["threshold2"])
    hits_at_k = analysis_utils.get_hits_at_k(topk, logits, labels)
    MRR = analysis_utils.get_MRR(logits, labels)

    logger.info(f"PR: {precision:0.4f}, RE: {recall:0.4f}")
    logger.info(f"F1: {f1:0.4f}")
    logger.info(f"Hits @ {topk}: {hits_at_k}")
    logger.info(f"MRR: {MRR:0.4f}")

    with open(os.path.join(kwargs["explanation_policy_path"], f"{kwargs['data_split']}_drex_outputs.txt"), "w") as f:
        for output in outputs:
            f.write("\t".join([str(o) for o in output]) + "\n")


if __name__ == "__main__":
    args = utils.parse_drex_args()
    main(**args)
