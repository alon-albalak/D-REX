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


def evaluate(
    dataloader,
    initial_ranker_model,
    initial_ranker_model_class,
    expl_model,
    expl_model_class,
    reranker_model,
    reranker_model_class,
    expl_tokenizer,
    reranker_tokenizer,
    **kwargs,
):
    logits = []
    labels = []
    outputs = []
    logger.info("******** EVALUATION ********")
    pbar = tqdm(enumerate(dataloader), total=len(dataloader))
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

            # create batch for relation extraction
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
    f1, precision, recall, bestT1, bestT2 = analysis_utils.get_f1_from_logits(logits, labels)
    hits_at_k = analysis_utils.get_hits_at_k(topk, logits, labels)
    MRR = analysis_utils.get_MRR(logits, labels)
    logger.info("*** Re-Ranked ***")
    logger.info(f"PR: {precision:0.4f}, RE: {recall:0.4f}")
    logger.info(f"F1: {f1:0.4f}, BEST T1: {bestT1:0.4f}, BEST T2: {bestT2:0.4f}")
    logger.info(f"Hits @ {topk}: {hits_at_k}")
    logger.info(f"MRR: {MRR:0.4f}")
    return f1, bestT1, bestT2, outputs


def main(**kwargs):
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

    # load training data
    kwargs["data_split"] = "train"
    kwargs["relation_extraction_pretraining"] = True
    train_dataset = data_utils.get_data(initial_ranker_tokenizer, include_samples=True, include_relations_in_sample=True, **kwargs)
    train_dataloader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=kwargs["gpu_batch_size"],
        shuffle=True,
        collate_fn=data_utils.default_collate,
    )

    # load dev data
    kwargs["data_split"] = "dev"
    dev_dataset = data_utils.get_data(initial_ranker_tokenizer, include_samples=True, **kwargs)
    dev_dataloader = torch.utils.data.DataLoader(
        dev_dataset,
        batch_size=kwargs["gpu_batch_size"],
        shuffle=False,
        collate_fn=data_utils.default_collate,
    )

    # load models
    initial_ranker_model = initial_ranker_model_class.from_pretrained(kwargs["relation_extraction_ranker_path"])
    initial_ranker_model.to(kwargs["device"])
    initial_ranker_model.eval()
    expl_model = expl_model_class.from_pretrained(kwargs["explanation_policy_path"])
    expl_model.to(kwargs["device"])
    reranker_model = reranker_model_class.from_pretrained(kwargs["relation_extraction_reranker_path"])
    reranker_model.to(kwargs["device"])

    # load optimizers
    gradient_accumulation_steps = kwargs["effective_batch_size"] / kwargs["gpu_batch_size"]
    total_optimization_steps = kwargs["num_epochs"] * (len(train_dataloader) // gradient_accumulation_steps)
    expl_optimizer = torch.optim.AdamW(expl_model.parameters(), lr=kwargs["expl_learning_rate"])
    reranker_optimizer = torch.optim.AdamW(reranker_model.parameters(), lr=kwargs["reranker_learning_rate"])

    if kwargs["fp16"]:
        expl_scaler = torch.cuda.amp.GradScaler()
        reranker_scaler = torch.cuda.amp.GradScaler()

    logger.info("\n******** Training ********")
    logger.info(f"    Num samples: {len(train_dataset)}")
    logger.info(f"    Num epochs: {kwargs['num_epochs']}")
    logger.info(f"    Total optimization steps: {total_optimization_steps}")

    softmax_temp = kwargs["initial_softmax_temp"]
    best_f1, best_T1, best_T2 = 0, 0, 0
    reranking_rewards = []
    LOO_rewards_initial_ranker, LOO_rewards_reranker = [], []
    for epoch in range(kwargs["num_epochs"]):
        logger.info(f"EPOCH: {epoch+1}")
        expl_model.train()
        reranker_model.train()

        (
            total_reranking_loss,
            total_expl_policy_reranking_loss,
            total_expl_policy_LOO_loss_initial_ranker,
            total_expl_policy_LOO_loss_reranker,
            total_supervised_expl_loss,
        ) = (0, 0, 0, 0, 0)

        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for step, initial_ranker_batch in pbar:
            initial_ranker_batch = utils.batch_to_device(initial_ranker_batch, kwargs["device"])
            guids, input_ids, attention_mask, segment_ids, relations, start_trigger_ids, end_trigger_ids, samples = initial_ranker_batch
            samples = [
                {k: v[i] if not (len(v) == 1 and isinstance(v[0], torch.Tensor)) else v[0][i] for k, v in samples.items()}
                for i in range(len(samples["dialogue"]))
            ]
            if initial_ranker_model_class.__name__ == "relation_extraction_RoBERTa":
                initial_ranker_inputs = (input_ids, attention_mask)
            else:
                initial_ranker_inputs = (input_ids, attention_mask, segment_ids)

            # calculate initial relation rankings
            with torch.no_grad():
                if kwargs["fp16"]:
                    with torch.cuda.amp.autocast():
                        initial_ranker_logits = initial_ranker_model(*initial_ranker_inputs)
                        initial_ranker_loss = torch.sum(initial_ranker_model.calculate_loss_from_logits(initial_ranker_logits, relations), dim=1)

                else:
                    initial_ranker_logits = initial_ranker_model(*initial_ranker_inputs)
                    initial_ranker_loss = torch.sum(initial_ranker_model.calculate_loss_from_logits(initial_ranker_logits, relations), dim=1)

            initial_sorted_rankings = initial_ranker_model.sort_relations_from_logits(initial_ranker_logits)

            # predict explanation for top k relations
            for i, (initial_sorted_rankings_idxs, sample, guid, sample_relations) in enumerate(
                zip(initial_sorted_rankings, samples, guids, relations)
            ):
                topk_initial_ranking_idxs = initial_sorted_rankings_idxs[: kwargs["topk_relations"]]

                # create batch for explanation policy
                explanation_prediction_samples = []
                samples_with_trigger_mask = []

                ground_truth_relation_in_topk = False
                for initial_ranking_idx in topk_initial_ranking_idxs:
                    if initial_ranking_idx in sample["relations"]:
                        ground_truth_relation_in_topk = True
                        samples_with_trigger_mask.append(1)
                        trigger_for_sample = sample["triggers"][sample["relations"].index(initial_ranking_idx)]
                    else:
                        samples_with_trigger_mask.append(0)
                        trigger_for_sample = ""

                    explanation_prediction_samples.append(
                        data_utils.Sample(
                            guid=guid,
                            dialogue=sample["dialogue"],
                            head=sample["head"],
                            tail=sample["tail"],
                            relations=initial_ranking_idx,
                            triggers=trigger_for_sample,
                        )
                    )

                # since we only calculate relation extraction loss on ground truth relations, if the model did not rank the GT relation
                #   in the topk, just skip this datum
                if not ground_truth_relation_in_topk:
                    continue

                # create explanation policy data
                expl_pred_features = data_utils.convert_samples_to_features(
                    samples=explanation_prediction_samples,
                    max_sequence_len=kwargs["max_sequence_len"],
                    tokenizer=expl_tokenizer,
                    predict_triggers=True,
                    append_relation=True,
                    logging=False,
                )
                expl_dataset = data_utils.get_data(expl_tokenizer, features=expl_pred_features, include_relation_entities_mask=True)
                expl_batch = utils.batch_to_device(expl_dataset.data[1:], kwargs["device"])
                expl_input_ids, expl_attention_mask, expl_segment_ids, _, expl_start_labels, expl_end_labels, relation_entities_mask = expl_batch
                if expl_model_class.__name__ == "explanation_policy_RoBERTa":
                    expl_inputs = (expl_input_ids, expl_attention_mask)
                else:
                    expl_inputs = (expl_input_ids, expl_attention_mask, expl_segment_ids)

                # predict explanations conditioned on relations
                if kwargs["fp16"]:
                    with torch.cuda.amp.autocast():
                        expl_start_logits, expl_end_logits = expl_model(*expl_inputs)
                        if kwargs["supervised_expl_loss"]:
                            supervised_expl_loss = expl_model.calculate_loss_from_logits(
                                expl_start_logits, expl_start_labels
                            ) + expl_model.calculate_loss_from_logits(expl_end_logits, expl_end_labels)
                else:
                    expl_start_logits, expl_end_logits = expl_model(*expl_inputs)
                    if kwargs["supervised_expl_loss"]:
                        supervised_expl_loss = expl_model.calculate_loss_from_logits(
                            expl_start_logits, expl_start_labels
                        ) + expl_model.calculate_loss_from_logits(expl_end_logits, expl_end_labels)
                if kwargs["supervised_expl_loss"]:
                    samples_with_trigger_mask = torch.BoolTensor(samples_with_trigger_mask).to(kwargs["device"])
                    supervised_expl_loss = torch.sum(supervised_expl_loss[samples_with_trigger_mask])

                # mask the [CLS] RELATION NAME [SEP] ENTITY1 [SEP] ENTITY2 [SEP] tokens
                min_start_logit = torch.min(expl_start_logits)
                min_end_logit = torch.min(expl_end_logits)
                expl_start_logits = expl_start_logits + (1 - relation_entities_mask) * min_start_logit
                expl_end_logits = expl_end_logits + (1 - relation_entities_mask) * min_end_logit

                expl_start_softmax = torch.nn.functional.softmax(expl_start_logits / softmax_temp, dim=1)
                expl_end_softmax = torch.nn.functional.softmax(expl_end_logits / softmax_temp, dim=1)

                m_start = torch.distributions.Categorical(expl_start_softmax)
                m_end = torch.distributions.Categorical(expl_end_softmax)
                expl_start_pred_idxs = m_start.sample()
                expl_end_pred_idxs = m_end.sample()

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
                # calculate leave one out reward on the initial ranker
                if kwargs["leave_one_out_loss_initial_ranker"]:
                    relation_extraction_features = data_utils.convert_samples_to_features(
                        samples=relation_extraction_samples,
                        max_sequence_len=kwargs["max_sequence_len"],
                        tokenizer=initial_ranker_tokenizer,
                        logging=False,
                        mask_triggers=True,
                    )
                    relation_extraction_dataset = data_utils.get_data(initial_ranker_tokenizer, features=relation_extraction_features)
                    relation_extraction_relations = relation_extraction_dataset.data[4].int()
                    relation_extraction_batch = utils.batch_to_device(relation_extraction_dataset.data[1:4], kwargs["device"])
                    relation_extraction_input_ids, relation_extraction_attention_mask, relation_extraction_segment_ids = relation_extraction_batch
                    sample_has_GT_relation = torch.BoolTensor([1 if sample_relations[idx] else 0 for idx in relation_extraction_relations])
                    relation_labels = []
                    labels_w_GT_relation = []

                    for s, r in zip(sample_has_GT_relation, relation_extraction_relations):
                        label = [0 for _ in range(kwargs["num_relations"])]
                        label[r] = 1
                        relation_labels.append(label)
                        if s:
                            labels_w_GT_relation.append(label)
                    relation_labels = torch.tensor(relation_labels, dtype=torch.float).to(kwargs["device"])
                    if labels_w_GT_relation:
                        labels_w_GT_relation = torch.tensor(labels_w_GT_relation, dtype=torch.float).to(kwargs["device"])

                        if initial_ranker_model.__name__ == "relation_extraction_RoBERTa":
                            relation_extraction_inputs = (relation_extraction_input_ids, relation_extraction_attention_mask)
                        else:
                            relation_extraction_inputs = (
                                relation_extraction_input_ids,
                                relation_extraction_attention_mask,
                                relation_extraction_segment_ids,
                            )

                        with torch.no_grad():
                            if kwargs["fp16"]:
                                with torch.cuda.camp.autocast():
                                    relation_logits = initial_ranker_model(*relation_extraction_inputs)
                                    logits_w_GT_relation = relation_logits[sample_has_GT_relation]
                                    relation_loss_leave_one_out_initial_ranker = torch.sum(
                                        initial_ranker_model.calculate_loss_from_logits(logits_w_GT_relation, labels_w_GT_relation), dim=1
                                    )
                            else:
                                relation_logits = initial_ranker_model(*relation_extraction_inputs)
                                logits_w_GT_relation = relation_logits[sample_has_GT_relation]
                                relation_loss_leave_one_out_initial_ranker = torch.sum(
                                    initial_ranker_model.calculate_loss_from_logits(logits_w_GT_relation, labels_w_GT_relation), dim=1
                                )

                            leave_one_out_reward_initial_ranker = relation_loss_leave_one_out_initial_ranker - initial_ranker_loss[i]
                            LOO_rewards_initial_ranker.extend(leave_one_out_reward_initial_ranker.detach().cpu().numpy().flatten())

                # calculate reranking reward and loss
                relation_extraction_features = data_utils.convert_samples_to_features(
                    samples=relation_extraction_samples,
                    max_sequence_len=kwargs["max_sequence_len"],
                    tokenizer=reranker_tokenizer,
                    append_trigger_tokens=True,
                    logging=False,
                    triggers_pretokenized=True,
                )
                relation_extraction_dataset = data_utils.get_data(reranker_tokenizer, features=relation_extraction_features)
                relation_extraction_relations = relation_extraction_dataset.data[4].int()
                relation_extraction_batch = utils.batch_to_device(relation_extraction_dataset.data[1:4], kwargs["device"])
                relation_extraction_input_ids, relation_extraction_attention_mask, relation_extraction_segment_ids = relation_extraction_batch
                sample_has_GT_relation = torch.BoolTensor([1 if sample_relations[idx] else 0 for idx in relation_extraction_relations])
                relation_labels = []
                labels_w_GT_relation = []
                for s, r in zip(sample_has_GT_relation, relation_extraction_relations):
                    label = [0 for _ in range(kwargs["num_relations"])]
                    label[r] = 1
                    relation_labels.append(label)
                    if s:
                        labels_w_GT_relation.append(label)
                relation_labels = torch.tensor(relation_labels, dtype=torch.float).to(kwargs["device"])
                if labels_w_GT_relation:
                    labels_w_GT_relation = torch.tensor(labels_w_GT_relation, dtype=torch.float).to(kwargs["device"])

                    if reranker_model_class.__name__ == "relation_extraction_RoBERTa":
                        relation_extraction_inputs = (relation_extraction_input_ids, relation_extraction_attention_mask)
                    else:
                        relation_extraction_inputs = (
                            relation_extraction_input_ids,
                            relation_extraction_attention_mask,
                            relation_extraction_segment_ids,
                        )

                    if kwargs["fp16"]:
                        with torch.cuda.amp.autocast():
                            relation_logits = reranker_model(*relation_extraction_inputs)
                            logits_w_GT_relation = relation_logits[sample_has_GT_relation]
                            reranking_loss = torch.sum(reranker_model.calculate_loss_from_logits(logits_w_GT_relation, labels_w_GT_relation))

                            reranking_loss = reranking_loss / (kwargs["gpu_batch_size"] * gradient_accumulation_steps)
                        reranker_scaler.scale(reranking_loss).backward()
                        total_reranking_loss += reranking_loss.item()

                    else:
                        relation_logits = reranker_model(*relation_extraction_inputs)
                        logits_w_GT_relation = relation_logits[sample_has_GT_relation]
                        reranking_loss = torch.sum(reranker_model.calculate_loss_from_logits(logits_w_GT_relation, labels_w_GT_relation))
                        reranking_loss = reranking_loss / (kwargs["gpu_batch_size"] * gradient_accumulation_steps)
                        reranking_loss.backward()
                        total_reranking_loss += reranking_loss.item()

                    with torch.no_grad():
                        reranking_loss_for_reward = torch.sum(
                            reranker_model.calculate_loss_from_logits(
                                reranker_model(*relation_extraction_inputs),
                                relation_labels,
                            ),
                            dim=1,
                        )

                        reranking_reward = initial_ranker_loss[i].repeat(kwargs["topk_relations"]) - reranking_loss_for_reward
                        reranking_rewards.extend(reranking_reward[sample_has_GT_relation].detach().cpu().numpy().flatten())

                else:
                    reranking_reward = 0

                # calculate leave one out reward on the re-ranker
                if kwargs["leave_one_out_loss_reranker"]:
                    relation_extraction_features = data_utils.convert_samples_to_features(
                        samples=relation_extraction_samples,
                        max_sequence_len=kwargs["max_sequence_len"],
                        tokenizer=reranker_tokenizer,
                        logging=False,
                        mask_triggers=True,
                        append_trigger_tokens=True,
                    )
                    relation_extraction_dataset = data_utils.get_data(reranker_tokenizer, features=relation_extraction_features)
                    relation_extraction_relations = relation_extraction_dataset.data[4].int()
                    relation_extraction_batch = utils.batch_to_device(relation_extraction_dataset.data[1:4], kwargs["device"])
                    relation_extraction_input_ids, relation_extraction_attention_mask, relation_extraction_segment_ids = relation_extraction_batch
                    sample_has_GT_relation = torch.BoolTensor([1 if sample_relations[idx] else 0 for idx in relation_extraction_relations])
                    relation_labels = []
                    labels_w_GT_relation = []
                    for s, r in zip(sample_has_GT_relation, relation_extraction_relations):
                        label = [0 for _ in range(kwargs["num_relations"])]
                        label[r] = 1
                        relation_labels.append(label)
                        if s:
                            labels_w_GT_relation.append(label)
                    relation_labels = torch.tensor(relation_labels, dtype=torch.float).to(kwargs["device"])
                    if labels_w_GT_relation:
                        labels_w_GT_relation = torch.tensor(labels_w_GT_relation, dtype=torch.float).to(kwargs["device"])

                        if reranker_model_class.__name__ == "relation_extraction_RoBERTa":
                            relation_extraction_inputs = (relation_extraction_input_ids, relation_extraction_attention_mask)
                        else:
                            relation_extraction_inputs = (
                                relation_extraction_input_ids,
                                relation_extraction_attention_mask,
                                relation_extraction_segment_ids,
                            )

                        with torch.no_grad():
                            if kwargs["fp16"]:
                                with torch.cuda.amp.autocast():
                                    relation_logits = reranker_model(*relation_extraction_inputs)
                                    logits_w_GT_relation = relation_logits[sample_has_GT_relation]
                                    relation_loss_leave_one_out_reranker = torch.sum(
                                        reranker_model.calculate_loss_from_logits(logits_w_GT_relation, labels_w_GT_relation), dim=1
                                    )
                            else:
                                relation_logits = reranker_model(*relation_extraction_inputs)
                                logits_w_GT_relation = relation_logits[sample_has_GT_relation]
                                relation_loss_leave_one_out_reranker = torch.sum(
                                    reranker_model.calculate_loss_from_logits(logits_w_GT_relation, labels_w_GT_relation), dim=1
                                )

                            leave_one_out_reward_reranker = relation_loss_leave_one_out_reranker - reranking_loss_for_reward[sample_has_GT_relation]
                            LOO_rewards_reranker.extend(leave_one_out_reward_reranker.detach().cpu().numpy().flatten())

                # calculate combined loss
                start_log_prob = m_start.log_prob(expl_start_pred_idxs)
                end_log_prob = m_end.log_prob(expl_end_pred_idxs)
                combined_log_prob = start_log_prob + end_log_prob

                expl_loss = 0

                if kwargs["reranking_reward"]:
                    if len(reranking_rewards) < 5:
                        reranking_baseline = 0
                        reranking_variance = 1
                    else:
                        reranking_baseline = torch.tensor(np.mean(reranking_rewards)).to(kwargs["device"])
                        reranking_variance = torch.tensor(np.std(reranking_rewards)).to(kwargs["device"])

                    expl_policy_reranking_loss = (
                        -combined_log_prob[sample_has_GT_relation]
                        * (reranking_reward[sample_has_GT_relation] - reranking_baseline)
                        / reranking_variance
                    )
                    expl_policy_reranking_loss = torch.sum(expl_policy_reranking_loss)
                    expl_loss += expl_policy_reranking_loss
                    total_expl_policy_reranking_loss += expl_policy_reranking_loss.item()

                if kwargs["supervised_expl_loss"]:
                    expl_loss += supervised_expl_loss
                    total_supervised_expl_loss += supervised_expl_loss.item()

                if kwargs["leave_one_out_loss_initial_ranker"]:
                    if len(LOO_rewards_initial_ranker) < 5:
                        LOO_baseline_initial_ranker = 0
                        LOO_variance_initial_ranker = 1
                    else:
                        LOO_baseline_initial_ranker = torch.tensor(np.mean(LOO_rewards_initial_ranker)).to(kwargs["device"])
                        LOO_variance_initial_ranker = torch.tensor(np.std(LOO_rewards_initial_ranker)).to(kwargs["device"])
                    expl_policy_LOO_loss_initial_ranker = (
                        -combined_log_prob[sample_has_GT_relation]
                        * (leave_one_out_reward_initial_ranker - LOO_baseline_initial_ranker)
                        / LOO_variance_initial_ranker
                    )
                    expl_policy_LOO_loss_initial_ranker = torch.sum(expl_policy_LOO_loss_initial_ranker)
                    expl_loss += expl_policy_LOO_loss_initial_ranker
                    total_expl_policy_LOO_loss_initial_ranker += expl_policy_LOO_loss_initial_ranker.item()

                if kwargs["leave_one_out_loss_reranker"]:
                    if len(LOO_rewards_reranker) < 5:
                        LOO_baseline_reranker = 0
                        LOO_variance_reranker = 1
                    else:
                        LOO_basline_reranker = torch.tensor(np.mean(LOO_rewards_reranker)).to(kwargs["device"])
                        LOO_variance_reranker = torch.tensor(np.std(LOO_rewards_reranker)).to(kwargs["device"])
                    expl_policy_LOO_loss_reranker = (
                        -combined_log_prob[sample_has_GT_relation] * (leave_one_out_reward_reranker - LOO_baseline_reranker) / LOO_variance_reranker
                    )
                    expl_policy_LOO_loss_reranker = torch.sum(expl_policy_LOO_loss_reranker)

                    expl_loss += expl_policy_LOO_loss_reranker
                    total_expl_policy_LOO_loss_reranker += expl_policy_LOO_loss_reranker.item()

                expl_loss = expl_loss / (kwargs["gpu_batch_size"] * gradient_accumulation_steps)
                if kwargs["fp16"]:
                    expl_scaler.scale(expl_loss).backward()
                else:
                    expl_loss.backward()

                if ((i + 1) == kwargs["gpu_batch_size"]) and ((step + 1) % gradient_accumulation_steps) == 0:
                    if kwargs["fp16"]:
                        reranker_scaler.unscale_(reranker_optimizer)
                        torch.nn.utils.clip_grad_norm_(reranker_model.parameters(), kwargs["max_grad_norm"])
                        reranker_scaler.step(reranker_optimizer)
                        reranker_scaler.update()
                        reranker_optimizer.zero_grad()

                        expl_scaler.unscale_(expl_optimizer)
                        torch.nn.utils.clip_grad_norm_(expl_model.parameters(), kwargs["max_grad_norm"])
                        expl_scaler.step(expl_optimizer)
                        expl_scaler.update()
                        expl_optimizer.zero_grad()
                    else:
                        torch.nn.utils.clip_grad_norm_(reranker_model.parameters(), kwargs["max_grad_norm"])
                        reranker_optimizer.step()
                        reranker_optimizer.zero_grad()

                        torch.nn.utils.clip_grad_norm_(expl_model.parameters(), kwargs["max_grad_norm"])
                        expl_optimizer.step()
                        expl_optimizer.zero_grad()

                    reranking_rewards = reranking_rewards[-100:]
                    LOO_rewards_initial_ranker = LOO_rewards_initial_ranker[-100:]
                    LOO_rewards_reranker = LOO_rewards_reranker[-100:]

            if (step + 1) % gradient_accumulation_steps == 0:
                desc = f"Reranker loss: {total_reranking_loss/(step+1):0.4f}"
                if kwargs["reranking_reward"]:
                    desc += f", Expl policy reranking loss: {total_expl_policy_reranking_loss/(step+1):0.4f}"
                if kwargs["leave_one_out_loss_initial_ranker"]:
                    desc += f", Expl policy LOO initial ranker LOSS: {total_expl_policy_LOO_loss_initial_ranker/(step+1):0.4f}"
                if kwargs["leave_one_out_loss_reranker"]:
                    desc += f", Expl policy LOO reranker loss: {total_expl_policy_LOO_loss_reranker/(step+1):0.4f}"
                if kwargs["supervised_expl_loss"]:
                    desc += f", Supervised expl loss: {total_supervised_expl_loss/(step+1):0.4f}"

                pbar.set_description(desc)

        expl_model.eval()
        reranker_model.eval()
        with torch.no_grad():
            f1, bestT1, bestT2, outputs = evaluate(
                dev_dataloader,
                initial_ranker_model,
                initial_ranker_model_class,
                expl_model,
                expl_model_class,
                reranker_model,
                reranker_model_class,
                expl_tokenizer,
                reranker_tokenizer,
                **kwargs,
            )

        if f1 > best_f1:
            best_f1 = f1
            best_T1 = bestT1
            best_T2 = bestT2
            if kwargs["output_dir"]:
                output_dir = os.path.join(kwargs["output_dir"], f"F1-{best_f1:0.4f}_T1-{best_T1:0.4f}_T2-{best_T2:0.4f}")
                expl_model.save_pretrained(output_dir + "expl_model")
                reranker_model.save_pretrained(output_dir + "reranker_model")
        if os.path.isdir(kwargs["output_dir"]):
            with open(os.path.join(kwargs["output_dir"], f"F1-{f1:0.4f}.dev_outputs.txt"), "w") as f:
                for output in outputs:
                    f.write("\t".join([str(o) for o in output]) + "\n")
            pred_rel_count, _ = analysis_utils.get_explanations_per_relation(os.path.join(kwargs["output_dir"], f"F1-{f1:0.4f}.dev_outputs.txt"))
            logger.info(f"Predicted explanations: {sum(pred_rel_count.values())}")

        softmax_temp = max(softmax_temp / kwargs["softmax_decay_ratio"], 0.1)

    if kwargs["output_dir"]:
        output_dir = os.path.join(kwargs["output_dir"], f"final.F1-{f1:0.4f}_T1-{bestT1:0.2f}_T2-{bestT2:0.2f}")
        expl_model.save_pretrained(output_dir + "_expl_model")
        reranker_model.save_pretrained(output_dir + "_reranker_model")


if __name__ == "__main__":
    args = utils.parse_drex_args()
    main(**args)
