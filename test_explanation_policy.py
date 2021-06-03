import logging
import os
import sys
from utils import utils, data_utils
import torch

# Setup logging
logger = logging.getLogger()
logger.setLevel(logging.INFO)

handler = logging.StreamHandler(sys.stdout)
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)


def main(**kwargs):
    assert kwargs["data_split"]

    # define model variables
    if kwargs["seed"] != -1:
        utils.set_seed(kwargs["seed"])

    # load model config and tokenizer
    kwargs["model_class"] = kwargs["model_class"].lower()
    config_class, model_class, tokenizer_class = utils.MODEL_CLASSES[kwargs["model_class"]]
    config = config_class.from_pretrained(kwargs["model_name_or_path"])
    config.update(kwargs)
    tokenizer = tokenizer_class.from_pretrained(kwargs["base_model"])

    # load data
    test_dataset = data_utils.get_data(tokenizer, include_samples=True, include_relation_entities_mask=True, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=kwargs["gpu_batch_size"], shuffle=False)

    # load model
    model = model_class.from_pretrained(kwargs["model_name_or_path"], config=config)
    model.to(kwargs["device"])
    model.eval()

    logger.info(f"******** Evaluating {kwargs['model_name_or_path']} ************")
    logger.info(f"  Num samples: {len(test_dataset)}")

    max_pred_trigger_dist = kwargs["max_pred_trigger_dist"]
    tokens_GT, tokens_correct, tokens_pred = 0, 0, 0
    total_GT_samples, total_GT_samples_correct = 0, 0
    num_preds, num_overlapping = 0, 0
    outputs = []
    for batch in test_dataloader:
        batch = utils.batch_to_device(batch, kwargs["device"])
        guids, input_ids, attention_mask, segment_ids, relation_ids, start_trigger_ids, end_trigger_ids, samples, relation_entities_mask = batch
        samples = [{k: samples[k][i] for k in samples} for i in range(len(samples["dialogue"]))]

        if model_class.__name__ == "explanation_policy_RoBERTa":
            inputs = (input_ids, attention_mask)
        else:
            inputs = (input_ids, attention_mask, segment_ids)

        with torch.no_grad():
            start_preds, end_preds = model.predict(*inputs)

        for start_pred, end_pred, start_trigger_id, end_trigger_id, token_ids, sample, guid, relation_id, rel_ent_mask in zip(
            start_preds, end_preds, start_trigger_ids, end_trigger_ids, input_ids, samples, guids, relation_ids, relation_entities_mask
        ):

            # only calculate EM, F1 on samples with labeled trigger
            if end_trigger_id > 0:
                total_GT_samples += 1
                if start_pred == start_trigger_id and end_pred == end_trigger_id:
                    total_GT_samples_correct += 1

                for i in range(max(end_pred, end_trigger_id) + 1):
                    if start_trigger_id <= i <= end_trigger_id:
                        tokens_GT += 1
                        if start_pred <= i <= end_pred and end_pred - start_pred < max_pred_trigger_dist:
                            tokens_correct += 1
                    if start_pred <= i <= end_pred and end_pred - start_pred < max_pred_trigger_dist:
                        tokens_pred += 1

            GT_expl = data_utils.decode_normalize_tokens(token_ids.squeeze(), start_trigger_id, end_trigger_id, tokenizer)

            # Only consider explanations which start from after the specification tokens <relation> [SEP] <entity1> [SEP] <entity2> [SEP]
            first_real_token = (rel_ent_mask == 1).nonzero(as_tuple=True)[0][0]
            pred_expl = "None"
            if start_pred <= end_pred and start_pred >= first_real_token and end_pred - start_pred < max_pred_trigger_dist:
                pred_expl = data_utils.decode_normalize_tokens(token_ids.squeeze(), start_pred, end_pred, tokenizer)

            if pred_expl != "None" and start_pred >= first_real_token:
                num_preds += 1
                if start_trigger_id <= start_pred <= end_trigger_id or start_trigger_id <= end_pred <= end_trigger_id:
                    num_overlapping += 1

            outputs.append(
                [
                    guid,
                    utils.get_relation(relation_id.item()),
                    start_pred.item(),
                    end_pred.item(),
                    pred_expl,
                    GT_expl,
                    sample["head"],
                    sample["tail"],
                ]
            )

    precision = 1 if tokens_pred == 0 else tokens_correct / tokens_pred
    recall = 0 if tokens_GT == 0 else tokens_correct / tokens_GT
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) != 0 else 0

    logger.info(f"TEST EM: {total_GT_samples_correct/total_GT_samples:0.4f}")
    logger.info(f"TEST PR: {precision:0.4f}, TEST RE: {recall:0.4f}")
    logger.info(f"TEST f1: {f1:0.4f}")

    logger.info(f"Total GT triggers: {total_GT_samples}")
    logger.info(f"Total correct triggers: {total_GT_samples_correct}")
    logger.info(f"Number predictions overlapping w/ GT: {num_overlapping}")
    logger.info(f"Number samples w/ prediction: {num_preds}")

    if os.path.isdir(kwargs["model_name_or_path"]):
        with open(os.path.join(kwargs["model_name_or_path"], f"{kwargs['data_split']}_explanation_evaluation_outputs.txt"), "w") as f:
            for output in outputs:
                f.write("\t".join([str(o) for o in output]) + "\n")


if __name__ == "__main__":
    args = utils.parse_args()
    main(**args)
