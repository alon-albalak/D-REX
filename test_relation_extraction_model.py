import logging
import os
import sys
import torch
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
    assert kwargs["data_split"]
    assert (kwargs["threshold1"] is not None) and (kwargs["threshold2"] is not None)

    if kwargs["seed"] != -1:
        utils.set_seed(kwargs["seed"])

    # load model config and tokenizer
    kwargs["model_class"] = kwargs["model_class"].lower()
    config_class, model_class, tokenizer_class = utils.MODEL_CLASSES[kwargs["model_class"]]
    config = config_class.from_pretrained(kwargs["model_name_or_path"])
    config.update(kwargs)
    tokenizer = tokenizer_class.from_pretrained(kwargs["base_model"])

    # load data
    test_dataset = data_utils.get_data(tokenizer, **kwargs)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=kwargs["gpu_batch_size"], shuffle=False)

    # load model
    model = model_class.from_pretrained(kwargs["model_name_or_path"], config=config)
    model.to(kwargs["device"])

    logger.info(f"******** Evaluating {kwargs['model_name_or_path']} ************")
    logger.info(f"  Num samples: {len(test_dataset)}")

    model.eval()
    topk = [1, 3, 5]
    logits, labels, all_guids = [], [], []
    for batch in test_dataloader:
        batch = utils.batch_to_device(batch, kwargs["device"])
        guids, input_ids, attention_mask, segment_ids, relations, start_trigger_ids, end_trigger_ids = batch

        if model_class.__name__ == "relation_extraction_RoBERTa":
            inputs = (input_ids, attention_mask)
        else:
            inputs = (input_ids, attention_mask, segment_ids)

        # detach and send to cpu for evaluation (labels, logits)
        with torch.no_grad():
            relation_logits = model(*inputs)
        for logit in relation_logits.detach().cpu().numpy():
            logits.append(logit)
        for label in relations:
            labels.append(label.cpu().numpy())
        all_guids.extend(guids)

    f1, precision, recall = analysis_utils.get_f1_from_logits(logits, labels, kwargs["threshold1"], kwargs["threshold2"])
    hits_at_k = analysis_utils.get_hits_at_k(topk, logits, labels)
    MRR = analysis_utils.get_MRR(logits, labels)

    logger.info(f"PR: {precision:0.4f}, RE: {recall:0.4f}")
    logger.info(f"F1: {f1:0.4f}")
    logger.info(f"Hits @ {topk}: {hits_at_k}")
    logger.info(f"MRR: {MRR:0.4f}")

    if os.path.isdir(kwargs["model_name_or_path"]):
        with open(os.path.join(kwargs["model_name_or_path"], "test_outputs.txt"), "w") as f:
            for guid, label, logit in zip(all_guids, labels, logits):
                f.write("\t".join([str(x) for x in [guid, label, logit]]) + "\n")


if __name__ == "__main__":
    args = utils.parse_args()
    main(**args)
