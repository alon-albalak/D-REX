import logging
import os
import sys
from tqdm import tqdm
from utils import utils, data_utils, analysis_utils
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
    if kwargs["seed"] != -1:
        utils.set_seed(kwargs["seed"])

    # load model config and tokenizer
    kwargs["model_class"] = kwargs["model_class"].lower()
    config_class, model_class, tokenizer_class = utils.MODEL_CLASSES[kwargs["model_class"]]
    config = config_class.from_pretrained(kwargs["model_name_or_path"])
    config.update(kwargs)
    tokenizer = tokenizer_class.from_pretrained(kwargs["base_model"])

    # load training data
    kwargs["data_split"] = "train"
    train_dataset = data_utils.get_data(tokenizer, **kwargs)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=kwargs["gpu_batch_size"], shuffle=True)

    # load dev data
    kwargs["data_split"] = "dev"
    dev_dataset = data_utils.get_data(tokenizer, **kwargs)
    dev_dataloader = torch.utils.data.DataLoader(dev_dataset, batch_size=kwargs["gpu_batch_size"], shuffle=False)

    # load model
    model = model_class.from_pretrained(kwargs["model_name_or_path"], config=config)
    model.to(kwargs["device"])

    # optimization vars
    gradient_accumulation_steps = kwargs["effective_batch_size"] / kwargs["gpu_batch_size"]
    total_optimization_steps = kwargs["num_epochs"] * (len(train_dataloader) // gradient_accumulation_steps)
    optimizer = torch.optim.AdamW(model.parameters(), lr=kwargs["learning_rate"])

    if kwargs["fp16"]:
        scaler = torch.cuda.amp.GradScaler()

    logger.info("******** Training ********")
    logger.info(f"    Num samples: {len(train_dataset)}")
    logger.info(f"    Num epochs: {kwargs['num_epochs']}")
    logger.info(f"    Total optimization steps: {total_optimization_steps}")

    best_f1, best_T1, best_T2 = 0, 0, 0
    topk = [1, 3, 5]
    for epoch in range(kwargs["num_epochs"]):
        logger.info(f"EPOCH: {epoch+1}")
        total_loss = 0
        optimizer.zero_grad()
        model.train()

        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for step, batch in pbar:
            batch = utils.batch_to_device(batch, kwargs["device"])
            guids, input_ids, attention_mask, segment_ids, relations, start_trigger_ids, end_trigger_ids = batch

            if model_class.__name__ == "relation_extraction_RoBERTa":
                inputs = (input_ids, attention_mask, relations)
            else:
                inputs = (input_ids, attention_mask, segment_ids, relations)

            if kwargs["fp16"]:
                with torch.cuda.amp.autocast():
                    loss = model.calculate_loss(*inputs)

                    loss = loss / gradient_accumulation_steps
                scaler.scale(loss).backward()
                total_loss += loss.item()

                if ((step + 1) % gradient_accumulation_steps) == 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), kwargs["max_grad_norm"])
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
            else:
                loss = model.calculate_loss(*inputs)
                loss = loss / gradient_accumulation_steps
                loss.backward()
                total_loss += loss.item()
                if ((step + 1) % gradient_accumulation_steps) == 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), kwargs["max_grad_norm"])
                    optimizer.step()
                    optimizer.zero_grad()

            desc = f"TRAIN LOSS: {total_loss/(step+1):0.4f}"
            pbar.set_description(desc)

        model.eval()
        # variables for post-hoc evaluation
        logits = []
        labels = []
        for batch in dev_dataloader:
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

        f1, precision, recall, bestT1, bestT2 = analysis_utils.get_f1_from_logits(logits, labels)
        hits_at_k = analysis_utils.get_hits_at_k(topk, logits, labels)
        MRR = analysis_utils.get_MRR(logits, labels)

        if f1 > best_f1:
            best_f1 = f1
            best_T1 = bestT1
            best_T2 = bestT2
            if kwargs["output_dir"]:
                output_dir = os.path.join(kwargs["output_dir"], f"F1-{best_f1:0.4f}_T1-{best_T1}_T2-{best_T2}")
                model.save_pretrained(output_dir)

        logger.info(f"PR: {precision:0.4f}, RE: {recall:0.4f}")
        logger.info(f"F1: {f1:0.4f}, BEST T1: {bestT1}, BEST T2: {bestT2}")
        logger.info(f"Hits @ {topk}: {hits_at_k}")
        logger.info(f"MRR: {MRR:0.4f}")

    if kwargs["output_dir"]:
        output_dir = os.path.join(kwargs["output_dir"], f"final.F1-{f1:0.4f}_T1-{bestT1}_T2-{bestT2}")
        model.save_pretrained(output_dir)


if __name__ == "__main__":
    args = utils.parse_args()
    main(**args)
