import logging
import os
import sys
from tqdm import tqdm
import torch

from utils import utils, data_utils

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

    # load data
    kwargs["data_split"] = "train"
    train_dataset = data_utils.get_data(tokenizer, **kwargs)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=kwargs["gpu_batch_size"], shuffle=True)

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

    max_pred_trigger_dist = kwargs["max_pred_trigger_dist"]
    best_exact_match, best_f1 = 0, 0
    for epoch in range(kwargs["num_epochs"]):
        logger.info(f"EPOCH: {epoch+1}")
        total_loss = 0
        optimizer.zero_grad()
        model.train()

        pbar = tqdm(enumerate(train_dataloader), total=len(train_dataloader))
        for step, batch in pbar:
            batch = utils.batch_to_device(batch, kwargs["device"])
            guids, input_ids, attention_mask, segment_ids, relation_ids, start_trigger_ids, end_trigger_ids = batch

            if model_class.__name__ == "explanation_policy_RoBERTa":
                inputs = (input_ids, attention_mask, start_trigger_ids, end_trigger_ids)
            else:
                inputs = (input_ids, attention_mask, segment_ids, start_trigger_ids, end_trigger_ids)

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
        total_samples, total_samples_correct = 0, 0

        tokens_GT, tokens_correct, tokens_pred = 0, 0, 0
        for batch in dev_dataloader:
            batch = utils.batch_to_device(batch, kwargs["device"])
            guids, input_ids, attention_mask, segment_ids, relation_ids, start_trigger_ids, end_trigger_ids = batch

            if model_class.__name__ == "explanation_policy_RoBERTa":
                inputs = (input_ids, attention_mask)
            else:
                inputs = (input_ids, attention_mask, segment_ids)

            with torch.no_grad():
                start_preds, end_preds = model.predict(*inputs)

            for start_pred, end_pred, start_trigger_id, end_trigger_id, token_ids in zip(
                start_preds, end_preds, start_trigger_ids, end_trigger_ids, input_ids
            ):
                total_samples += 1
                if start_pred == start_trigger_id and end_pred == end_trigger_id:
                    total_samples_correct += 1

                for i in range(max(end_pred, end_trigger_id) + 1):
                    if end_trigger_id > 0:
                        if start_trigger_id <= i <= end_trigger_id:
                            tokens_GT += 1
                            if start_pred <= i <= end_pred and end_pred - start_pred < max_pred_trigger_dist:
                                tokens_correct += 1
                    if start_pred <= i <= end_pred and end_pred - start_pred < max_pred_trigger_dist:
                        tokens_pred += 1

                # un-comment to view GT and predicted explanations
                # input_tokens = tokenizer.convert_ids_to_tokens(token_ids.squeeze())
                # GT_expl = decode_normalize_tokens(input_tokens, start_trigger_id, end_trigger_id)
                # if start_pred <= end_pred:
                #     pred_expl = decode_normalize_tokens(input_tokens, start_pred, end_pred)

        precision = 1 if tokens_pred == 0 else tokens_correct / tokens_pred
        recall = 0 if tokens_GT == 0 else tokens_correct / tokens_GT
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
        if f1 > best_f1:
            best_f1 = f1
            if kwargs["output_dir"]:
                output_dir = os.path.join(kwargs["output_dir"], f"F1-{best_f1:0.3f}")
                model.save_pretrained(output_dir)

        exact_match = total_samples_correct / total_samples
        if exact_match > best_exact_match:
            best_exact_match = exact_match
        logger.info(f"EM: {exact_match:0.3f}, BEST EM: {best_exact_match:0.3f}")
        logger.info(f"PR: {precision:0.3f}, RE: {recall:0.3f}")
        logger.info(f"f1: {f1:0.3f}, BEST f1: {best_f1:0.3f}")

    if kwargs["output_dir"]:
        output_dir = os.path.join(kwargs["output_dir"], f"final.F1-{f1:0.3f}")
        model.save_pretrained(output_dir)


if __name__ == "__main__":
    args = utils.parse_args()
    main(**args)
