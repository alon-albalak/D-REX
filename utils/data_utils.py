import logging
import json
import utils.utils as utils
import torch

import re
import collections
from torch._six import string_classes

logger = logging.getLogger(__name__)


def convert_to_unicode(text):
    """Converts `text` to Unicode (if it's not already), assuming utf-8 input."""
    if isinstance(text, str):
        return text
    elif isinstance(text, bytes):
        return text.decode("utf-8", "ignore")
    else:
        raise ValueError("Unsupported string type: %s" % (type(text)))


def decode_normalize_tokens(input_ids, start_idx, end_idx, tokenizer):
    pred_value = tokenizer.decode(input_ids[start_idx : end_idx + 1], clean_up_tokenization_spaces=True).lstrip()
    return pred_value


def tokenize(text, tokenizer):
    if "roberta" in tokenizer.name_or_path:
        unused_tokens = ["madeupword0001", "madeupword0002"]
    elif "bert" in tokenizer.name_or_path:
        unused_tokens = ["[unused1]", "[unused2]"]
    text_tokens = []
    textraw = [text]
    textraw = [text.replace(":", "")]
    for delimiter in unused_tokens:
        ntextraw = []
        for i in range(len(textraw)):
            t = textraw[i].split(delimiter)
            for j in range(len(t)):
                ntextraw += [t[j]]
                if j != len(t) - 1:
                    ntextraw += [delimiter]
        textraw = ntextraw
    text = []
    for t in textraw:
        if t in unused_tokens:
            text += [t]
        else:
            if "roberta" in tokenizer.name_or_path:
                tokens = tokenizer.tokenize(t, add_prefix_space=True)
            else:
                tokens = tokenizer.tokenize(t)
            for tok in tokens:
                text += [tok]
    return text


class Sample(object):
    """A single training/test sample for token/sequence classification"""

    def __init__(self, guid=None, dialogue=None, head=None, tail=None, relations=None, triggers=None):
        self.guid = guid
        self.dialogue = dialogue
        self.head = head
        self.tail = tail
        self.relations = relations
        self.triggers = triggers


def is_speaker(a):
    a = a.split()
    return len(a) == 2 and a[0].lower() == "speaker" and a[1].isdigit()


def rename(d, x, y, tokenizer_name_or_path):
    # replace
    if "roberta" in tokenizer_name_or_path:
        unused_tokens = ["madeupword0001", "madeupword0002"]
    elif "bert" in tokenizer_name_or_path:
        unused_tokens = ["[unused1]", "[unused2]"]
    # unused = ["[unused1]", "[unused2]"]
    a = []
    if is_speaker(x):
        a += [x]
    else:
        a += [None]
    if x != y and is_speaker(y):
        a += [y]
    else:
        a += [None]
    for i in range(len(a)):
        if a[i] is None:
            continue
        d = d.replace(a[i] + ":", unused_tokens[i] + " :")
        if x == a[i]:
            x = unused_tokens[i]
        if y == a[i]:
            y = unused_tokens[i]
    return d, x, y


def create_data(
    tokenizer,
    explanation_policy_pretraining,
    relation_extraction_pretraining,
    relation_extraction_conditioned_on_explanations,
    joint_model_training,
    include_relations_in_sample,
    **kwargs,
):
    """load data into python samples, and convert to features (tokens, labels, etc.) for model"""

    assert (
        explanation_policy_pretraining or relation_extraction_pretraining or relation_extraction_conditioned_on_explanations or joint_model_training
    ), "Must select a type of pretraining"
    assert not (
        explanation_policy_pretraining
        and relation_extraction_pretraining
        and relation_extraction_conditioned_on_explanations
        and joint_model_training
    ), "Must select only 1 type of pretraining"

    # load data and create samples
    if explanation_policy_pretraining:
        samples = load_dialogRE_explanation_policy_pretraining(
            data_path=kwargs["data_path"],
            data_split=kwargs["data_split"],
            predict_trigger_for_unlabelled=kwargs["predict_trigger_for_unlabelled"],
            rename_entities=kwargs["rename_entities"],
            keep_case=kwargs["keep_case"],
            tokenizer_name_or_path=tokenizer.name_or_path,
        )
    elif relation_extraction_pretraining:
        samples = load_dialogRE_relation_extraction_pretraining(
            data_path=kwargs["data_path"],
            data_split=kwargs["data_split"],
            keep_case=kwargs["keep_case"],
            tokenizer_name_or_path=tokenizer.name_or_path,
        )
    else:
        samples = load_dialogRE_relation_extraction_conditioned_on_explanation(
            data_path=kwargs["data_path"],
            data_split=kwargs["data_split"],
            use_predicted_explanations=kwargs["use_predicted_explanations"],
            predicted_explanation_path=kwargs["predicted_explanation_path"],
            keep_case=kwargs["keep_case"],
            tokenizer_name_or_path=tokenizer.name_or_path,
        )

    # convert samples into features (i.e. guid, input token ids, labels, etc.)
    features = convert_samples_to_features(
        samples,
        kwargs["max_sequence_len"],
        tokenizer,
        predict_triggers=explanation_policy_pretraining or joint_model_training,
        append_trigger_tokens=relation_extraction_conditioned_on_explanations,
        append_relation=explanation_policy_pretraining,
        include_relations_in_sample=include_relations_in_sample,
    )

    return features


def get_data(
    tokenizer,
    features=None,
    include_samples=False,
    explanation_policy_pretraining=False,
    relation_extraction_pretraining=False,
    relation_extraction_conditioned_on_explanations=False,
    joint_model_training=False,
    include_relations_in_sample=False,
    include_relation_entities_mask=False,
    **kwargs,
):
    """
    convert features into tensor dataset

    allow for features to be passed in, or to create them from scratch
    """
    # If we want to cache data, this is the place to either create from scratch, or just load from saved file
    # For now, we always create from scratch
    if not features:
        features = create_data(
            tokenizer,
            explanation_policy_pretraining=explanation_policy_pretraining,
            relation_extraction_pretraining=relation_extraction_pretraining,
            relation_extraction_conditioned_on_explanations=relation_extraction_conditioned_on_explanations,
            joint_model_training=joint_model_training,
            include_relations_in_sample=include_relations_in_sample,
            **kwargs,
        )

    guids = [f.guid for f in features]
    input_ids = torch.tensor([f.input_ids for f in features], dtype=torch.long)
    attention_mask = torch.tensor([f.attention_mask for f in features], dtype=torch.long)
    segment_ids = torch.tensor([f.segment_ids for f in features], dtype=torch.long)
    relations = torch.tensor([f.relations for f in features], dtype=torch.float)
    start_trigger_ids = torch.tensor([f.start_trigger_id for f in features], dtype=torch.long)
    end_trigger_ids = torch.tensor([f.end_trigger_id for f in features], dtype=torch.long)

    all_features = [guids, input_ids, attention_mask, segment_ids, relations, start_trigger_ids, end_trigger_ids]

    if include_samples:
        samples = [f.sample for f in features]
        all_features.append(samples)

    if include_relation_entities_mask:
        relation_entities_mask = []
        for f in features:
            num_sep = f.input_ids.count(tokenizer.sep_token_id)
            sep_count = 0
            mask = []
            for ids in f.input_ids:
                if sep_count < num_sep - 1:
                    mask.append(0)
                else:
                    mask.append(1)

                if ids == tokenizer.sep_token_id:
                    sep_count += 1
            relation_entities_mask.append(mask)
        relation_entities_mask = torch.tensor(relation_entities_mask, dtype=torch.float)
        all_features.append(relation_entities_mask)

    dataset = TensorListDataset(*all_features)

    return dataset


def load_dialogRE_explanation_policy_pretraining(
    data_path="data_v2",
    data_split="debugging",
    predict_trigger_for_unlabelled=False,
    rename_entities=True,
    keep_case=False,
    tokenizer_name_or_path="roberta",
):

    logger.info(f"Loading and processing data from {data_path}/{data_split}.json")
    with open(f"{data_path}/{data_split}.json", "r") as f:
        data = json.load(f)

    samples = []
    for i, datum in enumerate(data):
        for j, entity_pair in enumerate(datum[1]):

            entity_pair_positive_relation_ids = []

            if rename_entities:
                dialogue, head, tail = rename("\n".join(datum[0]), entity_pair["x"], entity_pair["y"], tokenizer_name_or_path)
            else:
                dialogue = "\n".join(datum[0])
                head = entity_pair["x"]
                tail = entity_pair["y"]

            if not keep_case:
                dialogue = dialogue.lower()
                head = head.lower()
                tail = tail.lower()

            dialogue = convert_to_unicode(dialogue)
            head = convert_to_unicode(head)
            tail = convert_to_unicode(tail)

            for k, (trigger, relation_id) in enumerate(zip(entity_pair["t"], entity_pair["rid"])):
                guid = f"{data_split}-{i}.{j}.{k}"

                trigger = convert_to_unicode(trigger)

                # relation_id-1 because relations go from 1 to 37
                sample = Sample(guid, dialogue, head, tail, relation_id - 1, trigger)
                if predict_trigger_for_unlabelled or trigger:
                    samples.append(sample)
    return samples


def load_dialogRE_relation_extraction_pretraining(data_path="data_v2", data_split="debugging", keep_case=False, tokenizer_name_or_path="roberta"):

    logger.info(f"Loading and processing data from {data_path}/{data_split}.json")
    with open(f"{data_path}/{data_split}.json", "r") as f:
        data = json.load(f)

    samples = []
    for i, datum in enumerate(data):
        for j, entity_pair in enumerate(datum[1]):
            guid = f"{data_split}-{i}.{j}"
            if keep_case:
                dialogue, head, tail = rename("\n".join(datum[0]), entity_pair["x"], entity_pair["y"], tokenizer_name_or_path)
            else:
                dialogue, head, tail = rename("\n".join(datum[0]).lower(), entity_pair["x"].lower(), entity_pair["y"].lower(), tokenizer_name_or_path)
            dialogue = convert_to_unicode(dialogue)
            head = convert_to_unicode(head)
            tail = convert_to_unicode(tail)
            triggers = [convert_to_unicode(trigger) for trigger in entity_pair["t"]]
            relations = [0 for _ in range(37)]
            for rid in entity_pair["rid"]:

                relations[rid - 1] = 1

            sample = Sample(guid, dialogue, head, tail, relations, triggers)
            samples.append(sample)

    return samples


def load_dialogRE_relation_extraction_conditioned_on_explanation(
    data_path="data_v2",
    data_split="debugging",
    use_predicted_explanations=False,
    rename_entities=True,
    predicted_explanation_path=None,
    keep_case=False,
    tokenizer_name_or_path="roberta",
):
    logger.info(f"Loading and processing data from {data_path}/{data_split}.json")
    with open(f"{data_path}/{data_split}.json", "r") as f:
        data = json.load(f)

    if use_predicted_explanations:
        # load precomputed explanations
        logger.info(f"loading precomputed explanations from {predicted_explanation_path}/{data_split}_explanation_evaluation_outputs.txt")
        outside_predictions = {}
        with open(f"{predicted_explanation_path}/{data_split}_explanation_evaluation_outputs.txt") as f:
            for row in f:
                r = row.replace("\n", "").split("\t")
                outside_predictions[r[0]] = r[1:]

    samples = []
    for i, datum in enumerate(data):
        for j, entity_pair in enumerate(datum[1]):
            for k, (trigger, relation_id) in enumerate(zip(entity_pair["t"], entity_pair["rid"])):
                guid = f"{data_split}-{i}.{j}.{k}"
                if rename_entities:
                    dialogue, head, tail = rename("\n".join(datum[0]), entity_pair["x"], entity_pair["y"], tokenizer_name_or_path)
                else:
                    dialogue = "\n".join(datum[0])
                    head = entity_pair["x"]
                    tail = entity_pair["y"]

                if not keep_case:
                    dialogue = dialogue.lower()
                    head = head.lower()
                    tail = tail.lower()
                dialogue = convert_to_unicode(dialogue)
                head = convert_to_unicode(head)
                tail = convert_to_unicode(tail)
                trigger = convert_to_unicode(trigger)
                relations = [0 for _ in range(37)]
                relations[relation_id - 1] = 1

                if use_predicted_explanations:
                    precomputed = outside_predictions[guid]
                    assert precomputed[5] == head and precomputed[6] == tail
                    if relation_id != 36:
                        trigger = precomputed[3]
                    else:
                        trigger = ""

                sample = Sample(guid, dialogue, head, tail, relations, trigger)
                samples.append(sample)
    return samples


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, guid, input_ids, attention_mask, segment_ids, relations, start_trigger_id, end_trigger_id, sample):
        self.guid = guid
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.segment_ids = segment_ids
        self.relations = relations
        self.start_trigger_id = start_trigger_id
        self.end_trigger_id = end_trigger_id
        self.sample = sample


def convert_samples_to_features(
    samples,
    max_sequence_len,
    tokenizer,
    model_token_correction=4,
    predict_triggers=False,
    append_trigger_tokens=False,
    append_relation=False,
    include_relations_in_sample=False,
    logging=True,
    triggers_pretokenized=False,
    mask_triggers=False,
):
    """
    model_token_correction is to account for model specific tokens:
            eg. 4 for BERT - [CLS] tokens [SEP] tokens [SEP] tokens [SEP]
    """

    def _truncate_sequence(
        max_sequence_len, dialogue_tokens, head_tokens, tail_tokens, trigger_tokens=[], relation_tokens=[], model_token_correction=4
    ):
        """
        Truncate a sequence so that it will fit into the model
        model_token_correction is to account for model specific tokens:
                eg. 4 for BERT - [CLS] tokens [SEP] tokens [SEP] tokens [SEP]
        """
        truncated = False
        sequence_len = len(dialogue_tokens) + len(head_tokens) + len(tail_tokens) + len(trigger_tokens) + len(relation_tokens)
        adjusted_max_sequence_len = max_sequence_len - model_token_correction

        while sequence_len > adjusted_max_sequence_len:
            truncated = True
            if len(dialogue_tokens) > 0:
                dialogue_tokens.pop()
            elif len(tail_tokens) > 0:
                tail_tokens.pop()
            elif len(head_tokens) > 0:
                head_tokens.pop()
            elif len(trigger_tokens) > 0:
                trigger_tokens.pop()
            else:
                relation_tokens.pop()
            sequence_len = len(dialogue_tokens) + len(head_tokens) + len(tail_tokens) + len(trigger_tokens) + len(relation_tokens)
        return truncated

    def _find_start_end_labels(tokens_to_find, tokens):
        num_tokens = len(tokens_to_find)
        if num_tokens > 0:
            for ind in (i for i, e in enumerate(tokens) if e == tokens_to_find[0]):
                if tokens[ind : ind + num_tokens] == tokens_to_find:
                    return ind, ind + num_tokens - 1
        return 0, 0

    assert not (append_trigger_tokens and append_relation), "Cannot append both triggers and relations"

    if logging:
        logger.info(f"Converting {len(samples)} samples to features")

    features = []
    num_truncated, truncated_triggers = 0, 0
    for idx, sample in enumerate(samples):
        dialogue_tokens = tokenize(sample.dialogue, tokenizer)
        head_tokens = tokenize(sample.head, tokenizer)
        tail_tokens = tokenize(sample.tail, tokenizer)
        trigger_tokens = []
        if append_trigger_tokens or predict_triggers or mask_triggers:
            if triggers_pretokenized:
                trigger_tokens = sample.triggers.split()
            else:
                if sample.triggers == "None":
                    trigger_tokens = []
                else:
                    trigger_tokens = tokenize(sample.triggers, tokenizer)

        relation_tokens = []
        if append_relation:
            relation_tokens = tokenize(utils.get_relation_in_NL(sample.relations), tokenizer)

        appended_tokens = []
        curr_token_correction = model_token_correction
        if append_trigger_tokens:
            curr_token_correction += 1
            appended_tokens += trigger_tokens + [tokenizer.sep_token]
            if mask_triggers:
                for i in range(len(trigger_tokens)):
                    appended_tokens[i] = tokenizer.mask_token

        if append_relation:
            assert len(relation_tokens) > 0
            curr_token_correction += 1
            appended_tokens += relation_tokens + [tokenizer.sep_token]

        truncated = _truncate_sequence(
            max_sequence_len, dialogue_tokens, head_tokens, tail_tokens, trigger_tokens, relation_tokens, curr_token_correction
        )

        if truncated:
            num_truncated += 1

        appended_tokens += head_tokens + [tokenizer.sep_token] + tail_tokens

        tokens = []
        segment_ids = []
        tokens.append(tokenizer.cls_token)
        segment_ids.append(0)
        for token in appended_tokens:
            tokens.append(token)
            segment_ids.append(0)
        tokens.append(tokenizer.sep_token)
        segment_ids.append(0)
        for token in dialogue_tokens:
            tokens.append(token)
            segment_ids.append(1)
        tokens.append(tokenizer.sep_token)
        segment_ids.append(1)

        if predict_triggers or mask_triggers:
            start_trigger_id, end_trigger_id = _find_start_end_labels(trigger_tokens, tokens[len(appended_tokens) + 2 :])

            if not (start_trigger_id == 0 and end_trigger_id == 0):
                start_trigger_id += len(appended_tokens) + 2
                end_trigger_id += len(appended_tokens) + 2
                assert tokens[start_trigger_id : end_trigger_id + 1] == trigger_tokens
            elif trigger_tokens:
                truncated_triggers += 1
        else:
            start_trigger_id, end_trigger_id = 0, 0

        if mask_triggers and start_trigger_id > 0:
            for i in range(start_trigger_id, end_trigger_id + 1):
                tokens[i] = tokenizer.mask_token

        input_ids = tokenizer.convert_tokens_to_ids(tokens)
        # real tokens get attention of 1, padding tokens get attention of 0
        attention_mask = [1] * len(input_ids)

        # zero-pad up to max sequence length
        while len(input_ids) < max_sequence_len:
            input_ids.append(0)
            attention_mask.append(0)
            segment_ids.append(0)

        assert len(input_ids) == max_sequence_len
        assert len(attention_mask) == max_sequence_len
        assert len(segment_ids) == max_sequence_len

        relations = sample.relations

        appended_sample = {
            "dialogue": sample.dialogue,
            "head": sample.head,
            "tail": sample.tail,
            "triggers": sample.triggers,
        }

        if include_relations_in_sample:
            sample_relations = []
            for idx, i in enumerate(relations):
                if i == 1:
                    sample_relations.append(idx)
            appended_sample["relations"] = sample_relations

        features.append(
            InputFeatures(
                guid=sample.guid,
                input_ids=input_ids,
                attention_mask=attention_mask,
                segment_ids=segment_ids,
                relations=relations,
                start_trigger_id=start_trigger_id,
                end_trigger_id=end_trigger_id,
                sample=appended_sample,
            )
        )
    if logging:
        logger.info(f"Truncated {num_truncated} samples, {truncated_triggers} triggers were truncated")

    return features


class TensorListDataset(torch.utils.data.Dataset):
    """Dataset wrapping tensors, tensor dicts, and tensor lists

    *data (Tensor or dict or list of Tensors): tensors that all have the same size in the first dimension
    """

    def __init__(self, *data):
        if isinstance(data[0], dict):
            size = list(data[0].values())[0].size(0)
        elif isinstance(data[0], list):
            if isinstance(data[0][0], str):
                size = len(data[0])
            else:
                size = data[0][0].size(0)
        else:
            size = data[0].size(0)
        for element in data:
            if isinstance(element, dict):
                if isinstance(list(element.values())[0], list):
                    assert all(size == len(l) for name, l in element.items())  # dict of lists
                else:
                    assert all(size == tensor.size(0) for name, tensor in element.items())  # dict of tensors

            elif isinstance(element, list):
                if element and isinstance(element[0], str):
                    continue
                if element and isinstance(element[0], dict):
                    continue
                if element and isinstance(element[0], list):
                    continue
                assert all(size == tensor.size(0) for tensor in element)  # list of tensors
            else:
                assert size == element.size(0)  # tensor
        self.size = size
        self.data = data

    def __getitem__(self, index):
        result = []
        for element in self.data:
            if isinstance(element, dict):
                result.append({k: v[index] for k, v in element.items()})
            elif isinstance(element, list):
                if isinstance(element[index], str):
                    result.append(element[index])
                elif isinstance(element[index], list):
                    result.append(element[index])
                elif isinstance(element[index], dict):
                    result.append(element[index])
                else:
                    result.append(v[index] for v in element)
            else:
                result.append(element[index])
        return tuple(result)

    def __len__(self):
        return self.size


# Custom collate function to allow for multiple lengthed lists for trigger words
np_str_obj_array_pattern = re.compile(r"[SaUO]")
default_collate_err_msg_format = "default_collate: batch must contain tensors, numpy arrays, numbers, " "dicts or lists; found {}"


def default_collate(batch):
    r"""Puts each data field into a tensor with outer dimension batch size"""

    elem = batch[0]
    elem_type = type(elem)
    if isinstance(elem, torch.Tensor):
        out = None
        if torch.utils.data.get_worker_info() is not None:
            # If we're in a background process, concatenate directly into a
            # shared memory tensor to avoid an extra copy
            numel = sum([x.numel() for x in batch])
            storage = elem.storage()._new_shared(numel)
            out = elem.new(storage)
        return torch.stack(batch, 0, out=out)
    elif elem_type.__module__ == "numpy" and elem_type.__name__ != "str_" and elem_type.__name__ != "string_":
        if elem_type.__name__ == "ndarray" or elem_type.__name__ == "memmap":
            # array of string classes and object
            if np_str_obj_array_pattern.search(elem.dtype.str) is not None:
                raise TypeError(default_collate_err_msg_format.format(elem.dtype))

            return default_collate([torch.as_tensor(b) for b in batch])
        elif elem.shape == ():  # scalars
            return torch.as_tensor(batch)
    elif isinstance(elem, float):
        return torch.tensor(batch, dtype=torch.float64)
    elif isinstance(elem, int):
        return torch.tensor(batch)
    elif isinstance(elem, string_classes):
        return batch
    elif isinstance(elem, collections.abc.Mapping):
        return {key: default_collate([d[key] for d in batch]) if key not in ["triggers", "relations"] else [d[key] for d in batch] for key in elem}
    elif isinstance(elem, tuple) and hasattr(elem, "_fields"):  # namedtuple
        return elem_type(*(default_collate(samples) for samples in zip(*batch)))
    elif isinstance(elem, collections.abc.Sequence):
        # check to make sure that the elements in batch have consistent size
        it = iter(batch)
        elem_size = len(next(it))
        if not all(len(elem) == elem_size for elem in it):
            raise RuntimeError("each element in list of batch should be of equal size")
        transposed = zip(*batch)
        return [default_collate(samples) for samples in transposed]

    raise TypeError(default_collate_err_msg_format.format(elem_type))
