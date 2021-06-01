from transformers import BertPreTrainedModel, BertModel
from torch.nn import Dropout, Linear, BCEWithLogitsLoss
import torch
import numpy as np


class relation_extraction_BERT(BertPreTrainedModel):
    def __init__(self, config):
        super(relation_extraction_BERT, self).__init__(config)
        self.num_relations = config.num_relations
        self.bert = BertModel(config)
        self.dropout = Dropout(config.dropout_rate)
        self.classifier = Linear(config.hidden_size, config.num_relations)
        self.loss_fct = BCEWithLogitsLoss(pos_weight=config.pos_weight_RE_samples * torch.ones([config.num_relations]), reduction="none")
        self.init_weights()

    def forward(self, input_ids, attention_mask, segment_ids):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        pooled_output = outputs[1]
        pooled_output = self.dropout(pooled_output)
        relation_logits = self.classifier(pooled_output)

        return relation_logits

    def calculate_loss(self, input_ids, attention_mask, segment_ids, relation_labels):
        relation_logits = self.forward(input_ids, attention_mask, segment_ids)
        relation_loss = torch.sum(self.loss_fct(relation_logits, relation_labels))
        return relation_loss

    def calculate_loss_from_logits(self, logits, labels):
        return self.loss_fct(logits, labels)

    def sort_relations_from_logits(self, relation_logits):
        relation_logits = relation_logits.detach().cpu().numpy()
        sorted_relation_idxs = np.argsort(-relation_logits, axis=1)
        return sorted_relation_idxs