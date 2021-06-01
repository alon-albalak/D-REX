from transformers import RobertaModel
from transformers.models.roberta.modeling_roberta import RobertaPreTrainedModel
from torch.nn import Dropout, Linear, BCEWithLogitsLoss, Tanh
import torch
import numpy as np


class relation_extraction_RoBERTa(RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.num_relations = config.num_relations
        self.roberta = RobertaModel(config, add_pooling_layer=False)
        self.dense = Linear(config.hidden_size, config.hidden_size)
        self.dropout = Dropout(config.dropout_rate)
        self.classifier = Linear(config.hidden_size, config.num_relations)
        self.loss_fct = BCEWithLogitsLoss(pos_weight=config.pos_weight_RE_samples * torch.ones([config.num_relations]), reduction="none")
        self.tanh = Tanh()
        self.init_weights()

    def forward(self, input_ids, attention_mask):
        outputs = self.roberta(input_ids=input_ids, attention_mask=attention_mask)
        cls_tokens = outputs[0][:, 0, :]
        x = self.dropout(cls_tokens)
        x = self.dense(x)
        x = self.tanh(x)
        x = self.dropout(x)
        relation_logits = self.classifier(x)

        return relation_logits

    def calculate_loss(self, input_ids, attention_mask, relation_labels):
        relation_logits = self.forward(input_ids, attention_mask)
        relation_loss = torch.sum(self.loss_fct(relation_logits, relation_labels))
        return relation_loss

    def calculate_loss_from_logits(self, logits, labels):
        return self.loss_fct(logits, labels)

    def sort_relations_from_logits(self, relation_logits):
        relation_logits = relation_logits.detach().cpu().numpy()
        sorted_relation_idxs = np.argsort(-relation_logits, axis=1)
        return sorted_relation_idxs
