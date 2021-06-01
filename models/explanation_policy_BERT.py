from transformers import BertPreTrainedModel, BertModel
from torch.nn import Dropout, Linear, CrossEntropyLoss

import torch


class explanation_policy_BERT(BertPreTrainedModel):
    def __init__(self, config):
        super(explanation_policy_BERT, self).__init__(config)
        self.bert = BertModel(config)
        self.dropout = Dropout(config.dropout_rate)
        self.token_layer = Linear(config.hidden_size, 2)
        self.loss_fct = CrossEntropyLoss(reduction="none")
        # initialize all weights in the model, calling from_pretrained() will override pretrained weights only
        self.init_weights()

    def forward(self, input_ids, attention_mask, segment_ids):
        outputs = self.bert(
            input_ids=input_ids, attention_mask=attention_mask, token_type_ids=segment_ids)
        last_hidden_layer_BERT = outputs[0]
        last_hidden_layer_BERT = self.dropout(last_hidden_layer_BERT)
        token_logits = self.token_layer(last_hidden_layer_BERT)
        start_logits, end_logits = token_logits.split(1, dim=-1)
        start_logits = start_logits.squeeze(-1)
        end_logits = end_logits.squeeze(-1)

        return start_logits, end_logits

    def calculate_loss(self, input_ids, attention_mask, segment_ids, start_labels, end_labels):
        start_logits, end_logits = self.forward(
            input_ids, attention_mask, segment_ids)

        start_loss = torch.mean(self.loss_fct(start_logits, start_labels))
        end_loss = torch.mean(self.loss_fct(end_logits, end_labels))

        return start_loss + end_loss

    def calculate_loss_from_logits(self, logits, labels):
        return self.loss_fct(logits, labels)

    def predict(self, input_ids, attention_mask, segment_ids):
        start_logits, end_logits = self.forward(
            input_ids, attention_mask, segment_ids)

        start_preds = torch.argmax(start_logits, dim=1)
        end_preds = torch.argmax(end_logits, dim=1)

        return start_preds, end_preds
