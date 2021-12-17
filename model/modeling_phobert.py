from transformers.models.roberta.modeling_roberta import RobertaModel, RobertaPreTrainedModel
import torch.nn as nn


class CrossEncoderPhoBERT(RobertaPreTrainedModel):
    def __init__(self, config):
        super(CrossEncoderPhoBERT, self).__init__(config)

        self.roberta = RobertaModel(config)
        self.classifier = nn.Sequential(nn.Dropout(p=config.hidden_dropout_prob),
                                        nn.Linear(config.hidden_size, 2))

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,
                is_relevant=None):
        outputs = self.roberta(input_ids=input_ids,
                               attention_mask=attention_mask,
                               token_type_ids=token_type_ids)

        loss = 0
        pooled_output = outputs[1]
        logits = self.classifier(pooled_output)

        if is_relevant is not None:
            loss_fct = nn.CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, 2), is_relevant.view(-1))

        return loss, logits
