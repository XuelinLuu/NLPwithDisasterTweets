import transformers
import torch.nn as nn

class TweetReconModel(nn.Module):
    def __init__(self, bert_path):
        super(TweetReconModel, self).__init__()
        self.bert = transformers.BertModel.from_pretrained(bert_path)
        self.dropout = nn.Dropout(p=0.3)
        self.classifier = nn.Linear(self.bert.config.hidden_size, 2)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, input_ids, token_type_ids, attention_mask):
        bert_output = self.bert(input_ids=input_ids, token_type_ids=token_type_ids, attention_mask=attention_mask)
        output = bert_output[1]
        output = self.dropout(output)
        output = self.classifier(output)
        return self.softmax(output)
