"""
"""
import torch
from torch import nn
import transformers
from sklearn.metrics import f1_score



class Summarizer(nn.Module):

    def __init__(self):
        """
        """
        super(Summarizer, self).__init__()
        self.encoder = transformers.BertModel.from_pretrained("beomi/KcELECTRA-base")
        self.fc = nn.Linear(768, 1)
        self.sigmoid = nn.Sigmoid()


    def forward(self, x, segs, clss, mask, mask_clss):
        """
        """
        top_vec = self.encoder(input_ids = x.long(), attention_mask = mask.float(), token_type_ids = segs.long()).last_hidden_state
        sents_vec = top_vec[torch.arange(top_vec.size(0)).unsqueeze(1), clss.long()]
        sents_vec = sents_vec * mask_clss[:, :, None].float()
        h = self.fc(sents_vec).squeeze(-1)
        sent_scores = self.sigmoid(h) * mask_clss.float()
        return sent_scores


