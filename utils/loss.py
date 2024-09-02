import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
class RelaHashLoss(nn.Module):
    def __init__(self,
                 beta=10,
                 m=0.5,
                 multiclass=False,
                 onehot=True,
                 **kwargs):
        super(RelaHashLoss, self).__init__()
        self.beta = beta
        self.m = m
        self.multiclass = multiclass
        self.onehot = onehot

    def compute_margin_logits(self, logits, labels):
        if self.multiclass:
            y_onehot = labels * self.m
            # print(y_onehot)
            margin_logits = self.beta * (logits - y_onehot)
        else:
            y_onehot = torch.zeros_like(logits)
            y_onehot.scatter_(1, torch.unsqueeze(labels, dim=-1), self.m)
            margin_logits = self.beta * (logits - y_onehot)
        return margin_logits

    def forward(self, logits,  labels):
        if self.multiclass:
            if not self.onehot:
                labels = F.one_hot(labels, logits.size(1))

            labels = labels.float()


            margin_logits = self.compute_margin_logits(logits, labels)

            log_logits = F.log_softmax(margin_logits, dim=1)
            # t2 = margin_logits.cpu().detach().numpy()
            # np.savetxt('log_logits.csv', t2, delimiter=',')
            # lable_num = labels.shape[1]
            A = ((labels==0).sum(dim=1) == labels.shape[1])
            labels[A==True] = 1
            labels_scaled = labels / labels.sum(dim=1, keepdim=True)
            # t3 = labels_scaled.cpu().detach().numpy()
            # np.savetxt('labels_scale.csv', t3, delimiter=',')
            loss = - (labels_scaled * log_logits).sum(dim=1)
            # print(loss)
            # t1 = loss.cpu().detach().numpy()
            # np.savetxt('loss.csv',t1,delimiter=',')
            loss = loss.mean()
        else:
            if self.onehot:
                labels = labels.argmax(1)

            margin_logits = self.compute_margin_logits(logits, labels)
            loss = F.cross_entropy(margin_logits, labels)
        return loss
