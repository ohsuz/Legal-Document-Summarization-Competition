import torch
import torch.nn as nn


def get_criterion(pred, target):
    loss = None
    if args.criterion == 'bce':
        loss = nn.BCELoss(reduction="none")
    if args.criterion == 'crossentropy':
        loss = nn.CrossEntropyLoss()
    return loss(pred, target)