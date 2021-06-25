import torch
import torch.nn as nn


def get_criterion():
    loss = None
    if args.criterion == 'bce':
        loss = nn.BCELoss(reduction="none")
    if args.criterion == 'crossentropy':
        loss = nn.CrossEntropyLoss()
    return loss# loss(pred, target)