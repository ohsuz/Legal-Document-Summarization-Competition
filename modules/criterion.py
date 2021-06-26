import torch
import torch.nn as nn


def get_criterion(pred, target, args):
    loss = None
    if args.criterion == 'bce':
        loss = nn.BCELoss()
    if args.criterion == 'crossentropy':
        loss = nn.CrossEntropyLoss()
    return loss(pred, target)