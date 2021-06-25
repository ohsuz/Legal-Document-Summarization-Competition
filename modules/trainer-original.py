"""Trainer 클래스 정의
"""

import torch
import wandb
from .dataset import get_train_dataloaders
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion
from .metric import get_metric
from .model import *

def get_model(args):
    """
    Load model and move tensors to a given devices.
    """
    if args.model == 'base': model = Summarizer(args)

    model.to(args.device)

    return model


class Trainer():
    """ Trainer
        epoch에 대한 학습 및 검증 절차 정의
    
    Attributes:
        model (`model`)
        device (str)
        loss_fn (Callable)
        metric_fn (Callable)
        optimizer (`optimizer`)
        scheduler (`scheduler`)
    """

    def __init__(self, args, logger=None):
        """ 초기화
        """
        self.device = args.device
        self.model = get_model(args)
        self.optimizer = get_optimizer(model, args)
        self.scheduler = get_scheduler(optimizer, args)
        self.criterion = get_criterion()
        self.metric_fn = get_metric()
        self.logger = logger

    def train_epoch(self, dataloader, epoch_index):
        """ 한 epoch에서 수행되는 학습 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
        """
        self.model.train()
        self.train_total_loss = 0
        pred_lst = []
        target_lst = []
        for step, (data, target) in enumerate(dataloader):
            self.optimizer.zero_grad()

            src = data[0].to(self.device)
            clss = data[1].to(self.device)
            segs = data[2].to(self.device)
            mask = data[3].to(self.device)
            mask_clss = data[4].to(self.device)

            target = target.float().to(self.device)

            sent_score = self.model(src, segs, clss, mask, mask_clss)

            loss = self.criterion(sent_score, target)
            loss = (loss * mask_clss.float()).sum()
            self.train_total_loss += loss

            loss.backward()
            self.optimizer.step()
            self.scheduler.step()

            pred_lst.extend(torch.topk(sent_score, 3, axis=1).indices.tolist())
            target_lst.extend(torch.where(target==1)[1].reshape(-1,3).tolist())

            if step % 100 == 0:
                print(f"step {step}, loss: {loss.item()}")
            
        self.train_mean_loss = self.train_total_loss / len(dataloader)
        self.train_score = self.metric_fn(y_true=target_lst, y_pred=pred_lst)
        msg = f'Epoch {epoch_index}, Train, loss: {self.train_mean_loss}, Score: {self.train_score}'
        print(msg)
        self.logger.info(msg) if self.logger else print(msg)


    def validate_epoch(self, dataloader, epoch_index):
        """ 한 epoch에서 수행되는 검증 절차

        Args:
            dataloader (`dataloader`)
            epoch_index (int)
        """
        self.model.eval()
        self.val_total_loss = 0
        pred_lst = []
        target_lst = []

        with torch.no_grad():
            for batch_index, (data, target) in enumerate(dataloader):
                src = data[0].to(self.device)
                clss = data[1].to(self.device)
                segs = data[2].to(self.device)
                mask = data[3].to(self.device)
                mask_clss = data[4].to(self.device)
                target = target.float().to(self.device)

                sent_score = self.model(src, segs, clss, mask, mask_clss)
                loss = self.criterion(sent_score, target)
                loss = (loss * mask_clss.float()).sum()
                self.val_total_loss += loss

                pred_lst.extend(torch.topk(sent_score, 3, axis=1).indices.tolist())
                target_lst.extend(torch.where(target==1)[1].reshape(-1,3).tolist())

            self.val_mean_loss = self.val_total_loss / len(dataloader)
            self.validation_score = self.metric_fn(y_true=target_lst, y_pred=pred_lst)
            msg = f'Epoch {epoch_index}, Validation, loss: {self.val_mean_loss}, Score: {self.validation_score}'
            print(msg)
            self.logger.info(msg) if self.logger else print(msg)



