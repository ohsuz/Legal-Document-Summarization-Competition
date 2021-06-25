"""Trainer 클래스 정의
"""

import torch

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

    def __init__(self, model, device, criterion, metric_fn, optimizer=None, scheduler=None, logger=None):
        """ 초기화
        """
        self.model = model
        self.device = device
        self.criterion = criterion
        self.metric_fn = metric_fn
        self.optimizer = optimizer
        self.scheduler = scheduler
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



