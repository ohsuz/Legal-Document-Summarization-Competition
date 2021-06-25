"""Trainer 클래스 정의
"""

import torch
import wandb
import os
from .dataset import get_train_loaders
from .optimizer import get_optimizer
from .scheduler import get_scheduler
from .criterion import get_criterion
from .metric import get_metric
from .model import Summarizer


def get_model(args):
    if args.model == 'base': model = Summarizer(args)

    model.to(args.device)

    return model


def update_params(loss, model, optimizer, args):
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip_grad)
    optimizer.step()
    optimizer.zero_grad()
    
    
def save_checkpoint(state, model_dir, model_filename):
    print('Saving model ...')
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)    
    torch.save(state, os.path.join(model_dir, model_filename))

    
def run(args):
    train_loader, val_loader = get_train_loaders(args)
    """
    # only when using warmup scheduler
    args.total_steps = int(len(train_loader.dataset) / args.batch_size) * (args.n_epochs)
    # 총 10번 warmup
    args.warmup_steps = args.total_steps // 10
    """
    
    model = get_model(args)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    
    early_stopping_counter = 0
  
    for epoch in range(args.n_epochs):
        print(f"Start Training: Epoch {epoch + 1}")
        
        train_loss, train_score = train(train_loader, model, optimizer, args)
        val_loss, val_score = validate(valid_loader, model, args)

        wandb.log({"epoch": epoch, 
                   "train_loss": train_loss, "train_score": train_score,
                   "val_loss":val_loss, "val_score":val_score})
        
        if val_score > best_score:
            best_score = val_score
            # torch.nn.DataParallel로 감싸진 경우 원래의 model을 가져옵니다.
            model_to_save = model.module if hasattr(model, 'module') else model
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model_to_save.state_dict(),
                },
                args.output_dir, f'{args.run_name}_{epoch}.pt',
            )
            early_stopping_counter = 0
        else:
            early_stopping_counter += 1
            if early_stopping_counter >= args.patience:
                print(f'EarlyStopping counter: {early_stopping_counter} out of {args.patience}')
                break

        # scheduler
        if args.scheduler == 'plateau':
            scheduler.step(best_auc)
        else:
            scheduler.step()

    
def train(train_loader, model, optimizer, args):
    model.train()
    train_total_loss = 0
    pred_lst = []
    target_lst = []
    
    for step, (data, target) in enumerate(train_loader):
        src = data[0].to(args.device)
        clss = data[1].to(args.device)
        segs = data[2].to(args.device)
        mask = data[3].to(args.device)
        mask_clss = data[4].to(args.device)

        target = target.float().to(args.device)
        sent_score = model(src, segs, clss, mask, mask_clss)

        # compute loss
        loss = get_criterion(sent_score, target)
        loss = (loss * mask_clss.float()).sum()
        train_total_loss += loss

        # update parameters
        update_params(loss, model, optimizer, args)

        pred_lst.extend(torch.topk(sent_score, 3, axis=1).indices.tolist())
        target_lst.extend(torch.where(target==1)[1].reshape(-1,3).tolist())

        if step % args.log_steps == 0:
            print(f"step {step}, loss: {loss.item()}")

    train_mean_loss = train_total_loss / len(train_loader)
    train_score = get_metric(targets=target_lst, preds=pred_lst)
    msg = f'[Train] Loss: {train_mean_loss}, Score: {train_score}'
    print(msg)
    #self.logger.info(msg) if self.logger else print(msg)
    return train_mean_loss, train_score
    
    
def validate(val_loader, model, args):
    """ 한 epoch에서 수행되는 검증 절차

    Args:
        dataloader (`dataloader`)
        epoch_index (int)
    """
    model.eval()
    val_total_loss = 0
    pred_lst = []
    target_lst = []

    with torch.no_grad():
        for batch_index, (data, target) in enumerate(val_loader):
            src = data[0].to(args.device)
            clss = data[1].to(args.device)
            segs = data[2].to(args.device)
            mask = data[3].to(args.device)
            mask_clss = data[4].to(args.device)
            target = target.float().to(args.device)

            sent_score = model(src, segs, clss, mask, mask_clss)
            loss = get_criterion(sent_score, target)
            loss = (loss * mask_clss.float()).sum()
            val_total_loss += loss

            pred_lst.extend(torch.topk(sent_score, 3, axis=1).indices.tolist())
            target_lst.extend(torch.where(target==1)[1].reshape(-1,3).tolist())

        val_mean_loss = val_total_loss / len(val_loader)
        validation_score = get_metric(targets=target_lst, preds=pred_lst)
        msg = f'[Validation] Loss: {val_mean_loss}, Score: {validation_score}'
        print(msg)
        #self.logger.info(msg) if self.logger else print(msg)
        
        return val_mean_loss, validation_score
