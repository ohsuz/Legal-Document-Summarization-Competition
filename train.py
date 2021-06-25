import warnings
warnings.filterwarnings('ignore')


from model.model import Summarizer
from modules.dataset import get_train_loaders
from modules.utils import get_logger, make_directory, seed_everything, save_json
from modules.earlystoppers import LossEarlyStopper
from modules.recorders import PerformanceRecorder

import wandb
from args import parse_args
from modules import trainer

import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd

from datetime import datetime, timezone, timedelta
from importlib import import_module
import os
import json
import random
import argparse



def main(args):
    wandb.login()
    
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    args.device = device
    print(f"PyTorch version:[{torch.__version__}]")
    print(f"device:[{args.device}]")
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")

    wandb.init(project='AI-Online-Competition', config=vars(args), entity="ohsuz", name=args.run_name)
    trainer.run(args)
    wandb.finish()


if __name__ == "__main__":
    args = parse_args(mode='train')
    main(args)
    
    
"""
def train(model, optimizer, scheduler, criterion, train_dataloader, validation_dataloader):
    # Set metrics
    metric_fn = Hitrate

    # Set trainer
    trainer = Trainer(model, CFG.device, criterion, metric_fn, optimizer, scheduler, logger=CFG.system_logger)

    # Set earlystopper
    early_stopper = LossEarlyStopper(patience=CFG.early_stopping_patience, verbose=True, logger=CFG.system_logger)

    # Set performance recorder
    key_column_value_list = [
        CFG.TRAIN_SERIAL,
        CFG.TRAIN_TIMESTAMP,
        CFG.early_stopping_patience,
        CFG.batch_size,
        CFG.epochs,
        CFG.learning_rate,
        CFG.seed]

    performance_recorder = PerformanceRecorder(column_name_list=CFG.PERFORMANCE_RECORD_COLUMN_NAME_LIST,
                                               record_dir=CFG.PERFORMANCE_RECORD_DIR,
                                               key_column_value_list=key_column_value_list,
                                               logger=CFG.system_logger,
                                               model=model,
                                               optimizer=optimizer,
                                               scheduler=scheduler)

    best_score = 0
    for epoch_index in range(CFG.epochs):
        trainer.train_epoch(train_dataloader, epoch_index=epoch_index)
        trainer.validate_epoch(validation_dataloader, epoch_index=epoch_index)

        # Performance record - csv & save elapsed_time
        performance_recorder.add_row(epoch_index=epoch_index,
                                     train_loss=trainer.train_mean_loss,
                                     validation_loss=trainer.val_mean_loss,
                                     train_score=trainer.train_score,
                                     validation_score=trainer.validation_score)

        # Performance record - plot
        performance_recorder.save_performance_plot(final_epoch=epoch_index)

        # early_stopping check
        early_stopper.check_early_stopping(loss=trainer.val_mean_loss)

        if early_stopper.stop:
            print('Early stopped')
            break

        if trainer.validation_score > best_score:
            best_score = trainer.validation_score
            performance_recorder.weight_path = os.path.join(CFG.PERFORMANCE_RECORD_DIR, 'best.pt')
            performance_recorder.save_weight()    
"""