import warnings
warnings.filterwarnings('ignore')


from model.model import Summarizer
from modules.dataset import get_train_loaders
from modules.trainer import Trainer
from modules.utils import get_logger, make_directory, get_train_config, seed_everything, save_json
from modules.criterion import create_criterion
from modules.optimizer import create_optimizer
from modules.scheduler import create_scheduler
from modules.earlystoppers import LossEarlyStopper
from modules.metrics import Hitrate
from modules.recorders import PerformanceRecorder
from args import parse_args

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


def get_model(train_dataloader):
    '''
        get defined model from model.py
        
        Returns:
            model: pytorch model that would be trained
            optimizer: pytorch optimizer for gradient descent
            scheduler: pytorch lr scheduler
            criterion: loss function
    '''

    # Load Model
    model_module = getattr(import_module("model.model"), CFG.model)
    model = model_module().to(CFG.device)

    CFG.system_logger.info('===== Review Model Architecture =====')
    CFG.system_logger.info(f'{model} \n')

    # get optimizer from optimizer.py
    optimizer = create_optimizer(
        CFG.optimizer,
        params = model.parameters(),
        lr = CFG.learning_rate,
        **CFG.optimizer_params)

    # get scheduler from scheduler.py
    scheduler = create_scheduler(
        CFG.scheduler,
        optimizer = optimizer,
        pct_start=0.1,
        div_factor=1e5,
        max_lr=0.0001,
        epochs=CFG.epochs,
        anneal_strategy='cos',
        steps_per_epoch=len(train_dataloader))
        # **CFG.scheduler_params)

    # get criterion from criterion.py
    criterion = create_criterion(
        CFG.criterion,
        **CFG.criterion_params)

    return model, optimizer, scheduler, criterion


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


def main(args):
    # check pytorch version & whether using cuda or not
    print(f"PyTorch version:[{torch.__version__}]")
    print(f"device:[{CFG.device}]")
    print(f"GPU 이름: {torch.cuda.get_device_name(0)}")

    parser = argparse.ArgumentParser(description="AIOnlineCompetition")
    parser.add_argument("--config", type=str, default="base_config.json", help=f'train config file (defalut: base_config.json)')
    args = parser.parse_args()

    # parsing config from custom config.json file
    get_train_config(CFG, os.path.join(CFG.PROJECT_DIR, 'configs', 'train', args.config))

    # set every random seed
    seed_everything(CFG.seed)

    # Set train result directory
    make_directory(CFG.PERFORMANCE_RECORD_DIR)

    # Save config json file
    with open(os.path.join(CFG.PROJECT_DIR, 'configs', 'train', args.config)) as f:
        config = json.load(f)
    save_json(os.path.join(CFG.PERFORMANCE_RECORD_DIR, 'train_config.json'), config)

    # Set system logger
    CFG.system_logger = get_logger(name='train',
                                   file_path=os.path.join(CFG.PERFORMANCE_RECORD_DIR, 'train_log.log'))

    # set pytorch dataset & loader
    train_dataloader, validation_dataloader = get_data_utils()

    # get model, optimizer, criterion(not for this task), and scheduler
    model, optimizer, scheduler, criterion = get_model(train_dataloader)

    # train
    train(model, optimizer, scheduler, criterion, train_dataloader, validation_dataloader)


if __name__ == "__main__":
    args = parse_args(mode='train')
    main(args)