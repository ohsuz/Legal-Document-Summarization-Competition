import os
import pandas as pd
import torch
import modules.utils as utils
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from itertools import chain
from sklearn.model_selection import train_test_split

from pprint import pprint


class CustomDataset(Dataset):
    def __init__(self, args, data, mode):
        self.data = data
        self.data_dir = args.data_dir
        self.mode = mode
        self.tokenizer = AutoTokenizer.from_pretrained(args.model_name)
        self.inputs, self.labels = self.data_loader()

    def data_loader(self):
        print('Loading ' + self.mode + ' dataset..')
        if os.path.isfile(os.path.join(self.data_dir, self.mode, self.mode + '_X.pt')):
            inputs = torch.load(os.path.join(self.data_dir, self.mode, self.mode + '_X.pt'))
            labels = torch.load(os.path.join(self.data_dir, self.mode, self.mode + '_Y.pt'))

        else:
            df = self.data
            inputs = pd.DataFrame(columns=['src'])
            labels = pd.DataFrame(columns=['trg'])
            inputs['src'] =  df['article_original']

            if self.mode != "test":
                labels['trg'] =  df['extractive']

            # Preprocessing
            inputs, labels = self.preprocessing(inputs, labels)
            print("preprocessing")

            # Save data
            torch.save(inputs, os.path.join(self.data_dir, self.mode, self.mode + '_X.pt'))
            torch.save(labels, os.path.join(self.data_dir, self.mode, self.mode + '_Y.pt'))

        inputs = inputs.values
        labels = labels.values

        return inputs, labels

    def pad(self, data, pad_id, max_len):
        padded_data = data.map(lambda x : torch.cat([x, torch.tensor([pad_id] * (max_len - len(x)))]))
        return padded_data

    def preprocessing(self, inputs, labels):
        print('Preprocessing ' + self.mode + ' dataset..')

        # Encoding original text
        inputs['src'] = inputs['src'].map(lambda x: torch.tensor(list(chain.from_iterable([self.tokenizer.encode(x[i], max_length = int(512 / len(x)), add_special_tokens=True) for i in range(len(x))]))))
        inputs['clss'] = inputs.src.map(lambda x : torch.cat([torch.where(x == 2)[0], torch.tensor([len(x)])]))
        inputs['segs'] = inputs.clss.map(lambda x : torch.tensor(list(chain.from_iterable([[0] * (x[i+1] - x[i]) if i % 2 == 0 else [1] * (x[i+1] - x[i]) for i, val in enumerate(x[:-1])]))))
        inputs['clss'] = inputs.clss.map(lambda x : x[:-1])
        
        # Padding
        max_encoding_len = max(inputs.src.map(lambda x: len(x)))
        max_label_len = max(inputs.clss.map(lambda x: len(x)))
        inputs['src'] = self.pad(inputs.src, 0, max_encoding_len)
        inputs['segs'] = self.pad(inputs.segs, 0, max_encoding_len)
        inputs['clss'] = self.pad(inputs.clss, -1, max_label_len)
        inputs['mask'] = inputs.src.map(lambda x: ~ (x == 0))
        inputs['mask_clss'] = inputs.clss.map(lambda x: ~ (x == -1))

        # Binarize label {Extracted sentence : 1, Not Extracted sentence : 0}

        if self.mode != 'test':
            labels = labels['trg'].map(lambda  x: torch.tensor([1 if i in x else 0 for i in range(max_label_len)]))

        return inputs, labels


    def __len__(self):
        return len(self.inputs)


    def __getitem__(self, index):
        if self.mode == 'test':
            return [self.inputs[index][i] for i in range(5)]
        else:
            return [self.inputs[index][i] for i in range(5)], self.labels[index]


def get_train_loaders(args):
    """
        define train/validation pytorch dataset & loader

        Returns:
            train_loader: pytorch data loader for train data
            val_loader: pytorch data loader for validation data
    """
    # get data from json
    with open(os.path.join(args.data_dir, "train.json"), "r", encoding="utf-8-sig") as f:
        data = pd.read_json(f) 
    train_df = pd.DataFrame(data)
    
    # split train & test data
    train_data, val_data = train_test_split(train_df, test_size=0.1, random_state=args.seed)
    
    # get train & valid dataset from dataset.py
    train_dataset = CustomDataset(args, train_data, mode='train')
    val_dataset = CustomDataset(args, val_data, mode='valid')

    # define data loader based on each dataset
    train_dataloader = DataLoader(dataset=train_dataset,
                                  batch_size=args.batch_size,
                                  num_workers=args.num_workers,
                                  pin_memory=True,
                                  drop_last=False,
                                  shuffle=True)
    val_dataloader = DataLoader(dataset=val_dataset,
                                batch_size=args.batch_size,
                                num_workers=args.num_workers,
                                pin_memory=True,
                                drop_last=False,
                                shuffle=False)

    return train_dataloader, val_dataloader
    
    
def get_test_loader(args):
    # Get data from json
    with open(os.path.join(args.data_dir, "test.json"), "r", encoding="utf-8-sig") as f:
        data = pd.read_json(f) 
    test_df = pd.DataFrame(data)
    
    # Load dataset & dataloader
    test_dataset = CustomDataset(args, test_df, mode='test')
    test_dataloader = DataLoader(dataset=test_dataset,
                                 batch_size=args.batch_size,
                                 num_workers=args.num_workers,
                                 pin_memory=True,
                                 drop_last=False,
                                 shuffle=False)
    
    return test_dataloader