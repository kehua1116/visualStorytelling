import argparse
import time
import torch
import torch.nn as nn
from bart import BART
import torch.nn.functional as F
import dill as pickle
from tqdm import tqdm
from build_vocab import Vocabulary
from dataset import get_loader
import itertools
import opts
import random
from torch.optim.lr_scheduler import MultiStepLR

LR = 4e-5
ADAM_EPSILON = 1e-8
WEIGHT_DECAY = 0.
WARMUP_PROPORTION = 0.1

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        print(param_group['lr'])


# Train Model
def train_model(opt, bart, dataset, test_dataset, epochs, name):
    bart._model.split_to_gpus(1)
    print("training model...")
    train_steps = 10 * (len(dataset) + 1)
    warmup_steps = train_steps*0.1
    print(train_steps, warmup_steps)
    print(bart._model)


    # Training Image Encoder
    bart.get_optimizer(
        lr=LR,
        train_steps=train_steps,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        adam_epsilon=ADAM_EPSILON)


    for epoch in range(5):
        # bart._model.train()
        total_loss = []       
        for i, (src, tgt_texts, keys) in enumerate(tqdm(dataset)):
            src = src.cuda()
            for idx in range(len(src)):            
                # print(tgt_texts[idx])  
                bart._optimizer.zero_grad() 
                loss = bart._get_seq2seq_loss(
                        src_feat = src[idx].unsqueeze(0), keys = keys[idx], tgt_text=tgt_texts[idx])
                loss = loss/len(src)
                loss.backward()
                total_loss.append(loss.item())
            bart._optimizer.step()
            # if i % 1000 == 0:
            #     bart.save_model(f'models/{epoch}_model.pt')
            #     print(f"total loss : ", sum(total_loss)/len(total_loss), "loss", loss.item())
            bart._lr_scheduler.step()
        print(f"Epoch {epoch} ce loss : ", sum(total_loss)/len(total_loss))
        with open (f'models/emotion_experiments/training_loss_{name}.txt', "a") as f:
            f.write(f"Epoch {epoch} ce loss : {sum(total_loss)/len(total_loss)}\n")
    bart.save_model(f'models/emotion_experiments/{name}_visualencoder.pt')


    # Training BART Model
    bart.get_bart_optimizer(
        lr=LR,
        train_steps=train_steps,
        warmup_steps=warmup_steps,
        weight_decay=WEIGHT_DECAY,
        adam_epsilon=ADAM_EPSILON)

    for epoch in range(5):
        # bart._model.train()
        total_loss = []       
        # eval_loss = evaluation(model, SRC, TRG, test_dataset)
        for i, (src, tgt_texts, keys) in enumerate(tqdm(dataset)):
            src = src.cuda()
            for idx in range(len(src)):            
                # print(tgt_texts[idx])  
                bart._optimizer.zero_grad() 
                loss = bart._get_seq2seq_loss(
                        src_feat = src[idx].unsqueeze(0), keys = keys[idx], tgt_text=tgt_texts[idx])
                loss = loss/len(src)
                loss.backward()
                total_loss.append(loss.item())
            bart._optimizer.step()
            # if i % 1000 == 0:
            #     bart.save_model(f'models/{epoch}_model.pt')
            #     print(f"total loss : ", sum(total_loss)/len(total_loss), "loss", loss.item())
            bart._lr_scheduler.step()
            
        print(f"Epoch {epoch} ce loss : ", sum(total_loss)/len(total_loss))
        with open (f'models/emotion_experiments/training_loss_{name}.txt', "a") as f:
            f.write(f"Epoch {epoch} ce loss : {sum(total_loss)/len(total_loss)}\n")
    bart.save_model(f'models/emotion_experiments/{epoch}_{name}.pt')


def main():

    opt = opts.parse_opt()

    opt.src_data = '../../data/vist/data/visualstorytelling/train.pkl'
    opt.test_src_data = '../../data/vist/data/visualstorytelling/test.pkl'

    # Load Dataset
    dataset = get_loader(opt.src_data, 
                        opt, 
                        opt.batch_size, 
                        train=True
                        )
    test_dataset = get_loader(opt.test_src_data, 
                            opt, 
                            opt.batch_size, 
                            train=False, 
                            shuffle=False
                            )

    # get model
    bart = BART(opt)
    # bart.load_model("models/emotion_experiments/multi_emotion_noconcept_0.9_visualencoder.pt")
    name = "multi_emotion_noconcept_0.9"
    train_model(opt, bart, dataset, test_dataset, 10, name=name)


if __name__ == "__main__":
    main()
