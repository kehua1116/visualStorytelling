import numpy as np
import os
from torch.utils.data import Dataset
import glob
import numpy as np
from tqdm import tqdm
import dill as pickle
import torch
from copy import deepcopy
import json
import random
from collections import Counter 

synonyms = {}
synonyms["happy"] = ["happy", "cheerful", "merry", "delighted", "joyful", "glad", "jubilant", "elated", "pleased", "lively", "thrilled", "upbeast", "contented", "overjoyed", ]
synonyms["angry"] = ["angry", "annoyed", "enraged", "furious", "exasperated", "heated", "indignant", "irritated", "impassioned", "outraged", "resentful", "uptight", "offended", "pissed"]
synonyms["sad"] = ["sad", "bitter", "dismal", "heartbroken", "melancholy", "mournful", "somber", "sorrowful", "unhappy", "wistful", "pessimistic", "down", "gloomy", "morbid"]
synonyms["surprise"] = ["surprised", "amazed", "astonished", "shocked", "bewildered", "dazed", "startled", "stunned", "frightened", "aghast", "appalled", "astounded", "alarmed", "stupefied"]
synonyms["neutral"] = ["neutral", "disinterested", "evenhanded", "fair-minded", "indifferent", "unbiased", "undecided", "uninvolved", "uncommitted", "nonaligned", "nonpartisan", "cool", "aloof", "bystading"]
synonyms["fear"] = ["fear", "alarm", "angst", "concern", "unease", "scared", "terror", "panic", "horror", "worry", "despair", "distress", "frightful", "trembling"]
synonyms["disgust"] = ["disgust", "outraged", "appalled", "queasy", "antipathy", "loathing", "repulsion", "revulsion", "hatred", "aversion", "dislike", "sick", "abominate", "surfeit"]

class ROCdataset():
    def __init__(self, src_dir, opt, train=True):

        self.emotion = opt.emotion
        self.with_concepts = opt.with_concepts
        self.use_synonyms = opt.use_synonyms
        print(f"emotion: {self.emotion}")
        print(f"with_concepts: {self.with_concepts}")
        print(f"use_synonyms: {self.use_synonyms}")

        print("Loading data ...  ")
        # loading training data
        with open(src_dir, 'rb') as f:
            self.src = pickle.load(f)
            
        print('src data length', len(self.src))
        self.data_len = len(self.src)

        print("Loading vocab ... ")
        # loading vocab data
        self.train = train
        self.keys = list(self.src.keys())

        self.story_emotions = {"angry": 0, "disgust": 0, "fear": 0, "happy": 0, "sad": 0, "surprise": 0, "neutral": 0, "no_face": 0}


    def __getitem__(self, index):
        i = index
        index = self.keys[index]
        src_idx = []
        src = self.src[index]
        images = self.src[index]['image']
        texts = self.src[index]['text']


        tokens = self.src[index]['target']
        
        # Get detected emotion class for each image
        emotion_keywords = []
        train_path = "train" if self.train else "test"
        for image in images:
            try:
                with open(f'../../data/vist/emotion_annotations/{train_path}/{image}.json','rb') as f:
                    emotion = json.load(f)
                    dominant_emotion = emotion["dominant_emotion"]
                    if dominant_emotion == "no_face_detected":
                        dominant_emotion = ""
                    emotion_keywords.append(dominant_emotion)
            except:
                with open(f'missing.txt','a') as f:
                    f.write(f"{image}.jpg")
                    f.write("\n")
                    print(f"Not found: {image}")
                emotion_keywords.append("")            

            

        # Accprding to emotion class, get corresponding synonyms based on configuration
        synonym_keywords = [[] for _ in range(len(tokens))]
        counter = Counter(emotion_keywords)

        if len(counter) == 1 and "" in counter: #no face detected in any images
            self.story_emotions["no_face"] += 1
            pass
        else:
            if self.emotion == "single_emotion":
                emo = counter.most_common(1)[0][0]

                if emo == "":
                    emo = counter.most_common(2)[-1][0]
                self.story_emotions[emo] += 1

                for i in range(len(tokens)):
                    if self.use_synonyms: 
                        synonym_keys = []
                        for t in synonyms[emo]:
                            if np.random.rand() > 0.5: # randomly sample synonyms to incorporate
                                synonym_keys.append(t)
                    else:
                        synonym_keys = [emo]

                    synonym_keywords[i] = synonym_keys
            elif self.emotion == "multi_emotion":
                for i in range(len(tokens)):
                    emo = emotion_keywords[i]
                    if emo == "":
                        continue
                    if self.use_synonyms:
                        synonym_keys = []
                        for t in synonyms[emo]:
                            if np.random.rand() > 0.9:
                                synonym_keys.append(t)
                    else:
                        synonym_keys = [emo]
                    synonym_keywords[i] = synonym_keys

        # Get Commonsense Concept Keywords   
        image_keywords = []
        for image in images:
            with open(f'../../data/vist/data/clarifai/train/{image}.json','rb') as f:
                image_key = json.load(f)
            image_keywords.append(image_key)

        all_keywords = []
        for k ,token in enumerate(tokens):
            new_token = []

            if self.emotion == "no_emotion": # baseline
                if self.with_concepts:
                    token = token + image_keywords[k]
                else:
                    token = token
            else: # single or multi emotion
                if self.with_concepts:
                    token = token  + image_keywords[k] + synonym_keywords[k]
                else:
                    token = token + synonym_keywords[k]

            if self.use_synonyms: 
                for t in token:
                    if np.random.rand() > 0.2:
                        new_token.append(t)
            else:
                new_token = token
            random.shuffle(new_token)
            all_keywords.append(new_token)

        # Concatenate concept keywords with emotion keywords
        keywords = " <SEP> "
        for keys in all_keywords:
            for k in keys:
                keywords += k+' '
            keywords += '<SEP> '
        keywords = keywords[:-1]

        im_feats = []
        for im in images:
            if self.train:
                f = np.load(f'../../data/vist/data/AREL/dataset/resnet_features/fc/train/{im}.npy')
                im_feats.append(f)
            else:
                f = np.load(f'../../data/vist/data/AREL/dataset/resnet_features/fc/test/{im}.npy')
                im_feats.append(f)             

        im_feats = torch.tensor(im_feats)
        return im_feats, " "+" ".join(texts), keywords

    def __len__(self):
        return self.data_len




def get_loader(src_dir, opt, batch_size, train, shuffle=True):
    dataset = ROCdataset(src_dir, opt)

    data_loader = torch.utils.data.DataLoader(dataset,
                                              batch_size=batch_size,
                                              shuffle=shuffle,
                                              num_workers=10,
                                              drop_last=True)
    return data_loader
