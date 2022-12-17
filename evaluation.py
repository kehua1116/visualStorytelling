import warnings
warnings.simplefilter("ignore", UserWarning)
import numpy as np
from tqdm import tqdm
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import dill as pickle
import argparse
from nltk.corpus import wordnet
from build_vocab import Vocabulary
from dataset import get_loader
from copy import deepcopy
import opts
import misc.utils as utils
from vist_eval.album_eval import AlbumEvaluator
import json
import random
from bart import BART
from dataset import synonyms
from collections import Counter 
total = 0
correct = 0


class Evaluator:
    def __init__(self, opt):
        ref_json_path = "../../data/vist/data/visualstorytelling/reference/test_reference.json"
        self.reference = json.load(open(ref_json_path))
        print("loading file {}".format(ref_json_path))
        self.eval = AlbumEvaluator()

    def measure(self, filename, emos):
        self.prediction_file = filename
        predictions = {}
        with open(self.prediction_file) as f:
            for line in f:
                vid, seq = line.strip().split('\t')
                if vid not in predictions:
                    predictions[vid] = [seq]
        self.eval.evaluate(self.reference, predictions)
        for key, val in emos.items():
            predictions[key].insert(0, val)
        with open(filename, 'w') as f:
            json.dump(predictions, f)
        print(self.eval.eval_overall)
        return self.eval.eval_overall



def generate_sentence(album, src, model):
    with open('../../data/vist/data/visualstorytelling/new_test_story.pkl','rb') as f:
        test_src = pickle.load(f)
    
    images = test_src[album]
    feats = []
    for im in images:
        feat = np.load(f'../../data/vist/data/AREL/dataset/resnet_features/fc/test/{im}.npy')
        feats.append(deepcopy(feat))
    feats = torch.tensor(feats).unsqueeze(0)

    keywords = src
    # keywords = ' '.join(keywords)
    story = model.generate(cond=feats, keys=keywords, top_k=-1, top_p=0.9)
    print(album, story)
    return story

def generate(opt, model, name):
    global total
    global correct
    evaluator = Evaluator(opt)
    with_concepts = opt.with_concepts
    emotion = opt.emotion
    use_synonyms = opt.use_synonyms

    with open('res/graph2keyword.json','r') as f:
        test_key = json.load(f)  
    album_ids = list(test_key.keys())
    with open('../../data/vist/data/visualstorytelling/album2story.json','r') as f:
        album2story = json.load(f)
    story2album = {v: k for k, v in album2story.items()}

    hypos = {}
    emos = {}
    images = []
    res = []

    with open('../../data/vist/data/visualstorytelling/new_test_story.pkl','rb') as f:
        test_src = pickle.load(f)
    for album_id in tqdm(album_ids):

        src = test_key[album_id]
        images = test_src[album_id]


        # Generate Emotion keywords According to Configuration 
        emotion_keywords = []
        for image in images:
            try:
                with open(f'../../data/vist/emotion_annotations/test/{image}.json','rb') as f:
                    emotions = json.load(f)
                    dominant_emotion = emotions["dominant_emotion"]
                    if dominant_emotion == "no_face_detected":
                        dominant_emotion = ""
                    emotion_keywords.append(dominant_emotion)
            except:
                with open(f'missing.txt','a') as f:
                    f.write(f"{image}.jpg")
                    f.write("\n")
                    print(f"Not found: {image}")
                emotion_keywords.append("")
    

        if emotion != "no_emotion":
            synonym_keywords = [[] for _ in range(len(images))]
            counter = Counter(emotion_keywords)
            if len(counter) == 1 and "" in counter: #no face detected in any images
                emos[album_id] = "no_face"
                pass
            else:
                if emotion == "single_emotion":
                    emo = counter.most_common(1)[0][0] #find the most common emotion in story
                    if emo == "":
                        emo = counter.most_common(2)[-1][0]
                    if use_synonyms:
                        for i in range(len(images)):
                            synonym_keywords[i] = synonyms[emo]
                    else:
                        for i in range(len(images)):
                            synonym_keywords[i] = [emo]
                elif emotion == "multi_emotion": # each image has its own emotion keywords
                    for i in range(len(images)):
                        emo = emotion_keywords[i]
                        if emo == "":
                            continue
                        if use_synonyms:
                            synonym_keywords[i] = synonyms[emo]
                        else:
                            synonym_keywords[i] = [emo]
            
                emos[album_id] = counter.most_common(1)[0][0] if counter.most_common(1)[0][0] != "" \
                            else counter.most_common(2)[-1][0]

        # Get Commonsense Concept Keywords
        image_keywords = []
        for image in images:
            with open(f'../../data/vist/data/clarifai/test/{image}.json','rb') as f:
                image_key = json.load(f)
            image_keywords.append(image_key)


        # Concatenate image keywords and concept keywords
        keywords = []
        keywords = " <SEP> "
        for l, keys in enumerate(src):
            
            if emotion == "no_emotion":
                if with_concepts:
                    keys = keys + image_keywords[l]
                else:
                    keys = ""
            else:
                if with_concepts:
                    keys = keys + image_keywords[l] + synonym_keywords[l]
                else:
                    keys = synonym_keywords[l]


            random.shuffle(keys)
            for k in keys:
                keywords += k+' '
            keywords += '<SEP> '
        keywords = keywords[:-1]

        # Generate prediction sentence
        hypo = generate_sentence(album_id, keywords, model)

        for keys in src:
            for k in keys:
                if k in hypo:
                    correct += 1
                total += 1
        print(correct/total)
        hypos[album_id] = hypo
        res.append(f'{album_id}\t {hypo}'+'\n')


    with open(f"res/emotion_experiments/{name}.txt", "w") as f:
        f.writelines(res)

    # Evaluate with automatic metrics
    evaluator.measure(f"res/emotion_experiments/{name}.txt", emos)





def main():
    global total
    global correct
    opt = opts.parse_opt()
    # load vocab
    # get model
    # opt.load_epoch = 3
    bart = BART(opt)
    name = "4_multi_emotion_noconcept_0.9"
    bart.load_model(f'models/emotion_experiments/{name}.pt')
    # generate(opt, model, SRC, TRG, opt.beam_size)
    generate(opt, bart, name=name)
    print(correct/total)

if __name__ == '__main__':
    main()
