import pandas as pd
from absl import app
from absl import flags

from sklearn.metrics import f1_score

import json
import numpy as np
import os
import random
import torch
import transformers

from datasets import load_dataset
from transformers.models.bert.configuration_bert import BertConfig
from transformers import AutoModel, AutoTokenizer, BertModel, DistilBertModel, DistilBertConfig
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_integer("batch_size", 16, "")
flags.DEFINE_integer("max_epochs", 15, "")
flags.DEFINE_integer("num_classes", 2, "")
flags.DEFINE_integer("seed", 1, "")

flags.DEFINE_string("results_file", "results.json", "")
flags.DEFINE_string("model", "bert-base-uncased", "")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

emotion_mapping = {
    "anger" : "anger",
    "disgust" : "disgust",
    "joy" : "joy",
    "sadness" : "sadness",
    "fear" : "fear",
    "nervousness" : "anticipation",
    "desire" : "anticipation",
    "surprise" : "surprise",
    "admiration" : "trust",
}
inverse_emotion_mapping = {
    "anger" : ["anger"],
    "disgust" : ["disgust"],
    "joy" : ["joy"],
    "sadness" : ["sadness"],
    "fear" : ["fear"],
    "anticipation" : ["nervousness", "desire"],
    "surprise" : ["surprise"],
    "trust" : ["admiration"],
}

covidemo_emotions = ["anger", "disgust", "joy", "sadness", "fear", "anticipation", "surprise", "trust"]

# Evaluate the model on the test dataloader
def evaluate(model, test_dataloader):
    full_predictions = []
    true_labels = []

    model.eval()

    for elem in test_dataloader:
        x = {key: elem[key].to(device)
             for key in elem if key not in ['text', 'idx']}
        logits = model(x)
        results = torch.max(logits, axis=1).indices

        full_predictions = full_predictions + \
            list(results.cpu().detach().numpy())
        true_labels = true_labels + list(torch.max(elem['labels'], axis=1).indices.cpu().detach().numpy())

    model.train()

    return f1_score(true_labels, full_predictions, average='macro')

# Tokenized emotion dataset.
class EmotionsDataset(torch.utils.data.Dataset):
    def __init__(self, text_list, labels_list, tokenizer, num_classes=28):
        self.text_list = text_list
        self.labels_list = labels_list
        self.tokenizer = tokenizer
        self.num_classes = num_classes

    def __getitem__(self, idx):

        tok = self.tokenizer(
            self.text_list[idx], padding='max_length', max_length=128, truncation=True)

        new_labels = [0] * self.num_classes
        new_labels[self.labels_list[idx]] = 1
        item = {key: torch.tensor(tok[key]) for key in tok}
        item['labels'] = torch.tensor(new_labels, dtype=torch.float)

        return item

    def __len__(self):
        return len(self.labels_list)

# Pytorch model
class EmotionModel(torch.nn.Module):
    def __init__(self, ckpt_file, num_classes=2):
        super().__init__()
        self.bert_model = AutoModel.from_pretrained(FLAGS.model)
        self.dropout = torch.nn.Dropout(0.3)
        self.fc = torch.nn.Linear(1024, num_classes)

    def forward(self, x):
        out = self.bert_model(
            x['input_ids'], x['attention_mask']).last_hidden_state[:, 0, :]
        out = torch.squeeze(out)
        out = self.dropout(out)
        out = self.fc(out)
        return out

# Return dataloader from dataset.
def give_dataloader(path, batch_size, emotion, tokenizer):

    df = pd.read_csv(path)

    texts = df['text'].tolist()
    labels = df[emotion].tolist()

    dataset_goemotion_train = EmotionsDataset(texts, labels, tokenizer, FLAGS.num_classes)
    train_dataloader = torch.utils.data.DataLoader(
        dataset_goemotion_train, batch_size=batch_size, shuffle=False)

    return train_dataloader

SAMPLE_EQUAL_NEGATIVES = True

def main(argv):

    for repetition in range(5):
        tokenizer = AutoTokenizer.from_pretrained(
            FLAGS.model, use_fast=True)
        with open("emotions.txt") as f:
            goemotions_emotions = f.read().split('\n')
        id_to_emotion = {}
        for i in range(len(goemotions_emotions)):
            id_to_emotion[i] = goemotions_emotions[i]
        dataset_goemotions = load_dataset("go_emotions", "simplified")
        texts = []
        labels = []

        for emotion in covidemo_emotions:
            if SAMPLE_EQUAL_NEGATIVES:
                texts_positives = []
                texts_negatives = []
            else:
                texts = []
                labels = []
            for elem in dataset_goemotions['train']:
                if SAMPLE_EQUAL_NEGATIVES:
                    found_positive = False
                    no_neutral = True
                    for emotion_id in elem['labels']:
                        if 27 in elem['labels']:
                            no_neutral = False
                            break
                        if id_to_emotion[emotion_id] in inverse_emotion_mapping[emotion]:
                            found_positive = True
                    if no_neutral:
                        if found_positive:
                            texts_positives.append(elem['text'])
                        else:
                            texts_negatives.append(elem['text'])
                else:
                    found_positive = False
                    no_neutral = True
                    for emotion_id in elem['labels']:
                        if 27 in elem['labels']:
                            no_neutral = False
                            break
                        if id_to_emotion[emotion_id] in inverse_emotion_mapping[emotion]:
                            found_positive = True
                    if no_neutral:
                        texts.append(elem['text'])
                        labels.append(1 if found_positive else 0)

            if SAMPLE_EQUAL_NEGATIVES:
                texts = texts_positives + random.sample(texts_negatives, len(texts_positives))
                labels = [1] * len(texts_positives) + [0] * len(texts_positives)

            train_dataset = EmotionsDataset(texts, labels, tokenizer, FLAGS.num_classes)
            train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=FLAGS.batch_size, shuffle=True)

            validation_dataloader = give_dataloader(
                'binary/' + emotion + '_dev.csv', FLAGS.batch_size, emotion, tokenizer)
            test_dataloader = give_dataloader(
                'binary/' + emotion + '_test.csv', FLAGS.batch_size, emotion, tokenizer)


            model = EmotionModel(FLAGS.model, num_classes=FLAGS.num_classes)
            model.to(device)

            optimizer = torch.optim.Adam(model.parameters(), lr=5e-05)
            loss_fn = torch.nn.CrossEntropyLoss()

            best_f1 = 0
            # Training loop.
            for epoch in range(FLAGS.max_epochs):
                for data in tqdm(train_dataloader):
                    cuda_tensors = {key: data[key].to(
                        device) for key in data if key not in ['text', 'idx']}
                        
                    optimizer.zero_grad()
                    logits = model(cuda_tensors)
                    loss = loss_fn(logits, cuda_tensors['labels'])
                    loss.backward()
                    optimizer.step()

                f1_validation = evaluate(
                    model, validation_dataloader)
                f1_test = evaluate(model, test_dataloader)

                if f1_validation >= best_f1:
                    best_f1 = f1_validation
                    best_idx = epoch
                    corresponding_test = f1_test

            experiment_id = str(emotion) + '-' + FLAGS.model
            if not os.path.exists(FLAGS.results_file):
                d = {}
            else:
                with open(FLAGS.results_file) as f:
                    d = json.load(f)
            if experiment_id not in d:
                d[experiment_id] = {}

            d[experiment_id][repetition] = corresponding_test

            with open(FLAGS.results_file, 'w') as f:
                json.dump(d, f)

if __name__ == "__main__":
    app.run(main)
