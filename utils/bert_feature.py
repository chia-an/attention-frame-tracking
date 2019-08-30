from pathlib import Path
import json
import pickle
from tqdm import tqdm

import torch
from torch.utils.data import ConcatDataset, DataLoader

from pytorch_pretrained_bert import BertTokenizer, BertModel
from utils.parse_frames import FramesDataset, \
                         load_dictionaries as load_frames_dicts
from utils.parse_multiwoz import MultiwozDataset, \
                           load_dictionaries as load_mw_dicts


save_embed_dir = Path(__file__).parent / '../../data/bert_feature'


def text_iterator(bert_version):
    subsample = None
    frames_dicts = load_frames_dicts()
    mw_dicts = load_mw_dicts()

    frames_dataset = FramesDataset(
        dicts=frames_dicts,
        folds=range(1, 11),
        tokenizer=bert_version,
        subsample=subsample)
    mw_train_dataset = MultiwozDataset(
        dicts=mw_dicts,
        data_filename='train_mixed_multiwoz.json',
        tokenizer=bert_version,
        subsample=subsample)
    mw_valid_dataset = MultiwozDataset(
        dicts=mw_dicts,
        data_filename='valid_mixed_multiwoz.json',
        tokenizer=bert_version,
        subsample=subsample)

    for sample in DataLoader(ConcatDataset(
            [frames_dataset, mw_train_dataset, mw_valid_dataset])):
        fs, sent, asvs, frames, active_frame, new_frames = sample
        yield sent
        for a, s, v in asvs:
            yield v
        for frame in frames:
            for s, v in frame:
                yield v


def main():
    bert_version = 'bert-base-uncased'
    device = torch.device('cuda:1')

    bert_model = BertModel.from_pretrained(bert_version).to(device)
    bert_model.eval()

    embeddings = {}
    with torch.no_grad():
        for text in tqdm(text_iterator(bert_version),
                         bar_format='{l_bar}{r_bar}'):
            key = str(text.tolist())

            if key in embeddings:
                continue

            encoded_layers, pooled_output = bert_model(text.to(device))

            embedding = encoded_layers[-1].mean(dim=1).view(-1)

            embeddings[key] = embedding

    filename = 'bert-base-uncased-last-layer.pickle'
    with open(str(save_embed_dir / filename), 'wb') as f_embed:
        pickle.dump(embeddings, f_embed)


if __name__ == '__main__':
    main()
