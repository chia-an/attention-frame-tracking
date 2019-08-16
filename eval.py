from pathlib import Path
from tqdm import tqdm
import json
import datetime
import random
import logging
import os
import argparse
from pprint import pformat

import torch
from torch.utils.data import Dataset, DataLoader, Subset

from utils.dictionary import merge
from utils.metrics import accuracy
from utils.pipeline import train_hook, eval_hook, init_model, load_checkpoint
from utils.parse_frames import load_dictionaries as frames_load_dicts, \
                               data_dir as frames_data_dir, \
                               preds_to_dial_json, \
                               FramesDataset
from utils.parse_multiwoz import load_dictionaries as multiwoz_load_dicts, \
                                 MultiwozDataset

from model.maluuba import Model
from model.attention import Model as AttentionModel


# Path constants
result_save_dir = Path(__file__).parent / 'results'
log_dir = Path(__file__).parent / 'runs'

# Initialize logger
logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)


def load_combined_dicts():
    f_word_to_index, f_index_to_word, \
    f_tri_to_index, f_index_to_tri, \
    f_act_to_index, f_index_to_act, \
    f_slot_to_index, f_index_to_slot = frames_load_dicts()

    m_word_to_index, m_index_to_word, \
    m_tri_to_index, m_index_to_tri, \
    m_act_to_index, m_index_to_act, \
    m_slot_to_index, m_index_to_slot = multiwoz_load_dicts()

    word_to_index, index_to_word = merge(
        f_word_to_index, f_index_to_word,
        m_word_to_index, m_index_to_word)
    tri_to_index, index_to_tri = merge(
        f_tri_to_index, f_index_to_tri,
        m_tri_to_index, m_index_to_tri)
    act_to_index, index_to_act = merge(
        f_act_to_index, f_index_to_act,
        m_act_to_index, m_index_to_act)
    slot_to_index, index_to_slot = merge(
        f_slot_to_index, f_index_to_slot,
        m_slot_to_index, m_index_to_slot)

    return word_to_index, index_to_word, \
           tri_to_index, index_to_tri, \
           act_to_index, index_to_act, \
           slot_to_index, index_to_slot


def eval(dicts_config,
         model_config,
         valid_datasets_config,
         device_config,
         metrics_config={}):
    """
    X_config [dict]: each (key, value) corresponds to a (class_name, args)
    valid_datasets_config [dict]: keys are names of valid set and 
                                  values are configs.
    """
    # # Make things deterministic
    # random.seed(0)
    # torch.manual_seed(0)
    # torch.backends.cudnn.benchmark = False
    # torch.backends.cudnn.deterministic = True

    # # Initialize run_id.
    # run_id = '{}-{}'.format(
    #     run_id_prefix, datetime.datetime.now().strftime('%m%d-%H%M%S'))

    # Add handlers.
    logger.handlers = []

    # os.mkdir(str(log_dir / run_id))
    # fh = logging.FileHandler(str(log_dir / run_id / 'log'))
    # fh.setLevel(logging.DEBUG)
    # logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    # Log everything.
    logger.info('\ndicts config = {}'.format(pformat(dicts_config)))
    logger.info('model config = {}'.format(pformat(model_config)))
    logger.info('valid datasets config = {}'.format(
        pformat(valid_datasets_config)))
    logger.info('device config = {} (may not be used)'.format(
        pformat(device_config)))
    logger.info('metrics config = {}'.format(pformat(metrics_config)))

    # Load dicts for model_args and datasets.
    assert len(dicts_config) == 1, dicts_config
    for name, args in dicts_config.items():
        if name == 'frames':
            dicts = frames_load_dicts()
        elif name == 'combined':
            dicts = load_combined_dicts()
        else:
            raise Exception('Unknown dicts name {}.'.format(name))

    word_to_index, index_to_word, \
    tri_to_index, index_to_tri, \
    act_to_index, index_to_act, \
    slot_to_index, index_to_slot = dicts

    # Initialize torch device.
    # TODO: set model.device.
    assert len(device_config) == 1, device_config
    for name, args in device_config.items():
        device = torch.device(name)

    # Initialize/load model.
    assert len(model_config) == 1, model_config
    for name, args in model_config.items():
        if name == 'checkpoint':
            # args is checkpoint name.
            model, code = load_checkpoint(args)
        else:
            args['n_acts'] = len(act_to_index) + 1
            args['n_slots'] = len(slot_to_index) + 1
            args['n_tris'] = len(tri_to_index) + 1

            # NOTE: remove this when the device is set during runtime.
            args['device'] = device

            if name == 'Model':
                model, code = init_model(Model, **args)
            elif name == 'AttentionModel':
                model, code = init_model(AttentionModel, **args)
            else:
                raise Exception('Unknown model {}.'.format(name))

    # Set deivce.
    model.device = device
    model = model.to(device)
    model.optimizer.load_state_dict(model.optimizer.state_dict())

    # Initialize datasets.
    valid_loaders = {}
    for valid_name, valid_config in valid_datasets_config.items():
        assert len(valid_config) == 1, valid_config
        for name, args in valid_config.items():
            if name == 'frames':
                valid_dataset = FramesDataset(dicts=dicts, **args)
            elif name == 'multiwoz':
                valid_dataset = MultiwozDataset(dicts=dicts, **args)
            else:
                raise Exception('Unknown dataset {}.'.format(name))
        valid_loaders[valid_name] = DataLoader(valid_dataset)

    # Initialize metrics.
    metrics = {}
    for name, args in metrics_config.items():
        if name == 'accuracy':
            metrics[name] = accuracy
        else:
            raise Exception('Unknown metric {}.\n'.format(name))

    # Run
    all_preds = []
    for valid_name, valid_config in valid_datasets_config.items():
        if 'frames' not in valid_config:
            continue
        output = eval_hook(
            model=model,
            data_loader=valid_loaders[valid_name],
            metrics=metrics,
        )
        all_preds += output

    all_dials = []
    for valid_name, valid_config in valid_datasets_config.items():
        if 'frames' not in valid_config:
            continue
        for fold in valid_config['frames']['folds']:
            with open(str(frames_data_dir / '{}.json'.format(fold)),
                      'r') as f_data:
                dials = json.load(f_data)
                all_dials += dials

    # all_preds = [output[0] for output in valid_output + test_output]
    all_preds = [output[0] for output in all_preds]

    gen_dials = preds_to_dial_json(all_dials, all_preds)

    model_filename = model_config['checkpoint']
    gen_filename = 'gen_{}.json'.format(model_filename.rsplit('.', 1)[0])
    with open(str(result_save_dir / gen_filename), 'w') as f_gen:
        json.dump(gen_dials, f_gen)


def main():
    subsample = None


    # Initialize configs.
    # dicts_config = {'frames': {}}
    dicts_config = {'combined': {}}

    # model_config = {'AttentionModel': {
    #     'embed_dim_act': 128,
    #     'embed_dim_slot': 128,
    #     'embed_dim_text': 128,
    #     'embed_dim_tri': 128,
    #     'embed_dim': 128,
    #     'asv_rnn_hidden_size': 128,
    #     'frame_rnn_hidden_size': 128,
    # }}
    model_config = {'checkpoint': 'finetune-0528-181443-best.pt'}

    valid_datasets_config = {}
    # valid_datasets_config['frames-user-slot'] = {'frames': {
    #     'folds': [9],
    #     'subsample': subsample,
    #     'slot_based': True,
    #     'user_only': True,
    # }}
    valid_datasets_config['test-frames-user-slot'] = {'frames': {
        'folds': [9, 10],
        'subsample': subsample,
        # 'slot_based': True,
        # 'user_only': True,
    }}

    device_config = {'cuda:1': {}}

    metrics_config = {'accuracy': {}}

    # run_id_prefix = 'scratch-eval'


    eval(
        dicts_config=dicts_config,
        model_config=model_config,
        valid_datasets_config=valid_datasets_config,
        device_config=device_config,
        metrics_config=metrics_config,
    )

if __name__ == '__main__':
    main()

