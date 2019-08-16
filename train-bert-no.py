from pathlib import Path
from tqdm import tqdm
import json
import datetime
import random
import logging
import os
import argparse
from pprint import pformat
import pickle
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset

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


# Make things deterministic
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Path constants
result_save_dir = Path(__file__).parent / 'results'
bert_feature_dir = Path(__file__).parent / '../data/bert_feature'
log_dir = Path(__file__).parent / 'runs'
master_log_path = log_dir / '{}.log'.format(
    datetime.datetime.now().strftime('%m%d-%H%M%S'))

# Initialize logger
logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)

fh = logging.FileHandler(str(master_log_path))
fh.setLevel(logging.INFO)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)

action_options = ['scratch', 'pretrain', 'finetune']


def init_arg_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--model_config', default='', type=str,
        help='a json config file to initialize a model, '
             'cannot use with --checkpoint')
    parser.add_argument(
        '--checkpoint', default='', type=str,
        help='a PyTorch checkpoint, cannot use with --model_config')

    parser.add_argument(
        '--action', type=str, required=True,
        help='train or pre-train or fine-tune')

    # parser.add_argument(
    #     '--data', type=str, required=True,
    #     help='the dataset to be used: frames or multiwoz')

    parser.add_argument(
        '--device', type=str, required=True,
        help='torch device(s) to be used')
    parser.add_argument(
        '--n_epochs', type=int, required=True,
        help='number of training epochs')
    parser.add_argument(
        '--dry', default=True, type=bool,
        help='use dry run to test and debug, small dataset and not save')

    return parser


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


def train(dicts_config,
          model_config,
          train_datasets_config,
          valid_datasets_config,
          device_config,
          train_loader_kwargs={},
          optim_config={'': {}},
          main_valid_metric=('', ''),
          metrics_config={},
          n_epochs=10,
          run_id_prefix='',
          save=True,
          save_best_only=True):
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

    # Initialize run_id.
    run_id = '{}-{}'.format(
        run_id_prefix, datetime.datetime.now().strftime('%m%d-%H%M%S'))

    # Add handlers.
    logger.handlers = []

    os.mkdir(str(log_dir / run_id))
    fh = logging.FileHandler(str(log_dir / run_id / 'log'))
    fh.setLevel(logging.DEBUG)
    logger.addHandler(fh)

    fh = logging.FileHandler(str(master_log_path))
    fh.setLevel(logging.INFO)
    logger.addHandler(fh)

    ch = logging.StreamHandler()
    ch.setLevel(logging.INFO)
    logger.addHandler(ch)

    # Log everything.
    configs = {
        'dicts_config': dicts_config,
        'model_config': model_config,
        'train_datasets_config': train_datasets_config,
        'valid_datasets_config': valid_datasets_config,
        'device_config': device_config,
        'train_loader_kwargs': train_loader_kwargs,
        'optim_config': optim_config,
        'main_valid_metric': main_valid_metric,
        'metrics_config': metrics_config,
        'n_epochs': n_epochs,
        'run_id_prefix': run_id_prefix,
        'save': save,
        'save_best_only': save_best_only
    }
    logger.info('\nRun id = {}'.format(run_id))
    logger.info('\n===== Configs =====')
    logger.info(json.dumps(configs, indent=2, sort_keys=True))
    # logger.info(pformat(configs))
    # logger.info('dicts config = {}'.format(pformat(dicts_config)))
    # logger.info('model config = {}'.format(pformat(model_config)))
    # logger.info('optim config = {}'.format(pformat(optim_config)))
    # logger.info('train datasets config = {}'.format(
    #     pformat(train_datasets_config)))
    # logger.info('train loader kwargs = {}'.format(
    #     pformat(train_loader_kwargs)))
    # logger.info('valid datasets config = {}'.format(
    #     pformat(valid_datasets_config)))
    # logger.info('device config = {} (may not be used)'.format(
    #     pformat(device_config)))
    # logger.info('metrics config = {}'.format(pformat(metrics_config)))
    # logger.info('main valid metric = {}'.format(main_valid_metric))
    # logger.info('n_epochs = {}'.format(n_epochs))
    # logger.info('save = {}'.format(save))
    # logger.info('save best only = {}'.format(save_best_only))

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

            # NOTE: Dirty hack
            # model.optimizer = torch.optim.Adam(
            #     model.parameters(), lr=1e-4, weight_decay=1e-4)
        else:
            args['n_acts'] = len(act_to_index) + 1
            args['n_slots'] = len(slot_to_index) + 1
            args['n_tris'] = len(tri_to_index) + 1

            if name == 'Model':
                model, code = init_model(Model, **args)
            elif name == 'AttentionModel':
                # NOTE: remove this when the device is set during runtime.
                args['device'] = device

                if args['embed_type'].startswith('bert'):
                    bert_embedding_file = args['embed_type'] + '-' + \
                                          args.get('bert_embedding', '')

                    # if bert_embedding_file in ['last-layer.pickle']:

                    with open(str(bert_feature_dir / bert_embedding_file),
                              'rb') as f_embed:
                        bert_embedding = pickle.load(f_embed)
                        for text, feature in bert_embedding.items():
                            bert_embedding[text] = feature.to(device)
                        args['bert_embedding'] = bert_embedding

                    # else:
                    #     raise Exception('Unknown bert embedding file {}.'.format(
                    #         bert_embedding_file))

                model, code = init_model(AttentionModel, **args)
            else:
                raise Exception('Unknown model {}.'.format(name))

    # TODO: move this into model class itself.
    # Set device.
    model.set_device(device)
    # model.device = device
    # model = model.to(device)
    # model.optimizer.load_state_dict(model.optimizer.state_dict())

    # Initialize optimizer.
    assert len(optim_config) == 1, optim_config
    for name, args in optim_config.items():
        if name == '':
            pass
        elif name == 'adam':
            model.optimizer = torch.optim.Adam(model.parameters(), **args)
        else:
            raise Exception('Unknown optimizer {}.'.format(name))

    # Initialize datasets.
    train_datasets = []
    for train_dataset_config in train_datasets_config:
        assert len(train_dataset_config) == 1, train_dataset_config
        for name, args in train_dataset_config.items():
            if name == 'frames':
                train_dataset = FramesDataset(dicts=dicts, **args)
            elif name == 'multiwoz':
                train_dataset = MultiwozDataset(dicts=dicts, **args)
            else:
                raise Exception('Unknown dataset {}.'.format(name))
        train_datasets.append(train_dataset)
    train_loader = DataLoader(
        ConcatDataset(train_datasets), **train_loader_kwargs)

    # train_loader = DataLoader(train_dataset, **train_loader_kwargs)
    # train_loaders.append(train_loader)

    valid_loaders = {}
    for valid_name, valid_config in valid_datasets_config.items():
        valid_datasets = []
        for valid_dataset_config in valid_config:
            assert len(valid_dataset_config) == 1, valid_dataset_config
            for name, args in valid_dataset_config.items():
                if name == 'frames':
                    valid_dataset = FramesDataset(dicts=dicts, **args)
                elif name == 'multiwoz':
                    valid_dataset = MultiwozDataset(dicts=dicts, **args)
                else:
                    raise Exception('Unknown dataset {}.'.format(name))
            valid_datasets.append(valid_dataset)
        valid_loaders[valid_name] = DataLoader(ConcatDataset(valid_datasets))

    # Initialize metrics.
    metrics = {}
    for name, args in metrics_config.items():
        if name == 'accuracy':
            metrics[name] = accuracy
        else:
            raise Exception('Unknown metric {}.\n'.format(name))

    # Run
    return train_hook(
        run_id=run_id,
        model=model,
        code=code,
        n_epochs=n_epochs,
        train_loader=train_loader,
        valid_loaders=valid_loaders,
        metrics=metrics,
        main_valid_metric=main_valid_metric,
        save=save,
        save_best_only=save_best_only,
    )


def init_kwargs(dry_run=True,
                action=action_options[0],
                tokenizer='trigram'):

    logger.info('dry_run = {}'.format(dry_run))
    logger.info('action = {}'.format(action))

    if dry_run:
        subsample = 10
        save = False
    else:
        subsample = None
        save = False

    # Initialize configs.
    if action in ['scratch']:
        dicts_config = {'frames': {}}
    elif action in ['pretrain', 'finetune']:
        dicts_config = {'combined': {}}
    else:
        raise Exception('Fail to set dicts_config. ' \
                        'Unknown action {}.'.format(action))

    if action in ['scratch', 'pretrain']:
        model_config = {'AttentionModel': {
            'embed_type': tokenizer,
            'bert_embedding': 'last-layer.pickle',
            'embed_dim_act': 128,
            'embed_dim_slot': 128,
            'embed_dim_text': 128,
            'embed_dim_tri': 128,
            'embed_dim': 128,
            'asv_rnn_hidden_size': 128,
            'frame_rnn_hidden_size': 128,
            'attention_type': 'simple',
            'asv_with_utterance': False,
            'asv_rnn_hidden': True,
            'asv_rnn_output': False,
        }}
    elif action in ['finetune']:
        model_config = {'checkpoint': ''}
    else:
        raise Exception('Fail to set model_config. ' \
                        'Unknown action {}.'.format(action))

    if action in ['scratch', 'pretrain']:
        optim_config = {'adam': {'lr': 1e-4, 'weight_decay': 1e-4}}
    elif action in ['finetune']:
        # optim_config = {'': {}}
        optim_config = {'adam': {'lr': 1e-4, 'weight_decay': 1e-4}}
    else:
        raise Exception('Fail to set optim_config.' \
                        'Unknown action {}.'.format(action))

    if action in ['scratch', 'finetune']:
        train_datasets_config = [{'frames': {
            'folds': list(range(1, 9)),
            'subsample': subsample,
            'tokenizer': tokenizer,
        }}]

        valid_datasets_config = {}
        valid_datasets_config['frames-all'] = [{'frames': {
            'folds': [9],
            'subsample': subsample,
            'tokenizer': tokenizer,
        }}]
        valid_datasets_config['frames-user-slot'] = [{'frames': {
            'folds': [9],
            'subsample': subsample,
            'slot_based': True,
            'user_only': True,
            'tokenizer': tokenizer,
        }}]
        valid_datasets_config['test-frames-user-slot'] = [{'frames': {
            'folds': [10],
            'subsample': subsample,
            'slot_based': True,
            'user_only': True,
            'tokenizer': tokenizer,
        }}]

        main_valid_metric = ('frames-user-slot', 'accuracy')
    elif action in ['pretrain']:
        train_datasets_config = [{'multiwoz': {
            'data_filename': 'train_mixed_multiwoz.json',
            'subsample': subsample,
            'tokenizer': tokenizer,
        }}]

        valid_datasets_config = {}
        valid_datasets_config['multiwoz'] = [{'multiwoz': {
            'data_filename': 'valid_mixed_multiwoz.json',
            'subsample': subsample,
            'tokenizer': tokenizer,
        }}]

        main_valid_metric = ('multiwoz', 'accuracy')
    else:
        raise Exception('Fail to set dataset configs. ' \
                        'Unknown action {}.'.format(action))

    train_loader_kwargs = {}
    if action in ['scratch', 'pretrain', 'finetune']:
        train_loader_kwargs['shuffle'] = True

    device_config = {'cuda:0': {}}

    metrics_config = {'accuracy': {}}

    if action in ['scratch', 'finetune']:
        n_epochs = 10
        # n_epochs = 20
    elif action in ['pretrain']:
        n_epochs = 20
        # n_epochs = 50
    else:
        raise Exception('Fail to set n_epochs. ' \
                        'Unknown action {}.'.format(action))

    run_id_prefix = action

    save_best_only = True

    return {
        'dicts_config': dicts_config,
        'model_config': model_config,
        'optim_config': optim_config,
        'train_datasets_config': train_datasets_config,
        'train_loader_kwargs': train_loader_kwargs,
        'valid_datasets_config': valid_datasets_config,
        'device_config': device_config,
        'metrics_config': metrics_config,
        'main_valid_metric': main_valid_metric,
        'n_epochs': n_epochs,
        'run_id_prefix': run_id_prefix,
        'save': save,
        'save_best_only': save_best_only
    }


def set_pretrain_datasets(datasets):
    def f(config):
        for i, (train, valid) in enumerate(datasets):
            if i != 0:
                config['train_datasets_config'].append(
                    deepcopy(config['train_datasets_config'][0]))
            config['train_datasets_config'] \
                  [-1] \
                  ['multiwoz'] \
                  ['data_filename'] = train
            if i != 0:
                config['valid_datasets_config']['multiwoz'].append(
                    deepcopy(config['valid_datasets_config']['multiwoz'][0]))
            config['valid_datasets_config'] \
                  ['multiwoz'] \
                  [-1] \
                  ['multiwoz'] \
                  ['data_filename'] = valid
        return config
    return f


def set_attention(attention_type):
    def f(config):
        config['model_config'] \
              ['AttentionModel'] \
              ['attention_type'] = attention_type
        # if attention_type == 'no':
        #     config['model_config']['AttentionModel'].update({
        #         'embed_dim_act': 256,
        #         'embed_dim_slot': 256,
        #         'embed_dim_text': 256,
        #         'embed_dim_tri': 256,
        #         'embed_dim': 256,
        #         'asv_rnn_hidden_size': 256,
        #         'frame_rnn_hidden_size': 256,
        #     })
        return config
    return f


def main():
    n_runs = 5

    dry_run = False
    # tokenizer = 'trigram'
    tokenizer = 'bert-base-uncased'
    # attention_type = 'no'
    attention_type = 'simple'

    print('dry run = {}'.format(dry_run))
    print('tokenizer = {}'.format(tokenizer))
    print('attention = {}'.format(attention_type))

    input()

    import time
    # time.sleep(1 * 60 * 60)
    # time.sleep(5)

    model_configs_args = [
        {
            # 'embed_type': tokenizer,
            'bert_embedding': 'last-layer.pickle',
            # 'attention_type': 'simple',
            # 'attention_type': 'no',
            # 'attention_type': 'content',
            # 'asv_with_utterance': False,
            # 'asv_rnn_hidden': True,
            # 'asv_rnn_output': False,
        },
    ]

    pretrain_1 = ('train_mixed_multiwoz.json', 'valid_mixed_multiwoz.json')
    pretrain_2 = ('train_mixed_hotel_restaurant_multiwoz.json',
                  'valid_mixed_hotel_restaurant_multiwoz.json')
    pretrain_3 = ('train_mixed_hotel_transport_multiwoz.json',
                  'valid_mixed_hotel_transport_multiwoz.json')


    pretrain_datasets = [
        [pretrain_1,],
        [pretrain_2,],
        [pretrain_1, pretrain_3],
        [pretrain_1, pretrain_2, pretrain_3],
        [pretrain_1, pretrain_2],
        [pretrain_3,],
    ]


    for model_config_args in model_configs_args:
        break

        # From scratch
        for run in range(n_runs):
            logger.info('Run {} / {}:'.format(run + 1, n_runs))

            kwargs = init_kwargs(
                dry_run=dry_run,
                action=action_options[0],
                tokenizer=tokenizer)

            kwargs['n_epochs'] = 1
            kwargs['device_config'] = {'cuda:0': {}}

            kwargs = set_attention(attention_type)(kwargs)

            random.seed(run)
            torch.manual_seed(run)

            train(**kwargs)

    for datasets in pretrain_datasets:
        # Transfer learning
        if True:
            kwargs = init_kwargs(
                dry_run=dry_run,
                action=action_options[1],
                tokenizer=tokenizer)

            kwargs['n_epochs'] = 1
            kwargs['device_config'] = {'cuda:0': {}}
            kwargs['save'] = True

            kwargs = set_pretrain_datasets(datasets)(kwargs)
            kwargs = set_attention(attention_type)(kwargs)

            random.seed(0)
            torch.manual_seed(0)

            history = train(**kwargs)
            best_checkpoint = history['best_checkpoint']

        for run in range(n_runs):
            logger.info('Run {} / {}:'.format(run + 1, n_runs))

            kwargs = init_kwargs(
                dry_run=dry_run,
                action=action_options[2],
                tokenizer=tokenizer)

            kwargs['n_epochs'] = 1
            kwargs['device_config'] = {'cuda:0': {}}
            kwargs['model_config']['checkpoint'] = best_checkpoint

            random.seed(run)
            torch.manual_seed(run)

            train(**kwargs)


if __name__ == '__main__':
    main()

