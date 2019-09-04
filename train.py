from pathlib import Path
import json
import datetime
import random
import logging
import os
import pickle
from copy import deepcopy

import torch
from torch.utils.data import Dataset, DataLoader, Subset, ConcatDataset

from utils.dictionary import merge
from utils.metrics import accuracy
from utils.pipeline import train_hook, eval_hook, init_model, load_checkpoint
from utils.parse_frames import load_dictionaries as frames_load_dicts, \
                               preds_to_dial_json, \
                               FramesDataset
from utils.parse_multiwoz import load_dictionaries as multiwoz_load_dicts, \
                                 MultiwozDataset

from model.attention import Model as AttentionModel


# Make things deterministic
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Path constants
result_save_dir = Path(__file__).parent / 'results'
bert_feature_dir = Path(__file__).parent / '../data/bert_feature'
log_dir = Path(__file__).parent / 'runs'
master_log = '{}.log'.format(datetime.datetime.now().strftime('%m%d-%H%M%S'))
master_log_path = log_dir / master_log
print('master log = {}'.format(master_log))

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
    Initialise everything for training and call trainhook.

    The config for a class is represented as {CLASSNAME: KWARGS}, where KWARGS
    is a dictionary.

    Most configs in the argument is a single config for one class.
    train_datasets_config is a list of configs for train datasets.
    valid_datasets_config [dict]: keys are names of valid set and
                                  values are configs.
    """
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
    assert len(device_config) == 1, device_config
    for name, args in device_config.items():
        device = torch.device(name)

    # Initialize/load model.
    assert len(model_config) == 1, model_config
    for name, args in model_config.items():
        if name == 'checkpoint':
            # args is checkpoint name
            model, code = load_checkpoint(args)
        else:
            args['n_acts'] = len(act_to_index) + 1
            args['n_slots'] = len(slot_to_index) + 1
            args['n_tris'] = len(tri_to_index) + 1

            if name == 'AttentionModel':
                if args['embed_type'].startswith('bert') and \
                   args['bert_embedding'] is not None:
                    bert_embedding_file = args['embed_type'] + '-' + \
                                          args.get('bert_embedding', '')

                    with open(str(bert_feature_dir / bert_embedding_file),
                              'rb') as f_embed:
                        bert_embedding = pickle.load(f_embed)
                        for text, feature in bert_embedding.items():
                            bert_embedding[text] = feature.to(device)
                        args['bert_embedding'] = bert_embedding

                model, code = init_model(AttentionModel, **args)
            else:
                raise Exception('Unknown model {}.'.format(name))

    # Set device.
    model.set_device(device)

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
    """
    Set up default configs.
    """
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
    if action in ['scratch', 'finetune']:
        train_loader_kwargs['shuffle'] = True

    device_config = {'cuda:0': {}}

    metrics_config = {'accuracy': {}}

    if action in ['scratch', 'finetune']:
        n_epochs = 10
    elif action in ['pretrain']:
        n_epochs = 20
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


#####
# Functions that modify the config
#####

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
        return config
    return f


def set_input_dim(dim):
    def f(config):
        config['model_config'] \
              ['AttentionModel'].update({
                'embed_dim_act': dim,
                'embed_dim_slot': dim,
                'embed_dim_text': dim,
                'embed_dim_tri': dim,
              })
        return config
    return f


def set_embed_dim(dim):
    def f(config):
        config['model_config'] \
              ['AttentionModel'] \
              ['embed_dim'] = dim
        return config
    return f


def set_hidden_dim(dim):
    def f(config):
        config['model_config'] \
              ['AttentionModel'].update({
                'asv_rnn_hidden_size': dim,
                'frame_rnn_hidden_size': dim,
              })
        return config
    return f


def set_bert_embedding(embedding):
    def f(config):
        config['model_config'] \
              ['AttentionModel'] \
              ['bert_embedding'] = embedding
        return config
    return f


def main():
    n_runs = 5

    dry_run = False
    dry_run = True
    device = 'cuda:0'
    device = 'cpu'
    # tokenizer = 'trigram'
    tokenizer = 'bert-base-uncased'
    print('dry run = {}'.format(dry_run))
    print('device = {}'.format(device))
    print('tokenizer = {}'.format(tokenizer))

    input()

    #####
    # Train from scratch
    #####
    for run in range(n_runs):
        logger.info('Run {} / {}:'.format(run + 1, n_runs))

        kwargs = init_kwargs(
            dry_run=dry_run,
            action=action_options[0],
            tokenizer=tokenizer)

        kwargs['device_config'] = {device: {}}

        kwargs = set_bert_embedding(None)(kwargs)
        kwargs = set_attention('simple')(kwargs)
        # kwargs = set_input_dim(256)(kwargs)
        # kwargs = set_hidden_dim(256)(kwargs)
        # kwargs = set_embed_dim(256)(kwargs)

        random.seed(run)
        torch.manual_seed(run)

        train(**kwargs)

    #####
    # Transfer learning: pre-train
    #####
    kwargs = init_kwargs(
        dry_run=dry_run,
        action=action_options[1],
        tokenizer=tokenizer)

    kwargs['device_config'] = {device: {}}
    kwargs['save'] = True
    # kwargs = set_pretrain_datasets(datasets)(kwargs)
    # kwargs = set_attention(attention_type)(kwargs)

    random.seed(0)
    torch.manual_seed(0)

    history = train(**kwargs)
    best_checkpoint = history['best_checkpoint']

    #####
    # Transfer learning: fine-tune
    #####
    # Uncomment to start from other pre-trained model
    # best_checkpoint = 'pretrain-0604-203951-best.pt'

    for run in range(n_runs):
        logger.info('Run {} / {}:'.format(run + 1, n_runs))

        kwargs = init_kwargs(
            dry_run=dry_run,
            action=action_options[2],
            tokenizer=tokenizer)

        kwargs['device_config'] = {device: {}}
        kwargs['model_config']['checkpoint'] = best_checkpoint

        random.seed(run)
        torch.manual_seed(run)

        train(**kwargs)


if __name__ == '__main__':
    main()

