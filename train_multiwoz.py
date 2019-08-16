from pathlib import Path
from tqdm import tqdm
import json
import datetime
import random
import logging
import os

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

# Make things deterministic
random.seed(0)
torch.manual_seed(0)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

# Initialize logger
logger = logging.getLogger('logger')
logger.setLevel(logging.DEBUG)

run_id = 'multiwoz-{}'.format(datetime.datetime.now().strftime('%m%d-%H%M%S'))
os.mkdir(str(log_dir / run_id))
fh = logging.FileHandler(str(log_dir / run_id / 'log'))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def train(model, code, dicts):
    print('Training')

    n_epochs = 50
    subsample = None
    pad_frame = False
    tokenizer = 'bert-base-uncased'

    train_dataset = MultiwozDataset(
        dicts=dicts,
        data_filename='train_mixed_multiwoz.json',
        subsample=subsample,
        pad_frame=pad_frame,
        tokenizer=tokenizer)
    valid_dataset = MultiwozDataset(
        dicts=dicts,
        data_filename='valid_mixed_multiwoz.json',
        subsample=subsample,
        pad_frame=pad_frame,
        tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset)
    valid_loader = DataLoader(valid_dataset)

    train_hook(
        run_id=run_id,
        model=model,
        code=code,
        n_epochs=n_epochs,
        train_loader=train_loader,
        valid_loaders={'multiwoz': valid_loader},
        metrics={'accuracy': accuracy},
        model_filename_prefix='multiwoz',
        train_id_prefix='multiwoz',
        # save=False,
    )


def fine_tune(model, code, dicts):
    print('Fine tune')

    n_epochs = 10
    subsample = None
    pad_frame = False
    tokenizer = 'bert-base-uncased'

    train_dataset = FramesDataset(
        dicts=dicts,
        folds=range(1, 9),
        # folds=range(7, 9),
        subsample=subsample,
        pad_frame=pad_frame,
        tokenizer=tokenizer)
    valid_dataset = FramesDataset(
        dicts=dicts,
        folds=[9],
        subsample=subsample,
        pad_frame=pad_frame,
        tokenizer=tokenizer)
    true_valid_dataset = FramesDataset(
        dicts=dicts,
        folds=[9],
        subsample=subsample,
        slot_based=True,
        user_only=True,
        pad_frame=pad_frame,
        tokenizer=tokenizer)
    true_test_dataset = FramesDataset(
        dicts=dicts,
        folds=[10],
        subsample=subsample,
        slot_based=True,
        user_only=True,
        pad_frame=pad_frame,
        tokenizer=tokenizer)

    train_loader = DataLoader(train_dataset)
    valid_loader = DataLoader(valid_dataset)
    true_valid_loader = DataLoader(true_valid_dataset)
    true_test_loader = DataLoader(true_test_dataset)

    train_hook(
        run_id=run_id,
        model=model,
        code=code,
        n_epochs=n_epochs,
        train_loader=train_loader,
        valid_loaders={'frames': valid_loader,
                       'user-slot-frames': true_valid_loader,
                       'test-frames-user-slot': true_test_loader},
        metrics={'accuracy': accuracy},
        model_filename_prefix='multiwoz',
        train_id_prefix='multiwoz',
        save=False,
    )


def test(model_filename, dicts):
    gen = False

    # Load model.
    model, code = load_checkpoint(model_filename)

    # Load datasets.
    pad_frame = True

    if gen:
        valid_dataset = FramesDataset(
            dicts=dicts, folds=[9], pad_frame=pad_frame)
        test_dataset = FramesDataset(
            dicts=dicts, folds=[10], pad_frame=pad_frame)

    true_valid_dataset = FramesDataset(
        dicts=dicts,
        folds=[9],
        slot_based=True,
        user_only=True,
        pad_frame=pad_frame)
    true_test_dataset = FramesDataset(
        dicts=dicts,
        folds=[10],
        slot_based=True,
        user_only=True,
        pad_frame=pad_frame)

    if gen:
        valid_loader = DataLoader(valid_dataset)
        test_loader = DataLoader(test_dataset)

    true_valid_loader = DataLoader(true_valid_dataset)
    true_test_loader = DataLoader(true_test_dataset)

    # Run
    eval_hook(
        model=model,
        data_loader=true_valid_loader,
        metrics={'user-slot-acc': accuracy}
    )
    if gen:
        valid_output = eval_hook(
            model=model,
            data_loader=valid_loader,
            metrics={'accuracy': accuracy})
    eval_hook(
        model=model,
        data_loader=true_test_loader,
        metrics={'user-slot-acc': accuracy}
    )
    if gen:
        test_output = eval_hook(
            model=model,
            data_loader=test_loader,
            metrics={'accuracy': accuracy})

    if not gen:
        return

    # Generate json prediction (for eval metrics)
    all_dials = []
    for fold in range(9, 11):
        with open(str(frames_data_dir / '{}.json'.format(fold)),
                  'r') as f_data:
            dials = json.load(f_data)
            all_dials += dials

    all_preds = [output[0] for output in valid_output + test_output]

    gen_dials = preds_to_dial_json(all_dials, all_preds)
    gen_filename = 'gen_{}.json'.format(model_filename.rsplit('.', 1)[0])
    with open(str(result_save_dir / gen_filename), 'w') as f_gen:
        json.dump(gen_dials, f_gen)


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


def main():
    # Load data, dicts.
    dicts = load_combined_dicts()
    word_to_index, index_to_word, \
    tri_to_index, index_to_tri, \
    act_to_index, index_to_act, \
    slot_to_index, index_to_slot = dicts

    device = torch.device('cuda:1')

    # Init model.
    static_kwargs = {
        'n_acts': len(act_to_index) + 1,
        'n_slots': len(slot_to_index) + 1,
        'n_tris': len(tri_to_index) + 1,
        'embed_type': 'bert-base-uncased',
        'embed_dim_act': 128,
        'embed_dim_slot': 128,
        'embed_dim_text': 128,
        'embed_dim_tri': 128,
        'embed_dim': 128,
        'asv_rnn_hidden_size': 128,
        'frame_rnn_hidden_size': 128,
    }
    other_kwargs = {
        'device': device,
    }
    new_model = False

    if new_model:
        logger.info('===== Model args =====')
        logger.info(json.dumps(static_kwargs))
        logger.info('')

        model, code = init_model(
            AttentionModel, **static_kwargs, **other_kwargs)
    else:
        checkpoint = 'multiwoz-0514-084431-epoch-46.pt'
        # checkpoint = 'multiwoz-0508-181024-epoch-40.pt'
        # checkpoint = 'multiwoz-0502-165055-epoch-18.pt'
        # checkpoint = 'multiwoz-0502-081141-epoch-47.pt'
        # checkpoint = 'multiwoz-0426-084045-epoch-99.pt'
        # checkpoint = 'multiwoz-0426-084045-epoch-10.pt'
        logger.info('===== Load model =====')
        logger.info('Checkpoint = {}'.format(checkpoint))

        model, code = load_checkpoint(checkpoint)

    # model = Model(**kwargs)
    # model = torch.load(str(model_save_dir / 'multiwoz-0329-180233.pt'))
    # model = torch.load(str(model_save_dir / 'multiwoz-0402-183015.pt'))
    # model = torch.load(str(model_save_dir / 'multiwoz-0403-163506.pt'))

    # model = AttentionModel(**kwargs)
    # model = torch.load(str(model_save_dir / 'multiwoz-epoch-26-0418-053929.pt'))

    # Run
    # train(model=model, code=code, dicts=dicts)
    fine_tune(model=model, code=code, dicts=dicts)


if __name__ == '__main__':
    logger.info('Run id = {}\n'.format(run_id))

    # import time
    # time.sleep(7 * 60 * 60)

    main()
    # test('multiwoz-0507-162712-epoch-1.pt', load_combined_dicts())

