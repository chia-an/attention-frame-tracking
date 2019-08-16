from pathlib import Path
from tqdm import tqdm
import json
import datetime
import random
import logging
import os

from utils.dictionary import trigram_iterator
from utils.parse_frames import sample_iterator, \
                               load_dictionaries, \
                               data_dir, \
                               preds_to_dial_json, \
                               FramesDataset

import torch
from torch.utils.data import Dataset, DataLoader

from model.maluuba import Model
from model.attention import Model as AttentionModel

from utils.pipeline import train_hook, eval_hook, init_model, load_checkpoint
from utils.metrics import accuracy


# Path constants
# model_save_dir = Path(__file__).parent / 'model/model'
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

run_id = 'frames-{}'.format(datetime.datetime.now().strftime('%m%d-%H%M%S'))
os.mkdir(str(log_dir / run_id))
fh = logging.FileHandler(str(log_dir / run_id / 'log'))
fh.setLevel(logging.DEBUG)
logger.addHandler(fh)

ch = logging.StreamHandler()
ch.setLevel(logging.INFO)
logger.addHandler(ch)


def train(model, code, dicts):
    print('Train on FRAMES.')

    n_epochs = 5
    subsample = 10
    pad_frame = False
    # tokenizer = 'bert-base-uncased'
    tokenizer = 'trigram'

    train_dataset = FramesDataset(
        dicts=dicts,
        folds=list(range(1, 9)),
        # folds=list(range(1, 5)),
        subsample=subsample,
        pad_frame=pad_frame,
        tokenizer=tokenizer)
    valid_dataset = FramesDataset(
        dicts,
        folds=[9],
        subsample=subsample,
        pad_frame=pad_frame,
        tokenizer=tokenizer)
    true_valid_dataset = FramesDataset(
        dicts,
        folds=[9],
        subsample=subsample,
        slot_based=True,
        user_only=True,
        pad_frame=pad_frame,
        tokenizer=tokenizer)
    true_test_dataset = FramesDataset(
        dicts,
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
                       'test-user-slot-frames': true_test_loader},
        metrics={'accuracy': accuracy},
        model_filename_prefix='frames',
        train_id_prefix='frames',
        main_valid_metric=('user-slot-frames', 'accuracy'),
        # save=False,
        # save_best_only=True,
    )


def test(model_filename, dicts):
    # Load model.
    model, code = load_checkpoint(model_filename)

    pad_frame = False

    valid_dataset = FramesDataset(dicts, folds=[9], pad_frame=pad_frame)
    test_dataset = FramesDataset(dicts, folds=[10], pad_frame=pad_frame)
    true_valid_dataset = FramesDataset(
        dicts, folds=[9],
        slot_based=True,
        user_only=True,
        pad_frame=pad_frame)
    true_test_dataset = FramesDataset(
        dicts,
        folds=[10],
        slot_based=True,
        user_only=True,
        pad_frame=pad_frame)

    valid_loader = DataLoader(valid_dataset)
    test_loader = DataLoader(test_dataset)
    true_valid_loader = DataLoader(true_valid_dataset)
    true_test_loader = DataLoader(true_test_dataset)

    eval_hook(
        model=model,
        data_loader=true_valid_loader,
        metrics={'user-slot-acc': accuracy}
    )
    valid_output = eval_hook(
        model=model,
        data_loader=valid_loader,
        metrics={'accuracy': accuracy}
    )
    eval_hook(
        model=model,
        data_loader=true_test_loader,
        metrics={'user-slot-acc': accuracy}
    )
    test_output = eval_hook(
        model=model,
        data_loader=test_loader,
        metrics={'accuracy': accuracy}
    )

    # # Debug
    # with open('debug.test.out', 'w') as f:
    #     for output in debug_output:
    #         turn_output = output[0]
    #         for asv_output in turn_output:
    #             ans = max(enumerate(asv_output), key=lambda x: x[1])[0]
    #             f.write(str(ans) + '\n')

    # Generate json prediction (for eval metrics)
    all_dials = []
    for fold in range(9, 11):
        with open(str(data_dir / '{}.json'.format(fold)), 'r') as f_data:
            dials = json.load(f_data)
            all_dials += dials

    all_preds = [output[0] for output in valid_output + test_output]

    gen_dials = preds_to_dial_json(all_dials, all_preds)
    gen_filename = 'gen_{}.json'.format(model_filename.rsplit('.', 1)[0])
    with open(str(result_save_dir / gen_filename), 'w') as f_gen:
        json.dump(gen_dials, f_gen)


def main():
    # Load data, dicts.
    dicts = load_dictionaries()
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
        # 'embed_type': 'bert-base-uncased',
        'embed_type': 'trigram',
        'embed_dim_act': 128,
        'embed_dim_slot': 128,
        'embed_dim_text': 128,
        'embed_dim_tri': 128,
        'embed_dim': 128,
        'asv_rnn_hidden_size': 128,
        'frame_rnn_hidden_size': 128,
    }
    other_kwargs = {
        'device': device
    }

    logger.info('===== Model args =====')
    logger.info(json.dumps(static_kwargs))
    logger.info('')

    model, code = init_model(AttentionModel, **static_kwargs, **other_kwargs)

    # model = Model(**kwargs)
    # model = AttentionModel(**kwargs)
    # model = torch.load(str(model_save_dir / 'frames-epoch-8-0417-195221.pt'))

    train(model=model, code=code, dicts=dicts)


if __name__ == '__main__':
    logger.info('Run id = {}\n'.format(run_id))

    # import time
    # time.sleep(7 * 60 * 60)

    main()
    # test('scratch-0507-113201-epoch-4.pt', load_dictionaries())

