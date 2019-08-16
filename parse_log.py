# -*- coding: utf-8 -*-

from pathlib import Path
import json
import re
import numpy as np


log_dir = Path(__file__).parent / 'runs'


def agg_runs(values):
    print(values)
    vs = np.array(values)
    return '{:.1f} ± {:.2f}'.format(vs.mean(), vs.var())


def main():
    # log_path = log_dir / '0621-141625.log'  # bert, no att
    # log_path = log_dir / '0621-142113.log'  # bert, simple
    # log_path = log_dir / '0621-143035.log'  # tri, no att
    # log_path = log_dir / '0621-143440.log'  # tri, simple

    # log_path = log_dir / '0708-162344.log'  # tri simple
    log_path = log_dir / '0708-162517.log'  # tri no
    # log_path = log_dir / '0708-162711.log'  # bert simple
    # log_path = log_dir / '0708-162833.log'  # bert no


    with open(str(log_path), 'r') as f:
        master_log = f.read()

    run_ids = re.findall('Run id = (.*)\n', master_log)

    valid_scores = []
    test_scores = []
    method = 'frames'
    
    for run_id in run_ids:
        # with open(str(log_dir / 'bert-0708' / run_id / 'log'), 'r') as f:
        # with open(str(log_dir / 'bert-0621' / run_id / 'log'), 'r') as f:
        # with open(str(log_dir / 'trigram-0621' / run_id / 'log'), 'r') as f:
        with open(str(log_dir / 'trigram-0708' / run_id / 'log'), 'r') as f:
            log = f.read()

            config_str = re.findall(
                '===== Configs =====\n(.*^}).*===== Model =====',
                log,
                re.MULTILINE | re.DOTALL)[0]
            config = json.loads(config_str)

            valid_score = re.findall(
                "\('frames-user-slot', 'accuracy'\): value = ([\d.]*)", log)
            test_score = re.findall(
                "\('test-frames-user-slot', 'accuracy'\): value = ([\d.]*)", log)

            text_embedding_map = {
                'trigram': 'Letter trigram',
                'bert-base-uncased': 'BERT'
            }

            asv_gru_hidden_map = {
                True: 'Hidden state',
                False: 'Output'
            }

            attention_map = {
                'no': 'No attention',
                'simple': 'Simple'
            }

            method_map = {
                "frames": 'From scratch',
                "['train_mixed_multiwoz.json']":
                    'Transfer',
                "['train_mixed_hotel_restaurant_multiwoz.json']":
                    'Transfer: hotel + restaurant',
                "['train_mixed_hotel_transport_multiwoz.json', 'train_mixed_multiwoz.json']":
                    'Transfer: hotel + restaurant + single domains',
                "['train_mixed_hotel_restaurant_multiwoz.json', 'train_mixed_hotel_transport_multiwoz.json', 'train_mixed_multiwoz.json']":
                    'Transfer: hotel + restaurant + transportation + single domains',
                "['train_mixed_hotel_restaurant_multiwoz.json', 'train_mixed_multiwoz.json']":
                    'Transfer: hotel + transportation + single domains',
                "['train_mixed_hotel_transport_multiwoz.json']":
                    'Transfer: hotel + transportation',
            }

            is_finetune = False
            if 'AttentionModel' in config['model_config']:
                model_config = config['model_config']['AttentionModel']
            else:
                assert 'checkpoint' in config['model_config']
                is_finetune = True

            is_pretrain = False
            if not is_finetune:
                datasets = config['train_datasets_config']
                is_pretrain = list(datasets[0].keys())[0] == 'multiwoz'
                
                if is_pretrain:
                    datasets = [dataset['multiwoz']['data_filename']
                                for dataset in datasets]
                    datasets.sort()
                    method = str(datasets)

            if not is_pretrain and len(valid_score) > 0:
                valid_scores.append(float(valid_score[0]) * 100)
            if not is_pretrain and len(test_score) > 0:
                test_scores.append(float(test_score[0]) * 100)

            if len(valid_scores) == 10 or len(test_score) == 0:
                data = {
                    '# of runs': len(valid_scores),
                    'Valid accuracy': agg_runs(valid_scores),
                    'Test accuracy': agg_runs(test_scores),
                    'Training subset': '100%',
                    'Methods': method_map[method],
                    'Text embedding': text_embedding_map[model_config['embed_type']],
                    'Utterance label encoding: with utterance': model_config['asv_with_utterance'],
                    'Utterance label encoding: GRU': asv_gru_hidden_map[model_config['asv_rnn_hidden']],
                    'Utterance label encoding: after GRU': 'Linear',
                    'Attention': attention_map[model_config['attention_type']]
                }

                data_str = json.dumps(data).replace('false', 'False') + ','
                print(data_str)

                valid_scores = []
                test_scores = []

        # break


if __name__ == '__main__':
    main()

