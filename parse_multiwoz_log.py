from pathlib import Path
import re
import json


def parse_train_log(train_log_path):
    with open(train_log_path, 'r') as f:
        train_log = f.read()

    train_loss = [float(v) for v in re.findall('Loss: ([\d.]+)', train_log)]
    valid_loss = [float(v) for v in re.findall('LOSS: ([\d.]+)', train_log)]
    grad = [float(v) for v in re.findall('Grad: ([\d.]+)', train_log)]

    return {'train': {
        'train_loss': train_loss,
        'valid_loss': valid_loss,
        'grad':       grad,
    }}


def parse_eval_log(eval_log_path):
    with open(eval_log_path, 'r') as f:
        eval_log = f.read()

    eval_log_json = {'greedy': {}, 'beam': {}}
    for i, s in enumerate(['greedy', 'beam']):
        eval_log_json[s] = {
            "valid_blue": [float(v) for v in
                re.findall('^Valid BLUES SCORE ([\d.]+)', eval_log, re.M)[i::2]],
            "valid_match": [float(v) for v in
                re.findall('^Valid Corpus Matches : ([\d.]+)', eval_log, re.M)[i::2]],
            "valid_success": [float(v) for v in
                re.findall('^Valid Corpus Success : ([\d.]+)', eval_log, re.M)[i::2]],
            "test_blue": [float(v) for v in
                re.findall('^Corpus BLUES SCORE ([\d.]+)', eval_log, re.M)[i::2]],
            "test_match": [float(v) for v in
                re.findall('^Corpus Matches : ([\d.]+)', eval_log, re.M)[i::2]],
            "test_success": [float(v) for v in
                re.findall('^Corpus Success : ([\d.]+)', eval_log, re.M)[i::2]],
        }

    return {'eval': eval_log_json}


def parse_logs(train_log_path, eval_log_path):
    logs_json = parse_train_log(train_log_path)
    logs_json.update(parse_eval_log(eval_log_path))
    return logs_json


if __name__ == '__main__':
    logs_json = (parse_logs(Path('multiwoz_train_output.txt'),
                            Path('multiwoz_eval_output.txt')))

    print(json.dumps(logs_json))

    # with open('multiwoz_log.json', 'w') as f:
    #     json.dump(logs_json, f)
