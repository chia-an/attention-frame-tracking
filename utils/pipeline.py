import logging
from pathlib import Path
from tqdm import tqdm
import datetime
import inspect

import torch
from tensorboardX import SummaryWriter


model_save_dir = Path(__file__).parent / '../model/model'
event_dir = Path(__file__).parent / '../runs'


def init_model(model_class, **kwargs):
    # # Copy the model file as a backup.
    # from importlib import import_module
    # from shutil import copy

    # model_file_path = inspect.getfile(model_class)
    # model_file_name = Path(model_file_path).name

    # # copy(model_class_path, str(event_dir / run_id / model_file_name))
    # test_dir = Path(__file__).parent / '../test'
    # copy(model_file_path, str(test_dir / model_file_name))

    # m = import_module('...test.test.{}'.format(Path(model_file_name).stem),
    #                   package=__name__)

    # return

    model = model_class(**kwargs)
    with open(inspect.getfile(model.__class__), 'r') as f_code:
        code = f_code.read()

    logger = logging.getLogger('logger.model')
    logger.debug('\n===== Model =====')
    logger.debug(model)
    logger.debug('\n===== Model optimizer =====')
    logger.debug(model.optimizer)
    logger.debug('\n===== Model file =====')
    logger.debug(code)

    return model, code


def load_checkpoint(file_name):
    model = torch.load(str(model_save_dir / file_name))
    with open(str((model_save_dir / file_name).with_suffix('.py')),
              'r') as f_code:
        code = f_code.read()

    # Check if the model file is the same.
    import filecmp
    saved_model_file = str((model_save_dir / file_name).with_suffix('.py'))
    current_model_file = inspect.getfile(model.__class__)
    assert filecmp.cmp(saved_model_file, current_model_file), \
           'Saved code and current code are not the same.\n' \
           'saved file = {}\ncurrent file = {}.'.format(
                saved_model_file, current_model_file)

    logger = logging.getLogger('logger.model')
    logger.info('[Pass] File check.\n' \
                'checkpoint file = {}\ncurrent file = {}\n'. \
                format(saved_model_file, current_model_file))
    logger.debug('\n===== Model =====')
    logger.debug(model)
    logger.debug('\n===== Model optimizer =====')
    logger.debug(model.optimizer)
    logger.debug('\n===== Model file =====')
    logger.debug(code)

    return model, code


def save_checkpoint(file_dir, model, code):
    torch.save(model, str(file_dir))
    with open(str(file_dir.with_suffix('.py')), 'w') as f_code:
        f_code.write(code)


def train_hook(run_id,
               model,
               code,
               n_epochs,
               train_loader,
               valid_loaders,
               metrics={},
               main_valid_metric=('', ''),
               model_filename_prefix='',
               train_id_prefix='',
               save=True,
               save_best_only=False):

    tqdm_bar_format = '{l_bar}{r_bar}'

    train_iter_writer = SummaryWriter(
        log_dir=str(event_dir / run_id / 'train_iter'))
    train_writer = SummaryWriter(
        log_dir=str(event_dir / run_id / 'train'))
    valid_writer = SummaryWriter(
        log_dir=str(event_dir / run_id / 'valid'))

    logger = logging.getLogger('logger.train')
    logger.info('\n===== Training =====')

    assert not main_valid_metric[0] or \
           main_valid_metric[0] in valid_loaders, \
           'Unknown valid set. {} is not in {}.'.format(
                main_valid_metric[0], list(valid_loaders.keys()))
    assert not main_valid_metric[1] or \
           main_valid_metric[1] in metrics, \
           'Unknown metric. {} is not in {}.'.format(
                main_valid_metric[1], list(metrics.keys()))
    if main_valid_metric == ('', ''):
        logger.warning('Main valid set and metric not assigned.')

    best_valid_metric_value = -float('Inf')
    current_valid_metric_value = -float('Inf')
    history = {}
    for valid_name in valid_loaders.keys():
        history[(valid_name, 'loss')] = []
        for k in metrics.keys():
            history[(valid_name, k)] = []

    iteration = 0
    for epoch in range(1, n_epochs + 1):
        logger.info('# Epoch {}/{}'.format(epoch, n_epochs))

        # Train
        if train_loader is None:
            iteration += 1
        else:
            model.train()
            train_loss = []
            train_metrics = {k: {'value': 0, 'state': None}
                             for k in metrics.keys()}
            for data in tqdm(train_loader, bar_format=tqdm_bar_format):
                iteration += 1
                target, *input = data

                if target.numel() == 0:
                    continue

                loss, output = model.step(input, target)

                train_iter_writer.add_scalar('train/loss', loss, iteration)

                train_loss.append(loss)

                for metric_name, metric in metrics.items():
                    value, _ = metric(target, output)
                    train_iter_writer.add_scalar(
                        'train/{}'.format(metric_name), value, iteration) 

                    value, state = metric(
                        target, output, train_metrics[metric_name]['state'])
                    train_metrics[metric_name] = {
                        'value': value, 'state': state}

            train_writer.add_scalar('train/loss',
                                    sum(train_loss) / len(train_loss),
                                    iteration)
            logger.info('train/loss = {}'.format(
                sum(train_loss) / len(train_loss)))

            for metric_name in metrics.keys():
                value = train_metrics[metric_name]['value']
                train_writer.add_scalar(
                    'train/{}'.format(metric_name),
                    value,
                    iteration
                ) 
                logger.info('train/{} = {}'.format(metric_name, value))

        # Valid
        model.eval()
        for valid_name, valid_loader in valid_loaders.items():
            valid_loss = []
            valid_metrics = {k: {'value': 0, 'state': None}
                             for k in metrics.keys()}
            for data in tqdm(valid_loader, bar_format=tqdm_bar_format):
                target, *input = data

                if target.numel() == 0:
                    continue

                loss, output = model.step(input, target, train=False)

                valid_loss.append(loss)

                for metric_name, metric in metrics.items():
                    value, state = metric(
                        target, output, valid_metrics[metric_name]['state'])
                    valid_metrics[metric_name] = {
                        'value': value, 'state': state}

            valid_writer.add_scalar('valid-{}/loss'.format(valid_name),
                                    sum(valid_loss) / len(valid_loss),
                                    iteration)
            logger.info('valid-{}/loss = {}'.format(
                valid_name, sum(valid_loss) / len(valid_loss)))
            history[(valid_name, 'loss')].append(
                sum(valid_loss) / len(valid_loss))

            for metric_name in metrics.keys():
                value = valid_metrics[metric_name]['value']
                if (valid_name, metric_name) == main_valid_metric:
                    current_valid_metric_value = value
                valid_writer.add_scalar(
                    'valid-{}/{}'.format(valid_name, metric_name),
                    value,
                    iteration
                ) 
                logger.info('valid-{}/{} = {}'.format(
                    valid_name, metric_name, value))
                history[(valid_name, metric_name)].append(value)

        # Save checkpoint
        if save:
            if not save_best_only:
                model_filename = '{}-epoch-{}.pt'.format(run_id, epoch)
                save_checkpoint(
                    file_dir=model_save_dir / model_filename,
                    model=model,
                    code=code
                )
                logger.info('Save checkpoint {}.'.format(model_filename))

            # Save if it is the new best.
            if current_valid_metric_value > best_valid_metric_value:
                logger.info('Update best model.')
                best_valid_metric_value = current_valid_metric_value

                model_filename = '{}-best.pt'.format(run_id)
                save_checkpoint(
                    file_dir=model_save_dir / model_filename,
                    model=model,
                    code=code
                )
                logger.info('Save checkpoint {}.'.format(model_filename))


    train_writer.close()
    valid_writer.close()

    # Show summary
    logger.info('\n===== Summary =====')
    main_best_epoch = None
    for k, v in history.items():
        if k[1] == 'loss':
            best_epoch, best_value = min(enumerate(v), key=lambda x: x[1])
        else:
            best_epoch, best_value = max(enumerate(v), key=lambda x: x[1])
        best_epoch += 1

        if k == main_valid_metric:
            main_best_epoch = best_epoch

        logger.info('{}: epoch = {}, value = {}'.format(
            k, best_epoch, best_value))

    logger.info('\nBest epoch = {}'.format(main_best_epoch))
    for k, v in history.items():
        logger.info('{}: value = {}'.format(k, v[main_best_epoch - 1]))

    return {
        'run_id': run_id,
        'best_checkpoint': '{}-best.pt'.format(run_id),
        'history': history,
    }


def eval_hook(model,
              data_loader=None,
              metrics={}):

    tqdm_bar_format = '{l_bar}{r_bar}'

    logger = logging.getLogger('logger.eval')

    model.eval()
    outputs = []
    eval_loss = []
    eval_metrics = {k: {'value': 0, 'state': None} for k in metrics.keys()}
    for data in tqdm(data_loader, bar_format=tqdm_bar_format):
        target, *input = data

        if target.numel() == 0:
            outputs.append([[]])
            continue

        loss, output = model.step(input, target, train=False)

        eval_loss.append(loss)
        outputs.append(output.tolist())
        for metric_name, metric in metrics.items():
            value, state = metric(
                target, output, eval_metrics[metric_name]['state'])
            eval_metrics[metric_name] = {'value': value, 'state': state}

    logger.info('eval/loss = {}'.format(sum(eval_loss) / len(eval_loss)))
    for metric_name in metrics.keys():
        value = eval_metrics[metric_name]['value']
        logger.info('eval/{} = {}'.format(metric_name, value))

    return outputs


if __name__ == '__main__':
    pass

