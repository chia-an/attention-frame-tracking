from torch.utils.data import ConcatDataset, DataLoader
from parse_frames import FramesDataset, \
                         load_dictionaries as frames_load_dicts
from parse_multiwoz import MultiwozDataset, \
                           load_dictionaries as multiwoz_load_dicts


def n_frames_stats():
    """
    Output:
    root@afa32b085c28:/work/src-git# python3 utils/test.py
    multiwoz
    avg = 292099 / 62711 = 4.65785906778715
    var = 27.41903334343257
    frames
    avg = 84289 / 19986 = 4.217402181527069
    var = 31.79740818573001
    """
    multiwoz_dicts = multiwoz_load_dicts()
    frames_dicts = frames_load_dicts()

    multiwoz_train_dataset = MultiwozDataset(
        dicts=multiwoz_dicts, data_filename='train_mixed_multiwoz.json')
    multiwoz_valid_dataset = MultiwozDataset(
        dicts=multiwoz_dicts, data_filename='valid_mixed_multiwoz.json')
    multiwoz_dataset = ConcatDataset([
        multiwoz_train_dataset,
        multiwoz_valid_dataset,
    ])

    frames_train_dataset = FramesDataset(dicts=frames_dicts, folds=range(1, 9))
    frames_valid_dataset = FramesDataset(dicts=frames_dicts, folds=[9])
    frames_test_dataset = FramesDataset(dicts=frames_dicts, folds=[10])
    frames_dataset = ConcatDataset([
        frames_train_dataset,
        frames_valid_dataset,
        frames_test_dataset,
    ])

    multiwoz_n_frames = []
    for data in DataLoader(multiwoz_dataset):
        fs, sent, asvs, frames, active_frame, new_frame = data
        multiwoz_n_frames.append(len(frames))

    frames_n_frames = []
    for data in DataLoader(frames_dataset):
        fs, sent, asvs, frames, active_frame, new_frame = data
        frames_n_frames.append(len(frames))

    print('multiwoz')
    print('avg = {} / {} = {}'.format(
        sum(multiwoz_n_frames),
        len(multiwoz_n_frames),
        sum(multiwoz_n_frames) / len(multiwoz_n_frames)))
    print('var = {}'.format(
        sum(x * x for x in multiwoz_n_frames) / len(multiwoz_n_frames)))

    print('frames')
    print('avg = {} / {} = {}'.format(
        sum(frames_n_frames),
        len(frames_n_frames),
        sum(frames_n_frames) / len(frames_n_frames)))
    print('var = {}'.format(
        sum(x * x for x in frames_n_frames) / len(frames_n_frames)))


if __name__ == '__main__':
    pass
