from pathlib import Path
import json
import random
random.seed(0)

from torch.utils.data import DataLoader
from utils.parse_frames import FramesDataset, \
                               load_dictionaries as frames_load_dicts, \
                               preds_to_dial_json, \
                               data_dir as frames_data_dir, \
                               sample_iterator
from utils.parse_multiwoz import MultiwozDataset, \
                                 load_dictionaries as multiwoz_load_dicts


def main():
    dials = []
    for fold in range(1, 11):
        with open(str(frames_data_dir / '{}.json'.format(fold)),
                  'r') as f_data:
            dials += json.load(f_data)

    dataset = MultiwozDataset(dicts=multiwoz_load_dicts(),
                              data_filename='valid_mixed_multiwoz.json')
    # dataset = FramesDataset(dicts=frames_load_dicts(), folds=range(1, 11))
    # dataset = FramesDataset(dicts=frames_load_dicts(), folds=[1])

    preds = []
    total = 0
    correct = 0
    for data in DataLoader(dataset):
        fs, sent, asvs, frames, active_frame, new_frames = data

        n_frames = len(frames)
        preds_turn = [[0] * n_frames for _ in range(len(asvs))]

        # Do something here.

        # Random
        # for i, asv in enumerate(asvs):
        #     frame = random.randrange(n_frames)
        #     preds_turn[i][frame] = 1

        # Always predict active frame (cheating)
        # for i, asv in enumerate(asvs):
        #     preds_turn[i][active_frame] = 1

        # Perfect answer
        # for i, asv in enumerate(asvs):
        #     preds_turn[i][0] = 1
        #     preds_turn[i][fs[i]] = 1
        # act_based = [0] * n_frames
        # for asv in asvs:
        #     if asv[2] == 'NOANNO':
        #         act_based[asv[0]] = 1
        # for i, asv in enumerate(asvs):
        #     if asv[2] == 'NOANNO':
        #         preds_turn[i] = act_based
        #     else:
        #         preds_turn[i][asv[0]] = 1

        # Compare sv to sv in frames
        #   ref to the first appearance
        #   o.w. ref to active frame
        for i, asv in enumerate(asvs):
            frame_id = active_frame.item()
            for f_id, frame in enumerate(frames):
                found = False
                for sv in frame:
                    is_equal = asv[1] == sv[0]
                    is_equal &= len(asv[2]) == len(sv[1])
                    if is_equal:
                        for w_id, w in enumerate(asv[2]):
                            is_equal &= asv[2][w_id].tolist() == \
                                        sv[1][w_id].tolist()
                    if is_equal:
                        frame_id = f_id
                        found = True
                        break
                if found:
                    break
            preds_turn[i][frame_id] = 1

        preds.append(preds_turn)
        total += fs.numel()
        for i in range(len(asvs)):
            correct += preds_turn[i][fs[0][i].item()] == 1

    print('{} / {} = {}'.format(correct, total, correct / total))
    return

    gen_dials = preds_to_dial_json(dials, preds)

    with open(str(Path(__file__).parent / \
                  '../results/gen.json'), 'w') as f_gen:
        json.dump(gen_dials, f_gen)


if __name__ == '__main__':
    main()

