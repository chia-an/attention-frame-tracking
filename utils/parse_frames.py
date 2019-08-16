from pathlib import Path
import json
from pprint import pprint
from torch.utils.data import Dataset
from difflib import get_close_matches


data_dir = Path(__file__).parent / '../../data/Frames-dataset'
data_path = data_dir / 'frames.json'


class FramesDataset(Dataset):
    def __init__(self, dicts,
                       folds=[1],
                       subsample=None,
                       slot_based=False,
                       user_only=False,
                       pad_frame=False,
                       tokenizer='trigram',
                       text_position=False):
        import torch
        from utils.dictionary import trigram_iterator

        if tokenizer.startswith('bert'):
            from pytorch_pretrained_bert import BertTokenizer
            bert_tknzr = BertTokenizer.from_pretrained(tokenizer)
        else:
            from nltk.tokenize import TweetTokenizer
            tknzr = TweetTokenizer()

        # def tokenize(word):
        def tokenize(sent):
            if tokenizer == 'trigram':
                return [torch.tensor([tri_to_index[tri]
                                      for tri in trigram_iterator(word)])
                        for word in tknzr.tokenize(sent)]
                        # for word in sent.split()]
            elif tokenizer == 'word':
                return [word_to_index[word] for word in tknzr.tokenize(sent)]
                # return [word_to_index[word] for word in sent.split()]
            elif tokenizer.startswith('bert'):
                tokenized_text = bert_tknzr.tokenize(
                    ' '.join(['[CLS]', sent, '[SEP]']))
                indexes = bert_tknzr.convert_tokens_to_ids(tokenized_text)
                return torch.tensor(indexes)
                # return indexes
            else:
                raise Exception('Unknown tokenizer {}.'.format(tokenizer))

        word_to_index, index_to_word, \
        tri_to_index, index_to_tri, \
        act_to_index, index_to_act, \
        slot_to_index, index_to_slot = dicts

        dials = []
        for fold in folds:
            with open(str(data_dir / '{}.json'.format(fold)), 'r') as f_data:
                fold_dials = json.load(f_data)
            dials += fold_dials

        if user_only:
            samples = sample_iterator(dials, authors=['user'])
        else:
            samples = sample_iterator(dials)

        self.data = []
        for sample in samples:
            # TODO: SOS, EOS, PAD token?
            # print(sample)
            sent, fasvs, frames, active_frame, new_frames = sample

            # sent = [tokenize(w) for w in sent.split()]
            sent = tokenize(sent)

            fs = []
            asvs = []
            for f, a, s, v in fasvs:
                if slot_based and s is 'NOANNO':
                    continue
                fs.append(f)
                asvs.append((act_to_index[a],
                             slot_to_index[s],
                             tokenize(v)))
                             # [tokenize(w) for w in v.split()]))
            fs = torch.tensor(fs)

            frames = [[(slot_to_index[s],
                        tokenize(v))
                        # [tokenize(w) for w in v.split()])
                       for s, v in frame['slot_value']]
                      for frame in frames]
            if pad_frame:
                null_sv = (slot_to_index['NOANNO'], [tokenize('NULL')])
                for i, frame in enumerate(frames):
                    n_repeat = 32 - len(frame)
                    frames[i] = frame + [null_sv] * n_repeat
            for frame in frames:
                assert frame, (frame, sample)
                assert not pad_frame or len(frame) == 32, \
                       (len(frame), frame, sample)

            new_frames = torch.tensor(new_frames)

            # Type:
            #   word: size[1, *], size[1] if no trigram
            #   text: list of word
            #   fs: size[n_asvs]
            #   sent: text
            #   asvs: list of (size[], size[], text)
            #   frames: list of list of (size[], text)
            #   active_frame: size[]
            #   new_frames: size[*]

            self.data.append((
                fs, sent, asvs, frames, active_frame, new_frames))

            if subsample is not None and len(self.data) >= subsample:
                break

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]



### Functions for dialogue jsons ###


def turn_iterator(dials):
    for dial in dials:
        for turn in dial['turns']:
            yield turn


# def text_iterator(dials):
#     for turn in turn_iterator(dials):
#         yield turn['text']


# def act_iterator(turn, authors=['user']):
#     if turn['author'] in authors:
#         for act in turn['labels']['acts']:
#             yield act


# def turn_asv_iterator(turn):
    """ Extract the tuples (action, slot, value).
    Use the labels in 'acts_without_refs' for each user turns.
    Ignore actions with no slots because it's not evaluated.
    """
    # for dial_id, dial in enumerate(dials):
    #     for turn_id, turn in enumerate(dial['turns']):
    # if turn['author'] == 'user': 
    #     for act in turn['labels']['acts_without_refs']:
    #         for arg in act['args']:
    #             yield act['name'], arg['key'], str(arg['val'])


def frame_key_value_iterator(act, current_frame=0,
        shuffle_slots=False, rng=None):
    """ Iterate over all key value pairs in an act.
    If a key value pair does not have a frame_id, yield frame_id=current_frame.
    If a ref without annotations is given, yields (frame_id, 'NOANNO', 'NULL').
    Unannotated refs are produced ordered by frame_id.

    Modify from https://github.com/Maluuba/frames/frames/utils.py
    """

    if shuffle_slots:
        rng.shuffle(act['args'])

    unannotated_refs = []
    has_refs = False

    for arg in act['args']:
        if arg['key'] in ('ref', 'read', 'write'):
            for frame in arg['val']:
                if not frame.get('annotations', []):
                    unannotated_refs += (frame['frame'] - 1, 'NOANNO', 'NULL'),
                for kv in frame.get('annotations', []):
                    has_refs = True
                    assert kv['key'] not in ('ref', 'read', 'write'), \
                           (kv, frame)
                    yield frame['frame'] - 1, \
                          kv['key'], \
                          str(kv.get('val', 'NULL'))
        elif arg['key'] != 'id':
            yield current_frame - 1, arg['key'], str(arg.get('val', 'NULL'))

    if unannotated_refs:
        for ua in sorted(unannotated_refs):
            yield ua
    elif not has_refs:
        yield current_frame - 1, 'NOANNO', 'NULL'


def preds_to_act_json(act, preds):
    """ Transform predictions of a act into json for evaluation.
    The order in prediction should be the same as in the act.
    """
    preds = iter(preds)
    gen_act = {'name': act['name']}
    gen_args = []

    has_noanno = False
    has_refs = False

    for arg in act['args']:
        if arg['key'] in ('ref', 'read', 'write'):
            for frame in arg['val']:
                if not frame.get('annotations', []):
                    has_noanno = True
                for kv in frame.get('annotations', []):
                    has_refs = True
                    assert kv['key'] not in ('ref', 'read', 'write'), \
                           (kv, frame)
                    gen_args.append({
                        'key': arg['key'],
                        'val': [{'annotations': [kv],
                                 'frame': next(preds)}]})
        elif arg['key'] != 'id':
            # The answer is the current frame.
            gen_args.append({
                'key': 'ref',  # Fake key to fit the evaluation format.
                'val': [{'annotations': [arg],
                         'frame': next(preds)}]})

    # For action based evaluation.
    if has_noanno or not has_refs:
        gen_args.append({
            'key': 'ref',  # Fake key to fit the evaluation format.
            'val': [{'annotations': [],
                     'frame': next(preds)}]})

    gen_act['args'] = gen_args
    return gen_act


def fasv_iterator(turn):
    """ Yield all (frame_id, action name, slot, value) in a turn.
    """
    for act in turn['labels']['acts']:
        for f, s, v in frame_key_value_iterator(act,
                current_frame=turn['labels']['active_frame']):
            yield f, act['name'], s, v


def frame_sv_iterator(frame):
    # "We encode a string representation of the most recent
    # non-negated value v as described in Section 5.1.1."
    # NOTE: add the value here to text dictionary?
    for s, vs in sorted(frame['info'].items()):
        for v in vs:
            if not v['negated']:
                yield s, str(v['val'])


def sample_iterator(dials=None,
                    authors=['user', 'wizard'],
                    text_position=False):
    # NOTE: write a data_loader or not?
    if dials is None:
        with open(str(data_path), 'r') as f_data:
            dials = json.load(f_data)

    # from nltk.tokenize import TweetTokenizer

    # tknzr = TweetTokenizer()
    last_frames = []
    turn_id = 0

    for turn in turn_iterator(dials):
        # # NOTE: consider only user turn or not?
        # if turn['author'] == 'user' or True:
        if turn['author'] in authors:
            # NOTE: Assume this starts a new dialogue.
            if len(turn['labels']['frames']) < len(last_frames):
                turn_id = 0

            # User utterance.
            # sent = ' '.join(tknzr.tokenize(turn['text']))
            sent = turn['text']

            active_frame = turn['labels']['active_frame'] - 1

            # Get list of tuples (f, a, s, v).
            fasvs = []
            for f, a, s, v in fasv_iterator(turn):
                # Compute position (if can find).
                if text_position:
                    assert s != 'ref_anaphora' or \
                           v.lower() in sent.lower(), (a, s, v, sent)
                    index = sent.lower().find(v.lower())
                    if index == -1:
                        index = None
                    v = (v, turn_id, index)

                # fasvs.append((f, a, s, ' '.join(tknzr.tokenize(v))))
                fasvs.append((f, a, s, v))

            # Get list of frames.
            frames = []
            for frame in turn['labels']['frames']:
                # Each frame has a list of s-v pairs.
                svs = []
                for s, v in frame_sv_iterator(frame):
                    # v = ' '.join(tknzr.tokenize(v))
                    if True or (s, v) not in svs:
                        svs.append((s, v))
                if svs == []:
                    svs.append(('NOANNO', 'NULL'))
                assert svs, (svs, turn)

                frames.append({
                    'frame_id': frame['frame_id'] - 1,
                    'slot_value': svs,
                    'parent_id': frame['frame_parent_id'] - 1
                        if frame['frame_parent_id'] is not None else -1,
                })

            # New frames created in current turn.
            # NOTE: Assume a frame doesn't disappear in a dialogue.
            frames.sort(key=lambda f: f['frame_id'])
            new_frames = []
            if len(frames) > len(last_frames):
                for frame in frames:
                    is_new = True
                    for old_frame in last_frames:
                        is_new &= frame['frame_id'] != old_frame['frame_id']
                    if is_new:
                        new_frames.append((frame['frame_id']))
            elif len(frames) < len(last_frames):
                assert len(frames) == 1, \
                       (len(frames), len(last_frames), frames, last_frames)
                new_frames.append(frames[0]['frame_id'])

            # Add text position to frames.
            if text_position:
                for frame_id, frame in enumerate(frames):
                    if frame['frame_id'] not in new_frames:
                        assert frame_id == frame['frame_id'], (frame_id, frame)
                        frames[frame_id] = last_frames[frame_id]
                    else:
                        parent_id = frame['parent_id']
                        parent_frame = []
                        if parent_id >= 0:
                            parent_frame = frames[parent_id]['slot_value']

                        # NOTE: Assume a slot-value comes from either parent
                        #       frame or utterance label. It is hard to check
                        #       if a value comes from utterance because the
                        #       text can be different (and there are wrong
                        #       labels).
                        for sv_id, (s, v) in enumerate(frame['slot_value']):
                            found = False
                            for ps, pv in parent_frame:
                                if (s, v) == (ps, pv[0]):
                                    v = pv
                                    found = True

                            if not found:
                                # NOTE: find() may not give the correct index,
                                #       e.g. '2' matches '2100'.
                                index = sent.lower().find(v.lower())
                                if index == -1:
                                    index = None
                                v = (v, turn_id, index)

                            frame['slot_value'][sv_id] = (s, v)

                        frames[frame_id] = frame

                            # in_parent = (s, v) in parent_frame
                            # in_utterance = False
                            # for fasv in fasvs:
                            #     if s == fasv[2] and \
                            #        (get_close_matches(v, [fasv[3]]) or v in fasv[3]):
                            #         in_utterance = True
                            #     if s.lower() == fasv[1].lower():
                            #         in_utterance = True

                            # assert in_parent or in_utterance, \
                            #        (in_parent,
                            #         in_utterance,
                            #         (s, v),
                            #         sent,
                            #         fasvs, parent_frame)

            # # TODO: keep a list of frame_id to speed up.
            # # TODO: BUG, new_frames is incorrect (at the beginning of the
            # #       dialogue) because last_frames is not cleared
            # new_frames = []
            # for frame in frames:
            #     is_new = True
            #     for old_frame in last_frames:
            #         # is_new &= frame['frame_id'] != old_frame['frame_id']
            #         if frame == old_frame:
            #             is_new = False
            #     if is_new:
            #         new_frames.append((frame['frame_id']))

            sample = (sent, fasvs, frames, active_frame, new_frames)

            yield sample

            turn_id += 1
            last_frames = frames


def preds_to_dial_json(dials, preds):
    """ Transform predictions to jsons for evaluation.
    preds: list[preds for each turn]

    The order in preds should be the same as in sample_iterator().
    """
    preds = iter(preds)
    gen_dials = []

    for dial in dials:
        gen_turns = []
        for turn, sample, pred in zip(turn_iterator([dial]),
                sample_iterator([dial]), preds):
            # Transform to json.
            offset = 0
            gen_acts = []

            # n_fasvs = sum(len(list(frame_key_value_iterator(act)))
            #               for act in turn['labels']['acts'])
            # assert n_fasvs <= len(pred), \
            #        (n_fasvs, len(pred), turn, sample)

            for act in turn['labels']['acts']:
                n_fsvs = len(list(frame_key_value_iterator(act)))
                if n_fsvs:
                    assert len(pred) >= offset + n_fsvs
                    gen_act = preds_to_act_json(
                        act, pred[offset: offset+n_fsvs])
                gen_acts.append(gen_act)
                offset += n_fsvs

            gen_turn = turn.copy()
            gen_turn['predictions'] = {'acts': gen_acts}
            gen_turns.append(gen_turn)

        gen_dial = dial.copy()
        gen_dial['turns'] = gen_turns
        gen_dials.append(gen_dial)

    return gen_dials



### Functions for dictionaries ###


def text_dictionaries(dials):
    """ Tokenize texts, and create letter trigram dict.
    """
    from nltk.tokenize import TweetTokenizer
    from dictionary import trigram_iterator

    tknzr = TweetTokenizer()
    index_to_word = {}
    word_to_index = {}

    index_to_tri = {}
    tri_to_index = {}

    # Create word dictionary.
    def add_word(w):
        if w not in word_to_index:
            index = len(word_to_index) + 1
            word_to_index[w] = index
            index_to_word[index] = w

    for turn in turn_iterator(dials):
        for w in tknzr.tokenize(turn['text']):
            add_word(w)

        for _, _, _, v in fasv_iterator(turn):
            for w in tknzr.tokenize(v):
                add_word(w)

        for frame in turn['labels']['frames']:
            for _, v in frame_sv_iterator(frame):
                for w in tknzr.tokenize(v):
                    add_word(w)

    # Create letter trigram dictionary.
    for w in word_to_index.keys():
        for trigram in trigram_iterator(w):
            if trigram not in tri_to_index:
                index = len(tri_to_index) + 1
                tri_to_index[trigram] = index
                index_to_tri[index] = trigram

    return word_to_index, index_to_word, \
           tri_to_index, index_to_tri


def action_dictionary(dials):
    act_to_index = {}
    index_to_act = {}

    for turn in turn_iterator(dials):
        for act in turn['labels']['acts']:
            if act['name'] not in act_to_index:
                index = len(act_to_index) + 1
                act_to_index[act['name']] = index
                index_to_act[index] = act['name']

    return act_to_index, index_to_act


def slot_dictionary(dials):
    slot_to_index = {}
    index_to_slot = {}

    for turn in turn_iterator(dials):
        for _, _, s, _ in fasv_iterator(turn):
            if s not in slot_to_index:
                index = len(slot_to_index) + 1
                slot_to_index[s] = index
                index_to_slot[index] = s

        for frame in turn['labels']['frames']:
            for s, _ in frame_sv_iterator(frame):
                if s not in slot_to_index:
                    index = len(slot_to_index) + 1
                    slot_to_index[s] = index
                    index_to_slot[index] = s

    return slot_to_index, index_to_slot


def create_dictionaries():
    with open(str(data_path), 'r') as f_data:    
        dials = json.load(f_data)

    word_to_index, index_to_word, \
    tri_to_index, index_to_tri = text_dictionaries(dials)
    act_to_index, index_to_act = action_dictionary(dials)
    slot_to_index, index_to_slot = slot_dictionary(dials)

    # return

    with open(str(data_dir / 'word_to_index.json'), 'w') as f_dict:
        json.dump(word_to_index, f_dict)
    with open(str(data_dir / 'index_to_word.json'), 'w') as f_dict:
        json.dump(index_to_word, f_dict)
    with open(str(data_dir / 'tri_to_index.json'), 'w') as f_dict:
        json.dump(tri_to_index, f_dict)
    with open(str(data_dir / 'index_to_tri.json'), 'w') as f_dict:
        json.dump(index_to_tri, f_dict)
    with open(str(data_dir / 'act_to_index.json'), 'w') as f_dict:
        json.dump(act_to_index, f_dict)
    with open(str(data_dir / 'index_to_act.json'), 'w') as f_dict:
        json.dump(index_to_act, f_dict)
    with open(str(data_dir / 'slot_to_index.json'), 'w') as f_dict:
        json.dump(slot_to_index, f_dict)
    with open(str(data_dir / 'index_to_slot.json'), 'w') as f_dict:
        json.dump(index_to_slot, f_dict)


def load_dictionaries():
    with open(str(data_dir / 'word_to_index.json'), 'r') as f_dict:
        word_to_index = json.load(f_dict)
    with open(str(data_dir / 'index_to_word.json'), 'r') as f_dict:
        index_to_word = json.load(f_dict)
    with open(str(data_dir / 'tri_to_index.json'), 'r') as f_dict:
        tri_to_index = json.load(f_dict)
    with open(str(data_dir / 'index_to_tri.json'), 'r') as f_dict:
        index_to_tri = json.load(f_dict)
    with open(str(data_dir / 'act_to_index.json'), 'r') as f_dict:
        act_to_index = json.load(f_dict)
    with open(str(data_dir / 'index_to_act.json'), 'r') as f_dict:
        index_to_act = json.load(f_dict)
    with open(str(data_dir / 'slot_to_index.json'), 'r') as f_dict:
        slot_to_index = json.load(f_dict)
    with open(str(data_dir / 'index_to_slot.json'), 'r') as f_dict:
        index_to_slot = json.load(f_dict)

    return word_to_index, index_to_word, \
           tri_to_index, index_to_tri, \
           act_to_index, index_to_act, \
           slot_to_index, index_to_slot


def split_dataset():
    folds = {'U21E41CQP': 1,
             'U23KPC9QV': 1,
             'U21RP4FCY': 2,
             'U22HTHYNP': 3,
             'U22K1SX9N': 4,
             'U231PNNA3': 5,
             'U23KR88NT': 6,
             'U24V2QUKC': 7,
             'U260BGVS6': 8,
             'U2709166N': 9,
             'U2AMZ8TLK': 10}

    with open(str(data_path), 'r') as f_data:    
        dials = json.load(f_data)

    fold_dials = [[] for _ in range(10)]
    for dial in dials:
        fold_dials[folds[dial['user_id']] - 1].append(dial)

    for i in range(10):
        with open(str(data_dir / '{}.json'.format(i + 1)), 'w') as f_data:
            json.dump(fold_dials[i], f_data)
    

def main():
    import random
    random.seed(2)

    with open(str(data_path), 'r') as f_data:    
        dials = json.load(f_data)

    preds = []
    # for sample in sample_iterator([dials[0]]):
    for sample in sample_iterator(dials):
        sent, fasvs, frames, active_frame, new_frames = sample
        n_frames = len(frames)
        preds_turn = [[0] * n_frames for _ in range(len(fasvs))]

        # Do something here.

        # Random
        for i, fasv in enumerate(fasvs):
            frame = random.randrange(n_frames)
            preds_turn[i][frame] = 1

        # # Perfect answer
        # for i, fasv in enumerate(fasvs):
        #     preds_turn[i][0] = 1
        #     preds_turn[i][fasv[0]] = 1
        # act_based = [0] * n_frames
        # for fasv in fasvs:
        #     if fasv[2] == 'NOANNO':
        #         act_based[fasv[0]] = 1
        # for i, fasv in enumerate(fasvs):
        #     if fasv[2] == 'NOANNO':
        #         preds_turn[i] = act_based
        #     else:
        #         preds_turn[i][fasv[0]] = 1

        # # Compare sv to sv in frames
        # #   ref to the first appearance
        # #   o.w. ref to active frame
        # for i, fasv in enumerate(fasvs):
        #     frame_id = active_frame
        #     for frame in frames:
        #         found = False
        #         for sv in frame['slot_value']:
        #             is_equal = (fasv[2], fasv[3]) == sv
        #             if is_equal:
        #                 frame_id = frame['frame_id']
        #                 found = True
        #                 break
        #         if found:
        #             break
        #     preds_turn[i][frame_id] = 1

        preds.append(preds_turn)

    gen_dials = preds_to_dial_json(dials, preds)

    with open(str(Path(__file__).parent / \
        '../results/gen_frames.json'), 'w') as f_gen:
        json.dump(gen_dials, f_gen)


if __name__ == '__main__':
    # create_dictionaries(); exit()
    main()
    # split_dataset()
