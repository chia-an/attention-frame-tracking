import random
import json
from pathlib import Path
from pprint import pprint
from torch.utils.data import Dataset
from copy import deepcopy


bool_slots = ['parking', 'internet']
special_slots = ['stars', 'poeple', 'stay']


data_dir = Path(__file__).parent / '../../data/MULTIWOZ2 2'

raw_dials_path = data_dir / 'data.json'


class MultiwozDataset(Dataset):
    def __init__(self, dicts,
                       data_filename='train_mixed_multiwoz.json',
                       subsample=None,
                       pad_frame=False,
                       tokenizer='trigram'):
        import torch
        from utils.dictionary import trigram_iterator

        if tokenizer.startswith('bert'):
            from pytorch_pretrained_bert import BertTokenizer
            bert_tknzr = BertTokenizer.from_pretrained(tokenizer)
        elif tokenizer in ['trigram', 'word']:
            from nltk.tokenize import TweetTokenizer
            tknzr = TweetTokenizer()
        else:
            raise Exception('Unknown tokenizer {}.'.format(tokenizer))

        def tokenize(sent):
            if tokenizer == 'trigram':
                return [torch.tensor([tri_to_index[tri]
                                      for tri in trigram_iterator(word)])
                        for word in tknzr.tokenize(sent)]
            elif tokenizer == 'word':
                return [word_to_index[word] for word in tknzr.tokenize(sent)]
            elif tokenizer.startswith('bert'):
                tokenized_text = bert_tknzr.tokenize(
                    ' '.join(['[CLS]', sent, '[SEP]']))
                indexes = bert_tknzr.convert_tokens_to_ids(tokenized_text)
                return torch.tensor(indexes)
            else:
                raise Exception('Unknown tokenizer {}.'.format(tokenizer))

        word_to_index, index_to_word, \
        tri_to_index, index_to_tri, \
        act_to_index, index_to_act, \
        slot_to_index, index_to_slot = dicts

        raw_dials = load_data()

        with open(str(data_dir / data_filename), 'r') as f_data:
            mixed_dials = json.load(f_data)

        # Turn samples into tensors
        self.data = []
        for mixed_dial in mixed_dials:
            dials = [raw_dials[dial_id] for dial_id in mixed_dial['dials']]
            for sample in mixed_dialogue_samples(dials, mixed_dial['turns']):
                sent, fasvs, frames, active_frame, new_frames = sample

                sent = tokenize(sent)

                fs = []
                asvs = []
                for f, a, s, v in fasvs:
                    fs.append(f)
                    asvs.append((act_to_index[a],
                                 slot_to_index[s],
                                 tokenize(v)))
                fs = torch.tensor(fs)

                frames = [[(slot_to_index[s],
                            tokenize(v))
                           for s, v in frame]
                          for frame in frames]
                if pad_frame:
                    null_sv = (slot_to_index['NOANNO'], [tokenize('NULL')])
                    for i, frame in enumerate(frames):
                        n_repeat = 32 - len(frame)
                        frames[i] = frame + [null_sv] * n_repeat
                for frame in frames:
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
                    return

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]


def dsv_iterator(metadata):
    """ Iterate through a metadata json.
    """
    for domain in metadata.keys():
        for slot in sorted(metadata[domain]['book'].keys()):
            value = metadata[domain]['book'][slot]
            if slot != 'booked':
                yield (domain, slot, value)
            else:
                for booked in value:
                    for booked_slot in sorted(booked.keys()):
                        booked_value = booked[booked_slot]
                        yield (domain, booked_slot, booked_value)
        for slot in sorted(metadata[domain]['semi'].keys()):
            value = metadata[domain]['semi'][slot]
            yield (domain, slot, value)


def parse_turns(turns, verbose=False):
    """ Extract information of each turn.
    turns: list of turns with the same format as in MultiWOZ.

    Yield for each turn a tuple (is_mixable, state, slot_values).
    is_mixable: a boolean flag.
    state: dict[(domain, slot)] = value, the state/frame of
           this turn.
    dsvs: list[(domain, slot, value)],
          all dsv appear in this turn.

    There are three types of frame reference: mention slot value,
    use coreference, implicit reference. A turn is mixable if it
    contains only frame reference with slot value mention and no
    other types.

    A slot value mention is detected
    if a domain-slot-value tuple appears in the utterance and
       the tuple appears already in previous turns.

    NOTE:
    - There may be some coreference/anaphora in the utterance.
    - Special case: the first turn should always be mixable.
    - Sometimes a d-s-v may disappear in the metadata, which is weird.
    - Mixable turns are called identifying turns in the report.
    """
    n_turns = len(turns) // 2
    last_state = {}

    for turn_id in range(n_turns):
        u_turn = turns[turn_id * 2]
        s_turn = turns[turn_id * 2 + 1]
        state = {}
        dsvs = []
        is_mixable = False
        is_new_frame = False

        # There are some errors in the label.
        if u_turn['metadata']:
            return

        for dsv in dsv_iterator(s_turn['metadata']):
            domain, slot, value = dsv
            not_null = bool(value) and value != 'not mentioned'

            # if verbose:
            #     print('not null = {}, dsv = {}'.format(not_null, dsv))

            if not not_null:
                continue

            # NOTE: "in text" is a strict criteria
            appear_value = str(value) in u_turn['text']
            appear_slot = slot in u_turn['text']
            appear = appear_value and appear_slot \
                        if slot in special_slots else \
                     appear_slot if slot in bool_slots else \
                     appear_value

            is_new_slot = (domain, slot) not in last_state.keys()
            is_new_value = is_new_slot or value != last_state[(domain, slot)]

            # (slot, value)
            # (new, new) = fill new slot
            # (new, old) = impossible
            # (old, new) = change slot value
            # (old, old) = mention slot value if appear = true

            is_mixable = is_mixable or (appear and not is_new_value)
            is_new_frame = is_new_frame or is_new_value

            state[(domain, slot)] = value

            if appear or is_new_value:
                dsvs.append((domain, slot, value))

            # if verbose:
            #     print('\n'.join(['dsv = ' + str((domain, slot, value)),
            #                      'not null = ' + str(not_null),
            #                      'appear_value = ' + str(appear_value),
            #                      'appear_slot = ' + str(appear_slot),
            #                      'appear = ' + str(appear),
            #                      'is new slot = ' + str(is_new_slot),
            #                      'is new value = ' + str(is_new_value)]))

        if verbose:
            print('\n'.join([u_turn['text'],
                             'mixable = ' + str(is_mixable),
                             'new frame = ' + str(is_new_frame),
                             'slot values = ' + str(dsvs),
                             'state = ' + str(state)]))
            pprint(s_turn['metadata'])

        last_state = state.copy()

        yield (is_mixable if turn_id else True,
               state if is_new_frame else None,
               dsvs)


def mixed_dialogue_samples(dials,
                           mixed_turns=None,
                           user_only=True,
                           text_position=False):
    """ Mix dialogues and transform it to training samples.
    dials: list of dialogues in MultiWOZ format.
    mixed_turns: list[(dial_id, turn_id)], the result of mix_turns.

    Yield a training sample for each turn of the mixed dialogue.
    """
    n_turns = [len(dial['log']) // 2 for dial in dials]
    mixables = []
    states = []
    dsvss = []
    for dial in dials:
        mixable, state, dsvs = zip(*parse_turns(dial['log']))
        mixables.append(iter(mixable))
        states.append(iter(state))
        dsvss.append(iter(dsvs))

    if mixed_turns is None:
        from mix_multiwoz import mix_turns
        random.seed(0)
        mixed_turns = mix_turns(n_turns, rng=random, mixables=mixables)

    frames = []
    last_frame = []
    dsv_to_f = {}
    active_frame = -1
    for index, turn_id in mixed_turns:
        dial = dials[index]

        sent = dial['log'][turn_id * 2]['text']

        state = next(states[index])
        dsvs = next(dsvss[index])
        fasvs = []
        new_frame = []

        if state is not None:
            # Assume parent frame is the last frame.
            frame = []
            for (d, s), v in sorted(state.items()):
                s = d + '-' + s
                if text_position:
                    found = False
                    for ps, pv in last_frame:
                        if (s, v) == (ps, pv[0]):
                            v = pv
                            found = True
                    if not found:
                        index = sent.lower().find(v.lower())
                        if index == -1:
                            index = None
                        v = (v, turn_id * 2, index)
                frame.append((s, v))
            last_frame = deepcopy(frame)

            frames.append(frame)
            active_frame = len(frames) - 1
            new_frame.append(active_frame)

        for d, s, v in dsvs:
            f = dsv_to_f.get((index, d, s, v), active_frame)

            v2 = v
            if text_position:
                index = sent.lower().find(v.lower())
                if index == -1:
                    index = None
                v2 = (v, turn_id * 2, index)
            fasvs.append((f, 'NULL', d + '-' + s, v2))

            dsv_to_f[(index, d, s, v)] = f

        yield (sent, fasvs, deepcopy(frames), active_frame, new_frame)

        if not user_only:
            sent = dial['log'][turn_id * 2 + 1]['text']
            fasvs = []
            new_frame = []
            yield (sent, fasvs, deepcopy(frames), active_frame, new_frame)


def load_data():
    with open(str(raw_dials_path), 'r') as raw_dial_f:
        raw_dials = json.load(raw_dial_f)

    return raw_dials


#####
# Dictionaries
#####


def text_dictionaries(dials):
    from nltk.tokenize import TweetTokenizer
    from dictionary import trigram_iterator

    tknzr = TweetTokenizer()

    word_to_index = {}
    index_to_word = {}
    tri_to_index = {}
    index_to_tri = {}

    def add_word(w):
        if w not in word_to_index:
            index = len(word_to_index) + 1
            word_to_index[w] = index
            index_to_word[index] = w

    for dial_id, dial in sorted(dials.items()):
        for turn in dial['log']:
            for w in tknzr.tokenize(turn['text']):
                add_word(w)

            for _, _, v in dsv_iterator(turn['metadata']):
                for w in tknzr.tokenize(v):
                    add_word(w)

    for w in word_to_index.keys():
        for trigram in trigram_iterator(w):
            if trigram not in tri_to_index:
                index = len(tri_to_index) + 1
                tri_to_index[trigram] = index
                index_to_tri[index] = trigram

    return word_to_index, index_to_word, \
           tri_to_index, index_to_tri


def slot_dictionary(dials):
    slot_to_index = {}
    index_to_slot = {}

    for dial_id, dial in sorted(dials.items()):
        for turn in dial['log']:
            for d, s, _ in dsv_iterator(turn['metadata']):
                slot = d + '-' + s
                if slot not in slot_to_index:
                    index = len(slot_to_index) + 1
                    slot_to_index[slot] = index
                    index_to_slot[index] = slot

    return slot_to_index, index_to_slot

 
def create_dictionaries():
    # Load datset
    dials = load_data()

    word_to_index, index_to_word, \
    tri_to_index, index_to_tri = text_dictionaries(dials)
    slot_to_index, index_to_slot = slot_dictionary(dials)

    with open(str(data_dir / 'word_to_index.json'), 'w') as f_dict:
        json.dump(word_to_index, f_dict)
    with open(str(data_dir / 'index_to_word.json'), 'w') as f_dict:
        json.dump(index_to_word, f_dict)
    with open(str(data_dir / 'tri_to_index.json'), 'w') as f_dict:
        json.dump(tri_to_index, f_dict)
    with open(str(data_dir / 'index_to_tri.json'), 'w') as f_dict:
        json.dump(index_to_tri, f_dict)
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


if __name__ == '__main__':
    pass
    # create_dictionaries()


