import random
import json
from pprint import pprint
from parse_multiwoz import load_data, data_dir, parse_turns


def mix_turns(lens, rng=None, orders=None, mixables=None):
    """ Mix turns given length of dialogues.
    This is the core function of mixing dialogues.
    lens: list[int]. The length of each dialogues.
    rng: Seeded python rng.
    orders: list[int] or list[char] or string. A predefined order to
            mix the dialogues, each entry indicates the dialogue which
            the turn comes from.
    mixables: list[list[int]] or list[iterator]. Information about the
              mixablility of each turn in each dialogue. 1 means mixable
              and 0 means not mixable.
              TODO: use a term more precise than "mixable"?
    
    return: iterator of (dialgue id, turn id).
    """

    def interval_iterator(it, length=None):
        """ Count the distance between 1s.
        ex: output of [1, 0, 1, 1, 0, 0] should be iter([2, 1, 3]).
        """
        last = 0
        for i, val in enumerate(it):
            if i == 0:
                assert val, (i, val, list(it))

            if val:
                if i != last:
                    yield i - last
                last = i

        if length is not None:
            assert i + 1 == length, (length, i + 1, list(it))

        yield i + 1 - last

    turns = [iter(range(length)) for length in lens]
    index = list(range(len(lens)))

    if orders is not None:
        orders = iter(list(orders))

    if mixables is not None:
        mixables = [interval_iterator(turns) for turns in mixables]
    else:
        mixables = [iter([length]) for length in lens]

    if rng is None:
        rng = random
        seed = rng.randrange(97)
        rng.seed(seed)
        print('seed = {}'.format(seed))
        # rng.seed(0)

    last_dial_id = -1
    while True:
        # Get dialogue index.
        if orders is None and index:
            dial_id = rng.choice(index)
            # Make sure the mixing is interesting.
            while dial_id == last_dial_id and len(index) > 1:
                dial_id = rng.choice(index)
        elif orders is not None:
            try:
                dial_id = int(next(orders))
            except StopIteration:
                break
        else:
            break

        last_dial_id = dial_id

        # Yield a batch of turns, stop at next mixable turn.
        try:
            count = next(mixables[dial_id])
            for _ in range(count):
                turn_id = next(turns[dial_id])
                yield (dial_id, turn_id)
        except StopIteration:
            index.remove(dial_id)


def generate_mixed_dataset():
    raw_dials = load_data()

    # Get all domains.
    domains = set()
    for dial_id in raw_dials.keys():
        for key in raw_dials[dial_id]['goal'].keys():
            if key != 'message' and key != 'topic':
                domains.add(key)

    # Separate single and multi domain dials.
    single_dial_ids = []
    mul_dial_ids = []

    for dial_id in raw_dials.keys():
        if 'SNG' in dial_id or 'WOZ' in dial_id:
            single_dial_ids.append(dial_id)
        elif 'MUL' in dial_id:
            mul_dial_ids.append(dial_id)
        else:
            assert False, dial_id
    single_dial_ids.sort()
    mul_dial_ids.sort()

    # Group dials by domain.
    domain_dial_ids = {}
    for dial_id in single_dial_ids:
        cnt = 0
        for domain in domains:
            if raw_dials[dial_id]['goal'].get(domain, {}):
                ids = domain_dial_ids.get(domain, [])
                ids.append(dial_id)
                domain_dial_ids[domain] = ids
                cnt += 1
        assert cnt == 1, raw_dials[dial_id]['goal']

    # Get mixable dials (with mixable turn in the middle).
    domain_mixable_dial_ids = {}
    for domain, dial_ids in domain_dial_ids.items():
        ids = []
        for dial_id in dial_ids:
            turns = raw_dials[dial_id]['log']
            mixables, _, _ = zip(*parse_turns(turns))
            # Check > 1 because the first turn is always set to mixable.
            if sum(mixables) > 1:
                ids.append(dial_id)
        domain_mixable_dial_ids[domain] = ids

    # Create a list of mixing recipes.
    # Option: every possible pairs.
    # Option: mix with other k dialogues. (k ~ 10)

    # # Single domain mixture.
    # recipes = []
    # n_appearance = 10
    # for _, dial_ids in domain_mixable_dial_ids.items():
    #     for i, dial_id in enumerate(dial_ids):
    #         for j in range(i + 1, min(len(dial_ids),
    #                                   i + n_appearance + 1)):
    #             recipes.append((dial_id, dial_ids[j]))

    # Multi domain mixture.
    recipes = []
    n_appearance = 20
    hotel_ids = domain_mixable_dial_ids['hotel']
    restaurant_ids = domain_mixable_dial_ids['restaurant']
    train_ids = domain_mixable_dial_ids['train']
    taxi_ids = domain_mixable_dial_ids['taxi']

    for i, hotel_id in enumerate(hotel_ids):
        for j in range(n_appearance):
            recipes.append(
                (hotel_id, restaurant_ids[(i + j) % len(restaurant_ids)])) 

    # transport_ids = train_ids + taxi_ids
    # for i, hotel_id in enumerate(hotel_ids):
    #     for j in range(n_appearance):
    #         recipes.append(
    #             (hotel_id, transport_ids[(i + j) % len(transport_ids)]))

    # Mix.
    mixed_dataset = []
    for dial_ids in recipes:
        n_turns = []
        mixables = []
        for dial_id in dial_ids:
            dial = raw_dials[dial_id]
            n_turns.append(len(dial['log']) // 2)
            mixable, _, _ = zip(*parse_turns(dial['log']))
            mixables.append(iter(mixable))

        random.seed(0)
        mixed_turns = mix_turns(n_turns, rng=random, mixables=mixables)

        mixed_dataset.append({
            'dials': dial_ids,
            'turns': list(mixed_turns),
        })

    # Save.
    # with open(str(data_dir / 'mixed_multiwoz.json'), 'w') as f_out:
    with open(str(data_dir / 'mixed_hotel_restaurant_multiwoz.json'),
              'w') as f_out:
    # with open(str(data_dir / 'mixed_hotel_transport_multiwoz.json'),
    #           'w') as f_out:
        json.dump(mixed_dataset, f_out)


def split_dataset(filename):
    import random
    random.seed(0)

    with open(str(data_dir / filename), 'r') as f_data:
        mixed_dataset = json.load(f_data)

    random.shuffle(mixed_dataset)

    train_ratio = 0.8
    n_train = int(len(mixed_dataset) * train_ratio)
    n_valid = len(mixed_dataset) - n_train

    print('total = {}'.format(n_train + n_valid))
    print('# train = {}'.format(n_train))
    print('# valid = {}'.format(n_valid))

    train_dials = mixed_dataset[:n_train]
    valid_dials = mixed_dataset[n_train:]

    with open(str(data_dir / ('train_' + filename)), 'w') as f_data:
        json.dump(train_dials, f_data)
    with open(str(data_dir / ('valid_' + filename)), 'w') as f_data:
        json.dump(valid_dials, f_data)


if __name__ == '__main__':
    pass
    # generate_mixed_dataset()
    # split_dataset('mixed_hotel_transport_multiwoz.json')

