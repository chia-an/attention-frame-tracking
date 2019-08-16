SOW = '\x00'
EOW = '\x01'


def trigram_iterator(word):
    word = SOW + word + EOW
    for i in range(0, len(word) - 2):
        yield word[i: i+3]


def merge(ti1, it1, ti2, it2):
    t_to_i = ti1.copy()
    i_to_t = it1.copy()

    for token in sorted(ti2.keys()):
        if token not in t_to_i:
            index = len(t_to_i) + 1
            t_to_i[token] = index
            i_to_t[index] = token

    # print('ori 1 size = {}'.format(len(ti1)))
    # print('ori 2 size = {}'.format(len(ti2)))
    # print('comb size = {}'.format(len(t_to_i)))

    return t_to_i, i_to_t

