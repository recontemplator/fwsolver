import os
from typing import List

DICT_FILE_ENV_KEY = 'DICT_FILE'
DICT_FILE_DEFAULT_PATH = 'dict/dict_filtered.txt'


# noinspection PyShadowingBuiltins
def list_possible_decompositions(words_with_masks, rows_count, print_fun=print):
    print = print_fun
    solutions = []
    cnt = [0, False]
    full_mask = (1 << (rows_count * rows_count)) - 1

    def recalculate_rarity(possible_now):
        rarity_map = {}
        for p in possible_now:
            for cell in p[3]:
                rarity_map[cell] = rarity_map.get(cell, 0) + 1
        for p in possible_now:
            p[4][0] = min(map(lambda c: rarity_map[c], p[3]))

    def list_decompositions_rec(possible_words_with_masks, already_used_words_masks, effective_mask, empty_cells, r):
        cnt[0] += 1
        if len(solutions) > 0 and cnt[0] > 9999:
            cnt[1] = True
            return
        if effective_mask == full_mask:
            solution_found = [(pw[0], pw[2], pw[3]) for pw in
                              sorted(already_used_words_masks, key=lambda t: (len(t[0]), t[0]), reverse=True)]
            # if len(solutions) == 0 or len(solution_found) < min(map(len, solutions)):
            #     print('Found solution in %d words' % len(solution_found))
            if not (solution_found in solutions):
                solutions.append(solution_found)
                #                cnt[2]=min(r,cnt[2])
        possible_mask = effective_mask
        for word_and_mask in possible_words_with_masks:
            possible_mask |= word_and_mask[1]
            if possible_mask == full_mask:
                break
        if possible_mask != full_mask:
            return

        already_used_words = set([used_wm[0] for used_wm in already_used_words_masks])
        # already_used_words=set()
        for idx, word_and_masks in enumerate(possible_words_with_masks):
            possible_now = [
                pwm for pwm in possible_words_with_masks[(idx + 1):]
                if (word_and_masks[1] | effective_mask) & pwm[1] == 0 and not (pwm[0] in already_used_words)]
            recalculate_rarity(possible_now)
            possible_now_sorted_by_rarity = sorted(possible_now, key=lambda p: p[4][0])
            # possible_now_sorted_by_rarity=sorted(possible_now,key=lambda p:-len(p[0]))
            list_decompositions_rec(
                possible_now_sorted_by_rarity,
                already_used_words_masks + [word_and_masks],
                effective_mask | word_and_masks[1],
                empty_cells - len(word_and_masks[0]), r + 1)

    words_with_masks_ext = []
    for wm in words_with_masks:
        positions_in_mask = [i for i in range(rows_count * rows_count) if ((1 << i) & wm[1]) != 0]
        words_with_masks_ext.append(wm + (positions_in_mask, [0]))

    list_decompositions_rec(words_with_masks_ext, [], 0, rows_count * rows_count, 1)
    # print 'rec fun called times,max_depth:',cnt
    other_cnt = 0
    result = []

    if len(solutions) > 0:
        solutions_sorted_by_wc: List[str] = sorted(solutions, key=len)
        result = solutions_sorted_by_wc[0]
        solutions_sorted_by_wc = [[t[0] for t in solution] for solution in solutions_sorted_by_wc]
        if len(solutions) == 1:
            print('Найдено, по моему, единственное решение:')
            print('  ', ', '.join(solutions_sorted_by_wc[0]))
        else:
            print(f'Найдено разных решений: {len(solutions)} ')
            if cnt[1]:
                print('но, скорее всего, их ещё больше.')
            # noinspection PyTypeChecker
            print('Один из вариантов:\n  %s' % ', '.join(solutions_sorted_by_wc[0]))
        # noinspection PyTypeChecker
        words_already_shown = set(solutions_sorted_by_wc[0])
        for solution in solutions_sorted_by_wc[1:]:
            # noinspection PyTypeChecker
            unseen_words = [w for w in solution if not (w in words_already_shown)]
            if len(unseen_words) == 0:
                other_cnt += 1
            else:
                print('Ещё вариант, включает: %s' % (', '.join(unseen_words)))
                words_already_shown.update(unseen_words)
        if other_cnt > 0:
            print('Есть ещё %d разных вариантов, из ранее названных слов' % other_cnt)
        # print('rec calls: %d' % cnt[0])
    else:
        if words_with_masks:
            print('Не могу найти полных решений, но вот несколькл слов, которые могли быть тут загаданы:')
            # noinspection PyTypeChecker
            print('  ', ', '.join(sorted(set(next(zip(*words_with_masks))), key=len, reverse=True)[:15]))
        else:
            print('Совсем не вижу знакомых слов.')
    return result


dominoes = [
    [(0, 0), (0, 1)],
    [(0, 0), (0, -1)],
    [(0, 0), (1, 0)],
    [(0, 0), (-1, 0)]
]


def init_dict():
    dict_file = os.environ[DICT_FILE_ENV_KEY] if DICT_FILE_ENV_KEY in os.environ else DICT_FILE_DEFAULT_PATH

    global words_hashed
    words_dict = {}
    with open(dict_file) as ff:
        for line in ff.readlines():
            word = line.strip('\n').replace('ё', 'е')
            key = word[:3]
            if key in words_dict:
                words_dict[key].append(word)
            else:
                words_dict[key] = [word]
    words_hashed = words_dict


words_hashed = None


def find_possible_words_hashed(board, rows_count):
    if words_hashed is None:
        init_dict()
    trominoes = []

    '''
    pref_ind list of positions of characters of possible words prefix on the map
    '''

    def words_by_prefix(pref_indices):
        return words_hashed.get(''.join([board[r + 1][c + 1] for r, c in pref_indices]), [])

    def find_possible_words_in_position(row, col, possible_words):
        for possible_word in possible_words:
            possible_positions = tuple(is_word_in_position(board, row + 1, col + 1, possible_word, rows_count))
            if len(possible_word) > 2 and len(possible_positions) > 0:
                v = possible_words_positions_pairs.get(possible_word, set())
                v.add((possible_positions, (row, col)))
                possible_words_positions_pairs[possible_word] = v
                # if possible_word in possible_words_positions_pairs:
                #     possible_words_positions_pairs[possible_word] = \
                #         list(set(possible_words_positions_pairs[possible_word] + (possible_positions, (row, col))))
                # else:
                #     possible_words_positions_pairs[possible_word] = set([(possible_positions, (row, col))])

    for d in dominoes:
        for d2 in dominoes:
            t = (d[1][0] + d2[1][0], d[1][1] + d2[1][1])
            if t != (0, 0):
                trominoes.append(d + [t])
                #    dominoes_and_trominoes=dominoes+trominoes
    dominoes_and_trominoes = trominoes

    possible_words_positions_pairs = {}
    for i in range(rows_count):
        for j in range(rows_count):
            for s in dominoes_and_trominoes:
                find_possible_words_in_position(i, j, words_by_prefix([(i + ii, j + jj) for ii, jj in s]))

    # print 'possible words/positions pairs',len(possible_words_positions_pairs)
    possible_words_sorted_by_len = []
    sorted_keys = sorted(possible_words_positions_pairs, reverse=True, key=len)
    for key in sorted_keys:
        for positions_and_start in possible_words_positions_pairs[key]:
            for position in positions_and_start[0]:
                possible_words_sorted_by_len.append((key, position, positions_and_start[1]))
    return possible_words_sorted_by_len


def is_word_in_position(board, i, j, word, rows_count):
    if len(word) == 0:
        return []  # should never happens
    if word[0] != board[i][j]:
        return []
    p = board[i][j]
    board[i][j] = '.'
    idx = (i - 1) * rows_count + (j - 1)
    res = int('0' * idx + '1' + '0' * (rows_count * rows_count - idx - 1), 2)

    if len(word) == 1:
        #        print_board()
        result = [res]
    else:
        result = \
            is_word_in_position(board, i - 1, j, word[1:], rows_count) + \
            is_word_in_position(board, i + 1, j, word[1:], rows_count) + \
            is_word_in_position(board, i, j - 1, word[1:], rows_count) + \
            is_word_in_position(board, i, j + 1, word[1:], rows_count)
        # update binary mask
        result = [res | r for r in result]
    board[i][j] = p
    return result


def solve_txt(rows):
    output = []
    rows_count = len(rows)
    # noinspection PyTypeChecker
    board = fill_the_board(rows, rows_count)

    possible_words_sorted_by_len = find_possible_words_hashed(board, rows_count)
    result = list_possible_decompositions(possible_words_sorted_by_len, rows_count,
                                          lambda *p: output.append(' '.join(str(pp) for pp in p)))
    return '\n'.join(output), result


def fill_the_board(rows, rows_count):
    board = [[' '] * (rows_count + 3) for _ in range(rows_count + 3)]

    for i in range(rows_count):
        for j in range(rows_count):
            board[i + 1][j + 1] = rows[i][j]
    return board


# def is_word_anywhere(board, word, rows_count):
#     result = []
#     for i in range(rows_count):
#         for j in range(rows_count):
#             result += is_word_in_position(board, i + 1, j + 1, word)
#     return list(set(result))


if __name__ == "__main__":
    print(solve_txt(['ачрбо', 'йвыок', 'омаро', 'нюрав', 'гварг']))
    # print(solve_txt(['бтксовтс', 'ойеашузе', 'рдогадаж', 'дбйкинму', 'ерездьтт', 'иббнелое', 'шатопуеп', 'усьледль']))
    # import timeit
    # print(solve_txt(['бтксовтс', 'ойеашузе', 'рдогадаж', 'дбйкинму', 'ерездьтт', 'иббнелое', 'шатопуеп', 'усьледль']))
    # print(timeit.timeit(
    #     stmt="solve_txt(['бтксовтс', 'ойеашузе', 'рдогадаж', "
    #          "'дбйкинму', 'ерездьтт', 'иббнелое', 'шатопуеп', 'усьледль'])",
    #     setup="from __main__ import solve_txt", number=10))
    # print(solve_txt(['мохог', 'ткарл', 'ечазе', 'нжукм', 'ищета']))
