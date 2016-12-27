#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Dec 13 18:01:49 2016

@author: zy
"""
import numpy as np
import pickle
import gzip
import sys
import cv2

DICT_FILE = 'dict/dict_filtered.txt'
LETTERS_DAT_FILE = 'letters.dat'
# PUZZLE_FILE='/Users/zy/Downloads/IMG_3943.PNG'
# check 3944 for errors
# PUZZLE_FILE='unrecognized_shapes.png'
# PUZZLE_FILE='/Users/zy/Downloads/IMG_3963.PNG'#bad case
PUZZLE_FILE = '/Users/zy/Downloads/IMG_3968_.PNG'  # bad case

DEFAULT_BOARD_SIZE = (1040, 1040)
DEFAULT_BOARD_OFFSET = (42, 498)
DEFAULT_BOARD_DIM = (3, 3)
# need to be resolution dependent to increase robustness
CELL_SIZE_TOLLERANCE = 15

kernel = np.ones((5, 5), np.uint8)


def is_img_the_same(i1, i2):
    return not cv2.morphologyEx(cv2.absdiff(i1, cv2.resize(i2, i1.shape[1::-1])), cv2.MORPH_OPEN, kernel).any()


def group_values(sorted_values, threshold=10):
    if len(sorted_values) == 0:
        return sorted_values
    values_grouped = []
    last_group = [sorted_values[0]]
    for idx, d in enumerate(np.diff(sorted_values)):
        if d < threshold:
            last_group.append(sorted_values[idx + 1])
        else:
            values_grouped.append(np.mean(last_group))
            last_group = [sorted_values[idx + 1]]
    values_grouped.append(np.mean(last_group))
    return values_grouped


def runs_of_ones_array(bits):
    # make sure all runs of ones are well-bounded
    bounded = np.hstack(([0], bits, [0]))
    # get 1 at run starts and -1 at run ends
    difs = np.diff(bounded)
    run_starts, = np.where(difs > 0)
    run_ends, = np.where(difs < 0)
    # noinspection PyRedundantParentheses
    return (run_starts, run_ends - run_starts)


def detect_board(img):
    lines = cv2.HoughLines(img, 1, np.pi / 2, int(min(img.shape[:2]) * 0.7))
    left, top = DEFAULT_BOARD_OFFSET
    right = DEFAULT_BOARD_OFFSET[0] + DEFAULT_BOARD_SIZE[0]
    bottom = DEFAULT_BOARD_OFFSET[1] + DEFAULT_BOARD_SIZE[1]
    cols, rows = DEFAULT_BOARD_DIM
    # cv2.imshow('crop',cv2.resize(img,(512,512))),   cv2.waitKey()

    if lines is None or len(lines) < 6:  # something go wrong, return some defaults
        return left, top, right, bottom, cols, rows

    lines = lines[:, 0, :]
    # print 'lines detected:'
    vert_lines_sorted = np.sort(lines[np.abs(lines[:, 1]) < 1e-3, 0])
    #    canvas=cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
    #    for l in vert_lines_sorted:
    #        cv2.line(canvas,(l,-2000),(l,2000),(0,0,255),5)
    vert_lines_grouped = group_values(vert_lines_sorted)
    #    for l in vert_lines_grouped:
    #        cv2.line(canvas,(l,-2000),(l,2000),(255,0,0),4)
    #        print l
    #    print '----'
    #    for l in np.abs(np.diff(np.diff(vert_lines_grouped))):
    #        print l
    #    cv2.imshow('canvas2',cv2.resize(canvas,(img.shape[1]/4,img.shape[0]/4))),   cv2.waitKey()

    vruns = runs_of_ones_array(np.abs(np.diff(np.diff(vert_lines_grouped))) < CELL_SIZE_TOLLERANCE)
    if len(vruns[1]) == 0:
        return left, top, right, bottom, cols, rows

    vidxmax = np.argmax(vruns[1])
    left = int(vert_lines_grouped[vruns[0][vidxmax]])
    cols = vruns[1][vidxmax] + 1
    right = int(vert_lines_grouped[vruns[0][vidxmax] + cols])

    hor_lines_sorted = np.sort(lines[np.abs(lines[:, 1] - np.pi / 2) < 1e-3, 0])
    hor_lines_grouped = group_values(hor_lines_sorted)
    #    for l in hor_lines_grouped:
    #        cv2.line(canvas,(-2000,l),(2000,l),(255,0,0),4)
    #        print l
    #    print '----'
    #    for l in np.abs(np.diff(np.diff(hor_lines_grouped))):
    #        print l
    # cv2.imshow('canvas2',cv2.resize(canvas,(img.shape[1]/4,img.shape[0]/4))),   cv2.waitKey()

    hruns = runs_of_ones_array(np.abs(np.diff(np.diff(hor_lines_grouped))) < CELL_SIZE_TOLLERANCE)
    if len(hruns[1]) == 0:
        return left, top, right, bottom, cols, rows

    hidxmax = np.argmax(hruns[1])
    rows = hruns[1][hidxmax] + 1
    glitch = 0
    if rows == 5 and cols == 4:
        # special case. top menu has the same height
        # as cell height
        rows = 4
        glitch = 1
    top = int(hor_lines_grouped[hruns[0][hidxmax] + glitch])
    bottom = int(hor_lines_grouped[hruns[0][hidxmax] + rows + glitch])
    return left, top, right, bottom, cols, rows


def scan_screen_image(img_name, ground_truth=''):
    img = cv2.imread(img_name, 0)
    ret3, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    left, top, right, bottom, cols, rows = detect_board(img)
    if rows != cols:
        print 'square board expected. %sx%s board detected.' % (rows, cols)
        return None
    img = img[top:bottom, left:right]
    # print left,top,right,bottom,cols,rows,top-bottom,right-left,img.shape

    if bottom - top != img.shape[0] or right - left != img.shape[1]:
        print 'Unable to extract board from image'
        return None
    # print left,top,right,bottom,cols,rows
    # cv2.imshow('crop',img),   cv2.waitKey()
    # noinspection PyBroadException
    # it is really doesn`t matter why we cannot load the OCR model, just discard it
    try:
        with gzip.open(LETTERS_DAT_FILE, 'r') as f:
            patterns = pickle.load(f)
    except:
        patterns = {}
        print 'Shape database in letters.dat has not been found or it is corrupted.\
Empty shape database has been initialized.'
    # for k in patterns:
    #        print k,
    #    print
    #    print len(patterns),'фэъ'
    result = []
    unknown_shapes = []
    for i in range(rows):
        row = []
        for j in range(cols):
            roi = img[
                  i * (bottom - top) / rows + 20:(i + 1) * (bottom - top) / rows - 20,
                  j * (right - left) / cols + 20:(j + 1) * (right - left) / cols - 20
                  ]

            b = np.argwhere(roi)
            (ystart, xstart), (ystop, xstop) = b.min(0), b.max(0) + 1
            # im2, contours, hierarchy = cv2.findContours(roi.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
            # x, y, w, h = cv2.boundingRect(contours[0])
            letter = '*'
            for k in patterns:
                if is_img_the_same(patterns[k], roi[ystart:ystop, xstart:xstop]):
                    if letter != '*':
                        print 'warn: letters collision:', letter, k
                    letter = k
            if letter == '*':
                collisions = np.argwhere(map(
                    lambda shape: is_img_the_same(shape, roi[ystart:ystop, xstart:xstop]),
                    unknown_shapes))
                if len(collisions) == 0:
                    unknown_shapes.append(roi[ystart:ystop, xstart:xstop])
                elif len(collisions) > 1:
                    print 'warn: shapes collision'
            row.append(letter)
        result.append(''.join(row).replace(u'ё', u'е'))
    if len(unknown_shapes) > 0:
        if len(ground_truth) == len(unknown_shapes):
            print '%d never seen shapes were found on "%s" image, \
groundtruth letters were provided for exactly %d shapes.\
 Shapes database has been updated. This change will take\
 effect the next time the script is started.' % \
                  (len(unknown_shapes), img_name, len(ground_truth))
            for i, l in enumerate(unknown_shapes):
                patterns[ground_truth[i]] = l
            with gzip.open(LETTERS_DAT_FILE, 'wb') as f:
                pickle.dump(patterns, f, 2)
        else:
            print '%d never seen shapes were found on "%s" image, \
groundtruth letters were provided for %d shapes. \
Shapes database has NOT been updated. Exactly %d groundtruth \
letters need to be provided to update the shapes database. \
Shapes are shown in unrecognized_shapes.png' % \
                  (min(len(unknown_shapes), 20), img_name,
                   len(ground_truth), min(len(unknown_shapes), 20))
            shapes_cumm_widths = \
                np.hstack((
                    [0],
                    np.cumsum([s.shape[1] + 10 for s in unknown_shapes])))
            max_shapes_heights = max(s.shape[0] for s in unknown_shapes)
            canvas = np.zeros(
                (max_shapes_heights + 10, shapes_cumm_widths[-1]), np.uint8)
            for i, l in enumerate(unknown_shapes):
                h, w = l.shape
                canvas[5:5 + h, 5 + shapes_cumm_widths[i]:5 + shapes_cumm_widths[i] + w] = l[:, :]
            cv2.imwrite('unrecognized_shapes.png', canvas)

    return result


def print_board():
    print_board_masked((1 << (rows_count * rows_count)) - 1)


def print_board_masked(mask):
    for i in range(rows_count):
        for j in range(rows_count):
            print board[i + 1, j + 1] if (
                                             1 << rows_count * rows_count - (
                                                 (i * rows_count) + j) - 1) & mask > 0 else '.',
        print


# print repr('\n  '.join([(''.join(l[1:-1])) for l in board[1:-1]])).decode("unicode-escape")

def is_word_in_position(i, j, word):
    if len(word) == 0:
        return []  # should never happens
    if word[0] != board[i, j]:
        return []
    p = board[i, j]
    board[i, j] = '.'
    idx = (i - 1) * rows_count + (j - 1)
    res = int('0' * idx + '1' + '0' * (rows_count * rows_count - idx - 1), 2)

    if len(word) == 1:
        #        print_board()
        result = [res]
    else:
        result = \
            is_word_in_position(i - 1, j, word[1:]) + \
            is_word_in_position(i + 1, j, word[1:]) + \
            is_word_in_position(i, j - 1, word[1:]) + \
            is_word_in_position(i, j + 1, word[1:])
        # update binary mask
        result = [res | r for r in result]
    board[i, j] = p
    return result


def is_word_anywhere(word):
    result = []
    for i in range(rows_count):
        for j in range(rows_count):
            result += is_word_in_position(i + 1, j + 1, word)
    return list(set(result))


min_empty_cells = 1e9


def list_posible_decompositions(words_with_masks):
    solutions = []
    cnt = [0, False]
    full_mask = (1 << (rows_count * rows_count)) - 1

    def recalculate_rarity(possible_now):
        rarity_map = {}
        for p in possible_now:
            for cell in p[2]:
                rarity_map[cell] = rarity_map.get(cell, 0) + 1
        for p in possible_now:
            p[3][0] = min(map(lambda c: rarity_map[c], p[2]))

    def list_decompositions_rec(possible_words_with_masks, already_used_words_masks, effective_mask, empty_cells, r):
        cnt[0] += 1
        if len(solutions) > 0 and cnt[0] > 9999:
            cnt[1] = True
            return
        if effective_mask == full_mask:
            solution_found = [pw[0] for pw in sorted(already_used_words_masks, key=len, reverse=True)]
            if len(solutions) == 0 or len(solution_found) < min(map(len, solutions)):
                print 'Found solution in %d words' % len(solution_found)
            if not (solution_found in solutions):
                solutions.append(solution_found)
                #                cnt[2]=min(r,cnt[2])
        possible_mask = effective_mask
        for word_and_mask in possible_words_with_masks:
            possible_mask |= word_and_mask[1]
        if possible_mask != full_mask:
            return
        # if (cnt[0] > 8e6):
        #     print_board_masked(effective_mask)
        #     print '^^^^^^^%d^^^^^^(%d),(%s)' % (
        #         cnt[0], len(solutions), ' '.join([pw[0] for pw in possible_words_with_masks]))
        # global min_empty_cells
        #        if empty_cells<min_empty_cells:
        #            min_empty_cells=empty_cells
        #  wmsk=[(t[0],msk) for t in words_with_masks for msk in t[1] ]
        #  print_board_masked([t for t in wmsk if t[1]&1<<(8*3+3)>0][1][1])
        #    print 'list_decompositions_rec==='
        #    print 'possible:', repr(possible_words_with_masks).decode("unicode-escape"),
        #     print 'already:', repr(already_used_words_masks).decode("unicode-escape"),bin(effective_mask)
        #    print 'list_decompositions_rec==='
        #        print_board_masked(effective_mask)
        #        print 'already:', repr(already_used_words_masks).decode("unicode-escape")

        already_used_words = set([used_wm[0] for used_wm in already_used_words_masks])
        # already_used_words=set()
        for idx, word_and_masks in enumerate(possible_words_with_masks):
            possible_now = [
                pwm for pwm in possible_words_with_masks[(idx + 1):]
                if (word_and_masks[1] | effective_mask) & pwm[1] == 0 and not (pwm[0] in already_used_words)]
            recalculate_rarity(possible_now)
            posible_now_sorted_by_rarity = sorted(possible_now, key=lambda p: p[3][0])
            # posible_now_sorted_by_rarity=sorted(possible_now,key=lambda p:-len(p[0]))
            list_decompositions_rec(
                posible_now_sorted_by_rarity,
                already_used_words_masks + [word_and_masks],
                effective_mask | word_and_masks[1],
                empty_cells - len(word_and_masks[0]), r + 1)

    words_with_masks_ext = []
    for wm in words_with_masks:
        positions_in_mask = [i for i in range(rows_count * rows_count) if ((1 << i) & wm[1]) != 0]
        words_with_masks_ext.append(wm + (positions_in_mask, [0]))

    list_decompositions_rec(words_with_masks_ext, [], 0L, rows_count * rows_count, 1)
    # print 'rec fun called times,max_depth:',cnt
    other_cnt = 0
    if len(solutions) > 0:
        print '%d different solutions found' % len(solutions)
        solutions_sorted_by_wc = sorted(solutions, key=len)
        if cnt[1]:
            print '* not all possible solutions were enumerated'
        print '[One of] shortest solution(%d words):\n%s' % \
              (len(solutions_sorted_by_wc[0]), ' '.join(solutions_sorted_by_wc[0]))
        words_already_shown = set(solutions_sorted_by_wc[0])
        for solution in solutions_sorted_by_wc[1:]:
            unseen_words = [w for w in solution if not (w in words_already_shown)]
            if len(unseen_words) == 0:
                other_cnt += 1
            else:
                print 'alternative (%d words), including: %s' \
                      % (len(solution), ' '.join(unseen_words))
                words_already_shown.update(unseen_words)
        if other_cnt > 0:
            print 'And %d other solutions, with words already mentioned' % \
                  other_cnt
        print 'rec calls:', cnt[0]
    else:
        print 'Solutions have not been found.'

dominos = [
    [(0, 0), (0, 1)],
    [(0, 0), (0, -1)],
    [(0, 0), (1, 0)],
    [(0, 0), (-1, 0)]
]


def find_possible_words_hashed():
    trominos = []

    '''
    pref_ind list of positions of characters of possible words prefix on the map
    '''
    def words_by_prefix(pref_indices):
        return words_hashed.get(''.join([board[r + 1, c + 1] for r, c in pref_indices]), [])

    def find_possible_words_in_position(row, col, possible_words):
        for possible_word in possible_words:
            possible_positions = is_word_in_position(row + 1, col + 1, possible_word)
            if len(possible_word) > 2 and len(possible_positions) > 0:
                if possible_word in possible_words_positions_pairs:
                    possible_words_positions_pairs[possible_word] = \
                        list(set(possible_words_positions_pairs[possible_word] + possible_positions))
                else:
                    possible_words_positions_pairs[possible_word] = possible_positions

    for d in dominos:
        for d2 in dominos:
            t = (d[1][0] + d2[1][0], d[1][1] + d2[1][1])
            if t != (0, 0):
                trominos.append(d + [t])
                #    dominos_and_trominos=dominos+trominos
    dominos_and_trominos = trominos
    import codecs
    words_hashed = {}
    with codecs.open(DICT_FILE, encoding='utf-8') as f:
        for line in f.readlines():
            word = unicode(line.strip('\n')).replace(u'ё', u'е')
            key = word[:3]
            if key in words_hashed:
                words_hashed[key].append(word)
            else:
                words_hashed[key] = [word]

    possible_words_positions_pairs = {}
    for i in range(rows_count):
        for j in range(rows_count):
            for s in dominos_and_trominos:
                find_possible_words_in_position(i, j, words_by_prefix([(i + ii, j + jj) for ii, jj in s]))

    # print 'possible words/positions pairs',len(possible_words_positions_pairs)
    possible_words_sorted_by_len = []
    sorted_keys = sorted(possible_words_positions_pairs, reverse=True, key=len)
    for key in sorted_keys:
        for position in possible_words_positions_pairs[key]:
            possible_words_sorted_by_len.append((key, position))
    return possible_words_sorted_by_len

# with open('possible_words.dat','wb') as f:
#        pickle.dump(possible_words_sorted_by_len, f)


def find_possible_words():
    import codecs

    with codecs.open('dict.txt', encoding='utf-8') as f:
        words = [unicode(line.strip('\n')) for line in f.readlines()]
    possible_words = []
    for word in words:
        possible_positions = is_word_anywhere(word)
        if len(word) > 2 and len(possible_positions) > 0:
            possible_words.append((word, possible_positions))
    print 'possible words count:', len(possible_words)
    possible_words_sorted_by_len = sorted(possible_words, reverse=True, key=lambda x: len(x[0]))
    with open('possible_words.dat', 'wb') as f:
        pickle.dump(possible_words_sorted_by_len, f)


def main():
    import codecs
    print 'args:', ', '.join(sys.argv)
    puzzle_file = sys.argv[1] if len(sys.argv) > 1 else PUZZLE_FILE
    if puzzle_file.lower().endswith('txt'):
        with codecs.open(puzzle_file, encoding='utf-8') as f:
            rows = ''.join(f.readlines()).split()
    elif puzzle_file.lower().endswith('png'):
        if len(sys.argv) > 2:
            # if groundtruth letters are provided as commandline argument
            # we do need only try to update the shapes database and do not
            # try to actually solve the puzzle
            # print sys.argv[2].decode('utf-8')
            scan_screen_image(puzzle_file, sys.argv[2].decode('utf-8'))
            return
        else:
            rows = scan_screen_image(puzzle_file)
            if rows is None:
                return
    else:
        print 'txt or png boards supported only.'
        return

    global rows_count
    rows_count = len(rows)
    if not all(len(l) == rows_count for l in rows):
        print "square board expected. Specified board:\
        \n             rows:%i\
        \n  columns per row:%s" % \
              (rows_count, [len(l) for l in rows])
        return
    global board
    # noinspection PyTypeChecker
    board = np.full((rows_count + 3, rows_count + 3), u' ', dtype=np.dtype(('U', 1)))
    for i in range(rows_count):
        for j in range(rows_count):
            board[i + 1, j + 1] = unicode(rows[i][j])

    # print repr(board).decode("unicode-escape")
    # print repr('\n  '.join([(''.join(l[1:-1])) for l in board[1:-1]]))ape.decode("unicode-esc")

    print_board()
    # print is_word_in_position(1,1,u'тир')
    # print repr(rows[0][0])
    # print repr(board[1,1])
    # print is_word_anywhere(u'перегрев')
    #    find_possible_words()

    possible_words_sorted_by_len = find_possible_words_hashed()
    #    possible_words_sorted_by_len = pickle.load(open(DICT_FILE_NAME,'rb'))

    #    for word in possible_words_sorted_by_len:
    #        print word[0],len(word[1])

    #    for word_and_masks in possible_words_sorted_by_len[:5]:
    #        for mask in word_and_masks[1]:
    #            print_board_masked(mask)
    #            print
    list_posible_decompositions(possible_words_sorted_by_len[:])


# import cProfile
# cProfile.run('main()')
main()
