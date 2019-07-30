import cv2
import numpy as np
import gzip
import pickle

kernel5 = np.ones((5, 5), np.uint8)

OCR_DATA = 'letters.dat'
patterns = None


def init_ocr(ocr_data=OCR_DATA):
    global patterns
    with gzip.open(ocr_data, 'r') as f:
        patterns = pickle.load(f, encoding='latin1')


def is_img_the_same(i1, i2):
    return not cv2.morphologyEx(cv2.absdiff(i1, cv2.resize(i2, i1.shape[1::-1])), cv2.MORPH_OPEN, kernel5).any()


def build_board(letters_to_recognize, cols=None):
    if patterns is None:
        init_ocr()
    if cols is None:
        cols = int(len(letters_to_recognize) ** 0.5)
        assert cols * cols == len(letters_to_recognize), \
            f"length of letters_to_recognize {len(letters_to_recognize)} expected to be the perfect square."
    ll = 0
    board = ''
    for l in letters_to_recognize:
        letter = '*'
        # noinspection PyTypeChecker
        for k in patterns:
            if is_img_the_same(patterns[k], l * 255):
                if letter != '*':
                    print('warn: letters collision:', letter, k)
                letter = 'е' if k == 'ё' else k
        board += letter
        ll += 1
        if not (ll % cols):
            board += '\n'
    return board.split()
