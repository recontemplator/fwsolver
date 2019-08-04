from skimage.filters import threshold_local
from skimage.morphology import remove_small_objects
import cv2
import numpy as np


def read_frames(file_name):
    vcf = cv2.VideoCapture(file_name)
    frames = []
    while True:
        is_ok, img = vcf.read()
        if not is_ok:
            break
        frames.append(np.transpose(img, (1, 0, 2))[:, ::-1, :])
    return np.array(frames)


def find_tip(contour, tip_function):
    return tuple(contour[max(enumerate([tip_function(*pt[0]) for pt in contour]), key=lambda p: p[1])[0]][0])


tip_bottom_right = lambda x, y: x + y
tip_bottom_left = lambda x, y: y - x
tip_top_right = lambda x, y: x - y
tip_top_left = lambda x, y: -x - y


def draw_points(frame, points, font_size=1, prefix=''):
    for idx, p in enumerate(points):
        cv2.circle(frame, p, 30, (0, 255, 0), -1)
        cv2.putText(frame,
                    prefix + str(idx + 1),
                    (p[0] - 5, p[1] + 5),
                    cv2.FONT_HERSHEY_TRIPLEX,
                    font_size, (255, 0, 0), 2)


def to_bw(frame):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray_blur = cv2.GaussianBlur(frame_gray, (5, 5), 100)
    frame_t = threshold_local(frame_gray_blur, 21, offset=2, method="gaussian")
    kernel = np.ones((3, 3), np.uint8)
    return cv2.morphologyEx((frame_gray_blur < frame_t).astype('uint8'), cv2.MORPH_OPEN, kernel)


def to_bw2(frame, kernel_size=27):
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    frame_gray_blur = cv2.GaussianBlur(frame_gray, (kernel_size, kernel_size), 160)
    frame_t = threshold_local(frame_gray_blur, 21, offset=2, method="gaussian")
    return (frame_gray < frame_t).astype('uint8')


def get_corners_by_contour(contour):
    return tuple(map(lambda f: find_tip(contour, f),
                     (tip_top_left, tip_top_right, tip_bottom_right, tip_bottom_left)))


def get_corners(frame):
    frame_morph = to_bw(frame)
    ext_contours, hierarchy = cv2.findContours(frame_morph, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    max_contour = max(ext_contours, key=cv2.contourArea)
    return get_corners_by_contour(max_contour)


def get_aoi(frame, side=512):
    corners = get_corners(frame)
    src = np.array(corners, dtype='float32')
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(src, dst), (int(side), int(side)))


def get_aoi_by_corners(frame, corners, side=512):
    src = np.array(corners, dtype='float32')
    dst = np.array([[0, 0], [side - 1, 0], [side - 1, side - 1], [0, side - 1]], dtype='float32')

    return cv2.warpPerspective(frame, cv2.getPerspectiveTransform(src, dst), (int(side), int(side)))


def find_stripes_count(sums):
    # noinspection PyBroadException
    try:
        sum_vert = sums.astype('float64')
        ws = 20
        #     sum_vert_copy = sum_vert.copy()
        sum_ver_idxs = (np.diff(np.sign(np.diff(sum_vert))) < 0).nonzero()[0] + 1
        sum_ver_idxs = sum_ver_idxs[sum_vert[sum_ver_idxs] > 0.8 * sum_vert.max()]
        for v in sum_ver_idxs:
            sum_vert[max(v - ws, 0):min(v + ws, len(sum_vert))] *= (sum_vert[
                                                                    max(v - ws, 0):min(v + ws, len(sum_vert))]) >= \
                                                                   sum_vert[v]
            if v > 0:
                sum_vert[max(v - ws, 0):v] *= (sum_vert[max(v - ws, 0):v]) > sum_vert[v]
        sum_ver_idxs = sum_ver_idxs[sum_vert[sum_ver_idxs] > 0.8 * sum_vert.max()]

        diffs = np.diff(sum_ver_idxs)
        if len(diffs) > 0:
            return int(sum_vert.shape[0] / np.median(diffs))
        else:
            return -1
    except:
        return -1


def get_frame_bw_adaptive(frame):
    kernel5 = np.ones((5, 5), np.uint8)
    frame_bw = cv2.morphologyEx(to_bw(get_aoi(frame, side=768)), cv2.MORPH_OPEN, kernel5)
    rows, cols = find_stripes_count(frame_bw.sum(axis=1)), find_stripes_count(frame_bw.sum(axis=0))
    if rows > 5:
        frame_bw = cv2.morphologyEx(to_bw(get_aoi(frame, side=2048)), cv2.MORPH_OPEN, kernel5)
        rows, cols = find_stripes_count(frame_bw.sum(axis=1)), find_stripes_count(frame_bw.sum(axis=0))
    return frame_bw, rows, cols


def get_frame_bw_adaptive2(frame, dbg=None):
    if dbg is None:
        dbg = dict()  # Debug information from this dictionary will be effectively discarded
    kernel5 = np.ones((5, 5), np.uint8)
    # side = 768
    frame_src_bw = to_bw(frame)
    dbg['frame_src_bw'] = frame_src_bw
    ext_contours, hierarchy = cv2.findContours(frame_src_bw, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    frame_area = np.prod(frame_src_bw.shape[:2])
    frame_bw_result, rows_result, cols_result = None, -1, -1
    all_considered_contours = []
    for contour in ext_contours:
        if cv2.contourArea(contour) > frame_area * .1:
            all_considered_contours.append(contour)
            corners = get_corners_by_contour(contour)
            aoi_by_corners = get_aoi_by_corners(frame, corners, 768)
            frame_bw = cv2.morphologyEx(to_bw(aoi_by_corners), cv2.MORPH_OPEN, kernel5)
            rows, cols = find_stripes_count(frame_bw.sum(axis=1)), find_stripes_count(frame_bw.sum(axis=0))
            if (rows == cols) and (rows > 2):
                dbg['corners'] = corners
                dbg['aoi_by_corners'] = aoi_by_corners
                dbg['contour'] = contour
                rows_result, cols_result = rows, cols
                # frame_bw_result = cv2.morphologyEx(to_bw2(get_aoi_by_corners(frame, corners, 1024)), cv2.MORPH_OPEN,
                #                                    kernel5)
                if rows > 5:
                    aoi_by_corners = get_aoi_by_corners(frame, corners, 2048)
                    dbg['aoi_by_corners'] = aoi_by_corners
                    frame_bw_result = to_bw2(aoi_by_corners)
                #                     frame_bw_result = cv2.morphologyEx(frame_bw_result, cv2.MORPH_OPEN, kernel5)
                else:
                    frame_bw_result = frame_bw
            else:
                dbg['corners_candidate'] = corners
                dbg['aoi_by_corners_candidate'] = aoi_by_corners

    dbg['all_considered_contours'] = all_considered_contours
    return frame_bw_result, rows_result, cols_result


def remove_zero_paddings(roi):
    b = np.argwhere(roi)
    (y_start, x_start), (y_stop, x_stop) = b.min(0), b.max(0) + 1
    return roi[y_start:y_stop, x_start:x_stop]


def extract_letters_to_recognize(img, size):
    assert img.shape[0] == img.shape[1]
    side = img.shape[0]
    padding = 30
    letters_to_recognize = []
    for i in range(size):
        for j in range(size):
            roi = img[
                  int(i * side / size + padding):int((i + 1) * side / size - padding),
                  int(j * side / size + padding):int((j + 1) * side / size - padding)
                  ]
            roi2 = remove_zero_paddings(roi).astype(bool)
            roi2 = remove_small_objects(roi2, min_size=np.prod(roi2.shape) // 25).astype('uint8')
            letters_to_recognize.append(remove_zero_paddings(roi2))
    return letters_to_recognize


def letters_to_recognize_to_pic(letters_to_recognize):
    size = int(len(letters_to_recognize) ** 0.5)
    assert size * size == len(letters_to_recognize), \
        f"length of letters_to_recognize {len(letters_to_recognize)} expected to be the perfect square."
    cell_size = max(max(img.shape) for img in letters_to_recognize) + 4
    cells_pic = np.zeros((cell_size * size, cell_size * size), dtype='uint8')
    ii = 0
    for i in range(size):
        for j in range(size):
            cell = letters_to_recognize[ii]
            cw, ch = cell.shape[:2]
            row_from = i * cell_size + (cell_size - cw) // 2
            col_from = j * cell_size + (cell_size - ch) // 2
            cells_pic[row_from:row_from + cw, col_from:col_from + ch] = cell
            ii += 1
    return cells_pic
