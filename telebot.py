import logging
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
import telegram
from computer_vision import *
from ocr import *
from solver2 import *
import pickle
from PIL import Image
import os
from io import BytesIO

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')


def reply_to_start_command(bot, update):
    logging.debug(f'reply to start call for bot: {bot}')
    first_name = update.effective_user.first_name
    update.message.reply_text("Привет! Я робот Федор Филвордович. Филвордович - фамилия. "
                              "Я умею разгадывать слова в игре ФИЛВОРДЫ."
                              " Пришли мне картинку из этой игры, {}".format(first_name))


def reply_to_debug_command(bot, update):
    logging.debug(f'reply to debug call for bot: {bot}')
    filename_user_log = os.path.join('downloads', 'usr_{}.log'.format(update.effective_user.id))
    if os.path.isfile(filename_user_log):
        with open(filename_user_log) as f:
            pkl_filename = f.readline().strip()
        with open(pkl_filename, 'rb') as f:
            dbg = pickle.load(f)
        reply_photo_by_array(update, visualise_contours(dbg))
        if 'answer_ar' in dbg:
            reply_photo_by_array(update, dbg['answer_ar'])
        if ('*' in ''.join(dbg.get('board', ''))) and ('letters_pic' in dbg):
            reply_photo_by_array(update, dbg['letters_pic'] * 255)

        logging.info("debug keys: " + ', '.join(dbg.keys()))
    else:
        update.message.reply_text("Не могу сделать debug, нет истории. Пришли хотя бы одну картинку.")


def reply_photo_by_array(update, img):
    if isinstance(img, np.ndarray):
        bytes_jpg = image_to_jpeg_bytes(img)
    elif isinstance(img, bytes):
        bytes_jpg = img
    else:
        return
    update.message.reply_photo(photo=BytesIO(bytes_jpg))


def image_to_jpeg_bytes(img):
    with BytesIO() as bio:
        bio.name = 'image.jpeg'
        Image.fromarray(img).save(bio, 'JPEG')
        bio.seek(0)
        return bio.read()


def check_pic(bot, update):
    update.message.reply_text("Спасибо за картинку!")
    photo_file = bot.getFile(update.message.photo[-1].file_id)
    filename = os.path.join('downloads', '{}.jpg'.format(photo_file.file_id))
    filename_pkl = os.path.join('downloads', '{}.pkl'.format(photo_file.file_id))
    filename_user_log = os.path.join('downloads', 'usr_{}.log'.format(update.effective_user.id))
    photo_file.download(filename)
    frame = cv2.imread(filename)
    dbg = {}
    update.message.reply_text(solve_frame(frame, dbg), parse_mode=telegram.ParseMode.MARKDOWN)
    if 'answer_pic' in dbg:
        reply_photo_by_array(update, dbg['answer_pic'])

    with open(filename_pkl, 'wb') as f:
        pickle.dump(dbg, f)

    with open(filename_user_log, 'w') as f:
        print(filename_pkl, file=f)


def solve_frame(frame, dbg):
    frame_bw, rows, cols = get_frame_bw_adaptive2(frame, dbg)
    solved = False
    if (rows == cols) and (rows > 2):
        letters_to_recognize = extract_letters_to_recognize(frame_bw, rows)
        dbg['letters_pic'] = image_to_jpeg_bytes(letters_to_recognize_to_pic(letters_to_recognize))
        board = build_board(letters_to_recognize)
        dbg['board'] = board
        board_printable = ('\n'.join([''.join(c + ' ' for c in l) for l in board]))
        answer_res, decomposition = solve_txt(build_board(letters_to_recognize))
        answer = f'Вот какое поле для игры, я увидел своим компютерным зрением:\n' \
                 f'`{board_printable}`\n\n{answer_res}'
        dbg['answer_pic'] = image_to_jpeg_bytes(
            make_answer_frame(cv2.resize(frame_bw, (512, 512)), decomposition, rows))
        dbg['answer_ar'] = image_to_jpeg_bytes(make_ar_frame(frame, decomposition, rows, dbg['corners']))
        solved = True
    else:
        answer = f'Видимо, ' \
                 f'моё компьютерное зрение подводит меня, я не вижу тут игрового поля.'
    dbg['answer'] = answer
    dbg['solved'] = solved
    return answer


def start_bot():
    with open('bot.id') as f:
        bot_id = f.readline().strip()
    my_bot = Updater(bot_id)

    dp = my_bot.dispatcher
    dp.add_handler(CommandHandler("start", reply_to_start_command))
    dp.add_handler(CommandHandler("debug", reply_to_debug_command))
    dp.add_handler(MessageHandler(Filters.photo, check_pic))
    my_bot.start_polling()
    logging.info('Started')
    my_bot.idle()


if __name__ == "__main__":
    start_bot()
