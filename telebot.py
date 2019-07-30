import logging
import os
from telegram.ext import Updater, CommandHandler, MessageHandler, Filters
from computer_vision import *
from ocr import *
from solver2 import *

logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(levelname)s - %(message)s')


def reply_to_start_command(bot, update):
    logging.debug(f'reply to start call for bot: {bot}')
    first_name = update.effective_user.first_name
    update.message.reply_text("Send me a fillword picture {}".format(first_name))


def check_pic(bot, update):
    update.message.reply_text("Обрабатываю фото")
    photo_file = bot.getFile(update.message.photo[-1].file_id)
    filename = os.path.join('downloads', '{}.jpg'.format(photo_file.file_id))
    photo_file.download(filename)
    frame = cv2.imread(filename)
    update.message.reply_text(solve_frame(frame))


def solve_frame(frame):
    frame_bw, rows, cols = get_frame_bw_adaptive2(frame)
    if (rows == cols) and (rows > 2):
        letters_to_recognize = extract_letters_to_recognize(frame_bw, rows)
        board = build_board(letters_to_recognize)
        return solve_txt(board)
    else:
        return f'Unable to detect the board. Best attempt is {rows}x{cols} field.'


def start_bot():
    with open('bot.id') as f:
        bot_id = f.readline().strip()
    my_bot = Updater(bot_id)

    dp = my_bot.dispatcher
    dp.add_handler(CommandHandler("start", reply_to_start_command))
    dp.add_handler(MessageHandler(Filters.photo, check_pic))

    my_bot.start_polling()
    my_bot.idle()


if __name__ == "__main__":
    start_bot()
