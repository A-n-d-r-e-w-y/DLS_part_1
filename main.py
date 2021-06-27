import os
import io
import logging
import shutil

import numpy as np
from PIL import Image
from aiogram import Bot, Dispatcher, executor, types
from aiogram.types import InlineKeyboardButton, InlineKeyboardMarkup

from __init__ import API_TOKEN, DIRECTORY_TO_SAVE_IMAGES, AVAILABLE_SIZES, MAX_MEMORY_MB
from model import style_transferring
import database


"""inline_kb_imsize отвечает за выбор размера изображения"""
inline_btns = []
button_id_to_size = dict()
for i, size in enumerate(AVAILABLE_SIZES):
    button_id = f"button{i}"
    inline_btns.append(InlineKeyboardButton(str(size), callback_data=button_id))
    button_id_to_size[button_id] = size
inline_kb_imsize = InlineKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True).row(*inline_btns)


"""inline_kb_num_steps отвечает за выбор числа шагов при обучении модели"""
inline_btns = []
button_id_to_num_steps = dict()
for i in range(4, 7):
    button_id = f"button{i}"
    num_iter = 300 + 200 * (i-4)
    inline_btns.append(InlineKeyboardButton(str(num_iter), callback_data=button_id))
    button_id_to_num_steps[button_id] = num_iter
inline_kb_num_steps = InlineKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True).row(*inline_btns)


"""inline_kb_weight отвечает за выбор вклада лосса стиля. для удобства вес контента всегда 1"""
inline_btns = []
button_id_to_style_weight = dict()
for i in range(7, 10):
    button_id = f"button{i}"
    weight = 10 ** (i - 4)
    inline_btns.append(InlineKeyboardButton(str(weight), callback_data=button_id))
    button_id_to_style_weight[button_id] = weight
inline_kb_weight = InlineKeyboardMarkup(resize_keyboard=True, one_time_keyboard=True).row(*inline_btns)


logging.basicConfig(level=logging.INFO)
user_data = database.DataStore()
bot = Bot(token=API_TOKEN)
dp = Dispatcher(bot)


def tensor2img(t):
    output = np.rollaxis(t.cpu().detach().numpy()[0], 0, 3)
    output = Image.fromarray(np.uint8(output * 255))
    bio = io.BytesIO()
    bio.name = 'result.jpeg'
    output.save(bio, 'JPEG')
    bio.seek(0)
    return bio


@dp.message_handler(commands=['start'])
async def start(message: types.Message):
    await message.answer(f"Привет, {message.from_user.first_name}!\n" +
                         "Я бот, который может перенести стиль с двух картинок на третью.\n"
                         "Вот мои команды:\n" +
                         "/start - вывести стартовое приветствие\n" +
                         "/info - подробное описание команд\n" +
                         "/cancel - отменить все предыдущие команды\n" +
                         "/settings (настроить параметры: число итераций и вес стиля)\n" +
                         "/run - запустить перенос стиля")


@dp.message_handler(commands=['info'])
async def info(message: types.Message):
    await message.answer(f"ИНФО:\n" +
                         "Бот позволяет перенести стиль с двух картинок.\n" +
                         "Сначала вы отправляете фото, на которое будут переноситься стили с двух других картинок.\n" +
                         "Затем отправляете по очереди две другие картинки (стили).\n" +
                         "В итоге вы получите картинку, содержащую одновременно два стиля.\n\n" +
                         "Доступны следующие команды:\n" +
                         "/start - вывести стартовое приветствие\n" +
                         "/info - подробное описание команд\n" +
                         "/cancel - отменить все предыдущие команды, кроме settings\n" +
                         "/settings (настроить параметры: число итераций и вес стиля)\n" +
                         "/run - запустить перенос стиля")


@dp.message_handler(commands=['cancel'])
async def cancel(message: types.Message):
    user_data.update_state(message.chat.id, "null_state")
    await message.answer(f"Все действия отменены")


@dp.message_handler(commands=['run'])
async def run(message: types.Message):
    await message.answer("Отправьте мне своё фото, на которое будет переноситься стиль")
    user_data.update_state(message.chat.id, "await_photo")


@dp.message_handler(commands=['settings'])
async def run(message: types.Message):
    await message.reply("Выберите число итераций", reply_markup=inline_kb_num_steps)


@dp.callback_query_handler(lambda c: c.data in ['button4', 'button5', 'button6'])
async def process_callback_num_steps(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    button_id = callback_query.data
    num_steps = button_id_to_num_steps[button_id]
    await bot.send_message(callback_query.from_user.id,
                           f'Вы выбрали {num_steps} итераций.\n'
                           f'Теперь выберите вес стиля', reply_markup=inline_kb_weight)
    user_data.update_params(callback_query.message.chat.id, {'num_steps': num_steps})


@dp.callback_query_handler(lambda c: c.data in ['button7', 'button8', 'button9'])
async def process_callback_weight(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    button_id = callback_query.data
    weight = button_id_to_style_weight[button_id]
    await bot.send_message(callback_query.from_user.id,
                           f'Вы выбрали вес стиля = {weight}')
    user_data.update_params(callback_query.message.chat.id,
                            {'style_weight1': weight, 'style_weight2': weight})


@dp.message_handler(content_types=[types.message.ContentType.PHOTO])
async def get_image(message: types.Message):
    state = user_data.get_state(message.chat.id)
    if state == "await_photo":
        file_id = message.photo[-1].file_id
        await message.answer("Получаю ваше фото...")
        file_to_save = os.path.join(DIRECTORY_TO_SAVE_IMAGES, file_id)
        await message.photo[-1].download(file_to_save)
        user_data.update_files(message.chat.id, file_id, file_type="photo")
        user_data.update_state(message.chat.id, "await_style_1")
        await message.reply("Теперь отправьте фото с первым стилем")
    elif state == "await_style_1":
        file_id = message.photo[-1].file_id
        await message.answer("Получаю ваше фото с первым стилем...")
        file_to_save = os.path.join(DIRECTORY_TO_SAVE_IMAGES, file_id)
        await message.photo[-1].download(file_to_save)
        user_data.update_files(message.chat.id, file_id, file_type="style_1")
        user_data.update_state(message.chat.id, "await_style_2")
        await message.reply("Теперь отправьте фото со вторым стилем")
    elif state == "await_style_2":
        file_id = message.photo[-1].file_id
        await message.answer("Получаю ваше фото со вторым стилем...")
        file_to_save = os.path.join(DIRECTORY_TO_SAVE_IMAGES, file_id)
        await message.photo[-1].download(file_to_save)
        user_data.update_files(message.chat.id, file_id, file_type="style_2")
        user_data.update_state(message.chat.id, "await_size")
        await message.reply("Выберите размер изображения", reply_markup=inline_kb_imsize)
    else:
        await message.reply("Это фото не нужно было отправлять сейчас)")


@dp.callback_query_handler(lambda c: c.data in ['button0', 'button1', 'button2', 'button3'])
async def process_callback_photo(callback_query: types.CallbackQuery):
    await bot.answer_callback_query(callback_query.id)
    button_id = callback_query.data
    size = button_id_to_size[button_id]
    await bot.send_message(callback_query.from_user.id,
                           f'Вы выбрали размер изображения {size}. Начинаю перенос стиля...')
    msg = await bot.send_message(callback_query.from_user.id,
                                 f'Обработано 0 %')
    await bot.edit_message_reply_markup(chat_id=callback_query.message.chat.id,
                                        message_id=callback_query.message.message_id)
    files = user_data.get_files(callback_query.message.chat.id)
    if files == -1 or files['photo'] is None or files['style_1'] is None or files['style_2'] is None:
        await bot.send_message(callback_query.from_user.id,
                               "Ошибка( Я не могу найти ваши фото и два стиля. Попробуйте снова")
    else:
        params = user_data.get_params(callback_query.message.chat.id)
        for output in style_transferring(path_to_content=os.path.join(DIRECTORY_TO_SAVE_IMAGES, files['photo']),
                                         path_to_style_1=os.path.join(DIRECTORY_TO_SAVE_IMAGES, files['style_1']),
                                         path_to_style_2=os.path.join(DIRECTORY_TO_SAVE_IMAGES, files['style_2']),
                                         imsize=size,
                                         content_weight=params['content_weight'],
                                         style_weight1=params['style_weight1'],
                                         style_weight2=params['style_weight2'],
                                         num_steps=params['num_steps']):
            if output[0] == "iteration":
                iter = output[1]
                await msg.edit_text(f"Обработано {iter / params['num_steps'] * 100:.1f} %")
            else:
                img = output[1]
                i = tensor2img(img)
                await bot.send_photo(callback_query.message.chat.id, types.InputFile(i), caption="Результат")
    user_data.update_state(callback_query.message.chat.id, "null_state")
    await bot.send_message(callback_query.from_user.id, 'Перенос стиля завершён завершен')


@dp.message_handler()
async def echo(message: types.Message):
    await message.reply("Я вас не понимаю. Для работы следуйте моим командам или выбери одну из комманд отсюда: /info" )


if __name__ == '__main__':

    if os.path.exists(DIRECTORY_TO_SAVE_IMAGES):
        files = os.listdir(DIRECTORY_TO_SAVE_IMAGES)
        total_size = sum((os.path.getsize(os.path.join(DIRECTORY_TO_SAVE_IMAGES, f)) for f in files))
        if total_size / 1024 / 1024 > MAX_MEMORY_MB:
            shutil.rmtree(DIRECTORY_TO_SAVE_IMAGES)
    os.makedirs(DIRECTORY_TO_SAVE_IMAGES, exist_ok=True)

    executor.start_polling(dp, skip_updates=True)
