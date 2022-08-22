import asyncio
from aiogram import Dispatcher, Bot, executor
from aiogram.contrib.fsm_storage.memory import MemoryStorage
from random import randint
from aiogram.utils.callback_data import CallbackData
from aiogram.types import ChatActions, InlineKeyboardButton, InlineKeyboardMarkup, Message, CallbackQuery, ReplyKeyboardRemove, Audio
from aiogram.types.input_file import InputFile
from new_file_featuring import to_df
from tensorflow import keras
from pydub import AudioSegment 
import os
from sklearn.preprocessing import MinMaxScaler
import pandas as pd
import numpy as np


def create_model():
    keras.backend.clear_session()
    model = keras.models.Sequential([
        keras.layers.InputLayer(input_shape=(57)),
        keras.layers.Dense(units=1024, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(units=512, activation='relu'),
        keras.layers.Dropout(0.4),
        keras.layers.Dense(units=256, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(units=128, activation='relu'),
        keras.layers.Dropout(0.5),
        keras.layers.Dense(units=64, activation='relu'),
        keras.layers.Dropout(0.3),
        keras.layers.Dense(units=10, activation='softmax')
    ])
    model.compile('rmsprop', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model


bot_token = '5274618716:AAFR-C9zDOTU_BAUGIUMAXMkbSWsSOGqEy4'
bot = Bot(bot_token)
storage = MemoryStorage()
dp = Dispatcher(bot, storage=storage)
vote_cb = CallbackData('vote', 'action')
inline_kb_reactions = InlineKeyboardMarkup().add(InlineKeyboardButton('👍', callback_data=vote_cb.new(action='up')), InlineKeyboardButton('👎', callback_data=vote_cb.new(action='down')))
inline_kb_start = InlineKeyboardMarkup().add(InlineKeyboardButton('Играем😏', callback_data='start'))
again_kb = InlineKeyboardMarkup().add(InlineKeyboardButton('Пробуем другой?', callback_data='start'))


@dp.message_handler(commands=['start'])
async def start(message: Message):
    await ChatActions.typing(0.5)
    await message.answer(f'Привет, *{message.from_user.first_name}*!\nПредлагаю тебе сыграть в одну игру...🎭\n*Нажми на кнопку* - получишь инструкции ...',
                         parse_mode='Markdown', reply_markup=inline_kb_start)
                

@dp.callback_query_handler(lambda c: c.data == 'start')
async def start_game(query: CallbackQuery):
    await query.answer('Начали!')
    await ChatActions.typing(0.5)
    await bot.send_message(query.from_user.id, '*✅Погнали...*\nПрисылай мне .wav-file(или .mp3), а я угадаю его жанр и скину рекомендацию🎶',  parse_mode='Markdown')
    

@dp.message_handler(content_types=['document', 'audio'])
async def process_wav(message: Message):
    await ChatActions.typing(0.3)
    await message.reply('Анализирую...(это *занимает время*)', parse_mode='Markdown')
    filename = f'{randint(1, 999999)}'
    if message.audio:
        await message.audio.download(filename + '.mp3')
        sound = AudioSegment.from_mp3(filename + '.mp3') 
        sound.export(filename + '.wav', format="wav")
    try:
        if message.document:
            await message.document.download(filename + '.wav')
        df = to_df(filename + '.wav')
    except asyncio.exceptions.TimeoutError:
        await message.reply('Файл слишком большой для нашего интернета😥', reply_markup=again_kb)
        return
    except Exception as e:
        await message.reply('Неверный формат(или мы дурачки)😳')
        return
    recs = {
        'Блюз': 'blues.wav',
        'Классика': 'bethoven.wav',
        'Кантри': 'country.wav',
        'Диско': 'disco.wav',
        'Хип-хоп': 'my_name_is.wav',
        'Джаз': 'jazz.wav',
        'Метал': 'metal.wav',
        'Поп': 'будильник.wav',
        'Регги': 'reggae.wav',
        'Рок': 'korol_i_shut.wav',
    }
    keras.backend.clear_session()
    model = create_model()
    scaler = MinMaxScaler()
    df = pd.DataFrame(scaler.fit_transform(df), columns=df.columns)
    model.load_weights('music.h5')
    predict_data = pd.DataFrame({
        'Id': range(0, df.shape[0]),
        'Label': np.argmax(model.predict(scaler.transform(df)), axis=1)
    })
    print(predict_data)
    mean_genre = list(recs.keys())[int(predict_data['Label'].mode()[0])]
    await message.reply(f'Жанр: {mean_genre}\nОцени:', reply_markup=inline_kb_reactions)
    await bot.send_audio(message.from_user.id, InputFile(recs[mean_genre]), caption='Рекомендация')
    os.remove(filename + '.wav')
    if filename + '.mp3' in os.listdir():
        os.remove(filename + '.mp3')
    

@dp.callback_query_handler(vote_cb.filter(action='up'))
async def like(query: CallbackQuery):
    await query.answer('Yoo!')
    await query.message.edit_text('УРААА!!!🎉🎉🎉')
    await bot.send_message(947243146, '+')


@dp.callback_query_handler(vote_cb.filter(action='down'))
async def dislike(query: CallbackQuery):
    await query.answer(':(')
    await query.message.edit_text('Ну ладно..😥')
    await bot.send_message(947243146, '-')


executor.start_polling(dp)
