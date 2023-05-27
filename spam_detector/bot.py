# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylint: disable=too-few-public-methods

import asyncio
import logging
import sys
from os import getenv

from aiogram import Bot, Dispatcher, F, Router, types
from aiogram.filters import Command
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import (
    KeyboardButton,
    Message,
    ReplyKeyboardMarkup,
    ReplyKeyboardRemove,
)

from spam_detector.predict_service import predict_if_spam
from .domen import vectorizers, classifiers, langs

router = Router()


class ModelConfiguration(StatesGroup):
    configuration = State()
    vectorizer_choice = State()
    classifier_choice = State()
    language_choice = State()
    finishing = State()


class DetectingMode(StatesGroup):
    spam_detecting_on = State()
    spam_detecting_off = State()


ModelConfigurationProcessingStates = [
    ModelConfiguration.configuration,
    ModelConfiguration.vectorizer_choice,
    ModelConfiguration.classifier_choice,
]


# @router.chat_member(ChatMemberUpdatedFilter(IS_MEMBER >> IS_NOT_MEMBER))
# async def on_user_leave(event: ChatMemberUpdated):
#     print("on_user_leave")
#
#
# @router.chat_member(ChatMemberUpdatedFilter(IS_NOT_MEMBER >> IS_MEMBER))
# async def on_user_join(event: ChatMemberUpdated):print(on_user_join)

# @router.message(Command("spam_detecting_on"))
# async def spam_detecting_on(message: Message, state: FSMContext) -> None:
#     await state.set_state(DetectingMode.spam_detecting_on)
#
#     await message.answer(
#         "Spam detecting mode on.",
#         reply_markup=ReplyKeyboardRemove(),
#     )


# @router.message(DetectingMode.spam_detecting_on, Command("spam_detecting_off"))
# async def spam_detecting_off(message: Message, state: FSMContext) -> None:
#     await state.set_state(DetectingMode.spam_detecting_off)
#     await message.answer(
#         "Spam detecting mode off.",
#         reply_markup=ReplyKeyboardRemove(),
#     )


@router.message(Command("get_config", "get_configuration"))
async def command_get_configuration(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    await message.answer(
        f"""Текущая конфигурация:
        
- метод векторизации: {data.get("vectorizer", "bag_of_words")}
- классификатор: {data.get("classifier", "naive_bayes")}
- язык: {data.get("language", "en")}
        """,
        reply_markup=ReplyKeyboardRemove(),
    )


@router.message(Command("config"))
@router.message(Command("configuration"))
async def command_configuration_start(message: Message, state: FSMContext) -> None:
    await state.set_state(ModelConfiguration.configuration)
    data = await state.get_data()
    await message.answer(
        f"""Вы хотите изменить настройки бота?
- метод векторизации: {data.get("vectorizer", "bag_of_words")}
- классификатор: {data.get("classifier", "naive_bayes")}
- язык: {data.get("language", "en")}
        """,
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="да", ),
                    KeyboardButton(text="нет"),
                ],
            ],
            resize_keyboard=True,
        ),
    )


@router.message(Command("отмена"), ModelConfiguration.configuration)
@router.message(Command("отмена"), ModelConfiguration.language_choice)
@router.message(Command("отмена"), ModelConfiguration.vectorizer_choice)
@router.message(Command("отмена"), ModelConfiguration.classifier_choice)
@router.message(F.text.casefold() == "нет", ModelConfiguration.configuration)
@router.message(F.text.casefold() == "нет", ModelConfiguration.language_choice)
@router.message(F.text.casefold() == "нет", ModelConfiguration.vectorizer_choice)
@router.message(F.text.casefold() == "нет", ModelConfiguration.classifier_choice)
async def cancel_handler(message: Message, state: FSMContext) -> None:
    """
    Allow user to cancel any action
    """
    current_state = await state.get_state()

    logging.info("Cancelling state %r", current_state)
    await state.set_state(ModelConfiguration.finishing)
    await message.answer(
        "Отменено.",
        reply_markup=ReplyKeyboardRemove(),
    )


@router.message(ModelConfiguration.configuration, F.text.casefold() == "да")
async def vectorizer_choice_handler(message: Message, state: FSMContext) -> None:
    logging.info("Handler %s", "vectorizer_choice_handler")
    await state.set_state(ModelConfiguration.vectorizer_choice)
    keyboard = list(map(lambda vectorizer: [KeyboardButton(text=vectorizer)], vectorizers))
    keyboard.append([KeyboardButton(text="отменить")])
    await message.answer(
        f"Рад тебя видеть, {message.from_user.full_name}!\n"
        f"Какой метод векторизации ты хочешь выбрать??",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=keyboard,
            resize_keyboard=True,
        ),
    )


@router.message(ModelConfiguration.vectorizer_choice, ~(F.text.casefold().in_(vectorizers)))
async def vectorizer_choice_handler_wrong_input(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()
    logging.info("Handler %s", "vectorizer_choice_handler_wrong_input")
    logging.info("State %r", current_state)
    logging.info("State %s", message.text)
    logging.info("State %s", str(message.text in vectorizers))
    await state.set_state(ModelConfiguration.vectorizer_choice)
    await message.answer(
        f"Что-то пошло не так! Выбери  "
        f"{','.join(vectorizers.keys())} или отправь `отмена`",
        reply_markup=ReplyKeyboardRemove(), )
    await vectorizer_choice_handler(message, state)


@router.message(ModelConfiguration.vectorizer_choice, F.text.casefold().in_(vectorizers))
async def classifier_choice_handler(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()
    logging.info("Handler %s", "classifier_choice_handler")
    logging.info("State %r", current_state)
    await state.update_data(vectorizer=message.text)
    await state.set_state(ModelConfiguration.classifier_choice)
    keyboard = list(map(lambda classifier: [KeyboardButton(text=classifier)], classifiers))
    keyboard.append([KeyboardButton(text="отмена")])
    await message.answer(
        f"Рад тебя видеть, {message.from_user.full_name}!\n"
        f"Какой классификатор ты хочешь выбрать??",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=keyboard,
            resize_keyboard=True,
        ),
    )


@router.message(ModelConfiguration.classifier_choice, ~(F.text.casefold().in_(classifiers)))
async def classifier_choice_handler_wrong_input(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()

    logging.info("Handler %s", "classifier_choice_handler_wrong_input")
    logging.info("State %r", current_state)

    await message.answer(f"Что-то пошло не так!"
                         f"Выбери {','.join(classifiers.keys())} или отправь `отмена`",
                         reply_markup=ReplyKeyboardRemove(), )
    await classifier_choice_handler(message, state)


@router.message(ModelConfiguration.classifier_choice, F.text.casefold().in_(classifiers))
async def language_choice_handler(message: Message, state: FSMContext) -> None:
    logging.info("Handler %s", "language_choice_handler")
    await state.update_data(classifier=message.text)
    await state.set_state(ModelConfiguration.language_choice)

    keyboard = list(map(lambda language: [KeyboardButton(text=language)], langs))
    keyboard.append([KeyboardButton(text="отмена")])
    await message.answer(
        f"Рад тебя видеть, {message.from_user.full_name}!\n"
        f"Какой язык ты хочешь выбрать?",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=keyboard,
            resize_keyboard=True,
        ),
    )


@router.message(ModelConfiguration.language_choice, ~F.text.casefold().in_(langs))
async def language_choice_handler_wrong_input(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()
    logging.info("Handler %s", "language_choice_handler_wrong_input")
    logging.info("State %r", current_state)

    await message.answer(f"Что-то пошло не так!"
                         f" Выбери {','.join(classifiers.keys())} или отправь `отмена`",
                         reply_markup=ReplyKeyboardRemove(), )
    await language_choice_handler(message, state)


@router.message(ModelConfiguration.language_choice, F.text.casefold().in_(langs))
async def model_configuration_finishing(message: Message, state: FSMContext) -> None:
    logging.info("Handler %s", "model_configuration_finishing")
    await state.update_data(language=message.text)
    await state.set_state(ModelConfiguration.finishing)

    data = await state.get_data()
    vectorizer_id = data.get("vectorizer", "bag_of_words")
    classifier_id = data.get("classifier", "naive_bayes")
    lang_id = data.get("language", "en")

    await message.answer(f"""Конфигурация завершена!

- метод векторизации: {data.get("vectorizer", "bag_of_words")}
- классификатор: {data.get("classifier", "naive_bayes")}
- язык модели: {data.get("language", "en")}
    """, reply_markup=ReplyKeyboardRemove(), )


@router.message(ModelConfiguration.finishing)
async def echo_handler(message: types.Message, state: FSMContext) -> None:
    try:

        data = await state.get_data()
        vectorizer_id = data.get("vectorizer", "bag_of_words")
        classifier_id = data.get("classifier", "naive_bayes")
        language = data.get("language", "en")
        await message.answer(
            f"""
метод векторизации: {data.get("vectorizer", "bag_of_words")}
классификатор: {data.get("classifier", "naive_bayes")}
язык модели: {data.get("language", "en")}
            """
        )

        is_probabilistic = classifier_id != "svc"
        probability = predict_if_spam(message.text, vectorizer_id,
                                      classifier_id, language, is_probabilistic)
        if is_probabilistic:
            await message.answer(f"Вероятность отнесения сообщения к категории *спам* равна {probability}")
            if probability > 0.50:
                await message.delete()
                await message.answer(f"Ваше сообщение `{message.text}` было удалено")
            # else:
            # await message.answer(f"Keep living on... for now... this time")
        else:
            if probability:
                await message.answer(f"Ваше сообщение - спам!!!")
                await message.answer(f"Ваше сообщение `{message.text}` было удалено")
                await message.delete()
            else:
                await message.answer(f"Ваше сообщение не спам.")
                # await message.answer(f"Keep living on... for now... this time")

    except TypeError:
        await message.answer("Nice try!")


async def main():
    bot = Bot(token=getenv("TELEGRAM_BOT_TOKEN"), parse_mode="HTML")
    dispatcher = Dispatcher()
    dispatcher.include_router(router)
    await dispatcher.start_polling(bot)


def runner():
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    asyncio.run(main())


if __name__ == "__main__":
    runner()
