# pylint: disable=missing-function-docstring
# pylint: disable=missing-module-docstring
# pylint: disable=missing-class-docstring
# pylint: disable=f-string-without-interpolation
# pylint: disable=too-few-public-methods

import asyncio
import logging
import os
import sys
from os import getenv
from pickle import load

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
from nltk import word_tokenize
import langdetect
from langdetect import DetectorFactory


from spam_detector.utils import vectorize
from .domen import vectorizers, classifiers, langs

form_router = Router()
DetectorFactory.seed = 0


class ModelConfiguration(StatesGroup):
    configuration = State()
    vectorizer_choice = State()
    classifier_choice = State()
    finishing = State()
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

@form_router.message(Command("spam_detecting_on"))
async def spam_detecting_on(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()

    await state.set_state(ModelConfiguration.spam_detecting_on)
    logging.info("New state is %r", current_state)

    await message.answer(
        "Spam detecting mode on.",
        reply_markup=ReplyKeyboardRemove(),
    )


@form_router.message(ModelConfiguration.spam_detecting_on, Command("spam_detecting_off"))
async def spam_detecting_off(_: Message, state: FSMContext) -> None:
    await state.set_state(ModelConfiguration.spam_detecting_off)


@form_router.message(Command("get_config", "get_configuration"))
async def command_get_configuration(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    await message.answer(
        f"""Okay, there are a setting of spam detector?
        
- vectorizer: {data.get("vectorizer", "bag_of_words")}
- classifier: {data.get("classifier", "naive_bayes")}
        """,
        reply_markup=ReplyKeyboardRemove(),
    )


@form_router.message(Command("config"))
@form_router.message(Command("configuration"))
async def command_configuration_start(message: Message, state: FSMContext) -> None:
    await state.set_state(ModelConfiguration.configuration)
    data = await state.get_data()
    await message.answer(
        f"""Okay, do you want to change this setting of spam detector
- vectorizer: {data.get("vectorizer", "bag_of_words")}
- classifier: {data.get("classifier", "naive_bayes")}
        language: {data.get("lang", "en")}
        """,
        reply_markup=ReplyKeyboardMarkup(
            keyboard=[
                [
                    KeyboardButton(text="yes", ),
                    KeyboardButton(text="no"),
                ],
            ],
            resize_keyboard=True,
        ),
    )


@form_router.message(Command("cancel"))
@form_router.message(Command("reset"))
@form_router.message(F.text.casefold().in_(["cancel", "reset"]))
@form_router.message(F.text.casefold() == "no", ModelConfiguration.configuration)
async def cancel_handler(message: Message, state: FSMContext) -> None:
    """
    Allow user to cancel any action
    """
    current_state = await state.get_state()

    if current_state is None:
        return

    logging.info("Cancelling state %r", current_state)
    await state.clear()
    await message.answer(
        "Cancelled.",
        reply_markup=ReplyKeyboardRemove(),
    )


@form_router.message(ModelConfiguration.configuration, F.text.casefold() == "yes")
async def vectorizer_choice_handler(message: Message, state: FSMContext) -> None:
    logging.info("Handler %s", "vectorizer_choice_handler")
    await state.set_state(ModelConfiguration.vectorizer_choice)
    keyboard = list(map(lambda vectorizer: [KeyboardButton(text=vectorizer)], vectorizers))
    keyboard.append([KeyboardButton(text="cancel")])
    await message.answer(
        f"Nice to meet you, {message.from_user.full_name}!\n"
        f"What vectorizer method do you want to choice?",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=keyboard,
            resize_keyboard=True,
        ),
    )


@form_router.message(ModelConfiguration.vectorizer_choice, ~(F.text.casefold().in_(vectorizers)))
async def vectorizer_choice_handler_wrong_input(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()
    logging.info("Handler %s", "vectorizer_choice_handler_wrong_input")
    logging.info("State %r", current_state)
    logging.info("State %s", message.text)
    logging.info("State %s", str(message.text in vectorizers))
    await state.set_state(ModelConfiguration.vectorizer_choice)
    await message.answer(
        f"Something went wrong! Try again choice vectorizer from "
        f"{','.join(vectorizers.keys())} or type `cancel`",
        reply_markup=ReplyKeyboardRemove(), )
    await message.answer(f"Anyway, set's start!")
    print('okey, let\'s again')
    print(f"`{message.text}`")
    print(vectorizers)
    print(message.text in vectorizers)
    await vectorizer_choice_handler(message, state)


@form_router.message(ModelConfiguration.vectorizer_choice, F.text.casefold().in_(vectorizers))
async def classifier_choice_handler(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()
    logging.info("Handler %s", "classifier_choice_handler")
    logging.info("State %r", current_state)
    await state.update_data(vectorizer=message.text)
    await state.set_state(ModelConfiguration.classifier_choice)
    keyboard = list(map(lambda classifier: [KeyboardButton(text=classifier)], classifiers))
    keyboard.append([KeyboardButton(text="cancel")])
    await message.answer(
        f"Nice to meet you, {message.from_user.full_name}!\n"
        f"What classifier method do you want to choice?",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=keyboard,
            resize_keyboard=True,
        ),
    )


@form_router.message(ModelConfiguration.classifier_choice, ~(F.text.casefold().in_(classifiers)))
async def classifier_choice_handler_wrong_input(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()
    logging.info("Handler %s", "classifier_choice_handler_wrong_input")
    logging.info("State %r", current_state)

    await state.set_state(ModelConfiguration.classifier_choice)
    await message.answer(f"Something went wrong!"
                         f" Try again from {','.join(classifiers.keys())} or type `cancel`",
                         reply_markup=ReplyKeyboardRemove(), )
    await message.answer(f"Anyway, set's start!")
    await classifier_choice_handler(message, state)


@form_router.message(ModelConfiguration.classifier_choice, F.text.casefold().in_(classifiers))
async def model_configuration_finishing(message: Message, state: FSMContext) -> None:
    logging.info("Handler %s", "model_configuration_finishing")
    await state.update_data(classifier=message.text)
    await state.set_state(ModelConfiguration.finishing)

    data = await state.get_data()
    await message.answer(f"""Anyway, you ended configuration!
    
- vectorizer: {data.get("vectorizer", "bag_of_words")}
- classifier: {data.get("classifier", "naive_bayes")}
    """, reply_markup=ReplyKeyboardRemove(), )


def predict_if_spam(message, vectorizer_id, classifier_id, lang, is_probabilistic):
    if classifier_id == "svc" and is_probabilistic:
        raise AssertionError("SVC doesn't support probabilistic prediction")

    vectorizer_path = [
        'models',
        lang,
        'vectorizers',
        f"{vectorizers[vectorizer_id]}_vectorizer.pkl"
    ]
    with open(os.path.join(*vectorizer_path), "rb") as vectorizer_file:
        vectorizer = load(vectorizer_file)

    classifier_path = [
        'models',
        lang,
        'classifiers',
        vectorizers[vectorizer_id],
        f"{classifiers[classifier_id]}.pkl",
    ]
    with open(os.path.join(*classifier_path), "rb") as classifier_file:
        classifier = load(classifier_file)

    if vectorizer_id == "bag_of_words":
        vectorized_message = vectorizer.transform([message])
    else:
        tokenized_message = [word_tokenize(text) for text in [message]]
        vectorized_message = vectorize(vectorizer, tokenized_message)

    if is_probabilistic:
        predicted_probably = classifier.predict_proba(vectorized_message)
        return predicted_probably[0][1]

    predicted = classifier.predict(vectorized_message)
    return predicted[0]


@form_router.message(ModelConfiguration.spam_detecting_on)
async def echo_handler(message: types.Message, state: FSMContext) -> None:
    try:

        data = await state.get_data()

        lang_res = langdetect.detect_langs(message.text)
        lang_res = map(lambda lang: str(lang).split(":"), lang_res)
        lang_res = {str(lang[0]): lang[1] for lang in lang_res}
        print(lang_res)
        intersect = tuple(sorted(set(lang_res).intersection(langs),
                                 key=lambda lang: lang[1]))
        print(intersect)
        if len(lang_res):
            if len(intersect) == 0:
                await message.answer(f"""Unknown languages""")
                await message.answer(f"""Bot support only english and russian languages""")
                return
            # if len(intersect) > 1:
            #     await message.answer(f"""Unknown languages""")
            # logging.info("Determined lang { %s }", ", ".join(intersect))
            # return
            logging.info("languages: { %s }", ", ".join(intersect))
            lang = intersect[0]
        else:
            await message.answer(f"Message too short to determine language!\n"
                                 f"Using english as default language ")
            lang = "en"

        await message.answer(f"""The message language is {lang}""")
        vectorizer_id = data.get("vectorizer", "bag_of_words")
        classifier_id = data.get("classifier", "naive_bayes")
        await message.answer(
            f"""
            vectorizer: {data.get("vectorizer", "bag_of_words")}
            classifier: {data.get("classifier", "naive_bayes")}
            """
        )
        is_probabilistic = classifier_id != "svc"
        probability = predict_if_spam(message.text, vectorizer_id,
                                      classifier_id, lang, is_probabilistic)
        if is_probabilistic:
            await message.answer(f"Probability your message is spam is {probability}")
            if probability > 0.50:
                await message.delete()
                await message.answer(f"You message `{message.text}` was justified deleted")
            else:
                await message.answer(f"Keep living on... for now... this time")
        else:
            if probability:
                await message.answer(f"Your message is a spam!!!")
                await message.delete()
            else:
                await message.answer(f"Your message is NOT a spam.")
                await message.answer(f"Keep living on... for now... this time")

    except TypeError:
        await message.answer("Nice try!")


async def main():
    bot = Bot(token=getenv("TELEGRAM_BOT_TOKEN"), parse_mode="HTML")
    dispatcher = Dispatcher()
    dispatcher.include_router(form_router)
    await dispatcher.start_polling(bot)


def runner():
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)
    asyncio.run(main())


if __name__ == "__main__":
    runner()
