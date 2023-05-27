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

form_router = Router()


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

# @form_router.message(Command("spam_detecting_on"))
# async def spam_detecting_on(message: Message, state: FSMContext) -> None:
#     await state.set_state(DetectingMode.spam_detecting_on)
#
#     await message.answer(
#         "Spam detecting mode on.",
#         reply_markup=ReplyKeyboardRemove(),
#     )


# @form_router.message(DetectingMode.spam_detecting_on, Command("spam_detecting_off"))
# async def spam_detecting_off(message: Message, state: FSMContext) -> None:
#     await state.set_state(DetectingMode.spam_detecting_off)
#     await message.answer(
#         "Spam detecting mode off.",
#         reply_markup=ReplyKeyboardRemove(),
#     )


@form_router.message(Command("get_config", "get_configuration"))
async def command_get_configuration(message: Message, state: FSMContext) -> None:
    data = await state.get_data()
    await message.answer(
        f"""Okay, there are a setting of spam detector?
        
- vectorizer: {data.get("vectorizer", "bag_of_words")}
- classifier: {data.get("classifier", "naive_bayes")}
- language: {data.get("language", "en")}
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
- language: {data.get("language", "en")}
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
# @form_router.message(F.text.casefold().in_(["cancel", "reset"]))
@form_router.message(F.text.casefold() == "no", ModelConfiguration.configuration)
async def cancel_handler(message: Message, state: FSMContext) -> None:
    """
    Allow user to cancel any action
    """
    current_state = await state.get_state()

    logging.info("Cancelling state %r", current_state)
    await state.set_state(ModelConfiguration.finishing)
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

    await message.answer(f"Something went wrong!"
                         f" Try again from {','.join(classifiers.keys())} or type `cancel`",
                         reply_markup=ReplyKeyboardRemove(), )
    await classifier_choice_handler(message, state)


@form_router.message(ModelConfiguration.classifier_choice, F.text.casefold().in_(classifiers))
async def language_choice_handler(message: Message, state: FSMContext) -> None:
    logging.info("Handler %s", "language_choice_handler")
    await state.update_data(classifier=message.text)
    await state.set_state(ModelConfiguration.language_choice)

    keyboard = list(map(lambda language: [KeyboardButton(text=language)], langs))
    keyboard.append([KeyboardButton(text="cancel")])
    await message.answer(
        f"Nice to meet you, {message.from_user.full_name}!\n"
        f"What language of models do you want to choice?",
        reply_markup=ReplyKeyboardMarkup(
            keyboard=keyboard,
            resize_keyboard=True,
        ),
    )


@form_router.message(ModelConfiguration.language_choice, ~F.text.casefold().in_(langs))
async def language_choice_handler_wrong_input(message: Message, state: FSMContext) -> None:
    current_state = await state.get_state()
    logging.info("Handler %s", "language_choice_handler_wrong_input")
    logging.info("State %r", current_state)

    await message.answer(f"Something went wrong!"
                         f" Try again from {','.join(classifiers.keys())} or type `cancel`",
                         reply_markup=ReplyKeyboardRemove(), )
    await language_choice_handler(message, state)


@form_router.message(ModelConfiguration.language_choice, F.text.casefold().in_(langs))
async def model_configuration_finishing(message: Message, state: FSMContext) -> None:
    logging.info("Handler %s", "model_configuration_finishing")
    await state.update_data(language=message.text)
    await state.set_state(ModelConfiguration.finishing)

    data = await state.get_data()
    vectorizer_id = data.get("vectorizer", "bag_of_words")
    classifier_id = data.get("classifier", "naive_bayes")
    lang_id = data.get("language", "en")

    await message.answer(f"""Anyway, you ended configuration!

- vectorizer: {data.get("vectorizer", "bag_of_words")}
- classifier: {data.get("classifier", "naive_bayes")}
- language: {data.get("language", "en")}
    """, reply_markup=ReplyKeyboardRemove(), )


@form_router.message(ModelConfiguration.finishing)
async def echo_handler(message: types.Message, state: FSMContext) -> None:
    try:

        data = await state.get_data()
        vectorizer_id = data.get("vectorizer", "bag_of_words")
        classifier_id = data.get("classifier", "naive_bayes")
        language = data.get("language", "en")
        await message.answer(
            f"""
vectorizer: {data.get("vectorizer", "bag_of_words")}
classifier: {data.get("classifier", "naive_bayes")}
language: {data.get("language", "en")}
            """
        )

        is_probabilistic = classifier_id != "svc"
        probability = predict_if_spam(message.text, vectorizer_id,
                                      classifier_id, language, is_probabilistic)
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
                await message.answer(f"You message `{message.text}` was justified deleted")
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
