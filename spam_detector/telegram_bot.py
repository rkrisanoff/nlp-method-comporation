import asyncio
import logging
import os
from pickle import load

from aiogram import Bot, Dispatcher, Router, types
from aiogram.types import Message
from aiogram.filters import Filter

TOKEN = os.getenv("TELEGAM_BOT_TOKEN")

router = Router()

vectorizers = {"bag_of_words": "bag_of_words", "fast_text": "fast_text", "word2vec": "word2vec"}
classifiers = {"naive_bayes": "MultinomialNB", "random_forest": "RandomForestClassifier", "svc": "SVC"}
langs = {"ru": "russian", "en": "english"}
vector_method = "bag_of_words"
class_method = "naive_bayes"
lang = "en"
is_probabilistic = True

vectorizer = load(
    open(f"models/{lang}/vectorizers/{vectorizers[vector_method]}_vectorizer.pkl", "rb")
)
classifier = load(
    open(f"models/{lang}/classifiers/{vectorizers[vector_method]}/{classifiers[class_method]}.pkl", 'rb'))


def predict_if_spam(message):
    vectorized_message = vectorizer.transform([message])

    if is_probabilistic:
        predicted_probably = classifier.predict_proba(vectorized_message)
        return predicted_probably[0][1]
    else:
        predicted = classifier.predict(vectorized_message)
        return predicted[0] == 1


class MyFilter(Filter):
    def __init__(self, my_text: str) -> None:
        self.my_text = my_text

    async def __call__(self, message: Message) -> bool:
        # return True
        return message.text is not None


@router.message(MyFilter("hello"))  # (event_name=enums.chat_member_status.ChatMemberStatus)
async def echo_handler(message: types.Message) -> None:
    """
    Handler will forward received message back to the sender

    By default, message handler will handle all message types (like text, photo, sticker and etc.)
    """
    try:
        # Send copy of the received message
        probability = predict_if_spam(message.text)
        await message.answer(f"Probability your message is spam if {probability}")
        if probability > 0.5:
            await message.delete()
            await message.answer(f"You silly message `{message.text}`  was justified deleted")
        else:
            await message.answer(f"Keep living on... for now... this time")

    except TypeError:
        # But not all the types is supported to be copied so need to handle it
        await message.answer("Nice try!")


async def start() -> None:
    # Dispatcher is a root router
    dp = Dispatcher()
    # ... and all other routers should be attached to Dispatcher
    dp.include_router(router)

    # Initialize Bot instance with a default parse mode which will be passed to all API calls
    bot = Bot(TOKEN, parse_mode="HTML")
    # And the run events dispatching
    await dp.start_polling(bot)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(start())
