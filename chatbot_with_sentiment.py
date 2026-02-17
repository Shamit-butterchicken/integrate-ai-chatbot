# Databricks notebook source
# Run this once in your Databricks notebook cluster
%pip install transformers torch --quiet

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

from transformers import pipeline


# COMMAND ----------

sentiment_analysis = pipeline("sentiment-analysis", model="distilbert-base-uncased-finetuned-sst-2-english")


# COMMAND ----------

def get_sentiment(text):
    """
    Returns a tuple (label, score) where label is 'POSITIVE'/'NEGATIVE' and score is float (0..1)
    """
    if not text or not text.strip():
        return ("NEUTRAL", 0.0)
    try:
        result = sentiment_analysis(text[:512])  # limit length just in case
        label = result[0]["label"]
        score = float(result[0]["score"])
        return (label, score)
    except Exception as e:
        return ("ERROR", 0.0)


# COMMAND ----------

import re
from difflib import get_close_matches

# Keep your previous intent keywords (add 'sentiment' trigger if you want on-demand)
INTENT_KEYWORDS = {
    "greeting": ["hi", "hello", "hey", "good morning", "good afternoon"],
    "goodbye": ["bye", "goodbye", "see you", "farewell"],
    "thanks": ["thanks", "thank you", "thx"],
    "capabilities": ["capabilities", "what can you do", "features", "help list"],
    "hours": ["hours", "open", "working hours"],
    "help": ["help", "support", "how to use"]
}

RESPONSES = {
    "greeting": "Hello! I'm SimpleBot — I can answer a few basic questions. Type 'capabilities' to see what I do.",
    "goodbye": "Goodbye! Feel free to run me again anytime.",
    "thanks": "You're welcome! Glad to help.",
    "capabilities": ("I can: 1) Greet you, 2) Tell my capabilities, "
                     "3) Tell support hours, 4) Run sentiment analysis on your messages."),
    "hours": "Our support hours are Mon-Fri 9am-5pm EST.",
    "help": "Type one short sentence, or ask 'what can you do' to see my capabilities. "
            "If I don't understand, I'll give suggestions.",
    "fallback": "Sorry, I didn't understand that. Try a short question like 'what can you do' or 'hours'.",
}

def detect_intent(user_input):
    s = user_input.lower().strip()
    # exact command match
    if s in ("capabilities", "help", "hours", "sentiment"):
        return s
    for intent, keywords in INTENT_KEYWORDS.items():
        for kw in keywords:
            if kw in s:
                return intent
    # fuzzy match
    tokens = re.findall(r"[a-z]+", s)
    all_keywords = [k for kws in INTENT_KEYWORDS.values() for k in kws]
    for t in tokens:
        close = get_close_matches(t, all_keywords, n=1, cutoff=0.8)
        if close:
            for intent, kws in INTENT_KEYWORDS.items():
                if close[0] in kws:
                    return intent
    return "fallback"


# COMMAND ----------

def run_chatbot():
    print("SimpleBot (Databricks). Type 'exit' to quit. Type 'sentiment' to see this feature described.")
    while True:
        try:
            user = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nExiting. Bye!")
            break
        if not user:
            print("Bot: Please type something or 'help'.")
            continue
        if user.lower() in ("exit", "quit"):
            print("Bot:", RESPONSES["goodbye"])
            break
        intent = detect_intent(user)
        resp = RESPONSES.get(intent, RESPONSES["fallback"])
        # Get sentiment for all user inputs (optional). If you prefer "on demand", do this only when intent == 'sentiment'
        label, score = get_sentiment(user)
        # Make friendly sentiment text
        if label == "ERROR":
            sentiment_text = "(Sentiment: error analyzing)"
        elif label == "NEUTRAL":
            sentiment_text = "(Sentiment: neutral)"
        else:
            pct = round(score * 100, 1)
            sentiment_text = f"(Sentiment: {label}, confidence {pct}%)"
        # Print bot response and sentiment
        print("Bot:", resp, sentiment_text)

# Run the chat loop (in Dtbricks, this cell will wait for input)
# run_chatbot()


# COMMAND ----------

# If interactive input is inconvenient in Databricks UI, test with a list of sample messages:
sample_messages = [
    "hi there",
    "what can you do?",
    "I hate this service, it's terrible",
    "I love how fast this is!",
    "what are your hours",
    "random gibberish qwerty"
]

for msg in sample_messages:
    intent = detect_intent(msg)
    resp = RESPONSES.get(intent, RESPONSES["fallback"])
    label, score = get_sentiment(msg)
    print("You:", msg)
    if label in ("ERROR", "NEUTRAL"):
        print("Bot:", resp, f"(Sentiment: {label})\n")
    else:
        print("Bot:", resp, f"(Sentiment: {label} {round(score*100,1)}%)\n")
