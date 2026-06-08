import os
import json
import re
import datetime

import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai

from autocorrect import Speller
from langdetect import detect

# ============================================================
# Haven Backend - V7 Ultimate AI Logic Upgrade
# ============================================================

# ================= Firebase Database Connection =================
db = None

try:
    cred = credentials.Certificate("serviceAccountKey.json")

    firebase_admin.initialize_app(cred)
    db = firestore.client()

    print("✅ Firebase database connected successfully!")

except Exception as e:
    print(f"❌ Firebase connection failed: {e}")

# ================= Gemini AI Connection =================
gemini_client = None

try:
    api_key = os.environ.get("GEMINI_API_KEY")

    if not api_key:
        raise ValueError("No GEMINI_API_KEY found in environment variable.")

    gemini_client = genai.Client(api_key=api_key)

    print("✅ Gemini AI connected successfully!")

except Exception as e:
    print(f"❌ Gemini connection failed: {e}")

# ================= Flask App =================
app = Flask(__name__)
CORS(app)

GEMINI_MODEL = "gemini-2.0-flash"

# V5: Keep demo recording stable without removing Gemini integration.
# When HAVEN_DEMO_MODE is true, the backend gives deterministic high-quality
# replies for common demo cases, then falls back to Gemini/local rules for others.
DEMO_MODE = os.environ.get("HAVEN_DEMO_MODE", "true").lower() in ["1", "true", "yes", "on"]

# ================= Smart Input Processing =================

spell = Speller(lang='en')

SLANG_MAP = {
    "im": "i am",
    "idk": "i don't know",
    "rn": "right now",
    "pls": "please",
    "u": "you",
    "ur": "your",
    "cant": "can't",
    "dont": "don't",
    "wanna": "want to",
    "gonna": "going to",
    "tbh": "to be honest",
    "bc": "because",
    "wtf": "what the heck",
    "omg": "oh my god",
    "lol": "laughing",
    "lmao": "laughing",
    "bro": "friend",
    "nah": "no",
    "yup": "yes",
    "tho": "though",
}


# Common spelling mistakes that appear during student demo / real student typing.
# These are corrected before the autocorrect package runs, because autocorrect
# may sometimes change short words into unrelated words.
TYPO_MAP = {
    "strssed": "stressed",
    "stresed": "stressed",
    "stressd": "stressed",
    "abot": "about",
    "exm": "exam",
    "focs": "focus",
    "focuss": "focus",
    "studdy": "study",
    "asignment": "assignment",
    "assignmnt": "assignment",
    "presntation": "presentation",
    "presentaion": "presentation",
    "anxius": "anxious",
    "lonly": "lonely",
    "depresed": "depressed",
    "suicdal": "suicidal",
}

INDIRECT_HIGH_RISK = [
    "tired of existing",
    "nothing matters anymore",
    "everyone would be happier without me",
    "i feel trapped",
    "i can't do this anymore",
    "i cant do this anymore",
    "i feel done with life",
    "life feels pointless",
    "i want to disappear",
    "i'm mentally exhausted",
    "im mentally exhausted",
    "everything feels meaningless",
    "i don't see a future",
    "i dont see a future",
    "i am done",
    "i'm done",
    "im done",
    "i want everything to stop",
    "i can't continue",
    "i cant continue",
    "i cannot continue",
]

SMALL_TALK_RESPONSES = {
    "hello": "Hi, I’m here for you. How has your day been going lately?",
    "hi": "Hey, I’m listening. What’s been on your mind today?",
    "hey": "Hey there. How are you feeling today?",
    "goodnight": "Goodnight 🌙 I hope tomorrow feels a little lighter for you.",
    "thanks": "You’re always welcome. I’m glad you reached out.",
    "thank you": "You’re welcome 💙 I’m here anytime you need someone to talk to.",
    "haha": "I’m glad I could make the conversation feel a little lighter 😄",
    "lol": "Glad that made you smile a little 😄",
}

POSITIVE_EMOTION_PHRASES = [
    "happy",
    "glad",
    "excited",
    "proud",
    "great",
    "good result",
    "good results",
    "passed",
    "pass my subject",
    "passed my subject",
    "received good results",
    "good news",
    "relieved",
    "grateful",
    "satisfied",
    "i did well",
    "i got good marks",
    "i got good results",
    "i feel better",
]



NEGATED_POSITIVE_CONTEXTS = [
    "used to make me happy",
    "doesn't make me happy",
    "doesnt make me happy",
    "do not make me happy",
    "don't make me happy",
    "not happy",
    "no longer happy",
    "not really happy",
    "cannot feel happy",
    "can't feel happy",
    "hard to feel happy",
]

EMOTIONAL_DISTRESS_PHRASES = [
    "emotionally exhausted",
    "mentally exhausted",
    "physically exhausted",
    "do not really enjoy",
    "don't really enjoy",
    "does not enjoy",
    "doesn't enjoy",
    "lost interest",
    "no longer enjoy",
    "nothing feels enjoyable",
    "things that used to make me happy",
    "feel empty",
    "feel numb",
    "burnt out",
    "burned out",
    "overwhelmed",
    "exhausted",
    "tired almost every day",
]

BURNOUT_PHRASES = [
    "emotionally exhausted",
    "mentally exhausted",
    "no motivation",
    "lost motivation",
    "feel numb",
    "nothing feels enjoyable",
    "no longer enjoy",
    "used to make me happy",
    "burnt out",
    "burned out",
    "drained",
]

LONELINESS_PHRASES = [
    "nobody understands me",
    "no one understands me",
    "feel alone",
    "feel very alone",
    "feel lonely",
    "i am lonely",
    "i'm lonely",
]

CRITICAL_CONTEXT_PHRASES = [
    "tonight",
    "right now",
    "i have a plan",
    "i plan to",
    "i will kill myself",
    "after this",
]


def is_positive_emotion(message: str) -> bool:
    """Detect genuinely positive messages without misreading negated happiness.

    Example that should NOT be positive:
    "I do not enjoy the things that used to make me happy."
    """
    msg = message.lower()

    if contains_any(msg, NEGATED_POSITIVE_CONTEXTS):
        return False

    if contains_any(msg, EMOTIONAL_DISTRESS_PHRASES):
        return False

    return contains_any(msg, POSITIVE_EMOTION_PHRASES)


DEMO_TYPO_STUDY_PHRASES = [
    "strssed",
    "abot",
    "exm",
    "focs",
    "studdy",
    "asignment",
    "assignmnt",
    "presntation",
    "presentaion",
    "anxius",
    "lonly",
    "depresed",
    "suicdal",
]


def normalize_message(message: str) -> str:
    if not message:
        return ""

    msg = message.lower().strip()

    msg = re.sub(r"[^\w\s,.?!']", " ", msg)
    msg = re.sub(r"\s+", " ", msg)

    words = msg.split()

    converted_words = []

    for word in words:
        word = TYPO_MAP.get(word, word)
        converted_words.append(SLANG_MAP.get(word, word))

    msg = " ".join(converted_words)

    try:
        corrected = spell(msg)

        if len(corrected.split()) >= len(msg.split()) * 0.7:
            msg = corrected

    except Exception:
        pass

    return msg.strip()


def detect_language_safe(text: str) -> str:
    try:
        return detect(text)
    except Exception:
        return "unknown"


def smart_local_response(user_message: str):
    """Return local small-talk replies only for very short greeting messages.

    This prevents demo messages such as "Hi Haven, I feel happy..." from being
    wrongly treated as a simple greeting.
    """
    msg = user_message.lower().strip()
    msg_clean = re.sub(r"[^a-zA-Z\s']", "", msg).strip()
    words = msg_clean.split()

    # Only trigger greeting fallback when the whole message is short.
    # Example: "hi", "hello", "hey haven".
    if len(words) <= 3:
        for key, value in SMALL_TALK_RESPONSES.items():
            if msg_clean == key or msg_clean.startswith(f"{key} "):
                return value

    # Thanks / goodnight can still be handled locally when short.
    if len(words) <= 5:
        for key in ["thanks", "thank you", "goodnight", "haha", "lol"]:
            if key in msg_clean:
                return SMALL_TALK_RESPONSES.get(key)

    return None


# ================= Risk Phrases =================

HIGH_RISK_PHRASES = [
    "suicide",
    "kill myself",
    "end my life",
    "hurt myself",
    "self harm",
    "self-harm",
    "cut myself",
    "i want to die",
    "i wanna die",
    "i don't want to live",
    "i dont want to live",
    "no reason to live",
    "want to end it",
    "i don't want to wake up",
    "i dont want to wake up",
    "don't want to wake up",
    "dont want to wake up",
    "everyone would be better without me",
    "people would be better without me",
    "my family would be better without me",
    "i want to disappear forever",
    "i wish i was dead",
    "i wish i were dead",
    "ending my life",
    "end my life",
    "give up on life",
    "giving up on life",
    "i cannot continue anymore",
    "i can't continue anymore",
    "i cannot go on anymore",
    "i can't go on anymore",
    "i don't want to continue",
    "i dont want to continue",
    "i will kill myself",
    "i might kill myself",
    "i plan to kill myself",
    "i want everything to stop",
    "i don't see a future",
    "i dont see a future",
    "nothing matters anymore",
]

MEDIUM_RISK_PHRASES = [
    "i feel hopeless",
    "i feel useless",
    "i am useless",
    "i'm useless",
    "i feel empty",
    "i hate my life",
    "i'm tired of everything",
    "im tired of everything",
    "i feel alone",
    "i feel lonely",
    "nobody understands me",
    "no one understands me",
    "i can't take it anymore",
    "i cant take it anymore",
    "i feel worthless",
    "i am worthless",
    "i'm worthless",
    "i feel like giving up",
    "i want to give up",
    "i feel overwhelmed",
    "i am overwhelmed",
    "i'm overwhelmed",
    "i feel exhausted",
    "i am exhausted",
    "i'm exhausted",
    "i cannot focus",
    "i can't focus",
]

STUDY_PHRASES = [
    "assignment",
    "exam",
    "study",
    "school",
    "class",
    "grade",
    "fail",
    "deadline",
    "presentation",
    "project",
    "fyp",
]

RELATIONSHIP_PHRASES = [
    "relationship",
    "girlfriend",
    "boyfriend",
    "breakup",
    "break up",
    "love",
    "crush",
    "friendship",
    "family problem",
]

CAREER_PHRASES = [
    "career",
    "future",
    "job",
    "internship",
    "work",
    "interview",
    "graduate",
    "salary",
]


def contains_any(message: str, phrases: list[str]) -> bool:
    msg = message.lower()
    return any(phrase in msg for phrase in phrases)



def detect_academic_category(message: str) -> str:
    msg = message.lower()
    if contains_any(msg, ["presentation", "present", "fyp"]):
        return "presentation anxiety"
    if contains_any(msg, ["exam", "quiz", "test"]):
        return "exam anxiety"
    if contains_any(msg, ["assignment", "deadline", "project work"]):
        return "assignment overload"
    if contains_any(msg, ["fail", "disappoint", "graduate"]):
        return "fear of failure"
    return "general academic stress"


def detect_burnout(message: str) -> bool:
    return contains_any(message, BURNOUT_PHRASES)


def detect_risk_severity(message: str, level: str) -> str:
    msg = message.lower()
    if level == "high" and contains_any(msg, CRITICAL_CONTEXT_PHRASES):
        return "critical"
    if level == "high":
        return "high"
    if level == "medium":
        return "medium"
    return "low"


def build_counselor_summary(user_message: str, result: dict) -> str:
    risk = result.get("risk_level", "low")
    emotion = result.get("emotion", "unknown")
    intent = result.get("intent", "general_chat")
    trend = result.get("emotion_trend", [])
    trend_text = " → ".join(trend) if trend else emotion
    if risk == "high":
        return (
            "Student expressed thoughts related to self-harm or suicide. "
            f"Current emotional state appears to be {emotion}, with intent category {intent}. "
            f"Emotion trend: {trend_text}. Immediate counselor review is recommended."
        )
    return (
        f"Student appears to be experiencing {emotion} with intent category {intent}. "
        f"Emotion trend: {trend_text}. Message indicates ongoing emotional pressure."
    )



def infer_secondary_emotion(message: str, primary: str) -> str:
    msg = message.lower()

    negated_positive_context = [
        "used to make me happy",
        "used to be happy",
        "do not really enjoy",
        "don't really enjoy",
        "do not enjoy",
        "don't enjoy",
        "no longer enjoy",
        "lost interest",
        "pretend to be happy",
        "not happy",
    ]

    allow_positive_secondary = not contains_any(msg, negated_positive_context)

    emotion_checks = []
    if allow_positive_secondary:
        emotion_checks.append(("happy", POSITIVE_EMOTION_PHRASES))

    emotion_checks += [
        ("anxious", ["worry", "worried", "anxious", "nervous", "presentation", "future"]),
        ("stressed", ["stress", "stressed", "assignment", "exam", "deadline", "project", "fyp"]),
        ("sad", ["sad", "cry", "empty", "worthless", "useless"]),
        ("lonely", ["alone", "lonely", "nobody understands", "no one understands"]),
        ("hopeless", ["hopeless", "giving up", "end my life", "die", "suicide"]),
    ]
    for emotion, phrases in emotion_checks:
        if emotion != primary and contains_any(msg, phrases):
            return emotion
    return "none"


def risk_score_from_level(level: str, message: str = "") -> int:
    msg = message.lower()
    if level == "high":
        if any(x in msg for x in ["tonight", "now", "plan", "will kill", "i will"]):
            return 95
        return 85
    if level == "medium":
        return 55
    return 15


def build_emotion_trend(history, current_emotion: str) -> list[str]:
    trend = []
    if isinstance(history, list):
        for item in history[-4:]:
            emotion = str(item.get("emotion", "")).strip().lower()
            if emotion and emotion not in ["unknown", "none"]:
                trend.append(emotion)
    if current_emotion and current_emotion not in ["unknown", "none"]:
        trend.append(current_emotion)
    # remove repeated neighbours only
    compact = []
    for item in trend:
        if not compact or compact[-1] != item:
            compact.append(item)
    return compact[-5:]


def ensure_result_schema(result: dict, base_risk: dict, user_message: str, history=None) -> dict:
    """Force a safe JSON shape so Flutter never receives missing fields."""
    if not isinstance(result, dict):
        result = {}

    risk = str(result.get("risk_level", base_risk.get("risk_level", "low"))).lower()
    if risk not in ["low", "medium", "high"]:
        risk = str(base_risk.get("risk_level", "low")).lower()

    # Rule-based safety is allowed to upgrade, but Gemini cannot downgrade clear risk.
    if base_risk.get("risk_level") == "high":
        risk = "high"
    if base_risk.get("risk_level") == "medium" and risk == "low":
        risk = "medium"

    emotion = str(result.get("emotion", base_risk.get("emotion", "neutral"))).lower()
    allowed_emotions = ["neutral", "happy", "stressed", "sad", "anxious", "angry", "lonely", "hopeless", "overwhelmed", "burnout", "tired"]
    if emotion not in allowed_emotions:
        emotion = str(base_risk.get("emotion", "neutral")).lower()
        if emotion not in allowed_emotions:
            emotion = "neutral"

    intent = str(result.get("intent", base_risk.get("intent", "general_chat"))).lower()
    allowed_intents = ["general_chat", "study", "relationship", "career", "emotional_support", "crisis"]
    if intent not in allowed_intents:
        intent = str(base_risk.get("intent", "general_chat")).lower()
        if intent not in allowed_intents:
            intent = "general_chat"

    result["risk_level"] = risk
    result["emotion"] = emotion
    result["intent"] = intent
    result["needs_counselor"] = bool(result.get("needs_counselor", risk == "high"))
    if risk == "high":
        result["needs_counselor"] = True

    result["reason"] = result.get("reason") or base_risk.get("reason", "")
    result["response"] = result.get("response") or local_reply(user_message, "General", result)
    # Give-up language should be treated as hopelessness, not only sadness.
    if risk == "medium" and contains_any(user_message.lower(), ["feel like giving up", "giving up", "everything feels too heavy", "too heavy for me"]):
        emotion = "hopeless"
        result["emotion"] = "hopeless"
        result["risk_score"] = max(int(result.get("risk_score", 0) or 0), 65)

    result["risk_score"] = int(result.get("risk_score", risk_score_from_level(risk, user_message)))
    result["primary_emotion"] = result.get("primary_emotion", emotion)
    if result["primary_emotion"] == "sad" and emotion == "hopeless":
        result["primary_emotion"] = "hopeless"
    result["secondary_emotion"] = result.get("secondary_emotion", infer_secondary_emotion(user_message, emotion))
    if result.get("secondary_emotion") == "happy" and contains_any(user_message.lower(), ["used to make me happy", "used to be happy", "do not really enjoy", "don't really enjoy", "do not enjoy", "don't enjoy", "no longer enjoy", "lost interest", "pretend to be happy", "not happy"]):
        result["secondary_emotion"] = "sad" if emotion != "sad" else "none"

    result["emotion_trend"] = result.get("emotion_trend", build_emotion_trend(history or [], emotion))
    result["risk_severity"] = result.get("risk_severity", detect_risk_severity(user_message, risk))
    result["academic_category"] = result.get("academic_category", detect_academic_category(user_message) if intent == "study" else "not academic")
    result["counselor_summary"] = result.get("counselor_summary", build_counselor_summary(user_message, result))

    return result


def demo_reply_and_analysis(user_message: str, topic: str, base_risk: dict, history=None) -> dict | None:
    """Deterministic replies for demo recording stability.

    This does not replace the whole AI system. It only catches common demo cases
    so the video remains stable even if Gemini is slow, quota-limited, or returns
    a weaker response.
    """
    if not DEMO_MODE:
        return None

    msg = normalize_message(user_message)
    raw = user_message.lower()

    # High-risk case must be stable and must create counselor alert.
    if base_risk.get("risk_level") == "high":
        result = {
            "response": (
                "💙 Haven Support Response\n"
                "I’m really sorry you’re feeling this way. What you shared sounds serious, and I’m concerned about your well-being. "
                "You do not have to face these feelings alone. A university counselor has been notified so additional support can be provided. "
                "For now, would you like to tell me more about what has been leading to these thoughts recently?"
            ),
            "risk_level": "high",
            "emotion": "hopeless",
            "intent": "crisis",
            "needs_counselor": True,
            "reason": "High-risk self-harm or suicidal ideation phrase detected.",
        }
        return ensure_result_schema(result, base_risk, user_message, history)

    # Emotional exhaustion / loss of enjoyment must be handled before positive words.
    # This prevents "things that used to make me happy" from becoming a happy reply.
    if detect_burnout(msg) or contains_any(msg, EMOTIONAL_DISTRESS_PHRASES):
        result = {
            "response": (
                "It sounds like you have been emotionally drained for a while, and losing interest in things you used to enjoy can feel really heavy. "
                "That does not mean you are weak; it may be a sign that you have been carrying too much for too long. What has been affecting you the most recently?"
            ),
            "risk_level": "medium",
            "emotion": "burnout",
            "intent": "emotional_support",
            "needs_counselor": False,
            "reason": "Burnout, emotional exhaustion, or loss of enjoyment detected.",
        }
        return ensure_result_schema(result, base_risk, user_message, history)

    # Loneliness / isolation case.
    if contains_any(msg, LONELINESS_PHRASES):
        result = {
            "response": (
                "Feeling alone while going through something difficult can make everything feel even heavier. "
                "I’m sorry it has felt like nobody really understands you. You do not have to carry this silently here — what part has been the hardest to explain to others?"
            ),
            "risk_level": "medium",
            "emotion": "lonely",
            "intent": "emotional_support",
            "needs_counselor": False,
            "reason": "Loneliness and emotional isolation detected.",
        }
        return ensure_result_schema(result, base_risk, user_message, history)

    # Positive + study transition.
    if is_positive_emotion(msg):
        result = {
            "response": (
                "That’s really good to hear. You should feel proud of yourself because your effort has paid off. "
                "It is also okay if you still have other worries later, but for now this is a positive moment worth recognising."
            ),
            "risk_level": "low",
            "emotion": "happy",
            "intent": "general_chat",
            "needs_counselor": False,
            "reason": "Positive emotional expression detected.",
        }
        return ensure_result_schema(result, base_risk, user_message, history)

    # Misspelled / long academic stress case.
    if contains_any(raw, DEMO_TYPO_STUDY_PHRASES) or (contains_any(msg, STUDY_PHRASES) and contains_any(msg, ["tired", "sleeping very late", "cannot focus", "can't focus", "worry", "fail"])):
        result = {
            "response": (
                "Your message sounds like academic stress is starting to affect your focus and energy. "
                "When exams, assignments, and lack of sleep happen together, it can become difficult to think clearly. Try choosing one small task to complete first, then take a short reset before continuing."
            ),
            "risk_level": "medium",
            "emotion": "stressed",
            "intent": "study",
            "needs_counselor": False,
            "reason": "Academic stress detected from typo or long study-related message.",
        }
        return ensure_result_schema(result, base_risk, user_message, history)

    # General study worry.
    if contains_any(msg, STUDY_PHRASES) or str(topic).lower() in ["study", "academic stress"]:
        result = {
            "response": (
                "It makes sense that you feel stressed when your presentation and assignments are happening close together. "
                "You do not need to solve everything at once. What is the most urgent task you need to handle first?"
            ),
            "risk_level": base_risk.get("risk_level", "low"),
            "emotion": "stressed",
            "intent": "study",
            "needs_counselor": False,
            "reason": "Study-related concern detected.",
        }
        return ensure_result_schema(result, base_risk, user_message, history)

    # Medium emotional distress.
    if base_risk.get("risk_level") == "medium":
        result = {
            "response": (
                "I’m sorry you’re feeling this overwhelmed. It sounds like you have been carrying a lot by yourself, "
                "and it can feel painful when nobody seems to understand. You do not need to explain everything perfectly here — I’m listening."
            ),
            "risk_level": "medium",
            "emotion": base_risk.get("emotion", "stressed"),
            "intent": base_risk.get("intent", "emotional_support"),
            "needs_counselor": False,
            "reason": base_risk.get("reason", "Medium emotional distress detected."),
        }
        return ensure_result_schema(result, base_risk, user_message, history)

    return None

# ================= Rule-Based Risk Analysis =================

def rule_based_risk_analysis(user_message: str, topic: str = "General") -> dict:

    msg = normalize_message(user_message)
    raw_msg = user_message.lower().strip()

    false_positive_phrases = [
        "assignment is killing me",
        "exam is killing me",
        "homework is killing me",
        "this project is killing me",
        "this fyp is killing me",
    ]

    if contains_any(msg, false_positive_phrases):
        return {
            "risk_level": "medium",
            "emotion": "stressed",
            "intent": "study",
            "needs_counselor": False,
            "reason": "Detected stress slang, not direct self-harm intent.",
        }

    # Demo typo protection: keep common misspelled academic stress messages stable.
    if contains_any(raw_msg, DEMO_TYPO_STUDY_PHRASES):
        return {
            "risk_level": "medium",
            "emotion": "stressed",
            "intent": "study",
            "needs_counselor": False,
            "reason": "Study stress detected from misspelled or unclear message.",
        }

    if contains_any(msg, INDIRECT_HIGH_RISK):
        return {
            "risk_level": "high",
            "emotion": "hopeless",
            "intent": "crisis",
            "needs_counselor": True,
            "reason": "Indirect high-risk emotional distress detected.",
        }

    if contains_any(msg, HIGH_RISK_PHRASES):
        return {
            "risk_level": "high",
            "emotion": "hopeless",
            "intent": "crisis",
            "needs_counselor": True,
            "reason": "High-risk self-harm or suicidal ideation phrase detected.",
        }

    if contains_any(msg, MEDIUM_RISK_PHRASES):
        return {
            "risk_level": "medium",
            "emotion": "sad",
            "intent": "emotional_support",
            "needs_counselor": False,
            "reason": "Medium-risk emotional distress phrase detected.",
        }

    if detect_burnout(msg) or contains_any(msg, EMOTIONAL_DISTRESS_PHRASES):
        return {
            "risk_level": "medium",
            "emotion": "burnout",
            "intent": "emotional_support",
            "needs_counselor": False,
            "reason": "Burnout, emotional exhaustion, or loss of enjoyment detected.",
        }

    if is_positive_emotion(msg):
        return {
            "risk_level": "low",
            "emotion": "happy",
            "intent": "general_chat",
            "needs_counselor": False,
            "reason": "Positive emotional expression detected.",
        }

    topic_lower = str(topic).lower()

    if contains_any(msg, STUDY_PHRASES) or topic_lower in ["study", "academic stress"]:
        return {
            "risk_level": "low",
            "emotion": "stressed",
            "intent": "study",
            "needs_counselor": False,
            "reason": "Study-related conversation detected.",
        }

    if contains_any(msg, RELATIONSHIP_PHRASES) or topic_lower == "relationship":
        return {
            "risk_level": "low",
            "emotion": "sad",
            "intent": "relationship",
            "needs_counselor": False,
            "reason": "Relationship-related conversation detected.",
        }

    if contains_any(msg, CAREER_PHRASES) or topic_lower == "career":
        return {
            "risk_level": "low",
            "emotion": "anxious",
            "intent": "career",
            "needs_counselor": False,
            "reason": "Career-related conversation detected.",
        }

    if any(x in msg for x in [
        "sad",
        "stress",
        "stressed",
        "anxious",
        "angry",
        "lonely",
        "cry",
        "burnt out",
        "overwhelmed",
        "exhausted",
    ]):
        return {
            "risk_level": "medium",
            "emotion": "stressed",
            "intent": "emotional_support",
            "needs_counselor": False,
            "reason": "General emotional distress detected.",
        }

    return {
        "risk_level": "low",
        "emotion": "neutral",
        "intent": "general_chat",
        "needs_counselor": False,
        "reason": "No significant risk phrase detected.",
    }


# ================= Local Reply =================

def local_reply(user_message: str, topic: str, risk_data: dict) -> str:

    msg = normalize_message(user_message)

    risk = risk_data.get("risk_level", "low")
    intent = risk_data.get("intent", "general_chat")

    small_talk = smart_local_response(msg)

    if small_talk:
        return small_talk

    if is_positive_emotion(msg):
        return (
            "That’s really good to hear. It sounds like your effort has paid off, "
            "and you deserve to feel proud of that progress. What helped you get through it?"
        )

    if any(x in msg for x in ["nvm", "never mind", "forget it"]):
        return (
            "That’s okay. Sometimes it can be hard to explain everything immediately. "
            "I’m still here if you want to talk about it."
        )

    if any(x in msg for x in ["tired", "exhausted", "burnt out", "burned out"]):
        return (
            "You sound emotionally exhausted lately. "
            "University life can become overwhelming when too many things pile up at once. "
            "What has been draining you the most recently?"
        )

    if any(x in msg for x in ["alone", "lonely"]):
        return (
            "Feeling alone can be really heavy emotionally, especially when it feels like nobody fully understands what you're going through. "
            "Do you want to talk about what has been making you feel this way?"
        )

    if risk == "high":
        return (
            "💙 Haven Support Response\n"
            "I’m really sorry you’re feeling this way. What you shared sounds serious, and I’m concerned about your well-being. "
            "You do not have to face these feelings alone. A university counselor has been notified so additional support can be provided. "
            "For now, would you like to tell me more about what has been leading to these thoughts recently?"
        )

    if "who are you" in msg or "what are you" in msg:
        return (
            "I’m Haven, a supportive AI companion designed for university students. "
            "You can talk to me about study stress, relationships, career worries, or daily problems. "
            "I’m not a professional counselor, but I can support you and help notify a counselor if something serious appears."
        )

    if intent == "study":
        return (
            "That sounds stressful, especially when university work starts piling up. "
            "What part has been affecting you the most recently?"
        )

    if intent == "relationship":
        return (
            "Relationship problems can feel emotionally heavy because they involve people you care about deeply. "
            "Do you want to tell me what happened?"
        )

    if intent == "career":
        return (
            "Thinking about the future can feel overwhelming sometimes, especially during university life. "
            "What has been worrying you the most recently?"
        )

    if risk == "medium":
        return (
            "I’m sorry you’re feeling this way. "
            "It sounds like you’ve been carrying a lot emotionally lately. "
            "You don’t need to explain everything perfectly — I’m listening."
        )

    return (
        "I’m here with you. "
        "Tell me a little more about what has been on your mind lately."
    )


# ================= Extract JSON =================

def extract_json(text: str) -> dict | None:
    try:
        return json.loads(text)

    except Exception:
        pass

    match = re.search(r"\{.*\}", text, re.DOTALL)

    if match:
        try:
            return json.loads(match.group(0))

        except Exception:
            return None

    return None


# ================= Build History =================

def build_history_text(history) -> str:

    if not isinstance(history, list):
        return ""

    lines = []

    for item in history[-10:]:

        sender = str(item.get("sender", "unknown"))
        text = str(item.get("text", "")).strip()

        emotion = str(item.get("emotion", "unknown"))
        risk = str(item.get("risk_level", "low"))

        if text:
            lines.append(
                f"{sender} ({emotion}, {risk}): {text}"
            )

    return "\n".join(lines)


# ================= Gemini AI =================

def gemini_reply_and_analysis(
    user_message: str,
    gender: str,
    topic: str,
    base_risk: dict,
    history_text: str = "",
    history=None,
) -> dict:

    demo_result = demo_reply_and_analysis(user_message, topic, base_risk, history)
    if demo_result:
        return demo_result

    if not gemini_client:

        result = base_risk.copy()

        result["response"] = local_reply(
            user_message,
            topic,
            result
        )

        return result

    topic_guidance = {
        "Study": "Focus on academic stress, assignments, exams, deadlines, motivation, and fear of failure.",
        "Relationship": "Focus on relationships, communication, conflict, loneliness, heartbreak, and emotional pain.",
        "Career": "Focus on future anxiety, internship, job worries, confidence, career planning, and decision making.",
        "Just Chat": "Respond as a supportive companion for general daily conversation.",
    }.get(topic, "Respond as a supportive AI companion for university students.")

    prompt = f"""
You are Haven, an empathetic and emotionally supportive AI companion designed for university students.

Important understanding rules:
- The user may make spelling mistakes.
- The user may use internet slang or short forms.
- The user may mix English, Malay, and Chinese.
- The user may express emotions indirectly.
- The user may send emotionally messy or unclear messages.
- Focus on emotional meaning, not only literal wording.
- If multiple emotional problems appear, identify the MAIN emotional concern first.
- Continue conversations naturally even when the user's message is short or vague.
- Never sound robotic or repetitive.

Your personality:
- Warm
- Calm
- Emotionally supportive
- Natural
- Human-like

Your goals:
1. Help students feel emotionally heard.
2. Continue conversations naturally.
3. Encourage emotional expression safely.
4. Ask meaningful follow-up questions.
5. Detect emotional distress carefully.
6. Encourage counselor support only when necessary.

Conversation style rules:
- Response should usually be 3–6 sentences.
- Keep most responses under 90 words.
- Avoid robotic wording.
- Avoid repetitive phrases.
- Do not sound like a textbook therapist.
- Maintain emotional continuity.
- Sound emotionally alive and conversational.

Safety rules:
- Never encourage self-harm or suicide.
- Never diagnose mental illness.
- In high-risk situations, encourage immediate human support.
- Mention counselor support if severe emotional distress appears.

Initial risk assessment:
{json.dumps(base_risk)}

Return ONLY valid JSON:

{{
  "response": "your chatbot reply",
  "risk_level": "low" | "medium" | "high",
  "emotion": "neutral" | "happy" | "stressed" | "sad" | "anxious" | "angry" | "lonely" | "hopeless" | "overwhelmed",
  "intent": "general_chat" | "study" | "relationship" | "career" | "emotional_support" | "crisis",
  "needs_counselor": true | false,
  "reason": "short reason"
}}

User profile:
- Gender: {gender}
- Selected topic: {topic}
- Topic guidance: {topic_guidance}

Recent conversation:
{history_text}

User message:
{user_message}
"""

    try:

        response = gemini_client.models.generate_content(
            model=GEMINI_MODEL,
            contents=prompt,
        )

        result = extract_json(response.text or "")

        if not result:

            print("⚠️ Gemini returned invalid JSON")

            result = base_risk.copy()

            result["response"] = local_reply(
                user_message,
                topic,
                result
            )

            return result

        risk = str(
            result.get(
                "risk_level",
                base_risk.get("risk_level", "low")
            )
        ).lower()

        if risk not in ["low", "medium", "high"]:
            risk = base_risk.get("risk_level", "low")

        if base_risk.get("risk_level") == "high":
            risk = "high"

        # Do not let Gemini downgrade clear rule-based medium distress to low.
        if base_risk.get("risk_level") == "medium" and risk == "low":
            risk = "medium"

        result["risk_level"] = risk

        result["emotion"] = result.get(
            "emotion",
            base_risk.get("emotion", "unknown")
        )

        result["intent"] = result.get(
            "intent",
            base_risk.get("intent", "general_chat")
        )

        result["needs_counselor"] = bool(
            result.get(
                "needs_counselor",
                risk == "high"
            )
        )

        result["reason"] = result.get(
            "reason",
            base_risk.get("reason", "")
        )

        result["response"] = result.get(
            "response"
        ) or local_reply(
            user_message,
            topic,
            result
        )

        if risk == "high":

            result["needs_counselor"] = True

            result["response"] = (
                "⚠️ CRITICAL ALERT\n"
                "I’m really sorry you’re feeling this way. "
                "You are not alone. "
                "A university counselor has been notified so you can receive proper support. "
                "If you are in immediate danger, please call 999 or contact someone you trust now."
            )

        return ensure_result_schema(result, base_risk, user_message, history)

    except Exception as e:

        print(f"❌ Gemini error: {e}")

        result = base_risk.copy()

        result["response"] = (
            local_reply(user_message, topic, result)
            or "I’m here with you. Tell me more about what’s been bothering you lately."
        )

        return ensure_result_schema(result, base_risk, user_message, history)


# ================= Update User Risk =================

def update_user_risk(user_uid, result, user_message):

    if not db or not user_uid:
        return

    try:

        db.collection("users").document(user_uid).set({

            "last_risk_level": result.get("risk_level", "low"),
            "last_emotion": result.get("emotion", "unknown"),
            "last_intent": result.get("intent", "general_chat"),
            "last_message": user_message,
            "needs_counselor": result.get("needs_counselor", False),
            "risk_score": result.get("risk_score", risk_score_from_level(result.get("risk_level", "low"), user_message)),
            "primary_emotion": result.get("primary_emotion", result.get("emotion", "unknown")),
            "secondary_emotion": result.get("secondary_emotion", "none"),
            "emotion_trend": result.get("emotion_trend", []),
            "last_updated": datetime.datetime.now(),

        }, merge=True)

        print("✅ User risk updated")

    except Exception as e:
        print(f"❌ User risk update error: {e}")


# ================= Save Conversation =================

def save_conversation(user_uid, user_message, ai_response, result):

    if not db or not user_uid:
        return

    try:

        db.collection("conversations") \
            .document(user_uid) \
            .collection("messages") \
            .add({

                "user_message": user_message,
                "ai_response": ai_response,
                "risk_level": result.get("risk_level", "low"),
                "emotion": result.get("emotion", "unknown"),
                "intent": result.get("intent", "general_chat"),
                "risk_score": result.get("risk_score", risk_score_from_level(result.get("risk_level", "low"), user_message)),
                "primary_emotion": result.get("primary_emotion", result.get("emotion", "unknown")),
                "secondary_emotion": result.get("secondary_emotion", "none"),
                "emotion_trend": result.get("emotion_trend", []),
                "risk_severity": result.get("risk_severity", result.get("risk_level", "low")),
                "academic_category": result.get("academic_category", "not academic"),
                "counselor_summary": result.get("counselor_summary", ""),
                "timestamp": datetime.datetime.now(),

            })

        print("✅ Conversation saved")

    except Exception as e:
        print(f"❌ Conversation save error: {e}")


# ================= Save Counselor Alert =================

def save_counselor_alert(user_uid, user_message, result, student_meta=None):

    if not db or not user_uid:
        return

    try:

        if result.get("risk_level") != "high":
            return

        student_meta = student_meta or {}
        now = datetime.datetime.now()

        message_preview = user_message[:250]

        existing_alert = db.collection("counselor_alerts") \
            .where("student_id", "==", user_uid) \
            .where("status", "==", "pending") \
            .limit(1) \
            .get()

        if len(existing_alert) > 0:

            doc_ref = existing_alert[0].reference

            old_data = existing_alert[0].to_dict() or {}

            risk_count = int(old_data.get("risk_count", 1)) + 1

            if risk_count >= 6:
                urgency = "critical"

            elif risk_count >= 3:
                urgency = "urgent"

            else:
                urgency = "monitor"

            doc_ref.update({

                "message_preview": message_preview,
                "triggerMessage": message_preview,
                "lastMessage": message_preview,
                "latest_message": message_preview,

                "student_name": student_meta.get("student_name", "Unknown Student"),
                "student_email": student_meta.get("student_email", ""),
                "topic": student_meta.get("topic", "General"),
                "gender": student_meta.get("gender", "Student"),

                "risk_level": "high",
                "riskLevel": "high",

                "emotion": result.get("emotion", "unknown"),
                "intent": result.get("intent", "crisis"),

                "summary": result.get("counselor_summary", result.get("reason", "")),
                "riskSummary": result.get("counselor_summary", result.get("reason", "")),
                "aiSummary": result.get("counselor_summary", result.get("reason", "")),
                "risk_severity": result.get("risk_severity", "high"),
                "academic_category": result.get("academic_category", "not academic"),

                "updated_at": now,
                "updatedAt": now,

                "risk_count": firestore.Increment(1),
                "urgency": urgency,
                "risk_score": result.get("risk_score", 85),
                "primary_emotion": result.get("primary_emotion", result.get("emotion", "hopeless")),
                "secondary_emotion": result.get("secondary_emotion", "none"),
                "emotion_trend": result.get("emotion_trend", []),

                "isResolved": False,

            })

            print("♻️ Existing counselor alert updated")

            return

        db.collection("counselor_alerts").add({

            "user_uid": user_uid,
            "student_id": user_uid,
            "user_id": user_uid,
            "uid": user_uid,

            "message_preview": message_preview,
            "triggerMessage": message_preview,
            "lastMessage": message_preview,
            "latest_message": message_preview,

            "student_name": student_meta.get("student_name", "Unknown Student"),
            "student_email": student_meta.get("student_email", ""),
            "topic": student_meta.get("topic", "General"),
            "gender": student_meta.get("gender", "Student"),

            "risk_level": "high",
            "riskLevel": "high",

            "emotion": result.get("emotion", "unknown"),
            "intent": result.get("intent", "crisis"),

            "summary": result.get("reason", ""),
            "riskSummary": result.get("reason", ""),
            "aiSummary": result.get("reason", ""),

            "status": "pending",
            "isResolved": False,

            "risk_count": 1,
            "urgency": "monitor",
            "risk_score": result.get("risk_score", 85),
            "primary_emotion": result.get("primary_emotion", result.get("emotion", "hopeless")),
            "secondary_emotion": result.get("secondary_emotion", "none"),
            "emotion_trend": result.get("emotion_trend", []),

            "created_at": now,
            "createdAt": now,

            "updated_at": now,
            "updatedAt": now,

        })

        print("🚨 Counselor alert created")

    except Exception as e:
        print(f"❌ Alert save error: {e}")


# ================= Chat Route =================

@app.route("/chat", methods=["POST"])
def chat():

    try:

        data = request.json or {}

        user_message = str(data.get("message", "")).strip()

        gender = str(data.get("gender", "Student"))

        topic = str(data.get("topic", "General"))

        student_name = str(data.get("student_name", data.get("name", "Unknown Student")))
        student_email = str(data.get("student_email", data.get("email", "")))

        user_uid = data.get("uid")

        history = data.get("history", [])

        normalized_message = normalize_message(user_message)

        language = detect_language_safe(user_message)

        history_text = build_history_text(history)

        print(f"🌐 Language: {language}")
        print(f"🧹 Normalized: {normalized_message}")

        if not user_message:

            return jsonify({

                "response": "Please type a message first.",
                "risk_level": "low",
                "emotion": "neutral",
                "intent": "general_chat",
                "needs_counselor": False,

            })

        print(f"📩 Message received: '{user_message}'")

        base_risk = rule_based_risk_analysis(
            normalized_message,
            topic
        )

        result = gemini_reply_and_analysis(
            normalized_message,
            gender,
            topic,
            base_risk,
            history_text,
            history,
        )

        print(f"🧠 Final result: {result}")

        update_user_risk(
            user_uid,
            result,
            user_message
        )

        save_conversation(
            user_uid,
            user_message,
            result.get("response", ""),
            result
        )

        save_counselor_alert(
            user_uid,
            user_message,
            result,
            {
                "student_name": student_name,
                "student_email": student_email,
                "topic": topic,
                "gender": gender,
            }
        )

        return jsonify({

            "response": result.get("response", "I’m here with you."),

            "risk_level": result.get("risk_level", "low"),

            "emotion": result.get("emotion", "unknown"),

            "intent": result.get("intent", "general_chat"),

            "needs_counselor": result.get("needs_counselor", False),

            "reason": result.get("reason", ""),
            "risk_score": result.get("risk_score", risk_score_from_level(result.get("risk_level", "low"), user_message)),
            "primary_emotion": result.get("primary_emotion", result.get("emotion", "unknown")),
            "secondary_emotion": result.get("secondary_emotion", "none"),
            "emotion_trend": result.get("emotion_trend", []),

        })

    except Exception as e:

        print(f"❌ System error: {e}")

        return jsonify({

            "response": "System Error. Please try again later.",

            "risk_level": "low",

            "emotion": "unknown",

            "intent": "unknown",

            "needs_counselor": False,

        }), 500


# ================= Home =================

@app.route("/", methods=["GET"])
def home():

    return jsonify({

        "status": "Haven backend is running",

        "gemini": "connected" if gemini_client else "not connected",

        "firebase": "connected" if db else "not connected",

        "model": GEMINI_MODEL,
        "demo_mode": DEMO_MODE,

    })


# ================= Main =================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )