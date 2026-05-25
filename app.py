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
# Haven Backend - AI Robustness V3 FINAL
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


def normalize_message(message: str) -> str:
    if not message:
        return ""

    msg = message.lower().strip()

    msg = re.sub(r"[^\w\s,.?!']", " ", msg)
    msg = re.sub(r"\s+", " ", msg)

    words = msg.split()

    converted_words = []

    for word in words:
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
    msg = user_message.lower().strip()

    for key, value in SMALL_TALK_RESPONSES.items():
        if key in msg:
            return value

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


# ================= Rule-Based Risk Analysis =================

def rule_based_risk_analysis(user_message: str, topic: str = "General") -> dict:

    msg = normalize_message(user_message)

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

    if contains_any(msg, STUDY_PHRASES) or topic == "Study":
        return {
            "risk_level": "low",
            "emotion": "stressed",
            "intent": "study",
            "needs_counselor": False,
            "reason": "Study-related conversation detected.",
        }

    if contains_any(msg, RELATIONSHIP_PHRASES) or topic == "Relationship":
        return {
            "risk_level": "low",
            "emotion": "sad",
            "intent": "relationship",
            "needs_counselor": False,
            "reason": "Relationship-related conversation detected.",
        }

    if contains_any(msg, CAREER_PHRASES) or topic == "Career":
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
            "⚠️ CRITICAL ALERT\n"
            "I’m really sorry you’re feeling this way. You are not alone. "
            "A university counselor has been notified so you can receive proper support. "
            "If you are in immediate danger, please call 999 or contact someone you trust now."
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
) -> dict:

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
  "emotion": "neutral" | "stressed" | "sad" | "anxious" | "angry" | "lonely" | "hopeless" | "overwhelmed",
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

        return result

    except Exception as e:

        print(f"❌ Gemini error: {e}")

        result = base_risk.copy()

        result["response"] = (
            local_reply(user_message, topic, result)
            or "I’m here with you. Tell me more about what’s been bothering you lately."
        )

        return result


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
                "timestamp": datetime.datetime.now(),

            })

        print("✅ Conversation saved")

    except Exception as e:
        print(f"❌ Conversation save error: {e}")


# ================= Save Counselor Alert =================

def save_counselor_alert(user_uid, user_message, result):

    if not db or not user_uid:
        return

    try:

        if result.get("risk_level") != "high":
            return

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

                "risk_level": "high",
                "riskLevel": "high",

                "emotion": result.get("emotion", "unknown"),
                "intent": result.get("intent", "crisis"),

                "summary": result.get("reason", ""),
                "riskSummary": result.get("reason", ""),
                "aiSummary": result.get("reason", ""),

                "updated_at": now,
                "updatedAt": now,

                "risk_count": firestore.Increment(1),
                "urgency": urgency,

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
            result
        )

        return jsonify({

            "response": result.get("response", "I’m here with you."),

            "risk_level": result.get("risk_level", "low"),

            "emotion": result.get("emotion", "unknown"),

            "intent": result.get("intent", "general_chat"),

            "needs_counselor": result.get("needs_counselor", False),

            "reason": result.get("reason", ""),

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

    })


# ================= Main =================

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=True
    )