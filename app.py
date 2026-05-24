import os
import json
import re
import datetime

import firebase_admin
from firebase_admin import credentials, firestore
from flask import Flask, request, jsonify
from flask_cors import CORS
from google import genai

# ============================================================
# Haven Backend - Stable Hybrid V2
# 1. Rule-based risk classification works without Gemini
# 2. Gemini improves normal conversation when available
# 3. Backend is the ONLY place that creates counselor alerts
# 4. Duplicate high-risk alerts are prevented
# 5. Existing pending alert will be updated as one active case
# ============================================================

# ================= Firebase Database Connection =================
db = None

try:
    cred_path = os.environ.get(
        "GOOGLE_APPLICATION_CREDENTIALS",
        "/etc/secrets/serviceAccountKey.json"
    )

    cred = credentials.Certificate(cred_path)

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


def rule_based_risk_analysis(user_message: str, topic: str = "General") -> dict:
    msg = user_message.lower()

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
            "emotion": "stressed"
            if any(x in msg for x in ["stress", "stressed", "pressure", "fail"])
            else "neutral",
            "intent": "study",
            "needs_counselor": False,
            "reason": "Study-related conversation detected.",
        }

    if contains_any(msg, RELATIONSHIP_PHRASES) or topic == "Relationship":
        return {
            "risk_level": "low",
            "emotion": "sad"
            if any(x in msg for x in ["sad", "lonely", "left", "break"])
            else "neutral",
            "intent": "relationship",
            "needs_counselor": False,
            "reason": "Relationship-related conversation detected.",
        }

    if contains_any(msg, CAREER_PHRASES) or topic == "Career":
        return {
            "risk_level": "low",
            "emotion": "anxious"
            if any(x in msg for x in ["worry", "worried", "scared", "confused"])
            else "neutral",
            "intent": "career",
            "needs_counselor": False,
            "reason": "Career-related conversation detected.",
        }

    if any(x in msg for x in ["sad", "stress", "stressed", "anxious", "angry", "lonely", "cry"]):
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


def local_reply(user_message: str, topic: str, risk_data: dict) -> str:
    msg = user_message.lower()
    risk = risk_data.get("risk_level", "low")
    intent = risk_data.get("intent", "general_chat")

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
            "Let’s make it smaller first — is the main problem the deadline, the workload, or not knowing where to start?"
        )

    if intent == "relationship":
        return (
            "Relationship problems can feel really heavy because they affect your emotions directly. "
            "Do you want to tell me what happened and how it has been affecting you?"
        )

    if intent == "career":
        return (
            "Thinking about your future can feel overwhelming, especially when you are unsure which direction to choose. "
            "What part worries you the most — internship, job options, salary, or confidence?"
        )

    if risk == "medium":
        return (
            "I’m sorry you’re feeling this way. It sounds like you’ve been carrying a lot recently. "
            "You don’t have to explain everything at once — what happened that made you feel this way?"
        )

    if any(x in msg for x in ["hi", "hello", "hey"]):
        return "Hi, I’m here. What would you like to talk about today?"

    return "I’m here with you. Tell me a bit more so I can understand what you mean."


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


def build_history_text(history) -> str:
    if not isinstance(history, list):
        return ""

    lines = []
    for item in history[-8:]:
        sender = str(item.get("sender", "unknown"))
        text = str(item.get("text", "")).strip()
        if text:
            lines.append(f"{sender}: {text}")

    return "\n".join(lines)


def gemini_reply_and_analysis(
    user_message: str,
    gender: str,
    topic: str,
    base_risk: dict,
    history_text: str = "",
) -> dict:
    if not gemini_client:
        result = base_risk.copy()
        result["response"] = local_reply(user_message, topic, result)
        return result

    topic_guidance = {
        "Study": "Focus on academic stress, assignments, exams, deadlines, motivation, and fear of failure.",
        "Relationship": "Focus on relationships, communication, conflict, loneliness, heartbreak, and emotional pain.",
        "Career": "Focus on future anxiety, internship, job worries, confidence, career planning, and decision making.",
        "Just Chat": "Respond as a supportive companion for general daily conversation.",
    }.get(topic, "Respond as a supportive AI companion for university students.")

    prompt = f"""
You are Haven, an empathetic and emotionally supportive AI companion designed for university students.

Your personality:
- Warm, calm, emotionally supportive, and natural.
- Speak like a caring human companion, not a robotic assistant.
- Never sound overly formal, repetitive, or scripted.
- Avoid generic responses like "I understand your feelings" repeatedly.
- Talk conversationally and naturally.
- Keep responses emotionally engaging and human-like.

Your goals:
1. Help students feel heard and emotionally supported.
2. Encourage students to express their thoughts safely.
3. Continue conversations naturally instead of ending too quickly.
4. Ask meaningful follow-up questions when appropriate.
5. Detect emotional distress carefully.
6. Encourage counselor support only when truly necessary.

Conversation style rules:
- Response should usually be 3–6 sentences.
- Keep most responses under 90 words unless the student is in crisis.
- Avoid sounding like a therapist textbook.
- Do not overuse motivational quotes.
- Do not give overly long lectures.
- Focus on understanding the student's feelings first.
- React emotionally to the situation naturally.
- Maintain conversation continuity using previous messages.
- Responses should feel emotionally alive, not generated from a template.
- Do not always start responses the same way.

Safety rules:
- Never encourage self-harm or suicide.
- If high-risk suicidal intent appears, stop casual conversation and strongly encourage immediate human help.
- In high-risk situations, say a counselor will be notified.
- Never diagnose mental illness.

Initial risk assessment:
{json.dumps(base_risk)}

Return ONLY valid JSON:
{{
  "response": "your chatbot reply",
  "risk_level": "low" | "medium" | "high",
  "emotion": "neutral" | "stressed" | "sad" | "anxious" | "angry" | "lonely" | "hopeless" | "unknown",
  "intent": "general_chat" | "study" | "relationship" | "career" | "emotional_support" | "crisis",
  "needs_counselor": true | false,
  "reason": "short reason"
}}

User profile:
- Gender: {gender}
- Selected topic: {topic}
- Topic guidance: {topic_guidance}

Recent conversation context:
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
            print("⚠️ Gemini returned non-JSON. Using local reply.")
            result = base_risk.copy()
            result["response"] = local_reply(user_message, topic, result)
            return result

        risk = str(result.get("risk_level", base_risk.get("risk_level", "low"))).lower()
        if risk not in ["low", "medium", "high"]:
            risk = base_risk.get("risk_level", "low")

        if base_risk.get("risk_level") == "high":
            risk = "high"

        result["risk_level"] = risk
        result["emotion"] = result.get("emotion", base_risk.get("emotion", "unknown"))
        result["intent"] = result.get("intent", base_risk.get("intent", "general_chat"))
        result["needs_counselor"] = bool(result.get("needs_counselor", risk == "high"))
        result["reason"] = result.get("reason", base_risk.get("reason", ""))
        result["response"] = result.get("response") or local_reply(user_message, topic, result)

        if risk == "high":
            result["needs_counselor"] = True
            result["response"] = (
                "⚠️ CRITICAL ALERT\n"
                "I’m really sorry you’re feeling this way. You are not alone. "
                "A university counselor has been notified so you can receive proper support. "
                "If you are in immediate danger, please call 999 or contact someone you trust now."
            )

        return result

    except Exception as e:
        print(f"❌ Gemini error: {e}")
        result = base_risk.copy()
        result["response"] = local_reply(user_message, topic, result)
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
            "topic": result.get("intent", "General"),

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


@app.route("/chat", methods=["POST"])
def chat():
    try:
        data = request.json or {}

        user_message = str(data.get("message", "")).strip()
        gender = str(data.get("gender", "Student"))
        topic = str(data.get("topic", "General"))
        user_uid = data.get("uid")
        history = data.get("history", [])
        history_text = build_history_text(history)

        if not user_message:
            return jsonify({
                "response": "Please type a message first.",
                "risk_level": "low",
                "emotion": "neutral",
                "intent": "general_chat",
                "needs_counselor": False,
            })

        print(f"📩 Message received: '{user_message}'")

        base_risk = rule_based_risk_analysis(user_message, topic)
        result = gemini_reply_and_analysis(
            user_message,
            gender,
            topic,
            base_risk,
            history_text,
        )

        print(f"🧠 Final result: {result}")

        update_user_risk(user_uid, result, user_message)
        save_conversation(user_uid, user_message, result.get("response", ""), result)
        save_counselor_alert(user_uid, user_message, result)

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


@app.route("/", methods=["GET"])
def home():
    return jsonify({
        "status": "Haven backend is running",
        "gemini": "connected" if gemini_client else "not connected",
        "firebase": "connected" if db else "not connected",
        "model": GEMINI_MODEL,
    })


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)