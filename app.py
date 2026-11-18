
import os
import re
import requests

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from careers import CAREERS


# --------------------------------
# ENVIRONMENT CONFIGURATION
# --------------------------------

load_dotenv()

app = Flask(__name__)

# Maximum request size: 2 MB
app.config["MAX_CONTENT_LENGTH"] = 2 * 1024 * 1024

# Add frontend URLs in Render's FRONTEND_ORIGIN environment variable.
# Separate multiple URLs using commas.
default_origins = (
    "http://localhost:3000,"
    "http://127.0.0.1:3000,"
    "http://192.168.1.6:3000,"
    "http://10.94.77.225:3000"
)

frontend_origins = [
    origin.strip().rstrip("/")
    for origin in os.getenv(
        "FRONTEND_ORIGIN",
        default_origins
    ).split(",")
    if origin.strip()
]

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": frontend_origins,
            "methods": ["GET", "POST", "OPTIONS"],
            "allow_headers": [
                "Content-Type",
                "Authorization"
            ],
            "max_age": 600
        }
    }
)


# --------------------------------
# HEALTH CHECK
# --------------------------------

@app.get("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "StudyHub Career Counselling API is running"
    })


@app.get("/api/health")
def health():
    return jsonify({
        "status": "ok",
        "ml_engine": "TF-IDF + cosine similarity",
        "gemini_enabled": bool(os.getenv("GEMINI_API_KEY"))
    })


# --------------------------------
# PROFILE PROCESSING
# --------------------------------

def clean_list(value):

    if isinstance(value, list):
        return [
            str(item).strip().lower()
            for item in value
            if str(item).strip()
        ]

    if isinstance(value, str):
        return [
            item.strip().lower()
            for item in re.split(r"[,;\n]", value)
            if item.strip()
        ]

    return []


def profile_text(profile):

    fields = [
        "interests",
        "skills",
        "subjects",
        "strengths"
    ]

    chunks = []

    for field in fields:
        chunks.extend(clean_list(profile.get(field)))

    for field in [
        "education",
        "preferred_work",
        "experience"
    ]:
        value = profile.get(field, "")

        if value:
            chunks.append(str(value).lower())

    return " ".join(chunks)


def career_text(career):

    fields = [
        career.get("title", ""),
        career.get("category", ""),
        career.get("description", "")
    ]

    for field in [
        "skills",
        "interests",
        "subjects",
        "strengths"
    ]:
        fields.extend(clean_list(career.get(field, [])))

    return " ".join(str(item) for item in fields)


# --------------------------------
# MACHINE LEARNING ENGINE
# --------------------------------

def recommend(profile):

    if not isinstance(profile, dict):
        raise ValueError("Invalid student profile.")

    user_text = profile_text(profile)

    if len(user_text.strip()) < 3:
        raise ValueError(
            "Please provide interests, skills, subjects or strengths."
        )

    if not CAREERS:
        raise ValueError("No career information is available.")

    documents = [
        career_text(career)
        for career in CAREERS
    ] + [user_text]

    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        stop_words="english"
    )

    matrix = vectorizer.fit_transform(documents)

    similarities = cosine_similarity(
        matrix[-1],
        matrix[:-1]
    ).flatten()

    user_terms = set(
        clean_list(profile.get("interests"))
        + clean_list(profile.get("skills"))
        + clean_list(profile.get("subjects"))
        + clean_list(profile.get("strengths"))
    )

    scored = []

    for career, similarity in zip(CAREERS, similarities):

        career_terms = set(
            clean_list(career.get("skills", []))
            + clean_list(career.get("interests", []))
            + clean_list(career.get("subjects", []))
            + clean_list(career.get("strengths", []))
        )

        overlap = (
            len(user_terms & career_terms)
            / max(1, len(user_terms))
        )

        score = round(
            (
                float(similarity) * 75
                + overlap * 25
            ),
            2
        )

        scored.append({
            **career,
            "match_score": min(100, score)
        })

    return sorted(
        scored,
        key=lambda item: item["match_score"],
        reverse=True
    )[:5]


# --------------------------------
# GEMINI AI INTEGRATION
# --------------------------------

def gemini_generate(prompt):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        app.logger.info("Gemini API key is not configured.")
        return None

    model = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.0-flash"
    ).strip()

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:generateContent"
    )

    try:

        response = requests.post(
            url,
            params={"key": api_key},
            json={
                "contents": [
                    {
                        "parts": [
                            {"text": prompt}
                        ]
                    }
                ],
                "generationConfig": {
                    "temperature": 0.4,
                    "maxOutputTokens": 1200
                }
            },
            timeout=(10, 35)
        )

        response.raise_for_status()

        data = response.json()

        candidates = data.get("candidates", [])

        if not candidates:
            app.logger.warning(
                "Gemini returned no response candidates."
            )
            return None

        parts = candidates[0].get(
            "content", {}
        ).get("parts", [])

        generated_text = " ".join(
            part.get("text", "")
            for part in parts
            if part.get("text")
        ).strip()

        return generated_text or None

    except requests.RequestException as exc:

        app.logger.warning(
            "Gemini API request failed: %s",
            exc
        )

        return None

    except (ValueError, KeyError, TypeError) as exc:

        app.logger.warning(
            "Unable to process Gemini response: %s",
            exc
        )

        return None


# --------------------------------
# CAREER RECOMMENDATIONS API
# --------------------------------

@app.post("/api/recommend")
def recommendations():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Please submit a valid JSON student profile."
        }), 400

    try:

        results = recommend(data)

        summary_prompt = (
            "You are a careful career counsellor. "
            "Based only on this student profile and ranked career matches, "
            "write a concise and supportive career analysis. "
            "Do not guarantee outcomes or claim a psychometric diagnosis. "
            "Mention that recommendations are exploratory.\n"
            f"Profile: {data}\n"
            f"Matches: {[(item['title'], item['match_score']) for item in results]}"
        )

        ai_summary = gemini_generate(summary_prompt)

        return jsonify({
            "recommendations": results,
            "summary": ai_summary or (
                "These career matches are based on text similarity "
                "between your profile and career requirements. "
                "Use them as starting points for further exploration."
            ),
            "engine": "TF-IDF + cosine similarity",
            "ai_enhanced": bool(ai_summary)
        }), 200

    except ValueError as exc:

        return jsonify({
            "error": str(exc)
        }), 400

    except Exception:

        app.logger.exception(
            "Career recommendation error"
        )

        return jsonify({
            "error": "Unable to generate recommendations. Please try again."
        }), 500


# --------------------------------
# PERSONALIZED ROADMAP API
# --------------------------------

@app.post("/api/roadmap")
def roadmap():

    data = request.get_json(silent=True)

    if not isinstance(data, dict):
        return jsonify({
            "error": "Please submit valid JSON data."
        }), 400

    career_id = data.get("career_id")

    career = next(
        (
            item for item in CAREERS
            if item.get("id") == career_id
        ),
        None
    )

    if not career:
        return jsonify({
            "error": "Choose a valid career."
        }), 400

    student_profile = data.get("profile", {})

    if not isinstance(student_profile, dict):
        student_profile = {}

    prompt = (
        "Create a practical 12-week beginner learning roadmap "
        "for this career. Include weekly milestones, free learning "
        "resources by name, one portfolio project and measurable outcomes. "
        "Adapt to the student's education and available hours. "
        "Do not invent URLs.\n"
        f"Career: {career.get('title', '')}\n"
        f"Student profile: {student_profile}"
    )

    ai_text = gemini_generate(prompt)

    return jsonify({
        "career": career.get("title", ""),
        "roadmap_text": ai_text or (
            "Follow the suggested roadmap below to begin "
            "developing the skills required for this career."
        ),
        "roadmap": career.get("roadmap", []),
        "ai_enhanced": bool(ai_text)
    }), 200


# --------------------------------
# REMOTIVE JOB SEARCH API
# --------------------------------

@app.get("/api/jobs")
def jobs():

    query = request.args.get(
        "search",
        "junior"
    ).strip()

    if not query:
        return jsonify({
            "error": "Please enter a job title.",
            "jobs": []
        }), 400

    try:

        response = requests.get(
            "https://remotive.com/api/remote-jobs",
            params={
                "search": query,
                "limit": 8
            },
            timeout=(10, 20)
        )

        response.raise_for_status()

        data = response.json()

        items = [
            {
                "title": job.get("title"),
                "company": job.get("company_name"),
                "url": job.get("url"),
                "location": job.get("candidate_required_location"),
                "category": job.get("category"),
                "published": job.get("publication_date")
            }
            for job in data.get("jobs", [])
        ]

        return jsonify({
            "jobs": items,
            "source": "Remotive",
            "note": (
                "Remote listings only. "
                "Verify eligibility and requirements."
            )
        }), 200

    except requests.RequestException:

        app.logger.exception(
            "Remotive jobs API request failed"
        )

        return jsonify({
            "jobs": [],
            "source": "Remotive",
            "error": "The jobs API is temporarily unavailable."
        }), 502

    except ValueError:

        app.logger.exception(
            "Invalid response received from Remotive"
        )

        return jsonify({
            "jobs": [],
            "source": "Remotive",
            "error": "Unable to process the jobs API response."
        }), 502


# --------------------------------
# ERROR HANDLERS
# --------------------------------

@app.errorhandler(413)
def request_too_large(error):

    return jsonify({
        "error": "Request is too large."
    }), 413


@app.errorhandler(404)
def not_found(error):

    return jsonify({
        "error": "API endpoint not found."
    }), 404


@app.errorhandler(405)
def method_not_allowed(error):

    return jsonify({
        "error": "This request method is not allowed."
    }), 405


# --------------------------------
# START APPLICATION
# --------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=False
    )
