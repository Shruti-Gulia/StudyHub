
import os
import re
import requests

from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from careers import CAREERS

load_dotenv()

app = Flask(__name__)

# CORS configuration
frontend_origins = [
    origin.strip()
    for origin in os.getenv(
        "FRONTEND_ORIGIN",
        "http://localhost:3000"
    ).split(",")
]

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": frontend_origins
        }
    }
)


# -------------------------------
# HEALTH CHECK
# -------------------------------

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


# -------------------------------
# PROFILE PROCESSING
# -------------------------------

def clean_list(value):
    if isinstance(value, list):
        return [
            str(x).strip().lower()
            for x in value
            if str(x).strip()
        ]

    if isinstance(value, str):
        return [
            x.strip().lower()
            for x in re.split(r"[,;\n]", value)
            if x.strip()
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
        career["title"],
        career["category"],
        career["description"]
    ]

    for field in [
        "skills",
        "interests",
        "subjects",
        "strengths"
    ]:
        fields.extend(career.get(field, []))

    return " ".join(fields)


# -------------------------------
# MACHINE LEARNING ENGINE
# -------------------------------

def recommend(profile):

    user_text = profile_text(profile)

    if len(user_text.strip()) < 3:
        raise ValueError(
            "Please provide interests, skills, subjects or strengths."
        )

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
            career.get("skills", [])
            + career.get("interests", [])
            + career.get("subjects", [])
            + career.get("strengths", [])
        )

        overlap = (
            len(user_terms & career_terms)
            / max(1, len(user_terms))
        )

        score = min(
            100,
            round(
                (
                    float(similarity) * 75
                    + overlap * 25
                ) * 100
            ) / 100
        )

        scored.append({
            **career,
            "match_score": score
        })

    return sorted(
        scored,
        key=lambda x: x["match_score"],
        reverse=True
    )[:5]


# -------------------------------
# GEMINI AI INTEGRATION
# -------------------------------

def gemini_generate(prompt):

    api_key = os.getenv("GEMINI_API_KEY")

    if not api_key:
        return None

    model = os.getenv(
        "GEMINI_MODEL",
        "gemini-2.0-flash"
    )

    url = (
        "https://generativelanguage.googleapis.com/"
        f"v1beta/models/{model}:generateContent"
        f"?key={api_key}"
    )

    try:
        response = requests.post(
            url,
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
            timeout=35
        )

        response.raise_for_status()

        data = response.json()

        return data["candidates"][0]["content"]["parts"][0]["text"]

    except Exception as exc:
        app.logger.warning(
            "Gemini request failed: %s",
            exc
        )
        return None


# -------------------------------
# CAREER RECOMMENDATIONS API
# -------------------------------

@app.post("/api/recommend")
def recommendations():

    data = request.get_json(silent=True) or {}

    try:

        results = recommend(data)

        summary_prompt = (
            "You are a careful career counsellor. "
            "Based only on this student profile and ranked career matches, "
            "write a concise and supportive career analysis. "
            "Do not guarantee outcomes or claim a psychometric diagnosis. "
            "Mention that recommendations are exploratory.\n"
            f"Profile: {data}\n"
            f"Matches: {[(x['title'], x['match_score']) for x in results]}"
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
        })

    except ValueError as exc:

        return jsonify({
            "error": str(exc)
        }), 400

    except Exception as exc:

        app.logger.exception("Recommendation error")

        return jsonify({
            "error": "Unable to generate recommendations."
        }), 500


# -------------------------------
# PERSONALIZED ROADMAP API
# -------------------------------

@app.post("/api/roadmap")
def roadmap():

    data = request.get_json(silent=True) or {}

    career = next(
        (
            c for c in CAREERS
            if c["id"] == data.get("career_id")
        ),
        None
    )

    if not career:
        return jsonify({
            "error": "Choose a valid career."
        }), 400

    prompt = (
        "Create a practical 12-week beginner learning roadmap "
        "for this career. Include weekly milestones, free learning "
        "resources by name, one portfolio project and measurable outcomes. "
        "Adapt to the student's education and available hours. "
        "Do not invent URLs.\n"
        f"Career: {career['title']}\n"
        f"Student profile: {data.get('profile', {})}"
    )

    ai_text = gemini_generate(prompt)

    return jsonify({
        "career": career["title"],
        "roadmap_text": ai_text,
        "roadmap": career["roadmap"],
        "ai_enhanced": bool(ai_text)
    })


# -------------------------------
# REMOTE JOB SEARCH API
# -------------------------------

@app.get("/api/jobs")
def jobs():

    query = request.args.get(
        "search",
        "junior"
    )

    try:

        response = requests.get(
            "https://remotive.com/api/remote-jobs",
            params={
                "search": query,
                "limit": 8
            },
            timeout=20
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
            "note": "Remote listings only. Verify eligibility and requirements."
        })

    except requests.RequestException:

        return jsonify({
            "jobs": [],
            "source": "Remotive",
            "error": "The jobs API is temporarily unavailable."
        }), 502


# -------------------------------
# START APPLICATION
# -------------------------------

if __name__ == "__main__":

    app.run(
        host="0.0.0.0",
        port=int(os.getenv("PORT", 5000)),
        debug=False
    )
