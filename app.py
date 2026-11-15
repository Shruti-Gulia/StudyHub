import os
from flask import Flask
from flask_cors import CORS

app = Flask(__name__)

CORS(
    app,
    resources={
        r"/api/*": {
            "origins": [
                "http://localhost:3000",
                "https://YOUR-STUDYHUB-DOMAIN"
            ]
        }
    }
)
# import os, re, requests
# from flask import Flask, request, jsonify
# from flask_cors import CORS
# from dotenv import load_dotenv
# from sklearn.feature_extraction.text import TfidfVectorizer
# from sklearn.metrics.pairwise import cosine_similarity
# from careers import CAREERS

# load_dotenv()
# app = Flask(__name__)
# CORS(app, resources={r"/api/*": {"origins": os.getenv("FRONTEND_ORIGIN", "http://localhost:3000")}})

# def clean_list(value):
#     if isinstance(value, list):
#         return [str(x).strip().lower() for x in value if str(x).strip()]
#     if isinstance(value, str):
#         return [x.strip().lower() for x in re.split(r"[,;\n]", value) if x.strip()]
#     return []

# def profile_text(profile):
#     fields = ["interests", "skills", "subjects", "strengths"]
#     chunks = []
#     for field in fields:
#         chunks.extend(clean_list(profile.get(field)))
#     for field in ["education", "preferred_work", "experience"]:
#         value = profile.get(field, "")
#         if value:
#             chunks.append(str(value).lower())
#     return " ".join(chunks)

# def career_text(career):
#     return " ".join([career["title"], career["category"], career["description"]] +
#                     career["skills"] + career["interests"] + career["subjects"] + career["strengths"])

# def recommend(profile):
#     user_text = profile_text(profile)
#     if len(user_text.strip()) < 3:
#         raise ValueError("Please provide at least a few interests, skills, subjects or strengths.")
#     docs = [career_text(c) for c in CAREERS] + [user_text]
#     matrix = TfidfVectorizer(ngram_range=(1, 2), stop_words="english").fit_transform(docs)
#     similarities = cosine_similarity(matrix[-1], matrix[:-1]).flatten()
#     user_terms = set(clean_list(profile.get("interests")) + clean_list(profile.get("skills")) +
#                      clean_list(profile.get("subjects")) + clean_list(profile.get("strengths")))
#     scored = []
#     for career, similarity in zip(CAREERS, similarities):
#         terms = set(career["skills"] + career["interests"] + career["subjects"] + career["strengths"])
#         overlap = len(user_terms & terms) / max(1, len(user_terms))
#         score = min(100, round((float(similarity) * 75 + overlap * 25) * 100) / 100)
#         scored.append({**career, "match_score": score})
#     return sorted(scored, key=lambda x: x["match_score"], reverse=True)[:5]

# def gemini_generate(prompt):
#     key = os.getenv("GEMINI_API_KEY")
#     if not key:
#         return None
#     model = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
#     url = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent?key={key}"
#     try:
#         response = requests.post(url, json={"contents":[{"parts":[{"text":prompt}]}],
#             "generationConfig":{"temperature":0.4,"maxOutputTokens":1200}}, timeout=35)
#         response.raise_for_status()
#         data = response.json()
#         return data["candidates"][0]["content"]["parts"][0]["text"]
#     except Exception as exc:
#         app.logger.warning("Gemini request failed: %s", exc)
#         return None

# @app.get("/api/health")
# def health():
#     return jsonify({"status":"ok", "ml_engine":"TF-IDF + cosine similarity", "gemini_enabled":bool(os.getenv("GEMINI_API_KEY"))})

# @app.get("/api/careers")
# def careers():
#     return jsonify({"careers":[{"id":c["id"],"title":c["title"],"category":c["category"],"description":c["description"]} for c in CAREERS]})

# @app.post("/api/recommend")
# def recommendations():
#     data = request.get_json(silent=True) or {}
#     try:
#         results = recommend(data)
#         summary_prompt = (
#             "You are a careful career counsellor. Based only on this profile and ranked matches, "
#             "write a concise, supportive explanation. Do not guarantee outcomes or claim a psychometric diagnosis. "
#             "Mention that recommendations are exploratory and should be validated through projects and counselling.\\n"
#             f"Profile: {data}\\nMatches: {[(x['title'], x['match_score']) for x in results]}"
#         )
#         ai_summary = gemini_generate(summary_prompt)
#         return jsonify({"recommendations":results, "summary":ai_summary or
#             "These matches are based on text similarity between your profile and career skill/interest profiles. "
#             "Treat them as starting points, then validate your choice with projects, conversations and further research.",
#             "engine":"TF-IDF + cosine similarity", "ai_enhanced":bool(ai_summary)})
#     except ValueError as exc:
#         return jsonify({"error":str(exc)}), 400

# @app.post("/api/roadmap")
# def roadmap():
#     data = request.get_json(silent=True) or {}
#     career = next((c for c in CAREERS if c["id"] == data.get("career_id")), None)
#     if not career:
#         return jsonify({"error":"Choose a valid career."}), 400
#     prompt = (
#       "Create a practical 12-week beginner roadmap for this career. Use weekly milestones, "
#       "free learning resources by name (do not invent URLs), one portfolio project, and measurable outcomes. "
#       "Adapt to the student's education and available hours. Be realistic and concise. "
#       "State that regulated professions require checking local accreditation and licensing.\\n"
#       f"Career: {career['title']}\\nStudent profile: {data.get('profile', {})}"
#     )
#     ai_text = gemini_generate(prompt)
#     return jsonify({"career":career["title"], "roadmap_text":ai_text,
#       "roadmap":career["roadmap"], "ai_enhanced":bool(ai_text)})

# @app.get("/api/jobs")
# def jobs():
#     query = request.args.get("search", "junior")
#     # Remotive provides a public remote-jobs API; results depend on its current listings.
#     try:
#         response = requests.get("https://remotive.com/api/remote-jobs",
#              params={"search":query, "limit":8}, timeout=20)
#         response.raise_for_status()
#         data = response.json()
#         items = [{"title":j.get("title"),"company":j.get("company_name"),
#                   "url":j.get("url"),"location":j.get("candidate_required_location"),
#                   "category":j.get("category"),"published":j.get("publication_date")}
#                  for j in data.get("jobs", [])]
#         return jsonify({"jobs":items, "source":"Remotive", "note":"Remote listings only; verify eligibility, experience and location requirements."})
#     except requests.RequestException:
#         return jsonify({"jobs":[], "source":"Remotive", "error":"The jobs API is temporarily unavailable. Try again later."}), 502

# if __name__ == "__main__":
#     app.run(host="0.0.0.0", port=int(os.getenv("PORT", 5000)), debug=True)
