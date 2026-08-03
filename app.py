from pathlib import Path
import re
from datetime import datetime
import os

from flask import Flask, render_template, request, send_from_directory, jsonify, session
from dotenv import load_dotenv

from model import get_artifacts, get_role_skill_profile, model, prepare_prediction_input, get_real_shap_explanation
from ai_assistant import get_assistant

# Load environment variables
load_dotenv()

BASE_DIR = Path(__file__).resolve().parent
app = Flask(__name__)

# Secret key for sessions
app.secret_key = os.getenv("SECRET_KEY", "your-secret-key-here-change-in-production")

ROLE_PROGRESSION = {
	"Data Scientist": "ML Engineer",
	"Data Engineer": "Data Scientist",
	"Backend Developer": "DevOps Engineer",
	"DevOps Engineer": "DevOps Engineer",
	"ML Engineer": "ML Engineer",
}


def _normalize_skill(skill: str) -> str:
	return re.sub(r"\s+", " ", skill.strip().lower())


def _parse_user_skills(skills_text: str) -> list[str]:
	parts = re.split(r"[,;\n/]+", skills_text or "")
	return [part.strip() for part in parts if part and part.strip()]


def _build_recommendation_context(predicted_role: str, user_skills_text: str) -> dict:
	role_profile = get_role_skill_profile(predicted_role)
	role_skills = role_profile.get("top_skills", [])
	user_skills = _parse_user_skills(user_skills_text)
	user_skill_set = {_normalize_skill(skill) for skill in user_skills}

	matched_skills = [
		skill
		for skill in role_skills
		if _normalize_skill(skill) in user_skill_set
	]
	recommended_skills = [
		skill
		for skill in role_skills
		if _normalize_skill(skill) not in user_skill_set
	]
	total_role_skills = len(role_skills)
	current_skills_count = len(matched_skills)
	missing_skills_count = len(recommended_skills)
	readiness_score = round((current_skills_count / total_role_skills) * 100, 2) if total_role_skills else 0.0

	return {
		"role_skill_profile": role_profile,
		"recommended_skills": recommended_skills,
		"matched_skills": matched_skills,
		"skills_already_possessed": matched_skills,
		"current_skills_count": current_skills_count,
		"missing_skills_count": missing_skills_count,
		"total_role_skills": total_role_skills,
		"readiness_score": readiness_score,
	}


def _parse_target_year(value: str) -> int:
	current_year = datetime.now().year
	try:
		parsed_year = int(str(value).strip())
	except (TypeError, ValueError):
		return current_year
	return parsed_year if parsed_year > 0 else current_year


def _build_future_projection(predicted_role: str, readiness_score: float, target_year: int, current_skills_count: int, missing_skills_count: int, recommended_skills: list[str]) -> dict:
	current_year = datetime.now().year
	years_ahead = max(0, target_year - current_year)
	current_role = predicted_role
	future_role = predicted_role

	if years_ahead <= 0:
		explanation = (
			f"Your target year is {current_year} or earlier, so the simulation focuses on your current {predicted_role} path."
		)
	else:
		future_role = ROLE_PROGRESSION.get(predicted_role, predicted_role)
		if readiness_score >= 60:
			explanation = (
				f"With {years_ahead} year(s) to grow and {current_skills_count} matching skills already in place, you are on track to progress from {predicted_role} to {future_role} by {target_year}."
			)
		else:
			explanation = (
				f"By {target_year}, focus on closing the {missing_skills_count} missing skill gap in {', '.join(recommended_skills[:3]) or 'your role skills'} to strengthen your {predicted_role} trajectory."
			)

	return {
		"current_year": current_year,
		"target_year": target_year,
		"years_ahead": years_ahead,
		"current_role": current_role,
		"future_role": future_role,
		"future_role_active": years_ahead > 0,
		"career_growth_explanation": explanation,
	}


@app.route("/")
def index():
	return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
	payload = {
		"skills": request.form.get("skills", ""),
		"experience": request.form.get("experience", ""),
		"education": request.form.get("education", ""),
		"target_year": request.form.get("target_year", ""),
	}
	input_frame = prepare_prediction_input(
		payload["skills"],
		payload["experience"],
		payload["education"],
	)
	predicted_role = model.predict(input_frame)[0]
	probabilities = model.predict_proba(input_frame)[0]
	classes = model.named_steps["classifier"].classes_
	probability_map = [
		{"role": role, "probability": round(float(probability) * 100, 2)}
		for role, probability in sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
	]
	prediction = {
		"predicted_role": predicted_role,
		"predicted_probability": probability_map[0]["probability"],
		"probabilities": probability_map,
		"accuracy": round(get_artifacts()["accuracy"] * 100, 2),
	}
	recommendation_context = _build_recommendation_context(prediction["predicted_role"], payload["skills"])
	target_year = _parse_target_year(payload["target_year"])
	projection_context = _build_future_projection(
		prediction["predicted_role"],
		recommendation_context["readiness_score"],
		target_year,
		recommendation_context["current_skills_count"],
		recommendation_context["missing_skills_count"],
		recommendation_context["recommended_skills"],
	)
	
	# Store prediction data in session for AI chat
	session['prediction_data'] = {
		"skills": payload["skills"],
		"experience": payload["experience"],
		"education": payload["education"],
		"predicted_role": prediction["predicted_role"],
		"readiness_score": recommendation_context["readiness_score"]
	}
	
	return render_template(
		"result.html",
		data=payload,
		prediction=prediction,
		**recommendation_context,
		**projection_context,
	)


@app.route("/logo.png")
def logo():
	return send_from_directory(BASE_DIR, "logo.png")


# SHAP Explanation Endpoint
@app.route("/explain", methods=["POST"])
def explain_prediction():
	"""API endpoint for REAL SHAP explanations"""
	data = request.get_json()
	if not data:
		return jsonify({"error": "No data provided"}), 400
	
	skills = data.get("skills", "")
	experience = data.get("experience", 0)
	education = data.get("education", "")
	
	try:
		explanation = get_real_shap_explanation(skills, experience, education)
		return jsonify(explanation)
	except Exception as e:
		return jsonify({"error": str(e), "success": False}), 500


# AI Chat Endpoints
@app.route("/ai_chat", methods=["POST"])
def ai_chat():
	"""Chat endpoint for career advice using Groq"""
	try:
		data = request.get_json()
		query = data.get("query", "")
		user_profile = data.get("profile", {})
		
		if not query:
			return jsonify({"error": "Query is required"}), 400
		
		# Get assistant and response
		assistant = get_assistant()
		response = assistant.get_career_advice(query, user_profile)
		
		return jsonify(response)
		
	except Exception as e:
		return jsonify({"error": str(e)}), 500


@app.route("/ai_chat_context", methods=["POST"])
def ai_chat_with_context():
	"""Chat endpoint with career context from model prediction using Groq"""
	try:
		data = request.get_json()
		query = data.get("query", "")
		prediction_data = data.get("prediction", {})
		
		if not query:
			return jsonify({"error": "Query is required"}), 400
		
		# If no prediction data sent, try to get from session
		if not prediction_data and 'prediction_data' in session:
			prediction_data = session['prediction_data']
		
		# Build user profile from prediction
		user_profile = {
			"skills": prediction_data.get("skills", ""),
			"experience": prediction_data.get("experience", 0),
			"education": prediction_data.get("education", ""),
			"predicted_role": prediction_data.get("predicted_role", ""),
			"readiness_score": prediction_data.get("readiness_score", 0)
		}
		
		assistant = get_assistant()
		response = assistant.get_career_advice(query, user_profile)
		
		return jsonify(response)
		
	except Exception as e:
		return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
	app.run(debug=True)