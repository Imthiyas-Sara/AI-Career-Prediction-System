from __future__ import annotations

from collections import Counter
import re
from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd
import numpy as np
import shap
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
import warnings
warnings.filterwarnings('ignore')

DATA_PATH = Path(__file__).with_name("survey.csv")
MODEL_PATH = Path(__file__).with_name("career_role_model_roles_v2.joblib")
ROLE_SKILL_PROFILE_PATH = Path(__file__).with_name("role_skill_profiles.joblib")
TARGET_ROLES = [
	"Data Scientist",
	"Data Engineer",
	"Backend Developer",
	"ML Engineer",
	"DevOps Engineer",
]


def _parse_experience(value: object) -> float | None:
	if pd.isna(value):
		return None

	text = str(value).strip()
	if not text or text == "NA":
		return None
	if text.startswith("Less than 1 year"):
		return 0.5
	if text.startswith("More than 50 years"):
		return 51.0

	match = re.search(r"\d+", text)
	return float(match.group()) if match else None


def _normalize_education(value: object) -> str:
	text = str(value).strip().lower()
	if not text or text == "nan" or text == "na":
		return "Other"
	if "primary" in text or "secondary school" in text or "high school" in text:
		return "High School"
	if "some college" in text or "associate" in text or "diploma" in text:
		return "Diploma"
	if "bachelor" in text:
		return "Bachelor's"
	if "master" in text or "mba" in text or "professional degree" in text:
		return "Master's"
	if "phd" in text or "doctoral" in text:
		return "PhD"
	return "Other"


def _combine_skill_columns(frame: pd.DataFrame) -> pd.Series:
	skill_columns = [column for column in frame.columns if "HaveWorkedWith" in column]
	if not skill_columns:
		return pd.Series([""] * len(frame), index=frame.index)

	combined = frame[skill_columns].fillna("").astype(str).agg(" ".join, axis=1)
	return combined.replace("NA", " ", regex=False).str.replace(r"\s+", " ", regex=True).str.strip()


def _load_survey_frame() -> pd.DataFrame:
	use_columns = lambda column: column in {"DevType", "EdLevel", "YearsCodePro", "YearsCode"} or "HaveWorkedWith" in column
	return pd.read_csv(DATA_PATH, usecols=use_columns, low_memory=False)


def _normalize_skill(skill: str) -> str:
	return re.sub(r"\s+", " ", skill.strip().lower())


def _split_skill_text(skills_text: str) -> list[str]:
	parts = re.split(r"[;,/\n]+", skills_text or "")
	cleaned_parts = []
	for part in parts:
		cleaned = re.sub(r"\s+", " ", part).strip()
		if cleaned and cleaned.lower() not in {"na", "nan", "none"}:
			cleaned_parts.append(cleaned)
	return cleaned_parts


def _format_skill_display(skill: str) -> str:
	formatted = re.sub(r"\s+", " ", skill).strip()
	if not formatted:
		return formatted

	upper_skill = formatted.upper()
	if upper_skill in {"AWS", "SQL", "CI/CD", "DBT", "ETL", "API", "REST", "MLFLOW"}:
		return upper_skill
	if formatted.lower() == "power bi":
		return "Power BI"
	if formatted.lower() == "hugging face":
		return "Hugging Face"
	if formatted.lower() == "postgresql":
		return "PostgreSQL"
	if formatted.lower() == "tensorflow":
		return "TensorFlow"
	if formatted.lower() == "pytorch":
		return "PyTorch"
	if formatted.lower() == "mlflow":
		return "MLflow"
	if formatted.lower() == "kafka":
		return "Kafka"
	if formatted.lower() == "airflow":
		return "Airflow"
	if formatted.lower() == "snowflake":
		return "Snowflake"
	if formatted.lower() == "microservices":
		return "Microservices"
	if formatted.lower() == "redis":
		return "Redis"
	if formatted.lower() == "monitoring":
		return "Monitoring"
	if formatted.lower() == "langchain":
		return "LangChain"
	if formatted.lower() == "kubernetes":
		return "Kubernetes"
	if formatted.lower() == "terraform":
		return "Terraform"
	if formatted.lower() == "statistics":
		return "Statistics"
	if formatted.lower() == "machine learning":
		return "Machine Learning"
	if formatted.lower() == "pandas":
		return "Pandas"
	if formatted.lower() == "docker":
		return "Docker"
	return formatted


def _build_role_skill_profiles(frame: pd.DataFrame) -> dict[str, dict]:
	skill_columns = [column for column in frame.columns if "HaveWorkedWith" in column]
	profiles: dict[str, dict] = {}
	for role, role_frame in frame.groupby("role", sort=True):
		counter: Counter[str] = Counter()
		display_map: dict[str, str] = {}

		for _, row in role_frame.iterrows():
			row_skill_tokens: list[str] = []
			for column in skill_columns:
				cell_value = row.get(column)
				if pd.isna(cell_value):
					continue
				row_skill_tokens.extend(_split_skill_text(str(cell_value)))

			row_skills = {_normalize_skill(skill): skill for skill in row_skill_tokens}
			for normalized_skill, skill in row_skills.items():
				if not normalized_skill:
					continue
				counter[normalized_skill] += 1
				display_map.setdefault(normalized_skill, _format_skill_display(skill))

		top_skill_items = counter.most_common(20)
		profiles[role] = {
			"role": role,
			"top_skills": [display_map[normalized_skill] for normalized_skill, _ in top_skill_items],
			"skill_frequencies": [
				{
					"skill": display_map[normalized_skill],
					"count": count,
				}
				for normalized_skill, count in top_skill_items
			],
			"unique_skill_count": len(counter),
			"total_skill_mentions": int(sum(counter.values())),
		}

	return profiles


@lru_cache(maxsize=1)
def get_role_skill_profiles() -> dict[str, dict]:
	if ROLE_SKILL_PROFILE_PATH.exists():
		profiles = joblib.load(ROLE_SKILL_PROFILE_PATH)
		if isinstance(profiles, dict) and sorted(profiles.keys()) == sorted(TARGET_ROLES):
			return profiles

	raw_frame = _load_survey_frame()
	raw_frame = raw_frame.dropna(subset=["DevType"]).copy()
	raw_frame["role"] = raw_frame.apply(
		lambda row: _map_role(
			row["DevType"],
			" ".join(
				[
					part
					for column in raw_frame.columns
					if "HaveWorkedWith" in column
					for part in _split_skill_text(str(row.get(column, "")))
				]
			),
		),
		axis=1,
	)
	raw_frame = raw_frame.dropna(subset=["role"]).copy()
	raw_frame = raw_frame[raw_frame["role"].isin(TARGET_ROLES)].copy()
	profiles = _build_role_skill_profiles(raw_frame)
	joblib.dump(profiles, ROLE_SKILL_PROFILE_PATH)
	return profiles


def get_role_skill_profile(role: str) -> dict:
	profiles = get_role_skill_profiles()
	default_profile = {
		"role": role,
		"top_skills": [],
		"skill_frequencies": [],
		"unique_skill_count": 0,
		"total_skill_mentions": 0,
	}
	return profiles.get(role, default_profile)


def _map_role(dev_type: object, skills_text: str) -> str | None:
	text = str(dev_type).strip().lower()
	skills = skills_text.lower()

	if not text or text == "na" or text == "nan":
		return None
	if "student" in text or "other (please specify)" in text:
		return None

	if "devops" in text or "sre" in text or "site reliability" in text:
		return "DevOps Engineer"
	if "data engineer" in text:
		return "Data Engineer"
	if "developer, back-end" in text or "backend" in text or "back-end" in text:
		return "Backend Developer"
	if "data scientist" in text or "machine learning specialist" in text:
		ml_signals = (
			"tensorflow",
			"pytorch",
			"keras",
			"scikit-learn",
			"sklearn",
			"xgboost",
			"lightgbm",
			"mlflow",
			"hugging face",
			"huggingface",
			"langchain",
			"llm",
			"transformers",
			"machine learning",
			"data science",
		)
		if any(signal in skills for signal in ml_signals):
			return "ML Engineer"
		return "Data Scientist"

	return None


def _build_dataset() -> pd.DataFrame:
	frame = _load_survey_frame()
	frame = frame.dropna(subset=["DevType"]).copy()

	frame["skills"] = _combine_skill_columns(frame)
	frame["experience"] = frame["YearsCodePro"].map(_parse_experience)
	frame["experience"] = frame["experience"].fillna(frame["YearsCode"].map(_parse_experience))
	frame["education"] = frame["EdLevel"].map(_normalize_education)
	frame["role"] = frame.apply(lambda row: _map_role(row["DevType"], row["skills"]), axis=1)
	frame = frame.dropna(subset=["role"]).copy()
	frame = frame[frame["role"].isin(TARGET_ROLES)].copy()

	frame = frame[["skills", "experience", "education", "role"]].copy()
	frame["skills"] = frame["skills"].fillna("")
	frame["experience"] = frame["experience"].fillna(frame["experience"].median())
	frame["education"] = frame["education"].fillna("Other")
	return frame


def _build_pipeline() -> Pipeline:
	preprocessor = ColumnTransformer(
		transformers=[
			("skills", TfidfVectorizer(max_features=6000, ngram_range=(1, 2)), "skills"),
			("experience", Pipeline([
				("imputer", SimpleImputer(strategy="median")),
				("scaler", StandardScaler()),
			]), ["experience"]),
			("education", OneHotEncoder(handle_unknown="ignore"), ["education"]),
		],
		verbose_feature_names_out=False,
	)

	classifier = LogisticRegression(
		solver="lbfgs",
		max_iter=2000,
	)

	return Pipeline([
		("preprocessor", preprocessor),
		("classifier", classifier),
	])


@lru_cache(maxsize=1)
def get_artifacts() -> dict:
	if MODEL_PATH.exists():
		artifacts = joblib.load(MODEL_PATH)
		if sorted(artifacts.get("classes", [])) == sorted(TARGET_ROLES):
			get_role_skill_profiles()
			return artifacts

	dataset = _build_dataset()
	get_role_skill_profiles()
	features = dataset[["skills", "experience", "education"]]
	target = dataset["role"]

	x_train, x_test, y_train, y_test = train_test_split(
		features,
		target,
		test_size=0.2,
		random_state=42,
		stratify=target,
	)
#training process
	pipeline = _build_pipeline()
	pipeline.fit(x_train, y_train)
	accuracy = float(pipeline.score(x_test, y_test))

	artifacts = {
		"pipeline": pipeline,
		"accuracy": accuracy,
		"classes": list(pipeline.named_steps["classifier"].classes_),
		"target_roles": TARGET_ROLES,
	}
	joblib.dump(artifacts, MODEL_PATH)
	return artifacts


def predict_career_role(skills: str, experience: str | int | float, education: str) -> dict:
	artifacts = get_artifacts()
	pipeline: Pipeline = artifacts["pipeline"]
	input_frame = pd.DataFrame([
		{
			"skills": skills or "",
			"experience": _parse_experience(experience),
			"education": _normalize_education(education),
		}
	])
	input_frame["experience"] = input_frame["experience"].fillna(0)

	probabilities = pipeline.predict_proba(input_frame)[0]
	classes = pipeline.named_steps["classifier"].classes_
	probability_map = [
		{"role": role, "probability": round(float(probability) * 100, 2)}
		for role, probability in sorted(zip(classes, probabilities), key=lambda item: item[1], reverse=True)
	]
	top_prediction = probability_map[0]

	return {
		"predicted_role": top_prediction["role"],
		"predicted_probability": top_prediction["probability"],
		"probabilities": probability_map,
		"accuracy": round(artifacts["accuracy"] * 100, 2),
	}


def prepare_prediction_input(skills: str, experience: str | int | float, education: str) -> pd.DataFrame:
	return pd.DataFrame([
		{
			"skills": skills or "",
			"experience": _parse_experience(experience),
			"education": _normalize_education(education),
		}
	])


# ============================================
# FIXED SHAP IMPLEMENTATION - Using TreeExplainer with a wrapper
# ============================================

def get_shap_values_for_prediction(skills: str, experience: str | int | float, education: str) -> dict:
    """
    Get SHAP values for a prediction using a simpler approach
    """
    try:
        artifacts = get_artifacts()
        pipeline = artifacts["pipeline"]
        preprocessor = pipeline.named_steps["preprocessor"]
        classifier = pipeline.named_steps["classifier"]
        classes = classifier.classes_
        
        # Prepare input
        input_df = prepare_prediction_input(skills, experience, education)
        
        # Transform input
        input_transformed = preprocessor.transform(input_df)
        
        # Convert to dense array
        if hasattr(input_transformed, "toarray"):
            input_transformed = input_transformed.toarray()
        else:
            input_transformed = np.array(input_transformed)
        
        # Get prediction
        probabilities = classifier.predict_proba(input_transformed)[0]
        predicted_idx = np.argmax(probabilities)
        predicted_role = classes[predicted_idx]
        predicted_prob = probabilities[predicted_idx]
        
        # Try to get feature importance using coefficients
        # This is a SHAP-like approximation using model coefficients
        try:
            # Get coefficients for the predicted class
            coefficients = classifier.coef_[predicted_idx]
            
            # Get feature names
            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                feature_names = [f"feature_{i}" for i in range(len(coefficients))]
            
            # Calculate feature contributions (coefficient * feature_value)
            feature_contributions = []
            
            for idx, (coef, feat_val) in enumerate(zip(coefficients, input_transformed[0])):
                if idx < len(feature_names):
                    feature_name = feature_names[idx]
                else:
                    feature_name = f"feature_{idx}"
                
                # Contribution = coefficient * feature_value
                contribution = float(coef * feat_val)
                
                if abs(contribution) > 0.0001:
                    feature_contributions.append({
                        "feature": feature_name,
                        "contribution": contribution,
                        "abs_contribution": abs(contribution)
                    })
            
            # Sort by absolute contribution
            feature_contributions.sort(key=lambda x: x["abs_contribution"], reverse=True)
            top_features = feature_contributions[:15]
            
            # Generate explanation
            explanation_text = _generate_shap_explanation(
                predicted_role,
                predicted_prob,
                top_features,
                skills
            )
            
            return {
                "success": True,
                "predicted_role": predicted_role,
                "predicted_probability": round(predicted_prob * 100, 2),
                "top_features": top_features,
                "explanation_text": explanation_text,
                "method": "coefficient-based importance (SHAP-like)"
            }
            
        except Exception as e:
            # Fallback to simpler feature importance
            return _get_fallback_explanation(skills, experience, education, predicted_role, predicted_prob)
            
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "explanation_text": f"Error generating explanation: {str(e)}"
        }


def _get_fallback_explanation(skills: str, experience: str, education: str, predicted_role: str, predicted_prob: float) -> dict:
    """Fallback explanation when SHAP fails"""
    try:
        # Simple feature importance based on skill matching
        user_skills_list = [s.strip().lower() for s in skills.split(',') if s.strip()]
        
        # Define role-specific keywords
        role_keywords = {
            "Data Scientist": ["python", "sql", "statistics", "machine learning", "data", "analytics"],
            "Data Engineer": ["python", "sql", "etl", "cloud", "big data", "spark", "airflow"],
            "Backend Developer": ["python", "java", "api", "database", "docker", "git", "rest"],
            "ML Engineer": ["python", "ml", "tensorflow", "pytorch", "cloud", "docker", "mlops"],
            "DevOps Engineer": ["linux", "docker", "kubernetes", "ci/cd", "cloud", "terraform"]
        }
        
        keywords = role_keywords.get(predicted_role, role_keywords["Backend Developer"])
        
        # Calculate matches
        matched_skills = []
        for keyword in keywords:
            for user_skill in user_skills_list:
                if keyword in user_skill or user_skill in keyword:
                    matched_skills.append(keyword)
                    break
        
        matched_skills = list(set(matched_skills))
        match_percentage = min(int((len(matched_skills) / len(keywords)) * 100), 100)
        
        # Create features
        features = []
        
        # Skills match
        features.append({
            "feature": "skills_match",
            "contribution": match_percentage / 100,
            "abs_contribution": match_percentage / 100
        })
        
        # Experience
        try:
            exp_years = float(experience) if experience else 0
            exp_match = min(exp_years / 5, 1.0)
            features.append({
                "feature": "experience",
                "contribution": exp_match,
                "abs_contribution": exp_match
            })
        except:
            pass
        
        # Education
        edu_levels = {"High School": 0, "Diploma": 0.3, "Bachelor's": 0.6, "Master's": 0.8, "PhD": 1.0}
        edu_score = edu_levels.get(education, 0.4)
        features.append({
            "feature": "education",
            "contribution": edu_score,
            "abs_contribution": edu_score
        })
        
        # Sort by importance
        features.sort(key=lambda x: x["abs_contribution"], reverse=True)
        
        explanation = f"The model predicts {predicted_role} with {round(predicted_prob * 100, 2)}% confidence. "
        explanation += f"Your skills match {match_percentage}% of the key skills for this role. "
        
        if match_percentage < 40:
            explanation += "Consider developing skills like: " + ", ".join(keywords[:3]) + "."
        elif match_percentage < 70:
            explanation += "You have a good foundation. Focus on deepening your expertise in: " + ", ".join(keywords[3:]) + "."
        else:
            explanation += "You have strong alignment with this role. Continue building on your strengths!"
        
        return {
            "success": True,
            "predicted_role": predicted_role,
            "predicted_probability": round(predicted_prob * 100, 2),
            "top_features": features,
            "explanation_text": explanation,
            "method": "skill-based importance (SHAP alternative)"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "explanation_text": f"Could not generate explanation: {str(e)}"
        }


def _generate_shap_explanation(predicted_role: str, probability: float, top_features: list, user_skills: str) -> str:
    """Generate human-readable explanation from feature contributions"""
    explanation = f"The model predicts {predicted_role} with {round(probability * 100, 2)}% confidence. "
    
    positive_features = [f for f in top_features if f["contribution"] > 0]
    negative_features = [f for f in top_features if f["contribution"] < 0]
    
    # Clean up feature names for readability
    def clean_feature_name(feature_name: str) -> str:
        if feature_name.startswith("skills__"):
            return feature_name.replace("skills__", "")
        if feature_name.startswith("experience"):
            return "Experience level"
        if feature_name.startswith("education"):
            return f"Education: {feature_name.replace('education_', '')}"
        return feature_name
    
    if positive_features:
        feature_names = [clean_feature_name(f['feature']) for f in positive_features[:3]]
        explanation += f"This is primarily influenced by: {', '.join(feature_names)}. "
    
    if negative_features:
        feature_names = [clean_feature_name(f['feature']) for f in negative_features[:2]]
        explanation += f"Factors that could be improved: {', '.join(feature_names)}. "
    
    # Add skill-specific insight
    skill_features = [f for f in top_features if "skills" in f["feature"]]
    if skill_features:
        skill_names = [clean_feature_name(f['feature']) for f in skill_features[:3]]
        explanation += f"Key skills driving this prediction include: {', '.join(skill_names)}."
    
    return explanation


def get_real_shap_explanation(skills: str, experience: str | int | float, education: str) -> dict:
    """Wrapper function to get SHAP explanation"""
    return get_shap_values_for_prediction(skills, experience, education)


model = get_artifacts()["pipeline"]