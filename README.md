# AI Career Prediction System

An AI-powered career recommendation system that predicts the most suitable tech career role based on your skills, experience, and education.

---

## 📌 Overview

**CareerSim AI** is a machine learning-powered web application that helps students and professionals make data-driven career decisions.
Built as part of the **PDP AI Program at SLIIT City Uni**, this project demonstrates the application of AI in career guidance.

The system analyzes user inputs (skills, experience, education) and predicts the most suitable career role from five tech categories,
providing transparent explanations and personalized career advice.

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 🎯 **Career Prediction** | Predicts the best matching career role using ML |
| 📊 **Probability Distribution** | Shows probability scores for all 5 career roles |
| 📈 **Readiness Score** | Calculates how prepared you are for the predicted role |
| 🔍 **SHAP Explanations** | Explains *why* a specific role was recommended |
| 💬 **AI Chat Assistant** | Personalized career advice via Groq API |
| 📚 **Skill Gap Analysis** | Identifies missing skills and recommends improvements |
| 🎨 **Modern UI** | Glassmorphism design with dark theme |

---

## 🏗️ Tech Stack

### Frontend
| Technology | Purpose |
|------------|---------|
| HTML5 | Structure |
| CSS3 | Glassmorphism design & styling |
| Bootstrap 5.3 | Responsive layout |
| JavaScript | Form validation & interactivity |
| Plotly.js | Interactive probability charts |
| Google Fonts | Space Grotesk typography |

### Backend
| Technology | Purpose |
|------------|---------|
| Python 3.x | Core programming language |
| Flask | Web framework |
| Jinja2 | Template rendering |

### Machine Learning
| Technology | Purpose |
|------------|---------|
| scikit-learn | ML algorithms & preprocessing |
| Logistic Regression | Classification model |
| SHAP (KernelExplainer) | Model explainability |
| pandas | Data processing |
| numpy | Numerical operations |
| joblib | Model persistence |

### AI & Integration
| Technology | Purpose |
|------------|---------|
| Groq API | AI chat assistant |
| OpenAI Client | Groq API integration |

---

## 📊 Dataset

**Source:** Stack Overflow Developer Survey 2023 from kaggle

## 🤖 Machine Learning Model

### Algorithm: Logistic Regression (Multiclass Classification)

**Why Logistic Regression?**
- ✅ Interpretable - easy to understand feature importance
- ✅ Provides probability scores for each role
- ✅ Fast training and prediction
- ✅ Well-calibrated confidence scores

### Feature Engineering

| Feature | Processing |
|---------|------------|
| Skills | TF-IDF Vectorization (6,000+ features) |
| Experience | StandardScaler Normalization |
| Education | One-Hot Encoding |

### Model Performance

| Metric | Value |
|--------|-------|
| **Accuracy** | ~70-75% |
| **Precision** | Balanced across roles |
| **Recall** | Good for major roles |

---

## 🚀 Installation & Setup

### Prerequisites

- Python 3.8+
- Git
- Groq API Key (optional, for AI chat)

### Step 1: Clone the Repository

```bash
git clone https://github.com/Imthiyas-Sara/AI-Career-Prediction-System.git
cd AI-Career-Prediction-System
```

### Step 2: Create Virtual Environment

```bash
# On Windows
python -m venv venv
venv\Scripts\activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

### Step 4: Set Up Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your-groq-api-key-here
GROQ_BASE_URL=https://api.groq.com/openai/v1
GROQ_MODEL=mixtral-8x7b-32768
SECRET_KEY=your-secret-key-here
```

### Step 5: Download the Dataset

The dataset (`survey.csv`) is excluded from GitHub due to file size limits.

1. Download the Stack Overflow Developer Survey 2023 from [Kaggle](https://www.kaggle.com/datasets/stackoverflow/stack-overflow-developer-survey-2023)
2. Place `survey.csv` in the project root directory

### Step 6: Run the Application

```bash
python app.py
```

The application will be available at `http://127.0.0.1:5000`

---

## 📊 How to Use

### Step 1: Enter Your Details

| Field | Description | Example |
|-------|-------------|---------|
| **Skills** | Your technical skills (comma-separated) | Python, SQL, Docker, TensorFlow |
| **Experience** | Years of professional experience | 4 |
| **Target Year** | Future year for career planning | 2030 |
| **Education** | Highest education level | Bachelor's, Master's, PhD |

### Step 2: Get Results

1. Click **"Predict"**
2. View your predicted career role
3. Explore the probability distribution chart
4. Check your readiness score
5. Review SHAP explanation for the prediction
6. See recommended skills to develop

### Step 3: Ask the AI Assistant

- Open the floating chat widget
- Ask career-related questions
- Get personalized advice based on your profile

---

## 💡 Key Features Explained

### 1. SHAP Explainability

```
┌─────────────────────────────────────────────────────────────┐
│  Feature Contributions (SHAP Values)                        │
│                                                             │
│  python        -64.96%  ⬇ Decreases chance                 │
│  java          +52.44%  ⬆ Increases chance                 │
│  docker        +41.74%  ⬆ Increases chance                 │
│                                                             │
│  🔍 SHAP shows how each skill influenced the prediction     │
└─────────────────────────────────────────────────────────────┘
```

### 2. AI Chat Assistant

**What you can ask:**
- "What career path should I choose based on my skills?"
- "How do I become a Data Scientist?"
- "What skills do I need for Backend Development?"
- "How do I prepare for a tech interview?"

**What is blocked:**
- Off-topic questions (e.g., "What is the capital of Sri Lanka?")
- Non-career related queries

---

## 📚 Learning Outcomes

### Technical Skills Acquired

| Area | Skills |
|------|--------|
| **Machine Learning** | Logistic Regression, TF-IDF, SHAP explainability |
| **Data Engineering** | Data cleaning, feature engineering, model persistence |
| **Web Development** | Flask, HTML/CSS, JavaScript, Bootstrap |
| **UI/UX** | Glassmorphism design, dark theme, responsive design |
| **AI Integration** | Groq API, prompt engineering, guardrails |

### Soft Skills

- ✅ Full-stack project development
- ✅ Problem-solving and debugging
- ✅ Version control with Git
- ✅ Security best practices
- ✅ Documentation and presentation

---

## 📧 Contact

**Developer:** Imthiyas Sara

**GitHub:** [Imthiyas-Sara](https://github.com/Imthiyas-Sara)

**Project Link:** [AI-Career-Prediction-System](https://github.com/Imthiyas-Sara/AI-Career-Prediction-System)

---

*Made with ❤️ as part of the PDP AI Program at SLIIT City Uni*
