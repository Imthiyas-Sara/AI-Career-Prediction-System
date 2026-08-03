import os
from typing import Dict, Optional
import json
import re

class CareerAIAssistant:
    """AI Assistant with career-specific responses and strict guardrails"""
    
    def __init__(self):
        self.career_roles = [
            "Data Scientist",
            "Data Engineer", 
            "Backend Developer",
            "ML Engineer",
            "DevOps Engineer",
            "Software Developer",
            "Full Stack Developer"
        ]
        
        # Career-related keywords for filtering
        self.career_keywords = [
            "career", "job", "work", "profession", "role", "position",
            "skill", "learn", "development", "growth", "advancement",
            "resume", "interview", "salary", "network", "leadership",
            "promotion", "transition", "planning", "mentor", "goal",
            "tech", "data", "engineering", "science", "software",
            "python", "java", "sql", "cloud", "aws", "docker", "kubernetes",
            "machine learning", "data science", "devops", "backend",
            "frontend", "full stack", "analytics", "product", "project",
            "coding", "programming", "developer", "engineer", "scientist",
            "certification", "course", "training", "bootcamp", "degree",
            "ml", "ai", "artificial intelligence", "deep learning",
            "framework", "library", "tool", "technology", "stack"
        ]
        
        # Off-topic keywords to block
        self.off_topic_keywords = [
            "fruit", "capital", "country", "city", "food", "recipe",
            "movie", "song", "music", "celebrity", "sport", "game",
            "weather", "history", "geography", "animal", "plant",
            "color", "shape", "number", "letter", "word", "dictionary",
            "what is the capital", "name 5", "list 5", "tell me 5",
            "banana", "apple", "mango", "orange", "grape", "cherry",
            "how to eat", "how to cook", "recipe for", "taste",
            "hi", "hello", "hey", "good morning", "good evening"
        ]
    
    def _is_career_related(self, query: str) -> tuple[bool, str]:
        """Check if the query is career-related"""
        query_lower = query.lower().strip()
        
        # Check for off-topic keywords first
        for off_topic in self.off_topic_keywords:
            if off_topic in query_lower:
                return False, f"🤖 I'm a career-focused AI assistant. I can only help with career-related questions like:\n\n• How do I become a Data Scientist?\n• What skills do I need for Backend Development?\n• How do I prepare for a tech interview?\n• What's the career growth path for ML Engineer?\n• How do I transition from [role] to [role]?\n\nPlease ask me about careers, skills, job searching, or professional growth!"
        
        # Check for career keywords
        for keyword in self.career_keywords:
            if keyword in query_lower:
                return True, ""
        
        return False, "🤖 I specialize in career advice! Please ask me questions about:\n\n• Career planning and growth\n• Skills development\n• Job searching and interviews\n• Career transitions\n• Tech industry insights\n\nI'm here to help you with your professional journey! 🚀"
    
    def _detect_role_from_query(self, query: str) -> str:
        """Detect which role is being asked about"""
        query_lower = query.lower()
        
        role_mappings = {
            "data scientist": "Data Scientist",
            "data engineer": "Data Engineer",
            "backend developer": "Backend Developer",
            "backend dev": "Backend Developer",
            "ml engineer": "ML Engineer",
            "machine learning engineer": "ML Engineer",
            "devops engineer": "DevOps Engineer",
            "devops": "DevOps Engineer",
            "software developer": "Software Developer",
            "software dev": "Software Developer",
            "software engineer": "Software Developer",
            "full stack developer": "Full Stack Developer",
            "full stack": "Full Stack Developer",
            "frontend developer": "Frontend Developer",
            "frontend dev": "Frontend Developer"
        }
        
        for key, role in role_mappings.items():
            if key in query_lower:
                return role
        
        # If no specific role detected, return None
        return None
    
    def _detect_topic_from_query(self, query: str) -> str:
        """Detect what topic is being asked about"""
        query_lower = query.lower()
        
        if "interview" in query_lower:
            return "interview"
        elif "salary" in query_lower:
            return "salary"
        elif "cloud" in query_lower or "aws" in query_lower or "azure" in query_lower:
            return "cloud"
        elif "python" in query_lower:
            return "python"
        elif "sql" in query_lower:
            return "sql"
        elif "certification" in query_lower or "cert" in query_lower:
            return "certification"
        elif "portfolio" in query_lower:
            return "portfolio"
        elif "resume" in query_lower:
            return "resume"
        elif "growth" in query_lower or "plan" in query_lower:
            return "growth"
        else:
            return "general"
    
    def _get_career_response(self, query: str, user_profile: Dict) -> str:
        """Generate a detailed career response"""
        
        predicted_role = user_profile.get('predicted_role', 'Backend Developer')
        readiness_score = user_profile.get('readiness_score', 0)
        user_skills = user_profile.get('skills', '')
        experience = user_profile.get('experience', 0)
        education = user_profile.get('education', '')
        
        # Detect the role being asked about
        detected_role = self._detect_role_from_query(query)
        
        # If a specific role is detected, use that instead of predicted_role
        target_role = detected_role if detected_role else predicted_role
        
        # Detect the topic
        topic = self._detect_topic_from_query(query)
        
        # Get advice for the target role
        role_advice = self.advice_templates.get(target_role, self.advice_templates["Backend Developer"])
        
        # ============================================
        # HANDLE DIFFERENT TOPICS
        # ============================================
        
        # Interview Questions
        if topic == "interview":
            return self._get_interview_advice(target_role, user_profile)
        
        # Salary Questions
        elif topic == "salary":
            return self._get_salary_advice(target_role)
        
        # Cloud Questions
        elif topic == "cloud":
            return self._get_cloud_advice(target_role)
        
        # Python Questions
        elif topic == "python":
            return self._get_python_advice(target_role)
        
        # SQL Questions
        elif topic == "sql":
            return self._get_sql_advice(target_role)
        
        # Certification Questions
        elif topic == "certification":
            return self._get_certification_advice(target_role)
        
        # Portfolio Questions
        elif topic == "portfolio":
            return self._get_portfolio_advice(target_role)
        
        # Growth/Plan Questions
        elif topic == "growth":
            return self._get_growth_plan(target_role, user_profile)
        
        # Career Path Questions
        elif "career path" in query.lower() or "which career" in query.lower() or "choose" in query.lower():
            return self._get_career_path_advice(user_profile)
        
        # General/Other Questions
        else:
            # Check if asking how to become a specific role
            if any(word in query.lower() for word in ["become", "how to", "get into", "learn"]) and detected_role:
                return self._get_detailed_role_advice(target_role, user_profile)
            
            # Default to general advice
            return self._get_general_advice(target_role, user_profile)
    
    def _get_detailed_role_advice(self, role: str, user_profile: Dict) -> str:
        """Get detailed advice for a specific role"""
        role_advice = self.advice_templates.get(role, self.advice_templates["Backend Developer"])
        
        # Get role-specific salary ranges
        salaries = {
            "Data Scientist": "95-160K",
            "Data Engineer": "100-170K",
            "Backend Developer": "90-150K",
            "ML Engineer": "110-180K",
            "DevOps Engineer": "105-165K",
            "Software Developer": "85-145K",
            "Full Stack Developer": "90-155K"
        }
        
        salary_range = salaries.get(role, "90-150K")
        
        # Get role-specific growth projections
        growth = {
            "Data Scientist": "22% (much faster than average)",
            "Data Engineer": "25% (much faster than average)",
            "Backend Developer": "18% (faster than average)",
            "ML Engineer": "28% (much faster than average)",
            "DevOps Engineer": "20% (faster than average)",
            "Software Developer": "15% (faster than average)",
            "Full Stack Developer": "17% (faster than average)"
        }
        
        growth_rate = growth.get(role, "20% (faster than average)")
        
        # Get role description
        descriptions = {
            "Data Scientist": "Data Scientists analyze complex data to help companies make better decisions. They use statistics, machine learning, and programming to extract insights from data.",
            "Data Engineer": "Data Engineers build and maintain the infrastructure that allows data to be collected, stored, and analyzed. They create data pipelines and ensure data quality.",
            "Backend Developer": "Backend Developers build and maintain the server-side logic, databases, and APIs that power web applications. They ensure everything works efficiently behind the scenes.",
            "ML Engineer": "ML Engineers design, build, and deploy machine learning models at scale. They bridge the gap between data science and software engineering.",
            "DevOps Engineer": "DevOps Engineers automate and optimize the software development lifecycle. They manage infrastructure, CI/CD pipelines, and deployment processes.",
            "Software Developer": "Software Developers design, code, and maintain software applications. They work on everything from mobile apps to enterprise systems.",
            "Full Stack Developer": "Full Stack Developers work on both frontend and backend development. They build complete web applications from database to user interface."
        }
        
        description = descriptions.get(role, f"{role}s build and maintain technology infrastructure.")
        
        return f"""🎯 **How to Become a {role}**

**About the Role:**
{description}

This role is in high demand with {growth_rate} job growth projected.

**📊 Quick Facts:**
• Average Salary: ${salary_range}
• Job Growth: {growth_rate}
• Typical Education: Bachelor's or Master's
• Remote Friendly: Yes (60%+ jobs)

**🛠️ Essential Skills You Need:**
{', '.join(role_advice['skills'])}

**📚 Step-by-Step Roadmap:**

**Phase 1: Foundation (0-6 months)**
• Learn {role_advice['skills'][0]} (core language)
• Understand {role_advice['skills'][1]} basics
• Take online courses
• Build small projects

**Phase 2: Intermediate (6-12 months)**
• Master {role_advice['skills'][2]}
• Learn {role_advice['skills'][3] if len(role_advice['skills']) > 3 else 'advanced concepts'}
• Contribute to open source
• Complete certifications

**Phase 3: Advanced (12-24 months)**
• Specialize in specific area
• Build complex projects
• Mentor others
• Lead technical initiatives

**🎓 Recommended Learning Resources:**
• {', '.join(role_advice['resources'][:3])}

**💼 How to Get Your First Job:**
1. Build a strong portfolio (3-5 projects)
2. Practice technical interviews daily
3. Network on LinkedIn and GitHub
4. Apply to 10-15 jobs per week
5. Prepare for system design questions

**⚡ Action Items for This Week:**
1. Pick one online course and start
2. Set up your development environment
3. Write your first line of code
4. Join a relevant community

Your current readiness for {role} is {user_profile.get('readiness_score', 0)}%. With consistent effort, you can be job-ready in 12-18 months!

Want specific advice on any of these areas? 🚀"""
    
    def _get_interview_advice(self, role: str, user_profile: Dict) -> str:
        """Get interview advice for a specific role"""
        readiness = user_profile.get('readiness_score', 0)
        
        return f"""🎯 **Interview Preparation for {role}**

**📋 Common Interview Rounds:**
1. **Phone Screen** (30 min)
   • Resume review
   • Basic technical questions
   • Cultural fit assessment

2. **Technical Round 1** (60 min)
   • Coding challenge
   • Language proficiency
   • Problem-solving

3. **Technical Round 2** (60 min)
   • System design
   • Architecture discussion
   • Scalability questions

4. **Behavioral Round** (60 min)
   • STAR method
   • Teamwork scenarios
   • Career goals

**💻 Technical Topics to Prepare:**
• Data structures and algorithms
• System design basics
• Language-specific questions
• Design patterns
• Best practices

**💰 Salary Expectations:**
• Entry-Level: $70-90K
• Mid-Level: $100-140K
• Senior: $150-200K+

**🎯 Interview Tips:**
1. Practice LeetCode daily (15-30 min)
2. Know your projects in detail
3. Prepare STAR method answers
4. Research the company

**📚 Resources:**
• LeetCode (Coding practice)
• "Cracking the Coding Interview"
• Pramp (Mock interviews)
• Glassdoor (Company reviews)

Your readiness is {readiness}%. With 2-3 months of focused practice, you can ace interviews! 💪"""
    
    def _get_cloud_advice(self, role: str) -> str:
        """Get cloud computing advice"""
        return f"""☁️ **Cloud Computing for {role}**

**Why Cloud?**
85% of companies use cloud services. Cloud skills can increase your salary by 30%.

**Top Platforms:**
1. **AWS** (65% market share) - Most jobs, best for beginners
2. **Azure** (20% market share) - Good for enterprise
3. **GCP** (10% market share) - Strong in AI/ML

**🎯 Recommended Path:**
1. Start with AWS (most job opportunities)
2. Use free tier for hands-on practice
3. Build a project using cloud
4. Consider certification

**📚 Learning Resources:**
• AWS YouTube channel (Free)
• A Cloud Guru (Best paid)
• Qwiklabs (Hands-on)
• Cloud Academy

**💼 Career Impact:**
• 2x faster promotions with cloud skills
• 30% higher salary potential
• Remote work opportunities increase

**Key Cloud Services to Learn:**
• Compute (EC2, Lambda)
• Storage (S3, EBS)
• Databases (RDS, DynamoDB)
• Networking (VPC, Route53)
• Security (IAM)

**Certification Path:**
1. AWS Cloud Practitioner (Foundational)
2. AWS Solutions Architect Associate (Most popular)
3. AWS DevOps Engineer (For DevOps)
4. AWS ML Specialty (For ML roles)

Your cloud journey starts now! 🚀"""
    
    def _get_python_advice(self, role: str) -> str:
        """Get Python learning advice"""
        return f"""🐍 **Learning Python for {role}**

**Why Python?**
Python is the #1 language for {role} with 85%+ usage rate.

**📚 Learning Roadmap:**

**Weeks 1-2: Basics**
• Variables, data types
• Lists, dictionaries
• Loops and conditionals
• Functions

**Weeks 3-4: Intermediate**
• OOP and classes
• File handling
• Error handling
• Modules and packages

**Weeks 5-8: Advanced**
• Decorators, generators
• Testing
• Optimization
• Framework basics

**🎯 Best Resources:**
• 📚 "Automate the Boring Stuff" (Free)
• 📺 Corey Schafer (YouTube)
• 🎓 Coursera Python for Everybody
• 💻 LeetCode/HackerRank

**💪 Practice Projects:**
1. To-Do List App
2. Web Scraper
3. REST API
4. Data Analysis Dashboard
5. Automation Script

**⏰ Recommended Schedule:**
• Daily: 1-2 hours coding
• Weekly: 5-10 hours total
• Monthly: Complete 1 project

Master Python in 3-6 months of consistent practice! 🚀"""
    
    def _get_sql_advice(self, role: str) -> str:
        """Get SQL learning advice"""
        return f"""🗄️ **SQL for {role}**

**Why SQL?**
SQL is essential for data roles. 75% of {role} jobs require SQL.

**📚 Learning Roadmap:**

**Basics (Weeks 1-2):**
• SELECT, FROM, WHERE
• ORDER BY, LIMIT
• JOINs (INNER, LEFT)
• GROUP BY, HAVING

**Intermediate (Weeks 3-4):**
• Subqueries
• Window functions
• CTEs
• Views

**Advanced (Weeks 5-8):**
• Query optimization
• Indexing strategies
• Stored procedures
• Database design

**🎯 Best Resources:**
• 📚 Mode Analytics SQL Tutorial (Free)
• 📺 SQL for Data Science (Coursera)
• 💻 LeetCode SQL problems
• 📱 DataCamp SQL courses

**💪 Practice Projects:**
1. Analyze a dataset using SQL
2. Build a database schema
3. Write complex queries
4. Optimize slow queries

Master SQL in 2-3 months of consistent practice! 🚀"""
    
    def _get_salary_advice(self, role: str) -> str:
        """Get salary advice"""
        salaries = {
            "Data Scientist": {"entry": "85-110K", "mid": "120-160K", "senior": "160-220K"},
            "Data Engineer": {"entry": "90-115K", "mid": "130-170K", "senior": "170-230K"},
            "Backend Developer": {"entry": "80-105K", "mid": "110-150K", "senior": "150-200K"},
            "ML Engineer": {"entry": "95-120K", "mid": "140-180K", "senior": "180-250K"},
            "DevOps Engineer": {"entry": "85-110K", "mid": "120-160K", "senior": "160-220K"},
            "Software Developer": {"entry": "75-100K", "mid": "105-140K", "senior": "140-190K"},
            "Full Stack Developer": {"entry": "80-105K", "mid": "110-150K", "senior": "150-200K"}
        }
        
        salary = salaries.get(role, {"entry": "70-95K", "mid": "100-140K", "senior": "140-190K"})
        
        return f"""💰 **Salary Guide for {role}**

**Average Salaries by Level:**

| Level | Salary Range | Experience |
|-------|-------------|------------|
| Entry Level | ${salary['entry']} | 0-2 years |
| Mid Level | ${salary['mid']} | 3-5 years |
| Senior Level | ${salary['senior']} | 6+ years |

**🏢 By Company Size:**
• Startup: 10-15% below average
• Mid-size: Average
• Large Tech (FAANG): 20-40% above average

**📍 By Location (US):**
• Bay Area/NYC: +30-50%
• Seattle/Austin: +10-20%
• Other US: Average
• Remote: Varies

**💡 Salary Negotiation Tips:**
1. Research market rates (Glassdoor, Levels.fyi)
2. Know your worth - add 10-20% to your ask
3. Consider total compensation (bonus, equity)
4. Practice negotiation conversations
5. Get multiple offers if possible

**📈 Salary Growth:**
• Year 1-2: 5-10% increase
• Year 3-5: 15-20% increase (promotion)
• Year 5+: 10-15% annual increase

Want specific advice on negotiating your next offer? 💪"""
    
    def _get_certification_advice(self, role: str) -> str:
        """Get certification advice"""
        certifications = {
            "Data Scientist": ["AWS ML Specialty", "Google ML", "IBM Data Science", "SAS Certified"],
            "Data Engineer": ["AWS Data Analytics", "Google Data Engineer", "Azure Data Engineer", "CDMP"],
            "Backend Developer": ["AWS Solutions Architect", "Azure Developer", "Google Cloud Developer", "Oracle Java"],
            "ML Engineer": ["AWS ML Specialty", "Google ML Engineer", "Azure AI Engineer", "MLOps Certification"],
            "DevOps Engineer": ["AWS DevOps", "Azure DevOps", "Google Cloud DevOps", "CKAD", "CKA"],
            "Software Developer": ["AWS Developer", "Google Cloud Developer", "Microsoft Developer", "Oracle Java"],
            "Full Stack Developer": ["AWS Developer", "Google Cloud Developer", "MongoDB", "Node.js Certified"]
        }
        
        cert_list = certifications.get(role, ["AWS Solutions Architect", "Azure Developer", "Google Cloud Developer"])
        
        return f"""📜 **Certification Guide for {role}**

**Top Certifications:**

1. **{cert_list[0]}**
   • Best for: Career growth
   • Cost: $100-300
   • Time: 2-3 months prep
   • Career impact: +20% salary

2. **{cert_list[1] if len(cert_list) > 1 else 'AWS Solutions Architect'}**
   • Best for: Cloud expertise
   • Cost: $100-250
   • Time: 1-2 months prep
   • Career impact: +15% salary

3. **{cert_list[2] if len(cert_list) > 2 else 'Google Cloud Developer'}**
   • Best for: Specialization
   • Cost: $100-200
   • Time: 1-2 months prep
   • Career impact: +10-15% salary

**📚 Study Resources:**
• Official certification guides
• Udemy courses
• Practice exams
• Study groups

**💡 Certification Strategy:**
1. Start with one certification
2. Build real projects alongside
3. Recertify every 2-3 years
4. Use certs to negotiate salary

**💰 ROI:**
Certifications typically pay for themselves within 3-6 months through salary increases.

Ready to start your certification journey? 🎯"""
    
    def _get_portfolio_advice(self, role: str) -> str:
        """Get portfolio advice"""
        return f"""💼 **Building a Portfolio for {role}**

**Why a Portfolio?**
85% of hiring managers look at portfolios before scheduling interviews.

**🎯 Portfolio Project Ideas:**

**Beginner (1-2 projects):**
1. To-Do App with API
2. Data Analysis Dashboard
3. Simple Website

**Intermediate (2-3 projects):**
1. Full Stack Application
2. ETL Pipeline
3. ML Model Deployment
4. CI/CD Pipeline

**Advanced (3-5 projects):**
1. Scalable Microservices
2. Data Platform
3. Production ML System
4. Cloud Infrastructure

**📁 Portfolio Structure:**
• 3-5 well-documented projects
• Clean, professional presentation
• Live demo links
• Source code on GitHub

**💡 Tips:**
1. Quality over quantity
2. Document your code
3. Explain your decisions
4. Show problem-solving
5. Highlight your role

**🛠️ Best Tools:**
• GitHub (Code hosting)
• Vercel/Netlify (Deployment)
• Heroku (Backend hosting)
• Tech stack aligned with role

Start building your portfolio today! 🚀"""
    
    def _get_career_path_advice(self, user_profile: Dict) -> str:
        """Get career path advice"""
        predicted_role = user_profile.get('predicted_role', 'Backend Developer')
        readiness = user_profile.get('readiness_score', 0)
        skills = user_profile.get('skills', '')
        experience = user_profile.get('experience', 0)
        education = user_profile.get('education', '')
        
        # Get alternative paths
        alternatives = {
            "Data Scientist": ["ML Engineer", "Data Engineer", "Research Scientist"],
            "Data Engineer": ["Data Scientist", "Backend Developer", "Cloud Engineer"],
            "Backend Developer": ["DevOps Engineer", "Full Stack Developer", "Tech Lead"],
            "ML Engineer": ["Data Scientist", "Research Engineer", "AI Architect"],
            "DevOps Engineer": ["SRE", "Cloud Architect", "Platform Engineer"],
            "Software Developer": ["Full Stack Developer", "Tech Lead", "Software Architect"],
            "Full Stack Developer": ["Tech Lead", "Software Architect", "Product Manager"]
        }
        
        alt_paths = alternatives.get(predicted_role, ["Senior " + predicted_role, "Tech Lead", "Architect"])
        
        return f"""🎯 **Career Path Analysis for You**

**Based on Your Profile:**
• **Current Skills**: {skills or 'Not specified'}
• **Experience**: {experience} years
• **Education**: {education}
• **Predicted Role**: {predicted_role} (readiness: {readiness}%)

**✅ Recommended Career Path**: {predicted_role}

**Why This Role?**
Your skills and background align well with {predicted_role} requirements:
• {predicted_role}s are in high demand
• Strong career progression opportunities
• Competitive compensation packages

**🔄 Alternative Paths to Consider:**
1. **{alt_paths[0]}** - Deepen expertise (2-4 years)
2. **{alt_paths[1]}** - Move into leadership (4-6 years)
3. **{alt_paths[2]}** - Broaden your scope (5-8 years)

**📊 Career Comparison:**

| Aspect | {predicted_role} | {alt_paths[0]} | {alt_paths[1]} |
|--------|-----------------|---------------|---------------|
| Learning Curve | Medium | High | Medium |
| Job Demand | High | Medium-High | High |
| Salary Range | $$ | $$$ | $$$ |

**🎯 Your Next Steps:**
1. **This Week:**
   • Research {predicted_role} roles
   • Update your skills list
   • Join relevant communities

2. **This Month:**
   • Start one online course
   • Build a small project
   • Network with professionals

3. **This Quarter:**
   • Complete a certification
   • Contribute to open source
   • Apply for relevant roles

Would you like me to elaborate on any specific career path? 🚀"""
    
    def _get_growth_plan(self, role: str, user_profile: Dict) -> str:
        """Get growth plan"""
        readiness = user_profile.get('readiness_score', 0)
        experience = user_profile.get('experience', 0)
        role_advice = self.advice_templates.get(role, self.advice_templates["Backend Developer"])
        
        return f"""📈 **5-Year Growth Plan: {role}**

**Current Status:**
• Role: {role}
• Readiness: {readiness}%
• Experience: {experience} years

**📆 Timeline:**

**Year 1: Build Foundation**
• Master core skills: {', '.join(role_advice['skills'][:3])}
• Complete 2-3 certifications
• Build 2 portfolio projects
• Read 5 tech books

**Year 2: Deepen Expertise**
• Specialize in {role_advice['skills'][0]}
• Contribute to open source
• Start mentoring juniors
• Present at meetups

**Year 3: Lead Projects**
• Lead a team project
• Design system architecture
• Consider promotion
• Network strategically

**Year 4: Expand Influence**
• Technical leadership
• Cross-team collaboration
• Process improvement
• Industry recognition

**Year 5: Senior/Lead Role**
• Senior {role}
• Team leadership
• Strategic decisions
• Mentorship program

**💰 Salary Progression:**
• Now: $90K average
• Year 3: $130K
• Year 5: $180K

**🎯 Key Milestones:**
✅ Complete one course per quarter
✅ Build one project per month
✅ Network with 5 professionals monthly
✅ Practice interview questions weekly

Your readiness score of {readiness}% shows you're beginning. Focus on Year 1 goals first! 🚀"""
    
    def _get_general_advice(self, role: str, user_profile: Dict) -> str:
        """Get general career advice"""
        readiness = user_profile.get('readiness_score', 0)
        role_advice = self.advice_templates.get(role, self.advice_templates["Backend Developer"])
        
        return f"""💼 **Career Advice for {role}**

**Key Recommendations:**
1. Build these core skills: {', '.join(role_advice['skills'][:3])}
2. Create a portfolio with real projects
3. Network with industry professionals
4. Stay updated with technology trends

**Current Status:**
• Role: {role}
• Readiness: {readiness}%
• Skills gap: {100 - readiness if readiness < 100 else 0}%

**⚡ Quick Actions (This Week):**
✅ Start one online course
✅ Build a small project
✅ Join a tech community
✅ Update your LinkedIn

**🎯 Goals (This Month):**
• Complete 1 course
• Build 1 portfolio project
• Network with 5 professionals
• Practice interview questions

**📚 Resources to Explore:**
• {', '.join(role_advice['resources'][:3])}

**💡 Remember:** Every expert was once a beginner. Your {readiness}% readiness is a great starting point!

Want specific advice on skills, interviews, or career growth? Just ask! 🚀"""
    
    # Pre-defined career advice templates
    advice_templates = {
        "Data Scientist": {
            "skills": ["Python", "SQL", "Statistics", "Machine Learning", "Data Visualization", "Deep Learning", "NLP", "Big Data"],
            "resources": ["Kaggle", "Coursera ML Course", "DataCamp", "Google Data Studio", "Fast.ai"]
        },
        "Data Engineer": {
            "skills": ["Python", "SQL", "ETL Pipelines", "Cloud", "Big Data", "Spark", "Airflow", "Kafka", "AWS"],
            "resources": ["AWS Certification", "Apache Airflow docs", "Data Engineering blogs", "Databricks", "dbt"]
        },
        "Backend Developer": {
            "skills": ["Python", "Java", "REST APIs", "Databases", "Docker", "Git", "Spring Boot", "Microservices"],
            "resources": ["FastAPI/Django docs", "System Design interviews", "LeetCode", "AWS", "PostgreSQL docs"]
        },
        "ML Engineer": {
            "skills": ["Python", "ML Frameworks", "Cloud", "DevOps", "MLOps", "Docker", "Kubernetes", "PyTorch"],
            "resources": ["MLOps course", "Papers with Code", "TensorFlow docs", "PyTorch docs", "AWS SageMaker"]
        },
        "DevOps Engineer": {
            "skills": ["Linux", "Docker", "Kubernetes", "CI/CD", "Cloud", "Terraform", "Ansible", "Prometheus"],
            "resources": ["Docker docs", "Kubernetes docs", "AWS/GCP certs", "DevOps blogs", "Prometheus docs"]
        },
        "Software Developer": {
            "skills": ["Python", "Java", "JavaScript", "Git", "Databases", "REST APIs", "Testing", "Agile"],
            "resources": ["LeetCode", "Stack Overflow", "GitHub", "HackerRank", "Codecademy"]
        },
        "Full Stack Developer": {
            "skills": ["JavaScript", "React", "Node.js", "Python", "SQL", "REST APIs", "Git", "Docker"],
            "resources": ["The Odin Project", "freeCodeCamp", "MDN Docs", "Codecademy", "Frontend Masters"]
        }
    }
    
    def get_response(self, query: str, user_profile: Optional[Dict] = None) -> Dict:
        """Get AI response with strict career guardrails"""
        try:
            if not user_profile:
                user_profile = {
                    "skills": "",
                    "experience": 0,
                    "education": "",
                    "predicted_role": "Backend Developer",
                    "readiness_score": 0
                }
            
            # Check if query is career-related
            is_career, message = self._is_career_related(query)
            
            if not is_career:
                return {
                    "response": message,
                    "blocked": True,
                    "reason": "Off-topic question",
                    "success": True
                }
            
            # Generate career response
            response_text = self._get_career_response(query, user_profile)
            
            return {
                "response": response_text,
                "blocked": False,
                "success": True
            }
            
        except Exception as e:
            return {
                "response": "I encountered an error. Please try again with a career-related question.",
                "blocked": True,
                "error": str(e),
                "success": False
            }
    
    def get_career_advice(self, query: str, user_data: Dict) -> Dict:
        """Get career advice based on user profile and query"""
        return self.get_response(query, user_data)


# Initialize singleton
_assistant = None

def get_assistant() -> CareerAIAssistant:
    global _assistant
    if _assistant is None:
        _assistant = CareerAIAssistant()
    return _assistant