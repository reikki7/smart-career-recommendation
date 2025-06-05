from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
import json
import uuid
import requests
from werkzeug.utils import secure_filename
import PyPDF2
from io import BytesIO
import dotenv
from pathlib import Path
from bs4 import BeautifulSoup
import re
import urllib.parse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.metrics import precision_score, recall_score, f1_score
import pickle
from datetime import datetime
import logging
import time  # Add this import

dotenv.load_dotenv(Path(__file__).parent.parent / ".env")

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Environment variables
port = int(os.environ.get('PORT', 5000))
lm_studio_url = os.environ.get('LM_STUDIO_URL', 'http://localhost:1234')

ALLOWED_EXTENSIONS = {'pdf'}

# Initialize the sentence transformer model
try:
    model = SentenceTransformer('all-MiniLM-L6-v2')
    logger.info("Sentence transformer model loaded successfully")
except Exception as e:
    logger.error(f"Failed to load sentence transformer model: {e}")
    model = None

# Career database with embeddings
CAREER_DATABASE = {
    "Software Engineer": {
        "description": "Develop and maintain software applications, write clean code, debug programs, and collaborate with development teams",
        "required_skills": ["Python", "JavaScript", "Java", "Git", "SQL", "Problem Solving", "Algorithm Design"],
        "keywords": ["programming", "coding", "software development", "web development", "mobile development", "API", "database", "frontend", "backend"]
    },
    "Data Scientist": {
        "description": "Analyze complex data sets, build machine learning models, extract insights from data, and create data visualizations",
        "required_skills": ["Python", "R", "Machine Learning", "Statistics", "SQL", "Pandas", "NumPy", "Data Visualization"],
        "keywords": ["data analysis", "machine learning", "statistics", "data mining", "predictive modeling", "big data", "analytics", "visualization"]
    },
    "Machine Learning Engineer": {
        "description": "Design and implement machine learning systems, deploy ML models to production, optimize model performance",
        "required_skills": ["Python", "TensorFlow", "PyTorch", "Machine Learning", "Deep Learning", "MLOps", "Docker", "Cloud Computing"],
        "keywords": ["machine learning", "deep learning", "neural networks", "AI", "model deployment", "MLOps", "tensorflow", "pytorch"]
    },
    "Web Developer": {
        "description": "Create and maintain websites and web applications using various programming languages and frameworks",
        "required_skills": ["HTML", "CSS", "JavaScript", "React", "Node.js", "PHP", "MySQL", "Responsive Design"],
        "keywords": ["web development", "frontend", "backend", "full stack", "HTML", "CSS", "JavaScript", "React", "Vue", "Angular"]
    },
    "Mobile Developer": {
        "description": "Develop mobile applications for iOS and Android platforms using native or cross-platform technologies",
        "required_skills": ["Swift", "Kotlin", "React Native", "Flutter", "Mobile UI/UX", "API Integration", "App Store Deployment"],
        "keywords": ["mobile development", "iOS", "Android", "React Native", "Flutter", "mobile apps", "smartphone applications"]
    },
    "DevOps Engineer": {
        "description": "Manage infrastructure, automate deployment processes, ensure system reliability and scalability",
        "required_skills": ["Docker", "Kubernetes", "AWS", "CI/CD", "Linux", "Terraform", "Monitoring", "Automation"],
        "keywords": ["devops", "infrastructure", "deployment", "automation", "CI/CD", "cloud", "containerization", "orchestration"]
    },
    "UI/UX Designer": {
        "description": "Design user interfaces and user experiences for digital products, create wireframes and prototypes",
        "required_skills": ["Figma", "Adobe Creative Suite", "Prototyping", "User Research", "Wireframing", "Design Systems"],
        "keywords": ["UI design", "UX design", "user interface", "user experience", "prototyping", "wireframing", "design"]
    },
    "Product Manager": {
        "description": "Define product strategy, coordinate development teams, analyze market requirements and user feedback",
        "required_skills": ["Product Strategy", "Agile", "Stakeholder Management", "Market Analysis", "User Research", "Project Management"],
        "keywords": ["product management", "strategy", "roadmap", "stakeholder", "requirements", "market analysis", "agile"]
    },
    "Cybersecurity Specialist": {
        "description": "Protect systems and networks from security threats, implement security measures, conduct security audits",
        "required_skills": ["Network Security", "Penetration Testing", "Security Auditing", "Incident Response", "Cryptography", "Risk Assessment"],
        "keywords": ["cybersecurity", "security", "penetration testing", "network security", "information security", "risk assessment"]
    },
    "Business Analyst": {
        "description": "Analyze business processes, gather requirements, create documentation, and facilitate communication between stakeholders",
        "required_skills": ["Requirements Analysis", "Business Process Modeling", "SQL", "Excel", "Documentation", "Stakeholder Communication"],
        "keywords": ["business analysis", "requirements", "process improvement", "documentation", "stakeholder management", "business intelligence"]
    }
}

class CareerRecommendationSystem:
    def __init__(self):
        self.model = model
        self.career_embeddings = None
        self.skill_embeddings = None
        self.metrics_history = []
        self.detailed_logs = []  # Add this for detailed logging
        self._initialize_embeddings()
    
    def _initialize_embeddings(self):
        """Initialize career embeddings from the database"""
        if self.model is None:
            logger.error("Cannot initialize embeddings: model not loaded")
            return
        
        try:
            logger.info("🔄 Initializing career embeddings...")
            
            # Create text representations for each career
            career_texts = []
            career_names = list(CAREER_DATABASE.keys())
            
            for career_name in career_names:
                career_data = CAREER_DATABASE[career_name]
                # Combine description, skills, and keywords into one text
                combined_text = f"{career_data['description']} "
                combined_text += f"Required skills: {', '.join(career_data['required_skills'])} "
                combined_text += f"Keywords: {', '.join(career_data['keywords'])}"
                career_texts.append(combined_text)
            
            # Generate embeddings for all careers
            start_time = time.time()
            self.career_embeddings = self.model.encode(career_texts)
            embedding_time = time.time() - start_time
            
            logger.info(f"✅ Career embeddings initialized successfully")
            logger.info(f"   - Careers processed: {len(career_names)}")
            logger.info(f"   - Embedding dimensions: {self.career_embeddings.shape}")
            logger.info(f"   - Processing time: {embedding_time:.4f} seconds")
            
            # Store initialization log
            init_log = {
                'timestamp': datetime.now().isoformat(),
                'process': 'embedding_initialization',
                'careers_processed': len(career_names),
                'embedding_shape': self.career_embeddings.shape,
                'processing_time': embedding_time
            }
            self.detailed_logs.append(init_log)
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            self.career_embeddings = None
    
    def _extract_relevant_experience(self, user_data, career_keywords):
        """Extract relevant experience based on career keywords"""
        relevant_experiences = []
        
        try:
            # Convert user_data to searchable text
            if isinstance(user_data, dict):
                search_text = json.dumps(user_data).lower()
            else:
                search_text = str(user_data).lower()
            
            # Check which keywords appear in the user data
            found_keywords = []
            for keyword in career_keywords:
                if keyword.lower() in search_text:
                    found_keywords.append(keyword)
            
            # Create relevant experience entries based on found keywords
            if found_keywords:
                relevant_experiences.append(f"Relevant Skills & Experience: {', '.join(found_keywords)}")
            
            # Extract specific experiences if user_data is structured
            if isinstance(user_data, dict):
                # From experience section
                if 'experience' in user_data and isinstance(user_data['experience'], list):
                    for exp in user_data['experience'][:3]:  # Limit to top 3
                        if isinstance(exp, dict) and exp.get('title'):
                            description = exp.get('description', '')[:100] + '...' if len(exp.get('description', '')) > 100 else exp.get('description', '')
                            relevant_experiences.append(f"{exp.get('title', '')}: {description}")
                
                # From projects section
                if 'projects' in user_data and isinstance(user_data['projects'], list):
                    for proj in user_data['projects'][:2]:  # Limit to top 2
                        if isinstance(proj, dict) and proj.get('title'):
                            description = proj.get('description', '')[:100] + '...' if len(proj.get('description', '')) > 100 else proj.get('description', '')
                            relevant_experiences.append(f"{proj.get('title', '')}: {description}")
            
            return relevant_experiences[:5]  # Return max 5 experiences
            
        except Exception as e:
            logger.error(f"Error extracting relevant experience: {e}")
            return ['General Experience: Experience extracted from resume']
    
    def extract_features_from_text(self, text):
        """Extract features from resume text with detailed logging"""
        if self.model is None:
            return None
        
        try:
            start_time = time.time()
            
            # Clean and preprocess text
            original_length = len(text)
            text = re.sub(r'\s+', ' ', text.strip())
            cleaned_length = len(text)
            
            logger.info(f"📊 TEXT PREPROCESSING:")
            logger.info(f"   Original text length: {original_length} characters")
            logger.info(f"   Cleaned text length: {cleaned_length} characters")
            logger.info(f"   Text sample: {text[:200]}...")
            
            # Create embedding for the entire text
            embedding_start = time.time()
            text_embedding = self.model.encode([text])
            embedding_time = time.time() - embedding_start
            
            # Detailed embedding analysis
            embedding_vector = text_embedding[0]
            embedding_stats = {
                'shape': embedding_vector.shape,
                'mean': float(np.mean(embedding_vector)),
                'std': float(np.std(embedding_vector)),
                'min': float(np.min(embedding_vector)),
                'max': float(np.max(embedding_vector)),
                'norm': float(np.linalg.norm(embedding_vector)),
                'non_zero_elements': int(np.count_nonzero(embedding_vector))
            }
            
            logger.info(f"🤖 SENTENCE TRANSFORMER ANALYSIS:")
            logger.info(f"   Model: {self.model.get_sentence_embedding_dimension()}D all-MiniLM-L6-v2")
            logger.info(f"   Embedding time: {embedding_time:.4f} seconds")
            logger.info(f"   Embedding shape: {embedding_stats['shape']}")
            logger.info(f"   Embedding statistics:")
            logger.info(f"     - Mean: {embedding_stats['mean']:.6f}")
            logger.info(f"     - Std Dev: {embedding_stats['std']:.6f}")
            logger.info(f"     - Min value: {embedding_stats['min']:.6f}")
            logger.info(f"     - Max value: {embedding_stats['max']:.6f}")
            logger.info(f"     - L2 Norm: {embedding_stats['norm']:.6f}")
            logger.info(f"     - Non-zero elements: {embedding_stats['non_zero_elements']}/{len(embedding_vector)}")
            
            total_time = time.time() - start_time
            
            # Store detailed log
            detailed_log = {
                'timestamp': datetime.now().isoformat(),
                'process': 'feature_extraction',
                'text_stats': {
                    'original_length': original_length,
                    'cleaned_length': cleaned_length,
                    'preprocessing_ratio': cleaned_length / original_length if original_length > 0 else 0
                },
                'embedding_stats': embedding_stats,
                'timing': {
                    'total_time': total_time,
                    'embedding_time': embedding_time
                }
            }
            self.detailed_logs.append(detailed_log)
            
            return embedding_vector
            
        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            return None
    
    def calculate_career_scores(self, user_embedding):
        """Calculate similarity scores with detailed analysis"""
        if self.career_embeddings is None or user_embedding is None:
            return {}
        
        try:
            start_time = time.time()
            
            logger.info(f"🎯 SIMILARITY CALCULATION PROCESS:")
            logger.info(f"   User embedding shape: {user_embedding.shape}")
            logger.info(f"   Career database size: {len(self.career_embeddings)} careers")
            logger.info(f"   Career embeddings shape: {self.career_embeddings.shape}")
            
            # Calculate cosine similarity with detailed logging
            similarities = cosine_similarity([user_embedding], self.career_embeddings)[0]
            
            # Detailed similarity analysis
            similarity_stats = {
                'mean_similarity': float(np.mean(similarities)),
                'std_similarity': float(np.std(similarities)),
                'min_similarity': float(np.min(similarities)),
                'max_similarity': float(np.max(similarities)),
                'median_similarity': float(np.median(similarities))
            }
            
            logger.info(f"   Similarity Statistics:")
            logger.info(f"     - Mean similarity: {similarity_stats['mean_similarity']:.6f}")
            logger.info(f"     - Std deviation: {similarity_stats['std_similarity']:.6f}")
            logger.info(f"     - Min similarity: {similarity_stats['min_similarity']:.6f}")
            logger.info(f"     - Max similarity: {similarity_stats['max_similarity']:.6f}")
            logger.info(f"     - Median similarity: {similarity_stats['median_similarity']:.6f}")
            
            scores = {}
            career_names = list(CAREER_DATABASE.keys())
            
            # Calculate scores with ranking analysis
            career_similarities = []
            
            for i, career in enumerate(career_names):
                similarity_val = float(similarities[i])
                confidence_val = float(min(max(similarity_val * 100, 0), 100))
                
                scores[career] = {
                    'similarity_score': similarity_val,
                    'confidence': confidence_val,
                    'rank': i + 1  # Will be updated after sorting
                }
                
                career_similarities.append((career, similarity_val))
                
                logger.info(f"     📋 {career}:")
                logger.info(f"       - Raw similarity: {similarity_val:.6f}")
                logger.info(f"       - Confidence %: {confidence_val:.2f}%")
            
            # Sort and update rankings
            career_similarities.sort(key=lambda x: x[1], reverse=True)
            
            logger.info(f"🏆 RANKING RESULTS:")
            for rank, (career, similarity) in enumerate(career_similarities, 1):
                scores[career]['rank'] = rank
                logger.info(f"   {rank}. {career}: {similarity:.6f} ({scores[career]['confidence']:.2f}%)")
            
            calculation_time = time.time() - start_time
            
            # Store detailed calculation log
            calculation_log = {
                'timestamp': datetime.now().isoformat(),
                'process': 'similarity_calculation',
                'similarity_stats': similarity_stats,
                'career_scores': scores,
                'timing': {
                    'calculation_time': calculation_time
                },
                'model_info': {
                    'similarity_metric': 'cosine_similarity',
                    'embedding_dimension': len(user_embedding),
                    'careers_compared': len(career_names)
                }
            }
            self.detailed_logs.append(calculation_log)
            
            logger.info(f"⏱️  Total calculation time: {calculation_time:.4f} seconds")
            
            return scores
            
        except Exception as e:
            logger.error(f"Failed to calculate career scores: {e}")
            return {}
    
    def get_recommendations(self, user_data, top_k=5):
        """Get top career recommendations with comprehensive logging"""
        logger.info(f"🚀 STARTING RECOMMENDATION PROCESS")
        logger.info(f"=" * 60)
        
        start_time = time.time()
        
        # Convert to text for embedding if it's structured data
        if isinstance(user_data, dict):
            user_text = json.dumps(user_data, indent=2)
            logger.info(f"📝 Input: Structured data converted to text ({len(user_text)} chars)")
        else:
            user_text = user_data
            logger.info(f"📝 Input: Raw text ({len(user_text)} chars)")
        
        # Extract features (this will log detailed embedding process)
        user_embedding = self.extract_features_from_text(user_text)
        if user_embedding is None:
            logger.error("❌ Failed to extract features from input")
            return []

        # Calculate scores (this will log detailed similarity process)
        scores = self.calculate_career_scores(user_embedding)
        
        # Sort by similarity score
        sorted_careers = sorted(scores.items(), key=lambda x: x[1]['similarity_score'], reverse=True)
        
        logger.info(f"🎯 GENERATING TOP {top_k} RECOMMENDATIONS:")
        
        recommendations = []
        for rank, (career, score_data) in enumerate(sorted_careers[:top_k], 1):
            career_info = CAREER_DATABASE[career]
            
            # Detailed recommendation analysis
            relevant_exp = self._extract_relevant_experience(user_data, career_info['keywords'])
            
            recommendation = {
                'jobTitle': career,
                'jobDescription': career_info['description'],
                'skills': career_info['required_skills'],
                'similarity_score': float(score_data['similarity_score']),
                'confidence': float(score_data['confidence']),
                'rank': rank,
                'relevantExperience': relevant_exp
            }
            
            logger.info(f"   🏅 #{rank}: {career}")
            logger.info(f"      Similarity: {score_data['similarity_score']:.6f}")
            logger.info(f"      Confidence: {score_data['confidence']:.2f}%")
            logger.info(f"      Relevant experiences: {len(relevant_exp)}")
            
            recommendations.append(recommendation)
        
        total_time = time.time() - start_time
        
        # Calculate recommendation quality metrics
        if len(recommendations) > 1:
            confidences = [r['confidence'] for r in recommendations]
            quality_metrics = {
                'confidence_spread': max(confidences) - min(confidences),
                'average_confidence': sum(confidences) / len(confidences),
                'top_recommendation_confidence': confidences[0],
                'recommendation_diversity': len(set(r['jobTitle'] for r in recommendations))
            }
            
            logger.info(f"📊 RECOMMENDATION QUALITY METRICS:")
            logger.info(f"   Confidence spread: {quality_metrics['confidence_spread']:.2f}%")
            logger.info(f"   Average confidence: {quality_metrics['average_confidence']:.2f}%")
            logger.info(f"   Top recommendation: {quality_metrics['top_recommendation_confidence']:.2f}%")
            logger.info(f"   Diversity score: {quality_metrics['recommendation_diversity']}/{len(recommendations)}")
        
        logger.info(f"⏱️  Total recommendation time: {total_time:.4f} seconds")
        logger.info(f"✅ RECOMMENDATION PROCESS COMPLETED")
        logger.info(f"=" * 60)
        
        return recommendations

    def get_detailed_analysis(self):
        """Get detailed analysis for thesis documentation"""
        return {
            'model_architecture': {
                'name': 'all-MiniLM-L6-v2',
                'type': 'Sentence Transformer',
                'embedding_dimension': 384,
                'max_sequence_length': 256,
                'pooling_mode': 'mean_pooling'
            },
            'processing_logs': self.detailed_logs[-10:],  # Last 10 operations
            'career_database_stats': {
                'total_careers': len(CAREER_DATABASE),
                'embedding_shape': self.career_embeddings.shape if self.career_embeddings is not None else None,
                'keywords_per_career': {career: len(data['keywords']) for career, data in CAREER_DATABASE.items()},
                'skills_per_career': {career: len(data['required_skills']) for career, data in CAREER_DATABASE.items()}
            }
        }
    
# Initialize the recommendation system
rec_system = CareerRecommendationSystem()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def parse_resume_structure(text):
    """Parse resume text into structured format"""
    # First, check if the text might be JSON
    try:
        # Try to parse as JSON first
        json_data = json.loads(text)
        experiences = []
        
        # Extract from experience section if available
        if 'experience' in json_data and isinstance(json_data['experience'], list):
            for exp in json_data['experience']:
                if isinstance(exp, dict):
                    experiences.append({
                        'title': exp.get('title', ''),
                        'description': exp.get('description', '')
                    })
        
        # Extract from projects section if available
        if 'projects' in json_data and isinstance(json_data['projects'], list):
            for proj in json_data['projects']:
                if isinstance(proj, dict):
                    experiences.append({
                        'title': proj.get('title', ''),
                        'description': proj.get('description', '')
                    })
        
        # If experiences were found, return them
        if experiences:
            return experiences
    except json.JSONDecodeError:
        # Not valid JSON, continue with text parsing
        pass
    
    # Improved text-based parsing
    lines = text.split('\n')
    experiences = []
    
    # Define patterns to ignore
    ignore_patterns = [
        r'^\d{1,2}/\d{4}$',  # Dates like 06/2024
        r'^[A-Z]+$',  # All caps section headers like "CERTIFICATES", "LANGUAGES"
        r'^[A-Za-z]+ (Native|Fluent|Intermediate|Advanced)$',  # Language proficiency
        r'^FreeCodeCamp$',  # Certificate providers
        r'^\d{1,2}/\d{1,2}$',  # Page numbers like 1/2
        r'^University$|College$|School$',  # Educational institutions
        r'^[A-Za-z]+ (Basic|Intermediate)$',  # Skill levels
        r'^Java Basic$|^React.js Intermediate$|^MySQL Intermediate$',  # Specific skills with levels
    ]
    
    # Function to check if a line should be ignored
    def should_ignore(line):
        line = line.strip()
        if not line or len(line) < 5:
            return True
        return any(re.match(pattern, line) for pattern in ignore_patterns)
    
    # Function to check if line looks like a project title
    def is_likely_project(line):
        # Project titles often contain these keywords
        project_indicators = [
            'web', 'app', 'website', 'application', 'platform', 'system', 
            'showcase', 'portfolio', 'translator', 'clone', 'product', 'design',
            '3d', 'browser', 'ecommerce', 'responsive'
        ]
        
        # Check if the line contains any project indicators
        line_lower = line.lower()
        return any(indicator in line_lower for indicator in project_indicators) and len(line) > 10
    
    # First pass: collect project titles
    for line in lines:
        line = line.strip()
        if not should_ignore(line) and is_likely_project(line):
            experiences.append({
                'title': line,
                'description': "Project extracted from resume"
            })
    
    # If no projects found, try a more lenient approach
    if not experiences:
        for i, line in enumerate(lines):
            line = line.strip()
            if not should_ignore(line) and len(line) > 10 and not line.startswith(('http', 'www')):
                experiences.append({
                    'title': line,
                    'description': "Project extracted from resume"
                })
                if len(experiences) >= 8:  # Limit to reasonable number of entries
                    break
    
    return experiences

@app.route('/api/parse-manual', methods=['POST'])
def parse_manual():
    try:
        resume_data = request.json
        
        # Get ML-based recommendations - pass the structured data
        recommendations = rec_system.get_recommendations(resume_data)
        
        # Extract experiences directly from structured data instead of parsing JSON text
        experiences = []
        
        # Extract from experience section
        if 'experience' in resume_data:
            for exp in resume_data['experience']:
                experiences.append({
                    'title': exp.get('title', ''),
                    'description': exp.get('description', '')
                })
        
        # Extract from projects section
        if 'projects' in resume_data:
            for proj in resume_data['projects']:
                experiences.append({
                    'title': proj.get('title', ''),
                    'description': proj.get('description', '')
                })
        
        # If no experiences found, create a placeholder
        if not experiences:
            experiences = [{
                'title': 'General Experience',
                'description': 'Experience extracted from provided information'
            }]
        
        # Create summary using simple text processing
        summary = f"Based on the provided information, this candidate shows experience in various areas. The profile has been analyzed using machine learning algorithms to provide the most suitable career recommendations with confidence scores."
        
        result = {
            "summary": summary,
            "Experience": experiences,
            "jobRecommendation": recommendations,
            "ml_metrics": {
                "model_used": "all-MiniLM-L6-v2",
                "similarity_algorithm": "cosine_similarity",
                "total_careers_analyzed": len(CAREER_DATABASE),
                "recommendations_count": len(recommendations)
            }
        }
        
        return jsonify(result)
    
    except Exception as error:
        app.logger.error(f"Error processing manual resume data: {error}")
        return "Failed to process manual resume data", 500

@app.route('/api/parse-pdf', methods=['POST'])
def parse_pdf():
    try:
        # Check if file was uploaded
        if 'pdfFile' not in request.files:
            return "No file uploaded", 400
            
        file = request.files['pdfFile']
        
        # Check if file has a name
        if file.filename == '':
            return "No file selected", 400
            
        if file and allowed_file(file.filename):
            # Create a secure filename and save it
            filename = secure_filename(str(uuid.uuid4()) + "-" + file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            # Extract text from PDF
            with open(filepath, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text()
            
            # Remove the file after extraction
            os.remove(filepath)
            
            # Get ML-based recommendations
            recommendations = rec_system.get_recommendations(text)
            
            # Parse experiences
            experiences = parse_resume_structure(text)
            
            # Create summary
            summary = f"This resume has been analyzed using advanced machine learning algorithms. The candidate's background and skills have been matched against our career database to provide personalized recommendations with quantitative confidence scores."
            
            result = {
                "summary": summary,
                "Experience": experiences,
                "jobRecommendation": recommendations,
                "ml_metrics": {
                    "model_used": "all-MiniLM-L6-v2",
                    "similarity_algorithm": "cosine_similarity",
                    "total_careers_analyzed": len(CAREER_DATABASE),
                    "recommendations_count": len(recommendations)
                }
            }
            
            return jsonify(result)
        
        return "Invalid file type. Please upload a PDF file.", 400
    
    except Exception as error:
        app.logger.error(f"Error processing document: {error}")
        return "Failed to process document", 500

@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get ML metrics for thesis reporting"""
    try:
        # Calculate some example metrics based on the recommendation system
        metrics_data = {
            "model_info": {
                "name": "Sentence Transformer Career Matching",
                "base_model": "all-MiniLM-L6-v2",
                "embedding_dimension": 384,
                "similarity_metric": "cosine_similarity"
            },
            "dataset_info": {
                "total_careers": len(CAREER_DATABASE),
                "career_categories": list(CAREER_DATABASE.keys()),
                "features_extracted": ["text_embeddings", "skill_keywords", "experience_keywords"]
            },
            "performance_metrics": {
                "average_similarity_threshold": 0.7,
                "recommendation_confidence_range": "0-100%",
                "processing_time_ms": "<100ms per request"
            },
            "system_metrics": {
                "total_recommendations_made": len(rec_system.metrics_history) if rec_system.metrics_history else 0,
                "system_uptime": "Active",
                "model_loaded": rec_system.model is not None
            }
        }
        
        return jsonify(metrics_data)
    
    except Exception as error:
        app.logger.error(f"Error getting metrics: {error}")
        return jsonify({"error": "Failed to get metrics"}), 500

@app.route('/api/linkedin-jobs', methods=['GET'])
def linkedin_jobs():
    try:
        title = request.args.get('title', '')
        sanitized_title = re.sub(r'[-:]', '', title)
        
        linkedin_url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={urllib.parse.quote(sanitized_title)}&location=Indonesia&start=0"
        
        # Fetch the HTML
        response = requests.get(linkedin_url)
        html = response.text
        
        # Load into BeautifulSoup to parse
        soup = BeautifulSoup(html, 'html.parser')
        job_listings = []
        
        for element in soup.select('.base-card'):
            job_title = element.select_one('.base-search-card__title').text.strip() if element.select_one('.base-search-card__title') else ''
            organization = element.select_one('.base-search-card__subtitle').text.strip() if element.select_one('.base-search-card__subtitle') else ''
            location = element.select_one('.job-search-card__location').text.strip() if element.select_one('.job-search-card__location') else ''
            
            # Extract job ID from data-entity-urn and build the URL
            entity_urn = element.get('data-entity-urn', '')
            url = ''
            
            if entity_urn:
                # Extract job ID from format "urn:li:jobPosting:4184053904"
                job_id = entity_urn.split(':')[-1]
                url = f"https://www.linkedin.com/jobs/view/{job_id}"
            
            company_logo = ''
            if element.select_one('img.artdeco-entity-image'):
                company_logo = element.select_one('img.artdeco-entity-image').get('data-delayed-url', '')
            
            posted_date = ''
            if element.select_one('time.job-search-card__listdate'):
                posted_date = element.select_one('time.job-search-card__listdate').get('datetime', '')
            
            job_listings.append({
                'title': job_title,
                'organization': organization,
                'locations_derived': [location],
                'url': url,
                'companyLogo': company_logo,
                'date_posted': posted_date
            })
        
        return jsonify(job_listings)
    
    except Exception as error:
        app.logger.error(f"Error fetching from LinkedIn: {error}")
        return jsonify({"error": "Failed to fetch data from LinkedIn"}), 500

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.json
        messages = data.get('messages', [])
        parsed_content = data.get('parsedContent', {})
        
        # Use ML system for chat responses if needed
        if 'ml_metrics' in parsed_content:
            system_message = {
                "role": "system",
                "content": f"""You have access to ML-based career analysis results:

{json.dumps(parsed_content, indent=2)}

The system uses sentence transformers and cosine similarity for career matching. You can discuss the quantitative metrics and confidence scores."""
            }
        else:
            system_message = {
                "role": "system", 
                "content": f"""You have the following analysis results:

{json.dumps(parsed_content, indent=2)}

Use these results to answer queries concisely."""
            }
        
        conversation = [system_message] + messages
        
        lm_payload = {
            "model": "local-model",
            "messages": conversation,
            "temperature": 0.7
        }
        
        lm_response = requests.post(f"{lm_studio_url}/v1/chat/completions", json=lm_payload)
        assistant_message = lm_response.json()['choices'][0]['message']
        assistant_message['content'] = re.sub(r'<think>[\s\S]*?</think>', '', assistant_message['content']).strip()
        
        return jsonify(assistant_message)
    
    except Exception as error:
        app.logger.error(f"Error in /api/chat: {error}")
        return jsonify({"error": "Failed to process chat request"}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)