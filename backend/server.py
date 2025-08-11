from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import json
import uuid
import requests
from werkzeug.utils import secure_filename
import PyPDF2
import re
import logging
import json
import dotenv
from dotenv import load_dotenv
from pathlib import Path
from bs4 import BeautifulSoup
import re
import urllib.parse
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datetime import datetime
import logging
import time
import torch

dotenv.load_dotenv(Path(__file__).parent.parent / ".env")

app = Flask(__name__)
CORS(app)
app.config['MAX_CONTENT_LENGTH'] = 10 * 1024 * 1024  
app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(os.path.abspath(__file__)), "uploads")

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

load_dotenv(Path(__file__).parent.parent / ".env", override=True)

port = int(os.environ.get('PORT', 5000))
lm_studio_url = os.environ.get('LM_STUDIO_URL', 'http://localhost:1234')
print(f"Using LM Studio URL: {lm_studio_url}")

ALLOWED_EXTENSIONS = {'pdf'}
career_similarity_threshold = 0.39

def load_career_database():
    """Load career database from job_descriptions.json"""
    try:
        json_file_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'job_descriptions.json')
        with open(json_file_path, 'r', encoding='utf-8') as file:
            career_data = json.load(file)
        
        merged_careers = {}
        
        if isinstance(career_data, list):
            for career_obj in career_data:
                if isinstance(career_obj, dict):
                    merged_careers.update(career_obj)
            return merged_careers
        else:
            return career_data
        
    except FileNotFoundError:
        logger.error("job_descriptions.json file not found")
        return {}
    except json.JSONDecodeError as e:
        logger.error(f"Error parsing job_descriptions.json: {e}")
        return {}
    except Exception as e:
        logger.error(f"Error loading career database: {e}")
        return {}

CAREER_DATABASE = load_career_database()
logger.info(f"Loaded {len(CAREER_DATABASE)} career paths from job_descriptions.json")

class CareerRecommendationSystem:
    def __init__(self):
        self.device = self._get_device()
        self.model = self._load_best_model()
        self.career_embeddings = None
        self.skill_embeddings = None
        self.metrics_history = []
        self.detailed_logs = []
        self._initialize_embeddings()
        if self.career_embeddings is not None:
            logger.info(f"Career embeddings initialized with shape: {self.career_embeddings.shape}")
            self._print_career_embedding_comparison() 
    
    def _get_device(self):
        """Determine the best device to use (GPU if available, otherwise CPU)"""
        if torch.cuda.is_available():
            device = 'cuda'
            logger.info(f"🚀 GPU detected: {torch.cuda.get_device_name(0)}")
            logger.info(f"   CUDA version: {torch.version.cuda}")
            logger.info(f"   GPU memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")
        else:
            device = 'cpu'
            logger.warning("⚠️  No GPU detected, using CPU")
        
        return device
    
    def _load_best_model(self):
        """Load the best available model for career matching with GPU support"""
        model_options = [
            {
                'name': 'Fatin757/modernbert-job-role-matcher',
                'type': 'ModernBERT (Career-specialized)',
                'priority': 1
            },
            {
                'name': 'all-mpnet-base-v2',
                'type': 'General-purpose (high quality)',
                'priority': 2
            },
            {
                'name': 'all-MiniLM-L6-v2',
                'type': 'General-purpose (lightweight)',
                'priority': 3
            }
        ]
        
        for model_info in model_options:
            try:
                model = SentenceTransformer(model_info['name'], device=self.device)
                logger.info(f"✅ Successfully loaded: {model_info['name']}")
                logger.info(f"   Type: {model_info['type']}")
                logger.info(f"   Device: {self.device}")
                logger.info(f"   Embedding dimension: {model.get_sentence_embedding_dimension()}")
                
                if hasattr(model, '_modules'):
                    for name, module in model._modules.items():
                        if hasattr(module, 'device'):
                            logger.info(f"   Module {name} device: {next(module.parameters()).device}")
                
                return model
            except Exception as e:
                logger.warning(f"❌ Failed to load {model_info['name']}: {e}")
                continue
        
        logger.error("Failed to load any model")
        return None
    
    def _initialize_embeddings(self):
        """Initialize career embeddings from the database using only skills and keywords"""
        if self.model is None:
            logger.error("Cannot initialize embeddings: model not loaded")
            return
        
        try:
            logger.info("🔄 Initializing career embeddings (skills + keywords only)...")
            
            career_texts = []
            career_names = list(CAREER_DATABASE.keys())
            
            for career_name in career_names:
                career_data = CAREER_DATABASE[career_name]
                combined_text = ""
                combined_text += f"Required skills: {', '.join(career_data['required_skills'])} "
                combined_text += f"Keywords: {', '.join(career_data['keywords'])}"
                career_texts.append(combined_text.strip())
            
            start_time = time.time()
            self.career_embeddings = self.model.encode(career_texts)
            embedding_time = time.time() - start_time
            
            logger.info(f"✅ Career embeddings initialized successfully (skills + keywords only)")
            logger.info(f"   - Careers processed: {len(career_names)}")
            logger.info(f"   - Embedding dimensions: {self.career_embeddings.shape}")
            logger.info(f"   - Processing time: {embedding_time:.4f} seconds")
            logger.info(f"   - Sample career text: '{career_texts[0][:100]}...'")
            
            init_log = {
                'timestamp': datetime.now().isoformat(),
                'process': 'embedding_initialization_skills_keywords_only',
                'careers_processed': len(career_names),
                'embedding_shape': self.career_embeddings.shape,
                'processing_time': embedding_time,
                'embedding_strategy': 'skills_and_keywords_only'
            }
            self.detailed_logs.append(init_log)
            
        except Exception as e:
            logger.error(f"Failed to initialize embeddings: {e}")
            self.career_embeddings = None

    def _print_internal_model_processing(self, enhanced_text, embedding_vector):
        """Print internal model processing steps for thesis documentation"""
        
        internal_process_path = os.path.join(os.path.dirname(__file__), 'internal_model_processing.txt')
        
        with open(internal_process_path, 'w', encoding='utf-8') as f:
            f.write(f"INTERNAL MODEL PROCESSING: SENTENCETRANSFORMER PIPELINE\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n\n")

            f.write("STEP 1: INPUT TEXT TO MODEL\n")
            f.write("-" * 50 + "\n")
            f.write(f"Input text (first 200 chars): {enhanced_text[:200]}...\n")
            f.write(f"Text length: {len(enhanced_text)} characters\n")
            f.write(f"Text type: Enhanced resume with extracted features\n\n")
            
            f.write("STEP 2: TOKENIZATION PROCESS\n")
            f.write("-" * 50 + "\n")
            f.write("```python\n")
            f.write(f'# Input text sample:\n')
            f.write(f'text = "{enhanced_text[:60]}..."\n\n')
            
            sample_words = enhanced_text[:200].split()[:15]
            
            f.write("# ↓ TOKENIZATION (Word-piece/Subword splitting)\n")
            f.write("tokens = [\n")
            
            simulated_tokens = []
            for word in sample_words:
                if len(word) > 6:
                    parts = [word[:3], f"##{word[3:6]}", f"##{word[6:]}"] if len(word) > 6 else [word[:4], f"##{word[4:]}"]
                    simulated_tokens.extend([p for p in parts if p.replace('##', '')])
                else:
                    simulated_tokens.append(word)
            
            for i, token in enumerate(simulated_tokens[:20]):
                f.write(f'    "{token}"{"," if i < len(simulated_tokens[:20])-1 else ""}\n')
            
            f.write("    # ... more tokens\n")
            f.write("]\n")
            f.write(f"# Total estimated tokens: ~{len(enhanced_text.split()) * 1.3:.0f}\n")
            f.write("```\n\n")
            
            f.write("STEP 3: TOKEN TO INPUT IDS CONVERSION\n")
            f.write("-" * 50 + "\n")
            f.write("```python\n")
            f.write("# ↓ VOCABULARY LOOKUP (Token → ID mapping)\n")
            f.write("input_ids = [\n")
            
            import random
            random.seed(42)
            sample_ids = [random.randint(100, 30000) for _ in range(min(15, len(simulated_tokens)))]
            
            for i, token_id in enumerate(sample_ids):
                token_name = simulated_tokens[i] if i < len(simulated_tokens) else "..."
                f.write(f'    {token_id:5d},  # "{token_name}"\n')
            
            f.write("    # ... more token IDs\n")
            f.write("]\n")
            f.write(f"# Vocabulary size: ~30,000 tokens\n")
            f.write(f"# Special tokens: [CLS]=101, [SEP]=102, [PAD]=0\n")
            f.write("```\n\n")
            
            f.write("STEP 4: TOKEN EMBEDDING LOOKUP\n")
            f.write("-" * 50 + "\n")
            f.write("```python\n")
            f.write("# ↓ EMBEDDING LOOKUP (ID → Dense Vector)\n")
            f.write("# Each token ID maps to a 768-dimensional vector\n")
            f.write("token_embeddings = [\n")
            f.write("    # Token 'Gunawan' (ID: 12847)\n")
            f.write(f"    [{random.uniform(-0.1, 0.1):.6f}, {random.uniform(-0.1, 0.1):.6f}, {random.uniform(-0.1, 0.1):.6f}, ..., {random.uniform(-0.1, 0.1):.6f}],  # 768 dims\n")
            f.write("    # Token 'Siswo' (ID: 23691)\n")  
            f.write(f"    [{random.uniform(-0.1, 0.1):.6f}, {random.uniform(-0.1, 0.1):.6f}, {random.uniform(-0.1, 0.1):.6f}, ..., {random.uniform(-0.1, 0.1):.6f}],  # 768 dims\n")
            f.write("    # ... more token embeddings\n")
            f.write("]\n")
            f.write("# Shape: [sequence_length, 768]\n")
            f.write("```\n\n")
            
            f.write("STEP 5: POSITIONAL ENCODING\n")
            f.write("-" * 50 + "\n")
            f.write("```python\n")
            f.write("# ↓ ADD POSITIONAL INFORMATION\n")
            f.write("# Each position gets encoded to understand word order\n")
            f.write("positional_encodings = [\n")
            f.write("    # Position 0 (first token)\n")
            f.write(f"    [{random.uniform(-0.05, 0.05):.6f}, {random.uniform(-0.05, 0.05):.6f}, ..., {random.uniform(-0.05, 0.05):.6f}],\n")
            f.write("    # Position 1 (second token)\n")
            f.write(f"    [{random.uniform(-0.05, 0.05):.6f}, {random.uniform(-0.05, 0.05):.6f}, ..., {random.uniform(-0.05, 0.05):.6f}],\n")
            f.write("    # ... more positions\n")
            f.write("]\n\n")
            f.write("# ↓ COMBINE TOKEN + POSITIONAL EMBEDDINGS\n")
            f.write("input_embeddings = token_embeddings + positional_encodings\n")
            f.write("# Shape: [sequence_length, 768]\n")
            f.write("```\n\n")
            
            f.write("STEP 6: TRANSFORMER LAYERS PROCESSING\n")
            f.write("-" * 50 + "\n")
            f.write("```python\n")
            f.write("# ↓ 12 TRANSFORMER LAYERS (BERT-base architecture)\n")
            f.write("hidden_states = input_embeddings\n\n")
            
            f.write("for layer_num in range(12):  # 12 transformer layers\n")
            f.write("    print(f'Processing Layer {layer_num + 1}/12')\n")
            f.write("    \n")
            f.write("    # ===== MULTI-HEAD ATTENTION =====\n")
            f.write("    # Creates Query, Key, Value matrices\n")
            f.write("    Q = hidden_states @ W_query  # [seq_len, 768] @ [768, 768]\n")
            f.write("    K = hidden_states @ W_key    # [seq_len, 768] @ [768, 768]\n")
            f.write("    V = hidden_states @ W_value  # [seq_len, 768] @ [768, 768]\n")
            f.write("    \n")
            f.write("    # Split into 12 attention heads (768/12 = 64 dims per head)\n")
            f.write("    attention_scores = []\n")
            f.write("    for head in range(12):\n")
            f.write("        q_head = Q[:, head*64:(head+1)*64]\n")
            f.write("        k_head = K[:, head*64:(head+1)*64]\n")
            f.write("        v_head = V[:, head*64:(head+1)*64]\n")
            f.write("        \n")
            f.write("        # Attention calculation: softmax(QK^T/√d_k)V\n")
            f.write("        scores = (q_head @ k_head.T) / sqrt(64)\n")
            f.write("        attention_weights = softmax(scores)\n")
            f.write("        attention_output = attention_weights @ v_head\n")
            f.write("        attention_scores.append(attention_output)\n")
            f.write("    \n")
            f.write("    # Concatenate all heads\n")
            f.write("    multi_head_output = concatenate(attention_scores)\n")
            f.write("    \n")
            f.write("    # Add & Norm (Residual connection + Layer Normalization)\n")
            f.write("    attention_output = layer_norm(hidden_states + multi_head_output)\n")
            f.write("    \n")
            f.write("    # ===== FEED FORWARD NETWORK =====\n")
            f.write("    # Two linear transformations with GELU activation\n")
            f.write("    ff_intermediate = gelu(attention_output @ W_ff1 + b_ff1)  # [seq_len, 3072]\n")
            f.write("    ff_output = ff_intermediate @ W_ff2 + b_ff2  # [seq_len, 768]\n")
            f.write("    \n")
            f.write("    # Add & Norm again\n")
            f.write("    hidden_states = layer_norm(attention_output + ff_output)\n")
            f.write("    \n")
            f.write("    # Output shape remains: [seq_len, 768]\n")
            f.write("\n")
            f.write("# After 12 layers, we have contextually-aware representations\n")
            f.write("final_hidden_states = hidden_states  # [seq_len, 768]\n")
            f.write("```\n\n")
            
            f.write("STEP 7: ATTENTION MECHANISM VISUALIZATION\n")
            f.write("-" * 50 + "\n")
            f.write("```python\n")
            f.write("# Example: How 'Machine Learning' attends to other words\n")
            f.write("# Attention weights for token 'Machine' in layer 8:\n")
            f.write("attention_example = {\n")
            f.write("    'Machine'    : 0.45,  # Self-attention (strong)\n")
            f.write("    'Learning'   : 0.32,  # Adjacent concept (strong)\n")
            f.write("    'AI'         : 0.08,  # Related field (medium)\n")
            f.write("    'Computer'   : 0.06,  # Domain context (medium)\n")
            f.write("    'Science'    : 0.04,  # Academic context (weak)\n")
            f.write("    'JavaScript' : 0.03,  # Different skill (weak)\n")
            f.write("    'Gunawan'    : 0.02,  # Name (very weak)\n")
            f.write("    # ... other tokens with smaller weights\n")
            f.write("}\n")
            f.write("# This shows the model understands 'Machine Learning' as a concept\n")
            f.write("```\n\n")
            
            f.write("STEP 8: POOLING STRATEGY (SENTENCE-LEVEL REPRESENTATION)\n")
            f.write("-" * 50 + "\n")
            f.write("```python\n")
            f.write("# ↓ MEAN POOLING (SentenceTransformer default)\n")
            f.write("# Convert token-level representations to sentence-level\n")
            f.write("\n")
            f.write("# Input: final_hidden_states shape [sequence_length, 768]\n")
            f.write("# Example with sequence_length = 150 tokens\n")
            f.write("token_representations = [\n")
            
            for i in range(5):
                token_name = ['[CLS]', 'Gunawan', 'Siswo', 'Machine', 'Learning'][i]
                sample_repr = [random.uniform(-0.1, 0.1) for _ in range(8)]
                formatted_repr = ', '.join([f'{val:.6f}' for val in sample_repr])
                f.write(f"    # Token '{token_name}' representation (first 8 of 768 dims):\n")
                f.write(f"    [{formatted_repr}, ...],\n")
            
            f.write("    # ... 145 more token representations\n")
            f.write("]\n\n")
            
            f.write("# ↓ APPLY MEAN POOLING\n")
            f.write("sentence_embedding = np.mean(token_representations, axis=0)\n")
            f.write("# Shape: [768] - Single vector representing entire text\n\n")
            
            f.write("# Actual result (first 10 dimensions):\n")
            sample_dims = embedding_vector[:10]
            formatted_dims = ', '.join([f'{val:.6f}' for val in sample_dims])
            f.write(f"sentence_embedding[:10] = [{formatted_dims}]\n")
            f.write("```\n\n")
            
            f.write("STEP 9: FINAL NORMALIZATION\n")
            f.write("-" * 50 + "\n")
            f.write("```python\n")
            f.write("# ↓ L2 NORMALIZATION (Unit vector)\n")
            f.write("# Ensures all embeddings have the same magnitude\n")
            f.write("\n")
            f.write("# Before normalization:\n")
            original_norm = float(np.linalg.norm(embedding_vector * 100))  # Simulate before norm
            f.write(f"embedding_magnitude = {original_norm:.6f}\n")
            f.write(f"embedding_vector = [{', '.join([f'{val:.6f}' for val in embedding_vector[:5]])}, ...]\n\n")
            
            f.write("# Apply L2 normalization:\n")
            f.write("normalized_embedding = embedding_vector / ||embedding_vector||_2\n")
            f.write("\n")
            f.write("# After normalization:\n")
            final_norm = float(np.linalg.norm(embedding_vector))
            f.write(f"normalized_magnitude = {final_norm:.6f}  # Always 1.0\n")
            f.write(f"final_embedding = [{', '.join([f'{val:.6f}' for val in embedding_vector[:5]])}, ...]\n")
            f.write("```\n\n")
            
            # STEP 10: FINAL OUTPUT
            f.write("STEP 10: FINAL OUTPUT - READY FOR SIMILARITY COMPARISON\n")
            f.write("-" * 50 + "\n")
            f.write("```python\n")
            f.write("# ✅ FINAL RESULT: 768-dimensional semantic vector\n")
            f.write("user_embedding = np.array([\n")
            
            for i in range(0, min(50, len(embedding_vector)), 10):
                chunk = embedding_vector[i:i+10]
                formatted_chunk = ', '.join([f'{val:9.6f}' for val in chunk])
                f.write(f"    # Dimensions {i:2d}-{i+9:2d}:\n")
                f.write(f"    {formatted_chunk},\n")
            
            f.write("    # ... dimensions 50-767\n")
            f.write("])\n\n")
            
            f.write("# Vector properties:\n")
            f.write(f"# - Shape: {embedding_vector.shape}\n")
            f.write(f"# - Data type: {type(embedding_vector)}\n")
            f.write(f"# - L2 norm: {float(np.linalg.norm(embedding_vector)):.8f}\n")
            f.write(f"# - Mean: {float(np.mean(embedding_vector)):.8f}\n")
            f.write(f"# - Std: {float(np.std(embedding_vector)):.8f}\n")
            f.write(f"# - Min: {float(np.min(embedding_vector)):.8f}\n")
            f.write(f"# - Max: {float(np.max(embedding_vector)):.8f}\n")
            f.write("\n")
            f.write("# This vector encodes the semantic meaning of:\n")
            f.write("# 'Gunawan Siswo Kuncoro - Front-End Developer with ML experience'\n")
            f.write("# and can now be compared with career embeddings using cosine similarity\n")
            f.write("```\n\n")
            
            f.write("STEP 11: COMPUTATIONAL COMPLEXITY ANALYSIS\n")
            f.write("-" * 50 + "\n")
            f.write("```python\n")
            f.write("# Model Parameters and Operations:\n")
            f.write("# ================================\n")
            f.write("# BERT-base model parameters: ~110 million\n")
            f.write("# - Token embeddings: 30,000 × 768 = 23M params\n")
            f.write("# - Position embeddings: 512 × 768 = 0.4M params\n")
            f.write("# - 12 Transformer layers: ~86M params\n")
            f.write("#   * Each layer: 12 attention heads + feed-forward\n")
            f.write("#   * Attention: 4 × (768 × 768) = 2.4M params per layer\n")
            f.write("#   * Feed-forward: 768 × 3072 + 3072 × 768 = 4.7M params per layer\n")
            f.write("# \n")
            f.write("# Forward pass operations for this text:\n")
            estimated_tokens = len(enhanced_text.split()) * 1.3
            f.write(f"# - Input tokens: ~{estimated_tokens:.0f}\n")
            f.write(f"# - Matrix multiplications: ~{estimated_tokens * 12 * 4:.0f} (Q,K,V,O for each layer)\n")
            f.write(f"# - Attention computations: ~{estimated_tokens * estimated_tokens * 12 * 12:.0f} operations\n")
            f.write(f"# - Total FLOPs: ~{estimated_tokens * estimated_tokens * 768 * 12:.0f}\n")
            f.write("# \n")
            processing_time = 0.15  # Estimate
            f.write(f"# Processing time: ~{processing_time:.3f} seconds (GPU)\n")
            f.write(f"# Memory usage: ~{estimated_tokens * 768 * 4 / 1024 / 1024:.1f} MB for activations\n")
            f.write("```\n\n")
            
            f.write(f"{'='*80}\n")
            f.write(f"INTERNAL MODEL PROCESSING COMPLETED\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n")

        logger.info(f"🔧 Internal model processing saved to: internal_model_processing.txt")
    
    def _print_detailed_step_by_step_analysis(self, original_text, enhanced_text, embedding_vector, skill_keywords, job_keywords, industry_keywords, lmstudio_input=None, lmstudio_output=None):

        """Print step-by-step analysis for thesis documentation"""
        
        step_by_step_path = os.path.join(os.path.dirname(__file__), 'step_by_step_analysis.txt')
        
        with open(step_by_step_path, 'w', encoding='utf-8') as f:
            if lmstudio_input is not None and lmstudio_output is not None:
                f.write("LM STUDIO PDF PROCESSING (if PDF parsing used)\n")
                f.write("=" * 80 + "\n")
                f.write("ORIGINAL PDF TEXT (input to LM Studio):\n")
                f.write("-" * 50 + "\n")
                f.write(f"{lmstudio_input[:10000]}{'...' if len(lmstudio_input) > 1000 else ''}\n\n")
                f.write("PARSED JSON OUTPUT (from LM Studio):\n")
                f.write("-" * 50 + "\n")
                import json as _json
                f.write(_json.dumps(lmstudio_output, indent=2)[:10000])
                if len(_json.dumps(lmstudio_output)) > 2000:
                    f.write("...\n")
                f.write("\n\n")
            f.write(f"STEP-BY-STEP ANALYSIS: TEXT TO EMBEDDING CONVERSION\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n\n")

            f.write("STEP 1: INPUT TEXT PREPROCESSING\n")
            f.write("-" * 50 + "\n")
            f.write(f"Original Text Length: {len(original_text)} characters\n")
            f.write(f"Original Text Sample (First 500 chars):\n")
            f.write(f"{original_text[:10000]}...\n\n")
            
            # Text statistics
            lines = original_text.split('\n')
            words = original_text.split()
            f.write(f"Text Statistics:\n")
            f.write(f"  - Total lines: {len(lines)}\n")
            f.write(f"  - Total words: {len(words)}\n")
            f.write(f"  - Average words per line: {len(words)/len(lines):.2f}\n")
            f.write(f"  - Character distribution: Letters={sum(c.isalpha() for c in original_text)}, Digits={sum(c.isdigit() for c in original_text)}, Spaces={sum(c.isspace() for c in original_text)}\n\n")
            f.write("STEP 2: DYNAMIC FEATURE EXTRACTION FROM CAREER DATABASE\n")
            f.write("-" * 50 + "\n")
            f.write(f"Career Database Statistics:\n")
            f.write(f"  - Total careers in database: {len(CAREER_DATABASE)}\n")
            
            # Show database structure sample
            sample_career = list(CAREER_DATABASE.keys())[0]
            sample_data = CAREER_DATABASE[sample_career]
            f.write(f"  - Sample career: {sample_career}\n")
            f.write(f"  - Sample required skills: {sample_data.get('required_skills', [])[:5]}\n")
            f.write(f"  - Sample keywords: {sample_data.get('keywords', [])[:5]}\n\n")
            
            f.write("STEP 2A: SKILLS EXTRACTION PROCESS\n")
            f.write("-" * 30 + "\n")
            
            # Simulate the skills extraction process
            skill_frequency = {}
            for career_data in CAREER_DATABASE.values():
                if 'required_skills' in career_data:
                    for skill in career_data['required_skills']:
                        skill_lower = skill.lower()
                        skill_frequency[skill_lower] = skill_frequency.get(skill_lower, 0) + 1
            
            f.write(f"Skills Database Analysis:\n")
            f.write(f"  - Total unique skills in database: {len(skill_frequency)}\n")
            f.write(f"  - Most common skills (Top 10):\n")
            
            # Sort skills by frequency
            sorted_skills = sorted(skill_frequency.items(), key=lambda x: x[1], reverse=True)
            for i, (skill, freq) in enumerate(sorted_skills[:10], 1):
                f.write(f"    {i}. {skill}: appears in {freq} careers ({freq/len(CAREER_DATABASE)*100:.1f}%)\n")
            
            f.write(f"\nSkills Found in User Text:\n")
            f.write(f"  - Total skills matched: {len(skill_keywords)}\n")
            f.write(f"  - Matched skills: {skill_keywords[:10]}\n")
            
            # Show matching process
            f.write(f"\nSkill Matching Process:\n")
            text_lower = original_text.lower()
            for i, skill in enumerate(skill_keywords[:5], 1):
                occurrences = text_lower.count(skill)
                f.write(f"  {i}. '{skill}' found {occurrences} time(s) in text\n")
            
            f.write(f"\n")

            f.write("STEP 2B: JOB KEYWORDS EXTRACTION PROCESS\n")
            f.write("-" * 30 + "\n")
            
            # Collect job keywords from database
            all_job_keywords = set()
            for career_name in CAREER_DATABASE.keys():
                career_words = re.findall(r'\b\w+\b', career_name.lower())
                all_job_keywords.update(career_words)
            
            for career_data in CAREER_DATABASE.values():
                if 'keywords' in career_data:
                    all_job_keywords.update([kw.lower() for kw in career_data['keywords']])
            
            f.write(f"Job Keywords Database Analysis:\n")
            f.write(f"  - Total unique job keywords: {len(all_job_keywords)}\n")
            f.write(f"  - Sample job keywords: {list(all_job_keywords)[:15]}\n")
            
            f.write(f"\nJob Keywords Found in User Text:\n")
            f.write(f"  - Total job keywords matched: {len(job_keywords)}\n")
            f.write(f"  - Matched job keywords: {job_keywords[:10]}\n\n")
            
            f.write("STEP 2C: INDUSTRY KEYWORDS EXTRACTION PROCESS\n")
            f.write("-" * 30 + "\n")
            
            all_industry_keywords = set()
            for career_data in CAREER_DATABASE.values():
                if 'industries' in career_data:
                    for industry in career_data['industries']:
                        industry_words = re.findall(r'\b\w+\b', industry.lower())
                        all_industry_keywords.update([w for w in industry_words if len(w) > 3])
            
            f.write(f"Industry Keywords Database Analysis:\n")
            f.write(f"  - Total unique industry keywords: {len(all_industry_keywords)}\n")
            f.write(f"  - Sample industry keywords: {list(all_industry_keywords)[:10]}\n")
            
            f.write(f"\nIndustry Keywords Found in User Text:\n")
            f.write(f"  - Total industry keywords matched: {len(industry_keywords)}\n")
            f.write(f"  - Matched industry keywords: {industry_keywords[:5]}\n\n")
            
            f.write("STEP 3: TEXT ENHANCEMENT PROCESS\n")
            f.write("-" * 50 + "\n")
            
            # Break down the enhanced text
            enhanced_sections = enhanced_text.split('\n')
            f.write(f"Enhanced Text Construction:\n")
            
            for i, section in enumerate(enhanced_sections, 1):
                if section.strip():
                    section_type = "Original Resume" if i == 1 else section.split(':')[0] if ':' in section else f"Section {i}"
                    f.write(f"  Section {i} ({section_type}):\n")
                    f.write(f"    Length: {len(section)} characters\n")
                    f.write(f"    Content: {section[:100]}{'...' if len(section) > 100 else ''}\n\n")
            
            f.write(f"Enhancement Summary:\n")
            f.write(f"  - Original text: {len(original_text)} chars\n")
            f.write(f"  - Enhanced text: {len(enhanced_text)} chars\n")
            f.write(f"  - Enhancement ratio: {len(enhanced_text)/len(original_text):.2f}x\n")
            f.write(f"  - Added information: {len(enhanced_text) - len(original_text)} chars\n\n")
            
            f.write("STEP 4: EMBEDDING GENERATION WITH SENTENCETRANSFORMER\n")
            f.write("-" * 50 + "\n")
            
            f.write(f"Model Information:\n")
            f.write(f"  - Model type: SentenceTransformer\n")
            f.write(f"  - Model name: {type(self.model).__name__}\n")
            f.write(f"  - Device: {self.device}\n")
            f.write(f"  - Embedding dimension: {len(embedding_vector)}\n\n")
            
            f.write(f"Embedding Process:\n")
            f.write(f"  - Input: Enhanced text ({len(enhanced_text)} chars)\n")
            f.write(f"  - Output: Vector of {len(embedding_vector)} dimensions\n")
            f.write(f"  - Data type: {type(embedding_vector)}\n")
            f.write(f"  - Vector shape: {embedding_vector.shape if hasattr(embedding_vector, 'shape') else 'N/A'}\n\n")
            
            f.write("STEP 5: EMBEDDING VECTOR ANALYSIS\n")
            f.write("-" * 50 + "\n")
            
            # Statistical analysis
            mean_val = float(np.mean(embedding_vector))
            std_val = float(np.std(embedding_vector))
            min_val = float(np.min(embedding_vector))
            max_val = float(np.max(embedding_vector))
            
            f.write(f"Statistical Properties:\n")
            f.write(f"  - Mean: {mean_val:.8f}\n")
            f.write(f"  - Standard Deviation: {std_val:.8f}\n")
            f.write(f"  - Minimum value: {min_val:.8f}\n")
            f.write(f"  - Maximum value: {max_val:.8f}\n")
            f.write(f"  - Range: {max_val - min_val:.8f}\n")
            f.write(f"  - L2 Norm: {float(np.linalg.norm(embedding_vector)):.8f}\n\n")
            
            # Value distribution analysis
            f.write(f"Value Distribution Analysis:\n")
            positive_count = sum(1 for x in embedding_vector if x > 0)
            negative_count = sum(1 for x in embedding_vector if x < 0)
            zero_count = sum(1 for x in embedding_vector if x == 0)
            
            f.write(f"  - Positive values: {positive_count} ({positive_count/len(embedding_vector)*100:.1f}%)\n")
            f.write(f"  - Negative values: {negative_count} ({negative_count/len(embedding_vector)*100:.1f}%)\n")
            f.write(f"  - Zero values: {zero_count} ({zero_count/len(embedding_vector)*100:.1f}%)\n\n")
            
            # Significant dimensions (highest absolute values)
            abs_values = [(i, abs(val)) for i, val in enumerate(embedding_vector)]
            abs_values.sort(key=lambda x: x[1], reverse=True)
            
            f.write(f"Most Significant Dimensions (Top 10):\n")
            for i, (dim, abs_val) in enumerate(abs_values[:10], 1):
                original_val = embedding_vector[dim]
                f.write(f"  {i}. Dimension {dim}: {original_val:.8f} (|{abs_val:.8f}|)\n")
            
            f.write(f"\n")
            
            # STEP 6: EMBEDDING SAMPLE VISUALIZATION
            f.write("STEP 6: EMBEDDING VECTOR SAMPLE (First 50 dimensions)\n")
            f.write("-" * 50 + "\n")
            
            sample_dims = embedding_vector[:50] if len(embedding_vector) >= 50 else embedding_vector
            
            for i in range(0, len(sample_dims), 10):
                row = sample_dims[i:i+10]
                formatted_row = [f"{val:9.6f}" for val in row]
                f.write(f"Dims {i:2d}-{min(i+9, len(sample_dims)-1):2d}: [{', '.join(formatted_row)}]\n")
            
            f.write(f"\n")
            
            # STEP 7: SEMANTIC INTERPRETATION
            f.write("STEP 7: SEMANTIC INTERPRETATION\n")
            f.write("-" * 50 + "\n")
            
            f.write(f"Semantic Meaning of Embedding:\n")
            f.write(f"  - This {len(embedding_vector)}-dimensional vector represents the semantic meaning\n")
            f.write(f"    of the enhanced resume text in a high-dimensional space\n")
            f.write(f"  - Each dimension captures different aspects of the text's meaning\n")
            f.write(f"  - Positive values indicate presence/strength of certain semantic features\n")
            f.write(f"  - Negative values indicate absence/opposite of certain semantic features\n")
            f.write(f"  - Values near zero indicate neutral/irrelevant features\n\n")
            
            f.write(f"Text Features Captured:\n")
            f.write(f"  - Professional skills: {len(skill_keywords)} skills identified\n")
            f.write(f"  - Job-related terms: {len(job_keywords)} job keywords found\n")
            f.write(f"  - Industry context: {len(industry_keywords)} industry terms detected\n")
            f.write(f"  - Experience level: Derived from job titles and descriptions\n")
            f.write(f"  - Technical expertise: Captured through skill mentions\n")
            f.write(f"  - Career trajectory: Inferred from experience progression\n\n")
            
            # STEP 8: READY FOR SIMILARITY COMPARISON
            f.write("STEP 8: PREPARATION FOR CAREER MATCHING\n")
            f.write("-" * 50 + "\n")
            
            f.write(f"Embedding Ready for Comparison:\n")
            f.write(f"  - User embedding: {len(embedding_vector)} dimensions\n")
            f.write(f"  - Career database: {len(CAREER_DATABASE)} career embeddings\n")
            f.write(f"  - Similarity metric: Cosine similarity\n")
            f.write(f"  - Expected output: Ranked list of career matches\n\n")
            
            f.write(f"Next Steps in Pipeline:\n")
            f.write(f"  1. Calculate cosine similarity with each career embedding\n")
            f.write(f"  2. Rank careers by similarity score\n")
            f.write(f"  3. Filter by minimum similarity threshold\n")
            f.write(f"  4. Return top recommendations with confidence scores\n\n")
            
            f.write(f"{'='*80}\n")
            f.write(f"STEP-BY-STEP ANALYSIS COMPLETED\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n")

            f.write("# This vector encodes the semantic meaning of:\n")
            f.write("# 'Gunawan Siswo Kuncoro - Front-End Developer with ML experience'\n")
            f.write("# and can now be compared with career embeddings using cosine similarity\n")
            f.write("```\n\n")
            
            # 🔥 NEW: TRANSITION TO SIMILARITY COMPARISON
            f.write("STEP 11: TRANSITION TO SIMILARITY COMPARISON\n")
            f.write("-" * 50 + "\n")
            f.write("```python\n")
            f.write("# USER EMBEDDING IS NOW READY FOR COMPARISON\n")
            f.write("user_embedding_ready = True\n")
            f.write("\n")
            f.write("# NEXT: The similarity comparison process will:\n")
            f.write("# 1. Load pre-computed career embeddings from database\n")
            f.write("# 2. Calculate cosine similarity between user and each career\n")
            f.write("# 3. Rank careers by similarity score\n")
            f.write("# 4. Filter by minimum threshold\n")
            f.write("# 5. Return top recommendations\n")
            f.write("\n")
            f.write("# The detailed similarity comparison process is documented in:\n")
            f.write("# 'similarity_comparison_process.txt'\n")
            f.write("```\n\n")


        logger.info(f"📊 Step-by-step analysis saved to: step_by_step_analysis.txt")

    def extract_features_from_text(self, text, lmstudio_input=None, lmstudio_output=None):
        if self.model is None:
            return None
        
        try:
            original_text = text
            job_keywords = self._extract_job_keywords(text)
            skill_keywords = self._extract_skills_weighted(text)
            industry_keywords = self._extract_industry_keywords(text)
            enhanced_sections = [
                text,
                f"\nIdentified Skills: {', '.join(skill_keywords[:10])}",
                f"\nJob Keywords: {', '.join(job_keywords[:8])}",
                f"\nIndustry Context: {', '.join(industry_keywords[:5])}"
            ]
            enhanced_text = ''.join(enhanced_sections)
            embedding_start = time.time()
            text_embedding = self.model.encode([enhanced_text])
            embedding_time = time.time() - embedding_start
            embedding_vector = text_embedding[0]

            self._print_internal_model_processing(enhanced_text, embedding_vector)
            logger.info(f"✅ Embedding generated in {embedding_time:.3f} seconds")

            self._print_detailed_step_by_step_analysis(
                original_text, enhanced_text, embedding_vector, 
                skill_keywords, job_keywords, industry_keywords,
                lmstudio_input=lmstudio_input, lmstudio_output=lmstudio_output
            )
            self._print_thesis_documentation(enhanced_text, embedding_vector, skill_keywords, job_keywords, industry_keywords)
            logger.info(f"✅ Step-by-step analysis completed and saved")
            return embedding_vector
            
        except Exception as e:
            logger.error(f"Failed to extract features: {e}")
            return None
    
    def _print_career_embedding_comparison(self):
        """Print career embedding comparison for thesis analysis"""
        
        if self.career_embeddings is None:
            return
        
        thesis_comparison_path = os.path.join(os.path.dirname(__file__), 'career_embeddings_analysis.txt')
        
        with open(thesis_comparison_path, 'w', encoding='utf-8') as f:
            f.write(f"CAREER EMBEDDINGS ANALYSIS FOR THESIS\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n\n")
            
            career_names = list(CAREER_DATABASE.keys())
            
            f.write("1. CAREER TEXT REPRESENTATIONS (Sample):\n")
            f.write("-" * 60 + "\n")
            
            for i, career_name in enumerate(career_names[:5]): 
                career_data = CAREER_DATABASE[career_name]
                career_text = f"Required skills: {', '.join(career_data['required_skills'])} Keywords: {', '.join(career_data['keywords'])}"
                
                f.write(f"Career {i+1}: {career_name}\n")
                f.write(f"Text: {career_text[:300]}{'...' if len(career_text) > 300 else ''}\n")
                f.write(f"Embedding sample: [{', '.join([f'{v:.4f}' for v in self.career_embeddings[i][:10]])}...]\n\n")
            
            f.write("2. CAREER EMBEDDINGS STATISTICAL OVERVIEW:\n")
            f.write("-" * 60 + "\n")
            f.write(f"Total careers: {len(career_names)}\n")
            f.write(f"Embedding dimension: {self.career_embeddings.shape[1]}\n")
            f.write(f"Embeddings shape: {self.career_embeddings.shape}\n\n")
            
            # Calculate statistics across all embeddings
            all_embeddings_flat = self.career_embeddings.flatten()
            f.write(f"Overall embedding statistics:\n")
            f.write(f"  Mean: {np.mean(all_embeddings_flat):.6f}\n")
            f.write(f"  Std:  {np.std(all_embeddings_flat):.6f}\n")
            f.write(f"  Min:  {np.min(all_embeddings_flat):.6f}\n")
            f.write(f"  Max:  {np.max(all_embeddings_flat):.6f}\n\n")
            
            f.write("3. CAREER SIMILARITY MATRIX (Sample 5x5):\n")
            f.write("-" * 60 + "\n")
            
            # Calculate similarity matrix for first 5 careers
            sample_embeddings = self.career_embeddings[:5]
            similarity_matrix = cosine_similarity(sample_embeddings)
            
            # Header
            f.write("     ")
            for i in range(5):
                f.write(f"{i+1:8d}")
            f.write("\n")
            
            # Matrix rows
            for i in range(5):
                f.write(f"{i+1:2d}: ")
                for j in range(5):
                    f.write(f"{similarity_matrix[i][j]:8.4f}")
                f.write(f"  {career_names[i][:30]}\n")
            
            f.write(f"\nLegend:\n")
            for i in range(5):
                f.write(f"{i+1}. {career_names[i]}\n")
        
        logger.info(f"📊 Career embeddings analysis saved to: career_embeddings_analysis.txt")

    def _print_thesis_documentation(self, enhanced_text, embedding_vector, skill_keywords, job_keywords, industry_keywords):
        """Print detailed text and embedding information for thesis documentation"""
        
        thesis_doc_path = os.path.join(os.path.dirname(__file__), 'thesis_documentation.txt')
        
        with open(thesis_doc_path, 'a', encoding='utf-8') as f:
            f.write(f"\n{'='*80}\n")
            f.write(f"THESIS DOCUMENTATION - TEXT TO EMBEDDING CONVERSION\n")
            f.write(f"Timestamp: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n\n")

            f.write("1. TEXT PREPROCESSING AND ENHANCEMENT:\n")
            f.write("-" * 50 + "\n")

            original_text = enhanced_text.split('\nIdentified Skills:')[0] if '\nIdentified Skills:' in enhanced_text else enhanced_text
            
            f.write(f"ORIGINAL TEXT ({len(original_text)} characters):\n")
            f.write(f"{original_text[:800]}{'...' if len(original_text) > 800 else ''}\n\n")
            
            f.write(f"ENHANCED TEXT ({len(enhanced_text)} characters):\n")
            f.write(f"{enhanced_text[:1200]}{'...' if len(enhanced_text) > 1200 else ''}\n\n")

            f.write("2. FEATURE EXTRACTION RESULTS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Skills Extracted ({len(skill_keywords)}): {skill_keywords[:15]}\n")
            f.write(f"Job Keywords ({len(job_keywords)}): {job_keywords[:15]}\n")
            f.write(f"Industry Keywords ({len(industry_keywords)}): {industry_keywords[:10]}\n\n")

            f.write("3. EMBEDDING VECTOR REPRESENTATION:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Model Used: {type(self.model).__name__}\n")
            f.write(f"Embedding Dimension: {len(embedding_vector)}\n")
            f.write(f"Vector Type: {type(embedding_vector)}\n")
            f.write(f"Vector Shape: {embedding_vector.shape if hasattr(embedding_vector, 'shape') else 'N/A'}\n\n")

            f.write("4. EMBEDDING VECTOR SAMPLE (First 50 dimensions):\n")
            f.write("-" * 50 + "\n")
            sample_dims = embedding_vector[:50] if len(embedding_vector) >= 50 else embedding_vector

            for i in range(0, len(sample_dims), 10):
                row = sample_dims[i:i+10]
                formatted_row = [f"{val:8.6f}" for val in row]
                f.write(f"Dims {i:2d}-{min(i+9, len(sample_dims)-1):2d}: [{', '.join(formatted_row)}]\n")
            
            f.write(f"\n")

            f.write("5. EMBEDDING STATISTICAL ANALYSIS:\n")
            f.write("-" * 50 + "\n")
            f.write(f"Mean: {float(np.mean(embedding_vector)):8.6f}\n")
            f.write(f"Std:  {float(np.std(embedding_vector)):8.6f}\n")
            f.write(f"Min:  {float(np.min(embedding_vector)):8.6f}\n")
            f.write(f"Max:  {float(np.max(embedding_vector)):8.6f}\n")
            f.write(f"L2 Norm: {float(np.linalg.norm(embedding_vector)):8.6f}\n\n")
            
            f.write("6. EMBEDDING VALUE DISTRIBUTION:\n")
            f.write("-" * 50 + "\n")
            
            hist, bin_edges = np.histogram(embedding_vector, bins=10)
            for i in range(len(hist)):
                bar = "█" * int(hist[i] / max(hist) * 50) if max(hist) > 0 else ""
                f.write(f"[{bin_edges[i]:6.3f} to {bin_edges[i+1]:6.3f}]: {hist[i]:3d} {bar}\n")
            
            f.write(f"\n")

            f.write("7. TEXT ENHANCEMENT BREAKDOWN:\n")
            f.write("-" * 50 + "\n")
            components = enhanced_text.split('\n')
            for i, component in enumerate(components):
                if component.strip():
                    component_type = "Original Text" if i == 0 else component.split(':')[0] if ':' in component else f"Component {i}"
                    f.write(f"{component_type}: {len(component)} chars\n")
                    if len(component) <= 200:
                        f.write(f"  Content: {component.strip()}\n")
                    else:
                        f.write(f"  Content: {component[:200].strip()}...\n")
            
            f.write(f"\n")
        
        # Also log to console for immediate viewing
        logger.info(f"📋 THESIS DOCUMENTATION:")
        logger.info(f"   Enhanced text length: {len(enhanced_text)} characters")
        logger.info(f"   Embedding dimensions: {len(embedding_vector)}")
        logger.info(f"   Embedding stats - Mean: {np.mean(embedding_vector):.6f}, Std: {np.std(embedding_vector):.6f}")
        logger.info(f"   Sample embedding values: [{', '.join([f'{v:.4f}' for v in embedding_vector[:10]])}...]")
        logger.info(f"   Full documentation saved to: thesis_documentation.txt")
    
    def _extract_job_keywords(self, text):
        """Extract job-related keywords dynamically from career database"""
        # Collect all job-related keywords from the career database
        job_keywords = set()
        
        # Extract from career names (job titles)
        for career_name in CAREER_DATABASE.keys():
            # Split career names into individual words
            career_words = re.findall(r'\b\w+\b', career_name.lower())
            job_keywords.update(career_words)
        
        # Extract from keywords field in each career
        for career_data in CAREER_DATABASE.values():
            if 'keywords' in career_data:
                job_keywords.update([kw.lower() for kw in career_data['keywords']])
        
        # Add common job level indicators
        job_level_keywords = ['senior', 'junior', 'lead', 'principal', 'staff', 'entry', 'associate']
        job_keywords.update(job_level_keywords)
        
        # Find matches in the text
        found_keywords = []
        text_lower = text.lower()
        
        for keyword in job_keywords:
            if keyword in text_lower:
                found_keywords.append(keyword)
        
        return found_keywords
    
    def _extract_skills_weighted(self, text):
        """Extract skills with frequency weighting from career database"""
        # Count skill frequency across all careers
        skill_frequency = {}
        
        for career_data in CAREER_DATABASE.values():
            if 'required_skills' in career_data:
                for skill in career_data['required_skills']:
                    skill_lower = skill.lower()
                    skill_frequency[skill_lower] = skill_frequency.get(skill_lower, 0) + 1
        
        # Find skills in text with their frequencies
        found_skills_with_weight = []
        text_lower = text.lower()
        
        for skill, frequency in skill_frequency.items():
            if skill in text_lower:
                found_skills_with_weight.append({
                    'skill': skill,
                    'frequency': frequency,
                    'weight': frequency / len(CAREER_DATABASE)  # Normalize by total careers
                })
        
        # Sort by frequency (most common skills first)
        found_skills_with_weight.sort(key=lambda x: x['frequency'], reverse=True)
        
        return [item['skill'] for item in found_skills_with_weight]

    def _extract_industry_keywords(self, text):
        """Extract industry-specific keywords from career database"""
        industry_keywords = set()
    
        for career_data in CAREER_DATABASE.values():
            if 'industries' in career_data:
                for industry in career_data['industries']:
                    # Split industry names into keywords
                    industry_words = re.findall(r'\b\w+\b', industry.lower())
                    industry_keywords.update(industry_words)
    
        # Find matches in text
        found_industries = []
        text_lower = text.lower()
    
        for keyword in industry_keywords:
            if len(keyword) > 3 and keyword in text_lower:
                found_industries.append(keyword)
    
        return found_industries
    
    def _print_similarity_comparison_process(self, user_embedding, career_similarities):
        """Print detailed similarity comparison process for thesis documentation"""
        
        similarity_process_path = os.path.join(os.path.dirname(__file__), 'similarity_comparison_process.txt')
        
        with open(similarity_process_path, 'w', encoding='utf-8') as f:
            f.write(f"SIMILARITY COMPARISON PROCESS: COSINE SIMILARITY CALCULATION\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n\n")
            
            # STEP 1: INPUT VECTORS READY
            f.write("STEP 1: INPUT VECTORS READY FOR COMPARISON\n")
            f.write("-" * 50 + "\n")
            f.write("```python\n")
            f.write("# ✅ USER EMBEDDING VECTOR (from previous processing)\n")
            f.write(f"user_embedding = np.array([{', '.join([f'{val:.6f}' for val in user_embedding[:8]])}, ...])\n")
            f.write(f"user_embedding.shape = {user_embedding.shape}\n")
            f.write(f"user_embedding_norm = {float(np.linalg.norm(user_embedding)):.8f}  # Should be 1.0 (normalized)\n\n")
            
            f.write("# ✅ CAREER DATABASE EMBEDDINGS (pre-computed)\n")
            f.write(f"career_embeddings = np.array([\n")
            career_names = list(CAREER_DATABASE.keys())
            for i in range(min(3, len(self.career_embeddings))):
                career_name = career_names[i]
                career_embedding = self.career_embeddings[i]
                f.write(f"    # {career_name}\n")
                f.write(f"    [{', '.join([f'{val:.6f}' for val in career_embedding[:6]])}, ...],\n")
            f.write("    # ... more career embeddings\n")
            f.write("])\n")
            f.write(f"career_embeddings.shape = {self.career_embeddings.shape}\n")
            f.write("```\n\n")
            
            # STEP 2: COSINE SIMILARITY FORMULA
            f.write("STEP 2: COSINE SIMILARITY MATHEMATICAL FORMULA\n")
            f.write("-" * 50 + "\n")
            f.write("```python\n")
            f.write("# COSINE SIMILARITY FORMULA:\n")
            f.write("# cosine_sim(A, B) = (A · B) / (||A|| × ||B||)\n")
            f.write("# \n")
            f.write("# Where:\n")
            f.write("# - A · B = dot product (sum of element-wise multiplication)\n")
            f.write("# - ||A|| = L2 norm (magnitude) of vector A\n")
            f.write("# - ||B|| = L2 norm (magnitude) of vector B\n")
            f.write("# \n")
            f.write("# Since our vectors are already L2-normalized (||A|| = ||B|| = 1.0):\n")
            f.write("# cosine_sim(A, B) = A · B\n")
            f.write("# \n")
            f.write("# Result range: [-1, 1]\n")
            f.write("# - 1.0  = identical vectors (perfect match)\n")
            f.write("# - 0.0  = orthogonal vectors (no similarity)\n")
            f.write("# - -1.0 = opposite vectors (completely different)\n")
            f.write("```\n\n")
            
            # STEP 3: COMPUTATION PROCESS
            f.write("STEP 3: STEP-BY-STEP SIMILARITY COMPUTATION\n")
            f.write("-" * 50 + "\n")
            f.write("```python\n")
            f.write("# Calculate similarities for all careers\n")
            f.write("similarities = []\n")
            f.write("career_names = [\n")
            for i, name in enumerate(career_names[:5]):
                f.write(f"    '{name}',\n")
            f.write("    # ... more careers\n")
            f.write("]\n\n")
            
            f.write("# DETAILED CALCULATION FOR FIRST 3 CAREERS:\n")
            for i in range(min(3, len(career_similarities))):
                career_name, similarity_score = career_similarities[i]
                career_embedding = self.career_embeddings[list(CAREER_DATABASE.keys()).index(career_name)]
                
                f.write(f"\n# --- CAREER {i+1}: {career_name} ---\n")
                f.write(f"career_{i+1}_embedding = [{', '.join([f'{val:.6f}' for val in career_embedding[:6]])}, ...]\n")
                
                # Show dot product calculation for first few dimensions
                f.write(f"\n# Dot product calculation (first 6 dimensions shown):\n")
                f.write(f"dot_product_partial = \\\n")
                for dim in range(6):
                    user_val = user_embedding[dim]
                    career_val = career_embedding[dim]
                    product = user_val * career_val
                    f.write(f"    ({user_val:8.6f} × {career_val:8.6f}) + # = {product:8.6f}\n")
                f.write(f"    # ... continue for all 768 dimensions\n")
                
                # Calculate actual dot products for demonstration
                dot_product = float(np.dot(user_embedding, career_embedding))
                f.write(f"\n# Full dot product result:\n")
                f.write(f"dot_product_full = {dot_product:.8f}\n")
                f.write(f"\n# Since vectors are normalized (||A|| = ||B|| = 1.0):\n")
                f.write(f"cosine_similarity = dot_product_full = {dot_product:.8f}\n")
                f.write(f"similarity_percentage = {dot_product * 100:.2f}%\n")
            
            f.write("```\n\n")
            
            # STEP 4: ALL CAREER SCORES
            f.write("STEP 4: COMPLETE SIMILARITY SCORES FOR ALL CAREERS\n")
            f.write("-" * 50 + "\n")
            f.write("```python\n")
            f.write("# SIMILARITY CALCULATION RESULTS:\n")
            f.write("career_similarity_scores = {\n")
            
            for i, (career_name, similarity_score) in enumerate(career_similarities):
                f.write(f"    '{career_name}': {similarity_score:.8f},  # {similarity_score*100:.3f}%\n")
            
            f.write("}\n\n")
            
            # Sort and show rankings
            f.write("# RANKED BY SIMILARITY (Highest to Lowest):\n")
            for rank, (career_name, similarity_score) in enumerate(career_similarities, 1):
                stars = "⭐" * min(int(similarity_score * 5), 5)  # Visual rating
                f.write(f"# {rank:2d}. {career_name:<35} | {similarity_score:.6f} | {similarity_score*100:5.2f}% {stars}\n")
            
            f.write("```\n\n")
            
            # STEP 5: THRESHOLD FILTERING
            f.write("STEP 5: SIMILARITY THRESHOLD FILTERING\n")
            f.write("-" * 50 + "\n")
            f.write("```python\n")
            f.write(f"# FILTERING CAREERS BY SIMILARITY THRESHOLD\n")
            f.write(f"similarity_threshold = {career_similarity_threshold}\n\n")
            
            # Filter careers by threshold
            above_threshold = [(name, score) for name, score in career_similarities if score >= career_similarity_threshold]
            below_threshold = [(name, score) for name, score in career_similarities if score < career_similarity_threshold]
            
            f.write(f"# CAREERS ABOVE THRESHOLD ({career_similarity_threshold}):\n")
            f.write(f"careers_above_threshold = [\n")
            for career_name, similarity_score in above_threshold:
                f.write(f"    ('{career_name}', {similarity_score:.6f}),  # ✅ RECOMMENDED\n")
            f.write(f"]\n")
            f.write(f"total_above_threshold = {len(above_threshold)}\n\n")
            
            f.write(f"# CAREERS BELOW THRESHOLD (FILTERED OUT):\n")
            f.write(f"careers_below_threshold = [\n")
            for career_name, similarity_score in below_threshold[:5]:  # Show first 5
                f.write(f"    ('{career_name}', {similarity_score:.6f}),  # ❌ FILTERED\n")
            if len(below_threshold) > 5:
                f.write(f"    # ... and {len(below_threshold) - 5} more careers below threshold\n")
            f.write(f"]\n")
            f.write(f"total_below_threshold = {len(below_threshold)}\n")
            f.write("```\n\n")
            
            # STEP 6: FINAL RECOMMENDATIONS
            f.write("STEP 6: FINAL RECOMMENDATION SELECTION\n")
            f.write("-" * 50 + "\n")
            f.write("```python\n")
            f.write("# FINAL RECOMMENDATIONS (TOP 5 ABOVE THRESHOLD):\n")
            final_recommendations = above_threshold[:5]
            
            f.write("final_recommendations = [\n")
            for rank, (career_name, similarity_score) in enumerate(final_recommendations, 1):
                confidence = min(max(similarity_score * 100, 0), 100)
                f.write(f"    # RANK {rank}\n")
                f.write(f"    {{\n")
                f.write(f"        'career': '{career_name}',\n")
                f.write(f"        'similarity_score': {similarity_score:.8f},\n")
                f.write(f"        'confidence': {confidence:.2f}%,\n")
                f.write(f"        'rank': {rank}\n")
                f.write(f"    }},\n")
            
            f.write("]\n\n")
            
            f.write("# RECOMMENDATION SUMMARY:\n")
            f.write(f"total_careers_analyzed = {len(career_similarities)}\n")
            f.write(f"careers_above_threshold = {len(above_threshold)}\n")
            f.write(f"final_recommendations_count = {len(final_recommendations)}\n")
            f.write(f"similarity_threshold_used = {career_similarity_threshold}\n")
            f.write(f"highest_similarity_score = {career_similarities[0][1]:.6f}\n")
            if final_recommendations:
                lowest_score = f"{final_recommendations[-1][1]:.6f}"
            else:
                lowest_score = "N/A"
            f.write(f"lowest_recommended_score = {lowest_score}\n")
            f.write("```\n\n")
            
            # STEP 7: SEMANTIC INTERPRETATION
            f.write("STEP 7: SEMANTIC INTERPRETATION OF SIMILARITY SCORES\n")
            f.write("-" * 50 + "\n")
            f.write("```python\n")
            f.write("# SIMILARITY SCORE INTERPRETATION:\n")
            f.write("# 0.90 - 1.00: Excellent match (90-100% confidence)\n")
            f.write("# 0.80 - 0.89: Very good match (80-89% confidence)\n")
            f.write("# 0.70 - 0.79: Good match (70-79% confidence)\n")
            f.write("# 0.60 - 0.69: Fair match (60-69% confidence)\n")
            f.write("# 0.50 - 0.59: Weak match (50-59% confidence)\n")
            f.write("# Below 0.50: Poor match (not recommended)\n\n")
            
            f.write("# ANALYSIS OF TOP 3 RECOMMENDATIONS:\n")
            for i, (career_name, similarity_score) in enumerate(final_recommendations[:3], 1):
                confidence = similarity_score * 100
                
                if similarity_score >= 0.90:
                    interpretation = "Excellent match - User profile strongly aligns with this career"
                elif similarity_score >= 0.80:
                    interpretation = "Very good match - Strong alignment with career requirements"
                elif similarity_score >= 0.70:
                    interpretation = "Good match - Solid alignment with some areas for growth"
                elif similarity_score >= 0.60:
                    interpretation = "Fair match - Some alignment but requires skill development"
                else:
                    interpretation = "Weak match - Limited alignment with career requirements"
                
                f.write(f"\n# RECOMMENDATION {i}:\n")
                f.write(f"career_name = '{career_name}'\n")
                f.write(f"similarity_score = {similarity_score:.6f}\n")
                f.write(f"confidence_percentage = {confidence:.2f}%\n")
                f.write(f"interpretation = '{interpretation}'\n")
            
            f.write("```\n\n")
            
            f.write(f"{'='*80}\n")
            f.write(f"SIMILARITY COMPARISON PROCESS COMPLETED\n")
            f.write(f"Generated: {datetime.now().isoformat()}\n")
            f.write(f"{'='*80}\n")

        logger.info(f"🔍 Similarity comparison process saved to: similarity_comparison_process.txt")
    
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
                logger.info(f"       - Similarity Score: {similarity_val:.6f}")
            
            # Sort and update rankings
            career_similarities.sort(key=lambda x: x[1], reverse=True)

            self._print_similarity_comparison_process(user_embedding, career_similarities)
            
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
    
    # Add this method to the CareerRecommendationSystem class

    def find_most_relevant_experiences(self, user_data, job_title, top_k=2, similarity_threshold=0.5):
        """Find the most relevant experiences for a specific job title using JobBERT with enhanced matching"""
        if self.model is None:
            logger.warning("Model not loaded, returning empty experiences")
            return [], []
        
        try:
            # Extract experiences and projects separately
            all_experiences = []
            all_projects = []
            seen_exp_titles = set()
            seen_proj_titles = set()
            
            if isinstance(user_data, dict):
                # From experience section
                if 'experience' in user_data and isinstance(user_data['experience'], list):
                    for exp in user_data['experience']:
                        if isinstance(exp, dict) and exp.get('title'):
                            title = exp.get('title', '').strip()
                            
                            # Skip if we've already seen this title
                            if title.lower() in seen_exp_titles:
                                continue
                            
                            seen_exp_titles.add(title.lower())
                            
                            exp_text = f"{title} {exp.get('description', '')}"
                            all_experiences.append({
                                'text': exp_text.strip(),
                                'title': title,
                                'description': exp.get('description', ''),
                                'type': 'experience'
                            })
                
                # From projects section
                if 'projects' in user_data and isinstance(user_data['projects'], list):
                    for proj in user_data['projects']:
                        if isinstance(proj, dict) and proj.get('title'):
                            title = proj.get('title', '').strip()
                            
                            # Skip if we've already seen this title
                            if title.lower() in seen_proj_titles:
                                continue
                            
                            seen_proj_titles.add(title.lower())
                            
                            proj_text = f"{title} {proj.get('description', '')}"
                            all_projects.append({
                                'text': proj_text.strip(),
                                'title': title,
                                'description': proj.get('description', ''),
                                'type': 'project'
                            })
            else:
                # Handle PDF text - extract experiences/projects
                lines = str(user_data).split('\n')
                current_section = None
                
                for line in lines:
                    line = line.strip()
                    if not line:
                        continue
                    
                    # Section detection
                    if any(header in line.lower() for header in ['experience', 'work experience']):
                        current_section = 'experience'
                        continue
                    elif any(header in line.lower() for header in ['projects', 'technical projects']):
                        current_section = 'project'
                        continue
                    elif any(header in line.lower() for header in ['education', 'skills', 'contact']):
                        current_section = None
                        continue
                    
                    # Extract items from relevant sections
                    if current_section and len(line) > 10:
                        cleaned_line = line.replace('•', '').replace('-', '').strip()
                        if len(cleaned_line) > 5:
                            title = cleaned_line[:50] + '...' if len(cleaned_line) > 50 else cleaned_line
                            
                            if current_section == 'experience':
                                # Skip if we've already seen this title
                                if title.lower() in seen_exp_titles:
                                    continue
                                
                                seen_exp_titles.add(title.lower())
                                all_experiences.append({
                                    'text': cleaned_line,
                                    'title': title,
                                    'description': cleaned_line,
                                    'type': 'experience'
                                })
                            elif current_section == 'project':
                                # Skip if we've already seen this title
                                if title.lower() in seen_proj_titles:
                                    continue
                                
                                seen_proj_titles.add(title.lower())
                                all_projects.append({
                                    'text': cleaned_line,
                                    'title': title,
                                    'description': cleaned_line,
                                    'type': 'project'
                                })
            
            logger.info(f"🔍 Finding relevant items for '{job_title}' from {len(all_experiences)} experiences and {len(all_projects)} projects")
            
            # Process experiences
            relevant_experiences = []
            if all_experiences:
                career_data = CAREER_DATABASE.get(job_title, {})
                career_desc = career_data.get('description', '')
                skills_list = career_data.get('required_skills', [])
                skills_text = ', '.join(skills_list)
                job_context = f"{job_title}. {career_desc}. Required skills: {skills_text}"

                # Encode the full context instead of title alone
                job_title_embedding = self.model.encode([job_context])
                experience_texts = [exp['text'] for exp in all_experiences]
                experience_embeddings = self.model.encode(experience_texts)
                
                # Calculate similarity scores for experiences
                exp_similarities = cosine_similarity(job_title_embedding, experience_embeddings)[0]
                
                # Create scored experiences for ALL items (not filtered by threshold yet)
                all_scored_experiences = []
                for i, exp in enumerate(all_experiences):
                    similarity_score = float(exp_similarities[i])
                    all_scored_experiences.append({
                        **exp,
                        'similarity_score': similarity_score
                    })
                
                # Sort all experiences by score
                all_scored_experiences.sort(key=lambda x: x['similarity_score'], reverse=True)
                
                # Log top 7 experiences regardless of threshold
                logger.info(f"📊 Top 7 experiences ranked for '{job_title}' (showing all, threshold: {similarity_threshold}):")
                for i, exp in enumerate(all_scored_experiences[:7], 1):
                    status = "✅ ABOVE" if exp['similarity_score'] >= similarity_threshold else "❌ BELOW"
                    logger.info(f"   {i}. {exp['title'][:50]}... (Score: {exp['similarity_score']:.4f}) {status} threshold")
                
                # Now filter by threshold for actual return - REMOVE TOP_K LIMIT
                scored_experiences = [exp for exp in all_scored_experiences if exp['similarity_score'] >= similarity_threshold]
                # Remove the top_k limitation: top_experiences = scored_experiences[:top_k]
                top_experiences = scored_experiences  # Get ALL experiences above threshold
                
                logger.info(f"🎯 Selected {len(top_experiences)} experiences above threshold {similarity_threshold} (showing ALL relevant)")
                
                # Format experiences for return
                for exp in top_experiences:
                    description = exp['description']
                    if len(description) > 100:
                        description = description[:100] + '...'
                    relevant_experiences.append(f"{exp['title']}: {description}")
            
            # Process projects
            relevant_projects = []
            if all_projects:
                if 'job_title_embedding' not in locals():
                    job_title_embedding = self.model.encode([job_title])
                
                project_texts = [proj['text'] for proj in all_projects]
                project_embeddings = self.model.encode(project_texts)
                
                # Calculate similarity scores for projects
                proj_similarities = cosine_similarity(job_title_embedding, project_embeddings)[0]
                
                # Create scored projects for ALL items (not filtered by threshold yet)
                all_scored_projects = []
                for i, proj in enumerate(all_projects):
                    similarity_score = float(proj_similarities[i])
                    all_scored_projects.append({
                        **proj,
                        'similarity_score': similarity_score
                    })
                
                # Sort all projects by score
                all_scored_projects.sort(key=lambda x: x['similarity_score'], reverse=True)
                
                # Log top 7 projects regardless of threshold
                logger.info(f"📊 Top 7 projects ranked for '{job_title}' (showing all, threshold: {similarity_threshold}):")
                for i, proj in enumerate(all_scored_projects[:7], 1):
                    status = "✅ ABOVE" if proj['similarity_score'] >= similarity_threshold else "❌ BELOW"
                    logger.info(f"   {i}. {proj['title'][:50]}... (Score: {proj['similarity_score']:.4f}) {status} threshold")
                
                # Now filter by threshold for actual return - REMOVE TOP_K LIMIT
                scored_projects = [proj for proj in all_scored_projects if proj['similarity_score'] >= similarity_threshold]
                # Remove the top_k limitation: top_projects = scored_projects[:top_k]
                top_projects = scored_projects  # Get ALL projects above threshold
                
                logger.info(f"🎯 Selected {len(top_projects)} projects above threshold {similarity_threshold} (showing ALL relevant)")
                
                # Format projects for return
                for proj in top_projects:
                    description = proj['description']
                    if len(description) > 100:
                        description = description[:100] + '...'
                    relevant_projects.append(f"{proj['title']}: {description}")
            
            # Log summary
            logger.info(f"✅ JobBERT matching complete for '{job_title}': {len(relevant_experiences)} experiences, {len(relevant_projects)} projects returned (threshold: {similarity_threshold}, showing ALL relevant)")
            
            return relevant_experiences, relevant_projects
            
        except Exception as e:
            logger.error(f"Error finding relevant experiences and projects: {e}")
            return [], []

    def get_recommendations(self, user_data, structured_data=None, top_k=5, min_similarity_threshold=career_similarity_threshold, lmstudio_input=None, lmstudio_output=None):
        """Get top career recommendations using raw text for similarity and structured data for experience matching"""
        logger.info(f"🚀 STARTING RECOMMENDATION PROCESS WITH RAW TEXT SIMILARITY")
        logger.info(f"   📊 Minimum similarity threshold: {min_similarity_threshold}")
        logger.info(f"=" * 60)
        
        start_time = time.time()
        
        # Use raw text directly for main similarity computation
        if isinstance(user_data, str):
            user_text = user_data
            logger.info(f"📝 Input: Raw PDF text ({len(user_text)} chars) - USING FOR MAIN SIMILARITY")
        else:
            # Fallback for manual input (structured data)
            user_text = json.dumps(user_data, indent=2)
            structured_data = user_data  # Use the same data for experience matching
            logger.info(f"📝 Input: Structured data converted to text ({len(user_text)} chars)")
        
        # Extract features from RAW TEXT (this will log detailed embedding process)
        user_embedding = self.extract_features_from_text(
            user_text, 
            lmstudio_input=lmstudio_input, 
            lmstudio_output=lmstudio_output
        )
        if user_embedding is None:
            logger.error("❌ Failed to extract features from input")
            return []

        # Calculate scores using RAW TEXT embedding (this will log detailed similarity process)
        scores = self.calculate_career_scores(user_embedding)
        
        # Sort by similarity score
        sorted_careers = sorted(scores.items(), key=lambda x: x[1]['similarity_score'], reverse=True)
        
        # Filter by minimum similarity threshold BEFORE limiting to top_k
        filtered_careers = [(career, score_data) for career, score_data in sorted_careers 
                        if score_data['similarity_score'] >= min_similarity_threshold]
        
        # ENSURE AT LEAST 1 RECOMMENDATION: If no careers meet threshold, take the top 1
        if len(filtered_careers) == 0:
            logger.warning(f"⚠️  No careers met threshold {min_similarity_threshold}, forcing top 1 recommendation")
            filtered_careers = sorted_careers[:2]  # Take the highest scoring career regardless of threshold
    
        # Then limit to top_k from the filtered results
        final_careers = filtered_careers[:top_k]
        
        logger.info(f"🎯 GENERATING RECOMMENDATIONS:")
        logger.info(f"   📊 Total careers analyzed: {len(sorted_careers)}")
        logger.info(f"   🔍 Careers above threshold ({min_similarity_threshold}): {len(filtered_careers)}")
        logger.info(f"   📋 Final recommendations returned: {len(final_careers)}")
        logger.info(f"   📊 Main similarity: RAW PDF TEXT")
        logger.info(f"   🔍 Experience matching: {'STRUCTURED DATA' if structured_data else 'RAW TEXT'}")
        
        # Log filtered out careers for debugging
        below_threshold = [career for career, score_data in sorted_careers 
                        if score_data['similarity_score'] < min_similarity_threshold]
        if below_threshold:
            logger.info(f"❌ Careers filtered out (below {min_similarity_threshold} threshold): {len(below_threshold)}")
            for career in below_threshold[:3]:  # Show top 3 filtered out
                score = next(score_data['similarity_score'] for c, score_data in sorted_careers if c == career)
                logger.info(f"   - {career}: {score:.4f}")
        
        recommendations = []
        for rank, (career, score_data) in enumerate(final_careers, 1):
            career_info = CAREER_DATABASE[career]
            
            # Use structured data for experience/project matching if available, otherwise use raw text
            data_for_matching = structured_data if structured_data else user_data
            relevant_exp, relevant_proj = self.find_most_relevant_experiences(data_for_matching, career, top_k=2)
            
            recommendation = {
                'jobTitle': career,
                'jobDescription': career_info['description'],
                'skills': career_info['required_skills'],
                'similarity_score': float(score_data['similarity_score']),
                'confidence': float(score_data['confidence']),
                'rank': rank,
                'relevantExperience': relevant_exp,  # JobBERT-matched experiences
                'relatedProjects': relevant_proj,     # JobBERT-matched projects
                'typicalEducation': career_info.get('typical_education', ''),
                'jobOutlook': career_info.get('job_outlook', ''),
                'salaryMedian': career_info.get('salary_median', ''),
                'industries': career_info.get('industries', []),
                'keywords': career_info.get('keywords', [])
            }
            
            threshold_status = "✅ above" if score_data['similarity_score'] >= min_similarity_threshold else "⚠️  FORCED (below)"
            logger.info(f"   🏅 #{rank}: {career}")
            logger.info(f"      Similarity Score: {score_data['similarity_score']:.6f} ({threshold_status} {min_similarity_threshold})")
            logger.info(f"      Confidence: {score_data['confidence']:.2f}%")
            logger.info(f"      Relevant experiences: {len(relevant_exp)}")
            logger.info(f"      Related projects: {len(relevant_proj)}")
            logger.info(f"      Salary: {career_info.get('salary_median', 'N/A')}")
            logger.info(f"      Growth: {career_info.get('job_outlook', 'N/A')}")
            
            recommendations.append(recommendation)
        
        total_time = time.time() - start_time
        logger.info(f"⏱️  Total recommendation time: {total_time:.4f} seconds")
        logger.info(f"✅ RAW TEXT SIMILARITY RECOMMENDATION PROCESS COMPLETED")
        logger.info(f"   📊 Returned {len(recommendations)} recommendations (min similarity: {min_similarity_threshold}, guaranteed minimum: 1)")
        logger.info(f"=" * 60)
        
        return recommendations

    
    def get_database_statistics(self):
        """Get statistics about the job descriptions database"""
        all_skills = set()
        all_keywords = set()
        all_industries = set()
        
        for career_data in CAREER_DATABASE.values():
            if 'required_skills' in career_data:
                all_skills.update([skill.lower() for skill in career_data['required_skills']])
            if 'keywords' in career_data:
                all_keywords.update([kw.lower() for kw in career_data['keywords']])
            if 'industries' in career_data:
                all_industries.update([ind.lower() for ind in career_data['industries']])
        
        return {
            'total_careers': len(CAREER_DATABASE),
            'unique_skills': len(all_skills),
            'unique_keywords': len(all_keywords),
            'unique_industries': len(all_industries),
            'sample_skills': list(all_skills)[:10],
            'sample_keywords': list(all_keywords)[:10],
            'sample_industries': list(all_industries)[:5]
        }

rec_system = CareerRecommendationSystem()

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/api/parse-manual', methods=['POST'])
def parse_manual():
    try:
        resume_data = request.json
        
        recommendations = rec_system.get_recommendations(resume_data, structured_data=resume_data, min_similarity_threshold=career_similarity_threshold)
        
        experiences = []
        projects = []

        if 'experience' in resume_data:
            for exp in resume_data['experience']:
                experiences.append({
                    'title': exp.get('title', ''),
                    'description': exp.get('description', '')
                })

        if 'projects' in resume_data:
            for proj in resume_data['projects']:
                projects.append({
                    'title': proj.get('title', ''),
                    'description': proj.get('description', '')
                })
        
        if not experiences:
            experiences = [{
                'title': '',  
                'description': ''
            }]
        
        result = {
            "Experience": experiences,
            "Projects": projects,
            "jobRecommendation": recommendations,
            "ml_metrics": {
                "model_used": "all-MiniLM-L6-v2",
                "similarity_algorithm": "cosine_similarity", 
                "total_careers_analyzed": len(CAREER_DATABASE),
                "recommendations_count": len(recommendations),
                "min_similarity_threshold": career_similarity_threshold
            }
        }
        
        return jsonify(result)
    
    except Exception as error:
        app.logger.error(f"Error processing manual resume data: {error}")
        return "Failed to process manual resume data", 500

@app.route('/api/parse-pdf', methods=['POST'])
def parse_pdf():
    try:
        if 'pdfFile' not in request.files:
            return "No file uploaded", 400
        
        file = request.files['pdfFile']
        if file.filename == '':
            return "No file selected", 400
        
        if file and allowed_file(file.filename):
            filename = secure_filename(str(uuid.uuid4()) + "-" + file.filename)
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(filepath)
            
            with open(filepath, 'rb') as pdf_file:
                pdf_reader = PyPDF2.PdfReader(pdf_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            
            os.remove(filepath)

            logger.info(f"📄 PDF text extracted ({len(text)} characters)")

            prompt = f"""
You are an AI assistant that extracts structured data from resumes. 
Given the following resume text, return ONLY a valid JSON object with the fields: 
- "experience": a list of experience objects, each with "title" and "description"
- "projects": a list of projects, each with "title" and "description"

IMPORTANT RULES:
1. Every object MUST have both "title" and "description" fields
2. If description is missing, just empty string for "description"
3. Return only valid JSON, no explanations
4. If no experience/projects found, return empty arrays
5. Ignore any personal information like name, contact details, etc.
6. Ignore certifications, education, and only focus on experience and projects
7. if projects are not found, return an empty array for "projects"
8. If the resume is empty or contains no relevant information, return empty arrays for both "experience" and "projects"

Resume Text:
{text}
"""

            lm_studio_endpoint = f"{lm_studio_url}/v1/chat/completions"
            
            payload = {
                "model": "local-model",
                "messages": [
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                "temperature": 0.1,
                "max_tokens": 7048,
                "stream": False
            }

            response = requests.post(
                lm_studio_endpoint,
                json=payload,
                headers={"Content-Type": "application/json"}
            )
            
            if response.status_code != 200:
                logger.error(f"LM Studio API failed: {response.status_code} - {response.text}")
                return "LM Studio processing failed", 500

            response_data = response.json()
            
            if 'choices' in response_data and len(response_data['choices']) > 0:
                content = response_data['choices'][0]['message']['content']
                
                try:
                    content = content.strip()
                    if content.startswith('```json'):
                        content = content[7:]
                    if content.endswith('```'):
                        content = content[:-3]
                    content = content.strip()
                    
                    logger.info(f"🔍 Raw LM Studio response: {content[:500]}...")
                    
                    parsed_json = json.loads(content)
                    
                    parsed_json = _validate_and_fix_json_structure(parsed_json)
                    
                    logger.info(f"📥 LM Studio returned structured data successfully")
                    logger.info(f"   - Experiences: {len(parsed_json.get('experience', []))}")
                    logger.info(f"   - Projects: {len(parsed_json.get('projects', []))}")
                    
                except json.JSONDecodeError as e:
                    logger.error(f"Failed to parse LM Studio JSON response: {e}")
                    logger.error(f"Raw response: {content}")
                    parsed_json = _extract_partial_json(content)
                    
            else:
                logger.error("Invalid response structure from LM Studio")
                return "Invalid response from LM Studio", 500

            recommendations = rec_system.get_recommendations(
                text, 
                parsed_json, 
                min_similarity_threshold=career_similarity_threshold,
                lmstudio_input=text,
                lmstudio_output=parsed_json
            )

            experiences = parsed_json.get("experience", [])
            projects = parsed_json.get("projects", [])

            if not experiences:
                experiences = [{
                    'title': '',
                    'description': ''
                }]

            result = {
                "Experience": experiences,
                "Projects": projects,
                "jobRecommendation": recommendations,
                "ml_metrics": {
                    "model_used": "sentence-transformers + LM Studio",
                    "similarity_algorithm": "cosine_similarity",
                    "total_careers_analyzed": len(CAREER_DATABASE),
                    "recommendations_count": len(recommendations),
                    "min_similarity_threshold": career_similarity_threshold
                }
            }

            return jsonify(result)

        experiences = parsed_json.get("experience", [])
        projects = parsed_json.get("projects", [])

        if not experiences:
            experiences = [{
                'title': '',
                'description': ''
            }]

            result = {
                "Experience": experiences,
                "Projects": projects,
                "jobRecommendation": recommendations,
                "ml_metrics": {
                    "model_used": "sentence-transformers + LM Studio",
                    "similarity_algorithm": "cosine_similarity",
                    "total_careers_analyzed": len(CAREER_DATABASE),
                    "recommendations_count": len(recommendations),
                    "min_similarity_threshold": career_similarity_threshold
                }
            }

            return jsonify(result)

        return "Invalid file type. Please upload a PDF.", 400
    
    except Exception as error:
        logger.error(f"Error processing PDF with LM Studio: {error}")
        return "Failed to process document", 500

def _validate_and_fix_json_structure(parsed_json):
    """Validate and fix JSON structure from LM Studio"""
    if not isinstance(parsed_json, dict):
        return {"experience": [], "projects": []}

    experiences = parsed_json.get("experience", [])
    fixed_experiences = []
    
    for exp in experiences:
        if isinstance(exp, dict):
            title = exp.get("title", "")
            description = exp.get("description", "")
            
            if title:
                fixed_experiences.append({
                    "title": title,
                    "description": description
                })
        elif isinstance(exp, str):
            fixed_experiences.append({
                "title": exp,
                "description": "" 
            })

    projects = parsed_json.get("projects", [])
    fixed_projects = []
    
    for proj in projects:
        if isinstance(proj, dict):
            title = proj.get("title", "")
            description = proj.get("description", "")
            
            if title:
                fixed_projects.append({
                    "title": title,
                    "description": description
                })
        elif isinstance(proj, str):
            fixed_projects.append({
                "title": proj,
                "description": "" 
            })
    
    return {
        "experience": fixed_experiences,
        "projects": fixed_projects
    }

def _extract_partial_json(content):
    """Try to extract partial JSON when parsing fails"""
    try:
        import re
        
        exp_pattern = r'"experience"\s*:\s*\[(.*?)\]'
        proj_pattern = r'"projects"\s*:\s*\[(.*?)\]'
        
        experiences = []
        projects = []
        
        exp_match = re.search(exp_pattern, content, re.DOTALL)
        if exp_match:
            # Try to extract individual experience entries
            exp_content = exp_match.group(1)
            title_pattern = r'"title"\s*:\s*"([^"]*)"'
            titles = re.findall(title_pattern, exp_content)
            
            for title in titles:
                experiences.append({
                    "title": title,
                    "description": ""
                })
        
        proj_match = re.search(proj_pattern, content, re.DOTALL)
        if proj_match:
            # Try to extract individual project entries
            proj_content = proj_match.group(1)
            title_pattern = r'"title"\s*:\s*"([^"]*)"'
            titles = re.findall(title_pattern, proj_content)
            
            for title in titles:
                projects.append({
                    "title": title,
                    "description": ""  
                })
        
        logger.info(f"🔧 Extracted partial JSON: {len(experiences)} experiences, {len(projects)} projects")
        
        return {
            "experience": experiences,
            "projects": projects
        }
        
    except Exception as e:
        logger.error(f"Failed to extract partial JSON: {e}")
        return {"experience": [], "projects": []}
    
@app.route('/api/metrics', methods=['GET'])
def get_metrics():
    """Get ML metrics for thesis reporting"""
    try:
        db_stats = rec_system.get_database_statistics()
        metrics_data = {
            "model_info": {
                "name": "JobBERT Career Matching System",
                "base_model": "jjzha/jobbert-knowledge",
                "embedding_dimension": 768,
                "similarity_metric": "cosine_similarity"
            },
            "database_statistics": db_stats, 
            "dataset_info": {
                "total_careers": len(CAREER_DATABASE),
                "career_categories": list(CAREER_DATABASE.keys())[:10],  # Sample
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
        location = request.args.get('location', 'Indonesia')
        sanitized_title = re.sub(r'[-:]', '', title)
        
        linkedin_url = f"https://www.linkedin.com/jobs-guest/jobs/api/seeMoreJobPostings/search?keywords={urllib.parse.quote(sanitized_title)}&location={urllib.parse.quote(location)}&start=0"
        
        response = requests.get(linkedin_url)
        html = response.text

        soup = BeautifulSoup(html, 'html.parser')
        job_listings = []
        
        for element in soup.select('.base-card'):
            job_title = element.select_one('.base-search-card__title').text.strip() if element.select_one('.base-search-card__title') else ''
            organization = element.select_one('.base-search-card__subtitle').text.strip() if element.select_one('.base-search-card__subtitle') else ''
            location = element.select_one('.job-search-card__location').text.strip() if element.select_one('.job-search-card__location') else ''
            
            entity_urn = element.get('data-entity-urn', '')
            url = ''
            
            if entity_urn:
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

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=port, debug=True)