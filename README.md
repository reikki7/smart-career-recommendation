# Jobify - Smart Career Recommendation Platform

![Jobify Logo](/public/webIcon.png)

## Overview

Jobify is an intelligent career recommendation platform that analyzes resume data to suggest optimal career paths and job opportunities. Built with React.js frontend and a Python Flask backend powered by machine learning models, Jobify offers personalized guidance based on your skills, experience, and education.

## Features

- **Dual Input Methods**:

  - Resume upload (PDF analysis with PyPDF2)
  - Manual input form for detailed profile creation

- **AI-Powered Analysis**:

  - BERT-based semantic embeddings using SentenceTransformers
  - Multiple transformer models: `all-mpnet-base-v2`, `all-MiniLM-L6-v2`, `Fatin757/modernbert-job-role-matcher`
  - Cosine similarity matching for career recommendations
  - LM Studio integration for enhanced PDF content extraction

- **Job Matching**:

  - Real-time job listings from LinkedIn via web scraping
  - Filters jobs by recommended career paths
  - Shows job details including company, location, and posting date

- **Interactive UI**:

  - Responsive design for mobile and desktop
  - Animated transitions with Framer Motion
  - Adaptive layout with collapsible sidebars
  - View similarity scores between career paths and uploaded resume

- **Advanced ML Features**:
  - GPU acceleration support (CUDA)
  - Career database with 100+ job descriptions
  - Skill gap analysis and recommendations

## Getting Started

### Prerequisites

- **Frontend**: Node.js (v14+), npm or pnpm
- **Backend**: Python 3.8+, pip
- **LM Studio**: PDF content extraction
- **Optional**: CUDA-capable GPU for faster ML processing

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/reikki7/smart-career-recommendation.git
   cd jobify
   ```

2. Install frontend dependencies:

   ```bash
   pnpm install
   ```

3. Set up Python backend:

   ```bash
   cd backend

   # Create virtual environment (recommended)
   python -m venv venv

   # Activate virtual environment
   # Windows:
   venv\Scripts\activate
   # macOS/Linux:
   source venv/bin/activate

   # Install Python dependencies
   pip install -r requirements.txt
   ```

4. Create a `.env` file in the root directory:
   ```env
   VITE_BACKEND_API_URL=http://localhost:5000
   LM_STUDIO_URL=http://localhost:1234
   ```

### Running the Application

1. Start the Flask backend server:

   ```bash
   cd backend
   python server.py
   ```

2. In a new terminal, start the React frontend:

   ```bash
   pnpm run dev
   ```

3. Open your browser and navigate to `http://localhost:5173`

## Technology Stack

### Backend

- **Flask** - Web framework
- **SentenceTransformers** - ML embeddings
- **PyTorch** - Deep learning framework
- **scikit-learn** - Machine learning utilities
- **PyPDF2** - PDF text extraction
- **BeautifulSoup4** - Web scraping
- **Flask-CORS** - Cross-origin resource sharing

### Frontend

- **React.js** - UI framework
- **Vite** - Build tool
- **Tailwind CSS** - Styling
- **Framer Motion** - Animations

### Machine Learning

- **BERT Models** - Semantic analysis
- **Cosine Similarity** - Career matching
- **CUDA Support** - GPU acceleration

## Usage

1. Choose between uploading a resume PDF or manually entering your information
2. Submit your information for analysis
3. Review your career path recommendations in the left sidebar
4. Explore matching job listings in the right sidebar
5. View similarity scores to understand recommendation accuracy

## API Endpoints

- `POST /api/parse-pdf` - Upload and analyze PDF resume
- `POST /api/parse-manual` - Submit manual profile data
- `GET /api/linkedin-jobs` - Fetch job listings
- `GET /api/metrics` - Get system metrics

## Deployment

### Building for Production

```bash
# Build frontend
pnpm run build

# The built files will be in the `dist` directory
```

### Backend Deployment

Ensure your production environment has:

- Python 3.8+
- All dependencies from [requirements.txt](backend/requirements.txt)
- Sufficient memory for ML models (recommended: 4GB+ RAM)

### Deploying to GitHub Pages

```bash
pnpm run deploy
```

### Development Scripts

For Windows users, there's a convenient batch script to start all necessary services:

**Note**: Before using this script, you need to set up zrok tunneling:

1. Install zrok from [zrok.io](https://zrok.io)
2. Create an account and configure your reserved shares: `jobifybackend` and `lmstudioserver`
3. Follow the zrok documentation for initial setup and authentication

```bash
# Start backend server and zrok tunnels (Windows only)
start-zrok-servers.bat
```

This script will:

- Start the Flask backend server in a virtual environment
- Launch zrok tunnels for both the backend API and LM Studio server
- Open everything in separate Windows Terminal tabs for easy monitoring

For manual startup or other operating systems, follow the individual steps in the "Running the Application" section above.

## License

[MIT License](LICENSE)

## Acknowledgments

- Hugging Face for transformer models
- LinkedIn for job data
- SentenceTransformers library for embeddings
