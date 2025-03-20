# Jobify - Smart Career Recommendation Platform

![Jobify Logo](/public/webIcon.png)

## Overview

Jobify is an intelligent career recommendation platform that analyzes resume data to suggest optimal career paths and job opportunities. Built with React and a Node.js backend, Jobify offers personalized guidance based on your skills, experience, and education.

## Features

- **Dual Input Methods**:

  - Resume upload (PDF analysis)
  - Manual input form for detailed profile creation

- **AI-Powered Analysis**:

  - Generates personalized career path recommendations
  - Maps skills to industry requirements
  - Identifies skill gaps and opportunities

- **Job Matching**:

  - Real-time job listings from LinkedIn
  - Filters jobs by recommended career paths
  - Shows job details including company, location, and posting date

- **Interactive UI**:

  - Responsive design for mobile and desktop
  - Animated transitions with Framer Motion
  - Adaptive layout with collapsible sidebars

- **Career Assistant**:
  - Chat interface for questions about results
  - Contextual guidance for career development

## Getting Started

### Prerequisites

- Node.js (v14+)
- npm or pnpm

### Installation

1. Clone the repository:

   ```bash
   git clone https://github.com/your-username/jobify.git
   cd jobify
   ```

2. Install dependencies for frontend:

   ```bash
   pnpm install
   ```

3. Install dependencies for backend:

   ```bash
   cd backend
   pnpm install
   cd ..
   ```

4. Create a `.env` file in the root directory with your API keys (if needed):
   ```
   VITE_BACKEND_API_URL=http://localhost:5000
   ```

### Running the Application

1. Start the backend server:

   ```bash
   cd backend
   pnpm run dev
   ```

2. In a new terminal, start the frontend:

   ```bash
   pnpm run dev
   ```

3. Open your browser and navigate to `http://localhost:5173`

## Usage

1. Choose between uploading a resume PDF or manually entering your information
2. Submit your information for analysis
3. Review your career path recommendations in the left sidebar
4. Explore matching job listings in the right sidebar
5. Chat with the career assistant for personalized advice

## Deployment

### Building for Production

```bash
pnpm run build
```

The built files will be located in the `dist` directory.

### Deploying to GitHub Pages

```bash
pnpm run deploy
```

## License

[MIT License](LICENSE)
