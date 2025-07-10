# Auto Research Agent

AI-powered research agent built with FastAPI backend and Next.js frontend, featuring intelligent research capabilities and modern web interface.

## Project Structure

```
auto_research_agent/
├── backend/                    # FastAPI backend
│   ├── app/                   # Application code
│   │   ├── agents/           # AI agents
│   │   ├── api/              # API endpoints
│   │   ├── rag/              # RAG implementation
│   │   └── tools/            # Research tools
│   ├── Dockerfile            # Production Docker image
│   ├── Dockerfile.dev        # Development Docker image
│   └── requirements.txt      # Python dependencies
├── frontend/                  # Next.js frontend
│   ├── app/                  # Next.js app directory
│   ├── components/           # React components
│   ├── lib/                  # Utilities
│   ├── Dockerfile            # Production Docker image
│   ├── Dockerfile.dev        # Development Docker image
│   └── package.json          # Node.js dependencies
├── docker-compose.yml        # Production environment
├── docker-compose.dev.yml    # Development environment
└── .env.example             # Environment variables template
```

## Prerequisites

- Docker and Docker Compose
- Node.js 18+ (for local development)
- Python 3.11+ (for local development)

## Quick Start

### Using Docker (Recommended)

#### Development Environment

1. **Clone the repository**
   ```bash
   git clone <repository-url>
   cd auto_research_agent
   ```

2. **Set up environment variables**
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start development environment**
   ```bash
   docker-compose -f docker-compose.dev.yml up
   ```

4. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000
   - API Documentation: http://localhost:8000/docs

#### Production Environment

1. **Build and run production containers**
   ```bash
   docker-compose up -d
   ```

2. **Access the application**
   - Frontend: http://localhost:3000
   - Backend API: http://localhost:8000

### Local Development (Without Docker)

#### Backend Setup

1. **Navigate to backend directory**
   ```bash
   cd backend
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Set environment variables**
   ```bash
   cp ../.env.example .env
   # Edit .env with your API keys
   ```

5. **Run the application**
   ```bash
   uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
   ```

#### Frontend Setup

1. **Navigate to frontend directory**
   ```bash
   cd frontend
   ```

2. **Install dependencies**
   ```bash
   npm install
   ```

3. **Start development server**
   ```bash
   npm run dev
   ```

## Environment Variables

Create a `.env` file in the project root with the following variables:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# Pinecone Configuration (if using vector database)
PINECONE_API_KEY=your_pinecone_api_key_here
PINECONE_ENVIRONMENT=your_pinecone_environment_here

# Other API keys as needed
ANTHROPIC_API_KEY=your_anthropic_api_key_here
PERPLEXITY_API_KEY=your_perplexity_api_key_here
```

## Docker Commands

### Development Environment

```bash
# Start development environment
docker-compose -f docker-compose.dev.yml up

# Start in background
docker-compose -f docker-compose.dev.yml up -d

# Stop services
docker-compose -f docker-compose.dev.yml down

# View logs
docker-compose -f docker-compose.dev.yml logs -f

# Rebuild containers
docker-compose -f docker-compose.dev.yml up --build
```

### Production Environment

```bash
# Start production environment
docker-compose up -d

# Stop services
docker-compose down

# View logs
docker-compose logs -f

# Rebuild containers
docker-compose up --build
```

## API Documentation

The FastAPI backend provides interactive API documentation:

- **Swagger UI**: http://localhost:8000/docs
- **ReDoc**: http://localhost:8000/redoc

## Development Features

### Hot Reload

Both development environments support hot reload:

- **Frontend**: Changes to React components are reflected immediately
- **Backend**: Changes to Python code trigger automatic server restart

### Volume Mounts

Development containers mount source code as volumes:

- Frontend: `./frontend:/app`
- Backend: `./backend:/app`

This enables real-time code editing without rebuilding containers.

## Architecture

### Backend (FastAPI)

- **API Layer**: RESTful API endpoints
- **Agent Layer**: AI-powered research agents
- **RAG System**: Retrieval-Augmented Generation for enhanced responses
- **Tools**: Research and data processing utilities

### Frontend (Next.js)

- **App Router**: Modern Next.js 13+ routing
- **React Components**: Modular UI components
- **Tailwind CSS**: Utility-first styling
- **TypeScript**: Type-safe development

## License

This project is licensed under the MIT License - see the LICENSE file for details.