# Overview

지능형 연구 보조 에이전트는 연구자, 학생, 그리고 지식 탐구자들을 위한 AI 기반 연구 도구입니다. 사용자가 특정 주제에 대해 질문하면, 에이전트가 자율적으로 사고하고, 여러 도구를 활용하여 종합적인 연구 결과를 제공합니다. 

**해결하는 문제:**
- 방대한 정보 속에서 신뢰할 수 있는 자료를 찾는 어려움
- 여러 출처의 정보를 종합하고 분석하는 시간적 비용
- 연구 과정의 체계적 관리 부재

**타겟 사용자:**
- 대학원생 및 연구원
- 기술 문서를 자주 참조하는 개발자
- 특정 주제에 대한 깊이 있는 학습을 원하는 학습자

**핵심 가치:**
- 자율적 사고와 추론 능력을 갖춘 AI 에이전트
- 다양한 문서 형식 지원 및 효율적인 정보 검색
- 연구 과정의 투명성과 추적 가능성

# Core Features

## 1. 자율적 연구 에이전트 (Autonomous Research Agent)
- **기능**: 사용자의 질문을 분석하고, 스스로 연구 계획을 수립하여 실행
- **중요성**: 단순 질의응답을 넘어 깊이 있는 연구 수행 가능
- **작동 방식**: 
  - ReAct (Reasoning + Acting) 패턴 구현
  - Chain-of-Thought 추론 과정 시각화
  - 자체 평가 및 개선 메커니즘

## 2. 멀티모달 RAG 시스템 (Multimodal RAG System)
- **기능**: PDF, 웹페이지, 코드, 이미지 등 다양한 형식의 문서 처리
- **중요성**: 실제 연구 환경의 다양한 자료 형식 지원
- **작동 방식**:
  - 문서 타입별 최적화된 chunking 전략
  - Hybrid search (벡터 + 키워드) 구현
  - 컨텍스트 인식 retrieval

## 3. 도구 통합 및 확장 시스템 (Tool Integration System)
- **기능**: 웹 검색, 코드 실행, 데이터 분석 등 다양한 도구 활용
- **중요성**: 에이전트의 능력 확장 및 실제 작업 수행
- **작동 방식**:
  - Function calling 인터페이스
  - 도구 선택 최적화
  - 결과 통합 및 검증

## 4. 연구 과정 추적 시스템 (Research Tracking System)
- **기능**: 에이전트의 사고 과정과 행동을 모두 기록하고 시각화
- **중요성**: 연구 과정의 투명성 확보 및 학습 기회 제공
- **작동 방식**:
  - 추론 단계별 로깅
  - 의사결정 트리 시각화
  - 성능 메트릭 수집

# User Experience

## User Personas

### 1. 김연구 (대학원생)
- 컴퓨터공학 석사과정
- 최신 논문과 기술 문서를 자주 참조
- 연구 주제에 대한 체계적인 정리 필요

### 2. 이개발 (시니어 개발자)
- 새로운 기술 스택 도입 검토 중
- 기술 문서와 실제 구현 예제 필요
- 빠른 프로토타이핑과 검증 중요

### 3. 박학습 (자기주도 학습자)
- AI/ML 분야 깊이 있는 학습 희망
- 체계적인 학습 경로 필요
- 실습과 이론의 균형 추구

## Key User Flows

### 1. 연구 주제 탐색
1. 사용자가 연구 질문 입력
2. 에이전트가 질문 분석 및 하위 주제 도출
3. 연구 계획 제시 및 사용자 확인
4. 단계별 연구 수행
5. 종합 보고서 생성

### 2. 문서 기반 학습
1. 사용자가 문서 업로드 (PDF, 웹링크 등)
2. 문서 자동 분석 및 인덱싱
3. 질의응답 세션 시작
4. 관련 내용 검색 및 설명
5. 추가 자료 추천

## UI/UX Considerations

- **대화형 인터페이스**: 자연스러운 대화를 통한 연구 진행
- **시각적 피드백**: 에이전트의 사고 과정 실시간 표시
- **진행 상황 표시**: 연구 단계별 진행률 시각화
- **결과 정리**: 구조화된 보고서 형식으로 결과 제공

# Technical Architecture

## System Components

### 1. Agent Core
- **Framework**: LangChain
  - Agent executor
  - Memory management
  - Tool orchestration
- **LLM Integration**: OpenAI (GPT-4, GPT-3.5)
- **Reasoning Engine**: ReAct agent (LangChain)
- **Planning Module**: LangChain task decomposition

### 2. RAG Pipeline
- **Framework**: LangChain
  - Document loaders
  - Text splitters
  - Vector store integration
- **Document Processor**: 
  - PDF: PyPDF/LangChain loaders
  - Web: BeautifulSoup + LangChain
  - Markdown/Code: Native loaders
- **Chunking**: LangChain text splitters
  - RecursiveCharacterTextSplitter
  - Document-aware splitting
- **Embedding**: OpenAI via LangChain
- **Vector Store**: Pinecone via LangChain

### 3. Tool Framework
- **LangChain Tools**:
  - Web search tool
  - Python REPL
  - Custom tools
- **Tool Registry**: LangChain tool management
- **Execution**: LangChain agent executor
- **Result Parser**: Structured output parsing

### 4. Frontend Application
- **Framework**: React with TypeScript
- **AI Interaction**: Vercel AI SDK
  - Streaming chat UI
  - Tool call visualization
- **UI Components**: shadcn/ui
- **Icons**: Lucide React
- **Styling**: Tailwind CSS
- **State Management**: Zustand
- **Deployment**: Vercel

## Data Models

```python
# Core Entities
class ResearchSession:
    id: str
    user_id: str
    topic: str
    status: str
    created_at: datetime
    messages: List[Message]
    artifacts: List[Artifact]

class Message:
    id: str
    role: str  # user, assistant, system
    content: str
    reasoning: Optional[ReasoningChain]
    tools_used: List[ToolInvocation]
    timestamp: datetime

class ReasoningChain:
    steps: List[ThoughtStep]
    conclusion: str
    confidence: float

class Document:
    id: str
    source: str
    type: str  # pdf, web, code, etc
    chunks: List[Chunk]
    metadata: Dict
    embeddings: Optional[np.array]

class Chunk:
    id: str
    document_id: str
    content: str
    embedding: np.array
    metadata: Dict  # page, section, etc
```

## APIs and Integrations

### Backend API (FastAPI)
- `/api/chat`: Vercel AI SDK 호환 스트리밍
- `/api/documents`: 문서 업로드 및 관리
- `/api/tools`: LangChain tool 실행
- `/api/sessions`: 연구 세션 관리

### LangChain Integrations
- **LLM**: OpenAI (GPT-4o, GPT-4o-mini)
- **Embeddings**: OpenAI text-embedding-3
- **Vector Store**: Pinecone
- **Tools**: Serper API, Python REPL

### Frontend Integration
- **Vercel AI SDK**: 
  - useChat hook
  - Tool call UI components
  - Streaming support
- **API Routes**: Next.js API routes

## Infrastructure Requirements

- **Compute**: 
  - CPU 서버 (API 서버 및 일반 처리)
  - 메모리: 최소 8GB RAM (벡터 연산용)
- **Storage**:
  - Pinecone (벡터 데이터베이스)
  - SQLite (메타데이터 및 세션 정보)
  - Local file storage (문서)
- **Services**:
  - Redis (캐싱 및 세션 관리)
  - RabbitMQ (비동기 작업 큐)
  - Elasticsearch (풀텍스트 검색)

# Development Roadmap

## Phase 1: Foundation (MVP)

### Core Agent Implementation
- LangChain ReAct agent 설정
- OpenAI GPT-4o-mini 기본 설정
- 단순 대화 인터페이스 (Vercel AI SDK)
- 기본 추론 과정 로깅

### Simple RAG System
- PDF 문서 처리 (LangChain PDF loader)
- 기본 텍스트 chunking (RecursiveCharacterTextSplitter)
- OpenAI embeddings (LangChain)
- Pinecone vector store 설정
- LangChain retriever chain

### Minimal UI
- Next.js + React 프로젝트 설정
- Vercel AI SDK useChat 구현
- shadcn/ui 컴포넌트 (Button, Input, Card)
- 파일 업로드 (react-dropzone)
- 기본 마크다운 렌더링

## Phase 2: Enhanced Intelligence

### Advanced Agent Capabilities
- LangChain agent types 실험
  - OpenAI Functions Agent
  - ReAct Agent
- GPT-4o 전환 옵션
- Memory: ConversationSummaryMemory
- Multi-step planning with LangChain

### Improved RAG Pipeline
- LangChain semantic chunking
- Hybrid search (Pinecone + BM25)
- LangChain reranking chains
- Parent document retriever

### Tool Integration
- LangChain tools:
  - SerperAPI (web search)
  - PythonREPL
  - Custom tools
- Tool selection with GPT-4o

## Phase 3: Production Features

### Scalable Architecture
- Model switching (GPT-4o-mini ↔ GPT-4o)
- Pinecone 인덱스 최적화
- Background jobs (Celery)
- Response caching (Redis)

### Advanced Features
- 멀티모달 지원 (GPT-4o Vision)
- Vercel AI SDK RSC (React Server Components)
- 고급 UI 컴포넌트 (shadcn/ui charts, data tables)
- LangChain Expression Language (LCEL)

### Analytics & Monitoring
- 사용 패턴 분석
- 성능 메트릭 대시보드
- A/B 테스팅 프레임워크
- 비용 최적화 도구

# Logical Dependency Chain

## 1. Foundation Layer (Week 1-2)
```
1. Project Setup
   └── 2. Basic LLM Integration
       └── 3. Simple Chat Interface
           └── 4. Message History Management
```

## 2. Core Agent Development (Week 3-4)
```
5. ReAct Pattern Implementation
   ├── 6. Reasoning Chain Logging
   └── 7. Action Execution Framework
       └── 8. Basic Tool Interface
```

## 3. RAG System Building (Week 5-6)
```
9. Document Processing Pipeline
   ├── 10. Text Extraction (PDF)
   ├── 11. Basic Chunking Strategy
   └── 12. Embedding Generation
       └── 13. Vector Store Integration
           └── 14. Similarity Search
```

## 4. Integration Phase (Week 7-8)
```
15. Agent + RAG Connection
    ├── 16. Context Injection
    └── 17. Source Attribution
        └── 18. Response Generation
            └── 19. UI Polish
```

## 5. Enhancement Phase (Week 9-10)
```
20. Advanced Chunking
    └── 21. Hybrid Search
        └── 22. Reranking
            └── 23. Performance Optimization
```

# Risks and Mitigations

## Technical Risks

### 1. LLM API 비용 관리
- **위험**: 과도한 API 호출로 인한 비용 증가
- **완화**: 
  - 응답 캐싱 구현
  - 로컬 LLM 옵션 제공
  - 사용량 제한 및 모니터링

### 2. RAG 정확도 문제
- **위험**: 잘못된 정보 검색 및 hallucination
- **완화**:
  - 멀티 스테이지 검증
  - 소스 명시 의무화
  - Confidence score 표시

### 3. 확장성 제한
- **위험**: 사용자/문서 증가 시 성능 저하
- **완화**:
  - 초기부터 비동기 아키텍처 설계
  - 수평적 확장 가능한 구조
  - 효율적인 인덱싱 전략

## MVP Scoping Risks

### 1. 과도한 기능 포함
- **위험**: MVP가 너무 복잡해짐
- **완화**:
  - 핵심 기능만 구현 (Agent + Basic RAG)
  - 사용자 피드백 기반 우선순위 조정

### 2. 기술적 난이도
- **위험**: ReAct 패턴 구현의 복잡성
- **완화**:
  - 단계별 구현 (simple → complex)
  - 오픈소스 참조 구현 활용

## Resource Constraints

### 1. 개발 시간
- **위험**: 예상보다 긴 개발 기간
- **완화**:
  - 명확한 마일스톤 설정
  - 기존 라이브러리 최대 활용

### 2. API 비용
- **위험**: LLM API 및 임베딩 API 비용
- **완화**:
  - 초기에는 무료 티어 활용
  - 효율적인 캐싱 전략
  - 점진적 스케일업

# Appendix

## Research Findings

### Agent Patterns
- **ReAct**: Reasoning과 Acting을 결합한 가장 효과적인 패턴
- **Chain-of-Thought**: 복잡한 문제 해결에 필수
- **Self-Reflection**: 에이전트 성능 향상의 핵심

### RAG Best Practices
- **Chunking**: 512-1024 토큰이 최적
- **Overlap**: 10-20% 오버랩 권장
- **Reranking**: 상위 10개 결과 재정렬 시 정확도 30% 향상

### Tool Integration
- **Function Calling**: OpenAI 스타일이 사실상 표준
- **Sandbox**: 코드 실행 시 필수
- **Rate Limiting**: 외부 API 호출 시 중요

## Technical Specifications

### Embedding Models
- **OpenAI text-embedding-3-small**: 비용 효율적, API 기반
- **text-embedding-3-large**: 높은 정확도, API 기반
- **Local Options**: all-MiniLM-L6-v2 (CPU 실행 가능)

### Vector Database - Pinecone
- **장점**: 
  - 완전 관리형 서비스 (인프라 관리 불필요)
  - 무료 티어 제공 (Starter plan)
  - 빠른 검색 성능
  - 메타데이터 필터링 지원
- **Starter Plan 제한사항**:
  - 1개 인덱스
  - 100K 벡터까지 무료
  - 충분한 학습/프로토타이핑 용량

### LLM Selection
- **GPT-4o**: 복잡한 추론, 도구 선택, 연구 계획 수립, 멀티모달 지원
- **GPT-4o-mini**: 일반 대화, 빠른 응답, 비용 효율적
- **모델 전환**: 사용자 선택 또는 작업 복잡도에 따라 자동 전환

## Development Tools
- **Backend Framework**: 
  - LangChain (Agent & RAG)
  - FastAPI (API server)
- **Frontend Stack**:
  - Next.js + React + TypeScript
  - Vercel AI SDK
  - shadcn/ui + Tailwind CSS
  - Lucide React (icons)
- **Database**: SQLite (Prisma ORM)
- **LLM**: OpenAI (GPT-4, GPT-3.5)
- **Deployment**: Vercel (frontend), Railway/Render (backend)