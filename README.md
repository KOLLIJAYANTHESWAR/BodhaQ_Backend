# BodhaQ Backend

Production-ready backend for **BodhaQ**, an AI-powered learning workspace that transforms study materials and topics into interactive learning experiences.

The backend is built with **FastAPI** and provides APIs for learning, study-material processing, RAG-based question answering, quizzes, resume preparation, coding practice, AI resource discovery, and secure code execution.

---

## 🚀 Overview

BodhaQ follows a continuous learning loop:

```text
STUDY
  ↓
ASSESS
  ↓
EVALUATE
  ↓
UNDERSTAND
  ↓
IDENTIFY WEAKNESS
  ↓
PRACTICE
  ↓
IMPROVE
  ↺
```

The backend is responsible for:

- Processing PDF, PPTX, and DOCX study materials
- Extracting and chunking educational content
- Generating embeddings
- Storing and retrieving document context with ChromaDB
- RAG-based question answering
- AI-generated quizzes
- Deterministic quiz evaluation
- Weak-topic identification
- Targeted practice generation
- Resume analysis and preparation
- AI coding-problem generation
- Test-case generation
- Secure Java/Python code execution
- Learning-resource discovery
- Video/resource search
- Anonymous session isolation
- Request-scoped Gemini and Tavily API keys

---

# 🏗️ Architecture

```text
                    ┌─────────────────────┐
                    │     BodhaQ Frontend │
                    │   React + Vite + JS │
                    └──────────┬──────────┘
                               │
                ┌──────────────┴──────────────┐
                │                             │
        X-BodhaQ-Session             Request-scoped API Keys
        X-Gemini-API-Key             X-Tavily-API-Key
                │                             │
                └──────────────┬──────────────┘
                               ↓
                    ┌─────────────────────┐
                    │    FastAPI Backend  │
                    └──────────┬──────────┘
                               │
          ┌────────────────────┼────────────────────┐
          ↓                    ↓                    ↓
     AI Services          Learning/RAG         Application
          │                    │                    │
     ┌────┴────┐         ┌─────┴─────┐       ┌─────┴─────┐
     │ Gemini  │         │ ChromaDB  │       │  SQLite   │
     │ Tavily  │         │ Embeddings│       │  Storage  │
     └─────────┘         └───────────┘       └───────────┘

                         ┌─────────────┐
                         │ Docker Code │
                         │   Sandbox   │
                         └─────────────┘
```

---

# 🔐 BYOK Security Model

BodhaQ uses a **Bring Your Own Key (BYOK)** architecture.

Users provide their own:

- Gemini API key
- Tavily API key

The backend does **not** maintain server-owned Gemini or Tavily API keys.

### API-key flow

```text
Browser
   │
   ├── Gemini API Key
   └── Tavily API Key
          │
          ↓
   HTTP Request Headers
          │
          ↓
      FastAPI
          │
          ├── Gemini
          └── Tavily
```

API keys are:

- Request-scoped
- Not stored in SQLite
- Not stored in ChromaDB
- Not written to files
- Not written to logs
- Not returned by API responses
- Not configured as backend provider secrets

The frontend keeps user-provided keys in browser `sessionStorage` and sends them only when required.

> **Security note:** Browser-based BYOK means the key is necessarily accessible to the user's browser session. BodhaQ therefore avoids persisting the key on the server and avoids logging it.

---

# 🔑 Anonymous Session Security

BodhaQ uses signed anonymous sessions instead of requiring traditional user accounts for the current architecture.

A session is created through:

```http
POST /api/session
```

The backend returns:

```json
{
  "session_id": "uuid",
  "session_token": "signed-session-token"
}
```

The frontend sends the token using:

```http
X-BodhaQ-Session: <session-token>
```

Session tokens are:

- Cryptographically signed
- Time-limited
- UUID-based
- Validated on protected requests
- Not used as permanent authentication credentials
- Not stored as user records

The current session token lifetime is **30 days**.

---

# 📚 Core Features

## 1. Study Materials

BodhaQ supports:

- PDF
- DOCX
- PPTX

Study documents are:

1. Uploaded
2. Validated
3. Parsed
4. Normalized
5. Chunked
6. Embedded
7. Stored in ChromaDB
8. Retrieved for RAG workflows

Supported document types are intentionally limited to reduce unsafe or unsupported file-processing paths.

---

## 2. RAG-Based Learning

The backend implements Retrieval-Augmented Generation for document-based learning.

```text
Document
   ↓
Parser
   ↓
Text Extraction
   ↓
Chunking
   ↓
Embeddings
   ↓
ChromaDB
   ↓
Retriever
   ↓
Relevant Context
   ↓
Gemini
   ↓
Answer
```

RAG data is isolated using the anonymous session and document identifiers.

---

## 3. AI Doubt Solving

Users can ask questions based on:

### Document mode

Questions are answered using relevant content retrieved from the user's uploaded material.

### Topic mode

Questions can be answered directly using the AI learning service without requiring an uploaded document.

The backend also handles:

- Conversation history
- Input validation
- API-key validation
- Controlled provider errors
- RAG retrieval failures

---

## 4. Quiz Generation

BodhaQ can generate quizzes from:

- Topics
- Documents
- Resume preparation items where internally required

Quiz generation supports:

- Configurable number of questions
- Difficulty levels
- AI-generated questions
- Multiple-choice answers
- Deterministic evaluation
- Weak-topic identification
- Targeted practice

Correct answers are kept private on the backend.

The public quiz response does **not** expose:

```text
correct_answer
explanation
```

until evaluation is performed by the backend.

---

## 5. Quiz Evaluation

Quiz evaluation is deterministic and performed by backend logic.

The backend calculates:

- Score
- Correct answers
- Incorrect answers
- Weak areas
- Topic-level performance

This avoids relying on an LLM to determine objective quiz scores.

---

## 6. Targeted Practice

Based on quiz performance, BodhaQ can identify weak areas and generate targeted practice.

```text
Quiz
 ↓
Evaluation
 ↓
Weak Topics
 ↓
Targeted Practice
 ↓
Improvement
```

Quiz and evaluation storage is session-isolated.

---

# 📄 Resume Preparation

The backend provides resume-preparation functionality including:

- Resume upload
- Resume parsing
- Resume-item generation
- Resume-related quiz generation
- Progress tracking
- Evaluation
- Weak-area identification

Supported resume formats are currently restricted to:

- PDF
- DOCX

Resume files and generated data are isolated by anonymous session.

---

# 💻 Coding Practice

BodhaQ includes an AI-powered coding workflow.

The backend can:

1. Generate coding problems
2. Generate test cases
3. Accept code submissions
4. Execute code
5. Evaluate output

Supported languages:

- Java
- Python

---

# 🐳 Secure Code Execution

Code execution is performed through Docker rather than directly on the host machine.

The sandbox includes controls such as:

- Network disabled
- Non-root execution
- Read-only root filesystem
- Limited memory
- Limited CPU
- Limited PIDs
- `no-new-privileges`
- Dropped Linux capabilities
- Temporary filesystem
- Execution timeout
- Compilation timeout
- Output-size limits
- Container cleanup
- Temporary workspace cleanup

Current execution limits include:

```text
Memory:              128 MB
CPU:                 0.5 CPU
PIDs:                64
Compile timeout:     10 seconds
Execution timeout:   5 seconds
Output limit:        100 KB
```

The backend currently uses:

```text
Java    → eclipse-temurin:17-alpine
Python  → python:3.10-alpine
```

> Docker isolation significantly reduces risk, but no general-purpose code-execution environment should be treated as an absolute security boundary without additional infrastructure hardening.

---

# 🔎 Learning Resource Search

BodhaQ can discover external learning resources through Tavily.

Resources may include:

- Documentation
- Tutorials
- Articles
- Stack Overflow discussions
- YouTube learning resources

Tavily API access uses the request-scoped BYOK model.

---

# 🧠 AI Integration

BodhaQ uses Google's Gemini API for AI-powered functionality.

The backend creates request-scoped Gemini clients rather than maintaining a global server-side API key.

AI functionality includes:

- Learning explanations
- Quiz generation
- Doubt solving
- Resume analysis
- Coding problem generation
- Test-case generation
- Other learning workflows

---

# 🛠️ Technology Stack

## Backend

- Python
- FastAPI
- Pydantic
- Uvicorn

## AI

- Google Gemini
- Tavily

## RAG

- ChromaDB
- Embeddings
- Retrieval pipeline

## Document Processing

- PyMuPDF
- python-docx
- python-pptx

## Database

- SQLite

## Code Execution

- Docker

## Testing

- Python test suite
- FastAPI API tests
- Docker execution tests
- Session isolation tests
- Production audit

---

# 📁 Project Structure

```text
BodhaQ_Backend/
│
├── app/
│   ├── __init__.py
│   ├── config.py
│   ├── main.py
│   │
│   ├── dependencies/
│   │   └── session.py
│   │
│   ├── ingestion/
│   │   ├── __init__.py
│   │   ├── chunker.py
│   │   ├── docx_loader.py
│   │   ├── pdf_loader.py
│   │   └── pptx_loader.py
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── requests.py
│   │   ├── responses.py
│   │   └── session.py
│   │
│   ├── rag/
│   │   ├── __init__.py
│   │   ├── embeddings.py
│   │   ├── retriever.py
│   │   └── vector_store.py
│   │
│   ├── routes/
│   │   ├── __init__.py
│   │   ├── coding.py
│   │   ├── documents.py
│   │   ├── doubts.py
│   │   ├── health.py
│   │   ├── learning.py
│   │   ├── quiz.py
│   │   ├── resume.py
│   │   ├── session.py
│   │   └── settings.py
│   │
│   ├── services/
│   │   ├── __init__.py
│   │   ├── code_execution_service.py
│   │   ├── document_service.py
│   │   ├── evaluation_service.py
│   │   ├── gemini_service.py
│   │   ├── problem_store.py
│   │   ├── quiz_service.py
│   │   ├── rag_service.py
│   │   ├── resource_search_service.py
│   │   ├── resume_service.py
│   │   └── session_service.py
│   │
│   └── utils/
│       ├── __init__.py
│       └── json_parser.py
│
├── tests/
│   ├── deep_test.py
│   ├── test_api.py
│   └── test_docker_execution.py
│
├── .env.example
├── .gitignore
├── backend_structure.txt
├── BodhaQ_Backend_API_Test_Cases.txt
├── production_audit.py
├── requirements.txt
├── test_problem_store_isolation.py
├── test_submit.py
└── README.md
```

---

# ⚙️ Installation

## Prerequisites

Install the following before running the backend:

- Python 3.10+
- Docker
- Git

Docker is required for the coding-execution functionality.

---

## 1. Clone the repository

```bash
git clone https://github.com/KOLLIJAYANTHESWAR/BodhaQ_Backend.git
cd BodhaQ_Backend
```

---

## 2. Create a virtual environment

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

### Linux/macOS

```bash
python3 -m venv venv
source venv/bin/activate
```

---

## 3. Install dependencies

```bash
pip install -r requirements.txt
```

---

# 🔧 Configuration

Create the local environment file from the example.

### Windows

```powershell
copy .env.example .env
```

### Linux/macOS

```bash
cp .env.example .env
```

Configure:

```env
FRONTEND_URL=http://localhost:5173
BODHAQ_SESSION_SECRET=replace_with_a_long_random_secret
```

Generate a strong session secret with:

```bash
python -c "import secrets; print(secrets.token_hex(32))"
```

Then place the generated value in:

```env
BODHAQ_SESSION_SECRET=...
```

---

# 🔐 API Keys

Do **not** add Gemini or Tavily API keys to `.env`.

BodhaQ uses request-scoped BYOK.

The frontend sends:

```http
X-Gemini-API-Key: <user-gemini-key>
```

and:

```http
X-Tavily-API-Key: <user-tavily-key>
```

only for requests that require those providers.

---

# ▶️ Running the Backend

Start the development server with:

```bash
uvicorn app.main:app --reload
```

The API will normally be available at:

```text
http://localhost:8000
```

FastAPI documentation:

```text
http://localhost:8000/docs
```

OpenAPI schema:

```text
http://localhost:8000/openapi.json
```

---

# ❤️ Health Check

The backend exposes a health endpoint.

```http
GET /health
```

A successful response indicates that the API is running.

---

# 🔌 API Areas

The backend currently exposes API areas for:

```text
/api/session
/api/documents
/api/quiz
/api/doubts
/api/settings
/api/resume
/api/coding
```

along with the health and root endpoints.

For the exact request and response schemas, use the generated FastAPI documentation:

```text
/docs
```

---

# 🧪 Testing

The backend contains multiple layers of testing.

Run the available test suite with:

```bash
pytest
```

Individual test files include:

```text
tests/test_api.py
tests/test_docker_execution.py
tests/deep_test.py
test_problem_store_isolation.py
test_submit.py
```

---

# 🔍 Production Audit

BodhaQ includes a dedicated production audit script:

```bash
python production_audit.py
```

The audit checks areas including:

- Python compilation
- FastAPI startup
- Route registration
- Service API compatibility
- BYOK configuration
- API-key handling
- Session security
- Session isolation
- Quiz answer privacy
- Evaluation storage
- SQLite schema
- Chroma/RAG isolation
- Upload path security
- Upload validation
- Temporary-file cleanup
- Docker availability
- Docker sandbox controls
- Runtime execution
- Timeout protection
- Output limits
- HTTP security headers
- CORS
- Dependency configuration
- Git ignore protection
- Debug-artifact scanning
- Session-aware routes

The production audit currently passes with:

```text
PASS: 37
FAIL: 0
SKIP: 0
```

---

# 🛡️ Security

BodhaQ implements several backend security controls.

## Request validation

Incoming requests are validated using Pydantic models and explicit service-level validation.

## Session isolation

Session-owned data is isolated using the anonymous session identifier.

This applies to areas including:

- Documents
- RAG collections
- Quizzes
- Evaluations
- Coding problems
- Resume data

## File validation

Uploaded files are:

- Size-limited
- Extension-validated
- Stored using controlled paths
- Assigned temporary UUID-based names where appropriate
- Cleaned up after processing

## Path safety

File operations are restricted to configured application directories.

## API-key protection

Provider keys are:

- Request-scoped
- Never persisted
- Never logged intentionally
- Never returned to clients

## HTTP security headers

The application applies headers including:

```text
X-Content-Type-Options: nosniff
X-Frame-Options: DENY
Referrer-Policy: strict-origin-when-cross-origin
Permissions-Policy
```

## CORS

CORS is restricted to the configured frontend origin.

---

# 💾 Runtime Data

The following directories/files are runtime-generated and should not be committed:

```text
data/
uploads/
*.db
*.sqlite
*.sqlite3
```

The `.gitignore` file protects these paths.

ChromaDB data is also treated as runtime application data.

---

# 🌐 Frontend Integration

The production frontend communicates with this backend through HTTP APIs.

The frontend is responsible for:

- Creating/maintaining the anonymous session token
- Storing user-provided BYOK keys for the browser session
- Sending the appropriate request headers
- Rendering API responses
- Managing client-side application state

The backend remains responsible for:

- Validation
- Session verification
- AI provider communication
- RAG
- Data isolation
- Quiz evaluation
- Code execution
- File processing
- Business logic

---

# 🔄 Request Flow Example

A protected request generally follows:

```text
1. Frontend creates session
          ↓
2. Backend returns signed session token
          ↓
3. Frontend stores token for the session
          ↓
4. Frontend sends X-BodhaQ-Session
          ↓
5. Backend validates token
          ↓
6. Backend extracts session ID
          ↓
7. Service verifies resource ownership
          ↓
8. Requested operation executes
          ↓
9. Controlled response returned
```

For AI operations:

```text
Frontend
   ↓
Session + provider API key
   ↓
FastAPI
   ↓
Validation
   ↓
Gemini / Tavily
   ↓
Controlled response
```

---

# 🚫 What the Backend Does Not Do

The current backend intentionally does not depend on:

- Server-owned Gemini API keys
- Server-owned Tavily API keys
- Traditional user authentication
- PostgreSQL
- Kubernetes
- Microservice infrastructure
- Multi-agent orchestration
- Custom-trained foundation models

The architecture is intentionally kept focused on the core BodhaQ learning workflow.

---

# 📌 Current Design Principles

BodhaQ backend follows these principles:

### Security first

User data and provider credentials should not be unnecessarily persisted.

### Stateless AI credentials

AI provider credentials are supplied per request.

### Session isolation

Anonymous users should not be able to access another session's data.

### Deterministic evaluation

Objective quiz scoring is handled by backend logic rather than relying on AI judgment.

### Controlled errors

Internal exceptions and provider details should not unnecessarily leak through API responses.

### Resource limits

File processing, AI inputs, quiz storage, evaluation storage, and code execution are bounded.

### Minimal architecture

The system avoids unnecessary infrastructure complexity while maintaining clear service boundaries.

---

# 📈 Production Readiness

The backend has undergone a dedicated production audit covering:

```text
Foundation
BYOK
Session Security
Data Isolation
Privacy
File Security
Docker Sandbox
Runtime Protection
HTTP Security
CORS
Dependencies
Deployment Configuration
Route Session Protection
```

Current audit result:

```text
========================================
BodhaQ Backend Production Audit
========================================

PASS: 37
FAIL: 0
SKIP: 0

RESULT: PRODUCTION AUDIT PASSED
```

---

# 🧩 Development Philosophy

BodhaQ is designed as a modular backend.

```text
Routes
  ↓
Services
  ↓
RAG / AI / Storage / Execution
```

Routes handle HTTP concerns.

Services handle business logic.

RAG modules handle retrieval and vector storage.

Ingestion modules handle document processing.

The code-execution service handles isolated execution.

This separation makes the backend easier to test, maintain, and extend.

---

# 🔮 Future Development

Potential future improvements may include:

- Persistent user accounts
- More advanced authentication
- Additional programming languages
- More document formats
- OCR support
- Improved retrieval strategies
- More advanced learning analytics
- Additional AI providers
- Cloud object storage
- Production-grade distributed rate limiting
- Dedicated production observability
- Scalable execution infrastructure

These are intentionally outside the current backend scope.

---

# 👨‍💻 Author

**Kolli Jayanth Eswar**

GitHub:

https://github.com/KOLLIJAYANTHESWAR

---

# 📄 License

Add the project's chosen license here before publishing the project as an open-source package.
