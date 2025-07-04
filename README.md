# AI & Backend Engineering Interview Preparation Notes

## Table of Contents
1. [RAG (Retrieval-Augmented Generation)](#rag)
2. [Agentic AI](#agentic-ai)
3. [FastAPI](#fastapi)
4. [WebSockets](#websockets)
5. [LLM Testing](#llm-testing)
6. [Fine-tuning](#fine-tuning)
7. [Vector Databases](#vector-databases)

---

## 1. RAG (Retrieval-Augmented Generation) {#rag}

### Overview
RAG combines the power of retrieval systems with generative AI models to provide more accurate, up-to-date, and contextual responses by referencing external knowledge bases.

### Architecture Components

| Component | Purpose | Common Tools |
|-----------|---------|--------------|
| Document Loader | Ingests various file formats | LangChain, LlamaIndex |
| Text Splitter | Chunks documents into manageable pieces | RecursiveCharacterTextSplitter, TokenTextSplitter |
| Embedding Model | Converts text to vectors | OpenAI Ada, Sentence-BERT, Cohere |
| Vector Store | Stores and retrieves embeddings | Pinecone, Weaviate, ChromaDB |
| Retriever | Finds relevant documents | Similarity search, MMR, Hybrid search |
| LLM | Generates responses | GPT-4, Claude, Llama |
| Prompt Template | Structures the context | LangChain PromptTemplate |

### Implementation Example

```python
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI

# 1. Load documents
loader = PyPDFLoader("document.pdf")
documents = loader.load()

# 2. Split documents
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)
chunks = text_splitter.split_documents(documents)

# 3. Create embeddings and vector store
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(
    documents=chunks,
    embedding=embeddings,
    persist_directory="./chroma_db"
)

# 4. Create retriever
retriever = vectorstore.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 4}
)

# 5. Create RAG chain
llm = OpenAI(temperature=0)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,
    return_source_documents=True
)

# 6. Query
result = qa_chain({"query": "What is the main topic of the document?"})
print(result["result"])
```

### Advanced RAG Techniques

#### Hybrid Search Implementation
```python
from langchain.retrievers import EnsembleRetriever
from langchain.retrievers import BM25Retriever

# Combine dense and sparse retrieval
bm25_retriever = BM25Retriever.from_documents(chunks)
bm25_retriever.k = 2

vector_retriever = vectorstore.as_retriever(search_kwargs={"k": 2})

# Ensemble retriever
ensemble_retriever = EnsembleRetriever(
    retrievers=[bm25_retriever, vector_retriever],
    weights=[0.5, 0.5]
)
```

#### Contextual Compression
```python
from langchain.retrievers import ContextualCompressionRetriever
from langchain.retrievers.document_compressors import LLMChainExtractor

compressor = LLMChainExtractor.from_llm(llm)
compression_retriever = ContextualCompressionRetriever(
    base_compressor=compressor,
    base_retriever=retriever
)
```

### Gotchas
1. **Chunk Size Trade-off**: Too small = loss of context, Too large = irrelevant info
2. **Embedding Model Consistency**: Must use same model for indexing and querying
3. **Token Limits**: Retrieved context + query must fit within LLM context window
4. **Semantic Similarity ≠ Relevance**: High similarity scores don't guarantee relevance
5. **Cold Start Problem**: Need sufficient data for effective retrieval

### Pros and Cons

| Pros | Cons |
|------|------|
| ✓ Reduces hallucinations | ✗ Increased latency (retrieval + generation) |
| ✓ Up-to-date information | ✗ Higher computational cost |
| ✓ Source attribution | ✗ Complex pipeline to maintain |
| ✓ Domain-specific knowledge | ✗ Quality depends on document corpus |
| ✓ Scalable to large knowledge bases | ✗ Retrieval failures can mislead LLM |

---

## 2. Agentic AI {#agentic-ai}

### Overview
Agentic AI refers to AI systems that can autonomously perform tasks, make decisions, and interact with various tools and APIs to achieve complex goals.

### Core Components

| Component | Function | Example Implementation |
|-----------|----------|----------------------|
| Planning | Breaks down complex tasks | Chain-of-Thought, ReAct |
| Memory | Maintains context and state | Short-term, Long-term, Episodic |
| Tools | Executes actions | APIs, Databases, Search |
| Reflection | Self-evaluates and improves | Critique, Retry logic |
| Execution | Carries out planned actions | Function calling, Code execution |

### Implementation Examples

#### Basic Agent with LangChain
```python
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.llms import OpenAI
from langchain.utilities import SerpAPIWrapper
from langchain.utilities import PythonREPL

# Define tools
search = SerpAPIWrapper()
python_repl = PythonREPL()

tools = [
    Tool(
        name="Search",
        func=search.run,
        description="Search for current information"
    ),
    Tool(
        name="Python",
        func=python_repl.run,
        description="Execute Python code"
    )
]

# Custom prompt template
template = """You are an AI assistant with access to tools.

Tools:
{tools}

Use this format:
Thought: Consider what to do
Action: the action to take, one of [{tool_names}]
Action Input: the input to the action
Observation: the result of the action
... (repeat as needed)
Thought: I now know the final answer
Final Answer: the final answer

Question: {input}
{agent_scratchpad}"""

class CustomPromptTemplate(StringPromptTemplate):
    template: str
    tools: List[Tool]
    
    def format(self, **kwargs) -> str:
        kwargs["tools"] = "\n".join([f"{tool.name}: {tool.description}" for tool in self.tools])
        kwargs["tool_names"] = ", ".join([tool.name for tool in self.tools])
        return self.template.format(**kwargs)

prompt = CustomPromptTemplate(
    template=template,
    tools=tools,
    input_variables=["input", "agent_scratchpad"]
)

# Create agent
llm = OpenAI(temperature=0)
agent = LLMSingleActionAgent(
    llm_chain=LLMChain(llm=llm, prompt=prompt),
    output_parser=CustomOutputParser(),
    stop=["\nObservation:"],
    allowed_tools=[tool.name for tool in tools]
)

agent_executor = AgentExecutor.from_agent_and_tools(
    agent=agent,
    tools=tools,
    verbose=True
)

# Run agent
result = agent_executor.run("What's the weather in NYC and calculate the temperature in Celsius if it's in Fahrenheit")
```

#### ReAct Agent Pattern
```python
class ReActAgent:
    def __init__(self, llm, tools):
        self.llm = llm
        self.tools = {tool.name: tool for tool in tools}
        self.max_iterations = 5
        
    def think(self, question, scratchpad):
        prompt = f"""Question: {question}
Thought: Let me think about this step by step.
{scratchpad}
Thought:"""
        return self.llm.generate(prompt)
    
    def act(self, thought, available_tools):
        action_prompt = f"""Based on this thought: {thought}
Available tools: {', '.join(available_tools)}
Action:"""
        action = self.llm.generate(action_prompt)
        return self.parse_action(action)
    
    def observe(self, action, action_input):
        if action in self.tools:
            return self.tools[action].run(action_input)
        return "Invalid action"
    
    def run(self, question):
        scratchpad = ""
        for i in range(self.max_iterations):
            thought = self.think(question, scratchpad)
            action, action_input = self.act(thought, list(self.tools.keys()))
            observation = self.observe(action, action_input)
            
            scratchpad += f"\nThought: {thought}\nAction: {action}\nAction Input: {action_input}\nObservation: {observation}\n"
            
            if "Final Answer:" in thought:
                return self.extract_final_answer(thought)
        
        return "Max iterations reached"
```

### Memory Systems

```python
from collections import deque
from typing import Dict, List, Any
import json

class AgentMemory:
    def __init__(self):
        self.short_term = deque(maxlen=10)  # Recent interactions
        self.long_term = {}  # Persistent storage
        self.episodic = []  # Specific task completions
        
    def add_interaction(self, interaction: Dict[str, Any]):
        self.short_term.append(interaction)
        
    def store_knowledge(self, key: str, value: Any):
        self.long_term[key] = value
        
    def recall_recent(self, n: int = 5) -> List[Dict]:
        return list(self.short_term)[-n:]
    
    def add_episode(self, task: str, result: str, steps: List[str]):
        self.episodic.append({
            "task": task,
            "result": result,
            "steps": steps,
            "timestamp": datetime.now()
        })
    
    def find_similar_episodes(self, task: str, similarity_threshold: float = 0.7):
        # Implement similarity search
        similar = []
        for episode in self.episodic:
            similarity = self.calculate_similarity(task, episode["task"])
            if similarity > similarity_threshold:
                similar.append(episode)
        return similar
```

### Tool Creation and Integration

```python
from typing import Optional
from pydantic import BaseModel, Field

class ToolSchema(BaseModel):
    name: str = Field(description="Tool name")
    description: str = Field(description="What the tool does")
    parameters: dict = Field(description="Input parameters")

class WeatherTool:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.schema = ToolSchema(
            name="get_weather",
            description="Get current weather for a location",
            parameters={
                "location": {"type": "string", "description": "City name"},
                "units": {"type": "string", "enum": ["celsius", "fahrenheit"]}
            }
        )
    
    async def run(self, location: str, units: str = "celsius") -> dict:
        # Implementation
        response = await fetch_weather_api(location, self.api_key)
        return {
            "temperature": convert_units(response["temp"], units),
            "description": response["description"],
            "humidity": response["humidity"]
        }

# Function calling example
def create_function_calling_agent(llm, tools):
    tool_schemas = [tool.schema.dict() for tool in tools]
    
    system_prompt = f"""You are an AI assistant with access to these tools:
{json.dumps(tool_schemas, indent=2)}

Always respond with a JSON object in this format:
{{
    "thought": "Your reasoning",
    "tool": "tool_name or null",
    "parameters": {{}} or null,
    "final_answer": "Your response or null"
}}"""
    
    return FunctionCallingAgent(llm, tools, system_prompt)
```

### Gotchas
1. **Infinite Loops**: Agents can get stuck in repetitive actions
2. **Context Window Limits**: Long conversations exceed token limits
3. **Tool Reliability**: External API failures can break agent flow
4. **Hallucinated Actions**: Agents might invent non-existent tools
5. **Cost Explosion**: Multiple LLM calls per task
6. **State Management**: Maintaining consistency across interactions

### Pros and Cons

| Pros | Cons |
|------|------|
| ✓ Autonomous task completion | ✗ Unpredictable behavior |
| ✓ Complex problem solving | ✗ High API costs |
| ✓ Tool integration flexibility | ✗ Difficult to debug |
| ✓ Adaptive to new scenarios | ✗ Requires extensive error handling |
| ✓ Can learn from interactions | ✗ Security concerns with tool access |

---

## 3. FastAPI {#fastapi}

### Overview
FastAPI is a modern, fast web framework for building APIs with Python 3.7+ based on standard Python type hints.

### Core Features

| Feature | Description | Code Example |
|---------|-------------|--------------|
| Type Hints | Automatic validation | `name: str, age: int` |
| Async Support | Native async/await | `async def endpoint()` |
| Auto Documentation | Swagger/ReDoc | `/docs`, `/redoc` |
| Dependency Injection | Reusable components | `Depends()` |
| Security | OAuth2, JWT support | `OAuth2PasswordBearer` |

### Basic Implementation

```python
from fastapi import FastAPI, HTTPException, Depends, status
from pydantic import BaseModel, Field, validator
from typing import Optional, List
from datetime import datetime
import uvicorn

app = FastAPI(
    title="AI Service API",
    description="Production-ready AI service",
    version="1.0.0"
)

# Pydantic models
class ChatMessage(BaseModel):
    role: str = Field(..., regex="^(user|assistant|system)$")
    content: str = Field(..., min_length=1, max_length=4000)
    timestamp: datetime = Field(default_factory=datetime.now)
    
    @validator('content')
    def content_must_not_be_empty(cls, v):
        if not v.strip():
            raise ValueError('Content cannot be empty')
        return v

class ChatRequest(BaseModel):
    messages: List[ChatMessage]
    model: str = "gpt-4"
    temperature: float = Field(0.7, ge=0, le=2)
    max_tokens: Optional[int] = Field(None, gt=0, le=4000)

class ChatResponse(BaseModel):
    id: str
    choices: List[dict]
    usage: dict
    created: datetime

# Dependency injection
async def get_api_key(api_key: str = Depends(lambda: ...)):
    if not validate_api_key(api_key):
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid API key"
        )
    return api_key

# Endpoints
@app.post("/chat/completions", response_model=ChatResponse)
async def create_chat_completion(
    request: ChatRequest,
    api_key: str = Depends(get_api_key)
):
    try:
        # Process request
        result = await process_chat_request(request)
        return ChatResponse(**result)
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Middleware
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.trustedhost import TrustedHostMiddleware
import time

app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://example.com"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.middleware("http")
async def add_process_time_header(request, call_next):
    start_time = time.time()
    response = await call_next(request)
    process_time = time.time() - start_time
    response.headers["X-Process-Time"] = str(process_time)
    return response

# Background tasks
from fastapi import BackgroundTasks

@app.post("/process-document")
async def process_document(
    file: UploadFile,
    background_tasks: BackgroundTasks
):
    # Save file
    file_path = await save_upload(file)
    
    # Add background task
    background_tasks.add_task(
        process_document_async,
        file_path
    )
    
    return {"message": "Processing started", "file_id": file_path}

# WebSocket endpoint
from fastapi import WebSocket, WebSocketDisconnect

@app.websocket("/ws/chat/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            response = await process_streaming_chat(data)
            await websocket.send_text(response)
    except WebSocketDisconnect:
        await manager.disconnect(client_id)
```

### Advanced Patterns

#### Database Integration with SQLAlchemy
```python
from sqlalchemy.ext.asyncio import AsyncSession, create_async_engine
from sqlalchemy.orm import sessionmaker
from fastapi import Depends

# Database setup
engine = create_async_engine("postgresql+asyncpg://user:pass@localhost/db")
AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession)

# Dependency
async def get_db():
    async with AsyncSessionLocal() as session:
        yield session

# Repository pattern
class ChatRepository:
    def __init__(self, db: AsyncSession):
        self.db = db
    
    async def save_chat(self, chat_data: dict):
        chat = ChatModel(**chat_data)
        self.db.add(chat)
        await self.db.commit()
        return chat
    
    async def get_chat_history(self, user_id: str, limit: int = 10):
        result = await self.db.execute(
            select(ChatModel)
            .where(ChatModel.user_id == user_id)
            .order_by(ChatModel.created_at.desc())
            .limit(limit)
        )
        return result.scalars().all()

# Using in endpoint
@app.get("/chat/history/{user_id}")
async def get_chat_history(
    user_id: str,
    db: AsyncSession = Depends(get_db)
):
    repo = ChatRepository(db)
    history = await repo.get_chat_history(user_id)
    return history
```

#### Rate Limiting
```python
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.depends import RateLimiter
import redis.asyncio as redis

@app.on_event("startup")
async def startup():
    redis_client = redis.from_url("redis://localhost", encoding="utf-8", decode_responses=True)
    await FastAPILimiter.init(redis_client)

@app.post("/api/generate", dependencies=[Depends(RateLimiter(times=10, seconds=60))])
async def generate_text(prompt: str):
    return {"result": "generated text"}
```

#### Custom Exception Handling
```python
class AIServiceException(Exception):
    def __init__(self, status_code: int, detail: str):
        self.status_code = status_code
        self.detail = detail

@app.exception_handler(AIServiceException)
async def ai_exception_handler(request, exc: AIServiceException):
    return JSONResponse(
        status_code=exc.status_code,
        content={"error": exc.detail, "type": "ai_service_error"}
    )

# Global error handler
@app.exception_handler(Exception)
async def global_exception_handler(request, exc):
    return JSONResponse(
        status_code=500,
        content={
            "error": "Internal server error",
            "detail": str(exc) if settings.DEBUG else None
        }
    )
```

### Testing FastAPI

```python
from fastapi.testclient import TestClient
import pytest
from httpx import AsyncClient

# Sync testing
def test_create_chat():
    client = TestClient(app)
    response = client.post(
        "/chat/completions",
        json={
            "messages": [{"role": "user", "content": "Hello"}],
            "model": "gpt-4"
        },
        headers={"Authorization": "Bearer test-key"}
    )
    assert response.status_code == 200
    assert "choices" in response.json()

# Async testing
@pytest.mark.asyncio
async def test_websocket():
    async with AsyncClient(app=app, base_url="http://test") as client:
        with client.websocket_connect("/ws/chat/123") as websocket:
            await websocket.send_text("Hello")
            data = await websocket.receive_text()
            assert data == "Response"
```

### Gotchas
1. **Async vs Sync**: Mixing can cause blocking
2. **Pydantic V1 vs V2**: Breaking changes between versions
3. **Background Tasks**: Not suitable for long-running tasks
4. **File Uploads**: Large files can consume memory
5. **Dependency Injection**: Circular dependencies
6. **Type Hints**: Runtime overhead for complex types

### Pros and Cons

| Pros | Cons |
|------|------|
| ✓ High performance | ✗ Smaller ecosystem than Flask/Django |
| ✓ Auto-generated docs | ✗ Learning curve for async |
| ✓ Type safety | ✗ Debugging async code |
| ✓ Modern Python features | ✗ Limited built-in features |
| ✓ Great developer experience | ✗ Requires Python 3.7+ |

---

## 4. WebSockets {#websockets}

### Overview
WebSockets provide full-duplex, bidirectional communication channels over a single TCP connection, ideal for real-time applications.

### Protocol Overview

| Phase | Description | Key Headers |
|-------|-------------|-------------|
| Handshake | HTTP upgrade request | `Upgrade: websocket`, `Sec-WebSocket-Key` |
| Connection | Persistent TCP connection | `Sec-WebSocket-Accept` |
| Data Transfer | Binary/text frames | Opcode, Payload length |
| Closing | Clean disconnect | Close frame with status |

### Implementation Examples

#### Basic WebSocket Server (FastAPI)
```python
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from typing import Dict, Set
import json
import asyncio

app = FastAPI()

class ConnectionManager:
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.message_queue: asyncio.Queue = asyncio.Queue()
        
    async def connect(self, websocket: WebSocket, client_id: str):
        await websocket.accept()
        self.active_connections[client_id] = websocket
        await self.send_personal_message(
            f"Connected as {client_id}", 
            websocket
        )
    
    def disconnect(self, client_id: str):
        if client_id in self.active_connections:
            del self.active_connections[client_id]
    
    async def send_personal_message(self, message: str, websocket: WebSocket):
        await websocket.send_text(message)
    
    async def broadcast(self, message: str, exclude: Set[str] = None):
        exclude = exclude or set()
        disconnected = []
        
        for client_id, connection in self.active_connections.items():
            if client_id not in exclude:
                try:
                    await connection.send_text(message)
                except:
                    disconnected.append(client_id)
        
        # Clean up disconnected clients
        for client_id in disconnected:
            self.disconnect(client_id)

manager = ConnectionManager()

@app.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    await manager.connect(websocket, client_id)
    try:
        while True:
            # Receive message
            data = await websocket.receive_text()
            message = json.loads(data)
            
            # Process different message types
            if message["type"] == "chat":
                await manager.broadcast(
                    json.dumps({
                        "type": "chat",
                        "from": client_id,
                        "content": message["content"],
                        "timestamp": datetime.now().isoformat()
                    }),
                    exclude={client_id}
                )
            elif message["type"] == "typing":
                await manager.broadcast(
                    json.dumps({
                        "type": "typing",
                        "user": client_id
                    }),
                    exclude={client_id}
                )
                
    except WebSocketDisconnect:
        manager.disconnect(client_id)
        await manager.broadcast(
            json.dumps({
                "type": "user_left",
                "user": client_id
            })
        )
```

#### Advanced WebSocket with Authentication
```python
from fastapi import WebSocket, Query, HTTPException, status
from jose import JWTError, jwt
import asyncio
from typing import Optional

class AuthWebSocketManager:
    def __init__(self, secret_key: str):
        self.secret_key = secret_key
        self.connections: Dict[str, Dict] = {}
        
    async def authenticate(self, token: str) -> Optional[dict]:
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=["HS256"])
            return payload
        except JWTError:
            return None
    
    async def connect_with_auth(self, websocket: WebSocket, token: str):
        user_info = await self.authenticate(token)
        if not user_info:
            await websocket.close(code=status.WS_1008_POLICY_VIOLATION)
            return None
            
        await websocket.accept()
        client_id = user_info["sub"]
        self.connections[client_id] = {
            "websocket": websocket,
            "user_info": user_info,
            "subscriptions": set()
        }
        return client_id

@app.websocket("/ws/authenticated")
async def authenticated_websocket(
    websocket: WebSocket,
    token: str = Query(...)
):
    client_id = await auth_manager.connect_with_auth(websocket, token)
    if not client_id:
        return
    
    try:
        while True:
            data = await websocket.receive_json()
            await process_authenticated_message(client_id, data)
    except WebSocketDisconnect:
        auth_manager.disconnect(client_id)
```

#### WebSocket with Rooms/Channels
```python
class RoomManager:
    def __init__(self):
        self.rooms: Dict[str, Set[str]] = {}
        self.user_rooms: Dict[str, Set[str]] = {}
        
    def join_room(self, user_id: str, room_id: str):
        if room_id not in self.rooms:
            self.rooms[room_id] = set()
        self.rooms[room_id].add(user_id)
        
        if user_id not in self.user_rooms:
            self.user_rooms[user_id] = set()
        self.user_rooms[user_id].add(room_id)
    
    def leave_room(self, user_id: str, room_id: str):
        if room_id in self.rooms:
            self.rooms[room_id].discard(user_id)
            if not self.rooms[room_id]:
                del self.rooms[room_id]
        
        if user_id in self.user_rooms:
            self.user_rooms[user_id].discard(room_id)
    
    def get_room_members(self, room_id: str) -> Set[str]:
        return self.rooms.get(room_id, set())
    
    async def broadcast_to_room(self, room_id: str, message: dict, exclude: Set[str] = None):
        exclude = exclude or set()
        members = self.get_room_members(room_id)
        
        for member_id in members:
            if member_id not in exclude and member_id in manager.active_connections:
                try:
                    await manager.active_connections[member_id].send_json(message)
                except:
                    pass
```

#### Client-Side Implementation
```javascript
class WebSocketClient {
    constructor(url, options = {}) {
        this.url = url;
        this.options = options;
        this.ws = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = options.maxReconnectAttempts || 5;
        this.reconnectDelay = options.reconnectDelay || 1000;
        this.messageQueue = [];
        this.isConnected = false;
    }
    
    connect() {
        this.ws = new WebSocket(this.url);
        
        this.ws.onopen = (event) => {
            console.log('WebSocket connected');
            this.isConnected = true;
            this.reconnectAttempts = 0;
            this.flushMessageQueue();
            if (this.options.onOpen) this.options.onOpen(event);
        };
        
        this.ws.onmessage = (event) => {
            const data = JSON.parse(event.data);
            if (this.options.onMessage) this.options.onMessage(data);
        };
        
        this.ws.onerror = (event) => {
            console.error('WebSocket error:', event);
            if (this.options.onError) this.options.onError(event);
        };
        
        this.ws.onclose = (event) => {
            this.isConnected = false;
            if (this.options.onClose) this.options.onClose(event);
            
            if (this.reconnectAttempts < this.maxReconnectAttempts) {
                setTimeout(() => {
                    this.reconnectAttempts++;
                    this.connect();
                }, this.reconnectDelay * Math.pow(2, this.reconnectAttempts));
            }
        };
    }
    
    send(data) {
        const message = typeof data === 'string' ? data : JSON.stringify(data);
        
        if (this.isConnected && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(message);
        } else {
            this.messageQueue.push(message);
        }
    }
    
    flushMessageQueue() {
        while (this.messageQueue.length > 0 && this.isConnected) {
            const message = this.messageQueue.shift();
            this.ws.send(message);
        }
    }
    
    close() {
        if (this.ws) {
            this.ws.close();
        }
    }
}

// Usage
const client = new WebSocketClient('ws://localhost:8000/ws/123', {
    onMessage: (data) => {
        console.log('Received:', data);
    },
    onOpen: () => {
        client.send({ type: 'subscribe', channel: 'updates' });
    }
});

client.connect();
```

#### Scaling WebSockets with Redis
```python
import aioredis
from typing import Dict, Set
import json

class ScalableWebSocketManager:
    def __init__(self, redis_url: str):
        self.redis_url = redis_url
        self.local_connections: Dict[str, WebSocket] = {}
        self.redis_client = None
        self.pubsub = None
        
    async def initialize(self):
        self.redis_client = await aioredis.create_redis_pool(self.redis_url)
        self.pubsub = self.redis_client.pubsub()
        await self.pubsub.subscribe("broadcast_channel")
        asyncio.create_task(self.redis_listener())
    
    async def redis_listener(self):
        async for message in self.pubsub.listen():
            if message["type"] == "message":
                data = json.loads(message["data"])
                await self.local_broadcast(data)
    
    async def broadcast(self, message: dict):
        # Publish to Redis for other instances
        await self.redis_client.publish(
            "broadcast_channel",
            json.dumps(message)
        )
    
    async def local_broadcast(self, message: dict):
        disconnected = []
        for client_id, ws in self.local_connections.items():
            try:
                await ws.send_json(message)
            except:
                disconnected.append(client_id)
        
        for client_id in disconnected:
            del self.local_connections[client_id]
```

### Gotchas
1. **Browser Connection Limits**: ~255 connections per domain
2. **Proxy/Load Balancer**: May timeout long connections
3. **Message Ordering**: Not guaranteed across multiple servers
4. **Memory Leaks**: Forgetting to clean up connections
5. **Security**: No built-in authentication
6. **Binary Data**: Different handling across browsers

### Pros and Cons

| Pros | Cons |
|------|------|
| ✓ Real-time bidirectional | ✗ Complex scaling |
| ✓ Low latency | ✗ Stateful connections |
| ✓ Efficient for frequent updates | ✗ Firewall/proxy issues |
| ✓ Native browser support | ✗ No built-in reconnection |
| ✓ Less overhead than polling | ✗ Debugging challenges |

---

## 5. LLM Testing {#llm-testing}

### Overview
Testing LLMs requires specialized approaches due to their non-deterministic nature and the subjective quality of outputs.

### Testing Categories

| Category | Purpose | Methods |
|----------|---------|---------|
| Unit Tests | Test specific functions | Mocking, fixtures |
| Integration Tests | Test API endpoints | Response validation |
| Evaluation Tests | Test output quality | Metrics, human eval |
| Regression Tests | Prevent degradation | Golden datasets |
| Load Tests | Test scalability | Concurrent requests |

### Implementation Examples

#### Unit Testing with Mocks
```python
import pytest
from unittest.mock import Mock, patch, AsyncMock
import asyncio

class LLMService:
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.client = OpenAI(api_key=api_key)
    
    async def generate_response(self, prompt: str, **kwargs):
        response = await self.client.completions.create(
            prompt=prompt,
            **kwargs
        )
        return response.choices[0].text

# Test file
@pytest.fixture
def mock_openai_response():
    return {
        "choices": [{
            "text": "Mocked response",
            "index": 0,
            "finish_reason": "stop"
        }],
        "usage": {
            "prompt_tokens": 10,
            "completion_tokens": 5,
            "total_tokens": 15
        }
    }

@pytest.mark.asyncio
async def test_generate_response(mock_openai_response):
    with patch('openai.OpenAI') as mock_client:
        mock_instance = AsyncMock()
        mock_client.return_value = mock_instance
        mock_instance.completions.create.return_value = mock_openai_response
        
        service = LLMService("test-key")
        result = await service.generate_response("Test prompt")
        
        assert result == "Mocked response"
        mock_instance.completions.create.assert_called_once()
```

#### Evaluation Framework
```python
from dataclasses import dataclass
from typing import List, Dict, Callable
import numpy as np
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer

@dataclass
class EvaluationMetric:
    name: str
    function: Callable
    threshold: float = None

class LLMEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        self.embedding_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.metrics = self._initialize_metrics()
    
    def _initialize_metrics(self) -> List[EvaluationMetric]:
        return [
            EvaluationMetric("rouge", self.calculate_rouge),
            EvaluationMetric("semantic_similarity", self.calculate_semantic_similarity),
            EvaluationMetric("length_ratio", self.calculate_length_ratio),
            EvaluationMetric("toxicity", self.calculate_toxicity, threshold=0.1),
            EvaluationMetric("factuality", self.calculate_factuality)
        ]
    
    def calculate_rouge(self, generated: str, reference: str) -> Dict[str, float]:
        scores = self.rouge_scorer.score(reference, generated)
        return {
            "rouge1_f1": scores['rouge1'].fmeasure,
            "rouge2_f1": scores['rouge2'].fmeasure,
            "rougeL_f1": scores['rougeL'].fmeasure
        }
    
    def calculate_semantic_similarity(self, generated: str, reference: str) -> float:
        embeddings = self.embedding_model.encode([generated, reference])
        similarity = np.dot(embeddings[0], embeddings[1]) / (
            np.linalg.norm(embeddings[0]) * np.linalg.norm(embeddings[1])
        )
        return float(similarity)
    
    def calculate_length_ratio(self, generated: str, reference: str) -> float:
        return len(generated.split()) / len(reference.split())
    
    async def calculate_factuality(self, generated: str, context: str = None) -> float:
        # Use another LLM to evaluate factuality
        prompt = f"""
        Evaluate the factual accuracy of the following text on a scale of 0-1:
        Text: {generated}
        Context: {context or 'General knowledge'}
        
        Respond with only a number between 0 and 1.
        """
        score = await self.fact_check_llm(prompt)
        return float(score)
    
    def evaluate(self, test_cases: List[Dict]) -> Dict[str, Dict]:
        results = {}
        
        for case in test_cases:
            case_id = case["id"]
            generated = case["generated"]
            reference = case.get("reference")
            
            results[case_id] = {}
            
            for metric in self.metrics:
                if metric.name in ["rouge", "semantic_similarity", "length_ratio"] and reference:
                    if metric.name == "rouge":
                        results[case_id].update(metric.function(generated, reference))
                    else:
                        results[case_id][metric.name] = metric.function(generated, reference)
                elif metric.name in ["toxicity", "factuality"]:
                    results[case_id][metric.name] = metric.function(generated)
        
        return results
```

#### A/B Testing Framework
```python
import random
from typing import Dict, List, Tuple
import pandas as pd
from scipy import stats

class ABTestFramework:
    def __init__(self, models: Dict[str, Callable]):
        self.models = models
        self.results = []
        
    async def run_test(
        self, 
        test_prompts: List[str], 
        evaluation_function: Callable,
        sample_size_per_model: int = 100
    ):
        for prompt in test_prompts:
            for _ in range(sample_size_per_model):
                # Randomly select model
                model_name = random.choice(list(self.models.keys()))
                model = self.models[model_name]
                
                # Generate response
                response = await model(prompt)
                
                # Evaluate
                score = await evaluation_function(prompt, response)
                
                self.results.append({
                    "prompt": prompt,
                    "model": model_name,
                    "response": response,
                    "score": score
                })
        
        return self.analyze_results()
    
    def analyze_results(self) -> Dict:
        df = pd.DataFrame(self.results)
        
        # Calculate statistics per model
        stats_summary = df.groupby('model')['score'].agg([
            'mean', 'std', 'count', 'min', 'max'
        ]).to_dict('index')
        
        # Perform statistical tests
        models = list(self.models.keys())
        if len(models) == 2:
            # T-test for two models
            scores_a = df[df['model'] == models[0]]['score']
            scores_b = df[df['model'] == models[1]]['score']
            t_stat, p_value = stats.ttest_ind(scores_a, scores_b)
            
            stats_summary['statistical_test'] = {
                'test': 't-test',
                't_statistic': t_stat,
                'p_value': p_value,
                'significant': p_value < 0.05
            }
        
        return stats_summary
```

#### Property-Based Testing
```python
from hypothesis import given, strategies as st, assume
import re

class LLMPropertyTester:
    def __init__(self, llm_function):
        self.llm = llm_function
    
    @given(st.text(min_size=10, max_size=500))
    async def test_output_format(self, input_text):
        """Test that output maintains expected format"""
        prompt = f"Summarize: {input_text}"
        output = await self.llm(prompt)
        
        # Properties to check
        assert isinstance(output, str)
        assert len(output) > 0
        assert len(output.split()) <= len(input_text.split())  # Summary shorter
    
    @given(st.lists(st.text(min_size=5, max_size=50), min_size=2, max_size=10))
    async def test_consistency(self, items):
        """Test that similar inputs produce consistent outputs"""
        prompt1 = f"List these items: {', '.join(items)}"
        prompt2 = f"These are the items: {', '.join(items)}"
        
        output1 = await self.llm(prompt1)
        output2 = await self.llm(prompt2)
        
        # Check all items appear in both outputs
        for item in items:
            assert item in output1 or item.lower() in output1.lower()
            assert item in output2 or item.lower() in output2.lower()
    
    @given(st.text(alphabet=st.characters(blacklist_categories=['Cc', 'Cs'])))
    async def test_safety(self, potentially_unsafe_input):
        """Test that model handles edge cases safely"""
        output = await self.llm(potentially_unsafe_input)
        
        # Should not throw errors
        assert output is not None
        # Should not leak sensitive patterns
        assert not re.search(r'api[_-]?key', output, re.I)
        assert not re.search(r'password|token|secret', output, re.I)
```

#### Load Testing
```python
import asyncio
import time
from dataclasses import dataclass
from typing import List
import aiohttp
from tqdm.asyncio import tqdm

@dataclass
class LoadTestResult:
    total_requests: int
    successful_requests: int
    failed_requests: int
    average_latency: float
    percentile_95: float
    percentile_99: float
    requests_per_second: float

class LLMLoadTester:
    def __init__(self, endpoint_url: str, api_key: str):
        self.endpoint_url = endpoint_url
        self.headers = {"Authorization": f"Bearer {api_key}"}
        
    async def single_request(self, session: aiohttp.ClientSession, prompt: str) -> Tuple[float, bool]:
        start_time = time.time()
        try:
            async with session.post(
                self.endpoint_url,
                json={"prompt": prompt, "max_tokens": 100},
                headers=self.headers
            ) as response:
                await response.json()
                success = response.status == 200
        except Exception:
            success = False
        
        latency = time.time() - start_time
        return latency, success
    
    async def run_load_test(
        self, 
        prompts: List[str], 
        concurrent_requests: int = 10,
        total_requests: int = 1000
    ) -> LoadTestResult:
        latencies = []
        successes = 0
        
        start_time = time.time()
        
        async with aiohttp.ClientSession() as session:
            semaphore = asyncio.Semaphore(concurrent_requests)
            
            async def bounded_request(prompt):
                async with semaphore:
                    return await self.single_request(session, prompt)
            
            tasks = []
            for i in range(total_requests):
                prompt = prompts[i % len(prompts)]
                tasks.append(bounded_request(prompt))
            
            results = await tqdm.gather(*tasks, desc="Load testing")
            
        total_time = time.time() - start_time
        
        for latency, success in results:
            latencies.append(latency)
            if success:
                successes += 1
        
        latencies.sort()
        
        return LoadTestResult(
            total_requests=total_requests,
            successful_requests=successes,
            failed_requests=total_requests - successes,
            average_latency=sum(latencies) / len(latencies),
            percentile_95=latencies[int(len(latencies) * 0.95)],
            percentile_99=latencies[int(len(latencies) * 0.99)],
            requests_per_second=total_requests / total_time
        )
```

### Gotchas
1. **Non-determinism**: Same input may produce different outputs
2. **Evaluation Subjectivity**: Human preference varies
3. **Test Data Contamination**: Model may have seen test data
4. **Context Window**: Long prompts may be truncated
5. **Rate Limits**: API constraints during testing
6. **Cost**: Testing can be expensive with large models

### Pros and Cons

| Pros | Cons |
|------|------|
| ✓ Catches regressions | ✗ Hard to define "correct" |
| ✓ Ensures consistency | ✗ Expensive to run |
| ✓ Validates safety | ✗ Non-deterministic outputs |
| ✓ Performance benchmarking | ✗ Requires large test sets |
| ✓ A/B testing insights | ✗ Human evaluation needed |

---

## 6. Fine-tuning {#fine-tuning}

### Overview
Fine-tuning adapts pre-trained language models to specific tasks or domains by training on custom datasets.

### Fine-tuning Approaches

| Method | Description | Use Case | Resource Requirements |
|--------|-------------|----------|----------------------|
| Full Fine-tuning | Update all parameters | Small models, abundant data | High |
| LoRA | Low-rank adaptation | Large models, limited resources | Medium |
| QLoRA | Quantized LoRA | Very large models | Low |
| Prefix Tuning | Trainable prefix tokens | Task-specific adaptation | Low |
| Adapter Layers | Additional trainable layers | Multi-task learning | Medium |

### Data Preparation

```python
import json
import pandas as pd
from typing import List, Dict
from datasets import Dataset, DatasetDict
from transformers import AutoTokenizer

class FineTuningDataProcessor:
    def __init__(self, model_name: str, max_length: int = 512):
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.max_length = max_length
        
    def prepare_chat_dataset(self, conversations: List[List[Dict]]) -> Dataset:
        """Prepare dataset for chat fine-tuning"""
        formatted_data = []
        
        for conversation in conversations:
            text = ""
            for message in conversation:
                role = message["role"]
                content = message["content"]
                
                if role == "system":
                    text += f"System: {content}\n"
                elif role == "user":
                    text += f"Human: {content}\n"
                elif role == "assistant":
                    text += f"Assistant: {content}\n"
            
            formatted_data.append({"text": text.strip()})
        
        dataset = Dataset.from_list(formatted_data)
        
        # Tokenize
        def tokenize_function(examples):
            return self.tokenizer(
                examples["text"],
                truncation=True,
                padding="max_length",
                max_length=self.max_length
            )
        
        tokenized_dataset = dataset.map(tokenize_function, batched=True)
        return tokenized_dataset
    
    def prepare_instruction_dataset(
        self, 
        instructions: List[Dict[str, str]]
    ) -> Dataset:
        """Prepare dataset for instruction fine-tuning"""
        formatted_data = []
        
        for item in instructions:
            prompt = f"""### Instruction:
{item['instruction']}

### Input:
{item.get('input', '')}

### Response:
{item['output']}"""
            
            formatted_data.append({"text": prompt})
        
        return Dataset.from_list(formatted_data)
    
    def validate_dataset(self, dataset: Dataset) -> Dict[str, any]:
        """Validate dataset quality"""
        stats = {
            "total_examples": len(dataset),
            "avg_length": 0,
            "max_length": 0,
            "min_length": float('inf'),
            "empty_examples": 0
        }
        
        lengths = []
        for example in dataset:
            length = len(self.tokenizer.tokenize(example["text"]))
            lengths.append(length)
            
            if length == 0:
                stats["empty_examples"] += 1
            if length > stats["max_length"]:
                stats["max_length"] = length
            if length < stats["min_length"]:
                stats["min_length"] = length
        
        stats["avg_length"] = sum(lengths) / len(lengths)
        stats["length_distribution"] = pd.Series(lengths).describe().to_dict()
        
        return stats
```

### Full Fine-tuning Implementation

```python
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
import torch
from torch.utils.data import DataLoader
import wandb

class FineTuner:
    def __init__(
        self,
        model_name: str,
        output_dir: str,
        learning_rate: float = 5e-5,
        batch_size: int = 4,
        gradient_accumulation_steps: int = 4,
        num_epochs: int = 3,
        warmup_steps: int = 100
    ):
        self.model_name = model_name
        self.output_dir = output_dir
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Load model and tokenizer
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
            device_map="auto"
        )
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        
        # Add padding token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Training arguments
        self.training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=gradient_accumulation_steps,
            warmup_steps=warmup_steps,
            learning_rate=learning_rate,
            fp16=torch.cuda.is_available(),
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=500,
            load_best_model_at_end=True,
            report_to="wandb"
        )
    
    def train(self, train_dataset, eval_dataset=None):
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Initialize trainer
        trainer = Trainer(
            model=self.model,
            args=self.training_args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
            tokenizer=self.tokenizer,
            callbacks=[EarlyStoppingCallback(early_stopping_patience=3)]
        )
        
        # Train
        trainer.train()
        
        # Save model
        trainer.save_model()
        self.tokenizer.save_pretrained(self.output_dir)
        
        return trainer
```

### LoRA Fine-tuning

```python
from peft import LoraConfig, get_peft_model, TaskType, prepare_model_for_kbit_training
from transformers import BitsAndBytesConfig
import bitsandbytes as bnb

class LoRAFineTuner:
    def __init__(
        self,
        model_name: str,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        target_modules: List[str] = None
    ):
        self.model_name = model_name
        
        # LoRA configuration
        self.lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            task_type=TaskType.CAUSAL_LM,
            target_modules=target_modules or ["q_proj", "v_proj", "k_proj", "o_proj"]
        )
        
        # Load model with quantization for QLoRA
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16
        )
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=bnb_config,
            device_map="auto"
        )
        
        # Prepare model for training
        self.model = prepare_model_for_kbit_training(self.model)
        self.model = get_peft_model(self.model, self.lora_config)
        
        # Print trainable parameters
        self.print_trainable_parameters()
    
    def print_trainable_parameters(self):
        trainable_params = 0
        all_param = 0
        for _, param in self.model.named_parameters():
            all_param += param.numel()
            if param.requires_grad:
                trainable_params += param.numel()
        
        print(f"Trainable params: {trainable_params:,} || All params: {all_param:,} || Trainable%: {100 * trainable_params / all_param:.2f}")
    
    def find_optimal_lora_modules(self, sample_data):
        """Find which modules to target for LoRA"""
        import torch.nn as nn
        
        # Get model architecture
        for name, module in self.model.named_modules():
            if isinstance(module, nn.Linear):
                print(f"Linear layer: {name}, shape: {module.weight.shape}")
```

### Custom Loss Functions and Training Objectives

```python
import torch.nn.functional as F

class CustomTrainer(Trainer):
    def __init__(self, alpha=0.5, beta=0.3, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha
        self.beta = beta
    
    def compute_loss(self, model, inputs, return_outputs=False):
        """Custom loss with KL divergence and entropy regularization"""
        labels = inputs.pop("labels")
        
        # Forward pass
        outputs = model(**inputs)
        logits = outputs.logits
        
        # Standard cross-entropy loss
        ce_loss = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            labels.view(-1),
            ignore_index=-100
        )
        
        # KL divergence from a reference distribution
        if hasattr(self, 'reference_model'):
            with torch.no_grad():
                reference_outputs = self.reference_model(**inputs)
                reference_logits = reference_outputs.logits
            
            kl_loss = F.kl_div(
                F.log_softmax(logits, dim=-1),
                F.softmax(reference_logits, dim=-1),
                reduction='batchmean'
            )
        else:
            kl_loss = 0
        
        # Entropy regularization
        probs = F.softmax(logits, dim=-1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=-1).mean()
        
        # Combined loss
        loss = ce_loss + self.alpha * kl_loss - self.beta * entropy
        
        return (loss, outputs) if return_outputs else loss
```

### Evaluation and Metrics

```python
from sklearn.metrics import accuracy_score, f1_score
import numpy as np
from rouge_score import rouge_scorer

class FineTuningEvaluator:
    def __init__(self):
        self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'])
        
    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # Calculate ROUGE scores
        rouge_scores = []
        for pred, label in zip(decoded_preds, decoded_labels):
            scores = self.rouge_scorer.score(label, pred)
            rouge_scores.append({
                'rouge1': scores['rouge1'].fmeasure,
                'rouge2': scores['rouge2'].fmeasure,
                'rougeL': scores['rougeL'].fmeasure
            })
        
        # Average scores
        avg_scores = {
            key: np.mean([score[key] for score in rouge_scores])
            for key in rouge_scores[0].keys()
        }
        
        return avg_scores
    
    def evaluate_on_benchmark(self, model, benchmark_name="mmlu"):
        """Evaluate on standard benchmarks"""
        from lm_eval import evaluator, tasks
        
        results = evaluator.simple_evaluate(
            model=model,
            tasks=[benchmark_name],
            batch_size=8,
            no_cache=True
        )
        
        return results
```

### Distributed Fine-tuning

```python
from accelerate import Accelerator, DeepSpeedPlugin
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP

class DistributedFineTuner:
    def __init__(self, model_name: str, use_deepspeed: bool = True):
        # DeepSpeed configuration
        deepspeed_config = {
            "train_batch_size": 32,
            "gradient_accumulation_steps": 4,
            "fp16": {
                "enabled": True
            },
            "zero_optimization": {
                "stage": 2,
                "offload_optimizer": {
                    "device": "cpu"
                },
                "offload_param": {
                    "device": "cpu"
                }
            }
        }
        
        if use_deepspeed:
            plugin = DeepSpeedPlugin(
                hf_ds_config=deepspeed_config,
                gradient_accumulation_steps=4
            )
            self.accelerator = Accelerator(deepspeed_plugin=plugin)
        else:
            self.accelerator = Accelerator()
        
    def train_distributed(self, model, train_dataloader, optimizer, num_epochs):
        # Prepare for distributed training
        model, optimizer, train_dataloader = self.accelerator.prepare(
            model, optimizer, train_dataloader
        )
        
        model.train()
        
        for epoch in range(num_epochs):
            for batch in train_dataloader:
                with self.accelerator.accumulate(model):
                    outputs = model(**batch)
                    loss = outputs.loss
                    self.accelerator.backward(loss)
                    optimizer.step()
                    optimizer.zero_grad()
            
            # Save checkpoint
            if self.accelerator.is_main_process:
                self.accelerator.save_state(f"checkpoint-epoch-{epoch}")
```

### Gotchas
1. **Catastrophic Forgetting**: Model loses general capabilities
2. **Overfitting**: Too few examples or too many epochs
3. **Learning Rate**: Too high causes instability
4. **Memory Requirements**: Full fine-tuning needs lots of VRAM
5. **Data Quality**: Bad data = bad model
6. **Tokenizer Mismatch**: Must use same tokenizer as pre-training

### Pros and Cons

| Pros | Cons |
|------|------|
| ✓ Task-specific performance | ✗ Requires quality data |
| ✓ Smaller models possible | ✗ Risk of overfitting |
| ✓ Reduced inference cost | ✗ Training infrastructure |
| ✓ Domain adaptation | ✗ Catastrophic forgetting |
| ✓ Control over behavior | ✗ Ongoing maintenance |

---

## 7. Vector Databases {#vector-databases}

### Overview
Vector databases are specialized systems designed to store, index, and query high-dimensional vector embeddings efficiently.

### Popular Vector Databases Comparison

| Database | Index Types | Scalability | Features | Best For |
|----------|------------|-------------|----------|----------|
| Pinecone | LSH, HNSW | Cloud-native | Managed service, filtering | Production SaaS |
| Weaviate | HNSW | Horizontal | GraphQL, modules | Complex queries |
| Chroma | HNSW | Embedded/Server | Simple API | Prototyping |
| Qdrant | HNSW | Distributed | Rust-based, filtering | High performance |
| Milvus | Multiple | Highly scalable | GPU support | Large scale |
| Faiss | Many options | Single node | Facebook library | Research |

### Implementation Examples

#### Basic Vector Store Operations
```python
from typing import List, Dict, Tuple
import numpy as np
from dataclasses import dataclass
import uuid

@dataclass
class Document:
    id: str
    content: str
    embedding: np.ndarray
    metadata: Dict

class VectorStore:
    def __init__(self, dimension: int, similarity_metric: str = "cosine"):
        self.dimension = dimension
        self.similarity_metric = similarity_metric
        self.documents: Dict[str, Document] = {}
        self.embeddings: List[np.ndarray] = []
        self.ids: List[str] = []
    
    def add_documents(self, documents: List[Document]):
        for doc in documents:
            if doc.id in self.documents:
                # Update existing
                idx = self.ids.index(doc.id)
                self.embeddings[idx] = doc.embedding
            else:
                # Add new
                self.documents[doc.id] = doc
                self.embeddings.append(doc.embedding)
                self.ids.append(doc.id)
    
    def similarity_search(
        self, 
        query_embedding: np.ndarray, 
        k: int = 5,
        filter_dict: Dict = None
    ) -> List[Tuple[Document, float]]:
        if not self.embeddings:
            return []
        
        # Calculate similarities
        embeddings_matrix = np.array(self.embeddings)
        
        if self.similarity_metric == "cosine":
            # Normalize vectors
            query_norm = query_embedding / np.linalg.norm(query_embedding)
            embeddings_norm = embeddings_matrix / np.linalg.norm(
                embeddings_matrix, axis=1, keepdims=True
            )
            similarities = np.dot(embeddings_norm, query_norm)
        elif self.similarity_metric == "euclidean":
            similarities = -np.linalg.norm(
                embeddings_matrix - query_embedding, axis=1
            )
        
        # Apply filters
        valid_indices = list(range(len(self.ids)))
        if filter_dict:
            valid_indices = [
                i for i in valid_indices
                if self._matches_filter(self.documents[self.ids[i]], filter_dict)
            ]
        
        # Get top k
        filtered_similarities = [
            (i, similarities[i]) for i in valid_indices
        ]
        filtered_similarities.sort(key=lambda x: x[1], reverse=True)
        
        results = []
        for i, score in filtered_similarities[:k]:
            doc_id = self.ids[i]
            results.append((self.documents[doc_id], float(score)))
        
        return results
    
    def _matches_filter(self, doc: Document, filter_dict: Dict) -> bool:
        for key, value in filter_dict.items():
            if key not in doc.metadata or doc.metadata[key] != value:
                return False
        return True
```

#### Pinecone Implementation
```python
import pinecone
from typing import List, Dict
import hashlib

class PineconeVectorStore:
    def __init__(
        self, 
        api_key: str, 
        environment: str,
        index_name: str,
        dimension: int,
        metric: str = "cosine"
    ):
        pinecone.init(api_key=api_key, environment=environment)
        
        # Create index if doesn't exist
        if index_name not in pinecone.list_indexes():
            pinecone.create_index(
                name=index_name,
                dimension=dimension,
                metric=metric,
                pod_type="p1.x1"
            )
        
        self.index = pinecone.Index(index_name)
        self.dimension = dimension
    
    def upsert_documents(
        self, 
        documents: List[Dict],
        namespace: str = "",
        batch_size: int = 100
    ):
        """Upsert documents with embeddings"""
        vectors = []
        
        for doc in documents:
            # Generate ID if not provided
            doc_id = doc.get("id") or self._generate_id(doc["content"])
            
            vector = {
                "id": doc_id,
                "values": doc["embedding"],
                "metadata": {
                    "content": doc["content"][:1000],  # Metadata size limit
                    **doc.get("metadata", {})
                }
            }
            vectors.append(vector)
            
            # Batch upsert
            if len(vectors) >= batch_size:
                self.index.upsert(vectors=vectors, namespace=namespace)
                vectors = []
        
        # Upsert remaining
        if vectors:
            self.index.upsert(vectors=vectors, namespace=namespace)
    
    def query(
        self,
        query_embedding: List[float],
        top_k: int = 5,
        namespace: str = "",
        filter: Dict = None,
        include_metadata: bool = True
    ):
        """Query similar documents"""
        results = self.index.query(
            vector=query_embedding,
            top_k=top_k,
            namespace=namespace,
            filter=filter,
            include_metadata=include_metadata
        )
        
        return [
            {
                "id": match["id"],
                "score": match["score"],
                "metadata": match.get("metadata", {})
            }
            for match in results["matches"]
        ]
    
    def _generate_id(self, content: str) -> str:
        return hashlib.md5(content.encode()).hexdigest()
    
    def delete(self, ids: List[str], namespace: str = ""):
        """Delete vectors by ID"""
        self.index.delete(ids=ids, namespace=namespace)
    
    def update_metadata(self, id: str, metadata: Dict, namespace: str = ""):
        """Update metadata for a vector"""
        # Fetch existing vector
        fetch_response = self.index.fetch(ids=[id], namespace=namespace)
        if id in fetch_response["vectors"]:
            vector = fetch_response["vectors"][id]
            
            # Update metadata
            vector["metadata"].update(metadata)
            
            # Reinsert
            self.index.upsert(
                vectors=[{
                    "id": id,
                    "values": vector["values"],
                    "metadata": vector["metadata"]
                }],
                namespace=namespace
            )
```

#### Chroma Implementation
```python
import chromadb
from chromadb.config import Settings
from typing import List, Dict, Optional
import uuid

class ChromaVectorStore:
    def __init__(
        self,
        collection_name: str,
        persist_directory: Optional[str] = None,
        embedding_function = None
    ):
        # Initialize client
        if persist_directory:
            self.client = chromadb.PersistentClient(path=persist_directory)
        else:
            self.client = chromadb.Client()
        
        # Get or create collection
        self.collection = self.client.get_or_create_collection(
            name=collection_name,
            embedding_function=embedding_function
        )
    
    def add_documents(
        self,
        documents: List[str],
        embeddings: Optional[List[List[float]]] = None,
        metadatas: Optional[List[Dict]] = None,
        ids: Optional[List[str]] = None
    ):
        """Add documents to collection"""
        if ids is None:
            ids = [str(uuid.uuid4()) for _ in documents]
        
        self.collection.add(
            documents=documents,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids
        )
    
    def query(
        self,
        query_texts: Optional[List[str]] = None,
        query_embeddings: Optional[List[List[float]]] = None,
        n_results: int = 5,
        where: Optional[Dict] = None,
        where_document: Optional[Dict] = None
    ):
        """Query similar documents"""
        results = self.collection.query(
            query_texts=query_texts,
            query_embeddings=query_embeddings,
            n_results=n_results,
            where=where,
            where_document=where_document
        )
        
        return results
    
    def update_document(self, id: str, document: str = None, metadata: Dict = None):
        """Update existing document"""
        update_params = {"ids": [id]}
        if document:
            update_params["documents"] = [document]
        if metadata:
            update_params["metadatas"] = [metadata]
        
        self.collection.update(**update_params)
    
    def delete(self, ids: List[str] = None, where: Dict = None):
        """Delete documents"""
        self.collection.delete(ids=ids, where=where)
```

#### Advanced Indexing Strategies
```python
import faiss
import numpy as np
from typing import List, Tuple
import pickle

class FaissVectorIndex:
    def __init__(
        self, 
        dimension: int,
        index_type: str = "IVF",
        nlist: int = 100,
        nprobe: int = 10
    ):
        self.dimension = dimension
        self.index_type = index_type
        
        if index_type == "Flat":
            # Exact search
            self.index = faiss.IndexFlatL2(dimension)
        elif index_type == "IVF":
            # Inverted file index
            quantizer = faiss.IndexFlatL2(dimension)
            self.index = faiss.IndexIVFFlat(quantizer, dimension, nlist)
            self.nprobe = nprobe
        elif index_type == "HNSW":
            # Hierarchical NSW
            self.index = faiss.IndexHNSWFlat(dimension, 32)
        elif index_type == "LSH":
            # Locality sensitive hashing
            self.index = faiss.IndexLSH(dimension, dimension * 2)
        
        self.id_map = {}
        self.metadata = {}
        self.is_trained = index_type in ["Flat", "HNSW", "LSH"]
    
    def train(self, training_vectors: np.ndarray):
        """Train index if required"""
        if not self.is_trained:
            self.index.train(training_vectors)
            self.is_trained = True
    
    def add_vectors(
        self, 
        vectors: np.ndarray, 
        ids: List[str],
        metadata: List[Dict] = None
    ):
        """Add vectors to index"""
        if not self.is_trained:
            raise ValueError("Index must be trained first")
        
        start_idx = self.index.ntotal
        self.index.add(vectors)
        
        # Map external IDs to internal indices
        for i, ext_id in enumerate(ids):
            internal_idx = start_idx + i
            self.id_map[internal_idx] = ext_id
            if metadata:
                self.metadata[ext_id] = metadata[i]
    
    def search(
        self, 
        query_vectors: np.ndarray, 
        k: int = 5
    ) -> List[List[Tuple[str, float, Dict]]]:
        """Search for similar vectors"""
        if hasattr(self.index, 'nprobe'):
            self.index.nprobe = self.nprobe
        
        distances, indices = self.index.search(query_vectors, k)
        
        results = []
        for i in range(len(query_vectors)):
            query_results = []
            for j in range(k):
                idx = indices[i][j]
                if idx >= 0 and idx in self.id_map:
                    ext_id = self.id_map[idx]
                    distance = distances[i][j]
                    meta = self.metadata.get(ext_id, {})
                    query_results.append((ext_id, float(distance), meta))
            results.append(query_results)
        
        return results
    
    def save(self, path: str):
        """Save index to disk"""
        faiss.write_index(self.index, f"{path}.index")
        with open(f"{path}.meta", "wb") as f:
            pickle.dump({
                "id_map": self.id_map,
                "metadata": self.metadata,
                "dimension": self.dimension,
                "index_type": self.index_type
            }, f)
    
    def load(self, path: str):
        """Load index from disk"""
        self.index = faiss.read_index(f"{path}.index")
        with open(f"{path}.meta", "rb") as f:
            data = pickle.load(f)
            self.id_map = data["id_map"]
            self.metadata = data["metadata"]
            self.dimension = data["dimension"]
            self.index_type = data["index_type"]
```

#### Hybrid Search Implementation
```python
from rank_bm25 import BM25Okapi
import numpy as np
from typing import List, Tuple

class HybridSearchEngine:
    def __init__(self, vector_store, documents: List[str]):
        self.vector_store = vector_store
        self.documents = documents
        
        # Initialize BM25 for keyword search
        tokenized_docs = [doc.lower().split() for doc in documents]
        self.bm25 = BM25Okapi(tokenized_docs)
        
    def search(
        self,
        query: str,
        query_embedding: np.ndarray,
        k: int = 10,
        alpha: float = 0.5  # Weight for vector search
    ) -> List[Tuple[int, float]]:
        """Hybrid search combining vector and keyword search"""
        
        # Vector search
        vector_results = self.vector_store.search(
            query_embedding.reshape(1, -1), 
            k=k*2  # Get more results for reranking
        )[0]
        
        # Keyword search
        query_tokens = query.lower().split()
        bm25_scores = self.bm25.get_scores(query_tokens)
        
        # Get top k from BM25
        bm25_top_k = np.argsort(bm25_scores)[-k*2:][::-1]
        
        # Combine scores
        combined_scores = {}
        
        # Add vector search scores
        for doc_id, distance, _ in vector_results:
            # Convert distance to similarity score
            similarity = 1 / (1 + distance)
            combined_scores[doc_id] = alpha * similarity
        
        # Add BM25 scores
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        for idx in bm25_top_k:
            doc_id = str(idx)
            normalized_score = bm25_scores[idx] / max_bm25
            if doc_id in combined_scores:
                combined_scores[doc_id] += (1 - alpha) * normalized_score
            else:
                combined_scores[doc_id] = (1 - alpha) * normalized_score
        
        # Sort by combined score
        sorted_results = sorted(
            combined_scores.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:k]
        
        return sorted_results
```

### Performance Optimization

```python
class OptimizedVectorStore:
    def __init__(self, dimension: int):
        self.dimension = dimension
        self.shards = {}  # Sharding by metadata
        self.cache = LRUCache(maxsize=1000)
        
    def add_documents_batch(self, documents: List[Document], batch_size: int = 1000):
        """Optimized batch insertion"""
        # Group by shard
        sharded_docs = {}
        for doc in documents:
            shard_key = self._get_shard_key(doc.metadata)
            if shard_key not in sharded_docs:
                sharded_docs[shard_key] = []
            sharded_docs[shard_key].append(doc)
        
        # Process each shard in parallel
        import concurrent.futures
        with concurrent.futures.ThreadPoolExecutor() as executor:
            futures = []
            for shard_key, shard_docs in sharded_docs.items():
                future = executor.submit(
                    self._insert_shard_batch, 
                    shard_key, 
                    shard_docs, 
                    batch_size
                )
                futures.append(future)
            
            # Wait for completion
            concurrent.futures.wait(futures)
    
    def search_with_caching(self, query_embedding: np.ndarray, k: int = 5):
        """Search with result caching"""
        # Generate cache key
        cache_key = hash(query_embedding.tobytes())
        
        # Check cache
        if cache_key in self.cache:
            return self.cache[cache_key]
        
        # Perform search
        results = self._search_all_shards(query_embedding, k)
        
        # Cache results
        self.cache[cache_key] = results
        
        return results
    
    def _get_shard_key(self, metadata: Dict) -> str:
        """Determine shard based on metadata"""
        # Example: shard by date
        if "date" in metadata:
            return metadata["date"][:7]  # YYYY-MM
        return "default"
```

### Gotchas
1. **Dimension Mismatch**: Query and indexed vectors must match
2. **Index Building Time**: Large datasets take time to index
3. **Memory Usage**: Indexes can be memory intensive
4. **Similarity Metrics**: Different metrics for different use cases
5. **Data Drift**: Embeddings change with model updates
6. **Filtering Performance**: Post-filtering can be slow

### Pros and Cons

| Pros | Cons |
|------|------|
| ✓ Fast similarity search | ✗ Memory intensive |
| ✓ Scalable to billions | ✗ Index building time |
| ✓ Multiple index types | ✗ Approximate results |
| ✓ Metadata filtering | ✗ Dimension limitations |
| ✓ Real-time updates | ✗ Infrastructure complexity |

---

## Interview Tips

### Key Concepts to Master
1. **System Design**: How to build scalable AI systems
2. **Trade-offs**: Understand pros/cons of each approach
3. **Implementation Details**: Be ready to code key components
4. **Performance**: Optimization techniques and bottlenecks
5. **Best Practices**: Production considerations

### Common Interview Questions
1. How would you design a RAG system for a customer support chatbot?
2. Explain the difference between fine-tuning and RAG
3. How do you handle WebSocket scaling?
4. What metrics would you use to evaluate an LLM?
5. How do you prevent catastrophic forgetting in fine-tuning?
6. Compare different vector database indexing strategies

### Practical Exercises
1. Implement a simple RAG pipeline
2. Build a WebSocket chat server
3. Create an LLM evaluation framework
4. Fine-tune a small model
5. Design a vector search system

Remember to focus on understanding the concepts deeply rather than memorizing implementations. Good luck with your interviews!
