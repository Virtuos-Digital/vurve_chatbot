# 🤖 Chatbot API - Product Recommendation with Vector Search

## 🎯 Overview

This chatbot API uses **LangGraph orchestration** with **HNSW vector search** to provide intelligent product recommendations based on natural language queries.

---

## 📡 API Endpoint

### **POST** `/api/v1/talk`

Chatbot endpoint for natural language interaction with vector-based product search.

---

## 🌐 Local URL

```
http://localhost:8000/api/v1/talk
```

**Production URL:**
```
https://bot.api.vurve.ai/api/v1/talk
```

---

## 📥 Request Format

### Headers
```
Content-Type: application/json
```

### Body (JSON)
```json
{
    "user_uuid": "e2ae5050-44b7-4e44-badc-7c48c799ac96",
    "msg": "i want gift for gf"
}
```

| Field | Type | Required | Description |
|-------|------|----------|-------------|
| `user_uuid` | string | ✅ Yes | Unique identifier for the user |
| `msg` | string | ✅ Yes | User's message/query |

---

## 📤 Response Format

### Success Response (200 OK)
```json
{
    "user_uuid": "e2ae5050-44b7-4e44-badc-7c48c799ac96",
    "bot_message": "I found these products that might interest you: 8522431103196, 8064257851612, 7763737510108, 8522431299804, 8522430185692"
}
```

| Field | Type | Description |
|-------|------|-------------|
| `user_uuid` | string | User identifier (echoed back) |
| `bot_message` | string | Bot's response with product IDs |

### Error Response (400/500)
```json
{
    "user_uuid": "e2ae5050-44b7-4e44-badc-7c48c799ac96",
    "error": "Error message here"
}
```

---

## 🧪 Postman Examples

### Example 1: Product Search
**Request:**
```json
{
    "user_uuid": "user-123",
    "msg": "red dress for party"
}
```

**Response (200 OK):**
```json
{
    "user_uuid": "user-123",
    "bot_message": "I found these products that might interest you: 8522431103196, 8064257851612, 7763737510108"
}
```

### Example 2: Gift Recommendation
**Request:**
```json
{
    "user_uuid": "user-456",
    "msg": "gift for girlfriend birthday"
}
```

**Response:**
```json
{
    "user_uuid": "user-456",
    "bot_message": "I found these products that might interest you: 7697784668380, 8358100369628, 7754616504540"
}
```

### Example 3: Order Tracking
**Request:**
```json
{
    "user_uuid": "user-789",
    "msg": "where is my order?"
}
```

**Response:**
```json
{
    "user_uuid": "user-789",
    "bot_message": "Your order is being processed."
}
```

---

## 🚀 Starting the Server

### Development Mode (with auto-reload)
```bash
cd /Users/virtuos/Desktop/vurve/bot_api_vurve_ai/chatbot
uvicorn src.app.api.v1.chat:app --reload --host 0.0.0.0 --port 8000
```

### Production Mode
```bash
cd /Users/virtuos/Desktop/vurve/bot_api_vurve_ai/chatbot
uvicorn src.app.api.v1.chat:app --host 0.0.0.0 --port 8000 --workers 4
```

### Alternative (using Python directly)
```bash
cd /Users/virtuos/Desktop/vurve/bot_api_vurve_ai/chatbot
python3 src/app/api/v1/chat.py
```

---

## 🔧 Configuration

### Environment Variables (.env)
```bash
# API Configuration
TEST_VARIABLE=true
REDIS_HOST=localhost
REDIS_PORT=6379
INTENT_CLASSIFIER_API_URL=https://infer.bot.vurve.ai/predict
EMBED_API_URL=https://gift-intelligence.vurve.ai/api/v1/embed

# Vector Database Configuration
VECTOR_DB_HOST=localhost
VECTOR_DB_PORT=5430
VECTOR_DB_NAME=postres2
VECTOR_DB_USER=postgres
VECTOR_DB_PASSWORD=2001
```

---

## 🏗️ Architecture

```
User Input: "red dress for party"
          ↓
[Intent Classifier API]
Classifies as: "Product recommendation"
          ↓
[Orchestrator - LangGraph]
Routes to: product_recommender node
          ↓
[Vector Search]
1. Convert text to 384D embedding
   → API: https://gift-intelligence.vurve.ai/api/v1/embed
2. Search similar products using HNSW
   → DB: localhost:5430/postres2
   → Table: embedding_minilm
3. Return top 5 product IDs
          ↓
[Response Generator]
Formats bot message with product IDs
          ↓
[FastAPI Response]
Returns JSON to client
```

---

## 🔍 How Product Search Works

### 1. **Text to Vector Conversion**
User query → Transformer API → 384-dimensional embedding

**API:** `https://gift-intelligence.vurve.ai/api/v1/embed`

**Request:**
```json
{
  "texts": ["red dress for party"],
  "normalize": true
}
```

**Response:**
```json
{
  "embeddings": [{
    "text": "red dress for party",
    "embedding": [0.051, -0.023, ...],  // 384 dimensions
    "tokens": null
  }],
  "dimension": 384,
  "model": "sentence-transformers/all-MiniLM-L6-v2"
}
```

### 2. **Vector Similarity Search (HNSW)**
- Database: PostgreSQL 18 with pgvector extension
- Index: HNSW (Hierarchical Navigable Small World)
- Distance metric: Cosine distance
- Returns: Top 5 most similar products

**SQL Query:**
```sql
SELECT shopify_product_id
FROM embedding_minilm
ORDER BY embedding <=> [query_vector]::vector
LIMIT 5
```

### 3. **Response Generation**
Product IDs are formatted into a user-friendly message:
```
"I found these products that might interest you: [ID1], [ID2], [ID3], [ID4], [ID5]"
```

---

## 🎭 Intent Classification

The orchestrator classifies user messages into these categories:

| Intent | Description | Example |
|--------|-------------|---------|
| **Product recommendation** | User wants product suggestions | "red dress", "gift for mom" |
| **Order tracking** | User asks about order status | "where is my order?" |
| **General conversation** | General questions | "hello", "help me" |
| **Escalation** | Complex issues requiring support | "speak to agent" |

---

## 📊 Database Schema

### Table: `embedding_minilm` (postres2 database)
```sql
Column              | Type           | Description
--------------------|----------------|---------------------------
shopify_product_id  | VARCHAR(100)   | PRIMARY KEY
product_text        | TEXT           | Product description
embedding           | VECTOR(384)    | 384D vector embedding
created_at          | TIMESTAMP      | Creation timestamp
updated_at          | TIMESTAMP      | Last update timestamp
```

**Indexes:**
- PRIMARY KEY on `shopify_product_id`
- HNSW index on `embedding` (m=16, ef_construction=64)

**Current data:** 550+ product embeddings

---

## 📁 Project Structure

```
bot_api_vurve_ai/chatbot/
├── .env                          # Environment configuration
├── requirements.txt              # Python dependencies
└── src/
    ├── __init__.py
    ├── api_types/               # Type definitions
    │   ├── __init__.py
    │   └── api.py              # ChatRequest, ChatResponse
    ├── app/                     # FastAPI application
    │   ├── __init__.py
    │   └── api/
    │       └── v1/
    │           └── chat.py     # Main API endpoint
    ├── logger.py               # Logging utility
    ├── orchestrator.py         # LangGraph workflow
    ├── redis_utils.py          # Redis cache operations
    └── vector_search.py        # Vector search logic
```

---

## 🧪 Testing

### Quick Test - Vector Search
```bash
cd /Users/virtuos/Desktop/vurve/bot_api_vurve_ai/chatbot
PYTHONPATH=$PWD python3 src/vector_search.py
```

**Expected Output:**
```
SUCCESS: Generated embedding for text (dimension: 384)
SUCCESS: Found 5 matching products for query: 'red dress for party'
Product IDs: ['8522431103196', '8064257851612', '7763737510108', ...]
```

### Test with cURL
```bash
curl -X POST http://localhost:8000/api/v1/talk \
  -H "Content-Type: application/json" \
  -d '{
    "user_uuid": "test-user-123",
    "msg": "gift for girlfriend"
  }'
```

### Test with Postman

1. **Open Postman**
2. **Create New Request:**
   - Method: `POST`
   - URL: `http://localhost:8000/api/v1/talk`
   - Headers: `Content-Type: application/json`
   - Body (raw JSON):
     ```json
     {
       "user_uuid": "e2ae5050-44b7-4e44-badc-7c48c799ac96",
       "msg": "red dress for party"
     }
     ```
3. **Click Send**
4. **Expected Response:**
   ```json
   {
     "user_uuid": "e2ae5050-44b7-4e44-badc-7c48c799ac96",
     "bot_message": "I found these products..."
   }
   ```

---

## ⚡ Performance

- **Embedding generation:** ~50ms per query
- **Vector search (HNSW):** ~5-10ms for 550 embeddings
- **Total response time:** ~100-200ms
- **Concurrent requests:** Supports multiple simultaneous users

---

## 🔗 Dependencies

### Core Libraries
- **FastAPI** - Web framework
- **LangGraph** - Workflow orchestration
- **asyncpg** - PostgreSQL async driver
- **httpx** - HTTP client for API calls
- **Redis** - Chat history storage
- **python-dotenv** - Environment management

### Vector Search
- **PostgreSQL 18** with pgvector extension
- **HNSW indexing** for fast similarity search
- **sentence-transformers/all-MiniLM-L6-v2** model (384D)

---

## 🛠️ Troubleshooting

### Server won't start
```bash
# Check if port 8000 is in use
lsof -i :8000

# Kill process if needed
kill -9 <PID>

# Or use different port
uvicorn src.app.api.v1.chat:app --reload --port 8001
```

### Database connection error
```bash
# Verify PostgreSQL is running
PGPASSWORD=2001 psql -h localhost -p 5430 -U postgres -d postres2 -c "SELECT 1"

# Check embeddings count
PGPASSWORD=2001 psql -h localhost -p 5430 -U postgres -d postres2 -c "SELECT COUNT(*) FROM embedding_minilm"
```

### Redis connection error
```bash
# Check Redis is running
redis-cli ping  # Should return: PONG

# Start Redis if needed
redis-server
```

### Import errors
```bash
# Make sure you're in the chatbot directory
cd /Users/virtuos/Desktop/vurve/bot_api_vurve_ai/chatbot

# Run with PYTHONPATH
PYTHONPATH=$PWD uvicorn src.app.api.v1.chat:app --reload
```

---

## 📝 API Response Codes

| Code | Status | Description |
|------|--------|-------------|
| 200 | OK | Request successful |
| 400 | Bad Request | Invalid request format or classification error |
| 500 | Internal Server Error | Server-side error (workflow/database) |

---

## 🎯 Summary

**To use the product search API:**

1. ✅ Start the server: `uvicorn src.app.api.v1.chat:app --reload --host 0.0.0.0 --port 8000`
2. ✅ Open Postman
3. ✅ POST to: `http://localhost:8000/api/v1/talk`
4. ✅ Body format:
   ```json
   {
     "user_uuid": "your-uuid-here",
     "msg": "your search query"
   }
   ```
5. ✅ Response will contain array of product IDs in the bot_message

**Output:** Array of top 5 shopify_product_ids based on vector similarity!

---

## 📞 Support

For issues or questions:
- Check logs for detailed error messages
- Verify all environment variables are set correctly
- Ensure PostgreSQL and Redis are running
- Test vector search standalone first (`python3 src/vector_search.py`)

---

**Last Updated:** March 16, 2026  
**Version:** 1.0.0  
**Status:** ✅ Production Ready
