# Agentic AI System - Customer Support & Marketing Agents

A proof-of-concept implementation of autonomous AI agents for customer support and marketing automation, built with LangChain, PostgreSQL, and FastAPI.

## 🚀 Features

- **Customer Support Agent**: AI-powered customer service with context awareness, sentiment analysis, and intelligent escalation
- **Marketing Agent**: Autonomous marketing campaign creation, content generation, and performance analysis
- **Multi-Modal Knowledge Base**: Integration of structured (PostgreSQL), semi-structured (CSV), and unstructured (Markdown) data
- **FastAPI Integration**: RESTful API for seamless integration with external systems
- **Web Interface**: Interactive demonstration platform with real-time agent testing
- **Comprehensive Testing**: Full integration test suite with performance validation

## 🏗️ Architecture

```mermaid
graph TD
    A["🌐 Web Interface<br/>(HTML/JS/CSS)"] <--> B["⚡ FastAPI<br/>Gateway"]
    B <--> C["🤖 AI Agents<br/>(LangChain)"]
    B --> D["🗄️ PostgreSQL<br/>(Vector Store)"]
    C --> E["📚 Knowledge Base<br/>(Vector Store)"]

    classDef frontend fill:#e1f5fe
    classDef api fill:#f3e5f5
    classDef agents fill:#e8f5e8
    classDef storage fill:#fff3e0

    class A frontend
    class B api
    class C agents
    class D,E storage
```

## 🛠️ Technology Stack

- **Backend**: Python 3.11, FastAPI, LangChain
- **Database**: PostgreSQL 14, SQLAlchemy ORM
- **AI/ML**: OpenAI GPT-4, Sentence Transformers
- **Frontend**: HTML5, CSS3, Vanilla JavaScript
- **Testing**: Pytest, Requests
- **Deployment**: Uvicorn ASGI Server

## 📋 Prerequisites

- Python 3.11 or higher
- PostgreSQL 14 or higher
- OpenAI API key
- 4GB+ RAM (8GB recommended for production)

## 🚀 Quick Start

### 1. Clone and Setup

```bash
# Clone the repository (if using git)
git clone [https://github.com/Behordeun/que_agents.git](https://github.com/Behordeun/que_agents.git)
cd que_agents

# Create virtual environment
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Database Setup

```bash
# Install PostgreSQL (Ubuntu/Debian)
sudo apt update
sudo apt install postgresql postgresql-contrib

# Start PostgreSQL service
sudo systemctl start postgresql
sudo systemctl enable postgresql

# Create database and user
sudo -u postgres createdb agentic_ai
sudo -u postgres psql -c "CREATE USER agentic_user WITH PASSWORD 'agentic_pass';"
sudo -u postgres psql -c "GRANT ALL PRIVILEGES ON DATABASE agentic_ai TO agentic_user;"
```

### 3. Environment Configuration

Configurations are now managed via YAML files in the `configs/` directory. You can modify these files to suit your environment.

- `configs/agent_config.yaml`: Agent-specific settings (model names, temperatures).
- `configs/api_config.yaml`: FastAPI application settings (host, port, CORS, authentication token).
- `configs/database_config.yaml`: PostgreSQL connection details.
- `configs/knowledge_base_config.yaml`: Knowledge base settings (e.g., SQLite DB path).

**Example `configs/database_config.yaml`:**

```yaml
database:
  url: postgresql://agentic_user:agentic_pass@localhost/agentic_ai
  echo: false
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
  pool_recycle: 3600
```

**Note**: For production, it is highly recommended to use a secure secrets management system for sensitive information like database credentials and API tokens, rather than hardcoding them in YAML files.

### 4. Initialize Database & Knowledge Base

```bash
# Create database tables
python -m src.que_agents.core.database.

# Populate with sample data
python -m src.que_agents.utils.data_populator

# Initialize knowledge base (loads data from PostgreSQL and files into SQLite FTS)
python -m src.que_agents.knowledge_base.kb_manager
```

### 5. Start Services

```bash
# Terminal 1: Start API server
python -m src.que_agents.api.main

# Terminal 2: Start web interface
cd template
python3 -m http.server 8080
```

### 6. Access the System

- **Web Interface**: [http://localhost:8080](http://localhost:8080)
- **API Documentation**: [http://localhost:8000/docs](http://localhost:8000/docs)
- **Health Check**: [http://localhost:8000/health](http://localhost:8000/health)

## 🧪 Testing

### Run Integration Tests

```bash
# Run comprehensive test suite
python -m src.que_agents.utils.tests.integration_test

# Run API tests
python -m src.que_agents.utils.tests.test_api
```

### Test Individual Agents

```bash
# Test Customer Support Agent
python3 src.que_agents.agents.customer_support_agent

# Test Marketing Agent
python3 src.que_agents.agents.marketing_agent
```

## 📚 API Usage Examples

### Customer Support

```bash
# Send customer support request
curl -X POST "http://localhost:8000/api/v1/customer-support/chat" \
  -H "Authorization: Bearer demo-token-123" \
  -H "Content-Type: application/json" \
  -d '{
    "customer_id": 1,
    "message": "I cannot access my account"
  }'
```

### Marketing Campaign

```bash
# Create marketing campaign
curl -X POST "http://localhost:8000/api/v1/marketing/campaign/create" \
  -H "Authorization: Bearer demo-token-123" \
  -H "Content-Type: application/json" \
  -d '{
    "campaign_type": "product_launch",
    "target_audience": "tech professionals",
    "budget": 10000.0,
    "duration_days": 30,
    "goals": ["increase awareness", "generate leads"],
    "channels": ["linkedin", "twitter"],
    "content_requirements": ["social_media", "email"]
  }'
```

### Knowledge Base Search

```bash
# Search knowledge base
curl -X GET "http://localhost:8000/api/v1/knowledge-base/search?query=password%20reset&limit=5" \
  -H "Authorization: Bearer demo-token-123"
```

## 🎯 Agent Capabilities

### Customer Support Agent

- **Natural Language Understanding**: Processes complex customer queries
- **Context Awareness**: Retrieves customer history and account information
- **Sentiment Analysis**: Detects customer emotional state
- **Knowledge Integration**: Searches FAQs, documentation, and policies
- **Escalation Logic**: Determines when human intervention is needed
- **Confidence Scoring**: Provides response confidence metrics

### Marketing Agent

- **Campaign Strategy**: Develops comprehensive marketing strategies
- **Content Generation**: Creates platform-specific marketing content
- **Audience Segmentation**: Analyzes and targets specific customer segments
- **Performance Analysis**: Evaluates campaign effectiveness
- **Multi-Platform Optimization**: Adapts content for different channels
- **Budget Allocation**: Optimizes spending across marketing channels

## 📊 Data Sources

The system integrates three types of data as specified:

### Structured Data (PostgreSQL)

- Customer information and account details
- Campaign metrics and performance data
- System configuration and user management
- Interaction history and ticket tracking

### Semi-Structured Data (CSV)

- Customer feedback and survey responses
- Campaign performance analytics
- Market research data
- A/B testing results

### Unstructured Data (Documents)

- FAQ databases and help articles
- Product documentation and manuals
- Company policies and procedures
- Troubleshooting guides and tutorials

### Configuration

All critical system settings are now managed through dedicated YAML configuration files located in the `configs/` directory. This approach centralizes configuration, making it easier to adapt the system to different environments (development, staging, production) without modifying the core codebase.

- `configs/agent_config.yaml`: Agent-specific settings (model names, temperatures).
- `configs/api_config.yaml`: FastAPI application settings (host, port, CORS, authentication token).
- `configs/database_config.yaml`: PostgreSQL connection details.
- `configs/knowledge_base_config.yaml`: Knowledge base settings (e.g., SQLite DB path).

**Example `configs/database_config.yaml`:**

```yaml
database:
  url: postgresql://agentic_user:agentic_pass@localhost/agentic_ai
  echo: false
  pool_size: 10
  max_overflow: 20
  pool_timeout: 30
  pool_recycle: 3600
```

**Note**: For production, it is highly recommended to use a secure secrets management system for sensitive information like database credentials and API tokens, rather than hardcoding them in YAML files.`

## 🚀 Deployment

### Production Deployment

```bash
# Install production dependencies
pip install gunicorn

# Run with Gunicorn
gunicorn -w 4 -k uvicorn.workers.UvicornWorker src.que_agents.api.main:app --bind 0.0.0.0:8000

# Or use Docker (if Dockerfile provided)
docker build -t agentic-ai .
docker run -p 8000:8000 -e OPENAI_API_KEY=your-key agentic-ai
```

### Environment-Specific Configuration

```bash
# Development
export ENVIRONMENT=development
export DEBUG=true

# Production
export ENVIRONMENT=production
export DEBUG=false
export DATABASE_URL=postgresql://prod_user:prod_pass@prod_host/prod_db
```

## 📈 Performance

### Benchmarks

- **Health Check**: < 10ms response time
- **Knowledge Base Search**: < 100ms average
- **Customer Support Response**: 3-7 seconds
- **Marketing Campaign Creation**: 10-15 seconds
- **Concurrent Users**: 50+ without degradation

### Optimization Tips

1. **Database Indexing**: Ensure proper indexes on frequently queried fields
2. **Connection Pooling**: Use connection pooling for database connections
3. **Caching**: Implement Redis caching for frequent knowledge base queries
4. **Load Balancing**: Use multiple API server instances for high traffic

## 🔒 Security

### Authentication

The system uses Bearer token authentication:

```bash
# Include token in all API requests
Authorization: Bearer demo-token-123
```

### Security Features

- Input validation and sanitization
- SQL injection prevention
- CORS configuration
- Rate limiting
- Audit logging
- Encrypted database connections

## 🐛 Troubleshooting

### Common Issues

#### Database Connection Error

```bash
# Check PostgreSQL status
sudo systemctl status postgresql

# Verify database exists
sudo -u postgres psql -l | grep agentic_ai
```

#### OpenAI API Error

```bash
# Verify API key
echo $OPENAI_API_KEY

# Test API connection
curl -H "Authorization: Bearer $OPENAI_API_KEY" https://api.openai.com/v1/models
```

#### Import Errors

```bash
# Reinstall dependencies
pip install --force-reinstall -r requirements.txt

# Check Python version
python3 --version  # Should be 3.11+
```

### Debug Mode

```bash
# Enable debug logging
export LOG_LEVEL=DEBUG
python -m src.que_agents.api.main.py
```

## 📝 File Structure

```plain-text
que_agents/
├── configs
│   ├── agent_config.yaml
│   ├── api_config.yaml
│   ├── database_config.yaml
│   └── knowledge_base_config.yaml
├── data
│   ├── semi_structured
│   │   ├── campaign_performance.csv
│   │   └── customer_feedback.csv
│   ├── structured
│   │   ├── api_endpoints.json
│   │   ├── integration_test_report.json
│   │   └── sample_responses.json
│   └── unstructured
│       ├── company_policies.md
│       ├── faq_database.md
│       └── product_documentation.md
├── docs
│   ├── Agentic AI System - Customer Support & Marketing Agents.md
│   ├── Agentic AI System - Project Summary.md
│   ├── Agentic AI System: Technical Documentation.md
│   └── framework_research.md
├── Makefile
├── pyproject.toml
├── README.md
├── requirements.txt
├── src
│   └── que_agents
│       ├── __init__.py
│       ├── agents
│       │   ├── __init__.py
│       │   ├── customer_support_agent.py
│       │   └── marketing_agent.py
│       ├── api
│       │   ├── __init__.py
│       │   └── main.py
│       ├── core
│       │   ├── __init__.py
│       │   └── database.py
│       ├── knowledge_base
│       │   ├── __init__.py
│       │   └── kb_manager.py
│       ├── template
│       │   └── index.html
│       └── utils
│           ├── __init__.py
│           └── data_populator.py
└── tests
    ├── __init__.py
    └── integration_test.py
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](../LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain**: For providing the agent framework
- **OpenAI**: For GPT models and API
- **FastAPI**: For the excellent web framework
- **PostgreSQL**: For robust database capabilities

## 📞 Support

For questions or support:

1. Check the [Technical Documentation](./Technical%20Documentation.md)
2. Review the [API Documentation](http://localhost:8000/docs)
3. Run the integration tests to verify setup
4. Check the troubleshooting section above

---
