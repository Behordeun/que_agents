# Que Agents - Multi-Agent AI Platform

A comprehensive multi-agent AI platform featuring four specialized agents for customer support, marketing automation, personal assistance, and financial trading.

## 🚀 Features

### 🎯 Four Specialized Agents

1. **Customer Support Agent** - Intelligent customer service with sentiment analysis and escalation
2. **Marketing Agent** - Autonomous campaign management and content generation  
3. **Personal Virtual Assistant** - Smart home control, reminders, and personal productivity
4. **Financial Trading Bot** - Automated market analysis and trading decisions

### 🏗️ Platform Capabilities

- **Multi-LLM Support**: OpenAI, Anthropic, Groq, Azure OpenAI, and local models
- **Conversation Memory**: Context-aware interactions across sessions
- **Knowledge Base**: Semantic search with ChromaDB integration
- **RESTful API**: FastAPI with automatic documentation
- **Database Integration**: SQLAlchemy with SQLite/PostgreSQL support
- **Configuration Management**: YAML-based configuration system

## 📋 Quick Start

### Prerequisites

- Python 3.11+
- Git

### Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/behordeun/que_agents.git
   cd que_agents
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure the system**
   ```bash
   # Copy example configurations
   cp configs/database_config_example.yaml configs/database_config.yaml
   cp configs/llm_config_example.yaml configs/llm_config.yaml
   
   # Edit configurations with your settings
   nano configs/llm_config.yaml  # Add your API keys
   ```

4. **Initialize the database**
   ```bash
   python src/que_agents/core/database.py
   ```

5. **Start the API server**
   ```bash
   uvicorn src.que_agents.api.main:app --host 0.0.0.0 --port 8000 --reload
   ```

6. **Access the API documentation**
   - Open http://localhost:8000/docs for interactive API documentation
   - Health check: http://localhost:8000/health

## 🤖 Agent Capabilities

### Customer Support Agent
- **Intent Recognition**: Automatically categorizes customer inquiries
- **Sentiment Analysis**: Detects customer emotions and adjusts responses
- **Knowledge Base Integration**: Searches relevant support articles
- **Escalation Logic**: Identifies when human intervention is needed
- **Conversation Memory**: Maintains context across interactions

**API Endpoints:**
- `POST /api/v1/customer-support/chat` - Handle customer inquiries
- `GET /api/v1/customer-support/customer/{id}` - Get customer context

### Marketing Agent
- **Campaign Planning**: Creates comprehensive marketing strategies
- **Content Generation**: Produces platform-specific content
- **Audience Segmentation**: Identifies target demographics
- **Performance Optimization**: Adjusts campaigns based on metrics
- **Multi-Channel Support**: LinkedIn, email, social media, ads

**API Endpoints:**
- `POST /api/v1/marketing/campaign/create` - Create marketing campaigns
- `POST /api/v1/marketing/content/generate` - Generate content

### Personal Virtual Assistant
- **Smart Home Control**: Manage lights, thermostats, security systems
- **Reminder Management**: Set, list, and cancel reminders
- **Weather Information**: Current conditions and forecasts
- **General Queries**: Answer questions using knowledge base
- **Personalized Recommendations**: Restaurants, movies, books

**API Endpoints:**
- `POST /api/v1/pva/chat` - Chat with personal assistant
- `GET /api/v1/pva/user/{id}/reminders` - Get user reminders
- `GET /api/v1/pva/user/{id}/devices` - Get smart devices

### Financial Trading Bot
- **Market Analysis**: Technical indicators (RSI, MACD, moving averages)
- **Risk Management**: Position sizing, stop-loss, diversification
- **Strategy Execution**: Momentum, mean reversion, arbitrage
- **Portfolio Management**: Real-time tracking and optimization
- **Performance Reporting**: Detailed analytics and metrics

**API Endpoints:**
- `POST /api/v1/trading/analyze` - Analyze market and make decisions
- `POST /api/v1/trading/cycle` - Run complete trading cycle
- `GET /api/v1/trading/portfolio` - Get portfolio status
- `GET /api/v1/trading/performance` - Get performance report
- `GET /api/v1/trading/market/{symbol}` - Get market data

## 🔧 Configuration

### LLM Configuration
```yaml
llm:
  default_provider: "openai"
  providers:
    openai:
      api_key: "your-openai-api-key"
      model: "gpt-4"
      temperature: 0.7
```

### Agent Configuration
```yaml
agents:
  customer_support:
    model: "gpt-4"
    temperature: 0.5
    memory_window: 10
  
  personal_virtual_assistant:
    model: "gpt-4"
    temperature: 0.5
    memory_window: 15
    
  financial_trading_bot:
    model: "gpt-4"
    temperature: 0.3
    risk_tolerance: 0.1
```

## 🧪 Testing

Run the comprehensive test suite:

```bash
python test_agents_simple.py
```

This validates:
- ✅ Agent module imports
- ✅ Database model creation
- ✅ API structure and endpoints
- ✅ System integration

## 📊 Database Schema

The platform uses a comprehensive database schema supporting all four agents:

### Core Tables
- `Customer` - Customer information and context
- `UserPreferences` - Personal assistant user settings  
- `Portfolio` - Trading bot portfolio management
- `MarketingCampaign` - Marketing campaign tracking

### Interaction Tables
- `CustomerInteraction` - Support conversations
- `PVAInteraction` - Personal assistant sessions
- `TradeLog` - Trading decisions and outcomes
- `MarketingPost` - Generated content and performance

## 🔐 Authentication

All API endpoints require authentication using Bearer tokens:

```bash
curl -H "Authorization: Bearer demo-token-123" \
     http://localhost:8000/api/v1/customer-support/chat
```

## 📈 Monitoring

### Health Checks
- `GET /health` - Overall system health
- `GET /debug` - Detailed system information

### Performance Metrics
- Response times for each agent
- Database query performance
- Memory usage and conversation management
- Trading bot performance analytics

## 🚀 Deployment

### Development
```bash
uvicorn src.que_agents.api.main:app --reload
```

### Production
```bash
gunicorn src.que_agents.api.main:app -w 4 -k uvicorn.workers.UvicornWorker
```

### Docker (Optional)
```dockerfile
FROM python:3.11-slim
COPY . /app
WORKDIR /app
RUN pip install -r requirements.txt
CMD ["uvicorn", "src.que_agents.api.main:app", "--host", "0.0.0.0"]
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **LangChain** for agent orchestration framework
- **FastAPI** for high-performance web framework
- **SQLAlchemy** for database ORM
- **OpenAI, Anthropic, Groq** for LLM services

## 📞 Support

For questions and support:
- 📧 Email: abiodun.msulaiman@gmail.com
- 🐛 Issues: [GitHub Issues](https://github.com/behordeun/que_agents/issues)
- 📖 Documentation: [API Docs](http://localhost:8000/docs)

---

**Built with ❤️ by the Que Agents team**

