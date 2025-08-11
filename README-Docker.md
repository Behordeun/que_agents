# Docker Setup for Que Agents

This document provides instructions for running Que Agents using Docker and Docker Compose.

## Prerequisites

- Docker Engine 20.10+
- Docker Compose 2.0+

## Quick Start

### 1. Environment Setup

Copy the environment example file and configure your API keys:

```bash
cp .env.example .env
```

Edit `.env` and add your API keys:
- `OPENAI_API_KEY`
- `GROQ_API_KEY`
- `ANTHROPIC_API_KEY`

### 2. Production Deployment

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f que_agents

# Stop services
docker-compose down
```

### 3. Development Mode

```bash
# Start development environment with hot reload
docker-compose -f docker-compose.dev.yaml up -d

# View logs
docker-compose -f docker-compose.dev.yaml logs -f que_agents_dev
```

## Services

### Production Stack
- **que_agents**: Main application (port 8000)
- **postgres**: PostgreSQL database (port 5432)
- **redis**: Redis cache (port 6379)
- **nginx**: Reverse proxy (port 80)

### Development Stack
- **que_agents_dev**: Development app with hot reload (port 8001)
- **postgres**: PostgreSQL database (port 5433)
- **redis_dev**: Redis cache (port 6380)

## Access Points

### Production
- API: http://localhost/api/
- Documentation: http://localhost/docs
- Web Interface: http://localhost/

### Development
- API: http://localhost:8001/
- Documentation: http://localhost:8001/docs
- Database: localhost:5433

## Useful Commands

```bash
# View running containers
docker-compose ps

# Execute commands in container
docker-compose exec que_agents bash

# View application logs
docker-compose logs -f que_agents

# Restart specific service
docker-compose restart que_agents

# Rebuild and restart
docker-compose up -d --build

# Clean up everything
docker-compose down -v --remove-orphans
```

## Database Management

```bash
# Run migrations
docker-compose exec que_agents python -m alembic upgrade head

# Access PostgreSQL
docker-compose exec postgres psql -U postgres -d que_agents

# Backup database
docker-compose exec postgres pg_dump -U postgres que_agents > backup.sql
```

## Troubleshooting

### Common Issues

1. **Port conflicts**: Change ports in docker-compose.yaml if needed
2. **Permission issues**: Ensure proper file permissions
3. **API key errors**: Verify environment variables are set correctly

### Health Checks

```bash
# Check application health
curl http://localhost:8000/health

# Check database connection
docker-compose exec postgres pg_isready -U postgres
```

### Logs and Debugging

```bash
# Application logs
docker-compose logs que_agents

# Database logs
docker-compose logs postgres

# All services logs
docker-compose logs
```