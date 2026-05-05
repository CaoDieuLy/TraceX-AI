# infra/

Infrastructure configuration for Docker deployment.

## Structure

```
infra/
├── docker-compose.yml     # Main stack definition
└── postgres/
    └── init/
        └── 01-init.sql    # Database initialization
```

## Services

| Service | Container | Port | Description |
|---------|-----------|------|-------------|
| postgres | mcpt-postgres | 5432 | PostgreSQL database |
| metadata-service | mcpt-metadata-service | 8001 | Metadata & API service |
| ai-service | mcpt-ai-service | 8002 | AI inference (GPU) |
| frontend | mcpt-frontend | 3000 | Next.js web app |

## Quick Start

```bash
# Start all services
docker compose -f infra/docker-compose.yml up -d --build

# View status
docker compose -f infra/docker-compose.yml ps

# View logs
docker compose -f infra/docker-compose.yml logs -f

# Stop all services
docker compose -f infra/docker-compose.yml down
```

## Environment Variables

Create a `.env` file or set these variables:

| Variable | Default | Description |
|----------|---------|-------------|
| POSTGRES_DATABASE | mcpt | Database name |
| POSTGRES_USER | mcpt_user | Database user |
| POSTGRES_PASSWORD | mcpt | Database password |
| JWT_SECRET_KEY | change-me | JWT signing key |
| METADATA_DOMAIN | metadata.local | Metadata service domain |
| APP_DOMAIN | tracex-ai.smartnovi.tech | Frontend domain |
