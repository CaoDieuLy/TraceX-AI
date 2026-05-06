# Trace Service

Service responsible for:
- Select candidate as target for trace
- Build trace (24h window) from selected candidate's tracklets
- Merge video segments into complete trace video
- Handle trace feedback and verification

## API Endpoints

### Select Candidate
```
POST /api/v1/trace/select
Body: { "query_id": UUID, "candidate_id": UUID }
```

### Build Trace
```
POST /api/v1/trace/build
Body: { "query_id": UUID, "candidate_id": UUID, "time_window_hours": 24 }
```

### Get Trace Status
```
GET /api/v1/trace/status/{evidence_id}
```

### Get Trace Timeline
```
GET /api/v1/trace/timeline/{evidence_id}
```

### Submit Feedback
```
POST /api/v1/trace/feedback
Body: { "evidence_id": UUID, "is_correct": bool, "feedback_text": str? }
```

### Get Candidate Detail
```
GET /api/v1/trace/candidate-detail?query_id=UUID&candidate_id=UUID
```

### Continue Trace
```
POST /api/v1/trace/continue
Body: { "query_id": UUID, "candidate_id": UUID, "new_time_window_hours": 24 }
```

## Environment Variables

- `DATABASE_URL`: PostgreSQL connection URL
- `STORAGE_BASE_URL`: Base URL for video storage
- `POSTGRES_HOST`: PostgreSQL host
- `POSTGRES_PORT`: PostgreSQL port
- `POSTGRES_USER`: PostgreSQL user
- `POSTGRES_PASSWORD`: PostgreSQL password
- `POSTGRES_DB`: Database name
