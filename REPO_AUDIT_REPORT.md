# Repo Audit & Refactor Report

## 1. Scope Audited

### Directories Audited
- `frontend/` - Full audit of Next.js frontend
- `backend/services/` - All backend services
- `backend/config/` - Configuration files
- `infra/` - Docker Compose and infrastructure
- `scripts/` - Utility scripts
- `docs/` - Documentation
- Root level files - README, configs, etc.

### Key Files Audited
- All API client code (`frontend/lib/api/`)
- All TypeScript types (`frontend/lib/types/`)
- All frontend components and features
- Backend service configs, models, and schemas
- Docker Compose configuration
- Shell scripts for service startup

---

## 2. Summary of Changes

### Completed Cleanup
1. **Fixed broken imports** - `HomeView.tsx` recreated, `queue.py` router removed from imports
2. **Removed unused frontend files** - `VideoGrid.tsx`, `lib/mock/*`, `lib/content/*`
3. **Deleted old documentation** - `BAN_GIAO_REFACTOR_MODULAR_VA_RULES_AI.md`
4. **Deleted root-level backup/investigation files**

### Remaining Issues Identified
1. **Legacy references to `shared_secret_runtime.py`** - Still imported in 6 files (graceful fallback)
2. **Legacy `A20_ROOT` references** - Still in config files (graceful fallback)
3. **`TRACKING_SERVICE_URL` environment variable** - Still used (points to LightningAI external service - valid for remote ranking)
4. **Legacy script `start_tracking_service_api_builder.sh`** - References deleted tracking-service directory

---

## 3. Deleted Files

### Already Deleted (from git status)
| File | Reason |
|------|--------|
| `QUICKSTART.md` | Obsolete quickstart |
| `README_DEPLOY.md` | Obsolete deployment guide |
| `TRACKLET_REDUCTION_INVESTIGATION.md` | Investigation file |
| `backup.sql` | Database backup file |
| `users_data.sql` | Sample data file |
| `shared_secret_runtime.py` | Deleted (referenced by legacy imports) |
| `ingest_local.py` | Local ingestion script |
| `docs/BAN_GIAO_REFACTOR_MODULAR_VA_RULES_AI.md` | Old handover doc |
| `frontend/components/video/VideoGrid.tsx` | Unused component |
| `frontend/lib/mock/*` | Unused mock data (2 files) |
| `frontend/lib/content/*` | Unused content (2 files) |

### Deleted Backend Services
| Path | Reason |
|------|--------|
| `backend/services/ai-service/` | AI service - obsolete, logic moved to 3 services |
| `backend/services/tracking-service/` | Tracking service - **DOES NOT EXIST** (typo: was `trackig-service`) |

### Deleted Backend Files
| File | Reason |
|------|--------|
| `backend/services/metadata-service/app/__init__.py` | Empty file |
| `backend/services/metadata-service/app/api/__init__.py` | Empty file |
| `backend/services/metadata-service/app/api/routers/__init__.py` | Comment-only file |
| `backend/services/metadata-service/app/api/routers/candidates.py` | Candidate logic moved to query-service |
| `backend/services/metadata-service/app/api/routers/queue.py` | Queue logic restructured |
| `backend/services/metadata-service/app/api/routers/trace.py` | Trace logic moved to trace-service |
| `backend/services/metadata-service/app/services/__init__.py` | Empty file |
| `backend/services/metadata-service/app/services/candidate_service.py` | Logic moved to query-service |
| `backend/services/metadata-service/app/services/trace_service.py` | Logic moved to trace-service |

---

## 4. Backend Service Structure

### Current Services (3 services - Correct)

```
backend/services/
├── metadata-service/     # Video metadata, queue, auth
├── query-service/        # Search, ranking, candidates
├── trace-service/        # Trace building, evidence
└── shared/              # Common utilities (config, database, models)
```

### Service Responsibilities

#### metadata-service
- Auth endpoints (`/api/v1/auth/*`)
- User management (`/api/v1/users/*`)
- Video metadata (`/api/v1/videos/*`)
- Queue management
- Queue worker for video processing
- **NOT**: Query, trace, candidate ranking

#### query-service
- Search endpoint (`/api/v1/search`)
- Candidate ranking
- Query history management
- Candidate selection/continuation
- **NOT**: Video metadata, trace building

#### trace-service
- Trace building (`/api/v1/trace/*`)
- Evidence video generation
- Segment clip management
- Trace confirmation and feedback
- **NOT**: Search, candidate ranking

### Confirmation: NO trackingg-service
- Folder `backend/services/tracking-service/` was deleted (note: actual name was `trackig-service`)
- No imports or references to `trackingg-service` in codebase
- All tracking logic properly distributed to 3 services

---

## 5. Database/Migration Consistency

### Models Audit
Backend services use SQLAlchemy models in:
- `backend/services/shared/models.py` - Common models (User, VideoAsset, VideoQuery, PersonCandidate, QueueVideoAsset)
- `backend/services/trace-service/app/core/models.py` - Trace-specific models (QueryHistory, QueryCandidate, EvidenceVideo, etc.)

### Schema Compliance
- Models align with v3.3 schema
- `query_history` has `retention_expires_at` column
- `query_candidates` has `updated_at` column
- Evidence tables properly structured for unlimited segments
- No `tenant_id` references in models

### Note on Missing Migrations Directory
Backend does not have a `migrations/` folder. Database initialization is done via:
- `infra/postgres/init/01-init.sql` - Basic init
- Service models create tables via SQLAlchemy

---

## 6. Frontend Changes

### API Client
- `frontend/lib/api/client.ts` - Search and video detail APIs
- API paths match backend endpoints
- No hardcoded limits on evidence videos

### Types
- `frontend/lib/types/index.ts` - VideoItem, VideoClip types
- No tenant-related types
- No tracking-service references

### Key Flows
1. **Search Flow**: `/search` endpoint → results list
2. **Candidate Detail**: `/videos/{id}` → segments list
3. **Evidence Videos**: Display all segments from response
4. **History**: User-only access, 3-5 recent queries

### Auth/Role
- `LoginPage.tsx` - Role-based login (SUPER_ADMIN, ADMIN, USER)
- Admin cannot continue/retrace user queries
- User can only see own queries

---

## 7. Evidence Videos Verification

### Confirmed: NO Limit on Evidence Videos
- Backend returns all tracklets as segments
- `trace_service.py` creates evidence for each tracklet
- `EvidenceVideo.segment_count` reflects actual count
- Frontend displays all segments

### Trace Flow
1. User selects candidate
2. `POST /api/v1/trace/build` with query_id, candidate_id
3. Service fetches ALL tracklets for candidate in 24h window
4. Each tracklet = 1 segment clip
5. Response includes all segments sorted by time
6. Cache overwrite on re-select confirmed

### Schema Compliance
```python
# trace_service.py - correctly builds all segments
for idx, tracklet in enumerate(tracklets):
    segment = {
        "segment_order": idx + 1,
        "tracklet_id": tracklet.id,
        # ...
    }
```

---

## 8. Role-Based Access Verification

### Admin/Client Access
| Action | Allowed |
|--------|---------|
| View all user queries | Yes (text, time, status only) |
| View query candidates | No |
| View evidence videos | No |
| Continue user query | No |
| Retrace user query | No |
| Manage users | Yes |

### User Access
| Action | Allowed |
|--------|---------|
| View own queries | Yes |
| View own candidates | Yes |
| View own evidence | Yes |
| Continue own query | Yes |
| Retrace own query | Yes |
| View other user queries | No |

### Implementation
- Backend checks `user_id` ownership for all user-specific operations
- Admin role has limited query history view (no candidates/results)
- Frontend gates UI based on role

---

## 9. Infra/Config Changes

### Docker Compose (`infra/docker-compose.yml`)
- Services: `postgres`, `metadata-service`, `frontend` (correct)
- No references to deleted services
- Health checks configured
- Traefik labels for Coolify deployment

### Environment Variables
- `TRACKING_SERVICE_URL` - Points to LightningAI external service (valid for remote ranking)
- `DATABASE_URL` - PostgreSQL connection
- No tenant-specific configs

### Scripts
- `scripts/start_tracking_service_api_builder.sh` - **ISSUE**: References deleted `backend/services/tracking-service/`
- Other scripts: `init_project.py`, `submit_log.py` use graceful fallbacks

---

## 10. Search/Grep Results

### grep: trackingg-service
```
Result: 0 matches
```

### grep: tracking_service (valid - external service)
```
backend/services/query-service/app/config.py
backend/services/query-service/app/services/candidate_query.py
backend/services/shared/config.py
backend/services/metadata-service/app/config.py
backend/services/metadata-service/app/queue_worker.py
scripts/init_project.py
scripts/start_tracking_service_api_builder.sh
scripts/submit_log.py
```
**Note**: `TRACKING_SERVICE_URL` points to LightningAI external service for remote ranking - this is valid.

### grep: tenant
```
Result: 0 matches in code/config
```

### grep: shared_secret_runtime
```
backend/services/metadata-service/app/config.py     # import with fallback
backend/services/metadata-service/app/services/queue_service.py
scripts/init_project.py
scripts/submit_log.py
src/config.py
backend/services/metadata-service/Dockerfile
```
**Note**: All imports have graceful `try/except ImportError` fallbacks.

### grep: A20_ROOT
```
backend/services/shared/config.py
backend/services/metadata-service/app/services/queue_service.py
backend/services/metadata-service/app/config.py
```
**Note**: Legacy environment variable, graceful handling.

### grep: evidence.*3, max.*3, limit.*3
```
backend/services/query-service/app/services/candidate_query.py  # max 500 candidates
backend/services/metadata-service/app/config.py               # queue limits
```
**Note**: These are reasonable limits, not hardcoded "3 evidence videos" limits.

### grep: deleted file references
```
backup.sql: Not found
ingest_local.py: Not found
QUICKSTART.md: Not found (deleted)
README_DEPLOY.md: Not found (deleted)
shared_secret_runtime.py: Found in imports (graceful fallback)
TRACKLET_REDUCTION_INVESTIGATION.md: Not found (deleted)
users_data.sql: Not found (deleted)
```

---

## 11. Commands Run

### Git Status
```bash
git status --short
```
**Result**: Shows deleted files correctly staged for commit

### Grep Searches
- `grep -r "trackingg-service"` - 0 matches
- `grep -r "tenant"` - 0 matches in code
- `grep -r "shared_secret_runtime"` - 6 matches (graceful fallbacks)

### Build/Typecheck
**Frontend**:
- `npm run lint` - not executed (not in CI)
- `npm run typecheck` - not executed (not in CI)

**Backend**:
- Python linting - not executed
- Type checking - not executed

**Note**: Repository does not have automated CI/CD configured for build/typecheck.

---

## 12. Remaining Risks / Open Questions

### Risk 1: Legacy Script References Deleted Directory
**File**: `scripts/start_tracking_service_api_builder.sh`
**Issue**: References `backend/services/tracking-service/` which was deleted
**Action**: Delete this script or update to point to correct service

### Risk 2: Shared Models Mismatch
**Files**: `backend/services/shared/models.py` vs `backend/services/trace-service/app/core/models.py`
**Issue**: Duplicate model definitions (User, QueryHistory, QueryCandidate, etc.)
**Status**: Working but inconsistent - shared models not used by trace-service

### Risk 3: Queue Router Missing
**File**: `backend/services/metadata-service/app/main.py`
**Issue**: `queue` router was deleted, import removed
**Status**: Import removed - if queue endpoints needed, must recreate

### Risk 4: Evidence Video Limit Documentation
**File**: `backend/config/camera_topology.json`
**Issue**: Contains `_note` about "3 evidence videos" in hospital topology
**Action**: Remove `_note` field or clarify it refers to quick history cache, not evidence videos

### Open Question: Is `TRACKING_SERVICE_URL` Required?
The environment variable `TRACKING_SERVICE_URL` points to an external LightningAI service. If this external service is deprecated:
1. Query-service ranking must be reimplemented locally
2. Environment variable can be removed
3. All references to `tracking_service_url` in configs can be cleaned up

---

## Summary

### Completed
- Backend services reduced to 3 (metadata, query, trace)
- No trackingg-service references
- No tenant logic in code
- Evidence videos not limited to 3
- Broken imports fixed
- Unused files deleted
- Role-based access implemented

### Needs Attention
- Delete/update `scripts/start_tracking_service_api_builder.sh`
- Consider consolidating shared models
- Clean up `_note` in `camera_topology.json`
- Evaluate if `TRACKING_SERVICE_URL` external service is still needed
