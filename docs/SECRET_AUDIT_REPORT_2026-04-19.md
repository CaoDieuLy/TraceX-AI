# Secret Audit Report (2026-04-19 09:53:15Z)

## Scope

- Repo: `/teamspace/studios/this_studio/A20-App-119`
- Canonical secrets root: `/teamspace/studios/this_studio/A20-App-119/secrets`
- Findings: `15`

## Findings

| Risk | Type | File | Line | Detail |
| --- | --- | --- | ---: | --- |
| high | OAuth / service account | `secrets/google-drive/drive-sa.json` | 1 | Service account credential |
| high | OAuth | `secrets/oauth/oauth2_credentials.json` | 1 | OAuth client secret |
| high | Token | `secrets/oauth/oauth2_token.pickle` | 1 | OAuth refresh token |
| high | API key | `secrets/shared.env` | 8 | API token |
| high | DB config | `secrets/shared.env` | 14 | Database password |
| medium | Hardcoded URL / endpoint | `secrets/shared.env` | 4 | Drive folder identifier |
| medium | Hardcoded URL / endpoint | `secrets/shared.env` | 5 | Drive folder identifier |
| low | OAuth | `Multi-Camera-Person-Tracking-and-Re-Identification/backend/services/metadata-service/app/config.py` | 22 | Credential file path reference |
| low | OAuth | `Multi-Camera-Person-Tracking-and-Re-Identification/backend/services/metadata-service/app/queue_runtime.py` | 74 | Credential file path reference |
| low | OAuth | `Multi-Camera-Person-Tracking-and-Re-Identification/backend/services/tracking-service/app/config.py` | 22 | Credential file path reference |
| low | OAuth | `Multi-Camera-Person-Tracking-and-Re-Identification/backend/test_gdrive_connection.py` | 45 | Credential file path reference |
| low | OAuth | `scripts/migrate_secrets_to_canonical_layout.py` | 41 | Credential file path reference |
| low | OAuth | `secrets/shared.env` | 1 | Credential file path reference |
| low | OAuth | `shared_secret_runtime.py` | 63 | Credential file path reference |
| low | OAuth | `shared_secret_runtime.py` | 176 | Credential file path reference |

## Recommended Canonical Structure

```text
secrets/
  shared.env
  env/
  db/
  api/
  oauth/
    oauth2_credentials.json
    oauth2_token.pickle
  google-drive/
    drive-sa.json
  deploy/
  docker/
  gpu/
```

## Portable Run Rule

1. Clone repo.
2. Copy back the `secrets/` folder or decrypt the encrypted bundle.
3. Run `python scripts/init_project.py`.
4. Start the app stack.
