# Secrets Folder

This folder is intentionally ignored by git except for this note and optional `*.example` files.

Canonical layout:

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

Portable workflow:

1. Clone the repo.
2. Copy back the `secrets/` folder or import an encrypted bundle.
3. Run `python scripts/init_project.py`.
4. Start the application stack.
