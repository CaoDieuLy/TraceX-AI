"use client";

import { ChangeEvent, FormEvent, useEffect, useState } from "react";

type Overview = {
  metadata: {
    metrics: {
      total_users: number;
      total_managed_videos: number;
      total_queries: number;
      total_candidates: number;
      total_cameras: number;
      total_candidate_videos: number;
    };
  };
  ai: {
    provider: string;
    mode: string;
    lightning_api_base_url?: string | null;
    lightning_api_endpoint?: string;
  };
};

type User = {
  id: number;
  email: string;
  full_name: string;
};

type Video = {
  video_id: string;
  title: string;
  description?: string | null;
  storage_path: string;
  storage_backend: string;
  source_filename?: string | null;
  content_type?: string | null;
  created_at: string;
};

type QueryItem = {
  query_id: string;
  video_id: string;
  video_title: string;
  storage_path: string;
  query_text: string;
  status: string;
  ai_job_id?: string | null;
  ai_response?: {
    summary?: string;
    mode?: string;
    provider?: string;
    raw_response?: Record<string, unknown>;
  } | null;
  created_at: string;
  updated_at: string;
};

type AuthMode = "login" | "register";

const API_BASE = process.env.NEXT_PUBLIC_API_GATEWAY_URL ?? "http://localhost:8000";
const TOKEN_KEY = "mcpt_access_token";

const defaultRegister = {
  full_name: "",
  email: "",
  password: "",
};

const defaultLogin = {
  email: "",
  password: "",
};

export default function HomePage() {
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [token, setToken] = useState<string>("");
  const [user, setUser] = useState<User | null>(null);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [videos, setVideos] = useState<Video[]>([]);
  const [queries, setQueries] = useState<QueryItem[]>([]);
  const [selectedVideoId, setSelectedVideoId] = useState<string>("");
  const [queryText, setQueryText] = useState("doctor carrying a medical box");
  const [registerForm, setRegisterForm] = useState(defaultRegister);
  const [loginForm, setLoginForm] = useState(defaultLogin);
  const [uploadTitle, setUploadTitle] = useState("");
  const [uploadDescription, setUploadDescription] = useState("");
  const [storageUrl, setStorageUrl] = useState("");
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [loading, setLoading] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  function persistToken(nextToken: string) {
    localStorage.setItem(TOKEN_KEY, nextToken);
    setToken(nextToken);
  }

  function clearSession() {
    localStorage.removeItem(TOKEN_KEY);
    setToken("");
    setUser(null);
    setVideos([]);
    setQueries([]);
    setSelectedVideoId("");
  }

  async function apiFetch(path: string, init: RequestInit = {}) {
    const headers = new Headers(init.headers ?? {});
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
    const response = await fetch(`${API_BASE}${path}`, {
      ...init,
      headers,
      cache: "no-store",
    });
    if (response.status === 401) {
      clearSession();
      throw new Error("Session expired. Please log in again.");
    }
    return response;
  }

  async function refreshDashboard() {
    const [overviewResponse, meResponse, videosResponse, queriesResponse] = await Promise.all([
      fetch(`${API_BASE}/api/v1/overview`, { cache: "no-store" }),
      apiFetch("/api/v1/auth/me"),
      apiFetch("/api/v1/videos"),
      apiFetch("/api/v1/video-queries"),
    ]);

    if (!overviewResponse.ok) {
      throw new Error(`Overview failed: ${overviewResponse.status}`);
    }
    if (!meResponse.ok || !videosResponse.ok || !queriesResponse.ok) {
      throw new Error("Failed to load authenticated dashboard.");
    }

    const overviewPayload = (await overviewResponse.json()) as Overview;
    const mePayload = (await meResponse.json()) as User;
    const videosPayload = (await videosResponse.json()) as { items: Video[] };
    const queriesPayload = (await queriesResponse.json()) as { items: QueryItem[] };

    setOverview(overviewPayload);
    setUser(mePayload);
    setVideos(videosPayload.items);
    setQueries(queriesPayload.items);
    setSelectedVideoId((current) => current || videosPayload.items[0]?.video_id || "");
  }

  useEffect(() => {
    const stored = localStorage.getItem(TOKEN_KEY);
    if (stored) {
      setToken(stored);
    }
  }, []);

  useEffect(() => {
    if (!token) {
      fetch(`${API_BASE}/api/v1/overview`, { cache: "no-store" })
        .then(async (response) => {
          if (!response.ok) {
            throw new Error(`Overview failed: ${response.status}`);
          }
          const payload = (await response.json()) as Overview;
          setOverview(payload);
        })
        .catch((err: Error) => setError(err.message));
      return;
    }

    refreshDashboard().catch((err: Error) => setError(err.message));
  }, [token]);

  async function handleAuthSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const body = authMode === "register" ? registerForm : loginForm;
      const response = await fetch(`${API_BASE}/api/v1/auth/${authMode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as { access_token: string; user: User };
      persistToken(payload.access_token);
      setUser(payload.user);
      setRegisterForm(defaultRegister);
      setLoginForm(defaultLogin);
      setMessage(authMode === "register" ? "Account created successfully." : "Logged in successfully.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleUpload(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const formData = new FormData();
      formData.set("title", uploadTitle);
      formData.set("description", uploadDescription);
      if (selectedFile) {
        formData.set("file", selectedFile);
      }
      if (storageUrl.trim()) {
        formData.set("storage_url", storageUrl.trim());
      }

      const response = await apiFetch("/api/v1/videos", {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as Video;
      setMessage(`Video '${payload.title}' added successfully.`);
      setUploadTitle("");
      setUploadDescription("");
      setStorageUrl("");
      setSelectedFile(null);
      await refreshDashboard();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Video upload failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleRunQuery() {
    if (!selectedVideoId) {
      setError("Please select a video first.");
      return;
    }
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const response = await apiFetch("/api/v1/video-queries/run", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          video_id: selectedVideoId,
          query_text: queryText,
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as { query: QueryItem; ai_result: { summary: string } };
      setMessage(payload.ai_result.summary);
      await refreshDashboard();
    } catch (err) {
      setError(err instanceof Error ? err.message : "AI query failed");
    } finally {
      setLoading(false);
    }
  }

  function handleFileChange(event: ChangeEvent<HTMLInputElement>) {
    const nextFile = event.target.files?.[0] ?? null;
    setSelectedFile(nextFile);
    if (nextFile) {
      setStorageUrl("");
      if (!uploadTitle.trim()) {
        setUploadTitle(nextFile.name.replace(/\.[^.]+$/, ""));
      }
    }
  }

  const latestQuery = queries[0] ?? null;

  return (
    <main className="shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Video AI Management Platform</p>
          <h1>GPU-ready Video Ops Control Center</h1>
          <p className="lead">
            Next.js frontend, FastAPI backend, PostgreSQL metadata store, and LightningAI GPU orchestration for
            video-text processing.
          </p>
          <div className="hero-actions">
            {user ? (
              <>
                <button className="primary" onClick={() => refreshDashboard().catch((err: Error) => setError(err.message))}>
                  Refresh workspace
                </button>
                <button className="secondary" onClick={clearSession}>
                  Log out
                </button>
              </>
            ) : null}
          </div>
        </div>

        <div className="overview-grid">
          <article className="stat-card">
            <span>Users</span>
            <strong>{overview?.metadata.metrics.total_users ?? 0}</strong>
          </article>
          <article className="stat-card">
            <span>Managed videos</span>
            <strong>{overview?.metadata.metrics.total_managed_videos ?? 0}</strong>
          </article>
          <article className="stat-card">
            <span>AI queries</span>
            <strong>{overview?.metadata.metrics.total_queries ?? 0}</strong>
          </article>
          <article className="stat-card wide">
            <span>AI backend</span>
            <strong>
              {overview?.ai.provider ?? "lightningai"} / {overview?.ai.mode ?? "loading"}
            </strong>
          </article>
        </div>
      </section>

      {message ? <p className="message success">{message}</p> : null}
      {error ? <p className="message error">{error}</p> : null}

      <section className="workspace-grid">
        <section className="panel auth-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Access</p>
              <h2>{user ? `Welcome, ${user.full_name}` : "Authenticate"}</h2>
            </div>
            {user ? <p className="muted">{user.email}</p> : null}
          </div>

          {!user ? (
            <>
              <div className="tab-row">
                <button className={authMode === "login" ? "tab active" : "tab"} onClick={() => setAuthMode("login")}>
                  Login
                </button>
                <button className={authMode === "register" ? "tab active" : "tab"} onClick={() => setAuthMode("register")}>
                  Register
                </button>
              </div>
              <form className="stack" onSubmit={handleAuthSubmit}>
                {authMode === "register" ? (
                  <input
                    placeholder="Full name"
                    value={registerForm.full_name}
                    onChange={(event) => setRegisterForm((current) => ({ ...current, full_name: event.target.value }))}
                  />
                ) : null}
                <input
                  placeholder="Email"
                  type="email"
                  value={authMode === "register" ? registerForm.email : loginForm.email}
                  onChange={(event) =>
                    authMode === "register"
                      ? setRegisterForm((current) => ({ ...current, email: event.target.value }))
                      : setLoginForm((current) => ({ ...current, email: event.target.value }))
                  }
                />
                <input
                  placeholder="Password"
                  type="password"
                  value={authMode === "register" ? registerForm.password : loginForm.password}
                  onChange={(event) =>
                    authMode === "register"
                      ? setRegisterForm((current) => ({ ...current, password: event.target.value }))
                      : setLoginForm((current) => ({ ...current, password: event.target.value }))
                  }
                />
                <button className="primary" type="submit" disabled={loading}>
                  {loading ? "Working..." : authMode === "register" ? "Create account" : "Sign in"}
                </button>
              </form>
            </>
          ) : (
            <div className="detail-card">
              <p>
                <span>Account</span>
                <strong>{user.full_name}</strong>
              </p>
              <p>
                <span>Email</span>
                <strong>{user.email}</strong>
              </p>
              <p>
                <span>LightningAI mode</span>
                <strong>{overview?.ai.mode ?? "unknown"}</strong>
              </p>
            </div>
          )}
        </section>

        <section className="panel upload-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Video Library</p>
              <h2>Add a new video</h2>
            </div>
            <p className="muted">Database stores only the path/URL reference.</p>
          </div>
          <form className="stack" onSubmit={handleUpload}>
            <input placeholder="Video title" value={uploadTitle} onChange={(event) => setUploadTitle(event.target.value)} />
            <textarea
              placeholder="Description"
              value={uploadDescription}
              onChange={(event) => setUploadDescription(event.target.value)}
            />
            <label className="file-input">
              <span>{selectedFile ? selectedFile.name : "Choose a local video file"}</span>
              <input type="file" accept="video/*" onChange={handleFileChange} />
            </label>
            <div className="divider">
              <span>or</span>
            </div>
            <input
              placeholder="Existing storage URL / mounted path"
              value={storageUrl}
              onChange={(event) => setStorageUrl(event.target.value)}
            />
            <button className="primary" type="submit" disabled={loading || !user}>
              {loading ? "Saving..." : "Add video reference"}
            </button>
          </form>

          <div className="video-list">
            {videos.map((video) => (
              <button
                key={video.video_id}
                className={selectedVideoId === video.video_id ? "video-card active" : "video-card"}
                onClick={() => setSelectedVideoId(video.video_id)}
              >
                <strong>{video.title}</strong>
                <span>{video.source_filename ?? video.storage_backend}</span>
                <small>{video.storage_path}</small>
              </button>
            ))}
          </div>
        </section>
      </section>

      <section className="workspace-grid">
        <section className="panel query-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">AI Query</p>
              <h2>Run text search against a video</h2>
            </div>
            <p className="muted">FastAPI gateway persists the query then calls LightningAI GPU.</p>
          </div>
          <div className="stack">
            <select value={selectedVideoId} onChange={(event) => setSelectedVideoId(event.target.value)}>
              <option value="">Select a video</option>
              {videos.map((video) => (
                <option key={video.video_id} value={video.video_id}>
                  {video.title}
                </option>
              ))}
            </select>
            <textarea
              placeholder="Describe the event or object to find in the video"
              value={queryText}
              onChange={(event) => setQueryText(event.target.value)}
            />
            <button className="primary" onClick={handleRunQuery} disabled={loading || !user}>
              {loading ? "Processing..." : "Run AI query"}
            </button>
          </div>

          {latestQuery ? (
            <div className="detail-card">
              <p>
                <span>Latest query</span>
                <strong>{latestQuery.query_text}</strong>
              </p>
              <p>
                <span>Status</span>
                <strong>{latestQuery.status}</strong>
              </p>
              <p>
                <span>AI summary</span>
                <strong>{latestQuery.ai_response?.summary ?? "Awaiting AI response"}</strong>
              </p>
            </div>
          ) : null}
        </section>

        <section className="panel history-panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">History</p>
              <h2>Recent AI jobs</h2>
            </div>
            <p className="muted">Tracked per user and per video in PostgreSQL.</p>
          </div>
          <div className="history-list">
            {queries.map((item) => (
              <article key={item.query_id} className="history-card">
                <header>
                  <strong>{item.video_title}</strong>
                  <span>{item.status}</span>
                </header>
                <p>{item.query_text}</p>
                <small>{item.ai_response?.summary ?? "No AI summary yet."}</small>
              </article>
            ))}
            {!queries.length ? <div className="empty-state">No AI jobs yet. Upload a video and run your first query.</div> : null}
          </div>
        </section>
      </section>
    </main>
  );
}
