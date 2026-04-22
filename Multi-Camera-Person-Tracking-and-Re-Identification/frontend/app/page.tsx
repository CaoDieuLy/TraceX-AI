"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";

type Overview = {
  metadata: {
    metrics: {
      total_users: number;
      total_managed_videos: number;
      total_queries: number;
      total_candidates: number;
      total_cameras: number;
      total_candidate_videos: number;
      total_queue_videos: number;
    };
  };
  ai: {
    provider: string;
    mode: string;
  };
};

type User = {
  id: number;
  email: string;
  full_name: string;
};

type QueueVideo = {
  video_id: string;
  camera_id?: string | null;
  title: string;
  queue_position: number;
  storage_backend: string;
  available_link_video: string;
  available_link_metadata?: string | null;
  local_video_path?: string | null;
  local_metadata_path?: string | null;
  source_filename?: string | null;
};

type Candidate = {
  candidate_id: string;
  camera_id?: string | null;
  video_id?: string | null;
  track_id?: string | null;
  human_key?: string | null;
  frame_idx?: number | null;
  bbox?: number[];
  search_text?: string | null;
  appearance_summary?: string | null;
  semantic_attributes?: string[];
  matched_segments?: Array<{
    start_second?: number;
    end_second?: number;
    action_summary?: string;
  }>;
  score?: number | null;
  available_link_video?: string | null;
  available_link_metadata?: string | null;
  local_video_path?: string | null;
  local_metadata_path?: string | null;
  storage_path?: string | null;
  video_title?: string | null;
  preview_image_url?: string | null;
  raw_metadata: Record<string, unknown>;
};

type TrackingResult = {
  artifact_id: string;
  video_url: string;
  manifest_url: string;
  selected_candidate_id: string;
  manifest: {
    clip_count: number;
    written_frames: number;
    candidate_ids: string[];
  };
};

type MoveResult = {
  processed_videos: number;
  evicted_video_ids: string[];
  imported_source_files: string[];
};

type AuthMode = "login" | "register";

const API_BASE = (process.env.NEXT_PUBLIC_API_GATEWAY_URL ?? "").replace(/\/$/, "");
const TOKEN_KEY = "mcpt_access_token";
const defaultRegister = { full_name: "", email: "", password: "" };
const defaultLogin = { identifier: "admin", password: "admin" };

function withApiBase(path: string | null | undefined): string {
  if (!path) {
    return "";
  }
  if (path.startsWith("http://") || path.startsWith("https://")) {
    return path;
  }
  return `${API_BASE}${path.startsWith("/") ? path : `/${path}`}`;
}

function candidatePreviewUrl(candidate: Candidate | null | undefined): string {
  if (!candidate) {
    return "";
  }
  return withApiBase(candidate.preview_image_url ?? null);
}

function formatSeconds(value: number | undefined): string {
  if (typeof value !== "number" || Number.isNaN(value)) {
    return "--";
  }
  const minutes = Math.floor(value / 60);
  const seconds = Math.floor(value % 60)
    .toString()
    .padStart(2, "0");
  return `${minutes}:${seconds}`;
}

function candidatePrimaryMoment(candidate: Candidate | null | undefined) {
  const firstSegment = candidate?.matched_segments?.[0];
  if (!firstSegment) {
    return null;
  }
  return `${formatSeconds(firstSegment.start_second)} - ${formatSeconds(firstSegment.end_second)}`;
}

export default function HomePage() {
  const [authMode, setAuthMode] = useState<AuthMode>("login");
  const [token, setToken] = useState("");
  const [user, setUser] = useState<User | null>(null);
  const [overview, setOverview] = useState<Overview | null>(null);
  const [queueVideos, setQueueVideos] = useState<QueueVideo[]>([]);
  const [candidateQuery, setCandidateQuery] = useState("doctor carrying a medical box");
  const [candidateResults, setCandidateResults] = useState<Candidate[]>([]);
  const [selectedCandidateId, setSelectedCandidateId] = useState("");
  const [trackingResult, setTrackingResult] = useState<TrackingResult | null>(null);
  const [moveResult, setMoveResult] = useState<MoveResult | null>(null);
  const [registerForm, setRegisterForm] = useState(defaultRegister);
  const [loginForm, setLoginForm] = useState(defaultLogin);
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
    setCandidateResults([]);
    setSelectedCandidateId("");
    setTrackingResult(null);
    setMoveResult(null);
    setMessage(null);
    setError(null);
  }

  async function apiFetch(path: string, init: RequestInit = {}) {
    const headers = new Headers(init.headers ?? {});
    if (token) {
      headers.set("Authorization", `Bearer ${token}`);
    }
    const response = await fetch(`${API_BASE}${path}`, { ...init, headers, cache: "no-store" });
    if (response.status === 401) {
      clearSession();
      throw new Error("Session expired. Please log in again.");
    }
    return response;
  }

  async function refreshWorkspace() {
    const [overviewResponse, meResponse, queueResponse] = await Promise.all([
      fetch(`${API_BASE}/api/v1/overview`, { cache: "no-store" }),
      apiFetch("/api/v1/auth/me"),
      fetch(`${API_BASE}/api/v1/queue/videos`, { cache: "no-store" }),
    ]);

    if (!overviewResponse.ok || !meResponse.ok || !queueResponse.ok) {
      throw new Error("Failed to refresh the workspace.");
    }

    setOverview((await overviewResponse.json()) as Overview);
    setUser((await meResponse.json()) as User);
    const queuePayload = (await queueResponse.json()) as { items: QueueVideo[] };
    setQueueVideos(queuePayload.items);
  }

  useEffect(() => {
    const stored = localStorage.getItem(TOKEN_KEY);
    if (stored) {
      setToken(stored);
      return;
    }
    fetch(`${API_BASE}/api/v1/overview`, { cache: "no-store" })
      .then((response) => response.json())
      .then((payload: Overview) => setOverview(payload))
      .catch(() => undefined);
  }, []);

  useEffect(() => {
    if (!token) {
      return;
    }
    refreshWorkspace().catch((err: Error) => setError(err.message));
  }, [token]);

  async function handleAuthSubmit(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const payload = authMode === "register" ? registerForm : loginForm;
      const response = await fetch(`${API_BASE}/api/v1/auth/${authMode}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const data = (await response.json()) as { access_token: string; user: User };
      persistToken(data.access_token);
      setUser(data.user);
      setRegisterForm(defaultRegister);
      setLoginForm(defaultLogin);
      setMessage(authMode === "register" ? "Account created successfully." : "Logged in successfully.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Authentication failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleMove() {
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const response = await apiFetch("/api/v1/queue/process-imports", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({}),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as MoveResult;
      setMoveResult(payload);
      await refreshWorkspace();
      setMessage(
        payload.imported_source_files.length
          ? `Move finished. Imported ${payload.imported_source_files.length} file(s) from Import_New.`
          : "Move finished. No new .h265 files were waiting in Import_New.",
      );
    } catch (err) {
      setError(err instanceof Error ? err.message : "Move failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleSearch(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    setLoading(true);
    setError(null);
    setMessage(null);
    setTrackingResult(null);
    try {
      const response = await apiFetch("/api/v1/candidates/search", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ query_text: candidateQuery, limit: 5 }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as { items: Candidate[]; count: number };
      setCandidateResults(payload.items);
      setSelectedCandidateId(payload.items[0]?.candidate_id ?? "");
      setMessage(payload.count ? `Found ${payload.count} candidate match(es).` : "No candidates matched this query.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Candidate search failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleBuildTracking() {
    if (!selectedCandidateId) {
      setError("Choose a candidate first.");
      return;
    }
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const response = await apiFetch("/api/v1/candidates/track", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({
          selected_candidate_id: selectedCandidateId,
          candidate_ids: candidateResults.map((item) => item.candidate_id),
          query_text: candidateQuery,
          max_segments_per_candidate: 2,
        }),
      });
      if (!response.ok) {
        throw new Error(await response.text());
      }
      const payload = (await response.json()) as TrackingResult;
      setTrackingResult(payload);
      setMessage("Tracking video is ready.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Tracking compilation failed");
    } finally {
      setLoading(false);
    }
  }

  const selectedCandidate = useMemo(
    () => candidateResults.find((item) => item.candidate_id === selectedCandidateId) ?? candidateResults[0] ?? null,
    [candidateResults, selectedCandidateId],
  );

  if (!user) {
    return (
      <main className="shell auth-shell">
        <section className="hero auth-hero">
          <div className="hero-copy">
            <p className="eyebrow">MCPT Control Center</p>
            <h1>Login first, then move files, query people, and build tracking video output.</h1>
            <p className="lead">
              Screen 1 is access only. After login, you land in a dedicated workspace for `Move`, text query, top-k
              candidates, and final tracking video output.
            </p>
            <div className="badge-row">
              <span className="chip">Auto detect from `Import_New`</span>
              <span className="chip">Manual `Move` trigger</span>
              <span className="chip">GPU metadata on LightningAI</span>
            </div>
          </div>

          <section className="panel auth-panel">
            <div className="panel-heading">
              <div>
                <p className="eyebrow">Access</p>
                <h2>{authMode === "login" ? "Login" : "Create account"}</h2>
              </div>
            </div>

            <div className="tab-row">
              <button className={authMode === "login" ? "tab active" : "tab"} onClick={() => setAuthMode("login")} type="button">
                Login
              </button>
              <button className={authMode === "register" ? "tab active" : "tab"} onClick={() => setAuthMode("register")} type="button">
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
                placeholder={authMode === "login" ? "Username or email" : "Email"}
                value={authMode === "register" ? registerForm.email : loginForm.identifier}
                onChange={(event) =>
                  authMode === "register"
                    ? setRegisterForm((current) => ({ ...current, email: event.target.value }))
                    : setLoginForm((current) => ({ ...current, identifier: event.target.value }))
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

            <div className="detail-card">
              <p>
                <span>Admin login</span>
                <strong>Username: admin</strong>
              </p>
              <p>
                <span>Password</span>
                <strong>admin</strong>
              </p>
            </div>
          </section>
        </section>

        <section className="summary-strip">
          <article className="stat-card">
            <span>Queue videos</span>
            <strong>{overview?.metadata.metrics.total_queue_videos ?? 0}</strong>
          </article>
          <article className="stat-card">
            <span>Candidates</span>
            <strong>{overview?.metadata.metrics.total_candidates ?? 0}</strong>
          </article>
          <article className="stat-card">
            <span>Cameras</span>
            <strong>{overview?.metadata.metrics.total_cameras ?? 0}</strong>
          </article>
          <article className="stat-card">
            <span>GPU backend</span>
            <strong>
              {overview?.ai.provider ?? "lightningai"} / {overview?.ai.mode ?? "loading"}
            </strong>
          </article>
        </section>

        {message ? <p className="message success">{message}</p> : null}
        {error ? <p className="message error">{error}</p> : null}
      </main>
    );
  }

  return (
    <main className="shell workspace-shell">
      <section className="hero workspace-hero">
        <div className="hero-copy">
          <p className="eyebrow">Workspace</p>
          <h1>Move queue files, search people by text, then output one tracking video.</h1>
          <p className="lead">
            Screen 2 is the working area only. New `.h265` files can enter from auto detect or the manual `Move`
            button, then you query local metadata, choose a candidate, and export a combined tracking video.
          </p>
          <div className="hero-actions">
            <button className="primary" onClick={handleMove} disabled={loading}>
              {loading ? "Working..." : "Move"}
            </button>
            <button className="secondary" onClick={() => refreshWorkspace().catch((err: Error) => setError(err.message))} type="button">
              Refresh
            </button>
            <button className="secondary" onClick={clearSession} type="button">
              Log out
            </button>
          </div>
        </div>

        <section className="panel">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Session</p>
              <h2>{user.full_name}</h2>
            </div>
            <span className="chip">Signed in as {user.email}</span>
          </div>
          <div className="summary-strip compact">
            <article className="stat-card">
              <span>Queue videos</span>
              <strong>{queueVideos.length}</strong>
            </article>
            <article className="stat-card">
              <span>Candidates</span>
              <strong>{overview?.metadata.metrics.total_candidates ?? 0}</strong>
            </article>
            <article className="stat-card">
              <span>GPU</span>
              <strong>{overview?.ai.provider ?? "lightningai"}</strong>
            </article>
          </div>
          <div className="detail-card">
            <p>
              <span>Manual move</span>
              <strong>Button `Move` runs the same Import_New detection flow as the background worker.</strong>
            </p>
            <p>
              <span>Auto move</span>
              <strong>Queue worker still keeps watching `Import_New` in parallel.</strong>
            </p>
          </div>
        </section>
      </section>

      {message ? <p className="message success">{message}</p> : null}
      {error ? <p className="message error">{error}</p> : null}

      <section className="step-grid">
        <section className="panel step-card">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Step 1</p>
              <h2>Move from Import_New</h2>
            </div>
          </div>
          <div className="detail-card">
            <p>
              <span>What it does</span>
              <strong>Detect new `.h265`, call LightningAI for metadata, update local DB, then place the video into Queue logic.</strong>
            </p>
            <p>
              <span>Current queue size</span>
              <strong>{queueVideos.length} indexed queue video(s)</strong>
            </p>
          </div>
          {moveResult ? (
            <div className="detail-card">
              <p>
                <span>Last move</span>
                <strong>{moveResult.processed_videos} processed video(s)</strong>
              </p>
              <p>
                <span>Imported files</span>
                <strong>{moveResult.imported_source_files.join(", ") || "No new file"}</strong>
              </p>
            </div>
          ) : (
            <div className="empty-state">Use `Move` when you want to trigger detection immediately instead of waiting for auto polling.</div>
          )}
        </section>

        <section className="panel step-card">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Step 2</p>
              <h2>Text query to top-k candidates</h2>
            </div>
          </div>
          <form className="stack" onSubmit={handleSearch}>
            <textarea
              value={candidateQuery}
              onChange={(event) => setCandidateQuery(event.target.value)}
              placeholder="Describe the person you want to find across all indexed cameras."
            />
            <button className="primary" type="submit" disabled={loading}>
              {loading ? "Searching..." : "Find top candidates"}
            </button>
          </form>
          <div className="history-list">
            {candidateResults.map((candidate, index) => {
              const previewUrl = candidatePreviewUrl(candidate);
              const isSelected = selectedCandidateId === candidate.candidate_id;
              const primaryMoment = candidatePrimaryMoment(candidate);

              return (
                <article key={candidate.candidate_id} className={isSelected ? "candidate-card active" : "candidate-card"}>
                  <div className="candidate-card__preview">
                    {previewUrl ? (
                      <img
                        className="candidate-preview-image"
                        src={previewUrl}
                        alt={`${candidate.candidate_id} preview with bounding box`}
                      />
                    ) : (
                      <div className="preview-placeholder">Preview image unavailable</div>
                    )}
                    <span className="candidate-rank">Top {index + 1}</span>
                  </div>

                  <div className="candidate-card__body">
                    <div className="candidate-card__heading">
                      <strong>
                        {candidate.camera_id ?? candidate.video_title ?? "candidate"} | frame {candidate.frame_idx ?? "?"}
                      </strong>
                      <span>Score {(candidate.score ?? 0).toFixed(3)}</span>
                    </div>

                    <div className="candidate-meta">
                      <span>Track {candidate.track_id ?? "?"}</span>
                      <span>{candidate.human_key ?? "human_key unavailable"}</span>
                    </div>

                    <p className="candidate-summary">
                      {candidate.appearance_summary ?? candidate.search_text ?? "No summary yet."}
                    </p>

                    <div className="candidate-meta">
                      <span>{candidate.video_id ?? "unknown video"}</span>
                      <span>{primaryMoment ? `Best moment ${primaryMoment}` : "No clip timing yet"}</span>
                    </div>

                    {candidate.bbox?.length === 4 ? (
                      <div className="candidate-meta">
                        <span>
                          Bounding box [{candidate.bbox.join(", ")}]
                        </span>
                      </div>
                    ) : null}

                    {candidate.semantic_attributes?.length ? (
                      <div className="candidate-chip-row">
                        {candidate.semantic_attributes.slice(0, 5).map((attribute) => (
                          <span key={`${candidate.candidate_id}-${attribute}`} className="mini-chip">
                            {attribute}
                          </span>
                        ))}
                      </div>
                    ) : null}

                    <button className={isSelected ? "primary" : "secondary"} onClick={() => setSelectedCandidateId(candidate.candidate_id)} type="button">
                      {isSelected ? "Selected for output" : "Choose this candidate"}
                    </button>
                  </div>
                </article>
              );
            })}
            {!candidateResults.length ? <div className="empty-state">Top-k candidate results will appear here.</div> : null}
          </div>
        </section>

        <section className="panel step-card">
          <div className="panel-heading">
            <div>
              <p className="eyebrow">Step 3</p>
              <h2>Build tracking video output</h2>
            </div>
          </div>
          {selectedCandidate ? (
            <div className="stack">
              <div className="preview-stage">
                <div className="preview-stage__media">
                  {candidatePreviewUrl(selectedCandidate) ? (
                    <img
                      key={selectedCandidate.candidate_id}
                      className="preview-stage-image"
                      src={candidatePreviewUrl(selectedCandidate)}
                      alt={`${selectedCandidate.candidate_id} selected preview with bounding box`}
                    />
                  ) : (
                    <div className="preview-placeholder large">Candidate preview image unavailable</div>
                  )}
                </div>
                <div className="preview-stage__details">
                  <p className="eyebrow">Preview before export</p>
                  <h3>{selectedCandidate.candidate_id}</h3>
                  <p className="lead compact">
                    {selectedCandidate.appearance_summary ?? selectedCandidate.search_text ?? "No summary"}
                  </p>
                  <div className="detail-card">
                    <p>
                      <span>Camera / track</span>
                      <strong>
                        {selectedCandidate.camera_id ?? "unknown"} / {selectedCandidate.track_id ?? "?"}
                      </strong>
                    </p>
                    <p>
                      <span>Frame / human key</span>
                      <strong>
                        {selectedCandidate.frame_idx ?? "?"} / {selectedCandidate.human_key ?? "unavailable"}
                      </strong>
                    </p>
                    <p>
                      <span>Source video</span>
                      <strong>{selectedCandidate.video_id ?? "Unknown video"}</strong>
                    </p>
                    <p>
                      <span>Bounding box</span>
                      <strong>{selectedCandidate.bbox?.length === 4 ? `[${selectedCandidate.bbox.join(", ")}]` : "unavailable"}</strong>
                    </p>
                    <p>
                      <span>Best moment</span>
                      <strong>{candidatePrimaryMoment(selectedCandidate) ?? "Use full source preview"}</strong>
                    </p>
                  </div>
                </div>
              </div>

              <div className="detail-card">
                <p>
                  <span>Selected candidate</span>
                  <strong>{selectedCandidate.candidate_id}</strong>
                </p>
                <p>
                  <span>Camera / track</span>
                  <strong>
                    {selectedCandidate.camera_id ?? "unknown"} / {selectedCandidate.track_id ?? "?"}
                  </strong>
                </p>
                <p>
                  <span>Summary</span>
                  <strong>{selectedCandidate.appearance_summary ?? selectedCandidate.search_text ?? "No summary"}</strong>
                </p>
              </div>
              <button className="primary" onClick={handleBuildTracking} disabled={loading} type="button">
                {loading ? "Building..." : "Build tracking video"}
              </button>
            </div>
          ) : (
            <div className="empty-state">Pick one candidate from Step 2, then build the output video here.</div>
          )}

          {trackingResult ? (
            <div className="stack">
              <div className="preview-stage output-stage">
                <div className="preview-stage__media">
                  <video className="video-frame" controls preload="metadata" src={withApiBase(trackingResult.video_url)} />
                </div>
                <div className="preview-stage__details">
                  <p className="eyebrow">Output preview</p>
                  <h3>Tracking video ready</h3>
                  <p className="lead compact">
                    This is the end-user output video assembled from the selected candidate and the top-ranked supporting clips.
                  </p>
                  <div className="detail-card">
                    <p>
                      <span>Artifact</span>
                      <strong>{trackingResult.artifact_id}</strong>
                    </p>
                    <p>
                      <span>Selected candidate</span>
                      <strong>{trackingResult.selected_candidate_id}</strong>
                    </p>
                    <p>
                      <span>Clip count</span>
                      <strong>{trackingResult.manifest.clip_count}</strong>
                    </p>
                    <p>
                      <span>Frames written</span>
                      <strong>{trackingResult.manifest.written_frames}</strong>
                    </p>
                  </div>
                  <a className="secondary inline-link" href={withApiBase(trackingResult.video_url)} rel="noreferrer" target="_blank">
                    Open output video
                  </a>
                </div>
              </div>
              <div className="detail-card">
                <p>
                  <span>Artifact</span>
                  <strong>{trackingResult.artifact_id}</strong>
                </p>
                <p>
                  <span>Clips used</span>
                  <strong>{trackingResult.manifest.clip_count}</strong>
                </p>
                <p>
                  <span>Frames written</span>
                  <strong>{trackingResult.manifest.written_frames}</strong>
                </p>
              </div>
              <a className="secondary inline-link" href={withApiBase(trackingResult.manifest_url)} rel="noreferrer" target="_blank">
                Open manifest JSON
              </a>
            </div>
          ) : null}
        </section>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Queue</p>
            <h2>Indexed `.h265` videos</h2>
          </div>
        </div>
        <div className="video-list">
          {queueVideos.map((video) => (
            <a key={video.video_id} className="video-card" href={video.available_link_video} target="_blank" rel="noreferrer">
              <strong>{video.title}</strong>
              <span>{video.source_filename ?? video.camera_id ?? "queue"}</span>
              <small>{video.storage_backend}</small>
            </a>
          ))}
          {!queueVideos.length ? <div className="empty-state">Queue is still empty. Add `.h265` into `Import_New` or press `Move`.</div> : null}
        </div>
      </section>
    </main>
  );
}
