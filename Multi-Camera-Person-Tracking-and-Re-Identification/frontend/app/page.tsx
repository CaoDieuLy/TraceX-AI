"use client";

import { useEffect, useState } from "react";

type Candidate = {
  candidate_id: string;
  camera_id?: string | null;
  video_id?: string | null;
  track_id?: string | null;
  human_key?: string | null;
  frame_idx?: number | null;
  search_text?: string | null;
  metadata_path?: string | null;
  raw_metadata: Record<string, unknown>;
};

type Overview = {
  metadata: {
    total_candidates: number;
    total_cameras: number;
    total_videos: number;
  };
  pipeline: {
    text_query_model?: string;
    caption_model?: string;
    bootstrap_detection_mode?: string;
    incremental_detection_mode?: string;
    queue_size?: number;
  };
};

type TrackingResult = {
  output_path?: string | null;
  relative_output_path?: string | null;
  exists: boolean;
  tracking_use_mock: boolean;
};

const API_BASE = process.env.NEXT_PUBLIC_API_GATEWAY_URL ?? "http://localhost:8000";

export default function HomePage() {
  const [query, setQuery] = useState("doctor carrying a medical box");
  const [overview, setOverview] = useState<Overview | null>(null);
  const [items, setItems] = useState<Candidate[]>([]);
  const [selected, setSelected] = useState<Candidate | null>(null);
  const [trackingResult, setTrackingResult] = useState<TrackingResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [importing, setImporting] = useState(false);
  const [message, setMessage] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);

  async function fetchOverview() {
    const response = await fetch(`${API_BASE}/api/v1/overview`, { cache: "no-store" });
    if (!response.ok) {
      throw new Error(`Cannot load overview: ${response.status}`);
    }
    const payload = (await response.json()) as Overview;
    setOverview(payload);
  }

  useEffect(() => {
    fetchOverview().catch((err: Error) => setError(err.message));
  }, []);

  async function handleImportLegacy() {
    setImporting(true);
    setError(null);
    setMessage(null);
    try {
      const response = await fetch(`${API_BASE}/api/v1/candidates/import-legacy`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
      });
      if (!response.ok) {
        throw new Error(`Import failed: ${response.status}`);
      }
      const payload = (await response.json()) as {
        imported_count: number;
        updated_count: number;
        file_count: number;
      };
      setMessage(
        `Imported ${payload.imported_count} candidates, updated ${payload.updated_count}, scanned ${payload.file_count} files.`,
      );
      await fetchOverview();
    } catch (err) {
      setError(err instanceof Error ? err.message : "Import failed");
    } finally {
      setImporting(false);
    }
  }

  async function handleSearch() {
    setLoading(true);
    setError(null);
    setMessage(null);
    setTrackingResult(null);
    try {
      const response = await fetch(
        `${API_BASE}/api/v1/candidates?query=${encodeURIComponent(query)}&limit=12`,
        { cache: "no-store" },
      );
      if (!response.ok) {
        throw new Error(`Search failed: ${response.status}`);
      }
      const payload = (await response.json()) as { items: Candidate[] };
      setItems(payload.items);
      setSelected(payload.items[0] ?? null);
      setMessage(payload.items.length ? `Loaded ${payload.items.length} matching candidates.` : "No candidates found.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Search failed");
    } finally {
      setLoading(false);
    }
  }

  async function handleTrack() {
    if (!selected) {
      return;
    }
    setLoading(true);
    setError(null);
    setMessage(null);
    try {
      const response = await fetch(`${API_BASE}/api/v1/tracking/run`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ candidate_info: selected.raw_metadata ?? selected }),
      });
      if (!response.ok) {
        throw new Error(`Tracking failed: ${response.status}`);
      }
      const payload = (await response.json()) as TrackingResult;
      setTrackingResult(payload);
      setMessage(payload.exists ? "Tracking clip generated successfully." : "Tracking finished but no output file was found.");
    } catch (err) {
      setError(err instanceof Error ? err.message : "Tracking failed");
    } finally {
      setLoading(false);
    }
  }

  return (
    <main className="shell">
      <section className="hero">
        <div className="hero-copy">
          <p className="eyebrow">Multi-Service Surveillance Platform</p>
          <h1>MCPT Control Center</h1>
          <p className="lead">
            Frontend Next.js, backend FastAPI, metadata in PostgreSQL, and the current ReID engine wrapped as an
            independent tracking service.
          </p>
          <div className="hero-actions">
            <button className="primary" onClick={handleImportLegacy} disabled={importing}>
              {importing ? "Importing..." : "Import legacy metadata"}
            </button>
            <button className="secondary" onClick={() => fetchOverview().catch((err: Error) => setError(err.message))}>
              Refresh overview
            </button>
          </div>
        </div>

        <div className="overview-grid">
          <article className="stat-card">
            <span>Candidates</span>
            <strong>{overview?.metadata.total_candidates ?? 0}</strong>
          </article>
          <article className="stat-card">
            <span>Cameras</span>
            <strong>{overview?.metadata.total_cameras ?? 0}</strong>
          </article>
          <article className="stat-card">
            <span>Videos</span>
            <strong>{overview?.metadata.total_videos ?? 0}</strong>
          </article>
          <article className="stat-card wide">
            <span>Tracking mode</span>
            <strong>{overview?.pipeline.incremental_detection_mode ?? "loading..."}</strong>
          </article>
        </div>
      </section>

      <section className="panel">
        <div className="panel-heading">
          <div>
            <p className="eyebrow">Search</p>
            <h2>Candidate retrieval</h2>
          </div>
          <p className="muted">Gateway coordinates metadata lookup and tracking execution.</p>
        </div>

        <div className="search-row">
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Describe the person to search" />
          <button className="primary" onClick={handleSearch} disabled={loading}>
            {loading ? "Working..." : "Search candidates"}
          </button>
        </div>

        {message ? <p className="message success">{message}</p> : null}
        {error ? <p className="message error">{error}</p> : null}
      </section>

      <section className="workspace">
        <div className="results">
          <div className="section-title">
            <p className="eyebrow">Metadata Service</p>
            <h3>Search results</h3>
          </div>
          <div className="candidate-grid">
            {items.map((item) => (
              <button
                key={item.candidate_id}
                className={`candidate-card ${selected?.candidate_id === item.candidate_id ? "active" : ""}`}
                onClick={() => setSelected(item)}
              >
                <span>{item.camera_id ?? "Unknown camera"}</span>
                <strong>{item.candidate_id}</strong>
                <p>{item.search_text ?? "No description available."}</p>
                <small>
                  Track {item.track_id ?? "-"} | Frame {item.frame_idx ?? 0}
                </small>
              </button>
            ))}
          </div>
        </div>

        <aside className="detail-panel">
          <div className="section-title">
            <p className="eyebrow">Tracking Service</p>
            <h3>Selected candidate</h3>
          </div>

          {selected ? (
            <>
              <div className="detail-card">
                <p>
                  <span>ID</span>
                  <strong>{selected.candidate_id}</strong>
                </p>
                <p>
                  <span>Camera</span>
                  <strong>{selected.camera_id ?? "N/A"}</strong>
                </p>
                <p>
                  <span>Video</span>
                  <strong>{selected.video_id ?? "N/A"}</strong>
                </p>
                <p>
                  <span>Description</span>
                  <strong>{selected.search_text ?? "No description"}</strong>
                </p>
              </div>

              <button className="primary block" onClick={handleTrack} disabled={loading}>
                {loading ? "Running..." : "Run tracking clip"}
              </button>

              {trackingResult ? (
                <div className="tracking-output">
                  <p className="eyebrow">Latest output</p>
                  <p>Mock mode: {trackingResult.tracking_use_mock ? "enabled" : "disabled"}</p>
                  <p>File exists: {trackingResult.exists ? "yes" : "no"}</p>
                  <code>{trackingResult.relative_output_path ?? trackingResult.output_path ?? "No output path"}</code>
                </div>
              ) : null}

              <div className="json-box">
                <pre>{JSON.stringify(selected.raw_metadata, null, 2)}</pre>
              </div>
            </>
          ) : (
            <div className="detail-card empty">
              Search metadata first, then choose one candidate to trigger cross-camera tracking.
            </div>
          )}
        </aside>
      </section>
    </main>
  );
}
