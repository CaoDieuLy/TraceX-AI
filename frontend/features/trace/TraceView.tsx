"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { getTraceTimeline, resolveMediaUrl } from "@/lib/api";
import { cameraIdToLocationLabel } from "@/lib/config";
import type { BuildTraceResult, TraceSegment } from "@/lib/types";

type Props = {
  evidenceId: number;
  queryId: string | null;
  candidateId: string | null;
};

export function TraceView({ evidenceId, queryId, candidateId }: Props) {
  const [trace, setTrace] = useState<BuildTraceResult | null>(null);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [activeIdx, setActiveIdx] = useState(0);

  useEffect(() => {
    let cancelled = false;
    let timer: ReturnType<typeof setTimeout> | null = null;
    setIsLoading(true);
    setError(null);

    // Build returns immediately with empty videoClipUrls and renders clips
    // in a background task. Poll the timeline until every segment has a URL
    // (or we've waited long enough that something's clearly wrong).
    const POLL_INTERVAL_MS = 2000;
    const MAX_POLL_MS = 30 * 60 * 1000;
    const startedAt = Date.now();

    const tick = async () => {
      try {
        const t = await getTraceTimeline(evidenceId);
        if (cancelled) return;
        setTrace(t);
        setIsLoading(false);
        const pending = t.segments.some((s) => !s.videoClipUrl);
        if (pending && Date.now() - startedAt < MAX_POLL_MS) {
          timer = setTimeout(tick, POLL_INTERVAL_MS);
        }
      } catch (err) {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Không thể tải timeline.");
        setIsLoading(false);
      }
    };

    void tick();
    return () => {
      cancelled = true;
      if (timer) clearTimeout(timer);
    };
  }, [evidenceId]);

  useEffect(() => {
    // Reset the active index only on the first successful load so that
    // polling updates (which keep `trace` set) don't reset the user's
    // currently selected segment.
    if (trace && activeIdx >= trace.segments.length) {
      setActiveIdx(0);
    }
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [trace?.segments.length]);

  const segments = trace?.segments ?? [];
  const active = segments[activeIdx];
  const readyCount = segments.filter((segment) => Boolean(segment.videoClipUrl)).length;
  const pendingCount = Math.max(segments.length - readyCount, 0);

  return (
    <div className="flex flex-col gap-6">
      <header className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-ink-subtle">Thông tin truy vết</p>
          <h1 className="text-2xl font-semibold text-ink">
            Evidence #{evidenceId}
            {candidateId ? <span className="ml-2 font-mono text-sm text-slate-500">· {candidateId}</span> : null}
          </h1>
        </div>
        <Link
          href={queryId ? `/history/${encodeURIComponent(queryId)}/candidates` : "/home"}
          className="rounded-xl border border-surface-muted bg-white px-4 py-2 text-sm font-medium text-ink-secondary shadow-card transition hover:border-accent hover:text-ink"
        >
          ← {queryId ? "Chọn candidate khác" : "Quay lại"}
        </Link>
      </header>

      {isLoading ? <p className="text-sm text-ink-secondary">Đang tải timeline…</p> : null}
      {error ? (
        <p className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</p>
      ) : null}

      {!isLoading && !error && trace ? (
        <>
          <RenderProgress readyCount={readyCount} pendingCount={pendingCount} totalCount={segments.length} />
          <CameraTimeline segments={segments} activeIdx={activeIdx} onSelect={setActiveIdx} />
          {active ? <SegmentPlayer segment={active} /> : (
            <p className="rounded-xl border border-dashed border-slate-300 px-4 py-6 text-sm text-slate-500">
              Trace không có segment nào.
            </p>
          )}
          <SegmentList segments={segments} activeIdx={activeIdx} onSelect={setActiveIdx} />
        </>
      ) : null}
    </div>
  );
}

function RenderProgress({
  readyCount,
  pendingCount,
  totalCount,
}: {
  readyCount: number;
  pendingCount: number;
  totalCount: number;
}) {
  if (!totalCount) return null;
  const percent = Math.round((readyCount / totalCount) * 100);
  const isComplete = pendingCount === 0;
  return (
    <section className="rounded-2xl border border-slate-200 bg-white p-4 shadow-sm">
      <div className="flex flex-wrap items-center justify-between gap-3">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-slate-500">Tiến độ video trace</p>
          <p className="mt-1 text-sm font-semibold text-slate-900">
            {readyCount}/{totalCount} video sẵn sàng
          </p>
        </div>
        <span className={[
          "rounded-full border px-3 py-1 text-xs font-semibold",
          isComplete
            ? "border-emerald-200 bg-emerald-50 text-emerald-700"
            : "border-amber-200 bg-amber-50 text-amber-700",
        ].join(" ")}>
          {isComplete ? "Đã ghép xong" : `Đang ghép ${pendingCount} video`}
        </span>
      </div>
      <div className="mt-4 h-2 overflow-hidden rounded-full bg-slate-100">
        <div
          className={isComplete ? "h-full bg-emerald-500" : "h-full bg-sky-500 transition-all"}
          style={{ width: `${percent}%` }}
        />
      </div>
      <p className="mt-3 text-sm text-slate-500">
        {isComplete
          ? "Tất cả video đã sẵn sàng để xem."
          : "Video nào ghép xong sẽ xem được ngay. Trang này tự cập nhật, bạn không cần tải lại."}
      </p>
    </section>
  );
}

function SegmentStatusBadge({ ready }: { ready: boolean }) {
  return (
    <span className={[
      "inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-[11px] font-semibold",
      ready
        ? "border-emerald-200 bg-emerald-50 text-emerald-700"
        : "border-amber-200 bg-amber-50 text-amber-700",
    ].join(" ")}>
      {!ready ? <span className="h-2 w-2 animate-pulse rounded-full bg-amber-500" /> : null}
      {ready ? "Sẵn sàng" : "Đang ghép"}
    </span>
  );
}

function CameraTimeline({
  segments,
  activeIdx,
  onSelect,
}: {
  segments: TraceSegment[];
  activeIdx: number;
  onSelect: (i: number) => void;
}) {
  if (!segments.length) return null;
  return (
    <div className="overflow-x-auto rounded-2xl border border-slate-200 bg-white p-3 shadow-sm">
      <ol className="flex items-stretch gap-3">
        {segments.map((s, i) => {
          const isActive = i === activeIdx;
          const ready = Boolean(s.videoClipUrl);
          return (
            <li key={`${s.segmentOrder}-${s.trackletId}`} className="flex items-center gap-3">
              <button
                type="button"
                onClick={() => onSelect(i)}
                className={`flex w-44 shrink-0 flex-col items-start gap-1 rounded-xl border px-3 py-2 text-left transition ${
                  isActive
                    ? "border-sky-500 bg-sky-50 shadow-sm"
                    : "border-slate-200 bg-slate-50 hover:border-slate-300"
                }`}
              >
                <span className="text-[10px] font-semibold uppercase tracking-wide text-slate-500">
                  Step {s.segmentOrder}
                </span>
                <SegmentStatusBadge ready={ready} />
                <CameraLabel cameraId={s.cameraId} compact />
                <span className="text-[11px] text-slate-500">{fmtTime(s.timeStart)}</span>
                {s.durationSeconds != null ? (
                  <span className="text-[11px] text-slate-500">{s.durationSeconds.toFixed(1)}s</span>
                ) : null}
              </button>
              {i < segments.length - 1 ? <span className="text-slate-300">→</span> : null}
            </li>
          );
        })}
      </ol>
    </div>
  );
}

function SegmentPlayer({ segment }: { segment: TraceSegment }) {
  const url = segment.videoClipUrl ? resolveMediaUrl(segment.videoClipUrl) : null;
  return (
    <div className="overflow-hidden rounded-2xl border border-slate-200 bg-black/90 shadow-card">
      {url ? (
        <video
          key={url}
          src={url}
          controls
          preload="metadata"
          className="aspect-video w-full bg-black"
        />
      ) : (
        <div className="flex aspect-video w-full flex-col items-center justify-center gap-3 px-6 text-center text-sm text-white/75">
          <span className="h-8 w-8 animate-spin rounded-full border-2 border-white/20 border-t-white/80" />
          <span className="font-semibold text-white">Video segment này đang được ghép</span>
          <span className="max-w-md text-white/60">
            Khi render xong, player sẽ tự hiện video. Bạn có thể chọn segment đã sẵn sàng ở danh sách bên dưới.
          </span>
        </div>
      )}
      <div className="flex flex-wrap items-baseline gap-x-4 gap-y-1 bg-white/95 px-4 py-3 text-xs text-slate-600">
        <SegmentStatusBadge ready={Boolean(url)} />
        <CameraLabel cameraId={segment.cameraId} />
        <span>{fmtRange(segment.timeStart, segment.timeEnd)}</span>
        {segment.durationSeconds != null ? <span>· {segment.durationSeconds.toFixed(1)}s</span> : null}
        {typeof segment.confidence === "number" ? (
          <span>· conf {Math.round(segment.confidence * 100)}%</span>
        ) : null}
        <span className="ml-auto font-mono text-[11px] text-slate-500" title={segment.trackletId}>
          {segment.trackletId}
        </span>
      </div>
    </div>
  );
}

function SegmentList({
  segments,
  activeIdx,
  onSelect,
}: {
  segments: TraceSegment[];
  activeIdx: number;
  onSelect: (i: number) => void;
}) {
  if (!segments.length) return null;
  return (
    <ol className="flex flex-col gap-3">
      {segments.map((s, i) => {
        const isActive = i === activeIdx;
        const thumb = s.thumbnailUrl ? resolveMediaUrl(s.thumbnailUrl) : null;
        const ready = Boolean(s.videoClipUrl);
        return (
          <li key={`${s.segmentOrder}-${s.trackletId}-row`}>
            <button
              type="button"
              onClick={() => onSelect(i)}
              className={`flex w-full items-center gap-4 rounded-2xl border bg-white p-3 text-left shadow-sm transition ${
                isActive ? "border-sky-500 ring-2 ring-sky-200" : "border-slate-200 hover:border-slate-300"
              }`}
            >
              <span className="w-8 shrink-0 text-center text-xs font-semibold text-slate-500">#{s.segmentOrder}</span>
              <div className="relative h-16 w-24 shrink-0 overflow-hidden rounded-lg bg-slate-100">
                {thumb ? (
                  // eslint-disable-next-line @next/next/no-img-element
                  <img
                    src={thumb}
                    alt={s.trackletId}
                    className={[
                      "h-full w-full object-cover",
                      ready ? "" : "opacity-60",
                    ].join(" ")}
                  />
                ) : null}
                {!ready ? (
                  <div className="absolute inset-0 flex items-center justify-center bg-slate-950/35 text-[10px] font-semibold text-white">
                    Đang ghép
                  </div>
                ) : null}
              </div>
              <div className="flex flex-1 flex-col gap-0.5">
                <div className="flex flex-wrap items-center gap-2">
                  <CameraLabel cameraId={s.cameraId} />
                  <SegmentStatusBadge ready={ready} />
                </div>
                <p className="text-xs text-slate-500">{fmtRange(s.timeStart, s.timeEnd)}</p>
                <p className="font-mono text-[11px] text-slate-400">{s.trackletId}</p>
              </div>
              <div className="text-right text-xs text-slate-600">
                {s.durationSeconds != null ? <p>{s.durationSeconds.toFixed(1)}s</p> : null}
                {typeof s.confidence === "number" ? (
                  <p className="text-slate-400">conf {Math.round(s.confidence * 100)}%</p>
                ) : null}
              </div>
            </button>
          </li>
        );
      })}
    </ol>
  );
}

function CameraLabel({ cameraId, compact = false }: { cameraId: string | null; compact?: boolean }) {
  const location = cameraIdToLocationLabel(cameraId);
  if (!cameraId && !location) {
    return <span className="font-mono text-sm text-slate-900">—</span>;
  }
  if (compact) {
    return (
      <span className="flex max-w-full flex-col leading-tight">
        {location ? <span className="truncate text-xs font-semibold text-slate-900">{location}</span> : null}
        <span className="font-mono text-[11px] text-slate-500">{cameraId ?? "—"}</span>
      </span>
    );
  }
  return (
    <span className="inline-flex flex-wrap items-baseline gap-x-1.5">
      {location ? <span className="font-semibold text-slate-900">{location}</span> : null}
      <span className="font-mono text-sm text-slate-600">{cameraId ?? "—"}</span>
    </span>
  );
}

function pad2(value: number): string {
  return String(value).padStart(2, "0");
}

function fmtTime(iso: string | null): string {
  if (!iso) return "—";
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return "—";
  return `${pad2(d.getUTCHours())}:${pad2(d.getUTCMinutes())}:${pad2(d.getUTCSeconds())}`;
}

function fmtRange(start: string | null, end: string | null): string {
  if (!start && !end) return "—";
  return `${fmtTime(start)} → ${fmtTime(end)}`;
}
