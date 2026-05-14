import Image from "next/image";
import Link from "next/link";

import type { TrackletSummary, VideoItem } from "@/lib/types";

type VideoCardProps = {
  video: VideoItem;
  onClick?: (video: VideoItem) => void;
  rank?: number;
};

function formatClock(iso: string | null): string | null {
  if (!iso) return null;
  const d = new Date(iso);
  if (Number.isNaN(d.getTime())) return null;
  return d.toLocaleTimeString("vi-VN", { hour: "2-digit", minute: "2-digit", hour12: false });
}

function formatTimeRange(start: string | null, end: string | null): string | null {
  const a = formatClock(start);
  const b = formatClock(end);
  if (a && b) return `${a}–${b}`;
  return a ?? b ?? null;
}

function uniqueCameras(tracklets: TrackletSummary[]): string[] {
  const seen = new Set<string>();
  const result: string[] = [];
  for (const t of tracklets) {
    if (t.cameraId && !seen.has(t.cameraId)) {
      seen.add(t.cameraId);
      result.push(t.cameraId);
    }
  }
  return result;
}

function overallTimeRange(tracklets: TrackletSummary[]): string | null {
  let earliest: number | null = null;
  let latest: number | null = null;
  let earliestIso: string | null = null;
  let latestIso: string | null = null;
  for (const t of tracklets) {
    if (t.timeStart) {
      const ts = new Date(t.timeStart).getTime();
      if (!Number.isNaN(ts) && (earliest === null || ts < earliest)) {
        earliest = ts;
        earliestIso = t.timeStart;
      }
    }
    if (t.timeEnd) {
      const ts = new Date(t.timeEnd).getTime();
      if (!Number.isNaN(ts) && (latest === null || ts > latest)) {
        latest = ts;
        latestIso = t.timeEnd;
      }
    }
  }
  return formatTimeRange(earliestIso, latestIso);
}

export function VideoCard({ video, onClick, rank }: VideoCardProps) {
  const className =
    "group flex flex-col overflow-hidden rounded-2xl border border-surface-muted bg-white text-left shadow-card transition hover:-translate-y-0.5 hover:shadow-elevated";

  const displayRank = video.rank ?? rank;
  const label = displayRank !== undefined ? `Candidate ${displayRank}` : video.title;

  const tracklets = video.tracklets ?? [];
  const trackletCount = video.trackletCount ?? tracklets.length;
  const cameras = uniqueCameras(tracklets);
  const isMulti = trackletCount > 1;

  let cameraLine: string | null = null;
  let timeLine: string | null = null;
  let countLine: string | null = null;
  if (tracklets.length > 0) {
    if (isMulti) {
      countLine = `${trackletCount} tracklets`;
      cameraLine = cameras.length ? `Cam: ${cameras.join(", ")}` : null;
      timeLine = overallTimeRange(tracklets);
    } else {
      const t = tracklets[0];
      const cam = t.cameraId ? `Cam ${t.cameraId}` : null;
      const time = formatTimeRange(t.timeStart, t.timeEnd);
      cameraLine = [cam, time].filter(Boolean).join(" · ") || null;
    }
  }

  const body = (
    <>
      <div className="relative aspect-video w-full overflow-hidden bg-surface-muted">
        <Image
          src={video.thumbnailUrl}
          alt={label || "Candidate"}
          fill
          unoptimized
          className="object-cover transition duration-300 group-hover:scale-105"
          sizes="(max-width: 768px) 50vw, 20vw"
        />
      </div>
      <div className="flex flex-1 flex-col gap-1 p-3">
        <p className="line-clamp-2 text-sm font-semibold text-ink">{label}</p>
        {video.description ? (
          <p className={`${isMulti ? "line-clamp-2" : "line-clamp-3"} text-xs leading-relaxed text-ink-secondary`}>
            {video.description}
          </p>
        ) : null}
        {countLine ? (
          <p className="truncate text-xs font-semibold text-blue-700 dark:text-blue-300">{countLine}</p>
        ) : null}
        {cameraLine ? (
          <p className="truncate text-xs text-ink-secondary dark:text-slate-400">{cameraLine}</p>
        ) : null}
        {timeLine ? (
          <p className="truncate text-xs text-ink-secondary dark:text-slate-400">{timeLine}</p>
        ) : null}
      </div>
    </>
  );

  if (onClick) {
    return (
      <button type="button" className={className} onClick={() => onClick(video)}>
        {body}
      </button>
    );
  }

  return (
    <Link href={`/detail/${video.id}`} className={className}>
      {body}
    </Link>
  );
}
