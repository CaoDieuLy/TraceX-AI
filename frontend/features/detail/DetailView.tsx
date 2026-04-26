"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { VideoList } from "@/components/video/VideoList";
import { getVideoDetail } from "@/lib/api";
import type { VideoClip } from "@/lib/types/video";

type DetailViewProps = {
  videoId: string;
};

export function DetailView({ videoId }: DetailViewProps) {
  const [clips, setClips] = useState<VideoClip[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setError(null);

    void getVideoDetail(videoId)
      .then((payload) => {
        if (cancelled) return;
        setClips(payload.segments);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Không thể tải chi tiết video.");
      })
      .finally(() => {
        if (cancelled) return;
        setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, [videoId]);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-ink-subtle">Chi tiết</p>
          <h1 className="text-2xl font-semibold text-ink">Video · {videoId}</h1>
        </div>
        <Link
          href="/home"
          className="rounded-xl border border-surface-muted bg-white px-4 py-2 text-sm font-medium text-ink-secondary shadow-card transition hover:border-accent hover:text-ink"
        >
          ← Quay lại lưới
        </Link>
      </div>

      {isLoading ? <p className="text-sm text-ink-secondary">Đang tải đoạn video...</p> : null}
      {error ? <p className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</p> : null}
      {!isLoading && !error ? <VideoList clips={clips} /> : null}
    </div>
  );
}
