import Image from "next/image";

import type { VideoClip } from "@/lib/types";

type VideoListProps = {
  clips: VideoClip[];
};

export function VideoList({ clips }: VideoListProps) {
  const isVideoSource = (url: string | undefined) =>
    Boolean(
      url &&
        (url.includes("/file") ||
          url.includes("usercontent.google.com") ||
          url.includes("drive.google.com") ||
          url.includes("video-stream") ||
          url.includes("litng.ai") ||
          /\.(mp4|webm|ogg|mov|m3u8)(\?.*)?$/i.test(url)),
    );

  return (
    <ul className="flex flex-col gap-4">
      {clips.map((clip) => (
        <li
          key={clip.id}
          className="flex gap-4 overflow-hidden rounded-2xl border border-surface-muted bg-white p-4 shadow-card"
        >
          <div className="relative h-40 w-64 shrink-0 overflow-hidden rounded-xl bg-surface-muted">
            {isVideoSource(clip.previewUrl) ? (
              <video className="h-full w-full object-cover" controls muted preload="metadata" src={clip.previewUrl} />
            ) : (
              <Image
                src={clip.previewUrl ?? clip.thumbnailUrl}
                alt={clip.title}
                fill
                unoptimized
                className="object-cover"
                sizes="256px"
              />
            )}
            {clip.durationLabel ? (
              <span className="absolute bottom-2 right-2 rounded-md bg-black/60 px-2 py-0.5 text-xs font-medium text-white">
                {clip.durationLabel}
              </span>
            ) : null}
          </div>
          <div className="flex min-w-0 flex-1 flex-col justify-center gap-2">
            <h3 className="text-lg font-semibold text-ink">{clip.title}</h3>
            <p className="text-sm leading-relaxed text-ink-secondary">{clip.description}</p>
          </div>
        </li>
      ))}
    </ul>
  );
}
