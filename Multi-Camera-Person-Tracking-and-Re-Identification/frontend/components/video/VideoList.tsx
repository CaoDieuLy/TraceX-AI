import Image from "next/image";

import type { VideoClip } from "@/lib/types/video";

type VideoListProps = {
  clips: VideoClip[];
};

export function VideoList({ clips }: VideoListProps) {
  return (
    <ul className="flex flex-col gap-4">
      {clips.map((clip) => (
        <li
          key={clip.id}
          className="flex gap-4 overflow-hidden rounded-2xl border border-surface-muted bg-white p-4 shadow-card"
        >
          <div className="relative h-40 w-64 shrink-0 overflow-hidden rounded-xl bg-surface-muted">
            <Image
              src={clip.previewUrl ?? clip.thumbnailUrl}
              alt={clip.title}
              fill
              className="object-cover"
              sizes="256px"
            />
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
