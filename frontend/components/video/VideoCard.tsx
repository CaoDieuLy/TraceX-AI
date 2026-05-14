import Image from "next/image";
import Link from "next/link";

import type { VideoItem } from "@/lib/types";

type VideoCardProps = {
  video: VideoItem;
  onClick?: (video: VideoItem) => void;
  rank?: number;
};

export function VideoCard({ video, onClick, rank }: VideoCardProps) {
  const className =
    "group flex flex-col overflow-hidden rounded-2xl border border-surface-muted bg-white text-left shadow-card transition hover:-translate-y-0.5 hover:shadow-elevated";

  const displayRank = video.rank ?? rank;
  const label = displayRank !== undefined ? `Candidate ${displayRank}` : video.title;

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
        <p className="line-clamp-3 text-xs leading-relaxed text-ink-secondary">{video.description}</p>
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
