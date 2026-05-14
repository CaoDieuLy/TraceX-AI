import type { VideoItem } from "@/lib/types";

import { VideoCard } from "./VideoCard";

type VideoGridProps = {
  items: VideoItem[];
  onItemClick?: (video: VideoItem) => void;
  onToggleSelect?: (video: VideoItem) => void;
  selectedIds?: Set<string>;
  startRank?: number;
};

export function VideoGrid({
  items,
  onItemClick,
  onToggleSelect,
  selectedIds,
  startRank = 1,
}: VideoGridProps) {
  return (
    <div className="grid grid-cols-1 gap-5 sm:grid-cols-2 xl:grid-cols-3 2xl:grid-cols-4">
      {items.map((video, index) => (
        <VideoCard
          key={video.id}
          video={video}
          onClick={onItemClick}
          onToggleSelect={onToggleSelect}
          isSelected={selectedIds?.has(video.id) ?? false}
          rank={startRank + index}
        />
      ))}
    </div>
  );
}
