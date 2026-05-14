import type { VideoItem } from "@/lib/types";

import { VideoCard } from "./VideoCard";

type VideoGridProps = {
  items: VideoItem[];
  onItemClick?: (video: VideoItem) => void;
  startRank?: number;
};

export function VideoGrid({ items, onItemClick, startRank = 1 }: VideoGridProps) {
  return (
    <div className="grid grid-cols-2 gap-4 md:grid-cols-3 lg:grid-cols-5">
      {items.map((video, index) => (
        <VideoCard key={video.id} video={video} onClick={onItemClick} rank={startRank + index} />
      ))}
    </div>
  );
}
