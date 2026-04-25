import type { VideoClip, VideoItem } from "@/lib/types/video";
import { TOP_N_MAX } from "@/lib/constants";

const PLACEHOLDER = "https://picsum.photos/seed";

export const MOCK_VIDEO_CATALOG: VideoItem[] = Array.from({ length: 80 }, (_, i) => {
  const n = i + 1;
  return {
    id: `vid-${String(n).padStart(3, "0")}`,
    title: `Camera C${(n % 5) + 1} · Segment ${n}`,
    description: `Khung hình liên quan tìm kiếm #${n}. Độ tin cậy mô phỏng cho giao diện demo.`,
    thumbnailUrl: `${PLACEHOLDER}/mcpt${n}/400/225`,
  };
});

export function mockSearchResults(query: string, topK: number): VideoItem[] {
  const q = query.trim().toLowerCase();
  const pool = [...MOCK_VIDEO_CATALOG];
  if (q) {
    pool.sort((a, b) => {
      const score = (v: VideoItem) =>
        [v.title, v.description].some((t) => t.toLowerCase().includes(q)) ? 1 : 0;
      return score(b) - score(a);
    });
  }
  const cap = Math.min(Math.max(topK, 1), TOP_N_MAX);
  return pool.slice(0, cap);
}

export function mockDetailClips(videoId: string): VideoClip[] {
  const base = MOCK_VIDEO_CATALOG.find((v) => v.id === videoId) ?? MOCK_VIDEO_CATALOG[0];
  return Array.from({ length: 6 }, (_, i) => ({
    ...base,
    id: `${base.id}-clip-${i + 1}`,
    title: `${base.title} — đoạn ${i + 1}`,
    description: `${base.description} (preview đoạn ${i + 1})`,
    thumbnailUrl: `${PLACEHOLDER}/clip${base.id}${i}/320/180`,
    previewUrl: `${PLACEHOLDER}/prev${base.id}${i}/640/360`,
    durationLabel: `${(i + 1) * 12}s`,
  }));
}
