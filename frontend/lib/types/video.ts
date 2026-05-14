export type TrackletSummary = {
  trackletId: string;
  cameraId: string | null;
  timeStart: string | null;
  timeEnd: string | null;
};

export type VideoItem = {
  id: string;
  title: string;
  description: string;
  thumbnailUrl: string;
  queryId?: string;
  rank?: number;
  trackletCount?: number;
  tracklets?: TrackletSummary[];
};

export type VideoClip = VideoItem & {
  previewUrl?: string;
  durationLabel?: string;
};
