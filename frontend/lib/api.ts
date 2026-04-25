import type { VideoClip, VideoItem } from "@/lib/types/video";

type SearchApiResponse = {
  results: Array<{
    id: string;
    thumbnail_url: string;
    description: string;
  }>;
};

type VideoDetailApiResponse = {
  id: string;
  segments: Array<{
    id: string;
    video_url: string;
    title: string;
    description: string;
  }>;
};

function isLikelyImageUrl(url: string): boolean {
  const value = url.toLowerCase();
  if (!value) return false;
  if (value.includes("/api/v1/candidates/") && value.endsWith("/preview")) {
    return true;
  }
  return /\.(png|jpg|jpeg|webp|gif|avif)(\?.*)?$/.test(value);
}

function placeholderThumbnail(seed: string): string {
  const safe = encodeURIComponent(seed || "mcpt");
  return `https://picsum.photos/seed/${safe}/400/225`;
}

function getApiBaseUrl(): string {
  const envBase = (process.env.NEXT_PUBLIC_API_BASE_URL ?? process.env.NEXT_PUBLIC_API_GATEWAY_URL ?? "").trim();
  if (envBase.startsWith("/")) {
    return envBase.replace(/\/$/, "");
  }
  if (typeof window !== "undefined") {
    // Browser dùng relative path qua Next rewrite để không phụ thuộc host networking.
    return "";
  }
  const internalApi = (process.env.INTERNAL_API_GATEWAY_URL ?? "").trim();
  if (internalApi.startsWith("http://") || internalApi.startsWith("https://")) {
    return internalApi.replace(/\/$/, "");
  }
  if (envBase.startsWith("http://") || envBase.startsWith("https://")) {
    return envBase.replace(/\/$/, "");
  }
  return "http://backend:8000";
}

function resolveMediaUrl(url: string, apiBaseUrl: string): string {
  const raw = (url ?? "").trim();
  if (!raw) {
    return "";
  }
  if (/^https?:\/\//i.test(raw)) {
    return raw;
  }
  if (raw.startsWith("/")) {
    return `${apiBaseUrl}${raw}`;
  }
  return `${apiBaseUrl}/${raw}`;
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const apiBaseUrl = getApiBaseUrl();
  const authHeaders: Record<string, string> = {};
  if (typeof window !== "undefined") {
    const token = localStorage.getItem("mcpt_access_token");
    if (token) {
      authHeaders.Authorization = `Bearer ${token}`;
    }
  }

  const response = await fetch(`${getApiBaseUrl()}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...authHeaders,
      ...(init?.headers ?? {}),
    },
    cache: "no-store",
  });

  if (!response.ok) {
    const text = await response.text();
    throw new Error(text || `Request failed (${response.status})`);
  }
  return (await response.json()) as T;
}

export async function searchVideos(query: string, topK: number, offset = 0): Promise<VideoItem[]> {
  const apiBaseUrl = getApiBaseUrl();
  const payload = await apiFetch<SearchApiResponse>("/search", {
    method: "POST",
    body: JSON.stringify({ query, top_k: topK, offset }),
  });

  return payload.results.map((item) => ({
    id: item.id,
    title: `Candidate ${item.id}`,
    description: item.description,
    thumbnailUrl: isLikelyImageUrl(item.thumbnail_url)
      ? resolveMediaUrl(item.thumbnail_url, apiBaseUrl)
      : placeholderThumbnail(item.id),
  }));
}

export async function getVideoDetail(videoId: string): Promise<{ id: string; segments: VideoClip[] }> {
  const apiBaseUrl = getApiBaseUrl();
  const payload = await apiFetch<VideoDetailApiResponse>(`/videos/${videoId}`);
  return {
    id: payload.id,
    segments: payload.segments.map((segment) => ({
      id: segment.id,
      title: segment.title,
      description: segment.description,
      thumbnailUrl: "https://picsum.photos/seed/mcpt-segment/320/180",
      previewUrl: resolveMediaUrl(segment.video_url, apiBaseUrl),
    })),
  };
}
