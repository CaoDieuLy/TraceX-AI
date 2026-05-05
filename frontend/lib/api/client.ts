import type { VideoClip, VideoItem } from "@/lib/types";
import { loadAccessToken } from "@/lib/auth";
import { parseJsonOrThrow, readApiErrorMessage } from "@/lib/api/errors";

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

export type SearchFilters = {
  camera_ids?: string[];
  time_from?: string;
  time_to?: string;
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

/** Base URL cho fetch từ browser (ưu tiên same-origin + rewrite) hoặc SSR/server. */
export function getApiBaseUrl(): string {
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
  return "http://metadata-service:8000";
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
    let resolved = `${apiBaseUrl}${raw}`;
    if (typeof window !== "undefined" && raw.includes("/api/v1/candidates/") && raw.endsWith("/preview")) {
      const token = loadAccessToken();
      if (token) {
        const separator = resolved.includes("?") ? "&" : "?";
        resolved = `${resolved}${separator}access_token=${encodeURIComponent(token)}`;
      }
    }
    return resolved;
  }
  return `${apiBaseUrl}/${raw}`;
}

async function apiFetch<T>(path: string, init?: RequestInit): Promise<T> {
  const apiBaseUrl = getApiBaseUrl();
  const authHeaders: Record<string, string> = {};
  if (typeof window !== "undefined") {
    const token = loadAccessToken();
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
    throw new Error(await readApiErrorMessage(response));
  }
  return parseJsonOrThrow<T>(response);
}

export async function searchVideos(query: string, topK: number, offset = 0, filters?: SearchFilters): Promise<VideoItem[]> {
  const apiBaseUrl = getApiBaseUrl();
  const payloadBody: {
    query: string;
    top_k: number;
    offset: number;
    camera_ids?: string[];
    time_from?: string;
    time_to?: string;
  } = { query, top_k: topK, offset };
  if (filters?.camera_ids?.length) {
    payloadBody.camera_ids = filters.camera_ids;
  }
  if (filters?.time_from) {
    payloadBody.time_from = filters.time_from;
  }
  if (filters?.time_to) {
    payloadBody.time_to = filters.time_to;
  }
  const payload = await apiFetch<SearchApiResponse>("/search", {
    method: "POST",
    body: JSON.stringify(payloadBody),
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
