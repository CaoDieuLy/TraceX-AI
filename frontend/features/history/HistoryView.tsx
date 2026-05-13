"use client";

import { useCallback, useEffect, useState } from "react";

import { useToast } from "@/components/ui/ToastProvider";
import { VideoGrid } from "@/components/video/VideoGrid";
import { VideoList } from "@/components/video/VideoList";
import { CandidateDetailModal } from "@/features/candidate/CandidateDetailModal";
import { useSearch } from "@/features/search/SearchContext";
import {
  getHistoryCandidates,
  getHistoryEvidence,
  getSearchHistory,
  getTraceTimeline,
  type SearchHistoryItem,
} from "@/lib/api";
import type { VideoClip, VideoItem } from "@/lib/types";

type ActivePanel =
  | { type: "candidates"; queryId: string; items: VideoItem[]; selectedCandidateId: string | null }
  | { type: "evidence"; queryId: string; clips: VideoClip[] }
  | null;

const STATUS_LABEL: Record<string, string> = {
  pending: "Đang chờ",
  searching: "Đang tìm",
  candidates_found: "Có kết quả",
  completed: "Đã truy vết",
};

function statusBadgeClass(status: string | null): string {
  if (status === "completed") {
    return "border-emerald-200 bg-emerald-50 text-emerald-700 dark:border-emerald-800/70 dark:bg-emerald-950/45 dark:text-emerald-300";
  }
  if (status === "candidates_found") {
    return "border-blue-200 bg-blue-50 text-blue-700 dark:border-blue-800/70 dark:bg-blue-950/45 dark:text-blue-300";
  }
  if (status === "searching" || status === "pending") {
    return "border-amber-200 bg-amber-50 text-amber-700 dark:border-amber-800/70 dark:bg-amber-950/35 dark:text-amber-300";
  }
  return "border-slate-200 bg-slate-100 text-slate-600 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-300";
}

export function HistoryView() {
  const { showToast } = useToast();
  const { setQuery } = useSearch();
  const [historyItems, setHistoryItems] = useState<SearchHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [activeRow, setActiveRow] = useState<string | null>(null);
  const [actionLoading, setActionLoading] = useState(false);
  const [activePanel, setActivePanel] = useState<ActivePanel>(null);
  const [selectedCandidate, setSelectedCandidate] = useState<
    { queryId: string; candidateId: string } | null
  >(null);

  useEffect(() => {
    let cancelled = false;
    setHistoryLoading(true);
    void getSearchHistory()
      .then((items) => {
        if (cancelled) return;
        setHistoryItems(items);
      })
      .catch((err) => {
        if (cancelled) return;
        showToast(err instanceof Error ? err.message : "Không thể tải lịch sử.", "error");
      })
      .finally(() => {
        if (cancelled) return;
        setHistoryLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [showToast]);

  const handleViewCandidates = useCallback(async (row: SearchHistoryItem) => {
    setActiveRow(row.queryId);
    setActionLoading(true);
    setQuery(row.queryText);
    try {
      const payload = await getHistoryCandidates(row.queryId);
      setActivePanel({
        type: "candidates",
        queryId: row.queryId,
        items: payload.items,
        selectedCandidateId: payload.selectedCandidateId,
      });
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Không tải được candidates.", "error");
    } finally {
      setActionLoading(false);
    }
  }, [setQuery, showToast]);

  const handleViewEvidence = useCallback(async (row: SearchHistoryItem) => {
    setActiveRow(row.queryId);
    setActionLoading(true);
    setQuery(row.queryText);
    try {
      const meta = await getHistoryEvidence(row.queryId);
      const timeline = await getTraceTimeline(meta.evidenceId);
      const clips: VideoClip[] = timeline.segments
        .filter((seg) => seg.videoClipUrl)
        .map((seg) => ({
          id: `${meta.evidenceId}-${seg.segmentOrder}`,
          title: seg.cameraId ? `Camera ${seg.cameraId}` : `Segment ${seg.segmentOrder}`,
          description: seg.timeStart ? new Date(seg.timeStart).toLocaleString("vi-VN") : "",
          thumbnailUrl: seg.thumbnailUrl ?? "",
          previewUrl: seg.videoClipUrl ?? undefined,
        }));
      if (clips.length === 0) {
        showToast("Evidence chưa render xong. Vui lòng thử lại sau.", "error");
        return;
      }
      setActivePanel({ type: "evidence", queryId: row.queryId, clips });
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Không tải được evidence.", "error");
    } finally {
      setActionLoading(false);
    }
  }, [setQuery, showToast]);

  const handleCandidateClick = useCallback((video: VideoItem) => {
    if (!video.queryId) {
      showToast("Không tìm thấy query_id cho candidate này.", "error");
      return;
    }
    setSelectedCandidate({ queryId: video.queryId, candidateId: video.id });
  }, [showToast]);

  return (
    <div className="mx-auto w-full max-w-4xl text-ink dark:text-slate-100">
      <div className="mb-8 text-center">
        <p className="text-xs font-black uppercase tracking-[0.16em] text-blue-700 dark:text-blue-400">Search Intelligence</p>
        <h1 className="mt-3 text-3xl font-black tracking-normal text-slate-950 dark:text-white md:text-5xl">Lịch sử truy vấn</h1>
        <p className="mx-auto mt-4 max-w-2xl text-base leading-7 text-slate-600 dark:text-slate-300">
          Chỉ hiển thị 20 truy vấn gần nhất. Video đã truy vết chỉ giữ lại cho 4 truy vấn gần nhất.
        </p>
      </div>
      {historyLoading ? <p className="mb-4 text-sm text-slate-500 dark:text-slate-400">Đang tải lịch sử...</p> : null}
      <ul className="space-y-3">
        {historyItems.map((row) => {
          const statusLabel = STATUS_LABEL[row.status ?? ""] ?? row.status ?? "—";
          const hasCandidates = row.candidateCount > 0;
          const isActive = activeRow === row.queryId;
          return (
            <li
              key={row.queryId}
              className={[
                "rounded-2xl border bg-white px-5 py-4 shadow-[0_10px_24px_rgba(15,23,42,0.05)] transition dark:bg-slate-900/90 dark:shadow-[0_20px_40px_rgba(2,6,23,0.36)]",
                isActive
                  ? "border-blue-300 dark:border-blue-500/60"
                  : "border-slate-200 hover:border-blue-200 hover:shadow-[0_16px_30px_rgba(37,99,235,0.1)] dark:border-slate-800 dark:hover:border-blue-500/60",
              ].join(" ")}
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="min-w-0 flex-1">
                  <div className="flex flex-wrap items-center gap-2">
                    <span className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                      {new Date(row.updatedAt).toLocaleString("vi-VN")}
                    </span>
                    <span className={[
                      "rounded-full border px-2 py-0.5 text-[11px] font-semibold",
                      statusBadgeClass(row.status),
                    ].join(" ")}>
                      {statusLabel}
                    </span>
                    {row.hasEvidence ? (
                      <span className="rounded-full border border-violet-200 bg-violet-50 px-2 py-0.5 text-[11px] font-semibold text-violet-700 dark:border-violet-800/70 dark:bg-violet-950/55 dark:text-violet-300">
                        Có video
                      </span>
                    ) : null}
                  </div>
                  <p className="mt-1 text-sm font-bold text-slate-900 dark:text-slate-100 md:text-base">{row.queryText}</p>
                  <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                    {row.candidateCount} candidate{row.candidateCount === 1 ? "" : "s"}
                    {row.selectedCandidateId ? " · đã chọn 1" : ""}
                  </p>
                </div>
                <div className="flex shrink-0 flex-wrap gap-2">
                  <button
                    type="button"
                    disabled={!hasCandidates || actionLoading}
                    onClick={() => void handleViewCandidates(row)}
                    className="rounded-xl border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:border-blue-300 hover:text-blue-700 disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-700 dark:bg-slate-950/40 dark:text-slate-200 dark:hover:border-blue-500/60 dark:hover:text-blue-300"
                  >
                    Xem kết quả
                  </button>
                  <button
                    type="button"
                    disabled={!row.hasEvidence || actionLoading}
                    onClick={() => void handleViewEvidence(row)}
                    className="rounded-xl border border-blue-300 bg-blue-50 px-3 py-1.5 text-xs font-semibold text-blue-700 transition hover:bg-blue-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-blue-700/60 dark:bg-blue-950/45 dark:text-blue-300 dark:hover:bg-blue-900/40"
                    title={row.hasEvidence ? "Xem video đã truy vết" : "Chưa có video truy vết (chỉ lưu 4 truy vấn gần nhất)"}
                  >
                    Xem video truy vết
                  </button>
                </div>
              </div>
            </li>
          );
        })}
      </ul>

      {activePanel?.type === "candidates" ? (
        <section className="mt-8">
          <h2 className="mb-4 text-lg font-black tracking-normal text-slate-950 dark:text-white">Candidates đã lưu</h2>
          {activePanel.items.length === 0 ? (
            <p className="text-sm text-slate-500 dark:text-slate-400">Không có candidate nào.</p>
          ) : (
            <VideoGrid items={activePanel.items} onItemClick={handleCandidateClick} />
          )}
        </section>
      ) : null}

      {activePanel?.type === "evidence" ? (
        <section className="mt-8">
          <h2 className="mb-4 text-lg font-black tracking-normal text-slate-950 dark:text-white">Video đã truy vết</h2>
          <VideoList clips={activePanel.clips} />
        </section>
      ) : null}

      <CandidateDetailModal
        open={selectedCandidate !== null}
        queryId={selectedCandidate?.queryId ?? null}
        candidateId={selectedCandidate?.candidateId ?? null}
        onClose={() => setSelectedCandidate(null)}
      />
    </div>
  );
}
