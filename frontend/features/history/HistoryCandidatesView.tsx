"use client";

import Link from "next/link";
import { useCallback, useEffect, useState } from "react";

import { useToast } from "@/components/ui/ToastProvider";
import { VideoGrid } from "@/components/video/VideoGrid";
import { CandidateDetailModal } from "@/features/candidate/CandidateDetailModal";
import { getHistoryCandidates, type HistoryCandidatesResult } from "@/lib/api";
import { GRID_BATCH_SIZE } from "@/lib/config";
import type { VideoItem } from "@/lib/types";

type Props = {
  queryId: string;
};

export function HistoryCandidatesView({ queryId }: Props) {
  const { showToast } = useToast();
  const [loading, setLoading] = useState(true);
  const [loadingMore, setLoadingMore] = useState(false);
  const [data, setData] = useState<HistoryCandidatesResult | null>(null);
  const [selectedCandidate, setSelectedCandidate] = useState<
    { queryId: string; candidateId: string } | null
  >(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setLoadingMore(false);
    setData(null);
    void getHistoryCandidates(queryId, 0, GRID_BATCH_SIZE)
      .then((payload) => {
        if (cancelled) return;
        setData(payload);
      })
      .catch((err) => {
        if (cancelled) return;
        showToast(err instanceof Error ? err.message : "Không tải được candidates.", "error");
      })
      .finally(() => {
        if (cancelled) return;
        setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [queryId, showToast]);

  const handleLoadMore = useCallback(async () => {
    if (!data || loadingMore || !data.hasMore) return;
    setLoadingMore(true);
    try {
      const next = await getHistoryCandidates(queryId, data.items.length, GRID_BATCH_SIZE);
      setData({
        ...next,
        items: [...data.items, ...next.items],
      });
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Không tải thêm được candidates.", "error");
    } finally {
      setLoadingMore(false);
    }
  }, [data, loadingMore, queryId, showToast]);

  const handleCandidateClick = useCallback((video: VideoItem) => {
    if (!video.queryId) {
      showToast("Không tìm thấy query_id cho candidate này.", "error");
      return;
    }
    setSelectedCandidate({ queryId: video.queryId, candidateId: video.id });
  }, [showToast]);

  return (
    <div className="mx-auto w-full max-w-6xl text-ink dark:text-slate-100">
      <div className="mb-6 flex items-center justify-between gap-4">
        <Link
          href="/history"
          className="inline-flex items-center gap-2 rounded-xl border border-slate-200 bg-white px-3 py-2 text-sm font-semibold text-slate-700 transition hover:border-blue-300 hover:text-blue-700 dark:border-slate-700 dark:bg-slate-900/70 dark:text-slate-200 dark:hover:border-blue-500/60 dark:hover:text-blue-300"
        >
          <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="2">
            <path d="M19 12H5" />
            <path d="m12 19-7-7 7-7" />
          </svg>
          Quay lại lịch sử
        </Link>
        <div className="text-right">
          <p className="text-xs font-black uppercase tracking-[0.16em] text-blue-700 dark:text-blue-400">Search Intelligence</p>
          <h1 className="mt-1 text-2xl font-black tracking-normal text-slate-950 dark:text-white md:text-3xl">Candidates đã lưu</h1>
        </div>
      </div>

      {loading ? (
        <p className="text-sm text-slate-500 dark:text-slate-400">Đang tải candidates...</p>
      ) : null}

      {!loading && data && data.items.length === 0 ? (
        <p className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-semibold text-amber-800 dark:border-amber-900/70 dark:bg-amber-950/35 dark:text-amber-200">
          Chưa có candidate nào được lưu cho truy vấn này.
        </p>
      ) : null}

      {!loading && data && data.items.length > 0 ? (
        <>
          <p className="mb-4 text-sm text-slate-500 dark:text-slate-400">
            Đang hiển thị {data.items.length}/{data.totalCount} candidate{data.totalCount === 1 ? "" : "s"} đã lưu
            {data.selectedCandidateId ? " · đã chọn 1" : ""}
          </p>
          <VideoGrid items={data.items} onItemClick={handleCandidateClick} />
          {data.hasMore ? (
            <div className="mt-6 flex justify-center">
              <button
                type="button"
                onClick={() => void handleLoadMore()}
                disabled={loadingMore}
                className="rounded-xl bg-slate-950 px-5 py-2.5 text-sm font-bold text-white shadow-[0_12px_28px_rgba(15,23,42,0.22)] transition duration-200 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-blue-600 dark:shadow-[0_16px_32px_rgba(37,99,235,0.28)] dark:hover:bg-blue-500"
              >
                {loadingMore ? "Đang tải..." : `Tải thêm ${GRID_BATCH_SIZE} candidate`}
              </button>
            </div>
          ) : null}
        </>
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
