"use client";

import Link from "next/link";
import { useCallback, useEffect, useMemo, useState } from "react";

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
  const [gridPage, setGridPage] = useState(0);
  const [selectedCandidate, setSelectedCandidate] = useState<
    { queryId: string; candidateId: string } | null
  >(null);

  useEffect(() => {
    let cancelled = false;
    setLoading(true);
    setLoadingMore(false);
    setData(null);
    setGridPage(0);
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

  const pageItems = useMemo(
    () => data?.items.slice(gridPage * GRID_BATCH_SIZE, gridPage * GRID_BATCH_SIZE + GRID_BATCH_SIZE) ?? [],
    [data?.items, gridPage],
  );
  const totalLoadedPages = data ? Math.ceil(data.items.length / GRID_BATCH_SIZE) || 1 : 1;
  const isLastLoadedPage = gridPage >= totalLoadedPages - 1 || pageItems.length === 0;
  const isLastPage = Boolean(data) && isLastLoadedPage && !data?.hasMore;
  const startRank = gridPage * GRID_BATCH_SIZE + 1;
  const endRank = data ? Math.min(startRank + pageItems.length - 1, data.totalCount) : 0;

  const handleLoadMore = useCallback(async (): Promise<boolean> => {
    if (!data || loadingMore || !data.hasMore) return false;
    setLoadingMore(true);
    try {
      const next = await getHistoryCandidates(queryId, data.items.length, GRID_BATCH_SIZE);
      if (!next.items.length) {
        setData({
          ...next,
          items: data.items,
          hasMore: false,
        });
        return false;
      }
      setData({
        ...next,
        items: [...data.items, ...next.items],
      });
      return true;
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Không tải thêm được candidates.", "error");
      return false;
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
            Đang hiển thị {startRank}–{endRank}/{data.totalCount} candidate{data.totalCount === 1 ? "" : "s"} đã lưu
            {data.selectedCandidateId ? " · đã chọn 1" : ""}
          </p>
          <VideoGrid items={pageItems} startRank={startRank} onItemClick={handleCandidateClick} />

          <div className="mt-6 rounded-2xl border border-slate-200 bg-white p-4 shadow-[0_10px_28px_rgba(15,23,42,0.06)] dark:border-slate-800 dark:bg-slate-900/90 dark:shadow-[0_20px_40px_rgba(2,6,23,0.36)]">
            <div className="flex flex-wrap items-center justify-between gap-4">
              <div className="flex items-center gap-3">
                {gridPage > 0 ? (
                  <button
                    type="button"
                    onClick={() => setGridPage(gridPage - 1)}
                    className="rounded-xl border border-slate-200 bg-white px-5 py-2.5 text-sm font-bold text-slate-700 shadow-[0_8px_20px_rgba(15,23,42,0.08)] transition duration-200 hover:border-slate-300 hover:bg-slate-50 dark:border-slate-700 dark:bg-slate-800 dark:text-slate-200 dark:hover:border-slate-600 dark:hover:bg-slate-700"
                  >
                    Trước
                  </button>
                ) : null}
                <p className="text-sm font-bold text-slate-700 dark:text-slate-200">
                  {isLastPage ? "Hết kết quả" : `Top ${startRank}–${endRank}`}
                </p>
              </div>
              <button
                type="button"
                disabled={isLastPage || loadingMore}
                onClick={async () => {
                  if (!isLastLoadedPage) {
                    setGridPage(gridPage + 1);
                    return;
                  }
                  if (data.hasMore) {
                    const added = await handleLoadMore();
                    if (added) setGridPage(gridPage + 1);
                  }
                }}
                className="rounded-xl bg-slate-950 px-5 py-2.5 text-sm font-bold text-white shadow-[0_12px_28px_rgba(15,23,42,0.22)] transition duration-200 hover:bg-slate-800 disabled:cursor-not-allowed disabled:opacity-40 dark:bg-blue-600 dark:shadow-[0_16px_32px_rgba(37,99,235,0.28)] dark:hover:bg-blue-500"
              >
                {loadingMore ? "Đang tải..." : "Tiếp theo"}
              </button>
            </div>
          </div>

          <p className="mt-10 text-center text-xs font-semibold tracking-wide text-slate-500 dark:text-slate-400">
            Mỗi trang {GRID_BATCH_SIZE} kết quả · Top n hiện tại = {data.totalCount}
          </p>
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
