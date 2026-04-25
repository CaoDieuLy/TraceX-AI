"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useMemo } from "react";

import { VideoGrid } from "@/components/video/VideoGrid";
import { GRID_BATCH_SIZE } from "@/lib/constants";
import { HOME_GUIDE_LINES, HOME_GUIDE_TITLE } from "@/lib/content/homeGuide";
import { MOCK_SEARCH_HISTORY } from "@/lib/mock/history";
import { useSearch } from "@/features/search/SearchContext";

const PAGE_SIZE = GRID_BATCH_SIZE;

function HomeViewInner() {
  const searchParams = useSearchParams();
  const view = searchParams.get("view");
  const { hasSearched, isLoading, error, hasMore, results, gridPage, setGridPage, topK, loadMore } = useSearch();

  const pageItems = useMemo(
    () => results.slice(gridPage * PAGE_SIZE, gridPage * PAGE_SIZE + PAGE_SIZE),
    [results, gridPage],
  );

  const totalPages = Math.ceil(results.length / PAGE_SIZE) || 1;
  const isLastLoadedPage = gridPage >= totalPages - 1 || pageItems.length === 0;
  const isLastPage = isLastLoadedPage && !hasMore;
  const startRank = gridPage * PAGE_SIZE + 1;
  const endRank = Math.min((gridPage + 1) * PAGE_SIZE, results.length);

  if (view === "history") {
    return (
      <div className="mx-auto max-w-2xl rounded-2xl border border-surface-muted bg-white p-8 shadow-card">
        <h1 className="text-xl font-semibold text-ink">Lịch sử tìm kiếm</h1>
        <p className="mt-2 text-sm text-ink-secondary">Dữ liệu mô phỏng cho giao diện.</p>
        <ul className="mt-6 divide-y divide-surface-muted">
          {MOCK_SEARCH_HISTORY.map((row) => (
            <li key={row.id} className="flex flex-col gap-1 py-4 first:pt-0">
              <span className="text-xs text-ink-subtle">{row.at}</span>
              <span className="text-sm font-medium text-ink">{row.query}</span>
            </li>
          ))}
        </ul>
      </div>
    );
  }

  if (!hasSearched) {
    return (
      <div className="mx-auto w-full max-w-3xl rounded-2xl border border-surface-muted bg-white p-8 shadow-card">
        <h2 className="text-lg font-semibold text-ink">{HOME_GUIDE_TITLE}</h2>
        <ol className="mt-4 list-decimal space-y-2 pl-5 text-sm leading-relaxed text-ink-secondary">
          {HOME_GUIDE_LINES.map((line) => (
              <li key={line} className="whitespace-pre-line">
                {line}
              </li>
          ))}
        </ol>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-8">
      {error ? <p className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</p> : null}
      {isLoading ? <p className="text-sm text-ink-secondary">Đang tải kết quả...</p> : null}
      {!isLoading && !error && results.length === 0 ? (
        <p className="rounded-xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm text-amber-800">
          Chưa có video để hiển thị. Backend đang trả danh sách rỗng.
        </p>
      ) : null}
      <VideoGrid items={pageItems} />

      <div className="flex flex-wrap items-center justify-between gap-4 border-t border-surface-muted pt-6">
        <p className="text-sm font-medium text-ink-secondary">
          {isLastPage ? "Top End" : `Top ${startRank}-${endRank}`}
        </p>
        <button
          type="button"
          disabled={isLastPage || isLoading}
          onClick={async () => {
            if (!isLastLoadedPage) {
              setGridPage(gridPage + 1);
              return;
            }
            if (hasMore) {
              const added = await loadMore();
              if (added) {
                setGridPage(gridPage + 1);
              }
            }
          }}
          className="rounded-xl bg-ink px-5 py-2.5 text-sm font-semibold text-white shadow-card transition hover:bg-ink/90 disabled:cursor-not-allowed disabled:opacity-40"
        >
          Next
        </button>
      </div>

      <p className="text-center text-xs text-ink-subtle">
        Mỗi lần loop {PAGE_SIZE} kết quả · Top n hiện tại = {topK}
      </p>
    </div>
  );
}

export function HomeView() {
  return (
    <Suspense
      fallback={
        <div className="flex flex-1 items-center justify-center py-24 text-sm text-ink-secondary">Đang tải…</div>
      }
    >
      <HomeViewInner />
    </Suspense>
  );
}
