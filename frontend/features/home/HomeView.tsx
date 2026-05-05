"use client";

import { useSearchParams } from "next/navigation";
import { Suspense, useEffect, useMemo, useRef } from "react";

import { VideoGrid } from "@/components/video/VideoGrid";
import { GRID_BATCH_SIZE } from "@/lib/config";
import { HOME_GUIDE_LINES, HOME_GUIDE_TITLE } from "@/lib/content";
import { MOCK_SEARCH_HISTORY } from "@/lib/mock";
import { useSearch } from "@/features/search/SearchContext";
import { useToast } from "@/components/ui/ToastProvider";

const PAGE_SIZE = GRID_BATCH_SIZE;

function HomeViewInner() {
  const { showToast } = useToast();
  const searchParams = useSearchParams();
  const view = searchParams.get("view");
  const { hasSearched, isLoading, error, hasMore, results, gridPage, setGridPage, topK, loadMore } = useSearch();
  const lastErrorRef = useRef<string | null>(null);

  const pageItems = useMemo(
    () => results.slice(gridPage * PAGE_SIZE, gridPage * PAGE_SIZE + PAGE_SIZE),
    [results, gridPage],
  );

  const totalPages = Math.ceil(results.length / PAGE_SIZE) || 1;
  const isLastLoadedPage = gridPage >= totalPages - 1 || pageItems.length === 0;
  const isLastPage = isLastLoadedPage && !hasMore;
  const startRank = gridPage * PAGE_SIZE + 1;
  const endRank = Math.min((gridPage + 1) * PAGE_SIZE, results.length);

  useEffect(() => {
    if (error && error !== lastErrorRef.current) {
      showToast(error, "error");
      lastErrorRef.current = error;
    }
  }, [error, showToast]);

  if (view === "history") {
    return (
      <div className="mx-auto w-full max-w-4xl rounded-3xl border border-slate-200/90 bg-gradient-to-br from-white/95 via-slate-50/90 to-blue-50/80 p-6 shadow-[0_18px_40px_rgba(15,23,42,0.08)] backdrop-blur-lg md:p-8">
        <p className="text-xs font-semibold uppercase tracking-[0.16em] text-sky-700">Search Intelligence</p>
        <h1 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950 md:text-3xl">Lịch sử truy vấn</h1>
        <p className="mt-2 text-sm leading-relaxed text-slate-600 md:text-base">Dữ liệu mô phỏng để kiểm tra trải nghiệm tìm kiếm và luồng phân tích truy vấn.</p>
        <ul className="mt-6 space-y-3">
          {MOCK_SEARCH_HISTORY.map((row) => (
            <li
              key={row.id}
              className="rounded-2xl border border-slate-200/80 bg-white/80 px-4 py-3 shadow-[0_8px_24px_rgba(15,23,42,0.05)] transition-[border-color,box-shadow] duration-200 hover:border-sky-200 hover:shadow-[0_14px_28px_rgba(14,116,144,0.12)]"
            >
              <span className="text-xs font-medium uppercase tracking-wide text-slate-500">{row.at}</span>
              <p className="mt-1 text-sm font-semibold text-slate-900 md:text-base">{row.query}</p>
            </li>
          ))}
        </ul>
      </div>
    );
  }

  if (!hasSearched) {
    return (
      <div className="mx-auto w-full max-w-4xl">
        <section className="rounded-3xl border border-slate-200/90 bg-gradient-to-br from-white/95 via-slate-50/95 to-blue-50/80 p-6 shadow-[0_16px_40px_rgba(15,23,42,0.09)] backdrop-blur-md md:p-8">
          <p className="text-xs font-semibold uppercase tracking-[0.16em] text-sky-700">Modern B2B SaaS</p>
          <h2 className="mt-2 text-2xl font-semibold tracking-tight text-slate-950 md:text-3xl">{HOME_GUIDE_TITLE}</h2>
          <p className="mt-3 text-sm leading-relaxed text-slate-600 md:text-base">
            Tập trung mô tả ngắn gọn và chính xác để hệ thống phân tích nhanh hơn, đối chiếu đặc điểm hiệu quả hơn và trả về kết quả phù hợp.
          </p>

          <ol className="mt-6 space-y-3">
            {HOME_GUIDE_LINES.map((line, idx) => (
              <li
                key={line}
                className="rounded-2xl border border-slate-200/80 bg-white/80 px-4 py-3 shadow-[0_8px_24px_rgba(15,23,42,0.05)]"
              >
                <p className="text-xs font-semibold uppercase tracking-[0.13em] text-sky-700">Step {idx + 1}</p>
                <p className="mt-1 text-sm leading-relaxed text-slate-800 md:text-base">{line}</p>
              </li>
            ))}
          </ol>
        </section>
      </div>
    );
  }

  return (
    <div className="flex flex-col gap-8">
      {isLoading ? (
        <p className="rounded-2xl border border-sky-100 bg-sky-50/70 px-4 py-3 text-sm font-medium text-sky-900">
          Đang tải kết quả...
        </p>
      ) : null}
      {!isLoading && !error && results.length === 0 ? (
        <p className="rounded-2xl border border-amber-200 bg-amber-50 px-4 py-3 text-sm font-medium text-amber-800">
          Chưa có video để hiển thị. Backend đang trả danh sách rỗng.
        </p>
      ) : null}
      <VideoGrid items={pageItems} />

      <div className="rounded-2xl border border-slate-200/90 bg-white/80 p-4 shadow-[0_10px_28px_rgba(15,23,42,0.06)] backdrop-blur-md">
        <div className="flex flex-wrap items-center justify-between gap-4">
          <p className="text-sm font-semibold text-slate-700">
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
            className="rounded-xl bg-[#0F172A] px-5 py-2.5 text-sm font-semibold text-white shadow-[0_12px_28px_rgba(15,23,42,0.26)] transition duration-200 hover:bg-[#1E293B] disabled:cursor-not-allowed disabled:opacity-40"
          >
            Next
          </button>
        </div>
      </div>

      <p className="text-center text-xs font-medium tracking-wide text-slate-500">
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
