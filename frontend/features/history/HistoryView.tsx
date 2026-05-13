"use client";

import { useEffect, useState } from "react";

import { useToast } from "@/components/ui/ToastProvider";
import { VideoList } from "@/components/video/VideoList";
import { useSearch } from "@/features/search/SearchContext";
import { getSearchHistory, getVideoDetail, type SearchHistoryItem } from "@/lib/api";
import type { VideoClip } from "@/lib/types";

export function HistoryView() {
  const { showToast } = useToast();
  const { setQuery } = useSearch();
  const [historyItems, setHistoryItems] = useState<SearchHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [selectedHistoryClips, setSelectedHistoryClips] = useState<VideoClip[]>([]);

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

  return (
    <div className="mx-auto w-full max-w-4xl text-ink dark:text-slate-100">
      <div className="mb-8 text-center">
        <p className="text-xs font-black uppercase tracking-[0.16em] text-blue-700 dark:text-blue-400">Search Intelligence</p>
        <h1 className="mt-3 text-3xl font-black tracking-normal text-slate-950 dark:text-white md:text-5xl">Lịch sử truy vấn</h1>
        <p className="mx-auto mt-4 max-w-2xl text-base leading-7 text-slate-600 dark:text-slate-300">
          Chỉ hiển thị 20 truy vấn gần nhất. Bấm vào một dòng để lặp lại query.
        </p>
      </div>
      {historyLoading ? <p className="mb-4 text-sm text-slate-500 dark:text-slate-400">Đang tải lịch sử...</p> : null}
      <ul className="space-y-3">
        {historyItems.map((row) => (
          <li
            key={row.queryId}
            className="rounded-2xl border border-slate-200 bg-white px-5 py-4 shadow-[0_10px_24px_rgba(15,23,42,0.05)] transition hover:border-blue-200 hover:shadow-[0_16px_30px_rgba(37,99,235,0.1)] dark:border-slate-800 dark:bg-slate-900/90 dark:shadow-[0_20px_40px_rgba(2,6,23,0.36)] dark:hover:border-blue-500/60 dark:hover:shadow-[0_24px_44px_rgba(30,64,175,0.24)]"
          >
            <button
              type="button"
              className="w-full text-left"
              onClick={() => {
                setQuery(row.queryText);
                if (!row.videoId) {
                  setSelectedHistoryClips([]);
                  return;
                }
                void getVideoDetail(row.videoId)
                  .then((payload) => setSelectedHistoryClips(payload.segments))
                  .catch((err) => showToast(err instanceof Error ? err.message : "Không thể tải video.", "error"));
              }}
            >
              <span className="text-xs font-semibold uppercase tracking-wide text-slate-500 dark:text-slate-400">
                {new Date(row.updatedAt).toLocaleString("vi-VN")}
              </span>
              <p className="mt-1 text-sm font-bold text-slate-900 dark:text-slate-100 md:text-base">{row.queryText}</p>
              <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                {row.videoId ? `Video: ${row.videoId}` : "Chưa có video được chọn"}
              </p>
            </button>
          </li>
        ))}
      </ul>
      {selectedHistoryClips.length > 0 ? (
        <div className="mt-8">
          <VideoList clips={selectedHistoryClips} />
        </div>
      ) : null}
    </div>
  );
}
