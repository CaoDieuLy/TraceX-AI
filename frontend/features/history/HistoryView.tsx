"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useCallback, useEffect, useState } from "react";

import { useToast } from "@/components/ui/ToastProvider";
import { useSearch } from "@/features/search/SearchContext";
import {
  getHistoryEvidence,
  getSearchHistory,
  type SearchHistoryItem,
} from "@/lib/api";

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
  const router = useRouter();
  const { showToast } = useToast();
  const { setQuery } = useSearch();
  const [historyItems, setHistoryItems] = useState<SearchHistoryItem[]>([]);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [evidenceLoading, setEvidenceLoading] = useState<string | null>(null);

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

  const handleViewEvidence = useCallback(async (row: SearchHistoryItem) => {
    setEvidenceLoading(row.queryId);
    setQuery(row.queryText);
    try {
      const meta = await getHistoryEvidence(row.queryId);
      const candidateParam = meta.candidateId
        ? `&candidate=${encodeURIComponent(meta.candidateId)}`
        : "";
      router.push(
        `/trace/${meta.evidenceId}?query=${encodeURIComponent(row.queryId)}${candidateParam}`,
      );
    } catch (err) {
      showToast(err instanceof Error ? err.message : "Không tải được evidence.", "error");
      setEvidenceLoading(null);
    }
  }, [router, setQuery, showToast]);

  return (
    <div className="mx-auto w-full max-w-4xl text-ink dark:text-slate-100">
      <div className="mb-8 text-center">
        <p className="text-xs font-black uppercase tracking-[0.16em] text-blue-700 dark:text-blue-400">Search Intelligence</p>
        <h1 className="mt-3 text-3xl font-black tracking-normal text-slate-950 dark:text-white md:text-5xl">Lịch sử truy vấn</h1>
        <p className="mx-auto mt-4 max-w-2xl text-base leading-7 text-slate-600 dark:text-slate-300">
          Chỉ hiển thị 20 truy vấn gần nhất. Mỗi truy vấn lưu một kết quả trace mới nhất.
        </p>
      </div>
      {historyLoading ? <p className="mb-4 text-sm text-slate-500 dark:text-slate-400">Đang tải lịch sử...</p> : null}
      <ul className="space-y-3">
        {historyItems.map((row) => {
          const statusLabel = STATUS_LABEL[row.status ?? ""] ?? row.status ?? "—";
          const hasCandidates = row.candidateCount > 0;
          const isEvidenceLoading = evidenceLoading === row.queryId;
          const queryLabel = row.queryText.trim() || (row.queryImageUrl ? "Tìm kiếm bằng ảnh" : "Truy vấn trống");
          return (
            <li
              key={row.queryId}
              className="rounded-2xl border border-slate-200 bg-white px-5 py-4 shadow-[0_10px_24px_rgba(15,23,42,0.05)] transition hover:border-blue-200 hover:shadow-[0_16px_30px_rgba(37,99,235,0.1)] dark:border-slate-800 dark:bg-slate-900/90 dark:shadow-[0_20px_40px_rgba(2,6,23,0.36)] dark:hover:border-blue-500/60 dark:hover:shadow-[0_24px_44px_rgba(30,64,175,0.24)]"
            >
              <div className="flex flex-wrap items-start justify-between gap-3">
                <div className="flex min-w-0 flex-1 gap-3">
                  {row.queryImageUrl ? (
                    <a
                      href={row.queryImageUrl}
                      target="_blank"
                      rel="noreferrer"
                      className="mt-0.5 h-20 w-20 shrink-0 overflow-hidden rounded-xl border border-slate-200 bg-slate-100 dark:border-slate-700 dark:bg-slate-800"
                      title="Mở ảnh truy vấn"
                    >
                      {/* eslint-disable-next-line @next/next/no-img-element */}
                      <img src={row.queryImageUrl} alt={queryLabel} className="h-full w-full object-cover" />
                    </a>
                  ) : null}
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
                    <p className="mt-1 text-sm font-bold text-slate-900 dark:text-slate-100 md:text-base">{queryLabel}</p>
                    <p className="mt-2 text-xs text-slate-500 dark:text-slate-400">
                      Tìm thấy {row.candidateCount} candidate{row.candidateCount === 1 ? "" : "s"}
                      {row.selectedCandidateId ? " · đã chọn 1" : ""}
                    </p>
                  </div>
                </div>
                <div className="flex shrink-0 flex-wrap gap-2">
                  {row.hasEvidence ? (
                    <button
                      type="button"
                      disabled={isEvidenceLoading}
                      onClick={() => void handleViewEvidence(row)}
                      className="rounded-xl border border-blue-300 bg-blue-50 px-3 py-1.5 text-xs font-semibold text-blue-700 transition hover:bg-blue-100 disabled:cursor-not-allowed disabled:opacity-50 dark:border-blue-700/60 dark:bg-blue-950/45 dark:text-blue-300 dark:hover:bg-blue-900/40"
                    >
                      {isEvidenceLoading ? "Đang mở..." : "Xem trace"}
                    </button>
                  ) : hasCandidates ? (
                    <Link
                      href={`/history/${encodeURIComponent(row.queryId)}/candidates`}
                      className="rounded-xl border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:border-blue-300 hover:text-blue-700 dark:border-slate-700 dark:bg-slate-950/40 dark:text-slate-200 dark:hover:border-blue-500/60 dark:hover:text-blue-300"
                    >
                      Xem kết quả
                    </Link>
                  ) : (
                    <button
                      type="button"
                      disabled
                      className="rounded-xl border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-400 dark:border-slate-700 dark:bg-slate-950/40 dark:text-slate-600"
                    >
                      Xem kết quả
                    </button>
                  )}
                  {row.hasEvidence && hasCandidates ? (
                    <Link
                      href={`/history/${encodeURIComponent(row.queryId)}/candidates`}
                      className="rounded-xl border border-slate-200 bg-white px-3 py-1.5 text-xs font-semibold text-slate-700 transition hover:border-blue-300 hover:text-blue-700 dark:border-slate-700 dark:bg-slate-950/40 dark:text-slate-200 dark:hover:border-blue-500/60 dark:hover:text-blue-300"
                    >
                      Đổi candidate
                    </Link>
                  ) : null}
                </div>
              </div>
            </li>
          );
        })}
      </ul>
    </div>
  );
}
