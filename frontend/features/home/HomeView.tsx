"use client";

import Link from "next/link";
import { useEffect, useState } from "react";

import { VideoList } from "@/components/video/VideoList";
import { listVideos, triggerIngest } from "@/lib/api";
import { loadSessionUser } from "@/lib/auth";
import type { AuthUser } from "@/lib/auth";
import type { VideoClip } from "@/lib/types";
import { useToast } from "@/components/ui/ToastProvider";

export function HomeView() {
  const [clips, setClips] = useState<VideoClip[]>([]);
  const [isLoading, setIsLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [currentUser, setCurrentUser] = useState<AuthUser | null>(null);
  const [isImporting, setIsImporting] = useState(false);
  const [isFullPipeline, setIsFullPipeline] = useState(false);
  const [importMessage, setImportMessage] = useState<string | null>(null);
  const { showToast } = useToast();

  const isAdmin = currentUser?.role === "SUPER_ADMIN" || currentUser?.role === "ADMIN";

  useEffect(() => {
    setCurrentUser(loadSessionUser());
  }, []);

  useEffect(() => {
    let cancelled = false;
    setIsLoading(true);
    setError(null);

    void listVideos({ page: 1, pageSize: 20 })
      .then((payload) => {
        if (cancelled) return;
        setClips(payload.videos);
      })
      .catch((err) => {
        if (cancelled) return;
        setError(err instanceof Error ? err.message : "Không thể tải danh sách video.");
      })
      .finally(() => {
        if (cancelled) return;
        setIsLoading(false);
      });

    return () => {
      cancelled = true;
    };
  }, []);

  const handleImport = async () => {
    if (isImporting) return;
    setIsImporting(true);
    setImportMessage("Đang di chuyển videos từ Temp → Storage...");

    try {
      const result = await triggerIngest({ action: "move_and_ingest" });
      setImportMessage(result.message);
      showToast(result.message, "success");

      // Refresh video list
      const payload = await listVideos({ page: 1, pageSize: 20 });
      setClips(payload.videos);
    } catch (err) {
      const msg = err instanceof Error ? err.message : "Import thất bại.";
      setImportMessage(msg);
      showToast(msg, "error");
    } finally {
      setIsImporting(false);
      setTimeout(() => setImportMessage(null), 6000);
    }
  };

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-ink-subtle">Trang chủ</p>
          <h1 className="text-2xl font-semibold text-ink">Video của bạn</h1>
        </div>
        <div className="flex items-center gap-3">
          {isAdmin && (
            <button
              onClick={handleImport}
              disabled={isImporting}
              className={[
                "rounded-xl border px-4 py-2 text-sm font-medium shadow-card transition",
                isImporting
                  ? "cursor-not-allowed border-surface-muted bg-surface-muted text-ink-subtle"
                  : "border-emerald-200 bg-emerald-50 text-emerald-700 hover:border-emerald-400 hover:bg-emerald-100",
              ].join(" ")}
            >
              {isImporting ? "Đang xử lý..." : "Import Videos"}
            </button>
          )}
          <Link
            href="/guide"
            className="rounded-xl border border-surface-muted bg-white px-4 py-2 text-sm font-medium text-ink-secondary shadow-card transition hover:border-accent hover:text-ink"
          >
            Hướng dẫn sử dụng
          </Link>
        </div>
      </div>

      {importMessage ? (
        <div className={[
          "rounded-xl border px-4 py-3 text-sm",
          importMessage.toLowerCase().includes("error") || importMessage.toLowerCase().includes("thất bại")
            ? "border-red-200 bg-red-50 text-red-700"
            : "border-emerald-200 bg-emerald-50 text-emerald-700",
        ].join(" ")}>
          {importMessage}
        </div>
      ) : null}

      {isLoading ? <p className="text-sm text-ink-secondary">Đang tải...</p> : null}
      {error ? <p className="rounded-xl border border-red-200 bg-red-50 px-4 py-3 text-sm text-red-700">{error}</p> : null}
      {!isLoading && !error ? <VideoList clips={clips} /> : null}
    </div>
  );
}
