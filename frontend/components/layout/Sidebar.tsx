"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";

const linkClass = (active: boolean) =>
  [
    "flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-colors",
    active
      ? "bg-white text-ink shadow-card"
      : "text-ink-secondary hover:bg-white/60 hover:text-ink",
  ].join(" ");

export function Sidebar() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const view = searchParams.get("view");

  const isHome = pathname === "/home";
  const isNew = isHome && view !== "history";
  const isHistory = isHome && view === "history";
  const isDetail = pathname.startsWith("/detail/");

  return (
    <aside className="flex h-screen w-[240px] shrink-0 flex-col border-r border-surface-muted bg-surface-card px-3 py-6 shadow-card">
      <div className="mb-8 px-2">
        <p className="text-xs font-semibold uppercase tracking-wide text-ink-subtle">Video Search</p>
        <p className="mt-1 text-lg font-semibold text-ink">MCPT</p>
      </div>

      <nav className="flex flex-1 flex-col gap-1">
        <Link href="/home" className={linkClass(isNew)}>
          <span className="text-lg leading-none">✦</span>
          New
        </Link>
        <Link href="/home?view=history" className={linkClass(isHistory)}>
          <span className="text-lg leading-none">⟲</span>
          History
        </Link>
        {isDetail ? (
          <p className="mt-4 px-3 text-xs text-ink-subtle">Đang xem chi tiết video</p>
        ) : null}
      </nav>

      <button
        type="button"
        className="mt-auto flex w-full items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium text-ink-secondary transition-colors hover:bg-white/60 hover:text-ink"
        aria-label="Cài đặt"
      >
        <span className="flex h-9 w-9 items-center justify-center rounded-lg border border-surface-muted bg-surface text-base">
          ⚙
        </span>
        Settings
      </button>
    </aside>
  );
}
