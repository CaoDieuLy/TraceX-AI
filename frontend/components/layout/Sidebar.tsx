"use client";

import Link from "next/link";
import { usePathname, useSearchParams } from "next/navigation";
import { type ReactNode, useEffect, useState } from "react";

import { loadSessionUser, type AuthUser } from "@/lib/auth";

function NavIcon({
  active,
  children,
}: {
  active: boolean;
  children: ReactNode;
}) {
  return (
    <span
      className={[
        "flex h-8 w-8 items-center justify-center rounded-lg border transition-colors",
        active ? "border-sky-300/80 bg-sky-500/15 text-sky-200" : "border-slate-600/80 bg-slate-900/40 text-slate-300",
      ].join(" ")}
    >
      {children}
    </span>
  );
}

const linkClass = (active: boolean) =>
  [
    "flex items-center gap-3 rounded-xl px-3 py-2.5 text-sm font-medium transition-[background-color,border-color,color,box-shadow] duration-200 cursor-pointer border",
    active
      ? "border-sky-400/50 bg-sky-500/20 text-slate-50 shadow-[0_10px_22px_rgba(14,116,144,0.28)]"
      : "border-transparent text-slate-300 hover:border-slate-500/70 hover:bg-slate-800/70 hover:text-white",
  ].join(" ");

export function Sidebar() {
  const pathname = usePathname();
  const searchParams = useSearchParams();
  const view = searchParams.get("view");
  const [sessionUser, setSessionUser] = useState<AuthUser | null>(null);

  useEffect(() => {
    setSessionUser(loadSessionUser());
  }, []);

  const isHome = pathname === "/home";
  const isNew = isHome && view !== "history";
  const isHistory = isHome && view === "history";
  const isDetail = pathname.startsWith("/detail/");
  const isUsers = pathname === "/admin/users";
  const isGuide = pathname === "/guide";
  const isDashboard = pathname === "/dashboard";
  const isSettings = pathname === "/settings";
  const role = sessionUser?.role ?? "USER";
  const canAccessHome = role !== "SUPER_ADMIN";
  const canAccessUsers = ["ADMIN", "SUPER_ADMIN"].includes(role);
  const canAccessDashboard = role === "SUPER_ADMIN";

  return (
    <aside className="flex h-screen w-[252px] shrink-0 flex-col border-r border-slate-700/40 bg-gradient-to-b from-slate-950 via-slate-900 to-[#0B2442] px-3 py-6 shadow-[0_20px_40px_rgba(2,6,23,0.55)]">
      <div className="mb-8 rounded-2xl border border-slate-700/70 bg-slate-900/50 px-3 py-3 backdrop-blur-md">
        <p className="text-[11px] font-semibold uppercase tracking-[0.14em] text-slate-300">Video Search</p>
        <p className="mt-1 text-xl font-semibold tracking-tight text-slate-50">MCPT</p>
        <p className="mt-1 text-xs text-slate-300">Intelligence Workspace</p>
      </div>

      <nav className="flex flex-1 flex-col gap-1.5">
        {canAccessHome ? (
          <>
            <Link href="/home" className={linkClass(isNew)}>
              <NavIcon active={isNew}>
                <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M5 12h14M12 5v14" />
                </svg>
              </NavIcon>
              New
            </Link>
            <Link href="/home?view=history" className={linkClass(isHistory)}>
              <NavIcon active={isHistory}>
                <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8">
                  <path d="M4 12a8 8 0 1 0 2.34-5.66" />
                  <path d="M4 4v6h6" />
                </svg>
              </NavIcon>
              History
            </Link>
          </>
        ) : null}
        {canAccessUsers ? (
          <Link href="/admin/users" className={linkClass(isUsers)}>
            <NavIcon active={isUsers}>
              <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8">
                <path d="M16 19c0-2.2-1.8-4-4-4s-4 1.8-4 4" />
                <circle cx="12" cy="9" r="3" />
                <path d="M20 19a3 3 0 0 0-3-3M18 9a2.5 2.5 0 0 0-2.5-2.5" />
              </svg>
            </NavIcon>
            Users
          </Link>
        ) : null}
        <Link href="/guide" className={linkClass(isGuide)}>
          <NavIcon active={isGuide}>
            <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8">
              <path d="M5 5.5A2.5 2.5 0 0 1 7.5 3H20v16H7.5A2.5 2.5 0 0 0 5 21V5.5Z" />
              <path d="M5 21h15" />
            </svg>
          </NavIcon>
          GUIDE
        </Link>
        {canAccessDashboard ? (
          <Link href="/dashboard" className={linkClass(isDashboard)}>
            <NavIcon active={isDashboard}>
              <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8">
                <rect x="3" y="3" width="8" height="8" />
                <rect x="13" y="3" width="8" height="5" />
                <rect x="13" y="10" width="8" height="11" />
                <rect x="3" y="13" width="8" height="8" />
              </svg>
            </NavIcon>
            Dashboard
          </Link>
        ) : null}
        {isDetail ? (
          <p className="mt-4 px-3 text-xs font-medium text-slate-400">Đang xem chi tiết video</p>
        ) : null}
      </nav>

      <Link
        href="/settings"
        className={["mt-auto", linkClass(isSettings)].join(" ")}
        aria-label="Cài đặt"
      >
        <NavIcon active={isSettings}>
          <svg viewBox="0 0 24 24" className="h-4 w-4" fill="none" stroke="currentColor" strokeWidth="1.8">
            <path d="M12 8.5A3.5 3.5 0 1 0 12 15.5 3.5 3.5 0 1 0 12 8.5Z" />
            <path d="M19.4 15a1.7 1.7 0 0 0 .33 1.87l.06.06a2 2 0 0 1 0 2.83 2 2 0 0 1-2.83 0l-.06-.06A1.7 1.7 0 0 0 15 19.4a1.7 1.7 0 0 0-1 .97 1.7 1.7 0 0 1-1.58 1.13h-.84A1.7 1.7 0 0 1 10 20.37a1.7 1.7 0 0 0-1-.97 1.7 1.7 0 0 0-1.87.33l-.06.06a2 2 0 0 1-2.83 0 2 2 0 0 1 0-2.83l.06-.06A1.7 1.7 0 0 0 4.6 15a1.7 1.7 0 0 0-.97-1 1.7 1.7 0 0 1-1.13-1.58v-.84A1.7 1.7 0 0 1 3.63 10a1.7 1.7 0 0 0 .97-1 1.7 1.7 0 0 0-.33-1.87l-.06-.06a2 2 0 0 1 0-2.83 2 2 0 0 1 2.83 0l.06.06A1.7 1.7 0 0 0 9 4.6a1.7 1.7 0 0 0 1-.97A1.7 1.7 0 0 1 11.58 2.5h.84A1.7 1.7 0 0 1 14 3.63a1.7 1.7 0 0 0 1 .97 1.7 1.7 0 0 0 1.87-.33l.06-.06a2 2 0 0 1 2.83 0 2 2 0 0 1 0 2.83l-.06.06A1.7 1.7 0 0 0 19.4 9c.45.2.8.56 1 .97a1.7 1.7 0 0 1 1.13 1.58v.84A1.7 1.7 0 0 1 20.4 14c-.2.45-.56.8-.97 1Z" />
          </svg>
        </NavIcon>
        Settings
      </Link>
    </aside>
  );
}
