"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import { useEffect, useMemo, useRef, useState } from "react";

import { clearSession, initialsFromUser, loadSessionUser, type AuthUser } from "@/lib/auth";

export function AccountMenu() {
  const router = useRouter();
  const [open, setOpen] = useState(false);
  const [sessionUser, setSessionUser] = useState<AuthUser | null>(null);
  const rootRef = useRef<HTMLDivElement | null>(null);

  useEffect(() => {
    setSessionUser(loadSessionUser());
  }, []);

  useEffect(() => {
    function onDocumentClick(event: MouseEvent) {
      if (!rootRef.current) return;
      if (!rootRef.current.contains(event.target as Node)) {
        setOpen(false);
      }
    }
    document.addEventListener("mousedown", onDocumentClick);
    return () => {
      document.removeEventListener("mousedown", onDocumentClick);
    };
  }, []);

  const initials = useMemo(() => initialsFromUser(sessionUser), [sessionUser]);

  function handleLogout() {
    clearSession();
    router.replace("/login");
  }

  return (
    <div className="relative" ref={rootRef}>
      <button
        type="button"
        onClick={() => setOpen((prev) => !prev)}
        className="flex h-10 w-10 items-center justify-center rounded-full border border-slate-200/90 bg-white/90 text-sm font-semibold text-slate-800 shadow-[0_10px_24px_rgba(15,23,42,0.12)] transition duration-200 hover:border-sky-300 hover:bg-white dark:border-slate-700 dark:bg-slate-800 dark:text-slate-100 dark:hover:border-blue-500 dark:hover:bg-slate-700"
        aria-label="Tài khoản"
      >
        {initials}
      </button>
      {open ? (
        <div className="absolute right-0 z-40 mt-2 w-72 rounded-2xl border border-slate-200/90 bg-white/90 p-3 shadow-[0_20px_40px_rgba(15,23,42,0.18)] backdrop-blur-md dark:border-slate-800 dark:bg-slate-900/95">
          <div className="rounded-xl border border-slate-200/80 bg-slate-50 px-3 py-2 dark:border-slate-700 dark:bg-slate-800">
            <p className="text-sm font-semibold text-slate-900 dark:text-white">{sessionUser?.full_name || "Tài khoản"}</p>
            <p className="text-xs text-slate-600 dark:text-slate-400">{sessionUser?.email || "unknown@email"}</p>
            <p className="mt-1 inline-flex rounded-full bg-white px-2 py-0.5 text-[11px] font-semibold text-slate-600 dark:bg-slate-700 dark:text-slate-300">
              Role: {sessionUser?.role || "USER"}
            </p>
          </div>
          <div className="mt-2 flex flex-col gap-1">
            <Link
              href="/settings"
              onClick={() => setOpen(false)}
              className="rounded-lg px-3 py-2 text-sm font-medium text-slate-700 transition duration-200 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-200 dark:hover:bg-slate-800 dark:hover:text-white"
            >
              Cài đặt
            </Link>
            {["ADMIN", "SUPER_ADMIN"].includes(sessionUser?.role ?? "") ? (
              <Link
                href="/admin/users"
                onClick={() => setOpen(false)}
                className="rounded-lg px-3 py-2 text-sm font-medium text-slate-700 transition duration-200 hover:bg-slate-100 hover:text-slate-900 dark:text-slate-200 dark:hover:bg-slate-800 dark:hover:text-white"
              >
                Quản lý người dùng
              </Link>
            ) : null}
            <button
              type="button"
              onClick={handleLogout}
              className="rounded-lg px-3 py-2 text-left text-sm font-medium text-red-600 transition duration-200 hover:bg-red-50 dark:text-red-400 dark:hover:bg-red-500/10"
            >
              Đăng xuất
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
