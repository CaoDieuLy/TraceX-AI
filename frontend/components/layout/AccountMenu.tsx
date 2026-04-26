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
        className="flex h-10 w-10 items-center justify-center rounded-full border border-surface-muted bg-white text-sm font-semibold text-ink shadow-card transition hover:bg-surface"
        aria-label="Tài khoản"
      >
        {initials}
      </button>
      {open ? (
        <div className="absolute right-0 z-40 mt-2 w-72 rounded-2xl border border-surface-muted bg-white p-3 shadow-elevated">
          <div className="rounded-xl bg-surface px-3 py-2">
            <p className="text-sm font-semibold text-ink">{sessionUser?.full_name || "Tai khoan"}</p>
            <p className="text-xs text-ink-secondary">{sessionUser?.email || "unknown@email"}</p>
            <p className="mt-1 inline-flex rounded-full bg-white px-2 py-0.5 text-[11px] font-medium text-ink-secondary">
              Role: {sessionUser?.role || "USER"}
            </p>
          </div>
          <div className="mt-2 flex flex-col gap-1">
            <Link
              href="/settings"
              onClick={() => setOpen(false)}
              className="rounded-lg px-3 py-2 text-sm text-ink-secondary transition hover:bg-surface hover:text-ink"
            >
              Settings
            </Link>
            {["ADMIN", "SUPER_ADMIN"].includes(sessionUser?.role ?? "") ? (
              <Link
                href="/admin/users"
                onClick={() => setOpen(false)}
                className="rounded-lg px-3 py-2 text-sm text-ink-secondary transition hover:bg-surface hover:text-ink"
              >
                User management
              </Link>
            ) : null}
            <button
              type="button"
              onClick={handleLogout}
              className="rounded-lg px-3 py-2 text-left text-sm text-red-600 transition hover:bg-red-50"
            >
              Dang xuat
            </button>
          </div>
        </div>
      ) : null}
    </div>
  );
}
