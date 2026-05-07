"use client";

import Link from "next/link";
import { useEffect, useMemo, useState } from "react";

import { clearSession, loadAccessToken, loadSessionUser, type AuthUser } from "@/lib/auth";
import { getApiBaseUrl } from "@/lib/api";
import { mapBackendErrorMessage, parseJsonOrThrow, readApiErrorMessage } from "@/lib/api/errors";
import { useToast } from "@/components/ui/ToastProvider";

type MeResponse = {
  id: number;
  email: string;
  full_name: string;
  role: "SUPER_ADMIN" | "ADMIN" | "USER";
  is_active: boolean;
  last_login?: string | null;
  created_at?: string;
};

type OverviewResponse = {
  metrics?: {
    total_users?: number;
    total_managed_videos?: number;
    total_queries?: number;
    total_candidates?: number;
    total_cameras?: number;
    total_candidate_videos?: number;
    total_queue_videos?: number;
  };
};

function formatDateTime(value?: string | null): string {
  if (!value) return "--";
  const date = new Date(value);
  if (Number.isNaN(date.getTime())) return "--";
  return date.toLocaleString("vi-VN");
}

export function SettingsPage() {
  const { showToast } = useToast();
  const [sessionUser, setSessionUser] = useState<AuthUser | null>(null);
  const [me, setMe] = useState<MeResponse | null>(null);
  const [overview, setOverview] = useState<OverviewResponse | null>(null);
  const [loading, setLoading] = useState(false);

  const isAdmin = useMemo(
    () => ["ADMIN", "SUPER_ADMIN"].includes((me?.role || sessionUser?.role || "").toUpperCase()),
    [me?.role, sessionUser?.role],
  );

  async function authFetch(path: string): Promise<Response> {
    const token = loadAccessToken();
    return fetch(`${getApiBaseUrl()}${path}`, {
      headers: {
        Authorization: `Bearer ${token ?? ""}`,
      },
      cache: "no-store",
    });
  }

  async function loadSettingsData() {
    setLoading(true);
    try {
      const meRes = await authFetch("/v1/auth/me");
      if (!meRes.ok) throw new Error(await readApiErrorMessage(meRes));
      const mePayload = await parseJsonOrThrow<MeResponse>(meRes);
      setMe(mePayload);

      // Overview is optional for settings UX; don't block page on downstream AI auth failures.
      const overviewRes = await authFetch("/v1/overview");
      if (overviewRes.ok) {
        const overviewPayload = await parseJsonOrThrow<OverviewResponse>(overviewRes);
        setOverview(overviewPayload);
      } else {
        setOverview(null);
      }
    } catch (err) {
      const message = err instanceof Error ? err.message : "Khong the tai settings.";
      showToast(mapBackendErrorMessage(message), "error");
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    setSessionUser(loadSessionUser());
    void loadSettingsData();
  }, []);

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-surface-muted bg-white p-5 shadow-card">
        <h1 className="text-xl font-semibold text-ink">Settings</h1>
        <p className="mt-1 text-sm text-ink-secondary">Quan ly tai khoan, phien dang nhap va thong tin he thong.</p>
      </section>

      <section className="grid gap-4 md:grid-cols-2">
        <div className="rounded-2xl border border-surface-muted bg-white p-5 shadow-card">
          <p className="text-sm font-semibold text-ink">Thong tin tai khoan</p>
          {loading ? <p className="mt-2 text-sm text-ink-secondary">Dang tai...</p> : null}
          {!loading ? (
            <dl className="mt-3 space-y-2 text-sm">
              <div className="flex justify-between gap-3">
                <dt className="text-ink-secondary">Email</dt>
                <dd className="font-medium text-ink">{me?.email || sessionUser?.email || "--"}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-ink-secondary">Role</dt>
                <dd className="font-medium text-ink">{me?.role || sessionUser?.role || "--"}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-ink-secondary">Trang thai</dt>
                <dd className="font-medium text-ink">{me?.is_active === false ? "Inactive" : "Active"}</dd>
              </div>
              <div className="flex justify-between gap-3">
                <dt className="text-ink-secondary">Lan dang nhap cuoi</dt>
                <dd className="font-medium text-ink">{formatDateTime(me?.last_login)}</dd>
              </div>
            </dl>
          ) : null}
        </div>

        <div className="rounded-2xl border border-surface-muted bg-white p-5 shadow-card">
          <p className="text-sm font-semibold text-ink">Tac vu nhanh</p>
          <div className="mt-3 flex flex-col gap-2">
            <button
              type="button"
              onClick={async () => {
                await loadSettingsData();
                showToast("Da refresh du lieu settings.", "success");
              }}
              className="rounded-xl border border-surface-muted px-3 py-2 text-left text-sm text-ink transition hover:bg-surface"
            >
              Refresh du lieu
            </button>
            {isAdmin ? (
              <Link
                href="/admin/users"
                className="rounded-xl border border-surface-muted px-3 py-2 text-sm text-ink transition hover:bg-surface"
              >
                Mo User management
              </Link>
            ) : null}
            <button
              type="button"
              onClick={() => {
                clearSession();
                window.location.href = "/login";
              }}
              className="rounded-xl border border-red-200 px-3 py-2 text-left text-sm text-red-600 transition hover:bg-red-50"
            >
              Dang xuat tai khoan
            </button>
          </div>
        </div>
      </section>

      <section className="rounded-2xl border border-surface-muted bg-white p-5 shadow-card">
        <p className="text-sm font-semibold text-ink">Thong tin he thong</p>
        <div className="mt-3 grid gap-3 sm:grid-cols-2 lg:grid-cols-4">
          <div className="rounded-xl border border-surface-muted bg-surface px-3 py-2">
            <p className="text-xs text-ink-secondary">Users</p>
            <p className="text-lg font-semibold text-ink">{overview?.metrics?.total_users ?? "--"}</p>
          </div>
          <div className="rounded-xl border border-surface-muted bg-surface px-3 py-2">
            <p className="text-xs text-ink-secondary">Managed videos</p>
            <p className="text-lg font-semibold text-ink">{overview?.metrics?.total_managed_videos ?? "--"}</p>
          </div>
          <div className="rounded-xl border border-surface-muted bg-surface px-3 py-2">
            <p className="text-xs text-ink-secondary">Queries</p>
            <p className="text-lg font-semibold text-ink">{overview?.metrics?.total_queries ?? "--"}</p>
          </div>
          <div className="rounded-xl border border-surface-muted bg-surface px-3 py-2">
            <p className="text-xs text-ink-secondary">Queue videos</p>
            <p className="text-lg font-semibold text-ink">{overview?.metrics?.total_queue_videos ?? "--"}</p>
          </div>
        </div>
      </section>
    </div>
  );
}
