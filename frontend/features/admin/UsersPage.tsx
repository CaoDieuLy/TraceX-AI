"use client";

import { FormEvent, useEffect, useMemo, useState } from "react";
import Link from "next/link";
import { useRouter } from "next/navigation";

import { loadAccessToken } from "@/lib/auth";
import { getApiBaseUrl } from "@/lib/api";
import { mapBackendErrorMessage, parseJsonOrThrow, readApiErrorMessage } from "@/lib/api/errors";
import { useToast } from "@/components/ui/ToastProvider";

type UserItem = {
  id: number;
  email: string;
  full_name: string;
  role: "SUPER_ADMIN" | "ADMIN" | "USER";
  is_active: boolean;
  last_login?: string | null;
};

const ROLE_RANK: Record<UserItem["role"], number> = {
  USER: 1,
  ADMIN: 2,
  SUPER_ADMIN: 3,
};
const ROLE_OPTIONS: UserItem["role"][] = ["SUPER_ADMIN", "ADMIN", "USER"];
const ROLE_BADGE_CLASS: Record<UserItem["role"], string> = {
  SUPER_ADMIN: "bg-violet-100 text-violet-700 border-violet-200 dark:bg-violet-500/15 dark:text-violet-200 dark:border-violet-500/30",
  ADMIN: "bg-blue-100 text-blue-700 border-blue-200 dark:bg-blue-500/15 dark:text-blue-200 dark:border-blue-500/30",
  USER: "bg-slate-100 text-slate-700 border-slate-200 dark:bg-slate-800 dark:text-slate-200 dark:border-slate-700",
};

const fieldClass =
  "rounded-xl border border-surface-muted bg-white px-3 py-2 text-slate-900 outline-none ring-blue-300/25 transition focus:border-blue-400 focus:ring-2 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-100 dark:placeholder:text-slate-500";

export function UsersPage() {
  const router = useRouter();
  const { showToast } = useToast();
  const [currentUser, setCurrentUser] = useState<UserItem | null>(null);
  const [items, setItems] = useState<UserItem[]>([]);
  const [loading, setLoading] = useState(false);
  const [email, setEmail] = useState("");
  const [fullName, setFullName] = useState("");
  const [password, setPassword] = useState("");
  const [role, setRole] = useState<UserItem["role"]>("USER");
  const [isActive, setIsActive] = useState(true);
  const currentUserRole = (currentUser?.role ?? "USER") as UserItem["role"];
  const currentUserRank = ROLE_RANK[currentUserRole] ?? 1;
  const currentUserId = currentUser?.id ?? 0;

  async function authFetch(path: string, init?: RequestInit): Promise<Response> {
    const token = loadAccessToken();
    return fetch(`${getApiBaseUrl()}${path}`, {
      ...init,
      headers: {
        "Content-Type": "application/json",
        Authorization: `Bearer ${token ?? ""}`,
        ...(init?.headers ?? {}),
      },
    });
  }

  async function loadUsers() {
    setLoading(true);
    try {
      const response = await authFetch("/users");
      if (response.status === 401) {
        showToast("Phien dang nhap het han. Vui long dang nhap lai.", "error");
        router.replace("/login");
        return;
      }
      if (response.status === 403) {
        showToast("Ban khong co quyen admin.", "error");
        return;
      }
      if (!response.ok) throw new Error(await readApiErrorMessage(response));
      const payload = await parseJsonOrThrow<{ items: UserItem[] }>(response);
      setItems(payload.items ?? []);
    } catch (err) {
      const message = err instanceof Error ? err.message : "Tai danh sach user that bai.";
      showToast(mapBackendErrorMessage(message), "error");
    } finally {
      setLoading(false);
    }
  }

  async function loadCurrentUser() {
    const response = await authFetch("/auth/me");
    if (response.status === 401) {
      showToast("Phien dang nhap het han. Vui long dang nhap lai.", "error");
      router.replace("/login");
      return null;
    }
    if (!response.ok) {
      throw new Error(await readApiErrorMessage(response));
    }
    return await parseJsonOrThrow<UserItem>(response);
  }

  useEffect(() => {
    void (async () => {
      try {
        const me = await loadCurrentUser();
        if (!me) return;
        setCurrentUser(me);
        if (!["ADMIN", "SUPER_ADMIN"].includes(me.role)) {
          showToast("Ban khong co quyen vao trang quan ly user.", "error");
          router.replace("/home");
          return;
        }
        await loadUsers();
      } catch (err) {
        const message = err instanceof Error ? err.message : "Khong the tai thong tin nguoi dung.";
        showToast(mapBackendErrorMessage(message), "error");
      }
    })();
  }, [router]);

  async function handleCreateUser(event: FormEvent<HTMLFormElement>) {
    event.preventDefault();
    const response = await authFetch("/users", {
      method: "POST",
      body: JSON.stringify({ email, full_name: fullName, password, role, is_active: isActive }),
    });
    if (!response.ok) {
      showToast(mapBackendErrorMessage(await readApiErrorMessage(response)), "error");
      return;
    }
    setEmail("");
    setFullName("");
    setPassword("");
    setRole("USER");
    setIsActive(true);
    showToast("Tao user thanh cong.", "success");
    await loadUsers();
  }

  async function handleUpdateUser(userId: number, patch: { role?: UserItem["role"]; is_active?: boolean }) {
    const response = await authFetch(`/users/${userId}`, {
      method: "PATCH",
      body: JSON.stringify(patch),
    });
    if (!response.ok) {
      showToast(mapBackendErrorMessage(await readApiErrorMessage(response)), "error");
      return;
    }
    showToast("Cap nhat user thanh cong.", "success");
    await loadUsers();
  }

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-surface-muted bg-white p-5 shadow-card dark:border-slate-800 dark:bg-slate-900 dark:shadow-[0_14px_34px_rgba(0,0,0,0.28)]">
        <h1 className="text-xl font-semibold text-ink dark:text-white">Người dùng</h1>
        <p className="mt-1 text-sm text-ink-secondary dark:text-slate-300">Quản lý người dùng (ADMIN và SUPER_ADMIN).</p>
      </section>

      <section className="rounded-2xl border border-surface-muted bg-white p-5 shadow-card dark:border-slate-800 dark:bg-slate-900 dark:shadow-[0_14px_34px_rgba(0,0,0,0.28)]">
        <form className="grid grid-cols-1 gap-3 md:grid-cols-12" onSubmit={handleCreateUser}>
          <input
            className={[fieldClass, "md:col-span-3"].join(" ")}
            placeholder="Email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
          />
          <input
            className={[fieldClass, "md:col-span-3"].join(" ")}
            placeholder="Họ tên"
            value={fullName}
            onChange={(e) => setFullName(e.target.value)}
            required
          />
          <input
            className={[fieldClass, "md:col-span-2"].join(" ")}
            placeholder="Mật khẩu"
            type="password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
          />
          <select
            className={[fieldClass, "md:col-span-2"].join(" ")}
            value={isActive ? "active" : "inactive"}
            onChange={(e) => setIsActive(e.target.value === "active")}
          >
            <option value="active">Đang hoạt động</option>
            <option value="inactive">Không hoạt động</option>
          </select>
          <div className="flex min-w-0 gap-2 md:col-span-2">
            <select
              className={[fieldClass, "min-w-0 flex-1"].join(" ")}
              value={role}
              onChange={(e) => setRole(e.target.value as UserItem["role"])}
            >
              {ROLE_OPTIONS.map((item) => (
                <option key={item} value={item}>
                  {item}
                </option>
              ))}
            </select>
            <button className="shrink-0 rounded-xl bg-accent px-4 py-2 font-semibold text-white transition hover:opacity-90" type="submit">
              Tạo
            </button>
          </div>
        </form>
      </section>

      <section className="rounded-2xl border border-surface-muted bg-white p-5 shadow-card dark:border-slate-800 dark:bg-slate-900 dark:shadow-[0_14px_34px_rgba(0,0,0,0.28)]">
        {loading ? <p className="text-sm text-ink-secondary dark:text-slate-300">Đang tải người dùng...</p> : null}
        <div className="space-y-2">
          {items.map((user) => {
            const isSelf = user.id === currentUserId;
            const targetRank = ROLE_RANK[user.role] ?? 1;
            const canManageTarget = targetRank < currentUserRank;
            return (
              <div key={user.id} className="flex items-center justify-between rounded-xl border border-surface-muted bg-white px-3 py-2 dark:border-slate-800 dark:bg-slate-950">
                <div>
                  <p className="flex items-center gap-2 text-sm font-medium text-ink dark:text-slate-100">
                    <span>{user.email}</span>
                    <span className={["rounded-full border px-2 py-0.5 text-[11px] font-semibold", ROLE_BADGE_CLASS[user.role]].join(" ")}>
                      {user.role}
                    </span>
                  </p>
                  <p className="text-xs text-ink-secondary dark:text-slate-400">
                    {user.full_name} · {user.is_active ? "Đang hoạt động" : "Không hoạt động"}
                  </p>
                </div>
                <div className="flex items-center gap-2">
                  <select
                    className="rounded-lg border border-surface-muted bg-white px-2 py-1 text-sm text-slate-900 outline-none dark:border-slate-800 dark:bg-slate-950 dark:text-slate-100"
                    value={user.role}
                    disabled={isSelf || !canManageTarget}
                    onChange={(e) => void handleUpdateUser(user.id, { role: e.target.value as UserItem["role"] })}
                  >
                    {ROLE_OPTIONS.map((item) => (
                      <option key={item} value={item}>
                        {item}
                      </option>
                    ))}
                  </select>
                  <button
                    type="button"
                    disabled={isSelf || !canManageTarget}
                    onClick={() => void handleUpdateUser(user.id, { is_active: !user.is_active })}
                    className="rounded-lg border border-surface-muted bg-white px-2 py-1 text-xs text-ink-secondary transition hover:bg-surface disabled:cursor-not-allowed disabled:opacity-50 dark:border-slate-800 dark:bg-slate-950 dark:text-slate-300 dark:hover:bg-slate-800"
                  >
                    {user.is_active ? "Vô hiệu hóa" : "Kích hoạt"}
                  </button>
                  <Link
                    href={`/admin/users/${user.id}/queries`}
                    className="rounded-lg border border-surface-muted bg-white px-2 py-1 text-xs text-ink-secondary transition hover:bg-surface dark:border-slate-800 dark:bg-slate-950 dark:text-slate-300 dark:hover:bg-slate-800"
                  >
                    Queries
                  </Link>
                </div>
              </div>
            );
          })}
        </div>
      </section>
    </div>
  );
}
