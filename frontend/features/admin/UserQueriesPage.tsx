"use client";

import Link from "next/link";
import { useEffect, useState } from "react";
import { useRouter } from "next/navigation";

import { useToast } from "@/components/ui/ToastProvider";
import { getAdminUserQueries, getApiBaseUrl, type SearchHistoryItem } from "@/lib/api";
import { loadAccessToken } from "@/lib/auth";
import { parseJsonOrThrow, readApiErrorMessage } from "@/lib/api/errors";

type UserItem = {
  id: number;
  email: string;
  full_name: string;
  role: "SUPER_ADMIN" | "ADMIN" | "USER";
};

type UserQueriesPageProps = {
  userId: number;
};

export function UserQueriesPage({ userId }: UserQueriesPageProps) {
  const router = useRouter();
  const { showToast } = useToast();
  const [currentUser, setCurrentUser] = useState<UserItem | null>(null);
  const [items, setItems] = useState<SearchHistoryItem[]>([]);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    async function loadPage() {
      try {
        const token = loadAccessToken();
        const response = await fetch(`${getApiBaseUrl()}/auth/me`, {
          headers: {
            "Content-Type": "application/json",
            Authorization: `Bearer ${token ?? ""}`,
          },
        });
        if (response.status === 401) {
          showToast("Phiên đăng nhập hết hạn. Vui lòng đăng nhập lại.", "error");
          router.replace("/login");
          return;
        }
        if (!response.ok) {
          throw new Error(await readApiErrorMessage(response));
        }
        const me = await parseJsonOrThrow<UserItem>(response);
        setCurrentUser(me);
        if (!["ADMIN", "SUPER_ADMIN"].includes(me.role)) {
          showToast("Bạn không có quyền vào trang này.", "error");
          router.replace("/home");
          return;
        }
        const queries = await getAdminUserQueries(userId);
        setItems(queries);
      } catch (err) {
        const message = err instanceof Error ? err.message : "Không thể tải danh sách query.";
        showToast(message, "error");
      } finally {
        setLoading(false);
      }
    }
    void loadPage();
  }, [router, showToast, userId]);

  return (
    <div className="space-y-6">
      <section className="rounded-2xl border border-surface-muted bg-white p-5 shadow-card">
        <div className="flex items-center justify-between gap-4">
          <div>
            <h1 className="text-xl font-semibold text-ink">Query của user #{userId}</h1>
            <p className="mt-1 text-sm text-ink-secondary">
              {currentUser ? `Đăng nhập bởi ${currentUser.email}` : "Đang kiểm tra quyền truy cập..."}
            </p>
          </div>
          <Link
            href="/admin/users"
            className="rounded-xl border border-surface-muted bg-white px-4 py-2 text-sm font-medium text-ink-secondary shadow-card transition hover:border-accent hover:text-ink"
          >
            Quay lại
          </Link>
        </div>
      </section>

      <section className="rounded-2xl border border-surface-muted bg-white p-5 shadow-card">
        {loading ? <p className="text-sm text-ink-secondary">Đang tải query...</p> : null}
        {!loading && items.length === 0 ? (
          <p className="text-sm text-ink-secondary">Chưa có query nào.</p>
        ) : null}
        <div className="space-y-3">
          {items.map((item) => (
            <div
              key={item.queryId}
              className="rounded-xl border border-surface-muted bg-white px-4 py-3"
            >
              <p className="text-sm font-semibold text-ink">{item.queryText}</p>
              <p className="mt-1 text-xs text-ink-secondary">
                {new Date(item.createdAt).toLocaleString("vi-VN")}
              </p>
            </div>
          ))}
        </div>
      </section>
    </div>
  );
}
