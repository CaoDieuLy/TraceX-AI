"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState, type ReactNode } from "react";

const TOKEN_STORAGE_KEY = "mcpt_access_token";

export function AuthGate({ children }: { children: ReactNode }) {
  const router = useRouter();
  const [ready, setReady] = useState(false);

  useEffect(() => {
    const token = localStorage.getItem(TOKEN_STORAGE_KEY);
    if (!token) {
      router.replace("/login");
      return;
    }
    setReady(true);
  }, [router]);

  if (!ready) {
    return <div className="flex min-h-screen items-center justify-center text-sm text-ink-secondary">Đang kiểm tra phiên...</div>;
  }

  return <>{children}</>;
}
