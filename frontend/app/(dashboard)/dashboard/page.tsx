"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { useToast } from "@/components/ui/ToastProvider";
import { loadSessionUser } from "@/lib/auth";

export default function DashboardPlaceholderPage() {
  const router = useRouter();
  const { showToast } = useToast();
  const [allowed, setAllowed] = useState(false);

  useEffect(() => {
    const user = loadSessionUser();
    if (user?.role !== "SUPER_ADMIN") {
      showToast("Trang Dashboard chi danh cho SUPER_ADMIN.", "error");
      router.replace("/home");
      return;
    }
    setAllowed(true);
  }, [router, showToast]);

  if (!allowed) return null;
  return <div className="min-h-[calc(100vh-12rem)] w-full rounded-2xl bg-white" aria-label="Dashboard" />;
}
