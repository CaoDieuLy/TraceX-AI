"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { useToast } from "@/components/ui/ToastProvider";
import { HistoryView } from "@/features/history/HistoryView";
import { loadSessionUser } from "@/lib/auth";

export default function HistoryPage() {
  const router = useRouter();
  const { showToast } = useToast();
  const [allowed, setAllowed] = useState(false);

  useEffect(() => {
    const user = loadSessionUser();
    if (user?.role === "SUPER_ADMIN") {
      showToast("SUPER_ADMIN khong duoc vao New/History.", "error");
      router.replace("/admin/users");
      return;
    }
    setAllowed(true);
  }, [router, showToast]);

  if (!allowed) return null;
  return <HistoryView />;
}
