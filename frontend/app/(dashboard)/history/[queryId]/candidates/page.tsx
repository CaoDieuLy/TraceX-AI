"use client";

import { useRouter } from "next/navigation";
import { useEffect, useState } from "react";

import { useToast } from "@/components/ui/ToastProvider";
import { HistoryCandidatesView } from "@/features/history/HistoryCandidatesView";
import { loadSessionUser } from "@/lib/auth";

type PageProps = {
  params: { queryId: string };
};

export default function HistoryCandidatesPage({ params }: PageProps) {
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
  return <HistoryCandidatesView queryId={params.queryId} />;
}
