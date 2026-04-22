import Link from "next/link";

import { VideoList } from "@/components/video/VideoList";
import { mockDetailClips } from "@/lib/mock/videos";

type PageProps = {
  params: { id: string };
};

export default function DetailPage({ params }: PageProps) {
  const clips = mockDetailClips(params.id);

  return (
    <div className="flex flex-col gap-6">
      <div className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <p className="text-xs font-semibold uppercase tracking-wide text-ink-subtle">Chi tiết</p>
          <h1 className="text-2xl font-semibold text-ink">Video · {params.id}</h1>
        </div>
        <Link
          href="/home"
          className="rounded-xl border border-surface-muted bg-white px-4 py-2 text-sm font-medium text-ink-secondary shadow-card transition hover:border-accent hover:text-ink"
        >
          ← Quay lại lưới
        </Link>
      </div>
      <VideoList clips={clips} />
    </div>
  );
}
