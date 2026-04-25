import { DetailView } from "@/features/detail/DetailView";

type PageProps = {
  params: { id: string };
};

export default function DetailPage({ params }: PageProps) {
  return <DetailView videoId={params.id} />;
}
