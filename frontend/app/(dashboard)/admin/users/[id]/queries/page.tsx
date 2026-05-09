import { UserQueriesPage } from "@/features/admin/UserQueriesPage";

export default function Page({ params }: { params: { id: string } }) {
  return <UserQueriesPage userId={Number(params.id)} />;
}
