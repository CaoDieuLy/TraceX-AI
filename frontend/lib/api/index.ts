export {
  getApiBaseUrl,
  resolveMediaUrl,
  searchVideos,
  getVideoDetail,
  listVideos,
  triggerIngest,
  getIngestStats,
  triggerFullPipeline,
  getSearchHistory,
  getHistoryCandidates,
  getHistoryEvidence,
  selectHistoryVideo,
  getAdminUserQueries,
  getCandidateDetail,
  selectCandidate,
  buildTrace,
  getTraceTimeline,
} from "./client";
export { parseJsonOrThrow, readApiErrorMessage, mapBackendErrorMessage } from "./errors";
export type {
  SearchFilters,
  SearchPage,
  IngestResponse,
  IngestStatsResponse,
  SearchHistoryItem,
  HistoryCandidatesResult,
  HistoryEvidenceMeta,
} from "./client";
