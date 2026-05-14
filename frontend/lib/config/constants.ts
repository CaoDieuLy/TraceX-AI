export const TOP_K_OPTIONS = [50] as const;
export const TOP_N_MAX = 50;
export const GRID_BATCH_SIZE = 12;

export type TopKOption = (typeof TOP_K_OPTIONS)[number];
