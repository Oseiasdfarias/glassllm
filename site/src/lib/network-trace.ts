export type TraceValue = number | string;

export interface TraceSummary {
  id: string;
  layer: number | null;
  head: number | null;
  order: number;
  shape: number[];
  equation: string;
  preview: TraceValue[];
  min: number | null;
  max: number | null;
  mean: number | null;
  nonFinite: number;
  source: { file: string; line: number; code: string } | null;
}

export interface TraceInspection extends TraceSummary {
  iteration: number;
  grid: TraceValue[][];
  rowLabels: string[];
  columnLabels: string[];
  rowOffset: number;
  columnOffset: number;
  fullRows: number;
  fullColumns: number;
}

export const HEAD_STEPS = new Set([
  "attn.q", "attn.k", "attn.v", "attn.rope_q", "attn.rope_k",
  "attn.scores", "attn.mask", "attn.softmax", "attn.mix",
]);
