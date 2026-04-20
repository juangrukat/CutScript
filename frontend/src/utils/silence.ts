import type { Word } from '../types/project';

export const SILENCE_GAP_THRESHOLD_S = 0.5;

export interface SilenceRange {
  start: number;
  end: number;
  afterWordIndex: number | null;
  beforeWordIndex: number | null;
}

export function getSilenceRanges(
  words: Word[],
  duration = 0,
  threshold = SILENCE_GAP_THRESHOLD_S,
): SilenceRange[] {
  if (words.length === 0) return [];

  const ranges: SilenceRange[] = [];
  let prevEnd = 0;
  let prevIndex: number | null = null;

  for (let i = 0; i < words.length; i++) {
    const word = words[i];
    const start = Math.max(0, prevEnd);
    const end = Math.max(start, word.start);
    if (end - start >= threshold) {
      ranges.push({
        start,
        end,
        afterWordIndex: prevIndex,
        beforeWordIndex: i,
      });
    }
    if (word.end > prevEnd) {
      prevEnd = word.end;
      prevIndex = i;
    }
  }

  if (duration > 0) {
    const start = Math.max(0, prevEnd);
    const end = Math.max(start, duration);
    if (end - start >= threshold) {
      ranges.push({
        start,
        end,
        afterWordIndex: prevIndex,
        beforeWordIndex: null,
      });
    }
  }

  return ranges;
}
