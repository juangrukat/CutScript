import { Fragment, useCallback, useRef, useEffect, useMemo, useState } from 'react';
import { useEditorStore } from '../store/editorStore';
import { Virtuoso } from 'react-virtuoso';
import { Trash2, RotateCcw, Eraser } from 'lucide-react';
import type { DeletedRange } from '../types/project';
import { getSilenceRanges, type SilenceRange } from '../utils/silence';

const RANGE_MATCH_TOLERANCE_S = 0.010;

function pauseLabel(duration: number) {
  return duration >= 10 ? `${duration.toFixed(0)}s` : `${duration.toFixed(1)}s`;
}

function findDeletedSilenceRange(range: SilenceRange, deletedRanges: DeletedRange[]) {
  return deletedRanges.find((deletedRange) => {
    const isSilence = deletedRange.kind === 'silence' || deletedRange.wordIndices.length === 0;
    return (
      isSilence
      && Math.abs(deletedRange.start - range.start) < RANGE_MATCH_TOLERANCE_S
      && Math.abs(deletedRange.end - range.end) < RANGE_MATCH_TOLERANCE_S
    );
  });
}

function PauseMarker({
  range,
  deletedRange,
  onDelete,
  onRestore,
}: {
  range: SilenceRange;
  deletedRange?: DeletedRange;
  onDelete: (range: SilenceRange) => void;
  onRestore: (rangeId: string) => void;
}) {
  const duration = range.end - range.start;
  const isDeleted = !!deletedRange;
  return (
    <button
      type="button"
      data-pause-marker
      title={isDeleted ? `Restore ${pauseLabel(duration)} pause` : `Cut ${pauseLabel(duration)} pause`}
      onMouseDown={(e) => e.stopPropagation()}
      onClick={(e) => {
        e.stopPropagation();
        if (deletedRange) onRestore(deletedRange.id);
        else onDelete(range);
      }}
      className={`
        mx-1 inline-flex items-center rounded border px-1.5 py-[1px]
        font-mono text-[10px] leading-4 transition-colors
        ${isDeleted
          ? 'border-editor-danger/40 bg-editor-word-deleted text-editor-danger/80 line-through hover:bg-editor-danger/25'
          : 'border-editor-border bg-editor-surface text-editor-text-muted/80 hover:border-editor-accent/50 hover:text-editor-text'
        }
      `}
    >
      ... {pauseLabel(duration)}
    </button>
  );
}

export default function TranscriptEditor() {
  const words = useEditorStore((s) => s.words);
  const segments = useEditorStore((s) => s.segments);
  const deletedRanges = useEditorStore((s) => s.deletedRanges);
  const duration = useEditorStore((s) => s.duration);
  const selectedWordIndices = useEditorStore((s) => s.selectedWordIndices);
  const hoveredWordIndex = useEditorStore((s) => s.hoveredWordIndex);
  const setSelectedWordIndices = useEditorStore((s) => s.setSelectedWordIndices);
  const setHoveredWordIndex = useEditorStore((s) => s.setHoveredWordIndex);
  const deleteSelectedWords = useEditorStore((s) => s.deleteSelectedWords);
  const deleteSilenceRange = useEditorStore((s) => s.deleteSilenceRange);
  const deleteAllSilences = useEditorStore((s) => s.deleteAllSilences);
  const restoreRange = useEditorStore((s) => s.restoreRange);
  const clearAllDeletions = useEditorStore((s) => s.clearAllDeletions);
  const getWordAtTime = useEditorStore((s) => s.getWordAtTime);

  const selectionStart = useRef<number | null>(null);
  const wasDragging = useRef(false);
  const virtuosoRef = useRef<any>(null);

  const deletedSet = useMemo(() => {
    const s = new Set<number>();
    for (const range of deletedRanges) {
      for (const idx of range.wordIndices) s.add(idx);
    }
    return s;
  }, [deletedRanges]);

  const selectedSet = useMemo(() => new Set(selectedWordIndices), [selectedWordIndices]);

  const silenceRanges = useMemo(() => getSilenceRanges(words, duration), [words, duration]);
  const silenceBeforeWord = useMemo(() => {
    const ranges = new Map<number, SilenceRange>();
    for (const range of silenceRanges) {
      if (range.beforeWordIndex !== null) ranges.set(range.beforeWordIndex, range);
    }
    return ranges;
  }, [silenceRanges]);
  const trailingSilence = useMemo(
    () => silenceRanges.find((range) => range.beforeWordIndex === null),
    [silenceRanges],
  );
  const uncutPauseCount = useMemo(
    () => silenceRanges.filter((range) => !findDeletedSilenceRange(range, deletedRanges)).length,
    [silenceRanges, deletedRanges],
  );

  const [activeWordIndex, setActiveWordIndex] = useState(-1);

  useEffect(() => {
    if (words.length === 0) return;
    let raf: number;
    const tick = () => {
      const video = document.querySelector('video') as HTMLVideoElement | null;
      if (video) {
        const idx = getWordAtTime(video.currentTime);
        setActiveWordIndex((prev) => (prev === idx ? prev : idx));
      }
      raf = requestAnimationFrame(tick);
    };
    raf = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf);
  }, [words, getWordAtTime]);

  // Auto-scroll to active segment via Virtuoso
  useEffect(() => {
    if (activeWordIndex < 0 || segments.length === 0) return;
    const segIdx = segments.findIndex((seg) => {
      const start = seg.globalStartIndex ?? 0;
      return activeWordIndex >= start && activeWordIndex < start + seg.words.length;
    });
    if (segIdx >= 0 && virtuosoRef.current) {
      virtuosoRef.current.scrollIntoView({ index: segIdx, behavior: 'smooth', align: 'center' });
    }
  }, [activeWordIndex, segments]);

  const handleWordMouseDown = useCallback(
    (index: number, e: React.MouseEvent) => {
      e.preventDefault();
      wasDragging.current = false;
      if (e.shiftKey && selectedWordIndices.length > 0) {
        const first = selectedWordIndices[0];
        const start = Math.min(first, index);
        const end = Math.max(first, index);
        const indices = [];
        for (let i = start; i <= end; i++) indices.push(i);
        setSelectedWordIndices(indices);
      } else {
        selectionStart.current = index;
        setSelectedWordIndices([index]);
      }
    },
    [selectedWordIndices, setSelectedWordIndices],
  );

  const handleWordMouseEnter = useCallback(
    (index: number) => {
      setHoveredWordIndex(index);
      if (selectionStart.current !== null) {
        wasDragging.current = true;
        const start = Math.min(selectionStart.current, index);
        const end = Math.max(selectionStart.current, index);
        const indices = [];
        for (let i = start; i <= end; i++) indices.push(i);
        setSelectedWordIndices(indices);
      }
    },
    [setHoveredWordIndex, setSelectedWordIndices],
  );

  const handleMouseUp = useCallback(() => {
    selectionStart.current = null;
  }, []);

  const handleClickOutside = useCallback(
    (e: React.MouseEvent) => {
      if (wasDragging.current) {
        wasDragging.current = false;
        return;
      }
      if ((e.target as HTMLElement).dataset.wordIndex === undefined) {
        setSelectedWordIndices([]);
      }
    },
    [setSelectedWordIndices],
  );

  const getRangeForWord = useCallback(
    (wordIndex: number) => deletedRanges.find((r) => r.wordIndices.includes(wordIndex)),
    [deletedRanges],
  );

  const renderSegment = useCallback(
    (index: number) => {
      const segment = segments[index];
      if (!segment) return null;
      return (
        <div className="mb-3 px-4">
          {segment.speaker && (
            <div className="text-xs text-editor-accent font-medium mb-1">
              {segment.speaker}
            </div>
          )}
          <p className="text-sm leading-relaxed flex flex-wrap">
            {segment.words.map((word, localIndex) => {
              const globalIndex = (segment.globalStartIndex ?? 0) + localIndex;
              const pauseBefore = silenceBeforeWord.get(globalIndex);
              const deletedPause = pauseBefore
                ? findDeletedSilenceRange(pauseBefore, deletedRanges)
                : undefined;
              const isDeleted = deletedSet.has(globalIndex);
              const isSelected = selectedSet.has(globalIndex);
              const isActive = globalIndex === activeWordIndex;
              const isHovered = globalIndex === hoveredWordIndex;
              const deletedRange = isDeleted ? getRangeForWord(globalIndex) : null;

              return (
                <Fragment key={globalIndex}>
                  {pauseBefore && (
                    <PauseMarker
                      range={pauseBefore}
                      deletedRange={deletedPause}
                      onDelete={(range) => deleteSilenceRange(range.start, range.end)}
                      onRestore={restoreRange}
                    />
                  )}
                  <span
                    id={`word-${globalIndex}`}
                    data-word-index={globalIndex}
                    onMouseDown={(e) => handleWordMouseDown(globalIndex, e)}
                    onMouseEnter={() => handleWordMouseEnter(globalIndex)}
                    onMouseLeave={() => setHoveredWordIndex(null)}
                    className={`
                      relative px-[2px] py-[1px] rounded cursor-pointer transition-colors
                      ${isDeleted ? 'line-through text-editor-text-muted/40 bg-editor-word-deleted' : ''}
                      ${isSelected && !isDeleted ? 'bg-editor-word-selected text-white' : ''}
                      ${isActive && !isDeleted && !isSelected ? 'bg-editor-accent/20 text-editor-accent' : ''}
                      ${isHovered && !isDeleted && !isSelected && !isActive ? 'bg-editor-word-hover' : ''}
                    `}
                  >
                    {word.word}{' '}
                    {isDeleted && isHovered && deletedRange && (
                      <button
                        onClick={(e) => {
                          e.stopPropagation();
                          restoreRange(deletedRange.id);
                        }}
                        className="absolute -top-5 left-1/2 -translate-x-1/2 flex items-center gap-0.5 px-1.5 py-0.5 bg-editor-surface border border-editor-border rounded text-[10px] text-editor-success whitespace-nowrap z-10"
                      >
                        <RotateCcw className="w-2.5 h-2.5" /> Restore
                      </button>
                    )}
                  </span>
                </Fragment>
              );
            })}
            {index === segments.length - 1 && trailingSilence && (
              <PauseMarker
                range={trailingSilence}
                deletedRange={findDeletedSilenceRange(trailingSilence, deletedRanges)}
                onDelete={(range) => deleteSilenceRange(range.start, range.end)}
                onRestore={restoreRange}
              />
            )}
          </p>
        </div>
      );
    },
    [
      segments,
      deletedSet,
      selectedSet,
      activeWordIndex,
      hoveredWordIndex,
      silenceBeforeWord,
      trailingSilence,
      deletedRanges,
      handleWordMouseDown,
      handleWordMouseEnter,
      setHoveredWordIndex,
      getRangeForWord,
      restoreRange,
      deleteSilenceRange,
    ],
  );

  return (
    <div className="flex-1 flex flex-col min-h-0">
      <div className="flex items-center gap-2 px-4 py-2 border-b border-editor-border shrink-0">
        <span className="text-xs text-editor-text-muted flex-1">
          {words.length} words &middot; {silenceRanges.length} pauses &middot; {deletedRanges.length} cuts
        </span>
        {uncutPauseCount > 0 && (
          <button
            onClick={() => deleteAllSilences()}
            className="flex items-center gap-1 px-2 py-1 text-xs bg-editor-warning/20 text-editor-warning rounded hover:bg-editor-warning/30 transition-colors"
            title="Cut every detected pause"
          >
            Cut {uncutPauseCount} pauses
          </button>
        )}
        {selectedWordIndices.length > 0 && (
          <button
            onClick={deleteSelectedWords}
            className="flex items-center gap-1 px-2 py-1 text-xs bg-editor-danger/20 text-editor-danger rounded hover:bg-editor-danger/30 transition-colors"
          >
            <Trash2 className="w-3 h-3" />
            Delete {selectedWordIndices.length} words
          </button>
        )}
        {deletedRanges.length > 0 && (
          <button
            onClick={() => {
              if (window.confirm(`Restore all ${deletedRanges.length} cuts?`)) clearAllDeletions();
            }}
            className="flex items-center gap-1 px-2 py-1 text-xs bg-editor-border text-editor-text-muted rounded hover:bg-editor-surface transition-colors"
            title="Restore every cut word and pause"
          >
            <Eraser className="w-3 h-3" />
            Clear all
          </button>
        )}
      </div>

      <div
        className="flex-1 min-h-0 select-none"
        onMouseUp={handleMouseUp}
        onClick={handleClickOutside}
      >
        <Virtuoso
          ref={virtuosoRef}
          totalCount={segments.length}
          itemContent={renderSegment}
          overscan={200}
          className="h-full"
          style={{ height: '100%' }}
        />
      </div>
    </div>
  );
}
