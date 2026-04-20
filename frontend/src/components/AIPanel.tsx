import { useCallback, useMemo, useState } from 'react';
import { useEditorStore } from '../store/editorStore';
import { useAIStore, type ClipDuration } from '../store/aiStore';
import {
  AlertCircle,
  Check,
  Crosshair,
  Download,
  Film,
  Folder,
  Loader2,
  Play,
  Scissors,
  Sparkles,
  Wand2,
  X,
} from 'lucide-react';
import type { ClipSuggestion, FocusMode, FocusPlan } from '../types/project';

type Tab = 'filler' | 'clips' | 'focus';

const CLIP_DURATION_OPTIONS: ClipDuration[] = [15, 30, 60, 90];

const FOCUS_MODE_META: Record<FocusMode, { label: string; description: string }> = {
  redundancy: {
    label: 'Remove repetition',
    description: 'Trim near-duplicate sentences and repeated points.',
  },
  tighten: {
    label: 'Tighten pace',
    description: 'Cut throat-clearing, false starts, and tangents.',
  },
  key_points: {
    label: 'Keep key points',
    description: 'Keep thesis and main claims, drop examples.',
  },
  qa_extract: {
    label: 'Q&A only',
    description: 'Keep only questions and direct answers.',
  },
  topic: {
    label: 'Focus on topic',
    description: 'Keep content related to a topic you specify.',
  },
};

export default function AIPanel() {
  const {
    words,
    videoPath,
    backendUrl,
    deletedRanges,
    deleteWordRange,
    setCurrentTime,
  } = useEditorStore();
  const {
    defaultProvider,
    providers,
    customFillerWords,
    fillerResult,
    clipSuggestions,
    clipRationale,
    clipWarnings,
    clipDurations,
    clipSaveLocation,
    focusPlan,
    isProcessing,
    processingMessage,
    setCustomFillerWords,
    setFillerResult,
    setClipResult,
    setClipDurations,
    setClipSaveLocation,
    setFocusPlan,
    setProcessing,
  } = useAIStore();

  const [activeTab, setActiveTab] = useState<Tab>('filler');
  const [aiError, setAiError] = useState<string | null>(null);
  const [focusMode, setFocusMode] = useState<FocusMode>('tighten');
  const [focusTopic, setFocusTopic] = useState('');

  const requiresApiKey = defaultProvider === 'openai' || defaultProvider === 'claude';
  const config = providers[defaultProvider];
  const missingKey = requiresApiKey && !config.apiKey;

  const deletedIndicesSet = useMemo(() => {
    const set = new Set<number>();
    deletedRanges.forEach((r) => r.wordIndices.forEach((i) => set.add(i)));
    return set;
  }, [deletedRanges]);

  const deletedIndicesArray = useMemo(
    () => [...deletedIndicesSet].sort((a, b) => a - b),
    [deletedIndicesSet],
  );

  // Full payload (used by clip export where the backend needs all words for
  // AcousticMap refinement).
  const wordsPayload = useMemo(
    () => words.map((w, i) => ({ index: i, word: w.word, start: w.start, end: w.end })),
    [words],
  );

  // Kept-only payload — what the AI should reason over. Feeding already-
  // deleted words to the AI caused confusing behaviour (fillers were being
  // "re-flagged" inside deleted ranges, so Apply All produced no visible
  // change). Original indices are preserved so the apply path still works.
  const keptWordsPayload = useMemo(
    () =>
      words
        .map((w, i) => ({ index: i, word: w.word, start: w.start, end: w.end }))
        .filter((w) => !deletedIndicesSet.has(w.index)),
    [words, deletedIndicesSet],
  );

  const keptTranscript = useMemo(
    () => keptWordsPayload.map((w) => w.word).join(' '),
    [keptWordsPayload],
  );

  // Per-duration bucket index for filename composition.
  const bucketIndices = useMemo(() => {
    const counts = new Map<number, number>();
    return clipSuggestions.map((c) => {
      const n = (counts.get(c.target_duration) || 0) + 1;
      counts.set(c.target_duration, n);
      return n;
    });
  }, [clipSuggestions]);

  const sourceBasename = useMemo(() => {
    if (!videoPath) return 'clip';
    const sep = videoPath.includes('\\') ? '\\' : '/';
    const last = videoPath.split(sep).pop() || 'clip';
    return last.replace(/\.[^.]+$/, '') || 'clip';
  }, [videoPath]);

  const sourceDir = useMemo(() => {
    if (!videoPath) return '';
    const sep = videoPath.includes('\\') ? '\\' : '/';
    return videoPath.substring(0, videoPath.lastIndexOf(sep));
  }, [videoPath]);

  const effectiveSaveDir = clipSaveLocation || sourceDir;

  const pickSaveLocation = useCallback(async () => {
    if (!window.electronAPI?.openDirectory) return;
    const picked = await window.electronAPI.openDirectory({
      defaultPath: effectiveSaveDir || undefined,
      title: 'Choose where to save clips',
    });
    if (picked) setClipSaveLocation(picked);
  }, [effectiveSaveDir, setClipSaveLocation]);

  function buildClipFilename(clip: ClipSuggestion, bucketIndex: number): string {
    const safeBase = sourceBasename.replace(/[/\\:*?"<>|]/g, '_').substring(0, 40);
    const safeTitle = (clip.title || '')
      .replace(/[/\\:*?"<>|]/g, '')
      .trim()
      .replace(/\s+/g, '_')
      .substring(0, 50);
    return safeTitle
      ? `${safeBase}_${clip.target_duration}s_${safeTitle}_${bucketIndex}.mp4`
      : `${safeBase}_${clip.target_duration}s_${bucketIndex}.mp4`;
  }

  const checkReady = useCallback((): boolean => {
    setAiError(null);
    if (words.length === 0) {
      setAiError('No transcript loaded.');
      return false;
    }
    if (missingKey) {
      setAiError(`No API key set for ${defaultProvider}. Add it in Settings.`);
      return false;
    }
    return true;
  }, [words.length, missingKey, defaultProvider]);

  // ---------------------------------------------------------------------
  // Filler detection
  // ---------------------------------------------------------------------
  const detectFillers = useCallback(async () => {
    if (!checkReady()) return;
    setProcessing(true, 'Detecting filler words...');
    try {
      const res = await fetch(`${backendUrl}/ai/filler-removal`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transcript: keptTranscript,
          words: keptWordsPayload.map(({ index, word }) => ({ index, word })),
          provider: defaultProvider,
          model: config.model,
          api_key: config.apiKey || undefined,
          base_url: config.baseUrl || undefined,
          custom_filler_words: customFillerWords || undefined,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      setFillerResult(await res.json());
    } catch (err: any) {
      console.error(err);
      setAiError(err.message ?? 'Filler detection failed');
    } finally {
      setProcessing(false);
    }
  }, [
    checkReady, backendUrl, keptTranscript, keptWordsPayload, defaultProvider,
    config, customFillerWords, setProcessing, setFillerResult,
  ]);

  const applyFillerDeletions = useCallback(() => {
    if (!fillerResult) return;
    const sorted = [...fillerResult.fillerWords].sort((a, b) => b.index - a.index);
    for (const fw of sorted) deleteWordRange(fw.index, fw.index);
    setFillerResult(null);
  }, [fillerResult, deleteWordRange, setFillerResult]);

  // ---------------------------------------------------------------------
  // Clip suggestions
  // ---------------------------------------------------------------------
  const toggleDuration = (d: ClipDuration) => {
    const next = clipDurations.includes(d)
      ? clipDurations.filter((x) => x !== d)
      : [...clipDurations, d].sort((a, b) => a - b);
    setClipDurations(next);
  };

  const createClips = useCallback(async () => {
    if (!checkReady()) return;
    const durations = clipDurations.length ? clipDurations : [60];
    setProcessing(true, 'Finding clip candidates...');
    try {
      const res = await fetch(`${backendUrl}/ai/create-clip`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transcript: keptTranscript,
          words: keptWordsPayload,
          provider: defaultProvider,
          model: config.model,
          api_key: config.apiKey || undefined,
          base_url: config.baseUrl || undefined,
          target_duration: durations[0],
          target_durations: durations,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();
      setClipResult(data.clips || [], data.rationale || '', data.warnings || []);
    } catch (err: any) {
      console.error(err);
      setAiError(err.message ?? 'Clip creation failed');
    } finally {
      setProcessing(false);
    }
  }, [
    checkReady, clipDurations, backendUrl, keptTranscript, keptWordsPayload,
    defaultProvider, config, setProcessing, setClipResult,
  ]);

  const handlePreviewClip = useCallback(
    (clip: ClipSuggestion) => {
      setCurrentTime(clip.startTime);
      const video = document.querySelector('video');
      if (video) {
        video.currentTime = clip.startTime;
        video.play();
      }
    },
    [setCurrentTime],
  );

  const stageClipInEditor = useCallback(
    (clip: ClipSuggestion) => {
      if (words.length === 0) return;
      // Replace current deletions with two ranges that leave only the clip
      // intact. User can then tweak and hit the normal Export button.
      useEditorStore.setState({ deletedRanges: [] });
      if (clip.startWordIndex > 0) deleteWordRange(0, clip.startWordIndex - 1);
      if (clip.endWordIndex < words.length - 1) {
        deleteWordRange(clip.endWordIndex + 1, words.length - 1);
      }
      setActiveTab('filler'); // move focus back to main editor UX
    },
    [words.length, deleteWordRange],
  );

  const [exportingClips, setExportingClips] = useState<Set<number>>(new Set());
  const [exportedClips, setExportedClips] = useState<Set<number>>(new Set());
  const [exportErrors, setExportErrors] = useState<Set<number>>(new Set());

  /**
   * Quick-export a single clip. Respects any deletions already applied in
   * the editor (e.g. fillers) by intersecting with the editor's keep
   * segments. Sends the full words payload so the backend can run
   * AcousticMap refinement for seamless boundaries.
   */
  const handleExportClip = useCallback(
    async (clip: ClipSuggestion, index: number) => {
      if (!videoPath) return;
      setExportingClips((prev) => new Set(prev).add(index));
      setExportErrors((prev) => {
        const next = new Set(prev);
        next.delete(index);
        return next;
      });
      try {
        // Intersect clip range with editor's current keep-segments so any
        // in-editor deletions (filler words, focus cuts) are honoured.
        const editorKeep = useEditorStore.getState().getKeepSegments();
        const intersected = editorKeep
          .map((s) => ({
            start: Math.max(s.start, clip.startTime),
            end: Math.min(s.end, clip.endTime),
          }))
          .filter((s) => s.end - s.start > 0.05);
        const keepSegments = intersected.length
          ? intersected
          : [{ start: clip.startTime, end: clip.endTime }];

        const dirSep = videoPath.includes('\\') ? '\\' : '/';
        const targetDir = (clipSaveLocation || sourceDir).replace(/[\\/]+$/, '');
        const filename = buildClipFilename(clip, bucketIndices[index] ?? index + 1);
        const outputPath = `${targetDir}${dirSep}${filename}`;

        const res = await fetch(`${backendUrl}/export`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            input_path: videoPath,
            output_path: outputPath,
            keep_segments: keepSegments,
            mode: 'fast',
            format: 'mp4',
            force_refine: true,
            words: wordsPayload.map(({ word, start, end }) => ({
              word,
              start: start ?? 0,
              end: end ?? 0,
              confidence: 0,
            })),
            deleted_indices: deletedIndicesArray,
          }),
        });
        if (!res.ok) throw new Error('Export failed');
        setExportedClips((prev) => {
          const next = new Set(prev).add(index);
          setTimeout(
            () =>
              setExportedClips((s) => {
                const n = new Set(s);
                n.delete(index);
                return n;
              }),
            4000,
          );
          return next;
        });
      } catch (err) {
        console.error(err);
        setExportErrors((prev) => new Set(prev).add(index));
      } finally {
        setExportingClips((prev) => {
          const next = new Set(prev);
          next.delete(index);
          return next;
        });
      }
    },
    [videoPath, backendUrl, wordsPayload, deletedIndicesArray, clipSaveLocation, sourceDir, bucketIndices, sourceBasename],
  );

  const handleExportAllClips = useCallback(async () => {
    for (let i = 0; i < clipSuggestions.length; i++) {
      await handleExportClip(clipSuggestions[i], i);
    }
  }, [clipSuggestions, handleExportClip]);

  // ---------------------------------------------------------------------
  // Focus mode
  // ---------------------------------------------------------------------
  const runFocus = useCallback(async () => {
    if (!checkReady()) return;
    if (focusMode === 'topic' && !focusTopic.trim()) {
      setAiError('Enter a topic to focus on, or pick a different mode.');
      return;
    }
    setProcessing(true, `Analyzing for ${FOCUS_MODE_META[focusMode].label.toLowerCase()}...`);
    try {
      const res = await fetch(`${backendUrl}/ai/focus`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transcript: keptTranscript,
          words: keptWordsPayload.map(({ index, word }) => ({ index, word })),
          mode: focusMode,
          topic: focusMode === 'topic' ? focusTopic.trim() : undefined,
          provider: defaultProvider,
          model: config.model,
          api_key: config.apiKey || undefined,
          base_url: config.baseUrl || undefined,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      setFocusPlan((await res.json()) as FocusPlan);
    } catch (err: any) {
      console.error(err);
      setAiError(err.message ?? 'Focus analysis failed');
    } finally {
      setProcessing(false);
    }
  }, [
    checkReady, focusMode, focusTopic, backendUrl, keptTranscript, keptWordsPayload,
    defaultProvider, config, setProcessing, setFocusPlan,
  ]);

  const applyFocusPlan = useCallback(() => {
    if (!focusPlan) return;
    const sorted = [...focusPlan.deletions].sort((a, b) => b.startIndex - a.startIndex);
    for (const d of sorted) deleteWordRange(d.startIndex, d.endIndex);
    setFocusPlan(null);
  }, [focusPlan, deleteWordRange, setFocusPlan]);

  const applySingleFocusDeletion = useCallback(
    (idx: number) => {
      if (!focusPlan) return;
      const d = focusPlan.deletions[idx];
      if (!d) return;
      deleteWordRange(d.startIndex, d.endIndex);
      setFocusPlan({
        ...focusPlan,
        deletions: focusPlan.deletions.filter((_, i) => i !== idx),
      });
    },
    [focusPlan, deleteWordRange, setFocusPlan],
  );

  // ---------------------------------------------------------------------
  // Render
  // ---------------------------------------------------------------------
  return (
    <div className="flex flex-col h-full">
      <div className="flex border-b border-editor-border shrink-0">
        <TabButton
          active={activeTab === 'filler'}
          onClick={() => setActiveTab('filler')}
          icon={<Scissors className="w-3.5 h-3.5" />}
          label="Filler"
        />
        <TabButton
          active={activeTab === 'clips'}
          onClick={() => setActiveTab('clips')}
          icon={<Film className="w-3.5 h-3.5" />}
          label="Clips"
        />
        <TabButton
          active={activeTab === 'focus'}
          onClick={() => setActiveTab('focus')}
          icon={<Crosshair className="w-3.5 h-3.5" />}
          label="Focus"
        />
      </div>

      {aiError && (
        <div className="flex items-start gap-2 mx-4 mt-3 px-3 py-2 bg-red-500/10 border border-red-500/30 rounded-lg text-xs text-red-400">
          <AlertCircle className="w-3.5 h-3.5 shrink-0 mt-0.5" />
          <span>{aiError}</span>
        </div>
      )}

      <div className="flex-1 overflow-y-auto p-4">
        {activeTab === 'filler' && (
          <div className="space-y-4">
            <p className="text-xs text-editor-text-muted">
              AI detects filler words in any language. It infers the language from the transcript — Spanish "este", French "euh", German "äh", and so on.
            </p>
            <div className="space-y-1.5">
              <label className="text-[11px] text-editor-text-muted font-medium">
                Custom filler words (comma-separated)
              </label>
              <input
                type="text"
                value={customFillerWords}
                onChange={(e) => setCustomFillerWords(e.target.value)}
                placeholder="e.g. okay, alright, anyway"
                className="w-full px-2.5 py-1.5 text-xs bg-editor-surface border border-editor-border rounded focus:border-editor-accent focus:outline-none"
              />
            </div>
            <PrimaryButton
              onClick={detectFillers}
              disabled={isProcessing || words.length === 0}
              icon={<Sparkles className="w-4 h-4" />}
              idleLabel="Detect Filler Words"
              busyLabel={processingMessage}
              busy={isProcessing}
            />

            <WarningList items={fillerResult?.warnings} />

            {fillerResult && fillerResult.fillerWords.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium">
                    Found {fillerResult.fillerWords.length} filler words
                    {fillerResult.language && fillerResult.language !== 'auto' && (
                      <span className="text-editor-text-muted"> · {fillerResult.language}</span>
                    )}
                  </span>
                  <div className="flex gap-1">
                    <button
                      onClick={applyFillerDeletions}
                      className="flex items-center gap-1 px-2 py-1 text-xs bg-editor-success/20 text-editor-success rounded hover:bg-editor-success/30"
                    >
                      <Check className="w-3 h-3" /> Apply All
                    </button>
                    <button
                      onClick={() => setFillerResult(null)}
                      className="flex items-center gap-1 px-2 py-1 text-xs bg-editor-border text-editor-text-muted rounded hover:bg-editor-surface"
                    >
                      <X className="w-3 h-3" /> Dismiss
                    </button>
                  </div>
                </div>
                <div className="space-y-1 max-h-64 overflow-y-auto">
                  {fillerResult.fillerWords.map((fw) => (
                    <div
                      key={fw.index}
                      className="flex items-center justify-between px-2 py-1.5 bg-editor-word-filler rounded text-xs"
                    >
                      <span className="truncate mr-2">
                        <strong>"{fw.word}"</strong>
                        <span className="text-editor-text-muted ml-1">— {fw.reason}</span>
                      </span>
                      <ConfidencePill value={fw.confidence} />
                    </div>
                  ))}
                </div>
              </div>
            )}

            {fillerResult && fillerResult.fillerWords.length === 0 && (
              <p className="text-xs text-editor-success">No filler words detected.</p>
            )}
          </div>
        )}

        {activeTab === 'clips' && (
          <div className="space-y-4">
            <p className="text-xs text-editor-text-muted">
              AI finds shareable segments for social. Pick one or more target durations.
            </p>
            <div className="space-y-1.5">
              <label className="text-[11px] text-editor-text-muted font-medium">Target durations</label>
              <div className="flex gap-1.5 flex-wrap">
                {CLIP_DURATION_OPTIONS.map((d) => {
                  const active = clipDurations.includes(d);
                  return (
                    <button
                      key={d}
                      onClick={() => toggleDuration(d)}
                      className={`px-2.5 py-1 text-xs rounded border transition-colors ${
                        active
                          ? 'bg-editor-accent text-white border-editor-accent'
                          : 'bg-editor-surface text-editor-text-muted border-editor-border hover:border-editor-accent/50'
                      }`}
                    >
                      {d}s
                    </button>
                  );
                })}
              </div>
            </div>
            <div className="space-y-1.5">
              <label className="text-[11px] text-editor-text-muted font-medium">Save location</label>
              <div className="flex items-center gap-1.5">
                <div
                  className="flex-1 px-2.5 py-1.5 text-xs bg-editor-surface border border-editor-border rounded truncate text-editor-text-muted"
                  title={effectiveSaveDir || 'Pick a folder'}
                >
                  {effectiveSaveDir || 'Same folder as source'}
                </div>
                {window.electronAPI?.openDirectory && (
                  <button
                    onClick={pickSaveLocation}
                    className="flex items-center gap-1 px-2 py-1.5 text-xs bg-editor-surface border border-editor-border rounded hover:border-editor-accent/50 transition-colors"
                    title="Choose a different folder"
                  >
                    <Folder className="w-3 h-3" /> Change
                  </button>
                )}
                {clipSaveLocation && (
                  <button
                    onClick={() => setClipSaveLocation(null)}
                    className="px-2 py-1.5 text-xs text-editor-text-muted hover:text-editor-text transition-colors"
                    title="Reset to the source folder"
                  >
                    Reset
                  </button>
                )}
              </div>
            </div>
            <PrimaryButton
              onClick={createClips}
              disabled={isProcessing || words.length === 0}
              icon={<Film className="w-4 h-4" />}
              idleLabel="Find Clip Candidates"
              busyLabel={processingMessage}
              busy={isProcessing}
            />

            <WarningList items={clipWarnings} />
            {clipRationale && clipSuggestions.length === 0 && (
              <p className="text-xs text-editor-text-muted italic">{clipRationale}</p>
            )}

            {clipSuggestions.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-editor-text-muted">
                    Seamless stream copy · respects editor edits
                  </span>
                  <button
                    onClick={handleExportAllClips}
                    disabled={exportingClips.size > 0}
                    className="flex items-center gap-1 px-2 py-1 text-[10px] bg-editor-success/20 text-editor-success rounded hover:bg-editor-success/30 disabled:opacity-50 transition-colors"
                  >
                    <Download className="w-3 h-3" />
                    Export All
                  </button>
                </div>
                {clipSuggestions.map((clip, i) => (
                  <div key={i} className="p-3 bg-editor-surface rounded-lg space-y-2">
                    <div className="flex items-center justify-between gap-2">
                      <span className="text-xs font-semibold truncate">{clip.title}</span>
                      <div className="flex items-center gap-1.5 shrink-0">
                        <ConfidencePill value={clip.confidence} />
                        <span className="text-[10px] text-editor-text-muted">
                          {Math.round(clip.endTime - clip.startTime)}s
                        </span>
                      </div>
                    </div>
                    <p className="text-[11px] text-editor-text-muted">{clip.reason}</p>
                    <p className="text-[10px] text-editor-text-muted font-mono truncate" title={buildClipFilename(clip, bucketIndices[i] ?? i + 1)}>
                      {buildClipFilename(clip, bucketIndices[i] ?? i + 1)}
                    </p>
                    <div className="flex gap-1.5">
                      <button
                        onClick={() => handlePreviewClip(clip)}
                        className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs bg-editor-accent/20 text-editor-accent rounded hover:bg-editor-accent/30 transition-colors"
                      >
                        <Play className="w-3 h-3" /> Preview
                      </button>
                      <button
                        onClick={() => stageClipInEditor(clip)}
                        className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs bg-editor-border text-editor-text rounded hover:bg-editor-surface transition-colors"
                        title="Load clip into the editor to tweak"
                      >
                        <Wand2 className="w-3 h-3" /> Edit
                      </button>
                      <button
                        onClick={() => handleExportClip(clip, i)}
                        disabled={exportingClips.has(i)}
                        className={`flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs rounded disabled:opacity-50 transition-colors ${
                          exportedClips.has(i)
                            ? 'bg-editor-success/30 text-editor-success'
                            : exportErrors.has(i)
                              ? 'bg-red-500/20 text-red-400 hover:bg-red-500/30'
                              : 'bg-editor-success/20 text-editor-success hover:bg-editor-success/30'
                        }`}
                      >
                        {exportingClips.has(i) ? (
                          <Loader2 className="w-3 h-3 animate-spin" />
                        ) : exportedClips.has(i) ? (
                          <Check className="w-3 h-3" />
                        ) : exportErrors.has(i) ? (
                          <X className="w-3 h-3" />
                        ) : (
                          <Download className="w-3 h-3" />
                        )}
                        {exportingClips.has(i)
                          ? '…'
                          : exportedClips.has(i)
                            ? 'Saved'
                            : exportErrors.has(i)
                              ? 'Failed'
                              : 'Export'}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}

        {activeTab === 'focus' && (
          <div className="space-y-4">
            <p className="text-xs text-editor-text-muted">
              Tighten or reshape the video without altering meaning. Suggested cuts are reviewable before they touch your edit.
            </p>
            <div className="space-y-1.5">
              <label className="text-[11px] text-editor-text-muted font-medium">Mode</label>
              <div className="grid grid-cols-2 gap-1.5">
                {(Object.keys(FOCUS_MODE_META) as FocusMode[]).map((m) => {
                  const meta = FOCUS_MODE_META[m];
                  const active = focusMode === m;
                  return (
                    <button
                      key={m}
                      onClick={() => setFocusMode(m)}
                      className={`text-left px-2.5 py-2 rounded border text-xs transition-colors ${
                        active
                          ? 'bg-editor-accent/15 border-editor-accent text-editor-accent'
                          : 'bg-editor-surface border-editor-border hover:border-editor-accent/50 text-editor-text'
                      }`}
                    >
                      <div className="font-medium">{meta.label}</div>
                      <div className="text-[10px] text-editor-text-muted mt-0.5">{meta.description}</div>
                    </button>
                  );
                })}
              </div>
            </div>

            {focusMode === 'topic' && (
              <div className="space-y-1.5">
                <label className="text-[11px] text-editor-text-muted font-medium">Topic</label>
                <input
                  type="text"
                  value={focusTopic}
                  onChange={(e) => setFocusTopic(e.target.value)}
                  placeholder="e.g. the product launch timeline"
                  className="w-full px-2.5 py-1.5 text-xs bg-editor-surface border border-editor-border rounded focus:border-editor-accent focus:outline-none"
                />
              </div>
            )}

            <PrimaryButton
              onClick={runFocus}
              disabled={isProcessing || words.length === 0}
              icon={<Crosshair className="w-4 h-4" />}
              idleLabel="Analyze"
              busyLabel={processingMessage}
              busy={isProcessing}
            />

            <WarningList items={focusPlan?.warnings} />
            {focusPlan?.summary && (
              <p className="text-xs text-editor-text-muted italic">{focusPlan.summary}</p>
            )}

            {focusPlan && focusPlan.deletions.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium">
                    {focusPlan.deletions.length} proposed cut{focusPlan.deletions.length === 1 ? '' : 's'}
                  </span>
                  <div className="flex gap-1">
                    <button
                      onClick={applyFocusPlan}
                      className="flex items-center gap-1 px-2 py-1 text-xs bg-editor-success/20 text-editor-success rounded hover:bg-editor-success/30"
                    >
                      <Check className="w-3 h-3" /> Apply All
                    </button>
                    <button
                      onClick={() => setFocusPlan(null)}
                      className="flex items-center gap-1 px-2 py-1 text-xs bg-editor-border text-editor-text-muted rounded hover:bg-editor-surface"
                    >
                      <X className="w-3 h-3" /> Dismiss
                    </button>
                  </div>
                </div>
                <div className="space-y-1.5 max-h-96 overflow-y-auto">
                  {focusPlan.deletions.map((d, i) => {
                    const preview = words
                      .slice(d.startIndex, Math.min(d.endIndex + 1, d.startIndex + 18))
                      .map((w) => w.word)
                      .join(' ');
                    const length = d.endIndex - d.startIndex + 1;
                    const truncated = length > 18;
                    return (
                      <div
                        key={i}
                        className="px-2 py-1.5 bg-editor-word-filler rounded text-xs space-y-1"
                      >
                        <div className="flex items-center justify-between gap-2">
                          <span className="text-[10px] text-editor-text-muted">
                            {length} word{length === 1 ? '' : 's'} · {d.reason}
                          </span>
                          <div className="flex items-center gap-1.5 shrink-0">
                            <ConfidencePill value={d.confidence} />
                            <button
                              onClick={() => applySingleFocusDeletion(i)}
                              className="text-editor-success hover:bg-editor-success/20 rounded p-0.5"
                              title="Apply this cut"
                            >
                              <Check className="w-3 h-3" />
                            </button>
                          </div>
                        </div>
                        <div className="text-editor-text truncate">
                          "{preview}
                          {truncated ? '…' : ''}"
                        </div>
                      </div>
                    );
                  })}
                </div>
              </div>
            )}

            {focusPlan && focusPlan.deletions.length === 0 && (
              <p className="text-xs text-editor-success">Nothing to cut for this mode.</p>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------
// Shared small components
// ---------------------------------------------------------------------

function TabButton({
  active,
  onClick,
  icon,
  label,
}: {
  active: boolean;
  onClick: () => void;
  icon: React.ReactNode;
  label: string;
}) {
  return (
    <button
      onClick={onClick}
      className={`flex-1 flex items-center justify-center gap-1.5 px-3 py-2.5 text-xs font-medium transition-colors border-b-2 ${
        active
          ? 'border-editor-accent text-editor-accent'
          : 'border-transparent text-editor-text-muted hover:text-editor-text'
      }`}
    >
      {icon}
      {label}
    </button>
  );
}

function PrimaryButton({
  onClick,
  disabled,
  icon,
  idleLabel,
  busyLabel,
  busy,
}: {
  onClick: () => void;
  disabled: boolean;
  icon: React.ReactNode;
  idleLabel: string;
  busyLabel: string;
  busy: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-editor-accent hover:bg-editor-accent-hover disabled:opacity-50 rounded-lg text-sm font-medium transition-colors"
    >
      {busy ? (
        <>
          <Loader2 className="w-4 h-4 animate-spin" />
          {busyLabel || idleLabel}
        </>
      ) : (
        <>
          {icon}
          {idleLabel}
        </>
      )}
    </button>
  );
}

function ConfidencePill({ value }: { value: number }) {
  const pct = Math.round(value * 100);
  const color =
    value >= 0.8
      ? 'bg-editor-success/20 text-editor-success'
      : value >= 0.6
        ? 'bg-editor-accent/20 text-editor-accent'
        : 'bg-yellow-500/20 text-yellow-400';
  return (
    <span className={`px-1.5 py-0.5 rounded text-[10px] font-medium ${color}`} title="AI confidence">
      {pct}%
    </span>
  );
}

function WarningList({ items }: { items?: string[] }) {
  if (!items || items.length === 0) return null;
  return (
    <div className="space-y-1">
      {items.map((w, i) => (
        <div
          key={i}
          className="flex items-start gap-2 px-2 py-1.5 bg-yellow-500/10 border border-yellow-500/30 rounded text-[11px] text-yellow-300"
        >
          <AlertCircle className="w-3 h-3 shrink-0 mt-0.5" />
          <span>{w}</span>
        </div>
      ))}
    </div>
  );
}
