import { useCallback, useState } from 'react';
import { useEditorStore } from '../store/editorStore';
import { useAIStore } from '../store/aiStore';
import { Sparkles, Scissors, Film, Loader2, Check, X, Play, Download, AlertCircle } from 'lucide-react';
import type { ClipSuggestion } from '../types/project';

export default function AIPanel() {
  const { words, videoPath, backendUrl, deleteWordRange, setCurrentTime } = useEditorStore();
  const {
    defaultProvider,
    providers,
    customFillerWords,
    fillerResult,
    clipSuggestions,
    isProcessing,
    processingMessage,
    setCustomFillerWords,
    setFillerResult,
    setClipSuggestions,
    setProcessing,
  } = useAIStore();

  const [activeTab, setActiveTab] = useState<'filler' | 'clips'>('filler');
  const [aiError, setAiError] = useState<string | null>(null);

  const requiresApiKey = defaultProvider === 'openai' || defaultProvider === 'claude';
  const config = providers[defaultProvider];
  const missingKey = requiresApiKey && !config.apiKey;

  const detectFillers = useCallback(async () => {
    if (words.length === 0) return;
    setAiError(null);
    if (missingKey) {
      setAiError(`No API key set for ${defaultProvider}. Add it in Settings.`);
      return;
    }
    setProcessing(true, 'Detecting filler words...');
    try {
      const transcript = words.map((w) => w.word).join(' ');
      const res = await fetch(`${backendUrl}/ai/filler-removal`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transcript,
          words: words.map((w, i) => ({ index: i, word: w.word })),
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
      const data = await res.json();
      setFillerResult(data);
    } catch (err: any) {
      console.error(err);
      setAiError(err.message ?? 'Filler detection failed');
    } finally {
      setProcessing(false);
    }
  }, [words, backendUrl, defaultProvider, config, missingKey, customFillerWords, setProcessing, setFillerResult]);

  const createClips = useCallback(async () => {
    if (words.length === 0) return;
    setAiError(null);
    if (missingKey) {
      setAiError(`No API key set for ${defaultProvider}. Add it in Settings.`);
      return;
    }
    setProcessing(true, 'Finding best clip segments...');
    try {
      const transcript = words.map((w) => w.word).join(' ');
      const res = await fetch(`${backendUrl}/ai/create-clip`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          transcript,
          words: words.map((w, i) => ({
            index: i,
            word: w.word,
            start: w.start,
            end: w.end,
          })),
          provider: defaultProvider,
          model: config.model,
          api_key: config.apiKey || undefined,
          base_url: config.baseUrl || undefined,
          target_duration: 60,
        }),
      });
      if (!res.ok) {
        const body = await res.json().catch(() => ({}));
        throw new Error(body.detail || `Request failed (${res.status})`);
      }
      const data = await res.json();
      setClipSuggestions(data.clips || []);
    } catch (err: any) {
      console.error(err);
      setAiError(err.message ?? 'Clip creation failed');
    } finally {
      setProcessing(false);
    }
  }, [words, backendUrl, defaultProvider, config, missingKey, setProcessing, setClipSuggestions]);

  const applyFillerDeletions = useCallback(() => {
    if (!fillerResult) return;
    const sorted = [...fillerResult.fillerWords].sort((a, b) => b.index - a.index);
    for (const fw of sorted) {
      deleteWordRange(fw.index, fw.index);
    }
    setFillerResult(null);
  }, [fillerResult, deleteWordRange, setFillerResult]);

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

  const [exportingClips, setExportingClips] = useState<Set<number>>(new Set());
  const [exportedClips, setExportedClips] = useState<Set<number>>(new Set());
  const [exportErrors, setExportErrors] = useState<Set<number>>(new Set());

  const handleExportClip = useCallback(
    async (clip: ClipSuggestion, index: number) => {
      if (!videoPath) return;
      setExportingClips((prev) => new Set(prev).add(index));
      setExportErrors((prev) => { const next = new Set(prev); next.delete(index); return next; });
      try {
        const safeName = clip.title.replace(/[^a-zA-Z0-9_-]/g, '_').substring(0, 40);
        const dirSep = videoPath.lastIndexOf('\\') >= 0 ? '\\' : '/';
        const dir = videoPath.substring(0, videoPath.lastIndexOf(dirSep));
        const outputPath = `${dir}${dirSep}${safeName}_clip.mp4`;

        const res = await fetch(`${backendUrl}/export`, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            input_path: videoPath,
            output_path: outputPath,
            keep_segments: [{ start: clip.startTime, end: clip.endTime }],
            mode: 'fast',
            format: 'mp4',
          }),
        });
        if (!res.ok) throw new Error('Export failed');
        setExportedClips((prev) => {
          const next = new Set(prev).add(index);
          setTimeout(() => setExportedClips((s) => { const n = new Set(s); n.delete(index); return n; }), 4000);
          return next;
        });
      } catch (err) {
        console.error(err);
        setExportErrors((prev) => new Set(prev).add(index));
      } finally {
        setExportingClips((prev) => { const next = new Set(prev); next.delete(index); return next; });
      }
    },
    [videoPath, backendUrl],
  );

  const handleExportAllClips = useCallback(async () => {
    for (let i = 0; i < clipSuggestions.length; i++) {
      await handleExportClip(clipSuggestions[i], i);
    }
  }, [clipSuggestions, handleExportClip]);

  return (
    <div className="flex flex-col h-full">
      <div className="flex border-b border-editor-border shrink-0">
        <TabButton
          active={activeTab === 'filler'}
          onClick={() => setActiveTab('filler')}
          icon={<Scissors className="w-3.5 h-3.5" />}
          label="Filler Words"
        />
        <TabButton
          active={activeTab === 'clips'}
          onClick={() => setActiveTab('clips')}
          icon={<Film className="w-3.5 h-3.5" />}
          label="Create Clips"
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
              Use AI to detect and remove filler words like "um", "uh", "like", "you know" from
              your transcript.
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
            <button
              onClick={detectFillers}
              disabled={isProcessing || words.length === 0}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-editor-accent hover:bg-editor-accent-hover disabled:opacity-50 rounded-lg text-sm font-medium transition-colors"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  {processingMessage}
                </>
              ) : (
                <>
                  <Sparkles className="w-4 h-4" />
                  Detect Filler Words
                </>
              )}
            </button>

            {fillerResult && fillerResult.fillerWords.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-xs font-medium">
                    Found {fillerResult.fillerWords.length} filler words
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
                      <span>
                        <strong>"{fw.word}"</strong>
                        <span className="text-editor-text-muted ml-1">— {fw.reason}</span>
                      </span>
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
              AI analyzes your transcript and suggests the most engaging segments for a
              YouTube Short or social media clip.
            </p>
            <button
              onClick={createClips}
              disabled={isProcessing || words.length === 0}
              className="w-full flex items-center justify-center gap-2 px-4 py-2.5 bg-editor-accent hover:bg-editor-accent-hover disabled:opacity-50 rounded-lg text-sm font-medium transition-colors"
            >
              {isProcessing ? (
                <>
                  <Loader2 className="w-4 h-4 animate-spin" />
                  {processingMessage}
                </>
              ) : (
                <>
                  <Film className="w-4 h-4" />
                  Find Best Clips
                </>
              )}
            </button>

            {clipSuggestions.length > 0 && (
              <div className="space-y-3">
                <div className="flex items-center justify-between">
                  <span className="text-[10px] text-editor-text-muted">
                    Stream copy · lossless · original quality
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
                    <div className="flex items-center justify-between">
                      <span className="text-xs font-semibold">{clip.title}</span>
                      <span className="text-[10px] text-editor-text-muted">
                        {Math.round(clip.endTime - clip.startTime)}s
                      </span>
                    </div>
                    <p className="text-[11px] text-editor-text-muted">{clip.reason}</p>
                    <div className="flex gap-2">
                      <button
                        onClick={() => handlePreviewClip(clip)}
                        className="flex-1 flex items-center justify-center gap-1 px-2 py-1.5 text-xs bg-editor-accent/20 text-editor-accent rounded hover:bg-editor-accent/30 transition-colors"
                      >
                        <Play className="w-3 h-3" /> Preview
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
                        {exportingClips.has(i) ? 'Exporting...' : exportedClips.has(i) ? 'Saved!' : exportErrors.has(i) ? 'Failed' : 'Export'}
                      </button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
}

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
