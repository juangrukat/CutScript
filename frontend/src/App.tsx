import { useEffect, useState, useRef } from 'react';
import { useEditorStore } from './store/editorStore';
import VideoPlayer from './components/VideoPlayer';
import TranscriptEditor from './components/TranscriptEditor';
import WaveformTimeline from './components/WaveformTimeline';
import AIPanel from './components/AIPanel';
import ExportDialog from './components/ExportDialog';
import SettingsPanel from './components/SettingsPanel';
import { useKeyboardShortcuts } from './hooks/useKeyboardShortcuts';
import {
  Film,
  FolderOpen,
  Settings,
  Sparkles,
  Download,
  Loader2,
  FolderSearch,
  FileInput,
  ChevronRight,
} from 'lucide-react';

const IS_ELECTRON = !!window.electronAPI;

type Panel = 'ai' | 'settings' | 'export' | null;

export default function App() {
  const {
    videoPath,
    words,
    isTranscribing,
    transcriptionProgress,
    loadVideo,
    setBackendUrl,
    setTranscription,
    setTranscribing,
    backendUrl,
  } = useEditorStore();

  const VALID_WHISPER_MODELS = ['tiny', 'base', 'small', 'medium', 'large-v3', 'distil-large-v3'] as const;
  type WhisperModel = typeof VALID_WHISPER_MODELS[number];

  const [activePanel, setActivePanel] = useState<Panel>(null);
  const [manualPath, setManualPath] = useState('');
  const [whisperModel, setWhisperModel] = useState<WhisperModel>(() => {
    const saved = localStorage.getItem('whisperModel') as WhisperModel | null;
    return saved && VALID_WHISPER_MODELS.includes(saved) ? saved : 'base';
  });
  const [vocabPrompt, setVocabPrompt] = useState('');
  const [transcribeStatus, setTranscribeStatus] = useState('');
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Advanced transcription options
  const [showAdvanced, setShowAdvanced] = useState(false);
  const [language, setLanguage] = useState(() => localStorage.getItem('txLanguage') ?? '');
  const [beamSize, setBeamSize] = useState(() => parseInt(localStorage.getItem('txBeamSize') ?? '5'));
  const [vadFilter, setVadFilter] = useState(() => localStorage.getItem('txVadFilter') === 'true');
  const [vadMinSilenceMs, setVadMinSilenceMs] = useState(() => parseInt(localStorage.getItem('txVadMinSilenceMs') ?? '500'));
  const [verbatim, setVerbatim] = useState(false);
  const [diarize, setDiarize] = useState(false);
  const [hfToken, setHfToken] = useState('');
  const [numSpeakers, setNumSpeakers] = useState('');

  useKeyboardShortcuts();

  useEffect(() => {
    if (IS_ELECTRON) {
      window.electronAPI!.getBackendUrl().then(setBackendUrl);
    }
  }, [setBackendUrl]);

  const handleLoadProject = async () => {
    if (!IS_ELECTRON) return;
    try {
      const projectPath = await window.electronAPI!.openProject();
      if (!projectPath) return;
      const content = await window.electronAPI!.readFile(projectPath);
      const data = JSON.parse(content);
      useEditorStore.getState().loadProject(data);
    } catch (err) {
      console.error('Failed to load project:', err);
      alert(`Failed to load project: ${err}`);
    }
  };

  const handleOpenFile = async () => {
    if (IS_ELECTRON) {
      const path = await window.electronAPI!.openFile();
      if (path) {
        loadVideo(path);
        await transcribeVideo(path);
      }
    } else {
      // Browser: use the manual path input
      const path = manualPath.trim();
      if (path) {
        loadVideo(path);
        await transcribeVideo(path);
      }
    }
  };

  const handleManualSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const path = manualPath.trim();
    if (!path) return;
    loadVideo(path);
    await transcribeVideo(path);
  };

  const transcribeVideo = async (path: string) => {
    setTranscribing(true, 0);
    setTranscribeStatus('Starting...');
    try {
      const res = await fetch(`${backendUrl}/transcribe/stream`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          file_path: path,
          model: whisperModel,
          initial_prompt: vocabPrompt.trim() || undefined,
          language: language || undefined,
          beam_size: beamSize,
          vad_filter: vadFilter,
          vad_min_silence_ms: vadMinSilenceMs,
          verbatim,
          diarize,
          hf_token: diarize && hfToken ? hfToken : undefined,
          num_speakers: diarize && numSpeakers ? parseInt(numSpeakers) : undefined,
        }),
      });
      if (!res.ok) throw new Error(`Transcription failed: ${res.statusText}`);
      if (!res.body) throw new Error('No response body from server');

      const reader = res.body.getReader();
      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        // SSE lines end with \n\n; split on \n and process complete data: lines
        const lines = buffer.split('\n');
        buffer = lines.pop() ?? '';

        for (const line of lines) {
          if (!line.startsWith('data: ')) continue;
          try {
            const event = JSON.parse(line.slice(6));
            if (event.type === 'progress') {
              setTranscribing(true, event.value);
              setTranscribeStatus(event.status ?? '');
            } else if (event.type === 'done') {
              setTranscription(event.result);
              setTranscribing(false, 100);
              setTranscribeStatus('');
              return;
            } else if (event.type === 'error') {
              throw new Error(event.message);
            }
          } catch {
            // skip malformed events
          }
        }
      }
    } catch (err) {
      console.error('Transcription error:', err);
      alert(`Transcription failed. Check the console for details.\n\n${err}`);
    } finally {
      setTranscribing(false);
      setTranscribeStatus('');
    }
  };

  const togglePanel = (panel: Panel) =>
    setActivePanel((prev) => (prev === panel ? null : panel));

  if (!videoPath) {
    return (
      <div className="h-screen flex flex-col items-center justify-center gap-8 bg-editor-bg px-6">
        <div className="flex flex-col items-center gap-3">
          <Film className="w-14 h-14 text-editor-accent opacity-80" />
          <h1 className="text-3xl font-semibold tracking-tight">CutScript</h1>
          <p className="text-editor-text-muted text-sm max-w-sm text-center">
            Open-source text-based video editing powered by AI.
          </p>
        </div>

        {/* Whisper model selector + vocabulary prompt */}
        <div className="flex flex-col items-center gap-3 w-full max-w-sm">
          <div className="flex items-center gap-3 w-full">
            <label className="text-xs text-editor-text-muted whitespace-nowrap">Whisper model:</label>
            <select
              value={whisperModel}
              onChange={(e) => {
                const v = e.target.value as WhisperModel;
                setWhisperModel(v);
                localStorage.setItem('whisperModel', v);
              }}
              className="flex-1 px-3 py-1.5 bg-editor-surface border border-editor-border rounded-lg text-xs text-editor-text focus:outline-none focus:border-editor-accent"
            >
              <option value="tiny">faster-whisper tiny (~75 MB, fastest)</option>
              <option value="base">faster-whisper base (~140 MB, fast)</option>
              <option value="small">faster-whisper small (~460 MB, good)</option>
              <option value="medium">faster-whisper medium (~1.5 GB, better)</option>
              <option value="large-v3">faster-whisper large-v3 (~3.1 GB, best, multilingual)</option>
              <option value="distil-large-v3">faster-whisper distil-large-v3 (~1.6 GB, great + fast)</option>
            </select>
          </div>
          <div className="flex flex-col gap-1 w-full">
            <label className="text-xs text-editor-text-muted">
              Vocabulary prompt{' '}
              <span className="opacity-50">(optional — names, acronyms, technical terms)</span>
            </label>
            <textarea
              value={vocabPrompt}
              onChange={(e) => setVocabPrompt(e.target.value)}
              placeholder="e.g. John Smith, Acme Corp, API, OAuth, QuickBooks"
              rows={2}
              className="w-full px-3 py-2 bg-editor-surface border border-editor-border rounded-lg text-xs text-editor-text placeholder:text-editor-text-muted/40 focus:outline-none focus:border-editor-accent resize-none"
            />
          </div>

          {/* Advanced options */}
          <div className="w-full">
            <button
              onClick={() => setShowAdvanced((v) => !v)}
              className="flex items-center gap-1 text-xs text-editor-text-muted hover:text-editor-text transition-colors"
            >
              <ChevronRight className={`w-3 h-3 transition-transform duration-150 ${showAdvanced ? 'rotate-90' : ''}`} />
              Advanced options
            </button>

            {showAdvanced && (
              <div className="mt-3 flex flex-col gap-4 border-l border-editor-border pl-3">

                {/* Language */}
                <div className="flex items-center gap-3">
                  <label className="text-xs text-editor-text-muted w-20 shrink-0">Language</label>
                  <select
                    value={language}
                    onChange={(e) => { setLanguage(e.target.value); localStorage.setItem('txLanguage', e.target.value); }}
                    className="flex-1 px-2 py-1.5 bg-editor-surface border border-editor-border rounded-lg text-xs text-editor-text focus:outline-none focus:border-editor-accent"
                  >
                    <option value="">Auto-detect</option>
                    <option value="en">English</option>
                    <option value="es">Spanish</option>
                    <option value="fr">French</option>
                    <option value="de">German</option>
                    <option value="it">Italian</option>
                    <option value="pt">Portuguese</option>
                    <option value="zh">Chinese</option>
                    <option value="ja">Japanese</option>
                    <option value="ko">Korean</option>
                    <option value="nl">Dutch</option>
                    <option value="ru">Russian</option>
                    <option value="ar">Arabic</option>
                    <option value="hi">Hindi</option>
                  </select>
                </div>

                {/* Beam size */}
                <div className="flex items-center gap-3">
                  <label className="text-xs text-editor-text-muted w-20 shrink-0">Beam size</label>
                  <select
                    value={beamSize}
                    onChange={(e) => { setBeamSize(Number(e.target.value)); localStorage.setItem('txBeamSize', e.target.value); }}
                    className="flex-1 px-2 py-1.5 bg-editor-surface border border-editor-border rounded-lg text-xs text-editor-text focus:outline-none focus:border-editor-accent"
                  >
                    <option value={1}>1 — fastest, least accurate</option>
                    <option value={3}>3</option>
                    <option value={5}>5 — default (balanced)</option>
                    <option value={8}>8</option>
                    <option value={10}>10 — slowest, most accurate</option>
                  </select>
                </div>

                {/* VAD filter */}
                <div className="flex flex-col gap-2">
                  <label className="flex items-start gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={vadFilter}
                      onChange={(e) => { setVadFilter(e.target.checked); localStorage.setItem('txVadFilter', String(e.target.checked)); }}
                      className="mt-0.5 accent-editor-accent"
                    />
                    <span className="text-xs text-editor-text leading-snug">
                      VAD filter
                      <span className="block text-editor-text-muted mt-0.5">
                        Voice activity detection — improves segmentation for recordings with pauses or long silences. Recommended for interviews and multi-speaker recordings.
                      </span>
                    </span>
                  </label>
                  {vadFilter && (
                    <div className="flex items-center gap-2 pl-5">
                      <label className="text-xs text-editor-text-muted whitespace-nowrap">Min silence</label>
                      <input
                        type="number"
                        value={vadMinSilenceMs}
                        onChange={(e) => { setVadMinSilenceMs(Number(e.target.value)); localStorage.setItem('txVadMinSilenceMs', e.target.value); }}
                        min={100}
                        max={2000}
                        step={50}
                        className="w-20 px-2 py-1 bg-editor-surface border border-editor-border rounded-lg text-xs text-editor-text focus:outline-none focus:border-editor-accent"
                      />
                      <span className="text-xs text-editor-text-muted">ms</span>
                    </div>
                  )}
                </div>

                {/* Verbatim mode */}
                <div className="flex flex-col gap-2">
                  <label className="flex items-start gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={verbatim}
                      onChange={(e) => setVerbatim(e.target.checked)}
                      className="mt-0.5 accent-editor-accent"
                    />
                    <span className="text-xs text-editor-text leading-snug">
                      Verbatim mode
                      <span className="block text-editor-text-muted mt-0.5">
                        Preserves repeated words, stutters, false starts, and filler sounds. Off by default — Whisper normally smooths disfluencies for readability. Turn this on when exact spoken content matters.
                      </span>
                    </span>
                  </label>
                </div>

                {/* Speaker diarization */}
                <div className="flex flex-col gap-2">
                  <label className="flex items-start gap-2 cursor-pointer">
                    <input
                      type="checkbox"
                      checked={diarize}
                      onChange={(e) => setDiarize(e.target.checked)}
                      className="mt-0.5 accent-editor-accent"
                    />
                    <span className="text-xs text-editor-text leading-snug">
                      Speaker diarization
                      <span className="block text-editor-text-muted mt-0.5">
                        Automatically detects and labels different speakers (Speaker 1, Speaker 2…). Useful for interviews, meetings, and any recording with multiple voices. Requires a free HuggingFace token.
                      </span>
                    </span>
                  </label>
                  {diarize && (
                    <div className="flex flex-col gap-2 pl-5">
                      <div className="flex items-center gap-2">
                        <label className="text-xs text-editor-text-muted w-16 shrink-0">HF Token</label>
                        <input
                          type="password"
                          value={hfToken}
                          onChange={(e) => setHfToken(e.target.value)}
                          placeholder="hf_..."
                          className="flex-1 px-2 py-1 bg-editor-surface border border-editor-border rounded-lg text-xs text-editor-text placeholder:text-editor-text-muted/40 focus:outline-none focus:border-editor-accent"
                        />
                      </div>
                      <div className="flex items-center gap-2">
                        <label className="text-xs text-editor-text-muted w-16 shrink-0">Speakers</label>
                        <input
                          type="number"
                          value={numSpeakers}
                          onChange={(e) => setNumSpeakers(e.target.value)}
                          placeholder="auto"
                          min={1}
                          max={20}
                          className="w-16 px-2 py-1 bg-editor-surface border border-editor-border rounded-lg text-xs text-editor-text placeholder:text-editor-text-muted/40 focus:outline-none focus:border-editor-accent"
                        />
                        <span className="text-xs text-editor-text-muted">optional</span>
                      </div>
                    </div>
                  )}
                </div>

              </div>
            )}
          </div>
        </div>

        {IS_ELECTRON ? (
          <div className="flex flex-col items-center gap-3">
            <button
              onClick={handleOpenFile}
              className="flex items-center gap-2 px-6 py-3 bg-editor-accent hover:bg-editor-accent-hover rounded-lg text-white font-medium transition-colors"
            >
              <FolderOpen className="w-5 h-5" />
              Open Video File
            </button>
            <button
              onClick={handleLoadProject}
              className="flex items-center gap-2 px-4 py-2 text-sm text-editor-text-muted hover:text-editor-text hover:bg-editor-surface rounded-lg transition-colors"
            >
              <FileInput className="w-4 h-4" />
              Load Project (.aive)
            </button>
          </div>
        ) : (
          /* Browser: manual path input */
          <div className="w-full max-w-lg space-y-3">
            <div className="flex items-center gap-2 px-3 py-1.5 bg-editor-warning/10 border border-editor-warning/30 rounded-lg">
              <span className="text-editor-warning text-xs">
                Running in browser — paste the full path to your video file below.
              </span>
            </div>
            <form onSubmit={handleManualSubmit} className="flex gap-2">
              <div className="flex-1 relative">
                <FolderSearch className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-editor-text-muted pointer-events-none" />
                <input
                  ref={fileInputRef}
                  type="text"
                  value={manualPath}
                  onChange={(e) => setManualPath(e.target.value)}
                  placeholder="C:\Videos\my-video.mp4"
                  className="w-full pl-9 pr-3 py-2.5 bg-editor-surface border border-editor-border rounded-lg text-sm text-editor-text placeholder:text-editor-text-muted/40 focus:outline-none focus:border-editor-accent"
                  autoFocus
                />
              </div>
              <button
                type="submit"
                disabled={!manualPath.trim()}
                className="flex items-center gap-2 px-5 py-2.5 bg-editor-accent hover:bg-editor-accent-hover disabled:opacity-40 rounded-lg text-sm text-white font-medium transition-colors whitespace-nowrap"
              >
                <Film className="w-4 h-4" />
                Load &amp; Transcribe
              </button>
            </form>
            <p className="text-[11px] text-editor-text-muted text-center">
              Supported: MP4, AVI, MOV, MKV, WebM, M4A
            </p>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className="h-screen flex flex-col bg-editor-bg overflow-hidden">
      {/* Top bar */}
      <header className="h-12 flex items-center justify-between px-4 border-b border-editor-border shrink-0">
        <div className="flex items-center gap-3">
          <Film className="w-5 h-5 text-editor-accent" />
          <span className="text-sm font-medium truncate max-w-[300px]">
            {videoPath.split(/[\\/]/).pop()}
          </span>
        </div>
        <div className="flex items-center gap-1">
          <ToolbarButton
            icon={<FolderOpen className="w-4 h-4" />}
            label="Open"
            onClick={IS_ELECTRON ? handleOpenFile : () => useEditorStore.getState().reset()}
          />
          <ToolbarButton
            icon={<Sparkles className="w-4 h-4" />}
            label="AI"
            active={activePanel === 'ai'}
            onClick={() => togglePanel('ai')}
            disabled={words.length === 0}
          />
          <ToolbarButton
            icon={<Download className="w-4 h-4" />}
            label="Export"
            active={activePanel === 'export'}
            onClick={() => togglePanel('export')}
            disabled={words.length === 0}
          />
          <ToolbarButton
            icon={<Settings className="w-4 h-4" />}
            label="Settings"
            active={activePanel === 'settings'}
            onClick={() => togglePanel('settings')}
          />
        </div>
      </header>

      {/* Main content */}
      <div className="flex-1 flex overflow-hidden">
        {/* Left: video + transcript */}
        <div className="flex-1 flex flex-col min-w-0">
          <div className="flex-1 flex min-h-0">
            {/* Video player */}
            <div className="w-1/2 p-3 flex items-center justify-center bg-black/20">
              <VideoPlayer />
            </div>

            {/* Transcript */}
            <div className="w-1/2 border-l border-editor-border flex flex-col min-h-0">
              {isTranscribing ? (
                <div className="flex-1 flex flex-col items-center justify-center gap-5 px-8">
                  <Loader2 className="w-8 h-8 text-editor-accent animate-spin" />
                  <div className="w-full max-w-xs space-y-2">
                    <div className="flex justify-between text-xs text-editor-text-muted">
                      <span>{transcribeStatus || 'Working...'}</span>
                      <span>{Math.round(transcriptionProgress)}%</span>
                    </div>
                    <div className="h-1.5 w-full bg-editor-surface rounded-full overflow-hidden">
                      <div
                        className="h-full bg-editor-accent rounded-full transition-all duration-300 ease-out"
                        style={{ width: `${transcriptionProgress}%` }}
                      />
                    </div>
                  </div>
                </div>
              ) : words.length > 0 ? (
                <TranscriptEditor />
              ) : (
                <div className="flex-1 flex items-center justify-center text-editor-text-muted text-sm">
                  No transcript yet
                </div>
              )}
            </div>
          </div>

          {/* Waveform timeline */}
          <div className="h-32 border-t border-editor-border shrink-0">
            <WaveformTimeline />
          </div>
        </div>

        {/* Right panel (AI / Export / Settings) */}
        {activePanel && (
          <div className="w-80 border-l border-editor-border overflow-y-auto shrink-0">
            {activePanel === 'ai' && <AIPanel />}
            {activePanel === 'export' && <ExportDialog />}
            {activePanel === 'settings' && <SettingsPanel />}
          </div>
        )}
      </div>
    </div>
  );
}

function ToolbarButton({
  icon,
  label,
  active,
  onClick,
  disabled,
}: {
  icon: React.ReactNode;
  label: string;
  active?: boolean;
  onClick: () => void;
  disabled?: boolean;
}) {
  return (
    <button
      onClick={onClick}
      disabled={disabled}
      title={label}
      className={`flex items-center gap-1.5 px-3 py-1.5 rounded-md text-xs font-medium transition-colors ${
        active
          ? 'bg-editor-accent text-white'
          : 'text-editor-text-muted hover:text-editor-text hover:bg-editor-surface'
      } ${disabled ? 'opacity-40 cursor-not-allowed' : ''}`}
    >
      {icon}
      {label}
    </button>
  );
}
