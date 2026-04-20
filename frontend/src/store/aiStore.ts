import { create } from 'zustand';
import { persist } from 'zustand/middleware';
import type {
  AIProvider,
  AIProviderConfig,
  ClipSuggestion,
  FillerWordResult,
  FocusPlan,
} from '../types/project';

const ENCRYPTED_KEY_PREFIX = 'aive_enc_';

export type ClipDuration = 15 | 30 | 60 | 90;

interface AIState {
  providers: Record<AIProvider, AIProviderConfig>;
  defaultProvider: AIProvider;
  customFillerWords: string;
  fillerResult: FillerWordResult | null;
  clipSuggestions: ClipSuggestion[];
  clipRationale: string;
  clipWarnings: string[];
  clipDurations: ClipDuration[];
  clipSaveLocation: string | null;
  focusPlan: FocusPlan | null;
  isProcessing: boolean;
  processingMessage: string;
  _keysHydrated: boolean;
}

interface AIActions {
  setProviderConfig: (provider: AIProvider, config: Partial<AIProviderConfig>) => void;
  setDefaultProvider: (provider: AIProvider) => void;
  setCustomFillerWords: (words: string) => void;
  setFillerResult: (result: FillerWordResult | null) => void;
  setClipResult: (clips: ClipSuggestion[], rationale?: string, warnings?: string[]) => void;
  setClipDurations: (durations: ClipDuration[]) => void;
  setClipSaveLocation: (location: string | null) => void;
  setFocusPlan: (plan: FocusPlan | null) => void;
  setProcessing: (active: boolean, message?: string) => void;
  clearResults: () => void;
  hydrateKeys: () => Promise<void>;
}

async function encryptAndStore(key: string, value: string): Promise<void> {
  if (!value) {
    localStorage.removeItem(ENCRYPTED_KEY_PREFIX + key);
    return;
  }
  if (window.electronAPI) {
    const encrypted = await window.electronAPI.encryptString(value);
    localStorage.setItem(ENCRYPTED_KEY_PREFIX + key, encrypted);
  } else {
    localStorage.setItem(ENCRYPTED_KEY_PREFIX + key, btoa(value));
  }
}

async function loadAndDecrypt(key: string): Promise<string> {
  const stored = localStorage.getItem(ENCRYPTED_KEY_PREFIX + key);
  if (!stored) return '';
  if (window.electronAPI) {
    try {
      return await window.electronAPI.decryptString(stored);
    } catch {
      return '';
    }
  }
  try {
    return atob(stored);
  } catch {
    return '';
  }
}

export const useAIStore = create<AIState & AIActions>()(
  persist(
    (set, get) => ({
      providers: {
        ollama: { provider: 'ollama', baseUrl: 'http://localhost:11434', model: 'llama3' },
        openai: { provider: 'openai', apiKey: '', model: 'gpt-4o' },
        claude: { provider: 'claude', apiKey: '', model: 'claude-sonnet-4-6' },
      },
      defaultProvider: 'ollama',
      customFillerWords: '',
      fillerResult: null,
      clipSuggestions: [],
      clipRationale: '',
      clipWarnings: [],
      clipDurations: [60],
      clipSaveLocation: null,
      focusPlan: null,
      isProcessing: false,
      processingMessage: '',
      _keysHydrated: false,

      setProviderConfig: (provider, config) => {
        set((state) => ({
          providers: {
            ...state.providers,
            [provider]: { ...state.providers[provider], ...config },
          },
        }));
        if (config.apiKey !== undefined) {
          encryptAndStore(`${provider}_apiKey`, config.apiKey);
        }
      },

      setDefaultProvider: (provider) => set({ defaultProvider: provider }),
      setCustomFillerWords: (words) => set({ customFillerWords: words }),
      setFillerResult: (result) => set({ fillerResult: result }),
      setClipResult: (clips, rationale = '', warnings = []) =>
        set({ clipSuggestions: clips, clipRationale: rationale, clipWarnings: warnings }),
      setClipDurations: (durations) => set({ clipDurations: durations.length ? durations : [60] }),
      setClipSaveLocation: (location) => set({ clipSaveLocation: location }),
      setFocusPlan: (plan) => set({ focusPlan: plan }),
      setProcessing: (active, message) =>
        set({ isProcessing: active, processingMessage: message ?? '' }),
      clearResults: () =>
        set({ fillerResult: null, clipSuggestions: [], clipRationale: '', clipWarnings: [], focusPlan: null }),

      hydrateKeys: async () => {
        const [openaiKey, claudeKey] = await Promise.all([
          loadAndDecrypt('openai_apiKey'),
          loadAndDecrypt('claude_apiKey'),
        ]);
        const state = get();
        set({
          providers: {
            ...state.providers,
            openai: { ...state.providers.openai, apiKey: openaiKey },
            claude: { ...state.providers.claude, apiKey: claudeKey },
          },
          _keysHydrated: true,
        });
      },
    }),
    {
      name: 'aive-ai-settings',
      partialize: (state) => ({
        providers: {
          ollama: { ...state.providers.ollama, apiKey: undefined },
          openai: { ...state.providers.openai, apiKey: '' },
          claude: { ...state.providers.claude, apiKey: '' },
        },
        defaultProvider: state.defaultProvider,
        customFillerWords: state.customFillerWords,
        clipDurations: state.clipDurations,
        clipSaveLocation: state.clipSaveLocation,
      }),
    },
  ),
);

useAIStore.getState().hydrateKeys();
