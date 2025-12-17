import { inlineSuggestion } from "./inline-suggestion";
import type { EditorState } from "@codemirror/state";

/**
 * The suggestion object that can be returned by the suggestion request callback.
 * It can be a simple string or an object with more detailed replacement information.
 */
type Suggestion =
  | string
  | {
      /** The suggested text to insert. */
      text: string;
      /** Number of lines to delete starting from the current cursor's line. */
      linesToDelete?: number;
    };

/**
 * Callback to fetch auto-suggestions from an AI service.
 * @param prefix The text before the cursor.
 * @param suffix The text after the cursor.
 * @returns A promise that resolves to the suggestion.
 */
export type SuggestionRequestCallback = (
  prefix: string,
  suffix: string,
) => Promise<Suggestion>;

/**
 * Wraps a user-provided fetch method.
 */
// eslint-disable-next-line @typescript-eslint/no-unused-vars
function wrapUserFetcher(onSuggestionRequest: SuggestionRequestCallback) {
  return async function (
    prefix: string,
    suffix: string,
    // eslint-disable-next-line @typescript-eslint/no-unused-vars
    _state: EditorState,
  ): Promise<Suggestion> {
    const prediction = await onSuggestionRequest(prefix, suffix);
    return prediction;
  };
}

/**
 * Options to configure the inline copilot extension.
 */
export type CopilotOptions = {
  /**
   * The function to call to fetch suggestions.
   */
  onSuggestionRequest: SuggestionRequestCallback;
  /**
   * Delay in ms to wait after typing to fetch suggestions.
   * Set to 0 or less to disable automatic suggestions.
   * @default 500
   */
  delay?: number;
  /**
   * Whether to accept the suggestion on click.
   * @default true
   */
  acceptOnClick?: boolean;
  /**
   * Keybinding to trigger suggestions manually.
   * @default "Alt-i"
   */
  hotkey?: string;
  /**
   * Maximum number of characters to use as prefix context.
   * @default 1024
   */
  maxPrefix?: number;
  /**
   * Maximum number of characters to use as suffix context.
   * @default 1024
   */
  maxSuffix?: number;
};

/**
 * The main extension to enable inline AI suggestions.
 * This is a wrapper around the more detailed `inlineSuggestion` extension,
 * providing a simpler API for fetching suggestions.
 *
 * @example
 * ```
 * import { EditorView } from "@codemirror/view";
 * import { EditorState } from "@codemirror/state";
 * import { inlineCopilot } from "codemirror-copilot";
 *
 * new EditorView({
 *   state: EditorState.create({
 *     extensions: [
 *       inlineCopilot({
 *         onSuggestionRequest: async (prefix, suffix) => {
 *           // Call your AI API here
 *           return Suggestion;
 *         }
 *       })
 *     ]
 *   }),
 *   parent: document.body
 * });
 * ```
 */
export const inlineCopilot = (options: CopilotOptions) => {
  const { onSuggestionRequest, ...restOptions } = options;
  return inlineSuggestion({
    fetchFn: wrapUserFetcher(onSuggestionRequest),
    ...restOptions,
  });
};
