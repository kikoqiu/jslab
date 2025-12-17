import {
  ViewPlugin,
  DecorationSet,
  EditorView,
  ViewUpdate,
  Decoration,
  WidgetType,
  keymap,
  Command,
} from "@codemirror/view";
import {
  StateEffect,
  Text,
  Facet,
  Prec,
  StateField,
  EditorState,
  TransactionSpec,
} from "@codemirror/state";
import { debouncePromise } from "./lib/utils";

/**
 * The suggestion object, which can be a simple string
 * or an object with more detailed replacement information.
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
 * The inner method to fetch suggestions.
 * @param prefix The text before the cursor, truncated to `maxPrefix`.
 * @param suffix The text after the cursor, truncated to `maxSuffix`.
 * @param state The editor state.
 */
type InlineFetchFn = (
  prefix: string,
  suffix: string,
  state: EditorState,
) => Promise<Suggestion>;

/**
 * The normalized suggestion format stored in the state.
 */
type InlineSuggestion = {
  text: string;
  /** Number of lines to delete starting from the current cursor's line. */
  linesToDelete: number;
};

const InlineSuggestionState = StateField.define<{
  suggestion: null | InlineSuggestion;
}>({
  create() {
    return { suggestion: null };
  },
  update(previousValue, tr) {
    const inlineSuggestion = tr.effects.find((e) =>
      e.is(InlineSuggestionEffect),
    );
    if (tr.state.doc) {
      if (inlineSuggestion && tr.state.doc == inlineSuggestion.value.doc) {
        return { suggestion: inlineSuggestion.value.suggestion };
      } else if (!tr.docChanged && !tr.selection) {
        return previousValue;
      }
    }
    return { suggestion: null };
  },
});

const InlineSuggestionEffect = StateEffect.define<{
  suggestion: InlineSuggestion | null;
  doc: Text;
}>();

const TriggerSuggestionEffect = StateEffect.define<void>();

function inlineSuggestionDecoration(
  view: EditorView,
  suggestion: InlineSuggestion,
) {
  const pos = view.state.selection.main.head;
  const w = Decoration.widget({
    widget: new InlineSuggestionWidget(suggestion),
    side: 1,
  });
  return Decoration.set([w.range(pos)]);
}

export const suggestionConfigFacet = Facet.define<
  Partial<InlineSuggestionOptions & { fetchFn?: InlineFetchFn }>,
  Required<InlineSuggestionOptions> & { fetchFn?: InlineFetchFn }
>({
  combine(values) {
    const last = values[values.length - 1] || {};
    return {
      fetchFn: last.fetchFn!,
      delay: last.delay ?? 500,
      acceptOnClick: last.acceptOnClick ?? true,
      hotkey: last.hotkey ?? "Alt-i",
      maxPrefix: last.maxPrefix ?? 1024,
      maxSuffix: last.maxSuffix ?? 1024,
    };
  },
});

class InlineSuggestionWidget extends WidgetType {
  suggestion: InlineSuggestion;

  constructor(suggestion: InlineSuggestion) {
    super();
    this.suggestion = suggestion;
  }

  toDOM(view: EditorView) {
    const span = document.createElement("span");
    span.style.opacity = "0.4";
    span.className = "cm-inline-suggestion";
    span.textContent = this.suggestion.text;
    span.onclick = (e) => this.accept(e, view);
    return span;
  }

  accept(e: MouseEvent, view: EditorView) {
    const config = view.state.facet(suggestionConfigFacet);
    if (!config.acceptOnClick) return;
    e.stopPropagation();
    e.preventDefault();
    acceptSuggestion(view);
    return true;
  }
}

export const fetchSuggestion = ViewPlugin.fromClass(
  class Plugin {
    debouncedFetch: (view: EditorView, state: EditorState) => Promise<void>;
    constructor(view: EditorView) {
      const config = view.state.facet(suggestionConfigFacet);
      this.debouncedFetch = debouncePromise(
        (v: EditorView, s: EditorState) => this.fetch(v, s),
        config.delay,
      );
      if (config.delay > 0) {
        this.debouncedFetch(view, view.state);
      }
    }

    async update(update: ViewUpdate) {
      const config = update.view.state.facet(suggestionConfigFacet);
      const shouldFetch =
        (update.docChanged && config.delay > 0) ||
        update.transactions.some((tr) =>
          tr.effects.some((e) => e.is(TriggerSuggestionEffect)),
        );

      if (shouldFetch) {
        this.debouncedFetch(update.view, update.state);
      }
    }

    async fetch(view: EditorView, state: EditorState) {
      const config = state.facet(suggestionConfigFacet);
      if (!config.fetchFn) return;

      const { doc, selection } = state;
      const pos = selection.main.head;
      let prefix = doc.sliceString(0, pos);
      let suffix = doc.sliceString(pos, doc.length);

      if (prefix.length > config.maxPrefix) {
        const truncated = prefix.substring(prefix.length - config.maxPrefix);
        const newlinePos = truncated.indexOf("\n");
        prefix =
          newlinePos !== -1 ? truncated.substring(newlinePos + 1) : truncated;
      }
      if (suffix.length > config.maxSuffix) {
        const truncated = suffix.substring(0, config.maxSuffix);
        const newlinePos = truncated.lastIndexOf("\n");
        suffix =
          newlinePos !== -1 ? truncated.substring(0, newlinePos) : truncated;
      }

      try {
        const result = await config.fetchFn(prefix, suffix, state);
        let suggestion: InlineSuggestion | null = null;
        if (result) {
          if (typeof result === "string") {
            suggestion = { text: result, linesToDelete: 0 };
          } else {
            suggestion = {
              text: result.text,
              linesToDelete: result.linesToDelete ?? 0,
            };
          }
        }

        view.dispatch({
          effects: InlineSuggestionEffect.of({ suggestion, doc: state.doc }),
        });
      } catch (e) {
        console.error("codemirror-copilot: fetchFn error", e);
      }
    }
  },
);

const suggestionRemovalTheme = EditorView.baseTheme({
  ".cm-suggestion-toremove": { textDecoration: "line-through", opacity: "0.5" },
});

const renderRemovalDecorationPlugin = ViewPlugin.fromClass(
  class {
    decorations: DecorationSet;
    constructor() {
      this.decorations = Decoration.none;
    }
    update(update: ViewUpdate) {
      const suggestion = update.state.field(InlineSuggestionState)?.suggestion;
      if (!suggestion || suggestion.linesToDelete === 0) {
        this.decorations = Decoration.none;
        return;
      }
      const pos = update.state.selection.main.head;
      const currentLine = update.state.doc.lineAt(pos);

      let from: number;
      let to: number;

      from = currentLine.from;
      const endLineNumber = Math.min(update.state.doc.lines, currentLine.number + suggestion.linesToDelete - 1);
      to = update.state.doc.line(endLineNumber).to;
      if(from === to){
        this.decorations = Decoration.none;
        return;
      }
      this.decorations = Decoration.set([
        Decoration.mark({
          class: "cm-suggestion-toremove",
        }).range(from, to),
      ]);
    }
  },
  {
    decorations: (v) => v.decorations,
  },
);

const renderInlineSuggestionPlugin = ViewPlugin.fromClass(
  class Plugin {
    decorations: DecorationSet;
    constructor() {
      this.decorations = Decoration.none;
    }
    update(update: ViewUpdate) {
      const suggestion = update.state.field(InlineSuggestionState)?.suggestion;
      this.decorations = suggestion
        ? inlineSuggestionDecoration(update.view, suggestion)
        : Decoration.none;
    }
  },
  {
    decorations: (v) => v.decorations,
  },
);

export const triggerSuggestion: Command = (view) => {
  view.dispatch({ effects: TriggerSuggestionEffect.of() });
  return true;
};

export const dismissSuggestion: Command = (view) => {
  if (!view.state.field(InlineSuggestionState)?.suggestion) {
    return false;
  }
  view.dispatch({
    effects: InlineSuggestionEffect.of({
      suggestion: null,
      doc: view.state.doc,
    }),
  });
  return true;
};

function acceptSuggestion(view: EditorView): boolean {
  const suggestion = view.state.field(InlineSuggestionState)?.suggestion;
  if (!suggestion) return false;

  const head = view.state.selection.main.head;
  const currentLine = view.state.doc.lineAt(head);

  let from_calculated: number;
  let to_calculated: number;

  // Determine 'from_calculated'
  if (suggestion.linesToDelete === 0) {
      from_calculated = head;
      to_calculated = head;
  } else {
      from_calculated = currentLine.from;
      const endLineNumber = Math.min(view.state.doc.lines, currentLine.number + suggestion.linesToDelete - 1);
      to_calculated = view.state.doc.line(endLineNumber).to;
  }

  view.dispatch({
    ...insertCompletionText(
      suggestion.text,
      from_calculated,
      to_calculated,
    ),
  });
  return true;
}

const inlineSuggestionKeymap = Prec.highest(
  keymap.of([
    { key: "Tab", run: acceptSuggestion },
    { key: "Escape", run: dismissSuggestion },
  ]),
);

function insertCompletionText(
  text: string,
  from: number,
  to: number,
): TransactionSpec {
  return {
    changes: { from, to, insert: text },
    selection: { anchor: from + text.length },
    userEvent: "input.complete",
  };
}

export type InlineSuggestionOptions = {
  /** The function to fetch suggestions. */
  fetchFn: InlineFetchFn;
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
 * The main extension to enable inline suggestions.
 */
export function inlineSuggestion(options: InlineSuggestionOptions) {
  const triggerKeymap = keymap.of([
    {
      key: options.hotkey || "Alt-i",
      run: triggerSuggestion,
    },
  ]);

  return [
    suggestionConfigFacet.of(options),
    InlineSuggestionState,
    fetchSuggestion,
    renderInlineSuggestionPlugin,
    renderRemovalDecorationPlugin,
    suggestionRemovalTheme,
    inlineSuggestionKeymap,
    triggerKeymap,
  ];
}
