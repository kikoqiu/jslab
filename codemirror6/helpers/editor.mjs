
import { javascript, javascriptLanguage } from "@codemirror/lang-javascript";
import { indentWithTab } from "@codemirror/commands";
import { linter, lintGutter } from "@codemirror/lint";

import { EditorState} from "@codemirror/state"
import {
  EditorView, keymap, highlightSpecialChars, drawSelection,
  highlightActiveLine, dropCursor, rectangularSelection,
  crosshairCursor, lineNumbers, highlightActiveLineGutter
} from "@codemirror/view"
import {
  defaultHighlightStyle, syntaxHighlighting, indentOnInput,
  bracketMatching, foldGutter, foldKeymap
} from "@codemirror/language"
import {
  defaultKeymap, history, historyKeymap
} from "@codemirror/commands"
import {
  searchKeymap, highlightSelectionMatches
} from "@codemirror/search"
import {
  autocompletion, completionKeymap, closeBrackets,
  closeBracketsKeymap, snippet
} from "@codemirror/autocomplete"
import {lintKeymap} from "@codemirror/lint"


// --- THEME ---
const theme = EditorView.theme({
  "&": { color: "black", backgroundColor: "white" },
  ".cm-content": { caretColor: "black" },
  "&.cm-focused": { outline: "none" },
  // --- Refined Light-theme tooltip style ---
  ".cm-tooltip.cm-tooltip-autocomplete": {
    background: "#ffffff", // White background for the autocomplete list
    border: "1px solid #e0e0e0", // Light gray border
    borderRadius: "6px",
    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)", // Subtle shadow
  },
  ".cm-tooltip.cm-tooltip-autocomplete > ul": {
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif",
    padding: "4px 0", // Some vertical padding for list items
    maxHeight: "300px", // Limit height for autocomplete list
    overflowY: "auto", // Enable scrolling for long lists
  },
  ".cm-tooltip.cm-tooltip-autocomplete ul li": {
    padding: "6px 12px", // Padding for each list item
    cursor: "pointer",
    color: "#333333", // Dark text for suggestions
  },
  ".cm-tooltip.cm-tooltip-autocomplete ul li.cm-active": {
    background: "#e9f2ff", // Light blue background for active item
    color: "#000000", // Darker text for active item
  },
  ".cm-tooltip-doc": {
    display: 'block',
    fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif", // Sans-serif for description
    padding: "10px 15px",
    maxWidth: "600px",
    whiteSpace: "pre-wrap",
    backgroundColor: "#ffffff", // White background
    color: "#333333", // Dark gray text
    border: "1px solid #e0e0e0", // Light gray border
    borderRadius: "6px",
    lineHeight: "1.6",
    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)", // Subtle shadow, matching autocomplete list
  },
  ".cm-tooltip-doc h6": {
    // Signature heading
    fontFamily: "Menlo, Monaco, Consolas, 'Courier New', monospace", // Monospace for signature
    fontWeight: "600",
    margin: "0 0 8px 0",
    color: "#007acc", // A nice VS Code-like blue for function names
    fontSize: "1em",
    borderBottom: "1px solid #f0f0f0", // Very light separator
    paddingBottom: "8px",
  },
  ".cm-tooltip-doc p": {
    // Documentation body
    margin: "8px 0 0 0",
    fontSize: "0.9em",
    color: "#555555", // Slightly lighter dark gray for body
    opacity: 0.9,
  },
  // Scrollbar styling for a consistent look
  ".cm-tooltip-doc::-webkit-scrollbar": {
    width: "8px",
  },
  ".cm-tooltip-doc::-webkit-scrollbar-thumb": {
    backgroundColor: "#cccccc", // Muted scrollbar color
    borderRadius: "4px",
    border: "2px solid #ffffff", // Padding around thumb
  },
  ".cm-tooltip-doc::-webkit-scrollbar-track": {
    background: "transparent",
  },
  ".cm-tooltip.cm-completionInfo": { // For older CodeMirror versions where doc tooltip is separate
    background: "#ffffff",
    border: "1px solid #e0e0e0",
    borderRadius: "6px",
    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)",
  },
});


// --- COMPLETION LOGIC ---
let function_info = [];
async function gendoc() {
  try {
    const babeldoc = await import('./babel-doc.mjs');
    Babel.registerPlugin("babeldoc", babeldoc.default);
    
    let boxjs = await fetch('./box.js');
    boxjs = await boxjs.text();
    
    Babel.transform("'babeldoc enable';\r\n" + boxjs, {
      parserOpts: { strictMode: false },
      plugins: ["babeldoc"],
    });
    for(let key of Object.keys(babeldoc.functions)){
      if(key.startsWith("box.")){
        function_info.push({key,'value':babeldoc.functions[key]});
      }
    }
    
    console.log("JSDoc info successfully generated for", function_info.length, "functions.");
  } catch (e) {
    console.error("Failed to generate JSDoc info:", e);
  }
}
gendoc();

function createSnippet(signature) {
    const openParen = signature.indexOf('(');
    const closeParen = signature.lastIndexOf(')');
    if (openParen === -1 || closeParen === -1) {
        return signature; // Not a function signature, return as is
    }

    const name = signature.substring(0, openParen);
    const argsStr = signature.substring(openParen + 1, closeParen);

    if (!argsStr.trim()) {
        return `${name}()`; // Function with no arguments
    }

    // Process arguments into snippet placeholders
    const snippetArgs = argsStr.split(',').map(arg => `\${${arg.trim()}}`).join(', ');

    return snippet(`${name}(${snippetArgs})`);
}

const customCompletions = (context) => {
    // This regex is more robust, capturing chains of properties.
    const match = context.matchBefore(/(?:[\w$]+\.)*[\w$]*/);
    if (!match || (match.from === match.to && !context.explicit)) {
        return null;
    }

    const found = [];
    const foundKey = new Set();

    function addCompletion(options) {
        const { label, apply, type = 'property', docString, boost = 0 } = options;
        // Avoid duplicates by label
        if (!label ) return;
        if (foundKey.has(label.split('(')[0])) return;

        foundKey.add(label.split('(')[0]);
        found.push({
            label,
            apply: apply || label.split('(')[0],
            type,
            boost,
            info: () => { // Lazily generate tooltip DOM
                if (!docString) return null;
                const container = document.createElement('div');
                container.className = 'cm-tooltip-doc';
                const sig = docString.split('\n')[0];
                const doc = docString.split('\n').slice(1).join('\n').trim();
                const sigElement = document.createElement('h6');
                sigElement.textContent = sig;
                container.appendChild(sigElement);
                if (doc) {
                    const docElement = document.createElement('p');
                    docElement.textContent = doc;
                    container.appendChild(docElement);
                }
                return container;
            }
        });
    }

    const text = match.text;
    const parts = text.split('.');    

    // --- Logic for general `window` and other objects ---
    let parentObj = window;
    let memberPrefix = text;
    let fromPos = match.from;

    // If there is a dot, we are accessing a property.
    if (parts.length > 1 || text.endsWith('.')) {
        let pathParts;
        if (text.endsWith('.')) {
            memberPrefix = "";
            pathParts = parts.slice(0, -1);
        } else {
            memberPrefix = parts.pop();
            pathParts = parts;
        }

        try {
            // Resolve the object path from `window`
            parentObj = pathParts.reduce((acc, part) => acc ? acc[part] : undefined, window);
            fromPos = match.to - memberPrefix.length;
        } catch (e) {
            parentObj = null;
        }
    }

    // --- Logic for `box` members ---
    // For `box.`, provide documented completions from `function_info`
    if (text.toLowerCase().startsWith('box.') || parentObj == window) {
        function_info.forEach(({key,value}) => {
          const memberDoc = value.split('\n',2)[0].substring(4);
          if (memberDoc.toLowerCase().startsWith(memberPrefix.toLowerCase())) {
              const snippet = createSnippet(memberDoc);
              addCompletion({ label: memberDoc, apply: snippet, type: 'function', docString:value, boost: 90 });
          }            
        });
        if(text.toLowerCase().startsWith('box.')){
          // We've handled `box.` completions, so we can return.
          return { from: match.from + 4, options: found, validFor: /^\w*$/ };
        }
    }


    // --- Get properties from the resolved parent object ---
    if (parentObj) {
        const props = new Set();
        Object.getOwnPropertyNames(parentObj).forEach(prop => {
            if (prop.toLowerCase().startsWith(memberPrefix.toLowerCase()) && !prop.startsWith('_')) {
                try {
                    const val = parentObj[prop];
                    const type = typeof val === 'function' ? 'function' : 'property';
                    addCompletion({ label: prop, type, boost:10});
                } catch (e) { // Handle security errors
                    addCompletion({ label: prop, type: 'property', boost:10 });
                }
            }
        });
    }

    if (found.length === 0) return null;

    return {
        from: fromPos,
        options: found,
        validFor: /^(?:[\w$]+\.)*[\w$]+?$/
    };
};


// --- LINTING LOGIC ---
const jshintLinter = linter(view => {
    return new Promise(resolve => {
        if (typeof JSHINT === 'undefined') {
            console.warn("JSHINT not loaded, skipping lint.");
            resolve([]);
            return;
        }
        let diagnostics = [];
        try {
            const globalsForJshint = { "console": false, "window": false, "document": false, "box":false };
            if (window.box) Object.keys(window.box).forEach(k => globalsForJshint[k] = false);
            if (window.math) globalsForJshint['math'] = false;
            if (window.d3) globalsForJshint['d3'] = false;
            
            JSHINT(view.state.doc.toString(), { esversion: 11, asi: true, undef: true, browser: true, globals: globalsForJshint });
            const errors = JSHINT.data()?.errors;
            
            if (errors) {
                diagnostics = errors.map(e => {
                    if (!e) return null;
                    const lineNo = Math.min(e.line, view.state.doc.lines);
                    const line = view.state.doc.line(lineNo);
                    let from = line.from + e.character - 1;
                    if (from < line.from || from > line.to) from = line.from;
                    let to = from + 1;
                    if (e.evidence) {
                        const match = e.evidence.match(/\S*$/);
                        if (match) to = from + match[0].length;
                    }
                    return { from, to, severity: e.code?.[0] === 'W' ? "warning" : "error", message: `${e.reason} (${e.code})` };
                }).filter(Boolean);
            }
        } catch (e) {
            console.error("[JSHint Linter Failure]:", e);
        }
        resolve(diagnostics);
    });
});


// --- VUE COMPONENT ---
const CodeMirror6VueComponent = {
  props: ['modelValue'], emits: ['update:modelValue', 'focus'], template: '<div ref="editor"></div>',
  mounted() {
    this.editorView = new EditorView({
      state: EditorState.create({
        doc: this.modelValue,
        extensions: [
            javascript(),
            //javascriptLanguage, // Provides default JS completions & snippets
            javascriptLanguage.data.of({ // Adds our custom source to the completion pool
                autocomplete: customCompletions
            }),
            autocompletion(),   // Enables the autocompletion UI

            jshintLinter,
            lintGutter(),
            theme,
            EditorView.updateListener.of(update => {
                if (update.docChanged) this.$emit('update:modelValue', update.state.doc.toString());
                if (update.focusChanged && update.view.hasFocus) this.$emit('focus');
            }),



            // A line number gutter
            lineNumbers(),
            // A gutter with code folding markers
            foldGutter(),
            // Replace non-printable characters with placeholders
            highlightSpecialChars(),
            // The undo history
            history(),
            // Replace native cursor/selection with our own
            drawSelection(),
            // Show a drop cursor when dragging over the editor
            dropCursor(),
            // Allow multiple cursors/selections
            EditorState.allowMultipleSelections.of(true),
            // Re-indent lines when typing specific input
            indentOnInput(),
            // Highlight syntax with a default style
            syntaxHighlighting(defaultHighlightStyle),
            // Highlight matching brackets near cursor
            bracketMatching(),
            // Automatically close brackets
            closeBrackets(),
            // Load the autocompletion system
            autocompletion(),
            // Allow alt-drag to select rectangular regions
            rectangularSelection(),
            // Change the cursor to a crosshair when holding alt
            crosshairCursor(),
            // Style the current line specially
            highlightActiveLine(),
            // Style the gutter for current line specially
            highlightActiveLineGutter(),
            // Highlight text that matches the selected text
            highlightSelectionMatches(),
            keymap.of([
                { key: "Ctrl-Enter", run: () => true },
                indentWithTab,
                // Closed-brackets aware backspace
                ...closeBracketsKeymap,
                // A large set of basic bindings
                ...defaultKeymap,
                // Search-related keys
                ...searchKeymap,
                // Redo/undo keys
                ...historyKeymap,
                // Code folding bindings
                ...foldKeymap,
                // Autocompletion keys
                ...completionKeymap,
                // Keys related to the linter system
                ...lintKeymap
            ])
        ]
      }),
      parent: this.$refs.editor
    });
  },
  watch: {
    modelValue(newValue) {
      if (this.editorView && newValue !== this.editorView.state.doc.toString()) {
        this.editorView.dispatch({ changes: { from: 0, to: this.editorView.state.doc.length, insert: newValue } });
      }
    }
  },
  beforeUnmount() { if (this.editorView) this.editorView.destroy(); },
};

window.CodeMirror6VueComponent = CodeMirror6VueComponent;