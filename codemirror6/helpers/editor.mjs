
import { javascript, javascriptLanguage } from "@codemirror/lang-javascript";
import { indentWithTab } from "@codemirror/commands";
import { linter, lintGutter } from "@codemirror/lint";
import { createCopilotExtension } from "./aiCopilot.mjs";
import { getStaticCompletions } from "./lsp_helper.mjs";

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
  autocompletion, completionKeymap, closeBrackets, acceptCompletion,
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
    minWidth: "calc( min(300px, 50vw) )",
    whiteSpace: "pre-wrap",
    backgroundColor: "#ffffff", // White background
    color: "#333333", // Dark gray text
    /*border: "1px solid #e0e0e0", // Light gray border
    borderRadius: "6px",
    lineHeight: "1.6",
    boxShadow: "0 4px 12px rgba(0, 0, 0, 0.1)", // Subtle shadow, matching autocomplete list*/
  },
  ".cm-tooltip-doc.noprewrap": {
    whiteSpace: "normal",
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

  ".cm-tooltip-doc span": {
    display: "inline-block",
    marginLeft: "8px",
    fontSize: "0.5em",
    color: "#555555", // Slightly lighter dark gray for body
    opacity: 0.9,
  },
  ".cm-tooltip-doc ul.cm-tooltip-list": {
    listStyle: "none",
    padding: "0",
    margin: "0 0 8px 0",
    border: "1px solid #eaecef",
    backgroundColor: "#f6f8fa", // Very light gray background
  },
  ".cm-tooltip-doc ul.cm-tooltip-list li": {
    fontFamily: "Menlo, Monaco, Consolas, 'Courier New', monospace",
    fontSize: "0.9em",
    color: "#24292e",
    marginBottom: "2px"
  },
  ".cm-tooltip-doc pre": {
    fontFamily: "Menlo, Monaco, Consolas, 'Courier New', monospace",
    fontSize: "0.85em",
    backgroundColor: "#f6f8fa", // Very light gray background
    padding: "8px",
    borderRadius: "4px",
    overflowX: "auto",
    margin: "4px 0 8px 0",
    color: "#24292e",
    border: "1px solid #eaecef"
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


function generateTooltipHtmlForMathHelp(info) {
    const headerHtml = `<h6>${info.name}<span>(${info.category})</span></h6>`;

    const syntaxHtml = `
        <ul class="cm-tooltip-list">
            ${info.syntax.map(s => `<li>${s}</li>`).join('')}
        </ul>
    `;

    const descHtml = `<p>${info.description}</p>`;

    const examplesHtml = `
        <p><strong>Examples:</strong></p>
        <pre>${info.examples.join('\n')}</pre>
    `;

    const seeAlsoHtml = `
        <p><strong>See also:</strong> ${info.seealso.join(', ')}</p>
    `;

    return `<div class="cm-tooltip-doc noprewrap">
            ${headerHtml}
            ${syntaxHtml}
            ${descHtml}
            ${examplesHtml}
            ${seeAlsoHtml}
        </div>`;
}



/**
 * Formats a JSDoc object into an HTML string.
 * @private
 * @param {object} doc - The JSDoc object for a single symbol.
 * @returns {string} A formatted HTML string representing the documentation.
 */
function formatDocHTML(doc) {
    let output = '<div class="cm-tooltip-doc noprewrap">';

    // Header
    output += `<h6>${doc.longname} <span>${doc.kind}</span></h6>`;

    if(doc.description){
        output += `<p >${doc.description }</p>`;
    }    

    if(doc.params && doc.params.length>0){
        output += `<p><strong>Params:</strong></p>`;
        output+= `<ul class="cm-tooltip-list">
                ${doc.params.map(param => {
                    const type = param.type ? `<span>{${param.type.names.join('|')}}</span>` : '';
                    const optional = param.optional ? `<span>[optional]</span>` : '';
                    const defaultValue = param.defaultvalue !== undefined ? `<span>(default: ${JSON.stringify(param.defaultvalue)})</span>` : '';
                    return `<li>
                        <strong>${param.name}</strong>
                        ${type} ${optional} ${defaultValue}
                        ${param.description ? `<p>${param.description}</p>` : ''}
                    </li>`;
                }).join('')}
            </ul>
        `;
    }


    // Returns
    if (doc.returns && doc.returns.length > 0) {
        output += `<p><strong>Returns:</strong></p>`;
        output+= `<ul class="cm-tooltip-list">`;
        for (const ret of doc.returns) {
            const type = ret.type ? `<span class=>{${ret.type.names.join('|')}}</span>` : '';
            output += `<li>${type} - ${ret.description || ''}</li>`;
        }
        output+= `</ul>`;
    }
    

    // Examples
    if (doc.examples && doc.examples.length > 0) {
        output += `<p><strong>Examples:</strong></p>`;   
        output+= `<ul class="cm-tooltip-list">`;     
        for (const example of doc.examples) {
            output += `</li><pre><code>${example}</code></pre></li>`;
        }        
        output+= `</ul>`;
    }

    return output + "</div>";
}



const customCompletions = async (context) => {
    // Regex updated to allow function calls in the chain (e.g., func().prop).
    // It captures:
    // 1. Identifiers: [\w$]+
    // 2. Optional arguments: (?:\( ... \))?
    // 3. Argument content: (?:[^();\n]|\([^();\n]*\))* 
    //    - Matches non-special chars, or one level of nested parentheses, excluding ';', '\n'.
    const match = context.matchBefore(/(?:[\w$]+(?:\((?:[^();\n]|\([^();\n]*\))*\))?\.)*[\w$]*/);
    if (!match || (match.from === match.to && !context.explicit)) {
        return null;
    }

    const found = [];
    const foundKey = new Set();

    function addCompletion(options) {
        const { label, fullMatch, apply, type = 'property', docString, boost = 0 } = options;
        // Avoid duplicates by label
        if (!label ) return;
        if (foundKey.has(label.split('(')[0])) return;

        foundKey.add(label.split('(')[0]);
        found.push({
            label,
            apply: apply || label.split('(')[0],
            type,
            boost,
            info: async () => { // Lazily generate tooltip DOM
              if(fullMatch && fullMatch.startsWith('math.')){
                try{
                  const help=math.help(fullMatch.substring(5));
                  if (!help) return null;
                  const html=generateTooltipHtmlForMathHelp(help.doc);
                  const container = document.createRange().createContextualFragment(html);
                  return container;
                }catch(e){
                  return null;
                }
              }
              if (docString){
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
              try{
                if(options.jsdoc){
                  return document.createRange().createContextualFragment(formatDocHTML(options.jsdoc));
                }
              }catch(e){
                console.log(e);
              }
              try{
                let doc=await workerhelper.getDoc([fullMatch]);
                if(doc && doc[0]){
                  return document.createRange().createContextualFragment(formatDocHTML(doc[0]));
                }
              }catch(e){
                console.log(e);
              }
              return null;
            
            }
        });
    }

    const text = match.text;
    const parts = text.split('.');    

    let memberPrefix = parts.pop();
    let pathParts = parts;
    let fromPos = match.to - memberPrefix.length;

    //force empty prefix to get all members, leave filtering to CM
    memberPrefix="";


    // --- Logic for `box` members ---
    // For `box.`, provide documented completions from `function_info`
    if (text.toLowerCase().startsWith('box.') || parts.length <= 0) {
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
    let ctx = {
        text,
        memberPrefix,
        pathParts,
        fulltext: context.state.doc.toString(),
        pos: context.pos
    };
    const staticCompletions = getStaticCompletions(ctx);    
    // --- Get properties ---
    // Only invoke dynamic completion if there are no function calls (parentheses) in the chain.
    // Dynamic completion supports full property chains but not function return types.
    let dynamicResults = null;
    if (text.indexOf('(') === -1) {
        dynamicResults = await workerhelper.getCompletions(ctx);
    }
    // Process and add completions from both sources
    if (dynamicResults) {
        for (let item of dynamicResults) {
            if (item['snippet']) {
                item.apply = snippet(item.snippet);
            }
            addCompletion(item);
        }
    }
    const staticResults = await staticCompletions;
    if (staticResults) {
        for (let item of staticResults) {
            if (item['snippet']) {
                item.apply = snippet(item.snippet);
            }
            addCompletion(item);
        }
    }

    if (found.length === 0) return null;

    return {
        from: fromPos,
        options: found,
        // Updated validFor to match the regex logic used in matchBefore
        validFor: /^(?:[\w$]+(?:\((?:[^();\n]|\([^();\n]*\))*\))?\.)*[\w$]+?$/
    };
};


// --- LINTING LOGIC ---
const jshintLinter = linter(async view => {
    if (typeof JSHINT === 'undefined') {
        console.warn("JSHINT not loaded, skipping lint.");
            resolve([]);
            return;
    }

    let diagnostics = [];
    try {
        // Get the execution context's globals from the worker
        const globalsForJshint = await workerhelper.getWorkerGlobals();

        let code="async function _noname(){\n"+view.state.doc.toString()+"\n}"
        JSHINT(code, { esversion: 11, asi: true, /*undef: true,*/ browser: true, devel: true, typed: true, globals: globalsForJshint });
        const errors = JSHINT.data()?.errors;
        
        if (errors) {
            diagnostics = errors.map(e => {
                if (!e) return null;
                const lineNo = Math.max(1, Math.min(e.line - 1, view.state.doc.lines));
                const line = view.state.doc.line(lineNo);
                let from = line.from + e.character - 1;
                if (from < line.from || from > line.to) from = line.from;
                let to = from + 1;
                if (e.evidence) {
                    const match = e.evidence.match(/\S*$/);
                    if (match) to = from + match[0].length;
                }
                return { from, to, severity: e.code?.[0] === 'W' ? "warning" : "error", message: `${e.reason} (${e.code})` };
            });
        }
    } catch (e) {
        console.error("[JSHint Linter Failure]:", e);
    }
    return diagnostics;
});

/*function acceptCompletionWithDot(view){
  acceptCompletion(view);
  //alway return false to pass the dot to the next
  return false;
}*/

// --- VUE COMPONENT ---
const CodeMirror6VueComponent = {
  props: ['modelValue', 'hasFocus'],
  emits: ['update:modelValue', 'focus'],
  template: '<div ref="editor"></div>',
  setup(props) {
  },
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
            EditorView.lineWrapping, // Add this line for word wrapping
            
            // AI Copilot Integration
            createCopilotExtension(function_info),

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
                //{ key: ".", run: acceptCompletionWithDot },
                // Keys related to the linter system
                ...lintKeymap
            ])
        ]
      }),
      parent: this.$refs.editor
    });
    this.editorView?.focus();
  },
  watch: {
    modelValue(newValue) {
      if (this.editorView && newValue !== this.editorView.state.doc.toString()) {
        this.editorView.dispatch({ changes: { from: 0, to: this.editorView.state.doc.length, insert: newValue } });
      }
    },
    hasFocus(newValue) {
      if (newValue) {
        this.editorView?.focus();
      }
    }
  },
  beforeUnmount() { if (this.editorView) this.editorView.destroy(); },
};

window.CodeMirror6VueComponent = CodeMirror6VueComponent;