// lsp_server.js
'use strict';

globalThis.lsp = {};
const lspServer = globalThis.lsp;

// =============================================================================
// 1. Virtual File System (VFS)
//    Manages in-memory file state, simulating a real file system hierarchy.
// =============================================================================

class VirtualFS {
    constructor() {
        /** @type {Map<string, {content: string, version: number, snapshot: any, kind: number}>} */
        this.files = new Map();
    }

    /**
     * Normalize path: remove 'file://' prefix, unify using '/'.
     */
    normalizePath(path) {
        if (path.startsWith('file://')) {
            path = path.substring(7);
        }
        // Simple handling for Windows style paths
        path = path.replace(/\\/g, '/');
        // Ensure it starts with '/' (like Unix root directory)
        if (!path.startsWith('/')) {
            path = '/' + path;
        }
        return path;
    }

    /**
     * Update or create a file in the VFS.
     * @param {string} fileName 
     * @param {string} content 
     */
    updateFile(fileName, content) {
        const path = this.normalizePath(fileName);
        const prev = this.files.get(path);
        
        // Performance optimization: If content hasn't changed, do not update version.
        // TS checks version to decide if re-parsing is needed.
        if (prev && prev.content === content) {
            return;
        }

        const version = prev ? prev.version + 1 : 1;
        const snapshot = ts.ScriptSnapshot.fromString(content);
        
        // Determine ScriptKind based on extension
        let kind = ts.ScriptKind.TS;
        if (path.endsWith('.js')) kind = ts.ScriptKind.JS;
        else if (path.endsWith('.json')) kind = ts.ScriptKind.JSON;
        else if (path.endsWith('.tsx')) kind = ts.ScriptKind.TSX;

        this.files.set(path, { content, version, snapshot, kind });
    }

    deleteFile(fileName) {
        this.files.delete(this.normalizePath(fileName));
    }

    getFile(fileName) {
        return this.files.get(this.normalizePath(fileName));
    }

    exists(fileName) {
        return this.files.has(this.normalizePath(fileName));
    }

    /**
     * Get all file names (used by the Host).
     */
    getFileNames() {
        return Array.from(this.files.keys());
    }

    /**
     * Simulate directoryExists.
     * Logic: If /a/b.ts exists, then /a directory implicitly exists.
     */
    directoryExists(dirName) {
        const normalizedDir = this.normalizePath(dirName) + '/';
        if (normalizedDir === '//') return true; // Root always exists
        for (const path of this.files.keys()) {
            if (path.startsWith(normalizedDir)) {
                return true;
            }
        }
        return false;
    }

    /**
     * Simulate readDirectory.
     * Critical for TS to find modules within the virtual environment.
     */
    readDirectory(rootDir, extensions, excludes, includes, depth) {
        const normalizedRoot = this.normalizePath(rootDir);
        const results = [];
        
        for (const path of this.files.keys()) {
            // Check if file is under the requested root
            if (!path.startsWith(normalizedRoot)) continue;
            
            // Check extensions
            if (extensions && extensions.length > 0) {
                const hasExt = extensions.some(ext => path.endsWith(ext));
                if (!hasExt) continue;
            }

            results.push(path);
        }
        return results;
    }
}

// =============================================================================
// 2. State & Initialization
// =============================================================================

const vfs = new VirtualFS();
let ts = null;
let languageService = null;
let isInitialized = false;

const COMPILER_OPTIONS = {
    target: 99, // ESNext
    module: 99, // ESNext
    moduleResolution: 2, // Node
    allowJs: true, 
    checkJs: true,
    strict: true,
    noEmit: true,
    esModuleInterop: true,
    jsx: 1, // Preserve
    allowSyntheticDefaultImports: true,
    lib: ["lib.es5.d.ts", "lib.es2015.core.d.ts", "lib.esnext.d.ts", "lib.dom.d.ts"]
};

const ROOT_FILE = "/main.js";

// =============================================================================
// 3. Language Service Host Implementation (Detailed)
// =============================================================================

function createLanguageServiceHost() {
    return {
        // --- Basic Necessary Interfaces ---

        getScriptFileNames: () => vfs.getFileNames(),
        
        getScriptVersion: (fileName) => {
            const file = vfs.getFile(fileName);
            return file ? file.version.toString() : "0";
        },
        
        getScriptSnapshot: (fileName) => {
            const file = vfs.getFile(fileName);
            return file ? file.snapshot : undefined;
        },
        
        getCurrentDirectory: () => "/",
        
        getCompilationSettings: () => COMPILER_OPTIONS,
        
        getDefaultLibFileName: (options) => "/node_modules/typescript/lib/lib.esnext.d.ts",

        // --- File System Simulation Interfaces (Comprehensive) ---

        fileExists: (fileName) => vfs.exists(fileName),
        
        readFile: (fileName) => {
            const file = vfs.getFile(fileName);
            return file ? file.content : undefined;
        },

        // Read directory: Critical for module resolution (e.g., finding @types)
        readDirectory: (rootDir, extensions, excludes, includes, depth) => {
            return vfs.readDirectory(rootDir, extensions, excludes, includes, depth);
        },

        // Directory exists: Used by TS when looking up node_modules/package.json
        directoryExists: (dirName) => {
            return vfs.directoryExists(dirName);
        },

        // Get subdirectories: Required for advanced module resolution
        getDirectories: (dirName) => {
            // Simple implementation: iterate all files to find direct subdirectories
            const normalized = vfs.normalizePath(dirName);
            const subDirs = new Set();
            for (const path of vfs.files.keys()) {
                if (path.startsWith(normalized + '/')) {
                    const relative = path.slice(normalized.length + 1);
                    const parts = relative.split('/');
                    if (parts.length > 1) {
                        subDirs.add(parts[0]);
                    }
                }
            }
            return Array.from(subDirs);
        },

        // --- Path Handling & Debugging Interfaces ---

        // Normalize path, handle symlinks (here simply return normalized path)
        realpath: (path) => vfs.normalizePath(path),

        // Case sensitive: usually true for VFS
        useCaseSensitiveFileNames: () => true,

        // Newline character
        getNewLine: () => "\n",

        // Help TS determine file type (TS, JS, JSON, etc.)
        getScriptKind: (fileName) => {
            const file = vfs.getFile(fileName);
            return file ? file.kind : ts.ScriptKind.Unknown;
        },

        // Debug trace (optional, useful for development)
        // trace: (s) => console.log("[LSP-Host]", s),
    };
}

// =============================================================================
// 4. Core Logic
// =============================================================================

async function initialize() {
    if (isInitialized) return;
    try {
        const tsVersion = '5.9.3';
        console.log(`LSP: Loading TypeScript v${tsVersion}...`);
        //ts = await import(`https://esm.sh/typescript@${tsVersion}`);
        ts = await import('./libs/typescript@5.9.3/index.js');

        if (!ts) throw new Error("TS load failed");

        // Initialize Service (create only once)
        const host = createLanguageServiceHost();
        languageService = ts.createLanguageService(host, ts.createDocumentRegistry());

        // Load default libraries
        await lspServer.loadDts([
            `./libs/typescript@${tsVersion}/lib/lib.es5.d.ts`,
            `./libs/typescript@${tsVersion}/lib/lib.es2015.core.d.ts`,
            `./libs/typescript@${tsVersion}/lib/lib.esnext.d.ts`,
            `./libs/typescript@${tsVersion}/lib/lib.dom.d.ts`,
            '3pty/ndarray.d.ts'
        ]);

        // Initialize main file
        vfs.updateFile(ROOT_FILE, "");
        
        isInitialized = true;
        console.log("LSP: Ready.");
    } catch (e) {
        console.error("LSP: Init failed", e);
    }
}

// =============================================================================
// 5. Public API
// =============================================================================

/**
 * Load .d.ts files from URLs and write them to the Virtual File System.
 */
lspServer.loadDts = async function(paths = []) {
    if (!ts) {
        console.warn("LSP: Not initialized");
        return;
    }

    const fetches = paths.map(async (url) => {
        try {
            let virtualPath;
            const fileName = url.split('/').pop();

            if (url.includes('/lib.')) {
                // Place in TS default library location
                virtualPath = `/node_modules/typescript/lib/${fileName}`;
            } else {
                // Third-party libs go to @types
                const pkgName = fileName.replace('.d.ts', '');
                virtualPath = `/node_modules/@types/${pkgName}/index.d.ts`;
            }

            if (vfs.exists(virtualPath)) return;

            const res = await fetch(url);
            if (!res.ok) throw new Error(res.statusText);
            const text = await res.text();

            vfs.updateFile(virtualPath, text);
            console.log(`LSP: Added ${virtualPath}`);
        } catch (e) {
            console.warn(`LSP: Failed to fetch ${url}`, e);
        }
    });

    await Promise.all(fetches);
};

function mapTsKindToCmType(kind) {
    if (!ts) return 'text';
    switch (kind) {
        case ts.ScriptElementKind.memberVariableElement:
        case ts.ScriptElementKind.variableElement:
        case ts.ScriptElementKind.localVariableElement: return 'variable';
        case ts.ScriptElementKind.memberFunctionElement:
        case ts.ScriptElementKind.functionElement:
        case ts.ScriptElementKind.localFunctionElement: return 'function';
        case ts.ScriptElementKind.classElement: return 'class';
        case ts.ScriptElementKind.interfaceElement: return 'interface';
        case ts.ScriptElementKind.typeElement: return 'type';
        case ts.ScriptElementKind.keyword: return 'keyword';
        case ts.ScriptElementKind.propertyElement: return 'property';
        case ts.ScriptElementKind.moduleElement: return 'namespace';
        default: return 'text';
    }
}

/**
 * Core completion method called by CodeMirror.
 */
lspServer.runStaticCompletions = function(context) {
    if (!isInitialized || !languageService) return [];

    const { fulltext, pos } = context;

    const prefix = "import * as ndarray from 'ndarray';\n(async () => {\n";
    const suffix = "\n})();";
    const wrappedCode = prefix + fulltext + suffix;
    const adjustedPos = pos + prefix.length;

    // 1. Update main file content (triggers version change in VFS)
    vfs.updateFile(ROOT_FILE, wrappedCode);

    try {
        // 2. Get completions from TS
        const completions = languageService.getCompletionsAtPosition(
            ROOT_FILE, 
            adjustedPos, 
            {
                includeCompletionsForModuleExports: true, // Allow auto-import suggestions
                includeInsertTextCompletions: true,
                includeCompletionsWithInsertText: true
            }
        );

        if (!completions || !completions.entries) return [];

        return completions.entries.filter(v=>!v?.name?.startsWith("_")).map((entry, index) => {
            const result = {
                label: entry.name,
                type: mapTsKindToCmType(entry.kind),
                boost: entry.sortText?.startsWith('0') ? 99 : 0
            };

            // Limit
            if (completions.isMemberCompletion) {
                // Fetch details to generate Snippet and JSDoc
                const details = languageService.getCompletionEntryDetails(
                    ROOT_FILE,
                    adjustedPos,
                    entry.name,
                    undefined,
                    undefined,
                    undefined,
                    undefined
                );

                if (details) {
                    // Store extracted types from displayParts signature
                    const paramTypes = {};
                    let returnTypeStr = null;

                    // --- 1. Generate Snippet & Extract Types ---
                    const isFunction = 
                        entry.kind === ts.ScriptElementKind.functionElement ||
                        entry.kind === ts.ScriptElementKind.memberFunctionElement ||
                        entry.kind === ts.ScriptElementKind.methodElement ||
                        entry.kind === ts.ScriptElementKind.localFunctionElement;

                    if (isFunction && details.displayParts) {
                        const params = [];
                        let currentParamName = null;

                        // Iterate parts to build snippet AND extract types
                        for (let i = 0; i < details.displayParts.length; i++) {
                            const part = details.displayParts[i];
                            
                            if (part.kind === 'parameterName') {
                                currentParamName = part.text;
                                paramTypes[currentParamName] = {"names":[""]}; // Init type string

                                // Snippet Logic: Check optionality
                                const nextPart = details.displayParts[i + 1];
                                const isOptional = nextPart && nextPart.text === '?';

                                // Only add parameters that are not optional (no default value)
                                if (part.text != "this" && !isOptional) {
                                    params.push(part.text);
                                }
                            } else if (currentParamName) {
                                // Type Extraction Logic
                                if (part.text === ',' || part.text === ')') {
                                    currentParamName = null; // End of this param
                                } else if (part.text !== ':' && part.text !== '?' && part.text.trim() !== '') {
                                    // Accumulate type text (skips punctuation usually)
                                    paramTypes[currentParamName].names[0]+=part.text;
                                }
                            }
                        }

                        // Rough extraction of return type (everything after the last "):")
                        const fullSig = ts.displayPartsToString(details.displayParts);
                        const retSplit = fullSig.split('):');
                        if (retSplit.length > 1) {
                            returnTypeStr = retSplit.pop().trim();
                        }

                        if(params.length === 0){
                            result.snippet = `${entry.name}()\${0}`;
                        }else{
                            // Build template: name(${param1}, ${param2})
                            const args = params.map((p, i) => `\${${i + 1}:${p}}`).join(', ');
                            result.snippet = `${entry.name}(${args})`;
                        }
                    }

                    // --- 2. Generate JSDoc Object ---
                    const jsdoc = {
                        longname: entry.name,
                        kind: entry.kind,
                        description: ts.displayPartsToString(details.documentation),
                        params: [],
                        returns: [],
                        examples: []
                    };

                    if (details.tags) {
                        details.tags.forEach(tag => {
                            if (tag.name === 'param') {
                                // Extract name and description from tag text usually format: "argName - description"
                                const text = tag.text ? tag.text.map(t => t.text).join(' ').trim() : '';
                                // Simple regex to split first word (param name) from description
                                const match = text.match(/^(\S+)(?:\s*-?\s*)(.*)$/s);
                                if (match) {
                                    jsdoc.params.push({
                                        name: match[1],
                                        description: match[2],
                                        // Match JSDoc param name with TS extracted type
                                        type: paramTypes[match[1]] || null 
                                    });
                                } else {
                                    // Fallback if format is weird
                                    jsdoc.params.push({ name: '', description: text, type: paramTypes[match[1]] || null });
                                }
                            } else if (tag.name === 'return' || tag.name === 'returns') {
                                const text = tag.text ? tag.text.map(t => t.text).join(' ').trim() : '';
                                jsdoc.returns.push({
                                    description: text,
                                    type: null
                                });
                            } else if (tag.name === 'example') {
                                const text = tag.text ? tag.text.map(t => t.text).join(' ').trim() : '';
                                jsdoc.examples.push(text);
                            }
                        });
                    }

                    
                    if (returnTypeStr) {
                        if(jsdoc.returns.length === 0){
                            // Fallback: If no @return tag but we parsed a return type, add it
                            jsdoc.returns.push({ description: '', type: { names : [ returnTypeStr ] }});
                        }else{
                            jsdoc.returns[0].type = { names : [ returnTypeStr ] };
                        }
                    }

                    result.jsdoc = jsdoc;
                }
            }

            return result;
        });

    } catch (e) {
        console.error("LSP: Completion error", e);
        return [];
    }
};

initialize();