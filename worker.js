'use strict';
globalThis.window=globalThis;

// --- Library Initialization ---
const workerLibs = [
    'libbf.js/libbf.js',
    'libbf.js/bf.js',
    '3pty/math.15.1.0.js',
    'box.js',
];
async function loadLibs() {
    try {
        globalThis.linkedom = await import('./3pty/linkedom.js');//'./3pty/linkedom-browser/dist/linkedom.browser.min.js'

        for (const lib of workerLibs) {
            importScripts(lib);            
        }

        let { _Op } = await import('./bpo.js');
        self._Op = _Op;

        bfjs.gc_ele_limit = 100;

        self.postMessage({ type: 'ready' });
    } catch (error) {
        self.postMessage({ type: 'error', payload: { message: `Failed to load libraries: ${error.message}` } });
    }
}

loadLibs();



// --- State and Proxy Setup ---
let pendingPromise = null;
let callid=0;
// --- Proxy Call Handler ---
async function proxyCall(target, property, args) {
    while(!!pendingPromise){
        //console.log('Waiting for pending proxy call to complete before starting a new one, callId:', pendingPromise.callId);
        await pendingPromise.promise;
    }

    const callId = `proxy-${Date.now()}-${callid++}`;
    let promise = new Promise((resolve, reject) => {
        pendingPromise = { callId, resolve, reject };
        //console.log(`Worker proxy call: ${target}.${property} (callId: ${callId})`);
        self.postMessage({
            type: 'proxy',
            payload: { callId, target, property, args }
        });
    });
    pendingPromise.promise = promise;
    return promise;
}

function globalEval(code) {
    //console.log(`Worker globalEval called, code ${code}`);
    return proxyCall('window', 'eval', [code]);
}

function workerhelperCall(funcName, ...args) {
    return proxyCall('workerhelper', funcName, args);
}


function getDocByRefs(refs){
    let ret=[];
    try{
        for(let ref of refs){
            if(globalThis["ndarray"]){      
                const doc=ndarray.help.getHelpDoc(ref);
                if(doc){
                   ret.push(doc);
                   continue;
                }                
            }
            ret.push(null);
        }

    }catch(e){
        console.info(e);
    }
    return ret;
}
function getDoc(fullnames){
    let ret=[];
    for(let fullname of fullnames){
        let ref=null;
        try{                   
            let eval1=eval;            
            ref=eval1(fullname);            
        }catch(e){
            console.info(e);
        }
        ret.push(ref);
    }
    return getDocByRefs(ret);
}

// --- Completion Logic (migrated from editor.mjs) ---
function runCustomCompletions(context) {
    let {text,memberPrefix,pathParts} = context;
    let parentObj = self;
    let ret=[];

    
    try {
        // Resolve the object path from `window`
        parentObj = pathParts.reduce((acc, part) => acc ? acc[part] : undefined, parentObj);
    } catch (e) {
        parentObj = null;
    }
    


    // --- Get properties from the resolved parent object ---
    if (parentObj) {
        //includes prototype
        let propsTemp = [];
        for(let prop in parentObj) {
            propsTemp.push(prop);
        }
        //includs static
        const props = new Set([...Object.getOwnPropertyNames(parentObj), ...propsTemp]);
        for(let prop of props){
            if(prop.toLowerCase().startsWith(memberPrefix.toLowerCase()) && !prop.startsWith('_') && prop !== "prototype") {
              let fullMatch = pathParts.length >= 1 ? pathParts.join('.') + '.' + prop : prop;
              try {
                  const val = parentObj[prop];                  
                  const type = typeof val === 'function' ? 'function' : 'property';
                  let c={ label: prop, fullMatch, type, boost:10};
                  if(typeof val ==="function"){
                    let snippet = createSmartSnippet(val);
                    if(snippet!==null){
                        c.snippet = snippet;
                    }
                  }
                  ret.push(c);
              } catch (e) { // Handle security errors
                  ret.push({ label: prop, fullMatch, type: 'property', boost:10 });
              }
            }
        }
    }
    return ret;
}

// --- Message Handler ---
self.onmessage = async function (e) {
    const { type, payload } = e.data;
    switch (type) {
        case 'execute':
            try {
                await box.runtimeEnter();
                let eval1=eval;
                const result = await eval1(payload.code);
                await box.runtimeExit();
                self.postMessage({ type: 'executionResult', payload: { result } });
            } catch (error) {
                console.error(error);
                self.postMessage({ type: 'executionResult', payload: { error: { message: `${error.name}: ${error.message}\n${error.stack}` } } });
            }
            break;
        case 'proxyResult':
            if(!pendingPromise) break;
            const promise = pendingPromise;
            pendingPromise=null;
            if(payload.callId!==promise.callId) {
                console.error(`Mismatched proxy callId in worker response: expected ${promise.callId}, got ${payload.callId}`);
                break;
            }
            if (promise) {
                if (payload.error) promise.reject(new Error(payload.error.message));
                else promise.resolve(payload.result);
            }
            break;
        case 'getCompletions':
            const result = runCustomCompletions(payload.context);
            self.postMessage({ type: 'completionResult', payload: { result, completionId: payload.completionId } });
            break;
        case 'getDoc':
            {
                const result = getDoc(payload.context);
                self.postMessage({ type: 'docResult', payload: { result, id: payload.id } });
                break;
            }
        case 'getGlobals':
            const globalsForJshint = {};
            Object.keys(self).forEach(k => globalsForJshint[k] = true);
            let readonlyGlobals=["console", "box", "math", "d3", "bfjs", "ndarray"];
            readonlyGlobals.forEach(k => globalsForJshint[k] = false);
            ["self", "thisGlobal"].forEach(k => globalsForJshint[k] = true);
            self.postMessage({ type: 'getGlobalsResult', payload: { globals: globalsForJshint, id: payload.id } });
            break;
    }
};








const functionSnippetCache = new WeakMap();

/**
 * Creates a CodeMirror snippet for a function.
 * Prioritizes JSDoc metadata from getDoc(), falls back to source parsing.
 * 
 * @param {Function} fn - The actual function reference.
 */
function createSmartSnippet(fn) {
    if (functionSnippetCache.has(fn)) {
        return functionSnippetCache.get(fn);
    }
    let ret = null;

    // 1. Attempt to get JSDoc metadata first
    const doc = typeof getDoc === 'function' ? getDocByRefs([fn])[0] : null;

    if (doc) {
        ret = createSnippetFromDoc(doc);
    }

    // 2. Fallback to source code parsing
    //return createSnippetFromSource(fn, fullname);

    if(ret){
        functionSnippetCache.set(fn,ret);
    }

    return ret;
}

/**
 * Generates snippet using JSDoc object structure.
 */
function createSnippetFromDoc(doc) {
    if(doc?.kind !== "function"){
        return null;
    }
    const fnName = doc.name || doc.longname.split(/\.|#/).pop() || "fn";
    
    const placeholders = doc.params?.length?doc.params.filter(param=>param.defaultvalue === undefined).map((param, i) => {
        // Clean param name (JSDoc sometimes uses "options.name" for sub-properties)
        const name = param.name.split('.').pop();
        
        // Construct a helpful placeholder label: name or name=default
        let label = name;
        if (param.defaultvalue !== undefined) {
            label = `${name}=${param.defaultvalue}`.replaceAll(/\{/g,'\\{').replace(/\}/g,'\\}');
        }
        
        return `\${${label}}`;
    }).join(', '):'';

    return `${fnName}(${placeholders})`;
}

/**
 * Generates snippet by parsing Function.toString().
 * Handles Lambda, Async, and Destructuring.
 */
function createSnippetFromSource(fn, fullname) {
    const fnStr = fn.toString().trim();
    
    // Remove comments
    const cleanSource = fnStr.replace(/\/\*[\s\S]*?\*\/|([^\\:]|^)\/\/.*$/gm, '$1');

    // Extract raw parameter string and function name
    const paramsStr = extractParamBlock(cleanSource);
    const params = smartSplitParams(paramsStr);
    
    // Use fullname if provided, otherwise try to parse from source
    let fnName = (fullname && fullname.split('.').pop()) || extractFunctionName(cleanSource) || "fn";
    
    const placeholders = params.filter(p=>p.indexOf("=")==-1).map((p, i) => {
        // For destructuring like {a, b} = {}, we take the left side
        let [name,defaultvalue]=p.split('=');        
        const label = name.trim().replaceAll(/\{/g,'\\{').replace(/\}/g,'\\}');
        return `\${${label}}`;
    }).join(', ');

    return `${fnName}(${placeholders})`;
}

/**
 * Helper: Extracts parameter block supporting (a, b) and x => ...
 */
function extractParamBlock(source) {
    const arrowIdx = source.indexOf('=>');
    const firstParen = source.indexOf('(');

    // Single argument arrow function: arg => ...
    if (arrowIdx !== -1 && (firstParen === -1 || arrowIdx < firstParen)) {
        return source.split('=>')[0].trim();
    }

    // Balanced parentheses extraction
    let depth = 0;
    let start = -1;
    for (let i = 0; i < source.length; i++) {
        if (source[i] === '(') {
            if (depth === 0) start = i;
            depth++;
        } else if (source[i] === ')') {
            depth--;
            if (depth === 0 && start !== -1) return source.slice(start + 1, i);
        }
    }
    return "";
}

/**
 * Helper: Splits parameters by comma but respects nesting {}, [], ()
 */
function smartSplitParams(str) {
    const result = [];
    let current = "";
    let depth = 0;

    for (let i = 0; i < str.length; i++) {
        const char = str[i];
        if (char === ',' && depth === 0) {
            if (current.trim()) result.push(current.trim());
            current = "";
        } else {
            if ('{[('.includes(char)) depth++;
            if ('}])'.includes(char)) depth--;
            current += char;
        }
    }
    if (current.trim()) result.push(current.trim());
    return result;
}

/**
 * Helper: Logic to find function name from source string
 */
function extractFunctionName(source) {
    const match = source.match(/(?:function\s+|const\s+|let\s+|var\s+)?([\w$]+)\s*[:=]?\s*\(|([\w$]+)\s*\(/);
    return match ? (match[1] || match[2]) : null;
}

// --- End of worker.js ---