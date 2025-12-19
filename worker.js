'use strict';
globalThis.window=globalThis;

// --- Library Initialization ---
const workerLibs = [
    'libbf.js/libbf.js',
    'libbf.js/bf.js',
    'math.15.1.0.js',
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
        const props = new Set();
        Object.getOwnPropertyNames(parentObj).forEach(prop => {
            if (prop.toLowerCase().startsWith(memberPrefix.toLowerCase()) && !prop.startsWith('_')) {
              let fullMatch = pathParts.length >= 1 ? pathParts.join('.') + '.' + prop : prop;
              try {
                  const val = parentObj[prop];
                  const type = typeof val === 'function' ? 'function' : 'property';
                  ret.push({ label: prop, fullMatch, type, boost:10});
              } catch (e) { // Handle security errors
                  ret.push({ label: prop, fullMatch, type: 'property', boost:10 });
              }
            }
        });
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
        case 'getGlobals':
            const globalsForJshint = {};
            Object.keys(self).forEach(k => globalsForJshint[k] = true);
            let readonlyGlobals=["console", "box", "math", "d3", "bfjs"];
            readonlyGlobals.forEach(k => globalsForJshint[k] = false);
            ["self", "thisGlobal"].forEach(k => globalsForJshint[k] = true);
            self.postMessage({ type: 'getGlobalsResult', payload: { globals: globalsForJshint, id: payload.id } });
            break;
    }
};
// --- End of worker.js ---