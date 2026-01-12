'use strict';

// --- CONFIGURATION ---
let staticCompletionPromises = new Map();
let lspWorker = new Worker('lsp_worker.js');

// Handle messages from the LSP worker
lspWorker.onmessage = function(e) {
    const { type, payload } = e.data;
    if (type === 'staticCompletionResult') {
        const promise = staticCompletionPromises.get(payload.completionId);
        if (promise) {
            promise.resolve(payload.result);
            staticCompletionPromises.delete(payload.completionId);
        }
    } else if (type === 'error') {
        console.error('LSP Worker error:', payload.message);
    }
};

lspWorker.onerror = function(error) {
    console.error("An error occurred in lsp_worker.js:", error);
    for (const promise of staticCompletionPromises.values()) {
        promise.reject(new Error("LSP Worker failed."));
    }
    staticCompletionPromises.clear();
};


/**
 * Gets static completions. Executes in a worker.
 * @param {object} context - The completion context from CodeMirror.
 * @returns {Promise<Array>} A promise that resolves to an array of completion items.
 */
export function getStaticCompletions(context) {
    const completionId = Date.now() + Math.random();
    const promise = new Promise((resolve, reject) => {
        staticCompletionPromises.set(completionId, { resolve, reject });
    });

    lspWorker.postMessage({
        type: 'getStaticCompletions',
        payload: { context, completionId }
    });

    setTimeout(() => {
        if (staticCompletionPromises.has(completionId)) {
            staticCompletionPromises.get(completionId).reject(new Error("LSP completion request timed out."));
            staticCompletionPromises.delete(completionId);
        }
    }, 5000);

    return promise;
}
