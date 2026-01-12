// lsp_worker.js
'use strict';

// Import the core logic. 
// It is designed to be environment-agnostic.
importScripts('lsp_server.js');

// The worker's job is to be a simple bridge:
// receive a message, call the LSP logic, and post the result back.
self.onmessage = async (e) => {
    const { type, payload } = e.data;

    if (type === 'getStaticCompletions') {
        if (!globalThis.lsp || typeof globalThis.lsp.runStaticCompletions !== 'function') {
            // LSP server isn't ready yet.
            self.postMessage({
                type: 'staticCompletionResult',
                payload: { result: [], completionId: payload.completionId }
            });
            return;
        }

        // Call the core logic and get the results
        const result = await globalThis.lsp.runStaticCompletions(payload.context);
        
        // Post the results back to the main thread
        self.postMessage({
            type: 'staticCompletionResult',
            payload: { result, completionId: payload.completionId }
        });
    }
};
