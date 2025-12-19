// Main-thread helper for Web Worker integration.
'use strict';

window.workerhelper = {};

// The real vm instance will be assigned to the global `vm` variable after the Vue app is mounted.

const worker = new Worker('worker.js');
let workerReady = false;
let executionPromise = null;
let completionPromises = new Map();
let getGlobalsPromises = new Map();

// --- Worker Communication ---

// Handle messages from the worker
worker.onmessage = function(e) {
    const { type, payload } = e.data;
    switch (type) {
        case 'ready':
            console.log('Worker is ready.');
            workerReady = true;
            // Now that the worker is ready, send it the documentation info.
            // This relies on `function_info` being populated globally by editor.mjs
            if (typeof function_info !== 'undefined') {
                worker.postMessage({ type: 'setFunctionInfo', payload: function_info });
            }
            break;
        case 'executionResult':
            if (executionPromise) {
                if (payload.error) {
                    executionPromise.reject(new Error(payload.error.message));
                } else {
                    executionPromise.resolve(payload.result);
                }
                executionPromise = null;
            }
            break;
        case 'completionResult':
            const completionPromise = completionPromises.get(payload.completionId);
            if (completionPromise) {
                completionPromise.resolve(payload.result);
                completionPromises.delete(payload.completionId);
            }
            break;
        case 'getGlobalsResult':
            const linterPromise = getGlobalsPromises.get(payload.id);
            if (linterPromise) {
                linterPromise.resolve(payload.globals);
                getGlobalsPromises.delete(payload.id);
            }
            break;
        case 'proxy':
            handleProxyCall(payload);
            break;
        case 'error':
            console.error('Worker error:', payload.message);
            if (executionPromise) {
                executionPromise.reject(new Error(payload.message));
                executionPromise = null;
            }
            break;
    }
};

// Executes calls that the worker proxies to the main thread
function handleProxyCall(payload) {
    const { callId, target, property, args } = payload;
    let result;
    try {
        let targetObj;
        if (target === 'window') {
            targetObj = window; // The real window object
        } else if (target === 'workerhelper') {
            targetObj = window.workerhelper; // The real helper object
        } else {
            throw new Error(`Proxy target '${target}' not supported.`);
        }

        const func = targetObj[property];
        if (typeof func !== 'function') {
            // This handles properties
            throw new Error(`Property '${property}' is not a function on target '${target}'.`);
        } else {
            result = func.apply(targetObj, args);
        }
        
        if (result instanceof Promise) {
            result.then(promiseResult => {
                worker.postMessage({ type: 'proxyResult', payload: { callId, result: promiseResult } });
            }).catch(error => {
                worker.postMessage({ type: 'proxyResult', payload: { callId, error: { message: error.message } } });
            });
        } else {
            worker.postMessage({ type: 'proxyResult', payload: { callId, result } });
        }

    } catch (error) {
        console.error(`Proxy call failed: ${target}.${property}`, error);
        worker.postMessage({ type: 'proxyResult', payload: { callId, error: { message: error.message } } });
    }
}


// --- Override original box.js functions ---

// This function is called by the Vue app. It delegates execution to the worker.
workerhelper.runcode = function (code, info) {
    if (!workerReady) {
        return Promise.reject(new Error("Worker is not ready yet. Please wait a moment."));
    }
    const tcode =   
`(async () => {'bpo enable';
${code}
})();`
    const compiledCode = Babel.transform(tcode,     { 
      presets: [
        /*[
          "env",
          {
            exclude:[
              "@babel/plugin-transform-async-to-generator",
              '@babel/plugin-transform-regenerator',
              '@babel/plugin-transform-destructuring',              
            ],
            useBuiltIns:false
          }
        ],*/
      ] ,//env Babel.availablePresets//"es2017"
      plugins: ["bpo"],
      sourceType: "script",
      sourceMaps:"inline",
    } ).code;

    if (info) {
        info.compiled_code = compiledCode;
    }
    console.log("Compiled code:", compiledCode);

    const promise = new Promise((resolve, reject) => {
        executionPromise = { resolve, reject };
    });

    // Send the *compiled* code to the worker
    worker.postMessage({
        type: 'execute',
        payload: { code: compiledCode }
    });
    
    return promise;
}


// This function is called by CodeMirror. It delegates to the worker.
workerhelper.getCompletions = function (context) {
    if (!workerReady) return Promise.resolve(null);
    
    const completionId = Date.now() + Math.random();
    const promise = new Promise((resolve) => {
        completionPromises.set(completionId, { resolve });
    });

    worker.postMessage({
        type: 'getCompletions',
        payload: {
            context,
            completionId: completionId,
        }
    });

    return promise;
};

workerhelper.getWorkerGlobals = function () {
    if (!workerReady) return Promise.resolve([]);
    const id = Date.now() + Math.random();
    const promise = new Promise((resolve) => {
        getGlobalsPromises.set(id, { resolve });
    });
    worker.postMessage({ type: 'getGlobals', payload: { id } });
    return promise;
};




workerhelper.readfile=function(type='text',encoding="utf-8"){
  return new Promise(async (resolve,rej) => {
    try {
      const [fileHandle] = await window.showOpenFilePicker();
      const selectedFile = await fileHandle.getFile();
      if (!selectedFile) {
        return rej(new Error('No file selected.'));
      }
      const name = selectedFile.name;
      const size = selectedFile.size;
      console.log("filename:" + name + ", size:" + size);

      const reader = new FileReader();
      reader.onload = function() {
        resolve([this.result, name, size]);
      };
      reader.onerror = function(event) {
        rej(new Error('File reading error: ' + event.target.error.message));
      };

      // Read file based on the specified type
      switch (type) {
        case 'text':
          reader.readAsText(selectedFile, encoding);
          break;
        case 'bin':
          reader.readAsArrayBuffer(selectedFile);
          break;
        case 'binstr':
          reader.readAsBinaryString(selectedFile);
          break;
        case 'dataurl':
          reader.readAsDataURL(selectedFile);
          break;
        default:
          rej(new Error('Unknown file type: ' + type));
          break;
      }
    } catch (e) {
      // Handle cases where the user cancels the file picker or other errors
      if (e.name === 'AbortError') {
        rej(new Error('File selection cancelled by user.'));
      } else {
        rej(e);
      }
    }
  });  
}

workerhelper.writefile = function(content, fileName) {
  return new Promise(async (resolve, reject) => {
    try {
      const pickerOpts = {};
      if (fileName) {
        pickerOpts.suggestedName = fileName;
      }
      
      const fileHandle = await window.showSaveFilePicker(pickerOpts);
      const writable = await fileHandle.createWritable();
      await writable.write(content);
      await writable.close();
      
      const file = await fileHandle.getFile();
      resolve(`File "${file.name}" of size ${file.size} saved successfully.`);
    } catch (e) {
      if (e.name === 'AbortError') {
        reject(new Error('File save cancelled by user.'));
      } else {
        reject(e);
      }
    }
  });
};

workerhelper.setVmSelectedResult = function(cnt){
  vm.selected.result = cnt;
}

workerhelper.setVmSelectedResultScript = function(cnt){
  vm.selected.resultScript = cnt;
}
