// Main-thread helper for Web Worker integration.
'use strict';

window.workerhelper = {};

// The real vm instance will be assigned to the global `vm` variable after the Vue app is mounted.

const worker = new Worker('worker.js');
let workerReady = false;
let executionPromise = null;
let completionPromises = new Map();
let docPromises = new Map();
let getGlobalsPromises = new Map();
let callbackPromises = new Map();

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
        case 'uiCallbackResult':
            {
                const promise = callbackPromises.get(payload.call_id);
                if (promise) {
                    if (payload.error) {
                      if(payload.error.notExist){
                        promise.reject(new NotExistError(payload.error.message));                        
                      }else{
                        promise.reject(new Error(payload.error.message));
                      }
                    } else {
                        promise.resolve(payload.result);
                    }
                    callbackPromises.delete(payload.call_id);
                }
                break;
            }
        case 'docResult':
            const docPromise = docPromises.get(payload.id);
            if (docPromise) {
                docPromise.resolve(payload.result);
                docPromises.delete(payload.id);
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
workerhelper.runcode = function (code, info, cell_uuid) {
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
        payload: { 
            code: compiledCode,
            cell_uuid: cell_uuid 
        }
    });
    
    return promise;
};

/**
 * Calls a function that was registered inside the worker.
 * @param {string} callbackId The UUID of the function to call.
 * @param {Array} args An array of arguments to pass to the function.
 * @returns {Promise<any>} A promise that resolves with the result of the function.
 */
workerhelper.callWorkerFunction = function(callbackId, args = []) {
    if (!workerReady) {
        return Promise.reject(new Error("Worker is not ready yet."));
    }
    const callId = `cb-call-${crypto.randomUUID()}`;
    const promise = new Promise((resolve, reject) => {
        callbackPromises.set(callId, { resolve, reject });
    });

    worker.postMessage({
        type: 'callUiCallback',
        payload: {
            callback_uuid: callbackId,
            args,
            call_id: callId
        }
    });

    return promise;
};



/**
 * Main-thread function to draw an interactive plot.
 * Invoked by the script generated from box.plotFunction.
 * Handles mixed dimensions by checking global maxNdim.
 * @param {object} plotOptions The specifications for the plot.
 */
workerhelper.drawPlot = function(plotOptions) {
  const { divId, callbackId, maxNdim, range, layout, config } = plotOptions;

  const plotDiv = document.getElementById(divId);
  if (!plotDiv) {
    console.error(`Plot container with id ${divId} not found.`);
    return;
  }

  // Basic layout adjustment
  const plotLayout = { autosize: true, ...layout };
  
  // Adjust margins for 3D plots to prevent cutting off labels
  if (maxNdim >= 2 && !plotLayout.margin) {
      plotLayout.margin = { l: 0, r: 0, b: 0, t: 0 };
  }

  const resetConfig = maxNdim === 1 ? {
    modeBarButtonsToRemove: ['resetScale2d'],
    modeBarButtonsToAdd: [{
      name: 'Reset',
      icon: Plotly.Icons.home,
      click: function(gd) {
        let newRange = [...range[0]];
        const update = {
          'xaxis.range': newRange,
          'yaxis.autorange': true
        };
        updatePlot([newRange]);
        Plotly.relayout(gd, update);
      }
    }]
  }:{};

  // Function to update the plot with new data from worker
  async function updatePlot(targetRange) {
    try {
      // Data received is now an array of results from multiple functions
      const dataArray = await workerhelper.callWorkerFunction(callbackId, [targetRange]);

      // Generate traces based on the specific dimension of each result vs global max dimension
      const traces = dataArray.map((data, index) => {
        // Case: Global view is 3D (Surface/Scatter3d), but this specific dataset is 1D
        if (maxNdim >= 2 && data.ndim === 1) {
          // Fill missing dimension (Z) with zeros to plot a line in 3D space
          const zArray = new Float32Array(data.x.length).fill(0);
          return {
            x: data.x,
            y: data.y,
            z: zArray,
            type: 'scatter3d',
            mode: 'lines',
            line: { width: 2 },
            name: data.name || `Trace ${index + 1}`
          };
        } 
        // Case: Standard 2D Surface
        else if (data.ndim === 2) {
          return {
            x: data.x,
            y: data.y,
            z: data.z,
            type: 'surface',
            name: data.name || `Trace ${index + 1}`,
            showscale: index === 0 // Only show scale for the first surface to reduce clutter
          };
        } 
        // Case: Standard 1D Line
        else if (data.ndim === 1) {
          return {
            x: data.x,
            y: data.y,
            type: "scattergl",
            mode: 'lines',
            line: { width: 1 },
            name: data.name || `Trace ${index + 1}`
          };
        }
        return null;
      }).filter(t => t !== null);

      await Plotly.react(plotDiv, traces, plotLayout, {...config, ...resetConfig});

    } catch (e) {
      if (e instanceof NotExistError) {
        plotDiv.innerText = 'Not ready.';
        return false;
      } else {
        console.error('Error updating plot:', e);
        plotDiv.innerText = 'Error updating plot: ' + e.message;
        return false;
      }
    }
    return true;
  }

  const relayoutHandler = async (eventData) => {
    let newRange;
    // Handle 1D zooming
    if (maxNdim === 1) {
      const xStart = eventData['xaxis.range[0]'];
      const xEnd = eventData['xaxis.range[1]'];
      if (xStart === undefined || xEnd === undefined) return;
      newRange = [[xStart, xEnd]];
    } 
    // Handle 2D/3D zooming (updates typically not passed efficiently in 3D, relies on initial range usually)
    else {
        // 3D plots in Plotly handle camera movement client-side, 
        // explicit range regeneration usually requires custom UI controls for 3D axes.
        return; 
    }
    
    updatePlot(newRange);
  };

  const resizeHandler = () => {
    if (document.getElementById(divId)) {
      Plotly.Plots.resize(divId);
    } else {
      resizeObserver.disconnect();
    }
  };
  const resizeObserver = new ResizeObserver(resizeHandler);
  resizeObserver.observe(plotDiv);

  async function initializePlot() {
    let ok = await updatePlot(range);
    
    // Attach event listeners mainly for 1D relayout optimization
    if (ok && maxNdim === 1) {
      if (plotDiv.on) {
        plotDiv.on('plotly_relayout', relayoutHandler);
      } else {
        console.error("plotDiv.on is not a function. Plotly might not have initialized correctly.");
      }
    }
  }
  initializePlot();
};



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

// This function is called by CodeMirror. It delegates to the worker.
workerhelper.getDoc = function (context) {
    if (!workerReady) return Promise.resolve(null);
    
    const id = Date.now() + Math.random();
    const promise = new Promise((resolve) => {
        docPromises.set(id, { resolve });
    });

    worker.postMessage({
        type: 'getDoc',
        payload: {
            context,
            id,
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


class NotExistError extends Error {
  constructor(message) {
    super(message);
    this.name = "NotExistError";
  }
}