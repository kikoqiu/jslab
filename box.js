var box=globalThis;

box.outputBuffer={'result':null,'resultScript':''};

// a map for callback_uuid -> { func, cell_uuid }
box.ui_callbacks = {}; 
// a map for cell_uuid -> [callback_uuid, ...]
box.cell_to_callbacks = {}; 

box.document = globalThis.linkedom.parseHTML('<html><body></body></html>');

box.initOutputBuffer=async function(){
  box.outputBuffer={'result':null,'resultScript':''};
  let {document} = globalThis.linkedom.parseHTML('<html><body></body></html>');
  globalThis.document=document;
}

/**
 * Register a function to be called from the UI.
 * @param {Function} func The function to register.
 * @param {string} cell_uuid The UUID of the cell this function is associated with.
 * @returns {string|null} The UUID for the registered callback, or null if no cell_uuid is provided.
 */
box.registerUiCallback = function(func, cell_uuid) {
    if (!cell_uuid || typeof func !== 'function') {
        console.error("Failed to register callback: cell_uuid and a valid function are required.");
        return null;
    }
    const callback_uuid = `cb-${crypto.randomUUID()}`;
    
    // Store the callback details
    box.ui_callbacks[callback_uuid] = { func: func, cell_uuid: cell_uuid };

    // Index for easy cleanup
    if (!box.cell_to_callbacks[cell_uuid]) {
        box.cell_to_callbacks[cell_uuid] = [];
    }
    box.cell_to_callbacks[cell_uuid].push(callback_uuid);

    return callback_uuid;
};

/**
 * Clears all registered UI callbacks for a specific cell.
 * @param {string} cell_uuid The UUID of the cell to clear callbacks for.
 */
box.clear_cell_callbacks = function(cell_uuid) {
    if (cell_uuid && box.cell_to_callbacks[cell_uuid]) {
        const callback_uuids = box.cell_to_callbacks[cell_uuid];
        for (const callback_uuid of callback_uuids) {
            delete box.ui_callbacks[callback_uuid];
        }
        delete box.cell_to_callbacks[cell_uuid];
    }
};

/**
 * flush the html output
 */
box.flushHTML=async function(){
  //let jsonContent=globalThis.linkedom.toJSON(globalThis.document.body);
  let html=globalThis.document.body.innerHTML;
  if(html!==box.outputBuffer.result){
    box.outputBuffer.result=html;
    //let jsonContent=JSON.stringify(html);
    //await globalEval(`vm.selected.result = ${jsonContent};`);
    await workerhelperCall('setVmSelectedResult',html);
  }
}

/**
 * flush the output
 */
box.flushOutputBuffer=async function(){
  await box.flushHTML();
  await workerhelperCall('setVmSelectedResultScript',box.outputBuffer.resultScript);
  //await globalEval(`vm.selected.resultScript = ${JSON.stringify(box.outputBuffer.resultScript)};`);
}


box.runtimeEnter=async function(){
  await box.initOutputBuffer();
}

box.runtimeExit=async function(){
  await box.stopAnimation();
  await box.flushOutputBuffer();
}

/**
 * async delay microseconds
 * @param {number} durationMs 
 * @returns 
 */
box.delay = function(durationMs) {
      return new Promise(resolve => setTimeout(resolve, durationMs));
};

/**
 * Range generator
 * @param {*} start
 * @param {*} end (exclusive)
 * @param {*} step step
 * @param {*} mapper mapper function eg. i=>sin(i)
 */
box.rangen=function *(start,end,step=1,mapper=undefined){
  if(end >= start){
    for(let i=start;i<end;i=i+step){
      if(!!mapper){
        yield mapper(i);
      }else{
        yield i;
      }
    }
  }else{
    for(let i=start;i>end;i=i+step){
      if(!!mapper){
        yield mapper(i);
      }else{
        yield i;
      }
    }
  }

};
/**
 * Range array
 * @param {*} start
 * @param {*} end (exclusive)
 * @param {*} step step
 * @param {*} mapper mapper function eg. i=>sin(i)
 * @returns 
 */
box.range=function(start,end,step=1,mapper=undefined){
  return [...box.rangen(start,end,step,mapper)];
}
/**
 * echo output
 * @param  {...any} o output 
 */
box.echo=function(...o){
  let str='';
  for(var i of o){
    if(str!='')str+=', ';
    str+=String(i);
  }
  str=str.replace(/</ig,'&lt;').replace(/>/ig,'&gt;')+'\n';
  str=`<pre><code>${str}</code></pre>`;
  globalThis.document.body.append(document.createRange().createContextualFragment(str));
};

/**
 * echo HTML to output
 * @param  {...any} o output 
 */
box.echoHTML=function(...o){
  let str='';
  for(var i of o){
    str+=String(i);
  }
  globalThis.document.body.append(document.createRange().createContextualFragment(str));
};

/**
 * JSON highlight
 * @param {*} json JSON object or string
 * @returns {string} highlighted HTML string
 */
box.jsonHighlight=function(json) {
  if (typeof json != 'string') {
    const replacer=(key, value) => {
      if (ArrayBuffer.isView(value) && !(value instanceof DataView)) {
        return Array.from(value);
      }
      return value;
    }
    json = JSON.stringify(json, replacer, 2);
  }
  json = json.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
  return json.replace(/("(\\u[a-zA-Z0-9]{4}|\\[^u]|[^\\"])*"(\s*:)?|\b(true|false|null)\b|-?\d+(?:\.\d*)?(?:[eE][+\-]?\d+)?)/g, function (match) {
    var cls = 'json-number';
    if (/^"/.test(match)) {
        if (/:$/.test(match)) {
            cls = 'json-key';
        } else {
            cls = 'json-string';
        }
    } else if (/true|false/.test(match)) {
        cls = 'json-boolean';
    } else if (/null/.test(match)) {
        cls = 'json-null';
    }
    return '<span class="' + cls + '">' + match + '</span>';
  });
}

/**
 * dump object to JSON to output
 * @param  {...any} o output 
 */
box.dumpJSON=function(...o){
  let str='';
  for(var i of o){
    str+=`<pre><code>${box.jsonHighlight(i)}</code></pre>`;
  }
  globalThis.document.body.append(document.createRange().createContextualFragment(str));
};

function _convertToArrayProps(arr) {
  if (!Array.isArray(arr)) return arr;

  return arr.map(item => {
    if (typeof item !== 'object' || item === null) {
      return item;
    }

    const newItem = { ...item };

    Object.keys(newItem).forEach(key => {
      const value = newItem[key];

      if (value && typeof value.toArray === 'function') {
        newItem[key] = value.toArray();
      }
    });

    return newItem;
  });
}

/**
 * Plot with plotly.js. General plot function. Parameters are passed to plotly.
 * @param {*} data data arrays
 * @param {*} layout the plotly layout
 * @param {*} config the plotly config
 * @param {*} style Extra plot html div css style
 * @param {*} frames the plotly frames
 * @returns the plotly div id
 */
box.plotly=function(data, layout, config, style, frames){
  if(!style){
    style=''
  }
  data = _convertToArrayProps(data);
  let obj={config:config,layout:{autosize: true,...layout},frames:frames,data:data};
  let json=JSON.stringify(obj); 

  var div='div-'+crypto.randomUUID();
  let node=document.createRange().createContextualFragment(`<div class="plot" id="${div}" style="${style};resize:both;overflow:auto;"></div>`);
  globalThis.document.body.append(node);

  let scr=
`Plotly.react("${div}", ${json});
(function(){
const ob=new ResizeObserver(entries => {
  if(document.getElementById('${div}')){
    Plotly.Plots.resize("${div}");
  }else{
    ob.disconnect();
  }
});
ob.observe(document.getElementById('${div}'));
})();
`;//newPlot
  box.outputBuffer.resultScript += scr;
  return div;
};


/**
 * Generates data for one or more functions and sets up the plot environment.
 * Supports single function, array of functions, or array of configuration objects.
 *
 * @param {Function|Function[]|Object[]} plotSpec - The function(s) or configuration object(s) to plot.
 * @param {Object} layout - Plotly layout configuration.
 * @param {Object} config - Plotly config options.
 * @param {String} style - CSS style for the container.
 */
box.plotFunction = function(plotSpec, layout, config, style) {
  // Normalize input to an array of specification objects
  let specs = [];
  if (typeof plotSpec === "function") {
    specs = [{ func: plotSpec }];
  } else if (Array.isArray(plotSpec)) {
    specs = plotSpec.map(s => (typeof s === "function" ? { func: s } : s));
  } else {
    specs = [plotSpec];
  }

  // Determine the maximum dimension among all datasets to decide the global plot type
  let maxNdim = 1;
  
  // Process each spec to apply defaults
  const normalizedSpecs = specs.map(spec => {
    if (typeof spec.func !== 'function') {
      box.echo("Error: plotSpec.func must be a function.");
      return null;
    }

    const ndim = spec.ndim || 1;
    if (ndim > maxNdim) maxNdim = ndim;

    // Default ranges and samples based on dimension
    const defaultRanges = [[-10, 10], [-10, 10], [-10, 10]];
    let defaultSamples;
    if (ndim == 1) {
      defaultSamples = [4000];
    } else if (ndim == 2) {
      defaultSamples = [500, 500];
    } else {
      defaultSamples = [100, 100, 50];
    }

    const range = [];
    const samples = [];
    const userRange = spec.range;
    const userSamples = spec.samples;

    for (let i = 0; i < ndim; i++) {
      if (userRange && userRange[i] && Array.isArray(userRange[i]) && userRange[i].length === 2) {
        range.push(userRange[i]);
      } else {
        range.push(defaultRanges[i]);
      }

      if (userSamples && typeof userSamples[i] === 'number') {
        samples.push(userSamples[i]);
      } else {
        samples.push(defaultSamples[i]);
      }
    }

    return {
      ...spec,
      ndim,
      range,
      samples,
      vectorized: spec.vectorized || false,
      supersample: spec.supersample || 8
    };
  }).filter(s => s !== null);

  if (normalizedSpecs.length === 0) return;

  /**
   * Internal helper to calculate data for a single specification.
   * reused from the original logic but encapsulated.
   */
  const calculateSingleSpec = (spec, currentRange) => {
    const { func, ndim, samples, vectorized, supersample, name } = spec;

    // Use currentRange if available (from zoom), otherwise use spec default.
    // Ensure we only take dimensions relevant to this spec.
    const activeRange = (currentRange && currentRange.length >= ndim) 
      ? currentRange.slice(0, ndim) 
      : spec.range;

    const ssFactor = (ndim === 1 && supersample > 1) ? Math.floor(supersample) : 1;
    const isSupersampling = ssFactor > 1;

    const nx = (ndim === 1 && isSupersampling) ? Math.floor(samples[0] * ssFactor) : samples[0];
    const ny = (ndim >= 2) ? samples[1] : 1;
    const nz = (ndim >= 3) ? samples[2] : 1;
    const totalPoints = nx * ny * nz;

    const fillLinspace = (outArray, start, end, n) => {
      if (n <= 1) {
        outArray[0] = start;
        return;
      }
      const step = (end - start) / (n - 1);
      for (let i = 0; i < n; i++) {
        outArray[i] = start + i * step;
      }
    };

    const xAxis = new Float64Array(nx);
    fillLinspace(xAxis, activeRange[0][0], activeRange[0][1], nx);

    let yAxis, zAxis;
    if (ndim >= 2) {
      yAxis = new Float64Array(ny);
      fillLinspace(yAxis, activeRange[1][0], activeRange[1][1], ny);
    }
    if (ndim >= 3) {
      zAxis = new Float64Array(nz);
      fillLinspace(zAxis, activeRange[2][0], activeRange[2][1], nz);
    }

    let resultFlat;

    if (vectorized) {
      const xMesh = new Float64Array(totalPoints);
      let yMesh, zMesh;
      if (ndim >= 2) yMesh = new Float64Array(totalPoints);
      if (ndim >= 3) zMesh = new Float64Array(totalPoints);

      let ptr = 0;
      for (let k = 0; k < nz; k++) {
        const zVal = (ndim >= 3) ? zAxis[k] : 0;
        for (let j = 0; j < ny; j++) {
          const yVal = (ndim >= 2) ? yAxis[j] : 0;
          xMesh.set(xAxis, ptr);
          if (ndim >= 2) {
            for (let i = 0; i < nx; i++) {
              yMesh[ptr + i] = yVal;
              if (ndim >= 3) zMesh[ptr + i] = zVal;
            }
          }
          ptr += nx;
        }
      }

      if (ndim === 1) resultFlat = func(xMesh);
      else if (ndim === 2) resultFlat = func(xMesh, yMesh);
      else resultFlat = func(xMesh, yMesh, zMesh);

      if (!(resultFlat instanceof Float64Array)) {
        resultFlat = new Float64Array(resultFlat);
      }
    } else {
      resultFlat = new Float64Array(totalPoints);
      let ptr = 0;
      if (ndim === 1) {
        for (let i = 0; i < nx; i++) {
          resultFlat[i] = func(xAxis[i]);
        }
      } else if (ndim === 2) {
        for (let j = 0; j < ny; j++) {
          const yVal = yAxis[j];
          for (let i = 0; i < nx; i++) {
            resultFlat[ptr++] = func(xAxis[i], yVal);
          }
        }
      } else {
        for (let k = 0; k < nz; k++) {
          const zVal = zAxis[k];
          for (let j = 0; j < ny; j++) {
            const yVal = yAxis[j];
            for (let i = 0; i < nx; i++) {
              resultFlat[ptr++] = func(xAxis[i], yVal, zVal);
            }
          }
        }
      }
    }

    // 1D Supersampling Decimation
    if (ndim === 1 && isSupersampling) {
      const outputCount = samples[0];
      const finalCount = outputCount * 2;
      const finalX = new Float64Array(finalCount);
      const finalY = new Float64Array(finalCount);
      const yLen = resultFlat.length;
      let outPtr = 0;

      for (let i = 0; i < outputCount; i++) {
        const startIdx = i * ssFactor;
        let endIdx = startIdx + ssFactor;
        if (endIdx > yLen) endIdx = yLen;
        if (startIdx >= endIdx) break;

        let minVal = Infinity;
        let maxVal = -Infinity;
        for (let k = startIdx; k < endIdx; k++) {
          const val = resultFlat[k];
          if (val < minVal) minVal = val;
          if (val > maxVal) maxVal = val;
        }
        if (minVal === Infinity) { minVal = NaN; maxVal = NaN; }

        const firstVal = resultFlat[startIdx];
        const lastVal = resultFlat[endIdx - 1];
        finalX[outPtr] = xAxis[startIdx];
        finalX[outPtr + 1] = xAxis[endIdx - 1];

        if (firstVal > lastVal) {
          finalY[outPtr] = maxVal;
          finalY[outPtr + 1] = minVal;
        } else {
          finalY[outPtr] = minVal;
          finalY[outPtr + 1] = maxVal;
        }
        outPtr += 2;
      }
      return { ndim, x: finalX, y: finalY, name };
    }

    // Structure Output
    if (ndim === 1) {
      return { ndim, x: xAxis, y: resultFlat, name};
    } else if (ndim === 2) {
      const zRows = new Array(ny);
      for (let j = 0; j < ny; j++) {
        const start = j * nx;
        zRows[j] = resultFlat.subarray(start, start + nx);
      }
      return { ndim, x: xAxis, y: yAxis, z: zRows, name };
    } else if (ndim === 3) {
      const valueSlices = new Array(nz);
      for (let k = 0; k < nz; k++) {
        const sliceRows = new Array(ny);
        const sliceOffset = k * ny * nx;
        for (let j = 0; j < ny; j++) {
          const rowStart = sliceOffset + (j * nx);
          sliceRows[j] = resultFlat.subarray(rowStart, rowStart + nx);
        }
        valueSlices[k] = sliceRows;
      }
      return { ndim, x: xAxis, y: yAxis, z: zAxis, value: valueSlices, name };
    }
  };

  /**
   * Main data generator called by the UI.
   * Iterates through all normalized specs and returns an array of results.
   */
  const dataGenerator = (newRange) => {
    return normalizedSpecs.map(spec => calculateSingleSpec(spec, newRange));
  };

  const callbackId = box.registerUiCallback(dataGenerator, box.cell_uuid);
  if (!callbackId) {
    box.echo("Error: Could not register plot callback. Ensure code is run inside a cell.");
    return;
  }

  const divId = 'plot-' + crypto.randomUUID();
  box.echoHTML(`<div class="plot" id="${divId}" style="${style || ''}; resize:both; overflow:auto;"></div>`);

  // Construct initial combined range based on maxNdim for the first view
  // Use the range of the first spec that matches maxNdim, or default
  const initialRange = normalizedSpecs.find(s => s.ndim === maxNdim)?.range || [[-10, 10]];

  const plotOptions = {
    divId,
    callbackId,
    maxNdim, // Send the largest dimension found
    range: initialRange,
    layout,
    config
  };

  const script = `
      if (typeof workerhelper.drawPlot === 'function') {
          workerhelper.drawPlot(${JSON.stringify(plotOptions)});
      } else {
          const el = document.getElementById(${JSON.stringify(divId)});
          if(el) el.innerText = 'Error: workerhelper.drawPlot is not defined.';
      }
  `;
  box.outputBuffer.resultScript += script;
};


/**
 * Plots an implicit function (e.g., x^2 + y^2 = 1) on the UI.
 * The actual computation happens in the worker, and rendering happens on the main thread.
 *
 * @param {string | Function | object | Array<string|Function|object>} plotSpec - The specification(s) for the implicit function(s).
 *   - If a `string`: An implicit equation like "x\*\*2 + y\*\*2 = 1".
 *   - If a `Function`: A scalar function `f(x, y)` which returns a numerical value. The plot will show where `f(x, y) = 0`.
 *   - If an `object`: { func, vectorized, isEquality }.
 *   - vectorized If `true`, `func` is expected to take `ndarray.NDArray` inputs for `x(spape=[1,w])` and `y(spape=[h,1])` and return a `ndarray.NDArray` of results, e.g. (x**2-y**2).sin()-(x+y).sin()-(x*y).cos() . If `false`, `func` takes scalar `x, y` and is called for each point.
 *   - isEquality If `true`, plots `f(x,y)=0`. If `false`, plots `f(x,y)<0` (an inequality region).
 *   - If an `Array`: A list containing any combination of the above.
 * @param {object} [options={}] - Configuration options for the plot.
 * @param {string[]} [options.colors] - Array of color strings corresponding to inputs.
 * @param {string[]} [options.names] - Array of name strings for the legend/tooltip.
 * @param {string} [options.style='height:600px; resize:both; overflow:auto;'] - CSS style for the plot container.
 * @param {object} [options.bounds={xMin:-10, xMax:10, yMin:-10, yMax:10}] - Initial mathematical bounds.
 * @param {number} [options.superSample=1] - Factor for supersampling.
 * @param {string} [options.gridColor='#e0e0e0'] - Color for grid lines.
 * @param {string} [options.axisColor='#444'] - Color for axis lines and labels.
 */
box.plotImplicit = function(plotSpec, options) {
  options = options ?? {};
  
  // Normalize input to an array to handle single or multiple plots uniformly
  const inputList = Array.isArray(plotSpec) ? plotSpec : [plotSpec];
  const specsForUI = [];

  // Step 1: Process all inputs
  for (const item of inputList) {
      if (typeof item === 'string') {
          specsForUI.push({ type: 'string', value: item });
      } else {
          let func, vectorized = false, isEquality = true;

          if (typeof item === 'function') {
              func = item;
          } else if (typeof item === 'object' && item !== null) {
              func = item.func;
              vectorized = item.vectorized || false;
              isEquality = item.isEquality !== undefined ? item.isEquality : true;
          }

          if (typeof func !== 'function') {
              box.echo("Error: plotSpec item must be a string, a function, or an object with a 'func' property.");
              return;
          }

          const dataGenerator = (w, h, xArr, yArr) => {          
              if (vectorized) {
                  const result = func(new ndarray.NDArray(xArr.slice(0, w),{shape:[1,w]}), new ndarray.NDArray(yArr.slice(0, h),{shape:[h,1]}));
                  return result.data;
              } else {
                  const outArr = new Float32Array(h*w);
                  for (let j = 0; j < h; j++) {
                    for(let i = 0; i < w; i++){
                      outArr[j * w + i] = func(xArr[i], yArr[j]);
                    }
                  }
                  return outArr;
              }
          };

          const callbackId = box.registerUiCallback(dataGenerator, box.cell_uuid);
          if (!callbackId) {
              box.echo("Error: Could not register plot callback. Ensure code is run inside a cell.");
              return;
          }

          specsForUI.push({ type: 'callback', callbackId, isEquality });
      }
  }

  // Step 2: Generate resultScript
  const divId = 'plot-' + crypto.randomUUID();
  const style = options.style || 'resize:both; overflow:auto;';
  box.echoHTML(`<div class="plot" id="${divId}" style="${style}"></div>`);

  const plotOptions = {
      divId,
      specs: specsForUI, // Pass array of specs
      ...options 
  };

  const script = `
      if (typeof workerhelper.plotImplicit === 'function') {
          workerhelper.plotImplicit(${JSON.stringify(plotOptions)});
      } else {
          const el = document.getElementById(${JSON.stringify(divId)});
          if(el) el.innerText = 'Error: workerhelper.plotImplicit is not defined.';
      }
  `;
  box.outputBuffer.resultScript += script;
};




/**
 * add animation to a plotly div
 * @param {*} div the plotly div id
 * @param {*} frameOrGroupNameOrFrameList frame name or group name or list of frames
 * @param {*} animationAttributes animation attributes , eg. {transition: {duration: 500, easing: 'linear-in-out'}, frame: {duration: 500} }
 */
box.plotlyAnimate=function(div, frameOrGroupNameOrFrameList, animationAttributes){
  let script=`Plotly.animate(${JSON.stringify(div)}, ${JSON.stringify(frameOrGroupNameOrFrameList)}, ${JSON.stringify(animationAttributes)});\n`;  
  box.outputBuffer.resultScript += script;
};


/**
 * Unpace array of objects to get array of properties
 * @param {Array} data 
 */
box.unpack=function(data,...props){
  if(props.length==1)return data.map(o=>o[props[0]]);
  let ret=[];
  for(let i=0;i<props.length;++i){
    let p=props[i];
    ret.push(data.map(o=>o[p]));
  }
  return ret;
};


/**
 * Plot a line figure
 * @param  {...any} xPoints_yPoints x-points,y-points,[x-point,y-point...],[{labels,style,layout,config}] 
 * @returns the plotly div id
 */
box.plot=function(...xPoints_yPoints){
  let args=xPoints_yPoints;
  let style=null;  
  let data=[];
  let layout=null;
  let config=null;
  if(args.length==1){
    let x=range(0,args[0].length??args[0].size);
    let y=args[0];
    data.push({x,y,type: 'scatter',mode:'lines',line:{width:1}});
  }else{
    for(var pos=0;pos<args.length;pos+=2){
      let x=args[pos];
      let y=args[pos+1];
      data.push({x,y,type: 'scatter',mode:'lines',line:{width:1}});
    }
    pos-=2;
    if(pos==args.length-1){
      if(typeof(args[pos])=='string'){
        style=args[pos];
      }else if(typeof(args[pos])=='object'){
        let dic=args[pos];
        style=dic['style'];
        let lbls=dic['labels']?dic['labels']:dic['names'];
        if(lbls){
          for(let i =0;i<data.length;++i){
            data[i].name=lbls[i];
          }
        }        
        layout=dic['layout'];
        config=dic['config'];
      }
      
    }
  }
  return box.plotly(data,layout,config,style);
};

/**
 * Plot a 3d line figure
 * @param  {...any} xPoints_yPoints_zPoints x-points,y-points,z-points,[x-point,y-point,z-points...],[{labels,style,layout,config}] 
 * @returns the plotly div id
 */
box.plot3d=function(...xPoints_yPoints_zPoints){
  let args=xPoints_yPoints_zPoints;
  let style=null;  
  let data=[];
  let layout=null;
  let config=null;
  
  for(var pos=0;pos<args.length;pos+=3){
    let x=args[pos];
    let y=args[pos+1];
    let z=args[pos+2];
    data.push({x,y,z,type: 'scatter3d',mode:'lines'});
  }
  pos-=3;
  if(pos==args.length-1){
    if(typeof(args[pos])=='string'){
      style=args[pos];
    }else if(typeof(args[pos])=='object'){
      let dic=args[pos];
      style=dic['style'];
      let lbls=dic['labels']?dic['labels']:dic['names'];
      if(lbls){
        for(let i =0;i<data.length;++i){
          data[i].name=lbls[i];
        }
      }
      layout=dic['layout'];
      config=dic['config'];
    }    
  }
  return box.plotly(data,layout,config,style || 'height:600px;');
};




/**
 * Read a file
 * @param {String} type text bin binstr dataurl
 * @param {String} encoding encoding for reading text file
 * @returns a promise with [content,filename,size]
 */
box.readfile=async function(type='text',encoding="utf-8"){
  return await workerhelperCall('readfile',type,encoding);
}

/**
 * 
 * @param {*} content 
 * @param {String} fileName default filename
 * @returns a promise that resolves when file is written
 */
box.writefile = async function(content, fileName) {
  return await workerhelperCall('writefile', content, fileName);
};

/**
 * 
 * @param {String} encoding 
 * @returns d3 csv parsed object
 */
box.readCsv=async function(encoding="utf-8"){
  let filecnt=await box.readfile(type='text',encoding)
  return d3.csvParse(filecnt[0])
};

/**
 * Compile a mathjs expression
 * @param {string} e The Expression
 * @returns node with eval(scope) function
 */
box.compile_expr=box.expr=function(e){
  let node=math.parse(e);
  let code=node.compile();
  code.eval=(s)=>{
    if(!s){
      s={};
      node.filter(function (n,path,parent) {
        return n.isSymbolNode && path!='fn';
      }).forEach(
        n=>{
          s[n.name]=box[n.name];
          }
        );
    }
    return code.evaluate(s);
  }
  return code;
};
/**
 * Evaluate a mathjs expression
 * @param {string} e The Expression
 * @param {Object} scope the evalutae scope
 * @returns 
 */
box.eval_expr=function(e,scope){
  return box.compile_expr(e).eval(scope);
}
/**
 * Calc derivative of a mathjs expr
 */
box.deriv=math.derivative;
box.symplify=math.symplify;
box.mathfrac=math.create({number:'Fraction'},math.all)
box.mathbn=math.create({number:'BigNumber'},math.all)
/**
 * Get latex string
 * @param  {...any} ex 
 */
box.latexstr=function(...ex){
  let result='';
  for(let data of ex){
    if(typeof(data)=='object' && data.toTex){
      data=data.toTex({
        parenthesis: 'auto',    // parenthesis option
        //handler: someHandler,   // handler to change the output
        implicit: 'hide'        // how to treat implicit multiplication
      });
    }else if(Array.isArray(data)){
      data=data.map(d=>box.latexstr(d));
      data.join(',');
      data='['+data+']';
    }
      
    result+=data;
  }
  return result;
}

box.latex_style=''
/**
 * Display a latex expressoin
 * @param {*} data the latex expression (string, mathjs expression or sympy sym)
 * @param {*} style extra style
 */
box.latex=function(...ex){
  let result=box.latexstr(...ex);
  let style=box.latex_style;
  var div='div-'+crypto.randomUUID();
  if(false){
    //let node=document.createRange().createContextualFragment('<div class="latex" id="'+div+'" style="'+style+'">'+MathJax.tex2svg(result, {em: 16, ex: 6, display: false}).outerHTML+'</div>');
    //globalThis.document.body.append(node);
  }else{
    let json=JSON.stringify(result);
    let node=document.createRange().createContextualFragment('<div class="latex" id="'+div+'" style="'+style+'"></div>')
    globalThis.document.body.append(node);
    let scr='document.getElementById("'+div+'").appendChild(MathJax.tex2svg('+json+', {em: 16, ex: 6, display: true}));'
    box.outputBuffer.resultScript += scr;
  }
};

/**
 * @typedef {Object} SolverAPI
 * @property {function(function(Array): boolean): SolverAPI} where - Adds a constraint filter.
 * @property {function(number=): Generator<Array, void, unknown>} solve - Executes the permutation algorithm.
 */

/**
 * Creates a high-performance solver for Partial (P(n,k)) and Full Permutations (N!).
 * Supports constraints with automatic pruning via Proxy.
 *
 * @param {Array} sourceArray - The pool of elements (e.g., [1, 2, 3, 4]).
 * @param {number} [k] - The length of the permutation (P(n, k)). Defaults to n.
 * @returns {SolverAPI} The solver interface.
 */
 box.permutationSolver = function(sourceArray, k) {
    // Default to full permutation if k is undefined
    const targetLength = (k === undefined || k === null) ? sourceArray.length : k;
    
    // Validation
    if (targetLength < 0 || targetLength > sourceArray.length) {
        throw new Error(`Invalid k: ${k}. Must be between 0 and ${sourceArray.length}`);
    }

    // Internal buffer (copy of source). 
    // We permute this array in-place to avoid memory allocation during recursion.
    const arr = [...sourceArray];
    const rules = [];

    // --- State Management ---
    // 'cursor' tracks the boundary of "valid/fixed" data.
    // Indices < cursor are fixed. Indices >= cursor are technically "future/dirty".
    let cursor = 0;

    // Flag to detect if a rule tried to look ahead at future data.
    let isRuleFullyEvaluated = true;

    // --- The Proxy ---
    // Allows us to run user rules safely. 
    // It detects if a rule depends on data that hasn't been generated yet.
    const proxy = new Proxy(arr, {
        get(target, prop, receiver) {
            if (prop === 'length') {
                return cursor + 1;
            }
            const index = Number(prop);

            // Pass through standard properties (.length, etc.)
            if (isNaN(index)) {
                return Reflect.get(target, prop, receiver);
            }

            // Detect Future Access
            // If we are at depth 2 (cursor=2), data at index 2 is currently being decided, 
            // but data at index 3 is completely unknown (dirty).
            if (index > cursor) {
                isRuleFullyEvaluated = false;
                // Return the dirty value anyway to prevent math errors (e.g., NaN, Infinity).
                // The solver will ignore the result of this rule later.
            }

            return target[index];
        }
    });

    /**
     * Checks all constraints.
     * @returns {boolean} False if we should prune (stop) this branch.
     */
    const checkConstraints = () => {
        for (const rule of rules) {
            isRuleFullyEvaluated = true;
            
            // Execute rule against the proxy
            const result = rule(proxy);

            // 1. If the rule accessed future data (index > cursor), we can't trust the result yet.
            //    Strategy: Optimistic Pass. Assume it's true and wait for deeper recursion.
            if (!isRuleFullyEvaluated) {
                continue;
            }

            // 2. If data was fully available and the rule returned false, Prune immediately.
            if (result === false) {
                return false;
            }
        }
        return true;
    };

    /**
     * Recursive Backtracker
     * @param {number} start - Current index to fill.
     * @param {number} k - Target length of the permutation.
     */
    function* permute(start, k) {
        // Base Case: We have filled 'k' slots.
        if (start === k) {
            yield arr.slice(0, k); // Yield only the valid part
            return;
        }

        const length = arr.length;

        for (let i = start; i < length; i++) {
            // 1. Swap: Choose element at 'i' for the current slot 'start'
            [arr[start], arr[i]] = [arr[i], arr[start]];

            // 2. Update Context: Data at 'start' is now fixed.
            cursor = start;

            // 3. Pruning: Check rules before going deeper
            if (checkConstraints()) {
                yield* permute(start + 1, k);
            }

            // 4. Backtrack: Restore array
            [arr[start], arr[i]] = [arr[i], arr[start]];
        }
    }

    // --- Public API ---
    return {
        where(predicate) {
            rules.push(predicate);
            return this;
        },

        /**
         * Generates the permutations.
         */
        solve() {
            return permute(0, targetLength);
        }
    };
}


/**
 * Generates permutations of length `k` from the source array.
 * 
 * @template T
 * @param {T[]} sourceArray - The pool of elements.
 * @param {number} [k] - The number of elements to select. Defaults to array length.
 * @returns {Generator<T[], void, unknown>}
 */
box.permute = function* (sourceArray, k) {
    // Create a local copy to protect the original array
    const arr = [...sourceArray];
    const n = arr.length;
    
    // Default to Full Permutation if k is not provided
    const targetLength = (k === undefined || k === null) ? n : k;

    // Safety check
    if (targetLength < 0 || targetLength > n) return;

    /**
     * Internal recursive worker.
     * @param {number} index - Current position to fill.
     */
    function* backtrack(index) {
        // Base Case: We have filled 'k' positions
        if (index === targetLength) {
            yield arr.slice(0, targetLength);
            return;
        }

        for (let i = index; i < n; i++) {
            // 1. Swap
            [arr[index], arr[i]] = [arr[i], arr[index]];

            // 2. Recurse
            yield* backtrack(index + 1);

            // 3. Backtrack (Restore)
            [arr[index], arr[i]] = [arr[i], arr[index]];
        }
    }

    yield* backtrack(0);
}



/**
 * Generates all combinations of `k` elements from the source array.
 * Order implies no distinction: [A, B] is considered the same as [B, A].
 *
 * Algorithm: Recursive Backtracking with Index Pruning.
 * Time Complexity: O(C(n, k))
 * Space Complexity: O(k) (recursion stack + buffer)
 *
 * @template T
 * @param {T[]} sourceArray - The pool of elements to choose from.
 * @param {number} k - The number of elements to select.
 * @returns {Generator<T[], void, unknown>}
 */
box.combinations = function* (sourceArray, k) {
    const n = sourceArray.length;

    // Boundary check
    if (k < 0 || k > n) return;

    // Pre-allocate a buffer to store the current combination.
    // This avoids pushing/popping dynamic arrays.
    const buffer = new Array(k);

    /**
     * Internal recursive worker.
     * @param {number} start - The index in sourceArray to start picking from.
     * @param {number} depth - The current count of items selected (index in buffer).
     */
    function* backtrack(start, depth) {
        // Base Case: If the buffer is full (depth == k), yield the result.
        if (depth === k) {
            // Must yield a copy (Array.from or .slice) because 'buffer' is reused.
            yield Array.from(buffer);
            return;
        }

        // Optimization: Calculate the furthest index we can go to.
        // We need (k - depth) more items.
        // If sourceArray has 'n' items, we must leave enough room for the remaining slots.
        // Example: n=5, k=3, depth=0. We need 3 items. Max index we can pick is 2 (0,1,2).
        // If we picked index 3, we'd only have index 4 left (1 item), but we need 2 more.
        const limit = n - (k - depth) + 1;

        for (let i = start; i < limit; i++) {
            // 1. Pick current element
            buffer[depth] = sourceArray[i];

            // 2. Recurse: Move to next element (i+1) and next depth level
            yield* backtrack(i + 1, depth + 1);
            
            // 3. Backtrack: No explicit code needed here as we overwrite buffer[depth] 
            //    in the next iteration of this loop.
        }
    }

    yield* backtrack(0, 0);
}
/**
 * load SheetJS library
 * @returns 
 */
box.loadSheetJS=async function(){
  if(self.XLSX)return;
  await importScripts('3pty/xlsx.full.min.js');
}

/**
 * Open an Excel file and return the workbook object
 * @returns workbook object
 */
box.openExcel=async function(){
  await box.loadSheetJS();
  let data=await box.readfile(type='bin');
  let workbook=XLSX.read(data[0],{type:'array'});
  return workbook;
}

/**
 * Write an Excel workbook object to file 
 * @param {Object} workbook 
 * @param {String} fileName default 'output.xlsx'
 */
box.saveExcel=async function(workbook, fileName='output.xlsx'){
  await box.loadSheetJS();
  let wbout=XLSX.write(workbook,{bookType:'xlsx',type:'array'});
  await box.writefile(wbout,fileName);
}

/**
 * Read an Excel file and parse the first sheet
 * @param {String} resultType 'json', 'array'
 * @param {number} sheetIndex Sheet index to read, default 0
 * @returns JSON array
 */
box.readExcel=async function(resultType='json', sheetIndex=0){
  await box.loadSheetJS();
  let workbook=await box.openExcel();
  let firstSheetName=workbook.SheetNames[sheetIndex];
  let worksheet=workbook.Sheets[firstSheetName];
  if(resultType=='array'){
    let arr=XLSX.utils.sheet_to_json(worksheet,{header:1,defval:null});
    return arr;
  }else{
    let json=XLSX.utils.sheet_to_json(worksheet,{defval:null});
    return json;
  }
}

/**
 * Write JSON array to an Excel file
 * @param {Array} data 
 * @param {String} type 'json', 'array'
 * @param {String} fileName default 'output.xlsx'
 * @param {String} sheetName default 'Sheet1'
 */
box.writeExcel=async function(data,type='json', fileName='output.xlsx',sheetName='Sheet1'){
  await box.loadSheetJS();
  let worksheet;
  if(type=='array'){
    worksheet=XLSX.utils.aoa_to_sheet(data);
  }else{
    worksheet=XLSX.utils.json_to_sheet(data);
  }
  var wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, worksheet, sheetName);
  await box.saveExcel(wb,fileName);
}



box.renderRafId = null;

box.startAnimation=function() {
  if (box.renderRafId) return;

  function loop() {
    box.flushHTML();
    box.renderRafId = self.requestAnimationFrame(loop);
  }

  box.renderRafId = self.requestAnimationFrame(loop);
};

box.stopAnimation=function() {
  if (box.renderRafId) {
    self.cancelAnimationFrame(box.renderRafId);
    box.renderRafId = null;
  }
};


/**
 * Load the d3 library with linkedom polyfill
 * @param {boolean} enableAnimation enable Transition animition
 */
box.loadD3=async function(enableAnimation=false){
  if(!globalThis.d3){
    await importScripts('./3pty/d3@7.js');
  }
  if(enableAnimation){
    box.startAnimation();
  }
}


/**
 * Load the NDArray library
 */
box.loadNDArray=async function(){
  if(!globalThis.ndarray){
    await importScripts("3pty/ndarray.browser.js");
    await ndarray.NDWasm.init('3pty/');

    ndarray.NDArray.prototype.operatorAdd=ndarray.NDArray.prototype.add;
    ndarray.NDArray.prototype.operatorSub=ndarray.NDArray.prototype.sub;
    ndarray.NDArray.prototype.operatorMul=ndarray.NDArray.prototype.mul;
    ndarray.NDArray.prototype.operatorDiv=ndarray.NDArray.prototype.div;
    ndarray.NDArray.prototype.operatorPow=ndarray.NDArray.prototype.pow;
    ndarray.NDArray.prototype.operatorBinaryAnd=ndarray.NDArray.prototype.bitwise_and;
    ndarray.NDArray.prototype.operatorBinaryOr=ndarray.NDArray.prototype.bitwise_or;
    ndarray.NDArray.prototype.operatorBinaryXor=ndarray.NDArray.prototype.bitwise_xor;
    ndarray.NDArray.prototype.operatorBinaryLShift=ndarray.NDArray.prototype.bitwise_lshift;
    ndarray.NDArray.prototype.operatorBinaryRShift=ndarray.NDArray.prototype.bitwise_rshift;
    ndarray.NDArray.prototype.operatorLess=ndarray.NDArray.prototype.lt;
    ndarray.NDArray.prototype.operatorGreater=ndarray.NDArray.prototype.gt;
    ndarray.NDArray.prototype.operatorGreaterEqual=ndarray.NDArray.prototype.gte;
    ndarray.NDArray.prototype.operatorLessEqual=ndarray.NDArray.prototype.lte;
    ndarray.NDArray.prototype.operatorEqual=ndarray.NDArray.prototype.eq;
    ndarray.NDArray.prototype.operatorNotEqual=ndarray.NDArray.prototype.ne;
  }  
  
}


/**
 * show the image
 * @param  {any} image NDArray | dataurl | image binary in Uint8Array
 */
box.showImage=function(image){
  let str=null;
  if(typeof image =="string"){
    str=image;
  }else {
    if(image instanceof ndarray.NDArray){
      image = ndarray.image.encodeJpeg(image);
    }
    if(image instanceof Uint8Array){      
      image = ndarray.image.convertUint8ArrrayToDataurl(image);
    }
  }
  let html=`<img src="${image}"/>`;
  globalThis.document.body.append(document.createRange().createContextualFragment(html));
};
