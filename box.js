var box=globalThis;

box.outputBuffer={'result':'','resultScript':'','plotdiv':0};

box.initOutputBuffer=async function(){
  box.outputBuffer={'result':'','resultScript':'','plotdiv':await globalEval('vm.selected.plotdiv') || 0};
}

box.flushOutputBuffer=async function(){
  await globalEval(`vm.selected.result += ${JSON.stringify(box.outputBuffer.result)};`);
  await globalEval(`vm.selected.resultScript += ${JSON.stringify(box.outputBuffer.resultScript)};`);
  await globalEval(`vm.selected.plotdiv = ${box.outputBuffer.plotdiv};`);
}

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
  box.outputBuffer.result += str;
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
  box.outputBuffer.result += str;
};

/**
 * dump object to JSON to output
 * @param  {...any} o output 
 */
box.dumpJSON=function(...o){
  let str='';
  for(var i of o){
    str+=`<pre><code>${JSON.stringify(i)}</code></pre>`;
  }
  box.outputBuffer.result += str;
};

/**
 * Plot with plotly.js. General plot function. Parameters are passed to plotly
 * @param {*} data data arrays
 * @param {*} layout 
 * @param {*} config 
 * @param {*} style Extra plot html div css style
 */
box.plotly=function(data, layout, config, style){
  if(!style){
    style=''
  }
  layout=JSON.stringify(layout);
  config=JSON.stringify(config);  
  let json=JSON.stringify(data); 
  var div='divplot'+new Date().getTime()+'-'+ box.outputBuffer.plotdiv++;
  //console.log('Plotly div:',div);
  box.outputBuffer.result += `<div class="plot" id="${div}" style="${style};"></div>`; 
  let scr=`Plotly.react("${div}", ${json},${layout},${config});`;//newPlot
  box.outputBuffer.resultScript += scr;
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
 */
box.plot=function(...xPoints_yPoints){
  let args=xPoints_yPoints;
  let style=null;  
  let data=[];
  let layout=null;
  let config=null;
  if(args.length==1){
    let x=range(0,args[0].length);
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
  box.plotly(data,layout,config,style);
};

/**
 * Plot a 3d line figure
 * @param  {...any} xPoints_yPoints_zPoints x-points,y-points,z-points,[x-point,y-point,z-points...],[{labels,style,layout,config}] 
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
  box.plotly(data,layout,config,style || 'height:600px;');
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
  var div='divplot'+new Date().getTime()+'-'+ box.outputBuffer.plotdiv++;
  if(false){
    //box.outputBuffer.result+= '<div class="latex" id="'+div+'" style="'+style+'">'+MathJax.tex2svg(result, {em: 16, ex: 6, display: false}).outerHTML+'</div>';
  }else{
    let json=JSON.stringify(result); 
    box.outputBuffer.result+= '<div class="latex" id="'+div+'" style="'+style+'"></div>';
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

/**
 * Load the d3 library with linkedom polyfill
 */
box.loadD3=async function(){
  if(globalThis.d3 && globalThis.linkedom){
    return;
  }

  await importScripts('./3pty/linkedom-browser/dist/linkedom.browser.min.js');
  let {document}= globalThis.linkedom.parseHTML('<html><body></body></html>');
  window.document=document;

  await importScripts('./3pty/d3@7.js')

}