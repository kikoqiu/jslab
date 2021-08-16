var box={};//window;

/**
 * Range generator
 * @param {*} a start
 * @param {*} b end
 * @param {*} step step
 * @param {*} mapper mapper function eg. i=>sin(i)
 */
box.rangen=function *(a,b,step=1,mapper){
    if(typeof(step)=='function'){
        mapper=step;
        step=a/a;
    }
    if(!!mapper){
        for(let i=a;i<b;i=i+step){    
            yield mapper(i);
        }
    }else{
        for(let i=a;i<b;i=i+step){    
            yield i;
        }
    }
};
/**
 * Range array
 * @param {*} a start
 * @param {*} b end
 * @param {*} step step
 * @param {*} mapper mapper function eg. i=>sin(i)
 * @returns 
 */
box.range=function(a,b,step=1,mapper){
  return [...box.rangen(a,b,step,mapper)];
}
/**
 * echo output
 * @param  {...any} o output 
 */
box.echo=function(...o){
  let str='';
  for(i of o){
    if(str!='')str+=",&nbsp;";
    str+=String(i)
  }
  vm.selected.result+=str+'<br/>\r\n';
};

/**
 * Plot with plotly
 * @param {*} data data arrays
 * @param {*} layout 
 * @param {*} config 
 * @param {*} style Extra plot div style
 */
box.plotly=function(data, layout, config, style){
  if(!style){
    style=''
  }
  layout=JSON.stringify(layout);
  config=JSON.stringify(config);  
  let json=JSON.stringify(data); 
  //let deed=window.pako.deflate(json,{level:9});
  //console.log(json.length);
  //console.log(deed.length);
  div='divplot'+new Date().getTime()+(vm.plotdiv++);
  vm.selected.result+=
    '<div class="plot" id="'+div+'" style="'+style+';"></div>';
  let scr=`Plotly.react("${div}", ${json},${layout},${config});`;//newPlot
  vm.selected.resultScript+=scr;
};

/**
 * Unpace array of objects to get array of properties
 * @param {Array} data 
 */
box.unpack=function(data,...props){
  if(props.length==1)return data.map(o=>o[props]);
  let ret=[];
  for(let i=0;i<props.length;++i){
    let p=props[i];
    ret.push(data.map(o=>o[p]));
  }
  return ret;
};


/**
 * Plot a line figure
 * @param  {...any} args x-points,y-points,x-point,y-point...,[{labels,style,layout,config}] 
 */
box.plot=function(...args){
  let style=null;  
  let data=[];
  let layout=null;
  let config=null;
  if(args.length==1){
    let x=range(0,args[0].length);
    let y=args[0];
    data.push({x,y,type: 'scatter'});
  }else{
    for(var pos=0;pos<args.length;pos+=2){
      let x=args[pos];
      let y=args[pos+1];
      data.push({x,y,type: 'scatter'});
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
 * @param  {...any} args x-points,y-points,z-points,x-point,y-point,z-points...,[{labels,style,layout,config}] 
 */
box.plot3d=function(...args){
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
 * @returns a promise with [content,filename,size]
 */
box.readfile=function(type='text'){
  return new Promise((resolve,rej) => {    
    let file=document.getElementById('file');
    let done=false;
    let doit=function() {
      if(done)return;
      done=true;
      try{
        var selectedFile = file.files[0];
        if(!selectedFile)return rej(new Error('file not selected'));
        var name = selectedFile.name; 
        var size = selectedFile.size; 
        console.log("filename:" + name + "size:" + size);
        var reader = new FileReader(); 
        switch(type){
          case 'text':
            reader.readAsText(selectedFile); 
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
            return rej(new Error('unknown type '+type));
        }        
        reader.onload = function() {
            //console.log(this.result);
            resolve([this.result,name,size]);
        }
        reader.onerror = function(event) {
          return rej(new Error('on error '+event));
        }
        file.value='';
      }catch(e){
        return rej(e);
      }
    }
    file.onchange=doit;
    setTimeout(doit,60000);
    file.click(); 
  });  
}

/*box.fplot=function(func,rng,  style='width:600px;height:300px;'){
  box.plot(rng,rng.map(x=>func(x)))
};*/

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

/**
 * Get latex string
 * @param  {...any} ex 
 */
box.latexstr=function(...ex){
  let result='';

  for(let data of ex){
    if(!!window.pyodide&&pyodide.isPyProxy){
      if(pyodide.isPyProxy(data)){
        data=box.importpy('sympy').latex(data);
      }
    }
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
  div='divplot'+new Date().getTime()+vm.plotdiv++;
  if(false){
 vm.selected.result+=
    '<div class="latex" id="'+div+'" style="'+style+'">'+MathJax.tex2svg(result, {em: 16, ex: 6, display: false}).outerHTML+'</div>';
  }else{
    let json=JSON.stringify(result); 
    vm.selected.result+=
      '<div class="latex" id="'+div+'" style="'+style+'"></div>';
    let scr='document.getElementById("'+div+'").appendChild(MathJax.tex2svg('+json+', {em: 16, ex: 6, display: true}));'
    vm.selected.resultScript+=scr;
  }
};
/*
//safe box
box.global_proxy=new Proxy(box, {    
        get: function(target, prop, receiver) {          
          return prop in target ? target[prop] : undefined;
        },
        has: function(target, prop) {
          return true;
        },
        set: function(target, prop, value, receiver) {
          target[prop]=value;
          return true;
        }
      }
    );

box.eval=eval;
box.window=window;
box.console=console;
box.bfjs=bfjs;
box._Op=_Op;
box.Math=Math;
box.Date=Date;
  */
box.global_proxy=new Proxy(window, {    
      get: function(target, prop, receiver) {  
        if(prop in box) return box[prop];        
        if(prop in target)return target[prop];        
        return undefined;
      },
      has: function(target, prop) {
        if(prop in box) return true;        
        if(prop in target)return true;
        return false;   
      },
      set: function(target, prop, value, receiver) {
        box[prop]=value;
        return true;
      }
    }
  );

/**
 * Runs a piece of code async
 * @param {*} code 
 * @returns the promise
 */
box.runcode=function (code,info){  
  var input = code;
  var currentCode = Babel.transform("(async function(){'bpo enable';\r\n"+input+"\r\n})();", 
    { 
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
    }  ).code;
  console.log(currentCode);
  if(info)info.compiled_code=currentCode;
  box.box=box;
  box.currentCode=currentCode;
  with(box.global_proxy){
    let ret= eval(currentCode);
    return ret;
  }  
}

/**
 * Runs a piece of code sync
 * @param {*} code 
 * @returns the result
 */
 box.runcodeSync_=function (code,info){  
  var input = code;
  var currentCode = Babel.transform("'bpo enable';\r\n"+input, 
  { 
    presets: [/*"es2017"*/] ,//env Babel.availablePresets
    plugins: ["bpo"]}
  ).code;
  if(info)info.compiled_code=currentCode;
  console.log(currentCode);
  box.box=box;
  box.currentCode=currentCode;

  with(box.proxy){
    let ret= eval(currentCode);
    return ret;
  }
  
}





box._pyloaded=false;
box._loadpy=async function(){
  if(box._pyloaded)return;

  box._pyloaded=true;
  window.pyodide=box.pyodide = await loadPyodide({
      indexURL : "https://cdn.jsdelivr.net/pyodide/v0.18.0/full/"
    });  
}
/**
 * Runs a piece of python code
 * @param {*} code 
 * @returns the result
 */
box.runpy=function(code){
  let ret=pyodide.runPython(`
    ${code}
  `);
  return ret;
}
box._loadedpypackages=new Set();

/**
 * Load and init python, relies on Pyodide
 * @param  {...any} modules python modules to load (eg. sympy, numpy) 
 */
box.loadpy=async function(...modules){
  await box._loadpy();
  for(let m of modules){
    if(box._loadedpypackages.has(m)){
      continue;
    }
    box._loadedpypackages.add(m);
    await pyodide.loadPackage(m);
  }
}
box._pypackages={};
/**
 * Import a python module, run loadpy first
 * @param {*} name module name
 * @param {*} asname as name
 * @returns 
 */
box.importpy=function(name,asname=name){
  let ret=box._pypackages[name];
  if(!ret){
    box.runpy(`
      import ${name} as ${asname}
      `);
    ret=pyodide.globals.get(asname);
    box._pypackages[name]=ret;
  }
  return ret;
}
/**
 * Get a sympy sym
 * @param {*} e the expression (eg. 'x sin(x)')
 * @param {*} sub substute dict (eg. {x:'tan(x)'})
 * @returns the symbols
 */
box.sym=function(e,sub=null){
  sympy=box.importpy('sympy');
  let ret= sympy.sympify(e);
  if(sub==null){
    return ret;
  }
  sub=pyodide.toPy(sub);
  if(ret.length){    
    return ret.map(s=>s.subs(sub));
  }else{
    return ret.subs(sub);
  }
}

/**
 * javascript object to python 
 * @param {*} jso 
 * @returns 
 */
box.toPy=function(jso){
  return pyodide.toPy(jso);
}

/**
 * get extremum of a func
 * @param {*} sym 'a**2+1/b/(a-b),a,b'
 * @param {*} domain 
 * @returns 
 */
box.extremum= function (sym,domain=sympy.S.Reals){
  let syms=[...sympy.sympify(sym)];
  let func=syms[0];
  let xs=syms.slice(1);
  let diffs=xs.map(x=>func.diff(x));    
  solu=sympy.nonlinsolve(diffs,xs);
  //latex(solu);
  let res=[...solu];
  if(domain){
    res=res.filter(
      (so)=>
      ![...so].some(
        k=>!domain.has(k)
      )
    );
  }
  res=res.map(x=>{
    let s=[...x];
    let scope={};
    for(let i=0;i<s.length;++i){
      scope[xs[i]]=s[i];
    }    
    let y=null;
    try{
      y=func.subs(box.toPy(scope));
    }catch(e){
      console.log('failed to get y');
    }
    return [...x,y];
  });

  return res;
}
