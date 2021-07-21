let box={};//window;

/**
 * range generator
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
 * range array
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
 * print output
 * @param  {...any} o output 
 */
box.echo=(...o)=>{
  let str='';
  for(i of o){
    str+=String(i);
  }
  vm.selected.result+=str+'<br/>\r\n';
};

/**
 * plotly plot (look at plotly for more infomation)
 * @param {*} data arrays
 * @param {*} layout 
 * @param {*} config 
 * @param {*} style extra plot div style
 */
box.plotly=function(data, layout, config, style){
  if(!style){
    style=''
  }
  let json=JSON.stringify(data); 
  div='divplot'+new Date().getTime()+(vm.plotdiv++);
  vm.selected.result+=
    '<div class="plot" id="'+div+'" style="'+style+';"></div>';
  let scr='Plotly.newPlot("'+div+'", '+json+');';
  vm.selected.resultScript+=scr;
  vm.$nextTick(()=>{eval(scr);});
};

/**
 * plot a scatter figure
 * @param  {...any} args [X1s,Y1s,X2s,Y2s,...] 
 */
box.plot=function(...args){
  let style=null;  
  let data=[];
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
      style=args[pos];
    }
  }
  box.plotly(data,undefined,undefined,style);
};

/*box.fplot=function(func,rng,  style='width:600px;height:300px;'){
  box.plot(rng,rng.map(x=>func(x)))
};*/

/**
 * compile a mathjs expression
 * return node with eval(scope) function
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
 * eval a mathjs expression
 * @param {*} e 
 * @param {*} scope 
 * @returns 
 */
box.eval_expr=function(e,scope){
  return box.compile_expr(e).eval(scope);
}
/**
 * Calc derivative of a mathjs expr
 */
box.deriv=math.derivative;


/**
 * display a latex expressoin
 * @param {*} data the latex
 * @param {*} style extra style
 */
box.latex=function(data, style=''){
  let json=JSON.stringify(data);
  div='divplot'+new Date().getTime()+vm.plotdiv++;
  if(false){
 vm.selected.result+=
    '<div class="latex" id="'+div+'" style="'+style+'">'+MathJax.tex2svg(data, {em: 16, ex: 6, display: false}).outerHTML+'</div>';
  }else{
    let json=JSON.stringify(data); 
    vm.selected.result+=
      '<div class="latex" id="'+div+'" style="'+style+'"></div>';
    let scr='document.getElementById("'+div+'").appendChild(MathJax.tex2svg('+json+', {em: 16, ex: 6, display: true}));'
    vm.selected.resultScript+=scr;
    vm.$nextTick(()=>{eval(scr);});
  }
};
/*
//safe box
box.proxy=new Proxy(box, {    
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
box.proxy=new Proxy(window, {    
      get: function(target, prop, receiver) {  
        if(prop in box) return box[prop];        
        if(prop in target)return target[prop];        
        return undefined;
      },
      has: function(target, prop) {
        return true;
      },
      set: function(target, prop, value, receiver) {
        box[prop]=value;
        return true;
      }
    }
  );
/**
 * runs a piece of code
 * @param {*} code 
 * @returns 
 */
box.runcode=function (code){
  
  var input = code;
  var currentCode = Babel.transform("'bpo enable';\r\n"+input, 
  { 
    presets: ["es2017"] ,//env Babel.availablePresets
    plugins: ["bpo"]}
  ).code;
  console.log(currentCode);

  box.currentCode=currentCode;
  with(box.proxy){
    let ret= eval(currentCode);
    return ret;
  }
  
}