/**
 * copyright © 2021-2021 kikoqiu
 * MIT Licence
 * 
 */

 var bfjs=(function(){



	
var Flags={};
Flags.BF_ST_INVALID_OP=  (1 << 0)
Flags.BF_ST_DIVIDE_ZERO= (1 << 1)
Flags.BF_ST_OVERFLOW =   (1 << 2)
Flags.BF_ST_UNDERFLOW  = (1 << 3)
Flags.BF_ST_INEXACT  =   (1 << 4)
/* indicate that a memory allocation error occured. NaN is returned */
Flags.BF_ST_MEM_ERROR =  (1 << 5) 

Flags.BF_RADIX_MAX= 36 /* maximum radix for bf_atof() and bf_ftoa() */

/* additional flags for bf_atof */
/* do not accept hex radix prefix (0x or 0X) if radix = 0 or radix = 16 */
Flags.BF_ATOF_NO_HEX =      (1 << 16)
/* accept binary (0b or 0B) or octal (0o or 0O) radix prefix if radix = 0 */
Flags.BF_ATOF_BIN_OCT =     (1 << 17)
/* Do not parse NaN or Inf */
Flags.BF_ATOF_NO_NAN_INF =  (1 << 18)
/* return the exponent separately */
Flags.BF_ATOF_EXPONENT   =    (1 << 19)


/* Conversion of floating point number to string. Return a null
terminated string or NULL if memory error. *plen contains its
length if plen != NULL.  The exponent letter is "e" for base 10,
"p" for bases 2, 8, 16 with a binary exponent and "@" for the other
bases. */

Flags.BF_FTOA_FORMAT_MASK =(3 << 16)

/* fixed format: prec significant digits rounded with (flags &
BF_RND_MASK). Exponential notation is used if too many zeros are
needed.*/
Flags.BF_FTOA_FORMAT_FIXED =(0 << 16)
/* fractional format: prec digits after the decimal point rounded with
(flags & BF_RND_MASK) */
Flags.BF_FTOA_FORMAT_FRAC  =(1 << 16)
/* free format: 

For binary radices with bf_ftoa() and for bfdec_ftoa(): use the minimum
number of digits to represent 'a'. The precision and the rounding
mode are ignored.

For the non binary radices with bf_ftoa(): use as many digits as
necessary so that bf_atof() return the same number when using
precision 'prec', rounding to nearest and the subnormal
configuration of 'flags'. The result is meaningful only if 'a' is
already rounded to 'prec' bits. If the subnormal flag is set, the
exponent in 'flags' must also be set to the desired exponent range.
*/
Flags.BF_FTOA_FORMAT_FREE=  (2 << 16)
/* same as BF_FTOA_FORMAT_FREE but uses the minimum number of digits
(takes more computation time). Identical to BF_FTOA_FORMAT_FREE for
binary radices with bf_ftoa() and for bfdec_ftoa(). */
Flags.BF_FTOA_FORMAT_FREE_MIN= (3 << 16)

/* force exponential notation for fixed or free format */
Flags.BF_FTOA_FORCE_EXP  =  (1 << 20)
/* add 0x prefix for base 16, 0o prefix for base 8 or 0b prefix for
base 2 if non zero value */
Flags.BF_FTOA_ADD_PREFIX =  (1 << 21)
/* return "Infinity" instead of "Inf" and add a "+" for positive
exponents */
Flags.BF_FTOA_JS_QUIRKS   = (1 << 22)
Flags.BF_POW_JS_QUIRKS= (1 << 16); /* (+/-1)^(+/-Inf) = NaN, 1^NaN = NaN */

Flags.BF_RNDN=0; /* round to nearest, ties to even */
Flags.BF_RNDZ=1; /* round to zero */
Flags.BF_RNDD=2; /* round to -inf (the code relies on (BF_RNDD xor BF_RNDU) = 1) */
Flags.BF_RNDU=3;/* round to +inf */
Flags.BF_RNDNA=4; /* round to nearest, ties away from zero */
Flags.BF_RNDA=5; /* round away from zero */
Flags.BF_RNDF=6; /* faithful rounding (nondeterministic, either RNDD or RNDU,
			inexact flag is always set)  */




var module=
{
	//faster but delayed gc, need more memory
	gc_use_finalization:false,//(!!window.FinalizationRegistry),
	gc_array: new Set(),
	//gc_debug_set: new Set(),
	gc_registry : new FinalizationRegistry(heldValue => {
		module.libbf._delete_(heldValue);
		//module.gc_debug_set.delete(heldValue);
	}),
	gc(){
		if(this.gc_use_finalization)return;
		if(this.gcing)return;
		this.gcing=true;
		let ele=[...this.gc_array].sort((a,b)=>{
				let diff=b.visited-a.visited;
				if(diff>(2**31) || diff <-(2**31)){
					diff*=-1;
				}
				return diff;
			}
		);
		let gcstartpos=Math.floor(this.gc_ele_limit/2);
		for(let i=gcstartpos;i<ele.length;++i){
			let e=ele[i];
			e.dispose();					
		}
		this.gc_array=new Set(ele.slice(0,gcstartpos));
		this.gcing=false;
	},
	visit_index:0,
	gc_track(f,addToArray=true){
		if(this.gc_use_finalization){
			return;
		}
		f.visited=this.visit_index;
		this.visit_index=(this.visit_index+1)%(2**32);
		//f.visited=new Date().getTime();
		if(addToArray){
			module.gc_array.add(f);
			if(this.gc_array.size>=this.gc_ele_limit){
				this.gc();
			}
		}
	},
	precision:500,
	precision_array:[],
	push_precision(prec){
		this.precision_array.push(this.precision);
		this.precision=prec;
	},
	pop_precision(){
		this.precision=this.precision_array.pop();
	},
	decimal_precision(dp){
		if(dp!=undefined){
			this.precision=Math.ceil(dp*Math.log2(10));
		}else{
			return Math.ceil(this.precision/Math.log2(10));
		}
	},
	push_decimal_precision(dp){
		this.push_precision(0);
		this.decimal_precision(dp);
	},
	gc_ele_limit:200,//maxmum elements before gc
	ready(){
		console.log('bfjs ready');
	},
	is_ready(){
		return !!this.libbf;
	},
	libbf:null,
	bf(val,radix=10){
		return new this._bf(val,radix);
	}
}; 


function bf(val,radix=10){
	this.h=module.libbf._new_();
	if(module.gc_use_finalization){
		module.gc_registry.register(this, this.h,this);
		//module.gc_debug_set.add(this.h);
	}else{	
		module.gc_track(this);
	}
	this.status=0;
	switch(typeof(val)){
		case "undefined":
			break;
		case "string":
			this.fromString(val,radix);
			break;
		case "number":
			this.fromNumber(val);
			break;
		case 'bigint':
			this.fromString(val.toString(),10);
			break;
		case 'object':
			if(!!val &&val.constructor==bf)	{
				this.copy(val);
				break;
			}
		default:
			throw new Error('bf: invalid constructor oprand '+ typeof(val));
	}
}
bf.prototype.toUint8Array = function() {
	const BF_T_STRUCT_SIZE = 20; // Size of bf_t struct in bytes (4*5 fields)
	const limb_size = 4; // Assuming limb_t is uint32_t (4 bytes)

	// Create a DataView to read bf_t struct members from WASM memory
	const dataView = new DataView(module.libbf.HEAPU8.buffer, this.h, BF_T_STRUCT_SIZE);
	
	// Read ctx, sign, expn, len, tab from the bf_t struct
	// For the purpose of backup, we only need len and the tab pointer's target data
	// ctx field is at offset 0 (4 bytes, pointer)
	// sign field is at offset 4 (4 bytes, int)
	// expn field is at offset 8 (4 bytes, slimb_t)
	const len = dataView.getUint32(12, true); // len field is at offset 12 (4 bytes, limb_t)
	const tab_ptr = dataView.getUint32(16, true); // tab field is at offset 16 (4 bytes, pointer)
	
	const limbs_byte_length = len * limb_size;
	const total_backup_size = BF_T_STRUCT_SIZE + limbs_byte_length;
	
	const h_bak_data = new Uint8Array(total_backup_size);
	
	// Copy bf_t struct data (20 bytes)
	h_bak_data.set(new Uint8Array(module.libbf.HEAPU8.buffer, this.h, BF_T_STRUCT_SIZE), 0);
	
	// Copy limb data if tab pointer is valid and len > 0
	if (tab_ptr !== 0 && len > 0) {
		h_bak_data.set(new Uint8Array(module.libbf.HEAPU8.buffer, tab_ptr, limbs_byte_length), BF_T_STRUCT_SIZE);
	}
	return h_bak_data;
};
module._bf=bf;
bf.prototype.dispose=function(recoverable=true){
	if(module.gc_use_finalization){
		if(!recoverable){
			module.gc_registry.unregister(this);
			if(this.h!=0){
				module.libbf._delete_(this.h);
				//module.gc_debug_set.delete(this.h);
				this.h=0;
			}
		}
	}else{
		if(this.h!=0){
			if(recoverable){
				this.h_bak = this.toUint8Array();
				
				// Clean up for GC
				module.libbf._delete_(this.h); // Free the bf_t struct itself
				this.h = 0;
			} else {
				module.libbf._delete_(this.h);
				this.h=0;
			}
		}
	}
}
module.fromUint8Array = function(data) {
	const BF_T_STRUCT_SIZE = 20; // Size of bf_t struct in bytes
	const limb_size = 4; // Assuming limb_t is uint32_t (4 bytes)

	// Read len from the backed-up data
	const dataView = new DataView(data.buffer, 0, BF_T_STRUCT_SIZE);
	const len = dataView.getUint32(12, true); // len field is at offset 12

	const limbs_byte_length = len * limb_size;

	// Allocate new memory in WASM for bf_t struct and for limbs
	const new_h = module.libbf._malloc(BF_T_STRUCT_SIZE);
	let new_tab_ptr = 0;
	if (len > 0) {
		new_tab_ptr = module.libbf._malloc(limbs_byte_length);
	}
	
	// Copy the backed-up bf_t struct data to the new WASM memory location
	module.libbf.HEAPU8.set(data.subarray(0, BF_T_STRUCT_SIZE), new_h);

	// Copy the backed-up limb data to the new WASM memory location
	if (new_tab_ptr !== 0) {
		module.libbf.HEAPU8.set(data.subarray(BF_T_STRUCT_SIZE, BF_T_STRUCT_SIZE + limbs_byte_length), new_tab_ptr);
	}
	
	// Update the 'tab' pointer within the newly allocated bf_t struct
	// tab field is at offset 16 bytes (i.e., 16/4 = 4th Uint32 element)
	module.libbf.HEAPU32[(new_h / 4) + 4] = new_tab_ptr;
	
	return new_h;
};
bf.prototype.geth=function(){	
	if(module.gc_use_finalization){
		return this.h;
	}else{		
		if(this.h==0){
			if(this.h_bak){
				this.h = module.fromUint8Array(this.h_bak); // Use the new function
				this.h_bak = null; // Clear the backup
			} else {
				// Original behavior for a truly new bf
				this.h = module.libbf._new_();
			}
			module.gc_track(this,true);
		}else{
			//this would not cause gc, because addToArray=false		
			module.gc_track(this,false);
		}
		return this.h;
	}
}
bf.prototype.checkstatus=function(s){
	//if(s&Flags.BF_ST_INEXACT)console.log("libbf BF_ST_INEXACT ");
	//if(s&Flags.BF_ST_DIVIDE_ZERO)console.log("libbf BF_ST_DIVIDE_ZERO "+s);
	if(s&Flags.BF_ST_INVALID_OP)throw new Error("libbf BF_ST_INVALID_OP ");
	return s;
}
bf.prototype.checktype=function (...ar){
	for(let a of ar){
		if(a!==null && a.constructor!=bf){
			throw new Error('is not a bigfloat '+ !!a.constructor);
		};
	}
}
bf.prototype.wraptypeh=function (...ar){
	let ret=[];
	let disposes=[];
	ret.push(function(){
		for(let e of disposes){
			e.dispose(false);
		}
	});
	for(let a of ar){
		if(a===null){
			ret.push(0);
		}else if(a.constructor==bf){
			ret.push(a.geth());
		}else if(typeof(a) == 'string' || typeof(a) == 'number' || typeof(a) == 'bigint'){
			let b=new bf(a);
			ret.push(b.h);
			disposes.push(b);
		}else{
			throw new Error('is not a bigfloat '+ !!a.constructor);
		}
	}
	return ret;
}
bf.prototype.flag=/*bf_set_exp_bits(15) MAXMUM | */ Flags.BF_RNDN | Flags.BF_FLAG_SUBNORMAL;

bf.prototype.calc=function(method,a=null,b=null,prec){
	if(prec<1)prec=module.precision;
	let [cleanup,ah,bh]=this.wraptypeh(a,b);
	this.status|=module.libbf._calc(method.charCodeAt(0),this.geth(),ah,bh,prec,this.flag);
	cleanup();
	this.checkstatus(this.status);
	return this;
}
bf.prototype.calc2=function(method,a=null,b=null,prec,rnd_mode=0,q=null){
	if(prec<1)prec=module.precision;
	let [cleanup,ah,bh,qh]=this.wraptypeh(a,b,q);
	this.status|=module.libbf._calc2(method.charCodeAt(0),this.geth(),ah,bh,prec,this.flag,rnd_mode,qh);
	cleanup();
	this.checkstatus(this.status);
	return this;
}
bf.prototype.checkoprand=function(...args){
	for(let a of args){
		if(a===null || typeof(a)=='undefined'){
			throw new Error('oprand missmatch');
		}
	}
}
bf.prototype.setadd=function(a,b,prec=0){
	this.checkoprand(a,b);
	return this.calc('+',a,b,prec);
}
bf.prototype.setsub=function(a,b,prec=0){	
	this.checkoprand(a,b);
	return this.calc('-',a,b,prec);
}
bf.prototype.setmul=function(a,b,prec=0){	
	this.checkoprand(a,b);
	return this.calc('*',a,b,prec);
}
bf.prototype.setdiv=function(a,b,prec=0){
	this.checkoprand(a,b);	
	return this.calc('/',a,b,prec);
}
bf.prototype.setmod=function(a,b,prec=0){
	this.checkoprand(a,b);
	return this.calc2('%',a,b,prec,Flags.BF_RNDZ,null);
}
bf.prototype.setrem=function(a,b,prec=0){
	this.checkoprand(a,b);
	return this.calc2('%',a,b,prec,Flags.BF_RNDN,null);
}
bf.prototype.setor=function(a,b,prec=0){
	this.checkoprand(a,b);	
	return this.calc('|',a,b,prec);
}
bf.prototype.setxor=function(a,b,prec=0){	
	this.checkoprand(a,b);
	return this.calc('^',a,b,prec);
}
bf.prototype.setand=function(a,b,prec=0){
	this.checkoprand(a,b);	
	return this.calc('&',a,b,prec);
}

bf.prototype.setsqrt=function(a,prec=0){	
	this.checkoprand(a);
	return this.calc('s',a,null,prec);
}
/*round to prec*/
bf.prototype.fpround=function(prec=0,flags=Flags.BF_RNDN){
	return this.calc('r',null,null,prec,flags,null);
}
/*round to int*/
bf.prototype.round=function(){
	return this.calc('i',null,null,0,Flags.BF_RNDNA,null);
}
bf.prototype.trunc=function(){	
	return this.calc('i',null,null,0,Flags.BF_RNDZ,null);
}
bf.prototype.floor=function(){	
	return this.calc('i',null,null,0,Flags.BF_RNDD,null);
}
bf.prototype.ceil=function(){
	return this.calc('i',null,null,0,Flags.BF_RNDU,null);
}
bf.prototype.neg=function(){	
	return this.calc('n',null,null,0);
}
bf.prototype.abs=function(){	
	return this.calc('b',null,null,0);
}

bf.prototype.setsign=function(a,prec=0){
	this.checkoprand(a);
	return this.calc('g',a,null,prec);
}
bf.prototype.setLOG2=function(prec=0){	
	return this.calc('2',null,null,prec);
}
bf.prototype.setPI=function(prec=0){	
	return this.calc('3',null,null,prec);
}
bf.prototype.setMIN_VALUE=function(prec=0){	
	return this.calc('z',null,null,prec);
}
bf.prototype.setMAX_VALUE=function(prec=0){	
	return this.calc('Z',null,null,prec);
}
bf.prototype.setEPSILON=function(prec=0){	
	return this.calc('y',null,null,prec);
}



bf.prototype.setexp=function(a,prec=0){	
	this.checkoprand(a);
	return this.calc('E',a,null,prec);
}
bf.prototype.setlog=function(a,prec=0){	
	this.checkoprand(a);
	return this.calc('L',a,null,prec);
}
bf.prototype.setpow=function(a,b,prec=0){
	this.checkoprand(a,b);
	return this.calc('P',a,b,prec,this.flag|Flags.BF_POW_JS_QUIRKS);
}
bf.prototype.setcos=function(a,prec=0){	
	this.checkoprand(a);
	return this.calc('c',a,null,prec);
}
bf.prototype.setsin=function(a,prec=0){	
	this.checkoprand(a);
	return this.calc('S',a,null,prec);
}
bf.prototype.settan=function(a,prec=0){	
	this.checkoprand(a);
	return this.calc('T',a,null,prec);
}
bf.prototype.setatan=function(a,prec=0){
	this.checkoprand(a);	
	return this.calc('4',a,null,prec);
}
bf.prototype.setatan2=function(a,b,prec=0){	
	this.checkoprand(a,b);
	return this.calc('5',a,null,prec);
}
bf.prototype.setasin=function(a,prec=0){	
	this.checkoprand(a);
	return this.calc('6',a,null,prec);
}
bf.prototype.setacos=function(a,prec=0){	
	this.checkoprand(a);
	return this.calc('7',a,null,prec);
}


bf.prototype.is_finit=function(){	
	return module.libbf._is_finite_(this.geth());
}
bf.prototype.is_nan=function(){	
	return module.libbf._is_nan_(this.geth());
}
bf.prototype.is_zero=function(){	
	return module.libbf._is_zero_(this.geth());
}
bf.prototype.copy=function(a){
	this.checkoprand(a);
	return module.libbf._set_(this.geth(),a.geth());
}
bf.prototype.clone=function(){	
	return new bf(this);
}
bf.prototype.fromNumber=function(a){
	return module.libbf._set_number_(this.geth(),a);
}
bf.prototype.toNumber=bf.prototype.f64=function(){	
	return module.libbf._get_number_(this.geth());
}
bf.prototype.cmp=function(b){
	this.checkoprand(b);
	let [cleanup,bh]=this.wraptypeh(b);
	let ret= module.libbf._cmp_(this.geth(),bh);
	cleanup();
	return ret;
}


let getFuncParameters=function (func) {
	if (typeof func == 'function') {
		var mathes = /[^(]+\(([^)]*)?\)/gm.exec(Function.prototype.toString.call(func));
		if (mathes[1]) {
			var args = mathes[1].replace(/[^,=\w]*/g, '').split(',');
			return args;
		}
	}
}
for(let k in bf.prototype){
	if(k.startsWith('set') && k!='set'){
		//console.log(k);
		let ofunc=bf.prototype[k];
		let ps=getFuncParameters(ofunc);
		//console.log(ps);
		let numps=ps.length;
		let nfunc=k.substr(3);
		bf.prototype[nfunc]=function(...args){
			if(numps==1){
				if(args.length!=0)throw new Error('oprands missmatch');
				let a=[];
				return ofunc.apply(new bf(), a);
			}else if(numps==2){
				if(args.length!=0)throw new Error('oprands missmatch');
				let a=[this];
				return ofunc.apply(new bf(), a);
			}else{
				if(args.length+2!=numps)throw new Error('oprands missmatch');
				let a=[this,...args];
				return ofunc.apply(new bf(), a);
			}
		}
		module[nfunc]=function(...args){			
			if(args.length+1!=numps)throw new Error('oprands missmatch');
			return ofunc.apply(new bf(), args);			
		}
	}
}

bf.prototype.operatorAdd=bf.prototype.add;
bf.prototype.operatorSub=bf.prototype.sub;
bf.prototype.operatorMul=bf.prototype.mul;
bf.prototype.operatorDiv=bf.prototype.div;
bf.prototype.operatorPow=bf.prototype.pow;
bf.prototype.operatorBinaryAnd=bf.prototype.and;
bf.prototype.operatorBinaryOr=bf.prototype.or;
bf.prototype.operatorBinaryXor=bf.prototype.xor;
//bf.prototype.operatorBinaryLShift=bf.prototype.mul2exp;
//bf.prototype.operatorBinaryRShift=bf.prototype.mul2exp;
bf.prototype.operatorLess=function(b){
	return this.cmp(b)<0;
}
bf.prototype.operatorGreater=function(b){
	return this.cmp(b)>0;
}
bf.prototype.operatorLessEqual=function(b){
	return this.cmp(b)<=0;
}
bf.prototype.operatorGreaterEqual=function(b){
	return this.cmp(b)>=0;
}
bf.prototype.operatorEqual=function(b){
	return this.cmp(b)==0;
}
bf.prototype.operatorNotEqual=function(b){
	return this.cmp(b)!=0;
}



bf.prototype.fromString=function(str,radix=10,prec=0){
	if(radix>64)throw new Error('radix error');
	if(prec<1)prec=module.precision;
	let hstr=module.libbf.allocateUTF8(str);
	let ret= module.libbf._atof_(this.geth(),hstr,radix,prec,0);
	module.libbf._free(hstr);
	this.checkstatus(ret);
	return this;
}
/**
 * 
 * @param {*} radix 
 * @param {*} prec precision digits in radix
 * @returns 
 */
bf.prototype.toString=function(radix=10,prec=-1){
	if(radix>64)throw new Error('radix error');
	if(prec<0)prec=Math.ceil(module.precision/Math.log2(radix));
	let flag=0;
	//Flags.BF_FTOA_FORMAT_FREE_MIN | Flags.BF_RNDZ | Flags.BF_FTOA_JS_QUIRKS;
	flag=Flags.BF_FTOA_FORMAT_FIXED| Flags.BF_RNDZ | Flags.BF_FTOA_JS_QUIRKS
	let ret= module.libbf._ftoa_(0,this.geth(),radix,prec,flag);
	let rets=module.libbf.AsciiToString(ret);
	module.libbf._free(ret);
	return rets;
}
/**
 * 
 * @param {*} radix 
 * @param {*} prec precision digits in radix
 * @returns 
 */
 bf.prototype.toFixed=function(radix=10,prec=-1,rnd_mode=Flags.BF_RNDNA){
	if(radix>64)throw new Error('radix error');
	if(prec<0)prec=Math.floor(module.precision/Math.log2(radix));
	let flag=0;
	flag= rnd_mode | Flags.BF_FTOA_FORMAT_FRAC
	let ret= module.libbf._ftoa_(0,this.geth(),radix,prec,flag);
	let rets=module.libbf.AsciiToString(ret);
	module.libbf._free(ret);
	return rets;
}



bf.prototype.toBigInt=function(){
	let s=this.toString(10,0);
	return BigInt(s);
}



module.helper={};

/**
 * Romberg integeration
 * @param {Function} f function
 * @param {*} _a start
 * @param {*} _b end
 * @param {*} _e Absolute error tolorance default 1e-30
 * @param {*} _re Relative error tolorance default =_e   (e < _e or re < _re)
 * @param {Object} info {max_step:20,max_acc:12,max_time:10000,steps:run steps,error:result error evaluation}
 * @returns result or null
 */
module.helper.romberg=function romberg(f,_a,_b,_e=1e-30,_re=_e,info={}){
  let max_step=info.max_step||20,
  	max_acc=info.max_acc||12,
  	max_time=info.max_time||10000;
  if(typeof(_e)!='number'|| typeof(_re)!='number' || typeof(info)!="object"){
	  throw new Error("arguments error");
  }
  let start_time=new Date().getTime();
  info.toString=function(){
	  return `lastresult=${this.lastresult}, 
	  effective_result=${this.eff_result},
	  steps=${this.steps}/${max_step}, 
	  error=${this.error.toString(10,3)},
	  rerror=${this.rerror.toString(10,3)},
	  eff_decimal_precision=${this.eff_decimal_precision}, 	  
	  exectime=${this.exectime}/${max_time}`
	};

  let a=module.bf(_a),b=module.bf(_b),e=module.bf(_e),re=module.bf(_re);
  const f0p5=module.bf(0.5);  
  const b_a_d=b.sub(a).mul(f0p5);
  let T=[0,b_a_d.mul(f(a).add(f(b)))];
  for(let m=2;m<=max_step;++m){  
    let Tm=[];    
  	let sum=module.bf(0);
  	for(let i=0;i<2**(m-2)/*do not overflow*/;++i){
      sum.setadd(sum,f(a.add(b_a_d.mul(i*2+1))));
    }
    Tm[1]=T[1].mul(f0p5).add(b_a_d.mul(sum));
  	b_a_d.setmul(b_a_d,f0p5);
    for(let j=2;j<=max_acc && j<=m;++j){
      let c=module.bf(4**(j-1)),c1=module.bf(4**(j-1)-1);
      Tm[j]=Tm[j-1].mul(c).sub(T[j-1]).div(c1);
    }
	let err=Tm[Tm.length-1].sub(T[T.length-1]).abs();
	let rerr=err.div(Tm[Tm.length-1]);
    if(!!info.debug && m>5){    	
        console.log('R['+m+']='+Tm[4]);
        console.log(err.toString(10,3));
    }

	info.exectime=new Date().getTime()-start_time;
	info.lastresult=Tm[Tm.length-1];	
	info.steps=m;
	info.error=err;
	info.rerror=rerr;
	info.eff_decimal_precision=Math.floor(-info.rerror.log().f64()/Math.log(10));
	if(info.eff_decimal_precision<=0){
		info.eff_decimal_precision=0;
		info.eff_result='';
	}else{
		if(info.eff_decimal_precision>bfjs.decimal_precision()){
			info.eff_result=info.lastresult.toString(10);
		}else{
			info.eff_result=info.lastresult.toString(10,info.eff_decimal_precision);
		}		
	}



	if(info.cb){
		info.cb();
	}
    if(m>5 && (err.cmp(e)<=0 || rerr.cmp(re)<=0)){
		info.result=info.lastresult;
		return info.result;
    }else if(m==max_step || info.exectime>max_time){
		info.result=null;
		return info.result;
    }
    T=Tm;
  }
}



module.Flags=Flags;
return module;
})();



createLibbf().then(function(m) {
	// this is reached when everything is ready, and you can call methods on Module		
	m._init_context_();	
	bfjs.libbf=m;
	bfjs.ready();
});

