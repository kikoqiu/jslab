let _Op = (function(){
	'bpo disable';
	return {
	add(a, b) {
		if(globalThis.pyodide?.isPyProxy){
			if(pyodide.isPyProxy(a)){
				return a.__add__(b);
			}else if(pyodide.isPyProxy(b)){
				return b.__radd__(a);
			}
		}
		if(typeof(a)!='object' && typeof(b)=='object'){
			a = new b.constructor(a);
		}else if(typeof(a)=='object' && typeof(b)=='object' && a.constructor!=b.constructor){
			throw new Error('bpo: not the same class');
		}
		if(typeof(a)=='object'&&a.operatorAdd) return a.operatorAdd(b);
		return a + b;
	},

	sub(a, b) {
		if(globalThis.pyodide?.isPyProxy){
			if(pyodide.isPyProxy(a)){
				return a.__sub__(b);
			}else if(pyodide.isPyProxy(b)){
				return b.__rsub__(a);
			}
		}
		if(typeof(a)!='object' && typeof(b)=='object'){
			a = new b.constructor(a);
		}else if(typeof(a)=='object' && typeof(b)=='object' && a.constructor!=b.constructor){
			throw new Error('bpo: not the same class');
		}
		if(typeof(a)=='object'&&a.operatorSub) return a.operatorSub(b);		    
		return a - b;
	},

	mul(a, b) {
		if(globalThis.pyodide?.isPyProxy){
			if(pyodide.isPyProxy(a)){
				return a.__mul__(b);
			}else if(pyodide.isPyProxy(b)){
				return b.__rmul__(a);
			}
		}
		if(typeof(a)!='object' && typeof(b)=='object'){
			a = new b.constructor(a);
		}else if(typeof(a)=='object' && typeof(b)=='object' && a.constructor!=b.constructor){
			throw new Error('bpo: not the same class');
		}
		if(typeof(a)=='object'&&a.operatorMul) return a.operatorMul(b);		    
		return a * b;
	},

	div(a, b) {
		if(globalThis.pyodide?.isPyProxy){
			if(pyodide.isPyProxy(a)){
				return a.__truediv__(b);
			}else if(pyodide.isPyProxy(b)){
				return b.__rtruediv__(a);
			}
		}
		if(typeof(a)!='object' && typeof(b)=='object'){
			a = new b.constructor(a);
		}else if(typeof(a)=='object' && typeof(b)=='object' && a.constructor!=b.constructor){
			throw new Error('bpo: not the same class');
		}
		if(typeof(a)=='object'&&a.operatorDiv) return a.operatorDiv(b);		    
		return a / b;
	},

	pow(a, b) {
		if(globalThis.pyodide?.isPyProxy){
			if(pyodide.isPyProxy(a)){
				return a.__pow__(b);
			}else if(pyodide.isPyProxy(b)){
				return b.__rpow__(a);
			}
		}
		if(typeof(a)!='object' && typeof(b)=='object'){
			a = new b.constructor(a);
		}else if(typeof(a)=='object' && typeof(b)=='object' && a.constructor!=b.constructor){
			throw new Error('bpo: not the same class');
		}
		if(typeof(a)=='object'&&a.operatorPow) return a.operatorPow(b);		    
		return a ** b;
	},

	binaryAnd(a, b) {
		if(globalThis.pyodide?.isPyProxy){
			if(pyodide.isPyProxy(a)){
				return a.__and__(b);
			}else if(pyodide.isPyProxy(b)){
				return b.__rand__(a);
			}
		}
		if(typeof(a)!='object' && typeof(b)=='object'){
			a = new b.constructor(a);
		}else if(typeof(a)=='object' && typeof(b)=='object' && a.constructor!=b.constructor){
			throw new Error('bpo: not the same class');
		}
		if(typeof(a)=='object'&&a.operatorBinaryAnd) return a.operatorBinaryAnd(b);		    
		return a & b;
	},

	binaryOr(a, b) {
		if(globalThis.pyodide?.isPyProxy){
			if(pyodide.isPyProxy(a)){
				return a.__or__(b);
			}else if(pyodide.isPyProxy(b)){
				return b.__ror__(a);
			}
		}
		if(typeof(a)!='object' && typeof(b)=='object'){
			a = new b.constructor(a);
		}else if(typeof(a)=='object' && typeof(b)=='object' && a.constructor!=b.constructor){
			throw new Error('bpo: not the same class');
		}
		if(typeof(a)=='object'&&a.operatorBinaryOr) return a.operatorBinaryOr(b);		    
		return a | b;
	},

	binaryXor(a, b) {
		if(globalThis.pyodide?.isPyProxy){
			if(pyodide.isPyProxy(a)){
				return a.__xor_(b);
			}else if(pyodide.isPyProxy(b)){
				return b.__rxor__(a);
			}
		}
		if(typeof(a)!='object' && typeof(b)=='object'){
			a = new b.constructor(a);
		}else if(typeof(a)=='object' && typeof(b)=='object' && a.constructor!=b.constructor){
			throw new Error('bpo: not the same class');
		}
		if(typeof(a)=='object'&&a.operatorBinaryXor) return a.operatorBinaryXor(b);		    
		return a ^ b;
	},

	binaryLShift(a, b) {
		if(globalThis.pyodide?.isPyProxy){
			if(pyodide.isPyProxy(a)){
				return a.__lshift__(b);
			}else if(pyodide.isPyProxy(b)){
				return b.__rlshift__(a);
			}
		}
		if(typeof(a)!='object' && typeof(b)=='object'){
			a = new b.constructor(a);
		}else if(typeof(a)=='object' && typeof(b)=='object' && a.constructor!=b.constructor){
			throw new Error('bpo: not the same class');
		}
		if(typeof(a)=='object'&&a.operatorBinaryLShift) return a.operatorBinaryLShift(b);		    
		return a << b;
	},

	binaryRShift(a, b) {
		if(globalThis.pyodide?.isPyProxy){
			if(pyodide.isPyProxy(a)){
				return a.__rshift__(b);
			}else if(pyodide.isPyProxy(b)){
				return b.__rrshift__(a);
			}
		}
		if(typeof(a)!='object' && typeof(b)=='object'){
			a = new b.constructor(a);
		}else if(typeof(a)=='object' && typeof(b)=='object' && a.constructor!=b.constructor){
			throw new Error('bpo: not the same class');
		}
		if(typeof(a)=='object'&&a.operatorBinaryRShift) return a.operatorBinaryRShift(b);		    
		return a >> b;
	},

	less(a, b) {
		if(globalThis.pyodide?.isPyProxy){
			if(pyodide.isPyProxy(a)){
				return a.__lt__(b);
			}else if(pyodide.isPyProxy(b)){
				return b.__gt__(a);
			}
		}
		if(typeof(a)=='object'&&a.operatorLess) return a.operatorLess(b);
		else if(typeof(b)=='object'&&b.operatorGreater) return b.operatorGreater(a);
		else if(typeof(a)=='object'&&a.operatorGreaterEqual) return !a.operatorGreaterEqual(b);		    
		return a < b;
	},

	greater(a, b) {
		if(globalThis.pyodide?.isPyProxy){
			if(pyodide.isPyProxy(a)){
				return a.__gt__(b);
			}else if(pyodide.isPyProxy(b)){
				return b.__lt__(a);
			}
		}
		if(typeof(a)=='object'&&a.operatorGreater) return a.operatorGreater(b);
		else if(typeof(b)=='object'&&b.operatorLess) return b.operatorLess(a);
		else if(typeof(a)=='object'&&a.operatorLessEqual) return !a.operatorLessEqual(b);		    
		return a > b;
	},

	lessEqual(a, b) {
		if(globalThis.pyodide?.isPyProxy){
			if(pyodide.isPyProxy(a)){
				return a.__le__(b);
			}else if(pyodide.isPyProxy(b)){
				return b.__ge__(a);
			}
		}
		if(typeof(a)=='object'&&a.operatorLessEqual) return a.operatorLessEqual(b);
		else if(typeof(b)=='object'&&b.operatorGreaterEqual) return b.operatorGreaterEqual(a);
		else if(typeof(a)=='object'&&a.operatorGreater) return !a.operatorGreater(b);		    
		return a <= b;
	},

	greaterEqual(a, b) {
		if(globalThis.pyodide?.isPyProxy){
			if(pyodide.isPyProxy(a)){
				return a.__ge__(b);
			}else if(pyodide.isPyProxy(b)){
				return b.__le__(a);
			}
		}
		if(typeof(a)=='object'&&a.operatorGreaterEqual) return a.operatorGreaterEqual(b);
		else if(typeof(b)=='object'&&b.operatorLessEqual) return b.operatorLessEqual(a);
		else if(typeof(a)=='object'&&a.operatorLess) return !a.operatorLess(b);		    
		return a >= b;
	},

	equal(a, b) {
		if(globalThis.pyodide?.isPyProxy){
			if(pyodide.isPyProxy(a)){
				return a.__eq__(b);
			}else if(pyodide.isPyProxy(b)){
				return b.__eq__(a);
			}
		}
		if(typeof(a)=='object'&&a.operatorEqual) return a.operatorEqual(b);
		else if(typeof(a)=='object'&&a.operatorNotEqual) return !a.operatorNotEqual(b);
		else if(typeof(b)=='object'&&b.operatorEqual) return b.operatorEqual(a);
		else if(typeof(b)=='object'&&b.operatorNotEqual) return !b.operatorNotEqual(a);		    
		return a == b;
	},

	notEqual(a, b) {
		if(globalThis.pyodide?.isPyProxy){
			if(pyodide.isPyProxy(a)){
				return a.__ne__(b);
			}else if(pyodide.isPyProxy(b)){
				return b.__ne__(a);
			}
		}
		if(typeof(a)=='object'&&a.operatorNotEqual) return a.operatorNotEqual(b);
		else if(typeof(a)=='object'&&a.operatorEqual) return !a.operatorEqual(b);
		else if(typeof(b)=='object'&&b.operatorNotEqual) return b.operatorNotEqual(a);
		else if(typeof(b)=='object'&&b.operatorEqual) return !b.operatorEqual(a);		    
		return a != b;
	},
};

})();	
	
export {_Op};

export function visitor(babel) {
    var t = babel.types;

	/*
	var preCode = (function() {
		var _Op=window._Op;
	}).toString();
	

    preCode = preCode.slice(preCode.indexOf('{') + 1, preCode.lastIndexOf('}'));

    var preCodeAST = babel.template(preCode)({});*/

    function initStatus(path) {
	var firstBlockStatement = path.findParent(path => t.isBlockStatement(path.node) || t.isProgram(path.node));
	if(firstBlockStatement) {
	    for(let directiveID in firstBlockStatement.node.directives) {
		let directive = firstBlockStatement.node.directives[directiveID];
		if(directive.value.value == 'bpo disable'){
		    path.node.BPO_HAVE_DEFAULT = true;
		    path.node.BPO_STATUS = false;
		    break;
		} else if(directive.value.value == 'bpo enable'){
		    path.node.BPO_HAVE_DEFAULT = true;
		    path.node.BPO_STATUS = true;
		    break;
		}
	    }
	    if(!path.node.BPO_HAVE_DEFAULT && firstBlockStatement.node.BPO_HAVE_DEFAULT) {
		path.node.BPO_HAVE_DEFAULT = true;
		path.node.BPO_STATUS = firstBlockStatement.node.BPO_STATUS;
	    }
	}
	if(!path.node.BPO_HAVE_DEFAULT) {
	    path.node.BPO_HAVE_DEFAULT = true;
	    path.node.BPO_STATUS = false;
	}
    }

    return {
	visitor: {
	    Program(path) {
		//path.unshiftContainer('body', preCodeAST);
	    },
	    BlockStatement(path) {
		initStatus(path);
	    },
	    BinaryExpression(path) {
		initStatus(path, true);
		if(!path.node.BPO_STATUS) return;
		var tab = {
		    '+': 'add',
		    '-': 'sub',
		    '*': 'mul',
		    '/': 'div',
		    '**': 'pow',

		    '&': 'binaryAnd',
		    '|': 'binaryOr',
		    '^': 'binaryXor',
		    '<<': 'binaryLShift',
		    '>>': 'binaryRShift',
		    
		    '<': 'less',
		    '>': 'greater',
		    '<=': 'lessEqual',
		    '>=': 'greaterEqual',
		    '==': 'equal',
		    '!=': 'notEqual',
		};
		if(!(path.node.operator in tab)) return;
		path.replaceWith(
		    t.callExpression(
			t.MemberExpression(t.identifier('_Op'), t.identifier(tab[path.node.operator])),
			[path.node.left, path.node.right]
		    )
		);
	    },
	},
    };
};

