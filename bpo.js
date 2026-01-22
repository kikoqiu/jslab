let _Op = (function () {
	'bpo disable';
	return {
		add(a, b) {
			if ((typeof (a) == 'number' && typeof (b) == 'number') || (typeof (a) == 'bigint' && typeof (b) == 'bigint')) return a + b;
			if (typeof (a) != 'object' && typeof (b) == 'object' && b.operatorAdd) {
				try { a = new b.constructor(a); } catch (e) { throw new Error(`bpo: cannot convert ${a} to ${b.constructor.name}, ${e.message}`); }
			} else if (typeof (a) == 'object' && typeof (b) == 'object' && a.constructor != b.constructor) {
				throw new Error('bpo: not the same class');
			}
			if (typeof (a) == 'object' && a.operatorAdd) return a.operatorAdd(b);
			return a + b;
		},

		sub(a, b) {
			if ((typeof (a) == 'number' && typeof (b) == 'number') || (typeof (a) == 'bigint' && typeof (b) == 'bigint')) return a - b;
			if (typeof (a) != 'object' && typeof (b) == 'object' && b.operatorSub) {
				try { a = new b.constructor(a); } catch (e) { throw new Error(`bpo: cannot convert ${a} to ${b.constructor.name}, ${e.message}`); }
			} else if (typeof (a) == 'object' && typeof (b) == 'object' && a.constructor != b.constructor) {
				throw new Error('bpo: not the same class');
			}
			if (typeof (a) == 'object' && a.operatorSub) return a.operatorSub(b);
			return a - b;
		},

		mul(a, b) {
			if ((typeof (a) == 'number' && typeof (b) == 'number') || (typeof (a) == 'bigint' && typeof (b) == 'bigint')) return a * b;
			if (typeof (a) != 'object' && typeof (b) == 'object' && b.operatorMul) {
				try { a = new b.constructor(a); } catch (e) { throw new Error(`bpo: cannot convert ${a} to ${b.constructor.name}, ${e.message}`); }
			} else if (typeof (a) == 'object' && typeof (b) == 'object' && a.constructor != b.constructor) {
				throw new Error('bpo: not the same class');
			}
			if (typeof (a) == 'object' && a.operatorMul) return a.operatorMul(b);
			return a * b;
		},

		div(a, b) {
			if ((typeof (a) == 'number' && typeof (b) == 'number') || (typeof (a) == 'bigint' && typeof (b) == 'bigint')) return a / b;
			if (typeof (a) != 'object' && typeof (b) == 'object' && b.operatorDiv) {
				try { a = new b.constructor(a); } catch (e) { throw new Error(`bpo: cannot convert ${a} to ${b.constructor.name}, ${e.message}`); }
			} else if (typeof (a) == 'object' && typeof (b) == 'object' && a.constructor != b.constructor) {
				throw new Error('bpo: not the same class');
			}
			if (typeof (a) == 'object' && a.operatorDiv) return a.operatorDiv(b);
			return a / b;
		},

		mod(a, b) {
			if ((typeof (a) == 'number' && typeof (b) == 'number') || (typeof (a) == 'bigint' && typeof (b) == 'bigint')) return a % b;
			if (typeof (a) != 'object' && typeof (b) == 'object' && b.operatorMod) {
				try { a = new b.constructor(a); } catch (e) { throw new Error(`bpo: cannot convert ${a} to ${b.constructor.name}, ${e.message}`); }
			} else if (typeof (a) == 'object' && typeof (b) == 'object' && a.constructor != b.constructor) {
				throw new Error('bpo: not the same class');
			}
			if (typeof (a) == 'object' && a.operatorMod) return a.operatorMod(b);
			return a % b;
		},

		pow(a, b) {
			if ((typeof (a) == 'number' && typeof (b) == 'number') || (typeof (a) == 'bigint' && typeof (b) == 'bigint')) return a ** b;
			if (typeof (a) != 'object' && typeof (b) == 'object' && b.operatorPow) {
				try { a = new b.constructor(a); } catch (e) { throw new Error(`bpo: cannot convert ${a} to ${b.constructor.name}, ${e.message}`); }
			} else if (typeof (a) == 'object' && typeof (b) == 'object' && a.constructor != b.constructor) {
				throw new Error('bpo: not the same class');
			}
			if (typeof (a) == 'object' && a.operatorPow) return a.operatorPow(b);
			return a ** b;
		},

		bitwiseAnd(a, b) {
			if ((typeof (a) == 'number' && typeof (b) == 'number') || (typeof (a) == 'bigint' && typeof (b) == 'bigint')) return a & b;
			if (typeof (a) != 'object' && typeof (b) == 'object' && b.operatorBitwiseAnd) {
				try { a = new b.constructor(a); } catch (e) { throw new Error(`bpo: cannot convert ${a} to ${b.constructor.name}, ${e.message}`); }
			} else if (typeof (a) == 'object' && typeof (b) == 'object' && a.constructor != b.constructor) {
				throw new Error('bpo: not the same class');
			}
			if (typeof (a) == 'object' && a.operatorBitwiseAnd) return a.operatorBitwiseAnd(b);
			return a & b;
		},

		bitwiseOr(a, b) {
			if ((typeof (a) == 'number' && typeof (b) == 'number') || (typeof (a) == 'bigint' && typeof (b) == 'bigint')) return a | b;
			if (typeof (a) != 'object' && typeof (b) == 'object' && b.operatorBitwiseOr) {
				try { a = new b.constructor(a); } catch (e) { throw new Error(`bpo: cannot convert ${a} to ${b.constructor.name}, ${e.message}`); }
			} else if (typeof (a) == 'object' && typeof (b) == 'object' && a.constructor != b.constructor) {
				throw new Error('bpo: not the same class');
			}
			if (typeof (a) == 'object' && a.operatorBitwiseOr) return a.operatorBitwiseOr(b);
			return a | b;
		},

		bitwiseXor(a, b) {
			if ((typeof (a) == 'number' && typeof (b) == 'number') || (typeof (a) == 'bigint' && typeof (b) == 'bigint')) return a ^ b;
			if (typeof (a) != 'object' && typeof (b) == 'object' && b.operatorBitwiseXor) {
				try { a = new b.constructor(a); } catch (e) { throw new Error(`bpo: cannot convert ${a} to ${b.constructor.name}, ${e.message}`); }
			} else if (typeof (a) == 'object' && typeof (b) == 'object' && a.constructor != b.constructor) {
				throw new Error('bpo: not the same class');
			}
			if (typeof (a) == 'object' && a.operatorBitwiseXor) return a.operatorBitwiseXor(b);
			return a ^ b;
		},

		bitwiseLShift(a, b) {
			if ((typeof (a) == 'number' && typeof (b) == 'number') || (typeof (a) == 'bigint' && typeof (b) == 'bigint')) return a << b;
			if (typeof (a) != 'object' && typeof (b) == 'object' && b.operatorBitwiseLShift) {
				try { a = new b.constructor(a); } catch (e) { throw new Error(`bpo: cannot convert ${a} to ${b.constructor.name}, ${e.message}`); }
			} else if (typeof (a) == 'object' && typeof (b) == 'object' && a.constructor != b.constructor) {
				throw new Error('bpo: not the same class');
			}
			if (typeof (a) == 'object' && a.operatorBitwiseLShift) return a.operatorBitwiseLShift(b);
			return a << b;
		},

		bitwiseRShift(a, b) {
			if ((typeof (a) == 'number' && typeof (b) == 'number') || (typeof (a) == 'bigint' && typeof (b) == 'bigint')) return a >> b;
			if (typeof (a) != 'object' && typeof (b) == 'object' && b.operatorBitwiseRShift) {
				try { a = new b.constructor(a); } catch (e) { throw new Error(`bpo: cannot convert ${a} to ${b.constructor.name}, ${e.message}`); }
			} else if (typeof (a) == 'object' && typeof (b) == 'object' && a.constructor != b.constructor) {
				throw new Error('bpo: not the same class');
			}
			if (typeof (a) == 'object' && a.operatorBitwiseRShift) return a.operatorBitwiseRShift(b);
			return a >> b;
		},

		bitwiseURShift(a, b) {
			if ((typeof (a) == 'number' && typeof (b) == 'number') || (typeof (a) == 'bigint' && typeof (b) == 'bigint')) return a >>> b;
			if (typeof (a) != 'object' && typeof (b) == 'object' && b.operatorBitwiseURShift) {
				try { a = new b.constructor(a); } catch (e) { throw new Error(`bpo: cannot convert ${a} to ${b.constructor.name}, ${e.message}`); }
			} else if (typeof (a) == 'object' && typeof (b) == 'object' && a.constructor != b.constructor) {
				throw new Error('bpo: not the same class');
			}
			if (typeof (a) == 'object' && a.operatorBitwiseURShift) return a.operatorBitwiseURShift(b);
			return a >>> b;
		},

		neg(a) {
			if (typeof (a) == 'number' || typeof (a) == 'bigint') return -a;
			if (typeof (a) == 'object' && a.operatorNeg) return a.operatorNeg();
			return -a;
		},

		bitwiseNot(a) {
			if (typeof (a) == 'number' || typeof (a) == 'bigint') return ~a;
			if (typeof (a) == 'object' && a.operatorBitwiseNot) return a.operatorBitwiseNot();
			return ~a;
		},


		less(a, b) {
			if ((typeof (a) == 'number' && typeof (b) == 'number') || (typeof (a) == 'bigint' && typeof (b) == 'bigint')) return a < b;
			if (typeof (a) == 'object' && a.operatorLess) return a.operatorLess(b);
			else if (typeof (b) == 'object' && b.operatorGreater) return b.operatorGreater(a);
			else if (typeof (a) == 'object' && a.operatorGreaterEqual) return !a.operatorGreaterEqual(b);
			return a < b;
		},

		greater(a, b) {
			if ((typeof (a) == 'number' && typeof (b) == 'number') || (typeof (a) == 'bigint' && typeof (b) == 'bigint')) return a > b;
			if (typeof (a) == 'object' && a.operatorGreater) return a.operatorGreater(b);
			else if (typeof (b) == 'object' && b.operatorLess) return b.operatorLess(a);
			else if (typeof (a) == 'object' && a.operatorLessEqual) return !a.operatorLessEqual(b);
			return a > b;
		},

		lessEqual(a, b) {
			if ((typeof (a) == 'number' && typeof (b) == 'number') || (typeof (a) == 'bigint' && typeof (b) == 'bigint')) return a <= b;
			if (typeof (a) == 'object' && a.operatorLessEqual) return a.operatorLessEqual(b);
			else if (typeof (b) == 'object' && b.operatorGreaterEqual) return b.operatorGreaterEqual(a);
			else if (typeof (a) == 'object' && a.operatorGreater) return !a.operatorGreater(b);
			return a <= b;
		},

		greaterEqual(a, b) {
			if ((typeof (a) == 'number' && typeof (b) == 'number') || (typeof (a) == 'bigint' && typeof (b) == 'bigint')) return a >= b;
			if (typeof (a) == 'object' && a.operatorGreaterEqual) return a.operatorGreaterEqual(b);
			else if (typeof (b) == 'object' && b.operatorLessEqual) return b.operatorLessEqual(a);
			else if (typeof (a) == 'object' && a.operatorLess) return !a.operatorLess(b);
			return a >= b;
		},

		equal(a, b) {
			if ((typeof (a) == 'number' && typeof (b) == 'number') || (typeof (a) == 'bigint' && typeof (b) == 'bigint')) return a == b;
			if (typeof (a) == 'object' && a.operatorEqual) return a.operatorEqual(b);
			else if (typeof (a) == 'object' && a.operatorNotEqual) return !a.operatorNotEqual(b);
			else if (typeof (b) == 'object' && b.operatorEqual) return b.operatorEqual(a);
			else if (typeof (b) == 'object' && b.operatorNotEqual) return !b.operatorNotEqual(a);
			return a == b;
		},

		notEqual(a, b) {
			if ((typeof (a) == 'number' && typeof (b) == 'number') || (typeof (a) == 'bigint' && typeof (b) == 'bigint')) return a != b;
			if (typeof (a) == 'object' && a.operatorNotEqual) return a.operatorNotEqual(b);
			else if (typeof (a) == 'object' && a.operatorEqual) return !a.operatorEqual(b);
			else if (typeof (b) == 'object' && b.operatorNotEqual) return b.operatorNotEqual(a);
			else if (typeof (b) == 'object' && b.operatorEqual) return !b.operatorEqual(a);
			return a != b;
		},
	};

})();

export { _Op };

export function visitor(babel) {
	var t = babel.types;

	function initStatus(path) {
		var firstBlockStatement = path.findParent(path => t.isBlockStatement(path.node) || t.isProgram(path.node));
		if (firstBlockStatement) {
			for (let directiveID in firstBlockStatement.node.directives) {
				let directive = firstBlockStatement.node.directives[directiveID];
				if (directive.value.value == 'bpo disable') {
					path.node.BPO_HAVE_DEFAULT = true;
					path.node.BPO_STATUS = false;
					break;
				} else if (directive.value.value == 'bpo enable') {
					path.node.BPO_HAVE_DEFAULT = true;
					path.node.BPO_STATUS = true;
					break;
				}
			}
			if (!path.node.BPO_HAVE_DEFAULT && firstBlockStatement.node.BPO_HAVE_DEFAULT) {
				path.node.BPO_HAVE_DEFAULT = true;
				path.node.BPO_STATUS = firstBlockStatement.node.BPO_STATUS;
			}
		}
		if (!path.node.BPO_HAVE_DEFAULT) {
			path.node.BPO_HAVE_DEFAULT = true;
			path.node.BPO_STATUS = false;
		}
	}

	function isPrimitive(nodePath) {
		// Quick check for literals
		if (t.isLiteral(nodePath.node)) {
			return true;
		}
		// Try to evaluate the expression to see if it results in a primitive value (e.g., constant variables)
		const evaluated = nodePath.evaluate();
		// If confident and value is null or not object (number, string, boolean, undefined)
		return evaluated.confident && (evaluated.value === null || typeof evaluated.value !== 'object');
	};

	return {
		visitor: {
			Program(path) {
			},
			BlockStatement(path) {
				initStatus(path);
			},			
			BinaryExpression(path) {
				initStatus(path, true);
				if (!path.node.BPO_STATUS) return;
				// Optimization: If both sides are primitives, skip overloading
				if (isPrimitive(path.get('left')) && isPrimitive(path.get('right'))) return;
				var tab = {
					'+': 'add',
					'-': 'sub',
					'*': 'mul',
					'/': 'div',
					'%': 'mod',
					'**': 'pow',

					'&': 'bitwiseAnd',
					'|': 'bitwiseOr',
					'^': 'bitwiseXor',
					'<<': 'bitwiseLShift',
					'>>': 'bitwiseRShift',
					'>>>': 'bitwiseURShift',

					'<': 'less',
					'>': 'greater',
					'<=': 'lessEqual',
					'>=': 'greaterEqual',
					'==': 'equal',
					'!=': 'notEqual',
				};
				if (!(path.node.operator in tab)) return;
				path.replaceWith(
					t.callExpression(
						t.MemberExpression(t.identifier('_Op'), t.identifier(tab[path.node.operator])),
						[path.node.left, path.node.right]
					)
				);
			},
			UnaryExpression(path) {
				initStatus(path, true);
				if (!path.node.BPO_STATUS) return;
				if (isPrimitive(path.get('argument'))) return;
				var tab = {
					'-': 'neg',
					'~': 'bitwiseNot',
				};
				if (!(path.node.operator in tab)) return;
				path.replaceWith(
					t.callExpression(
						t.MemberExpression(t.identifier('_Op'), t.identifier(tab[path.node.operator])),
						[path.node.argument]
					)
				);
			},
		},
	};
};