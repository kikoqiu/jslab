(function (global, factory) {
    typeof exports === 'object' && typeof module !== 'undefined' ? factory(exports) :
    typeof define === 'function' && define.amd ? define(['exports'], factory) :
    (global = typeof globalThis !== 'undefined' ? globalThis : global || self, factory(global.ginac = {}));
})(this, (function (exports) { 'use strict';

    /**
     * ginac.ts
     * TypeScript wrapper for GiNaC WebAssembly using Manual Bindings.
     */
    // Enum mapping to C++ ResultType
    var ResultType;
    (function (ResultType) {
        ResultType[ResultType["STRING"] = 1] = "STRING";
        ResultType[ResultType["JSON"] = 2] = "JSON";
        ResultType[ResultType["NUMBER"] = 3] = "NUMBER";
        ResultType[ResultType["MATRIX"] = 4] = "MATRIX";
        ResultType[ResultType["LATEX"] = 5] = "LATEX";
    })(ResultType || (ResultType = {}));
    class OperatorBase {
        operatorAdd(right) {
            return op_add(this, right);
        }
        operatorSub(right) {
            return op_subtract(this, right);
        }
        operatorMul(right) {
            return op_multiply(this, right);
        }
        operatorDiv(right) {
            return op_divide(this, right);
        }
        operatorMod(right) {
            return op_mod(this, right);
        }
        operatorPow(right) {
            return op_power(this, right);
        }
        operatorNeg() {
            return op_negate(this);
        }
        operatorLess(right) {
            return op_less(this, right);
        }
        operatorGreater(right) {
            return op_greater(this, right);
        }
        operatorGreaterEqual(right) {
            return op_greaterequal(this, right);
        }
        operatorLessEqual(right) {
            return op_lessequal(this, right);
        }
        operatorEqual(right) {
            return op_equal(this, right);
        }
        operatorNotEqual(right) {
            return op_notequal(this, right);
        }
    }
    /**
     * Represents a GiNaC expression.
     * Wraps the underlying binary archive data.
     */
    class Expr extends OperatorBase {
        // Holds the raw binary archive data
        _data;
        _ctx;
        constructor(ctx, rawData) {
            super();
            this._ctx = ctx;
            this._data = rawData;
        }
        /**
         * Returns the string representation of the expression.
         */
        toString() {
            return this._ctx._fmt(this._data, ResultType.STRING);
        }
        /**
         * Returns the numerical value.
         * Returns null if the expression cannot be evaluated to a number.
         */
        toNumber() {
            return this._ctx._fmt(this._data, ResultType.NUMBER);
        }
        /**
         * Returns a JSON object representation of the expression structure.
         */
        toJSON() {
            const jsonStr = this._ctx._fmt(this._data, ResultType.JSON);
            return JSON.parse(jsonStr);
        }
        /**
         * Returns a 2D array of numbers (or strings if symbolic) representing a matrix.
         */
        toMatrix() {
            return this._ctx._fmt(this._data, ResultType.MATRIX);
        }
        /**
         * Returns the LaTeX representation of the expression.
         */
        toLatex() {
            return this._ctx._fmt(this._data, ResultType.LATEX);
        }
    }
    class Symbol extends OperatorBase {
        name;
        constructor(name) {
            super();
            this.name = name;
        }
    }
    class SymbolicNumber extends OperatorBase {
        value;
        constructor(value) {
            super();
            this.value = value;
        }
    }
    class GiNaCContext {
        _mod;
        _enc = new TextEncoder();
        _dec = new TextDecoder();
        constructor(module) {
            this._mod = module;
        }
        /**
         * @internal
         * Reads a ReturnBox* from C++ memory.
         */
        _readBox(boxPtr) {
            // boxPtr is in bytes, HEAP32 is Int32 array (4 bytes per element)
            const idx = boxPtr >> 2;
            const heap = this._mod.HEAP32;
            return {
                type: heap[idx],
                len: heap[idx + 1],
                ptr: heap[idx + 2]
            };
        }
        /**
         * @internal
         * Formats raw binary data to target type by calling C++.
         */
        _fmt(data, type) {
            const m = this._mod;
            // Allocate temporary buffer in C++ heap
            const inPtr = m._malloc(data.length);
            m.HEAPU8.set(data, inPtr);
            try {
                // Call C++ conversion
                const boxPtr = m._raw_getResult(inPtr, data.length, 1, type);
                const ret = this._readBox(boxPtr);
                // Handle Error
                if (ret.type === 0) {
                    const msg = this._dec.decode(m.HEAPU8.subarray(ret.ptr, ret.ptr + ret.len));
                    throw new Error("GiNaC Format Error: " + msg);
                }
                // Handle Number (Type 3)
                if (type === ResultType.NUMBER) {
                    if (ret.len === 0)
                        return null;
                    // Read 8 bytes double from ptr
                    // Use slice to avoid alignment issues if ptr is not multiple of 8 (though malloc usually is)
                    const bytes = m.HEAPU8.slice(ret.ptr, ret.ptr + 8);
                    return new Float64Array(bytes.buffer)[0];
                }
                // Handle Matrix (Type 4)
                if (type === ResultType.MATRIX) {
                    return this._parseMatrix(ret.ptr);
                }
                // Handle Text/JSON (Type 1, 2, 5)
                // Zero-copy decoding using subarray
                return this._dec.decode(m.HEAPU8.subarray(ret.ptr, ret.ptr + ret.len));
            }
            finally {
                m._free(inPtr);
            }
        }
        /**
         * @internal
         * Parse binary matrix format from C++.
         */
        _parseMatrix(ptr) {
            const u8 = this._mod.HEAPU8;
            const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
            let off = ptr;
            const rows = dv.getInt32(off, true);
            off += 4;
            const cols = dv.getInt32(off, true);
            off += 4;
            const result = [];
            for (let r = 0; r < rows; r++) {
                const row = [];
                for (let c = 0; c < cols; c++) {
                    const cellType = u8[off++];
                    if (cellType === 1) { // Double
                        const val = dv.getFloat64(off, true);
                        off += 8;
                        row.push(val);
                    }
                    else { // String
                        const len = dv.getInt32(off, true);
                        off += 4;
                        const str = this._dec.decode(u8.subarray(off, off + len));
                        row.push(str);
                        off += len;
                    }
                }
                result.push(row);
            }
            return result;
        }
        /**
         * Executes the C++ function.
         */
        exec(name, ...args) {
            const m = this._mod;
            // 1. Serialize arguments into a single buffer
            // Calculate total size first
            let totalSize = 4; // NumArgs
            const prepared = args.map(arg => {
                if (arg instanceof Expr) {
                    totalSize += 5 + arg._data.length;
                    return { type: 1, data: arg._data };
                }
                else if (typeof arg === "number") {
                    const buf = new Uint8Array(new Float64Array([arg]).buffer);
                    totalSize += 5 + buf.length;
                    return { type: 2, data: buf };
                }
                else if (arg instanceof SymbolicNumber) {
                    const buf = this._enc.encode(arg.value);
                    totalSize += 5 + buf.length;
                    return { type: 3, data: buf };
                }
                else if (arg instanceof Symbol) {
                    const buf = this._enc.encode(arg.name);
                    totalSize += 5 + buf.length;
                    return { type: 4, data: buf };
                }
                else {
                    if (typeof arg !== "string") {
                        arg = String(arg);
                    }
                    const buf = this._enc.encode(arg);
                    totalSize += 5 + buf.length; // Type(1) + Len(4) + Data
                    return { type: 0, data: buf };
                }
            });
            const bufPtr = m._malloc(totalSize);
            const u8 = m.HEAPU8;
            const dv = new DataView(u8.buffer, u8.byteOffset, u8.byteLength);
            let off = bufPtr;
            // Write count
            dv.setInt32(off, prepared.length, true);
            off += 4;
            // Write args
            for (const p of prepared) {
                u8[off++] = p.type;
                dv.setInt32(off, p.data.length, true);
                off += 4;
                u8.set(p.data, off);
                off += p.data.length;
            }
            // 2. Prepare function name
            const nameLen = m.lengthBytesUTF8(name) + 1;
            const namePtr = m._malloc(nameLen);
            m.stringToUTF8(name, namePtr, nameLen);
            let resExpr;
            try {
                // 3. Call C++
                const boxPtr = m._raw_callFunc(namePtr, bufPtr);
                const ret = this._readBox(boxPtr);
                if (ret.type === 0) {
                    const msg = this._dec.decode(u8.subarray(ret.ptr, ret.ptr + ret.len));
                    throw new Error(`GiNaC Exec Error [${name}]: ${msg}`);
                }
                // 4. Copy result (Binary Archive)
                // Must copy because C++ buffer is reused
                const resData = new Uint8Array(ret.len);
                resData.set(u8.subarray(ret.ptr, ret.ptr + ret.len));
                resExpr = new Expr(this, resData);
            }
            finally {
                m._free(bufPtr);
                m._free(namePtr);
            }
            return resExpr;
        }
        /**
         * Returns a list of all exported function names.
         */
        getExportedFunctions() {
            const boxPtr = this._mod._raw_get_all_exported_functions();
            const ret = this._readBox(boxPtr);
            // Result is a JSON string
            const json = this._dec.decode(this._mod.HEAPU8.subarray(ret.ptr, ret.ptr + ret.len));
            return JSON.parse(json);
        }
    }
    exports.ginac = void 0;
    /**
     * Initializes the GiNaC WebAssembly module.
     * @param wasmModuleFactory - The function returned by the Emscripten script.
     */
    async function initGiNaC(module = undefined) {
        if (exports.ginac) {
            return exports.ginac;
        }
        if (!module) {
            let createGinacModule;
            // @ts-ignore
            const isNode = typeof window === 'undefined';
            if (isNode) {
                // @ts-ignore
                createGinacModule = require("./ginac.js");
            }
            else {
                // @ts-ignore
                createGinacModule = await importScripts("./ginac.js");
            }
            module = await createGinacModule();
        }
        exports.ginac = new GiNaCContext(module);
        //console.log("Preloaded exported functions:", ginac.getExportedFunctions());
        return exports.ginac;
    }
    // ========================================================================
    // Generated Helper Functions
    // ========================================================================
    function sym(name) {
        return new Symbol(name);
    }
    function symNum(value) {
        return new SymbolicNumber(String(value));
    }
    /** Computes the characteristic polynomial of a matrix. */
    function charpoly(matrix, variable) {
        return exports.ginac.exec("charpoly", matrix, variable);
    }
    /** Returns the coefficient of var^deg in expr. */
    function coeff(expr, variable, deg) {
        return exports.ginac.exec("coeff", expr, variable, deg);
    }
    /** Collects terms involving a variable. */
    function collect(expr, sym) {
        return exports.ginac.exec("collect", expr, sym);
    }
    /** Collects common factors from the terms of sums. */
    function collect_common_factors(expr) {
        return exports.ginac.exec("collect_common_factors", expr);
    }
    /** Collects coefficients of a distributed polynomial. */
    function collect_distributed(expr, syms) {
        return exports.ginac.exec("collect_distributed", expr, syms);
    }
    /** Returns the content of a polynomial (gcd of coefficients). */
    function content(expr, variable) {
        return exports.ginac.exec("content", expr, variable);
    }
    /** Converts Harmonic polylogarithms to Li functions. */
    function convert_H_to_Li(expr, parameter) {
        return exports.ginac.exec("convert_H_to_Li", expr, parameter);
    }
    /** Decomposes a rational function. */
    function decomp_rational(expr, variable) {
        return exports.ginac.exec("decomp_rational", expr, variable);
    }
    /** Returns the degree of the expression with respect to a symbol. */
    function degree(expr, sym) {
        return exports.ginac.exec("degree", expr, sym);
    }
    /** Returns the denominator of a rational function. */
    function denom(expr) {
        return exports.ginac.exec("denom", expr);
    }
    /** Computes the determinant of a matrix. */
    function determinant(matrix) {
        return exports.ginac.exec("determinant", matrix);
    }
    /** Creates a diagonal matrix. */
    function diag(...args) {
        return exports.ginac.exec("diag", ...args);
    }
    /**
     * Computes the derivative.
     * Supports: diff(expr, symbol) or diff(expr, symbol, order).
     */
    function diff(expr, symbol, order) {
        if (order !== undefined) {
            return exports.ginac.exec("diff", expr, symbol, order);
        }
        return exports.ginac.exec("diff", expr, symbol);
    }
    /** Polynomial division (exact). */
    function divide(expr1, expr2) {
        return exports.ginac.exec("divide", expr1, expr2);
    }
    /** Evaluates an expression numerically. */
    function evalf(expr) {
        return exports.ginac.exec("evalf", expr);
    }
    /** Evaluates sums, products and integer powers of matrices. */
    function evalm(expr) {
        return exports.ginac.exec("evalm", expr);
    }
    /** Evaluates integrals. */
    function eval_integ(expr) {
        return exports.ginac.exec("eval_integ", expr);
    }
    /** Expands an expression. */
    function expand(expr) {
        return exports.ginac.exec("expand", expr);
    }
    /** Factors a polynomial. */
    function factor(expr) {
        return exports.ginac.exec("factor", expr);
    }
    /** Finds occurrences of a pattern in an expression. */
    function find(expr, pattern) {
        return exports.ginac.exec("find", expr, pattern);
    }
    /** Numerical root finding. fsolve(eq, var, start, end). */
    function fsolve(eq, variable, start, end) {
        return exports.ginac.exec("fsolve", eq, variable, start, end);
    }
    /** Greatest Common Divisor. */
    function gcd(a, b) {
        return exports.ginac.exec("gcd", a, b);
    }
    /** Checks if expression contains a subexpression. */
    function has(expr, pattern) {
        return exports.ginac.exec("has", expr, pattern);
    }
    /** Integer content of a polynomial. */
    function integer_content(expr) {
        return exports.ginac.exec("integer_content", expr);
    }
    /**
     * Indefinite or definite integral.
     * integral(expr, var) or integral(expr, var, lower, upper).
     */
    function integral(variable, lower, upper, expr) {
        return exports.ginac.exec("integral", variable, lower, upper, expr);
    }
    /** Inverse of a matrix. */
    function inverse(matrix) {
        return exports.ginac.exec("inverse", matrix);
    }
    /** Dummy print for tab completion. */
    function iprint() {
        return exports.ginac.exec("iprint");
    }
    /** Logic check (returns numeric 1 or 0 inside Expr). */
    function is(relation) {
        return exports.ginac.exec("is", relation);
    }
    /** Least Common Multiple. */
    function lcm(a, b) {
        return exports.ginac.exec("lcm", a, b);
    }
    /** Leading coefficient. */
    function lcoeff(expr, sym) {
        return exports.ginac.exec("lcoeff", expr, sym);
    }
    /** Degree of the leading term. */
    function ldegree(expr, sym) {
        return exports.ginac.exec("ldegree", expr, sym);
    }
    /** Linear equation solver. */
    function lsolve(eqs, vars) {
        return exports.ginac.exec("lsolve", eqs, vars);
    }
    /** Map function over operands. */
    function map(expr, funcName) {
        return exports.ginac.exec("map", expr, funcName);
    }
    /** Pattern matching. */
    function match(expr, pattern) {
        return exports.ginac.exec("match", expr, pattern);
    }
    /** Number of operands. */
    function nops(expr) {
        return exports.ginac.exec("nops", expr);
    }
    /** Normalizes a rational function. */
    function normal(expr) {
        return exports.ginac.exec("normal", expr);
    }
    /** Numerator of a rational function. */
    function numer(expr) {
        return exports.ginac.exec("numer", expr);
    }
    /** Numerator and Denominator (returns list). */
    function numer_denom(expr) {
        return exports.ginac.exec("numer_denom", expr);
    }
    /** Get the i-th operand. */
    function op(expr, index) {
        return exports.ginac.exec("op", expr, index);
    }
    /** Power function (base^exp). */
    function pow(base, exp) {
        return exports.ginac.exec("pow", base, exp);
    }
    /** Pseudo-remainder. */
    function prem(expr1, expr2, sym) {
        return exports.ginac.exec("prem", expr1, expr2, sym);
    }
    /** Primitive part of a polynomial. */
    function primpart(expr, sym) {
        return exports.ginac.exec("primpart", expr, sym);
    }
    function print() {
        return exports.ginac.exec("print");
    }
    function print_csrc() {
        return exports.ginac.exec("print_csrc");
    }
    function print_latex() {
        return exports.ginac.exec("print_latex");
    }
    /** Quotient. */
    function quo(expr1, expr2, sym) {
        return exports.ginac.exec("quo", expr1, expr2, sym);
    }
    /** Rank of a matrix. */
    function rank(matrix) {
        return exports.ginac.exec("rank", matrix);
    }
    /** Remainder. */
    function rem(expr1, expr2, sym) {
        return exports.ginac.exec("rem", expr1, expr2, sym);
    }
    /** Resultant of two polynomials. */
    function resultant(expr1, expr2, sym) {
        return exports.ginac.exec("resultant", expr1, expr2, sym);
    }
    /** Series expansion. */
    function series(expr, relation, order) {
        return exports.ginac.exec("series", expr, relation, order);
    }
    /** Converts a series to a polynomial. */
    function series_to_poly(expr) {
        return exports.ginac.exec("series_to_poly", expr);
    }
    /** Sparse pseudo-remainder. */
    function sprem(expr1, expr2, sym) {
        return exports.ginac.exec("sprem", expr1, expr2, sym);
    }
    /**
     * Square-free factorization.
     * Supports sqrfree(expr) or sqrfree(expr, vars_list).
     */
    function sqrfree(expr, vars) {
        if (vars !== undefined) {
            return exports.ginac.exec("sqrfree", expr, vars);
        }
        return exports.ginac.exec("sqrfree", expr);
    }
    /** Square-free partial fraction decomposition. */
    function sqrfree_parfrac(expr, sym) {
        return exports.ginac.exec("sqrfree_parfrac", expr, sym);
    }
    /** Square root. */
    function sqrt(expr) {
        return exports.ginac.exec("sqrt", expr);
    }
    /**
     * Substitution.
     * subs(expr, relation_or_list) or subs(expr, pattern, replacement).
     */
    function subs(expr, arg2, arg3) {
        if (arg3 !== undefined) {
            return exports.ginac.exec("subs", expr, arg2, arg3);
        }
        return exports.ginac.exec("subs", expr, arg2);
    }
    /** Trailing coefficient. */
    function tcoeff(expr, sym) {
        return exports.ginac.exec("tcoeff", expr, sym);
    }
    function time() { return exports.ginac.exec("time"); }
    /** Trace of a matrix. */
    function trace(matrix) {
        return exports.ginac.exec("trace", matrix);
    }
    /** Matrix transposition. */
    function transpose(matrix) {
        return exports.ginac.exec("transpose", matrix);
    }
    /** Unit part. */
    function unit(expr, sym) {
        return exports.ginac.exec("unit", expr, sym);
    }
    // --- Kernel Functions ---
    function basic_log_kernel() {
        return exports.ginac.exec("basic_log_kernel");
    }
    function multiple_polylog_kernel(arg) {
        return exports.ginac.exec("multiple_polylog_kernel", arg);
    }
    function ELi_kernel(n, p, x, y) {
        return exports.ginac.exec("ELi_kernel", n, p, x, y);
    }
    function Ebar_kernel(n, p, x, y) {
        return exports.ginac.exec("Ebar_kernel", n, p, x, y);
    }
    function Kronecker_dtau_kernel(...args) {
        return exports.ginac.exec("Kronecker_dtau_kernel", ...args);
    }
    function Kronecker_dz_kernel(...args) {
        return exports.ginac.exec("Kronecker_dz_kernel", ...args);
    }
    function Eisenstein_kernel(...args) {
        return exports.ginac.exec("Eisenstein_kernel", ...args);
    }
    function Eisenstein_h_kernel(...args) {
        return exports.ginac.exec("Eisenstein_h_kernel", ...args);
    }
    function modular_form_kernel(...args) {
        return exports.ginac.exec("modular_form_kernel", ...args);
    }
    function user_defined_kernel(arg1, arg2) {
        return exports.ginac.exec("user_defined_kernel", arg1, arg2);
    }
    function q_expansion_modular_form(arg1, arg2, arg3) {
        return exports.ginac.exec("q_expansion_modular_form", arg1, arg2, arg3);
    }
    /** Multiple polylogarithm G(a, y) or G(a, s, y). */
    function G(a, y, s) {
        if (s !== undefined) {
            return exports.ginac.exec("G", a, y, s);
        }
        return exports.ginac.exec("G", a, y);
    }
    /** Harmonic polylogarithm H(x, y). */
    function H(x, y) {
        return exports.ginac.exec("H", x, y);
    }
    /** Polylogarithm Li(n, x). */
    function Li(n, x) {
        return exports.ginac.exec("Li", n, x);
    }
    /** Dilogarithm Li2(x). */
    function Li2(x) {
        return exports.ginac.exec("Li2", x);
    }
    /** Trilogarithm Li3(x). */
    function Li3(x) {
        return exports.ginac.exec("Li3", x);
    }
    /** Order term function. */
    function Order(x) {
        return exports.ginac.exec("Order", x);
    }
    /** Nielsen's generalized polylogarithm S(n, p, x). */
    function S(n, p, x) {
        return exports.ginac.exec("S", n, p, x);
    }
    /** Absolute value. */
    function abs(x) {
        return exports.ginac.exec("abs", x);
    }
    /** Inverse cosine. */
    function acos(x) {
        return exports.ginac.exec("acos", x);
    }
    /** Inverse hyperbolic cosine. */
    function acosh(x) {
        return exports.ginac.exec("acosh", x);
    }
    /** Inverse sine. */
    function asin(x) {
        return exports.ginac.exec("asin", x);
    }
    /** Inverse hyperbolic sine. */
    function asinh(x) {
        return exports.ginac.exec("asinh", x);
    }
    /** Inverse tangent. */
    function atan(x) {
        return exports.ginac.exec("atan", x);
    }
    /** Inverse tangent of y/x (arctangent with two arguments). */
    function atan2(y, x) {
        return exports.ginac.exec("atan2", y, x);
    }
    /** Inverse hyperbolic tangent. */
    function atanh(x) {
        return exports.ginac.exec("atanh", x);
    }
    /** Beta function. */
    function beta(x, y) {
        return exports.ginac.exec("beta", x, y);
    }
    /** Binomial coefficient. */
    function binomial(n, k) {
        return exports.ginac.exec("binomial", n, k);
    }
    /** Complex conjugate. */
    function conjugate(x) {
        return exports.ginac.exec("conjugate", x);
    }
    /** Cosine. */
    function cos(x) {
        return exports.ginac.exec("cos", x);
    }
    /** Hyperbolic cosine. */
    function cosh(x) {
        return exports.ginac.exec("cosh", x);
    }
    /** Sign of a complex number (csgn). */
    function csgn(x) {
        return exports.ginac.exec("csgn", x);
    }
    /** Eta function. */
    function eta(x, y) {
        return exports.ginac.exec("eta", x, y);
    }
    /** Exponential function. */
    function exp(x) {
        return exports.ginac.exec("exp", x);
    }
    /** Factorial function. */
    function factorial(n) {
        return exports.ginac.exec("factorial", n);
    }
    /** Imaginary part of a complex number. */
    function imag_part(x) {
        return exports.ginac.exec("imag_part", x);
    }
    /** Logarithm of the Gamma function. */
    function lgamma(x) {
        return exports.ginac.exec("lgamma", x);
    }
    /** Natural logarithm. */
    function log(x) {
        return exports.ginac.exec("log", x);
    }
    /** Psi function (Digamma) or Polygamma function psi(n, x). */
    function psi(arg0, arg1) {
        if (arg1 !== undefined) {
            return exports.ginac.exec("psi", arg0, arg1);
        }
        return exports.ginac.exec("psi", arg0);
    }
    /** Real part of a complex number. */
    function real_part(x) {
        return exports.ginac.exec("real_part", x);
    }
    /** Sine. */
    function sin(x) {
        return exports.ginac.exec("sin", x);
    }
    /** Hyperbolic sine. */
    function sinh(x) {
        return exports.ginac.exec("sinh", x);
    }
    /** Heaviside step function. */
    function step(x) {
        return exports.ginac.exec("step", x);
    }
    /** Tangent. */
    function tan(x) {
        return exports.ginac.exec("tan", x);
    }
    /** Hyperbolic tangent. */
    function tanh(x) {
        return exports.ginac.exec("tanh", x);
    }
    /** Gamma function. */
    function tgamma(x) {
        return exports.ginac.exec("tgamma", x);
    }
    /** Riemann Zeta function zeta(x) or Hurwitz Zeta zeta(s, x). */
    function zeta(arg0, arg1) {
        if (arg1 !== undefined) {
            return exports.ginac.exec("zeta", arg0, arg1);
        }
        return exports.ginac.exec("zeta", arg0);
    }
    /** Derivatives of the Riemann Zeta function. */
    function zetaderiv(n, x) {
        return exports.ginac.exec("zetaderiv", n, x);
    }
    /**
     * Helper to create an Expr from a raw string without calculation.
     */
    function parse(input) {
        return exports.ginac.exec("parse", input);
    }
    /**
     * Helper to create a list Expr from multiple inputs.
     */
    function list(...input) {
        return exports.ginac.exec("list", ...input);
    }
    /**
     * Simplify trigonometric expressions
     */
    function trigsimp(input) {
        return exports.ginac.exec("trigsimp", input);
    }
    let integWarn = true;
    function integ(expr, x) {
        if (integWarn) {
            integWarn = false;
            console.warn("Integ is unstable feature.");
        }
        return exports.ginac.exec("integ", expr, x);
    }
    function matrix(...input) {
        let ls = input[0];
        if (Array.isArray(ls)) {
            ls = `{${ls.map(r => `{${r.join(',')}}`).join(',')}}`;
            console.log(ls);
        }
        return exports.ginac.exec("matrix", ls);
    }
    function op_add(a, b) {
        return exports.ginac.exec("op_add", a, b);
    }
    function op_subtract(a, b) {
        return exports.ginac.exec("op_subtract", a, b);
    }
    function op_multiply(a, b) {
        return exports.ginac.exec("op_multiply", a, b);
    }
    function op_divide(a, b) {
        return exports.ginac.exec("op_divide", a, b);
    }
    function op_mod(a, b) {
        return exports.ginac.exec("op_mod", a, b);
    }
    function op_equal(a, b) {
        return exports.ginac.exec("op_equal", a, b);
    }
    function op_notequal(a, b) {
        return exports.ginac.exec("op_notequal", a, b);
    }
    function op_less(a, b) {
        return exports.ginac.exec("op_less", a, b);
    }
    function op_lessequal(a, b) {
        return exports.ginac.exec("op_lessequal", a, b);
    }
    function op_greater(a, b) {
        return exports.ginac.exec("op_greater", a, b);
    }
    function op_greaterequal(a, b) {
        return exports.ginac.exec("op_greaterequal", a, b);
    }
    function op_negate(a) {
        return exports.ginac.exec("op_negate", a);
    }
    function op_power(a, b) {
        return exports.ginac.exec("op_power", a, b);
    }
    function op_factorial(a) {
        return exports.ginac.exec("op_factorial", a);
    }

    exports.ELi_kernel = ELi_kernel;
    exports.Ebar_kernel = Ebar_kernel;
    exports.Eisenstein_h_kernel = Eisenstein_h_kernel;
    exports.Eisenstein_kernel = Eisenstein_kernel;
    exports.Expr = Expr;
    exports.G = G;
    exports.GiNaCContext = GiNaCContext;
    exports.H = H;
    exports.Kronecker_dtau_kernel = Kronecker_dtau_kernel;
    exports.Kronecker_dz_kernel = Kronecker_dz_kernel;
    exports.Li = Li;
    exports.Li2 = Li2;
    exports.Li3 = Li3;
    exports.OperatorBase = OperatorBase;
    exports.Order = Order;
    exports.S = S;
    exports.Symbol = Symbol;
    exports.SymbolicNumber = SymbolicNumber;
    exports.abs = abs;
    exports.acos = acos;
    exports.acosh = acosh;
    exports.asin = asin;
    exports.asinh = asinh;
    exports.atan = atan;
    exports.atan2 = atan2;
    exports.atanh = atanh;
    exports.basic_log_kernel = basic_log_kernel;
    exports.beta = beta;
    exports.binomial = binomial;
    exports.charpoly = charpoly;
    exports.coeff = coeff;
    exports.collect = collect;
    exports.collect_common_factors = collect_common_factors;
    exports.collect_distributed = collect_distributed;
    exports.conjugate = conjugate;
    exports.content = content;
    exports.convert_H_to_Li = convert_H_to_Li;
    exports.cos = cos;
    exports.cosh = cosh;
    exports.csgn = csgn;
    exports.decomp_rational = decomp_rational;
    exports.degree = degree;
    exports.denom = denom;
    exports.determinant = determinant;
    exports.diag = diag;
    exports.diff = diff;
    exports.divide = divide;
    exports.eta = eta;
    exports.eval_integ = eval_integ;
    exports.evalf = evalf;
    exports.evalm = evalm;
    exports.exp = exp;
    exports.expand = expand;
    exports.factor = factor;
    exports.factorial = factorial;
    exports.find = find;
    exports.fsolve = fsolve;
    exports.gcd = gcd;
    exports.has = has;
    exports.imag_part = imag_part;
    exports.initGiNaC = initGiNaC;
    exports.integ = integ;
    exports.integer_content = integer_content;
    exports.integral = integral;
    exports.inverse = inverse;
    exports.iprint = iprint;
    exports.is = is;
    exports.lcm = lcm;
    exports.lcoeff = lcoeff;
    exports.ldegree = ldegree;
    exports.lgamma = lgamma;
    exports.list = list;
    exports.log = log;
    exports.lsolve = lsolve;
    exports.map = map;
    exports.match = match;
    exports.matrix = matrix;
    exports.modular_form_kernel = modular_form_kernel;
    exports.multiple_polylog_kernel = multiple_polylog_kernel;
    exports.nops = nops;
    exports.normal = normal;
    exports.numer = numer;
    exports.numer_denom = numer_denom;
    exports.op = op;
    exports.op_add = op_add;
    exports.op_divide = op_divide;
    exports.op_equal = op_equal;
    exports.op_factorial = op_factorial;
    exports.op_greater = op_greater;
    exports.op_greaterequal = op_greaterequal;
    exports.op_less = op_less;
    exports.op_lessequal = op_lessequal;
    exports.op_mod = op_mod;
    exports.op_multiply = op_multiply;
    exports.op_negate = op_negate;
    exports.op_notequal = op_notequal;
    exports.op_power = op_power;
    exports.op_subtract = op_subtract;
    exports.parse = parse;
    exports.pow = pow;
    exports.prem = prem;
    exports.primpart = primpart;
    exports.print = print;
    exports.print_csrc = print_csrc;
    exports.print_latex = print_latex;
    exports.psi = psi;
    exports.q_expansion_modular_form = q_expansion_modular_form;
    exports.quo = quo;
    exports.rank = rank;
    exports.real_part = real_part;
    exports.rem = rem;
    exports.resultant = resultant;
    exports.series = series;
    exports.series_to_poly = series_to_poly;
    exports.sin = sin;
    exports.sinh = sinh;
    exports.sprem = sprem;
    exports.sqrfree = sqrfree;
    exports.sqrfree_parfrac = sqrfree_parfrac;
    exports.sqrt = sqrt;
    exports.step = step;
    exports.subs = subs;
    exports.sym = sym;
    exports.symNum = symNum;
    exports.tan = tan;
    exports.tanh = tanh;
    exports.tcoeff = tcoeff;
    exports.tgamma = tgamma;
    exports.time = time;
    exports.trace = trace;
    exports.transpose = transpose;
    exports.trigsimp = trigsimp;
    exports.unit = unit;
    exports.user_defined_kernel = user_defined_kernel;
    exports.zeta = zeta;
    exports.zetaderiv = zetaderiv;

}));
