/**
 * ginac.ts
 * TypeScript wrapper for GiNaC WebAssembly using Manual Bindings.
 */
declare enum ResultType {
    STRING = 1,
    JSON = 2,
    NUMBER = 3,
    MATRIX = 4,
    LATEX = 5
}
export declare class OperatorBase {
    operatorAdd(right: Param): Expr;
    operatorSub(right: Param): Expr;
    operatorMul(right: Param): Expr;
    operatorDiv(right: Param): Expr;
    operatorMod(right: Param): Expr;
    operatorPow(right: Param): Expr;
    operatorNeg(): Expr;
    operatorLess(right: Param): Expr;
    operatorGreater(right: Param): Expr;
    operatorGreaterEqual(right: Param): Expr;
    operatorLessEqual(right: Param): Expr;
    operatorEqual(right: Param): Expr;
    operatorNotEqual(right: Param): Expr;
}
/**
 * Represents a GiNaC expression.
 * Wraps the underlying binary archive data.
 */
export declare class Expr extends OperatorBase {
    readonly _data: Uint8Array;
    private readonly _ctx;
    constructor(ctx: GiNaCContext, rawData: Uint8Array);
    /**
     * Returns the string representation of the expression.
     */
    toString(): string;
    /**
     * Returns the numerical value.
     * Returns null if the expression cannot be evaluated to a number.
     */
    toNumber(): number | null;
    /**
     * Returns a JSON object representation of the expression structure.
     */
    toJSON(): object;
    /**
     * Returns a 2D array of numbers (or strings if symbolic) representing a matrix.
     */
    toMatrix(): (number | string)[][];
    /**
     * Returns the LaTeX representation of the expression.
     */
    toLatex(): string;
}
export declare class Symbol extends OperatorBase {
    name: string;
    constructor(name: string);
}
export declare class SymbolicNumber extends OperatorBase {
    value: string;
    constructor(value: string);
}
export type Param = any | Expr;
export declare class GiNaCContext {
    private _mod;
    private _enc;
    private _dec;
    constructor(module: any);
    /**
     * @internal
     * Reads a ReturnBox* from C++ memory.
     */
    private _readBox;
    /**
     * @internal
     * Formats raw binary data to target type by calling C++.
     */
    _fmt(data: Uint8Array, type: ResultType): any;
    /**
     * @internal
     * Parse binary matrix format from C++.
     */
    private _parseMatrix;
    /**
     * Executes the C++ function.
     */
    exec(name: string, ...args: Param[]): Expr;
    /**
     * Returns a list of all exported function names.
     */
    getExportedFunctions(): any[];
}
export declare let ginac: GiNaCContext;
/**
 * Initializes the GiNaC WebAssembly module.
 * @param wasmModuleFactory - The function returned by the Emscripten script.
 */
export declare function initGiNaC(module?: any): Promise<GiNaCContext>;
export declare function sym(name: string): Symbol;
export declare function symNum(value: string | number | bigint): SymbolicNumber;
/** Computes the characteristic polynomial of a matrix. */
export declare function charpoly(matrix: Param, variable: Param): Expr;
/** Returns the coefficient of var^deg in expr. */
export declare function coeff(expr: Param, variable: Param, deg: Param): Expr;
/** Collects terms involving a variable. */
export declare function collect(expr: Param, sym: Param): Expr;
/** Collects common factors from the terms of sums. */
export declare function collect_common_factors(expr: Param): Expr;
/** Collects coefficients of a distributed polynomial. */
export declare function collect_distributed(expr: Param, syms: Param): Expr;
/** Returns the content of a polynomial (gcd of coefficients). */
export declare function content(expr: Param, variable: Param): Expr;
/** Converts Harmonic polylogarithms to Li functions. */
export declare function convert_H_to_Li(expr: Param, parameter: Param): Expr;
/** Decomposes a rational function. */
export declare function decomp_rational(expr: Param, variable: Param): Expr;
/** Returns the degree of the expression with respect to a symbol. */
export declare function degree(expr: Param, sym: Param): Expr;
/** Returns the denominator of a rational function. */
export declare function denom(expr: Param): Expr;
/** Computes the determinant of a matrix. */
export declare function determinant(matrix: Param): Expr;
/** Creates a diagonal matrix. */
export declare function diag(...args: Param[]): Expr;
/**
 * Computes the derivative.
 * Supports: diff(expr, symbol) or diff(expr, symbol, order).
 */
export declare function diff(expr: Param, symbol: Param, order?: Param): Expr;
/** Polynomial division (exact). */
export declare function divide(expr1: Param, expr2: Param): Expr;
/** Evaluates an expression numerically. */
export declare function evalf(expr: Param): Expr;
/** Evaluates sums, products and integer powers of matrices. */
export declare function evalm(expr: Param): Expr;
/** Evaluates integrals. */
export declare function eval_integ(expr: Param): Expr;
/** Expands an expression. */
export declare function expand(expr: Param): Expr;
/** Factors a polynomial. */
export declare function factor(expr: Param): Expr;
/** Finds occurrences of a pattern in an expression. */
export declare function find(expr: Param, pattern: Param): Expr;
/** Numerical root finding. fsolve(eq, var, start, end). */
export declare function fsolve(eq: Param, variable: Param, start: Param, end: Param): Expr;
/** Greatest Common Divisor. */
export declare function gcd(a: Param, b: Param): Expr;
/** Checks if expression contains a subexpression. */
export declare function has(expr: Param, pattern: Param): Expr;
/** Integer content of a polynomial. */
export declare function integer_content(expr: Param): Expr;
/**
 * Indefinite or definite integral.
 * integral(expr, var) or integral(expr, var, lower, upper).
 */
export declare function integral(variable: Param, lower: Param, upper: Param, expr: Param): Expr;
/** Inverse of a matrix. */
export declare function inverse(matrix: Param): Expr;
/** Dummy print for tab completion. */
export declare function iprint(): Expr;
/** Logic check (returns numeric 1 or 0 inside Expr). */
export declare function is(relation: Param): Expr;
/** Least Common Multiple. */
export declare function lcm(a: Param, b: Param): Expr;
/** Leading coefficient. */
export declare function lcoeff(expr: Param, sym: Param): Expr;
/** Degree of the leading term. */
export declare function ldegree(expr: Param, sym: Param): Expr;
/** Linear equation solver. */
export declare function lsolve(eqs: Param, vars: Param): Expr;
/** Map function over operands. */
export declare function map(expr: Param, funcName: Param): Expr;
/** Pattern matching. */
export declare function match(expr: Param, pattern: Param): Expr;
/** Number of operands. */
export declare function nops(expr: Param): Expr;
/** Normalizes a rational function. */
export declare function normal(expr: Param): Expr;
/** Numerator of a rational function. */
export declare function numer(expr: Param): Expr;
/** Numerator and Denominator (returns list). */
export declare function numer_denom(expr: Param): Expr;
/** Get the i-th operand. */
export declare function op(expr: Param, index: Param): Expr;
/** Power function (base^exp). */
export declare function pow(base: Param, exp: Param): Expr;
/** Pseudo-remainder. */
export declare function prem(expr1: Param, expr2: Param, sym: Param): Expr;
/** Primitive part of a polynomial. */
export declare function primpart(expr: Param, sym: Param): Expr;
export declare function print(): Expr;
export declare function print_csrc(): Expr;
export declare function print_latex(): Expr;
/** Quotient. */
export declare function quo(expr1: Param, expr2: Param, sym: Param): Expr;
/** Rank of a matrix. */
export declare function rank(matrix: Param): Expr;
/** Remainder. */
export declare function rem(expr1: Param, expr2: Param, sym: Param): Expr;
/** Resultant of two polynomials. */
export declare function resultant(expr1: Param, expr2: Param, sym: Param): Expr;
/** Series expansion. */
export declare function series(expr: Param, relation: Param, order: Param): Expr;
/** Converts a series to a polynomial. */
export declare function series_to_poly(expr: Param): Expr;
/** Sparse pseudo-remainder. */
export declare function sprem(expr1: Param, expr2: Param, sym: Param): Expr;
/**
 * Square-free factorization.
 * Supports sqrfree(expr) or sqrfree(expr, vars_list).
 */
export declare function sqrfree(expr: Param, vars?: Param): Expr;
/** Square-free partial fraction decomposition. */
export declare function sqrfree_parfrac(expr: Param, sym: Param): Expr;
/** Square root. */
export declare function sqrt(expr: Param): Expr;
/**
 * Substitution.
 * subs(expr, relation_or_list) or subs(expr, pattern, replacement).
 */
export declare function subs(expr: Param, arg2: Param, arg3?: Param): Expr;
/** Trailing coefficient. */
export declare function tcoeff(expr: Param, sym: Param): Expr;
export declare function time(): Expr;
/** Trace of a matrix. */
export declare function trace(matrix: Param): Expr;
/** Matrix transposition. */
export declare function transpose(matrix: Param): Expr;
/** Unit part. */
export declare function unit(expr: Param, sym: Param): Expr;
export declare function basic_log_kernel(): Expr;
export declare function multiple_polylog_kernel(arg: Param): Expr;
export declare function ELi_kernel(n: Param, p: Param, x: Param, y: Param): Expr;
export declare function Ebar_kernel(n: Param, p: Param, x: Param, y: Param): Expr;
export declare function Kronecker_dtau_kernel(...args: Param[]): Expr;
export declare function Kronecker_dz_kernel(...args: Param[]): Expr;
export declare function Eisenstein_kernel(...args: Param[]): Expr;
export declare function Eisenstein_h_kernel(...args: Param[]): Expr;
export declare function modular_form_kernel(...args: Param[]): Expr;
export declare function user_defined_kernel(arg1: Param, arg2: Param): Expr;
export declare function q_expansion_modular_form(arg1: Param, arg2: Param, arg3: Param): Expr;
/** Multiple polylogarithm G(a, y) or G(a, s, y). */
export declare function G(a: Param, y: Param, s?: Param): Expr;
/** Harmonic polylogarithm H(x, y). */
export declare function H(x: Param, y: Param): Expr;
/** Polylogarithm Li(n, x). */
export declare function Li(n: Param, x: Param): Expr;
/** Dilogarithm Li2(x). */
export declare function Li2(x: Param): Expr;
/** Trilogarithm Li3(x). */
export declare function Li3(x: Param): Expr;
/** Order term function. */
export declare function Order(x: Param): Expr;
/** Nielsen's generalized polylogarithm S(n, p, x). */
export declare function S(n: Param, p: Param, x: Param): Expr;
/** Absolute value. */
export declare function abs(x: Param): Expr;
/** Inverse cosine. */
export declare function acos(x: Param): Expr;
/** Inverse hyperbolic cosine. */
export declare function acosh(x: Param): Expr;
/** Inverse sine. */
export declare function asin(x: Param): Expr;
/** Inverse hyperbolic sine. */
export declare function asinh(x: Param): Expr;
/** Inverse tangent. */
export declare function atan(x: Param): Expr;
/** Inverse tangent of y/x (arctangent with two arguments). */
export declare function atan2(y: Param, x: Param): Expr;
/** Inverse hyperbolic tangent. */
export declare function atanh(x: Param): Expr;
/** Beta function. */
export declare function beta(x: Param, y: Param): Expr;
/** Binomial coefficient. */
export declare function binomial(n: Param, k: Param): Expr;
/** Complex conjugate. */
export declare function conjugate(x: Param): Expr;
/** Cosine. */
export declare function cos(x: Param): Expr;
/** Hyperbolic cosine. */
export declare function cosh(x: Param): Expr;
/** Sign of a complex number (csgn). */
export declare function csgn(x: Param): Expr;
/** Eta function. */
export declare function eta(x: Param, y: Param): Expr;
/** Exponential function. */
export declare function exp(x: Param): Expr;
/** Factorial function. */
export declare function factorial(n: Param): Expr;
/** Imaginary part of a complex number. */
export declare function imag_part(x: Param): Expr;
/** Logarithm of the Gamma function. */
export declare function lgamma(x: Param): Expr;
/** Natural logarithm. */
export declare function log(x: Param): Expr;
/** Psi function (Digamma) or Polygamma function psi(n, x). */
export declare function psi(arg0: Param, arg1?: Param): Expr;
/** Real part of a complex number. */
export declare function real_part(x: Param): Expr;
/** Sine. */
export declare function sin(x: Param): Expr;
/** Hyperbolic sine. */
export declare function sinh(x: Param): Expr;
/** Heaviside step function. */
export declare function step(x: Param): Expr;
/** Tangent. */
export declare function tan(x: Param): Expr;
/** Hyperbolic tangent. */
export declare function tanh(x: Param): Expr;
/** Gamma function. */
export declare function tgamma(x: Param): Expr;
/** Riemann Zeta function zeta(x) or Hurwitz Zeta zeta(s, x). */
export declare function zeta(arg0: Param, arg1?: Param): Expr;
/** Derivatives of the Riemann Zeta function. */
export declare function zetaderiv(n: Param, x: Param): Expr;
/**
 * Helper to create an Expr from a raw string without calculation.
 */
export declare function parse(input: string): Expr;
/**
 * Helper to create a list Expr from multiple inputs.
 */
export declare function list(...input: Param[]): Expr;
/**
 * Simplify trigonometric expressions
 */
export declare function trigsimp(input: Param): Expr;
export declare function integ(expr: Param, x: Param): Expr;
export declare function matrix(...input: Param[]): Expr;
export declare function op_add(a: Param, b: Param): Expr;
export declare function op_subtract(a: Param, b: Param): Expr;
export declare function op_multiply(a: Param, b: Param): Expr;
export declare function op_divide(a: Param, b: Param): Expr;
export declare function op_mod(a: Param, b: Param): Expr;
export declare function op_equal(a: Param, b: Param): Expr;
export declare function op_notequal(a: Param, b: Param): Expr;
export declare function op_less(a: Param, b: Param): Expr;
export declare function op_lessequal(a: Param, b: Param): Expr;
export declare function op_greater(a: Param, b: Param): Expr;
export declare function op_greaterequal(a: Param, b: Param): Expr;
export declare function op_negate(a: Param): Expr;
export declare function op_power(a: Param, b: Param): Expr;
export declare function op_factorial(a: Param): Expr;
export {};
