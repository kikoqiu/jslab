"use strict";
var bfjs = (() => {
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __export = (target, all) => {
    for (var name in all)
      __defProp(target, name, { get: all[name], enumerable: true });
  };
  var __copyProps = (to, from, except, desc) => {
    if (from && typeof from === "object" || typeof from === "function") {
      for (let key of __getOwnPropNames(from))
        if (!__hasOwnProp.call(to, key) && key !== except)
          __defProp(to, key, { get: () => from[key], enumerable: !(desc = __getOwnPropDesc(from, key)) || desc.enumerable });
    }
    return to;
  };
  var __toCommonJS = (mod) => __copyProps(__defProp({}, "__esModule", { value: true }), mod);

  // src/bf.js
  var bf_exports = {};
  __export(bf_exports, {
    BigFloat: () => BigFloat,
    BigFraction: () => BigFraction,
    Complex: () => Complex,
    Constants: () => Constants,
    E: () => E,
    Flags: () => Flags,
    O: () => O,
    PI: () => PI,
    Poly: () => Poly,
    Scalar: () => Scalar,
    SparseMatrixCSC: () => SparseMatrixCSC,
    Vector: () => Vector,
    X: () => X,
    abs: () => abs,
    acos: () => acos,
    altZeta: () => altZeta,
    asin: () => asin,
    atan: () => atan,
    atan2: () => atan2,
    bernoulli: () => bernoulli,
    besseli: () => besseli,
    besselj: () => besselj,
    besselk: () => besselk,
    bessely: () => bessely,
    beta: () => beta,
    bf: () => bf,
    ceil: () => ceil,
    clearBernoulliCache: () => clearBernoulliCache,
    complex: () => complex,
    cos: () => cos,
    decimalPrecision: () => decimalPrecision,
    diff: () => diff,
    exp: () => exp,
    factorial: () => factorial,
    floor: () => floor,
    fminbnd: () => fminbnd,
    fpround: () => fpround,
    frac: () => frac,
    fzero: () => fzero,
    gamma: () => gamma,
    gc_ele_limit: () => gc_ele_limit,
    getEpsilon: () => getEpsilon,
    getGcEleLimit: () => getGcEleLimit,
    getGlobalFlag: () => getGlobalFlag,
    getPrecision: () => getPrecision,
    globalFlag: () => globalFlag,
    half: () => half,
    hankel1: () => hankel1,
    hankel2: () => hankel2,
    hyp0f1: () => hyp0f1,
    identify: () => identify,
    init: () => init,
    integral: () => quad,
    isReady: () => isReady,
    lambertw: () => lambertw,
    libbf: () => libbf,
    limit: () => limit,
    linspace: () => linspace,
    log: () => log,
    logGamma: () => logGamma,
    max: () => max,
    min: () => min,
    minus_one: () => minus_one,
    neg: () => neg,
    nsum: () => nsum,
    ode15s: () => ode15s,
    ode45: () => ode45,
    one: () => one,
    pdepe: () => pdepe,
    poly: () => poly,
    polyStr: () => polyStr,
    polyfit: () => polyfit,
    polyval: () => polyval,
    popPrecision: () => popPrecision,
    pow: () => pow,
    precision: () => precision,
    primeZeta: () => primeZeta,
    pushDecimalPrecision: () => pushDecimalPrecision,
    pushPrecision: () => pushPrecision,
    quad: () => quad,
    romberg: () => romberg,
    roots: () => roots,
    round: () => round,
    scalar: () => scalar,
    setGcEleLimit: () => setGcEleLimit,
    setGlobalFlag: () => setGlobalFlag,
    setPrecision: () => setPrecision,
    setThrowExceptionOnInvalidOp: () => setThrowExceptionOnInvalidOp,
    shanks: () => shanks,
    sign: () => sign,
    sin: () => sin,
    solveLinearSystem: () => solveLinearSystem,
    sqrt: () => sqrt,
    tan: () => tan,
    three: () => three,
    throwExceptionOnInvalidOp: () => throwExceptionOnInvalidOp,
    trunc: () => trunc,
    two: () => two,
    zero: () => zero,
    zeta: () => zeta
  });

  // src/complex.js
  var Complex = class _Complex {
    /**
     * @param {number|string|BigFloat|Complex} re - Real part or Complex object
     * @param {number|string|BigFloat} [im=undefined] - Imaginary part
     */
    constructor(re, im) {
      if (re instanceof _Complex) {
        this.re = re.re;
        this.im = re.im;
      } else {
        this.re = bf(re);
        this.im = im === void 0 ? zero : bf(im);
      }
    }
    // --- Basic Arithmetic ---
    /**
     * Adds another complex number.
     * @param {Complex|number|string|BigFloat} other
     * @returns {Complex}
     */
    add(other) {
      const b = this._wrap(other);
      return new _Complex(this.re.add(b.re), this.im.add(b.im));
    }
    /**
     * Subtracts another complex number.
     * @param {Complex|number|string|BigFloat} other
     * @returns {Complex}
     */
    sub(other) {
      const b = this._wrap(other);
      return new _Complex(this.re.sub(b.re), this.im.sub(b.im));
    }
    /**
     * Multiplies by another complex number.
     * @param {Complex|number|string|BigFloat} other
     * @returns {Complex}
     */
    mul(other) {
      if (other instanceof BigFloat || typeof other == "number") {
        const b2 = other instanceof BigFloat ? other : bf(other);
        const ac2 = this.re.mul(b2);
        const bc2 = this.im.mul(b2);
        return new _Complex(ac2, bc2);
      }
      const b = this._wrap(other);
      const ac = this.re.mul(b.re);
      const bd = this.im.mul(b.im);
      const ad = this.re.mul(b.im);
      const bc = this.im.mul(b.re);
      return new _Complex(ac.sub(bd), ad.add(bc));
    }
    /**
     * Divides by another complex number.
     * @param {Complex|number|string|BigFloat} other
     * @returns {Complex}
     */
    div(other) {
      if (other instanceof BigFloat || typeof other == "number") {
        const b2 = other instanceof BigFloat ? other : bf(other);
        const ac2 = this.re.mul(b2);
        const denom2 = b2.mul(b2);
        ac2.setdiv(ac2, denom2);
        const bc2 = this.im.mul(b2);
        bc2.setdiv(bc2, denom2);
        return new _Complex(ac2, bc2);
      }
      const b = this._wrap(other);
      let tmpa = bf(void 0, 10, false, false), tmpb = bf(void 0, 10, false, false);
      tmpa.setmul(b.re, b.re);
      tmpb.setmul(b.im, b.im);
      let denom = bf(void 0, 10, false, false).setadd(tmpa, tmpb);
      if (denom.isZero()) {
        tmpa.dispose(false);
        tmpb.dispose(false);
        denom.dispose(false);
        throw new Error("Complex division by zero");
      }
      const ac = tmpa.setmul(this.re, b.re);
      const bd = tmpb.setmul(this.im, b.im);
      let newRe = ac.add(bd);
      newRe.setdiv(newRe, denom);
      const bc = tmpa.setmul(this.im, b.re);
      const ad = tmpb.setmul(this.re, b.im);
      let newIm = bc.sub(ad);
      newIm.setdiv(newIm, denom);
      tmpa.dispose(false);
      tmpb.dispose(false);
      denom.dispose(false);
      return new _Complex(newRe, newIm);
    }
    /**
     * Raises this complex number to the power of another.
     * @param {Complex|number|string|BigFloat} other
     * @returns {Complex}
     */
    pow(other) {
      const b = this._wrap(other);
      if (this.re.isZero() && this.im.isZero()) {
        return new _Complex(zero, zero);
      }
      return this.log().mul(b).exp();
    }
    // --- Advanced Math & Transcendental Functions ---
    /**
     * Magnitude (Absolute value) |z|
     * @returns {BigFloat}
     */
    abs() {
      let a = this.re.mul(this.re);
      let b = this.im.mul(this.im);
      b.setadd(a, b);
      a.setsqrt(b);
      return a;
    }
    /**
     * Argument (Angle) arg(z)
     * @returns {BigFloat}
     */
    arg() {
      return this.im.atan2(this.re);
    }
    /**
     * Complex Square Root sqrt(z)
     * @returns {Complex}
     */
    sqrt() {
      const r = this.abs();
      let re = r.add(this.re);
      re = re.setmul(re, half).sqrt();
      let im = r.sub(this.re);
      im = im.setmul(im, half).sqrt();
      return new _Complex(re, this.im.cmp(zero) >= 0 ? im : im.neg());
    }
    /**
     * Complex Exponential e^z
     * @returns {Complex}
     */
    exp() {
      const expRe = this.re.exp();
      return new _Complex(expRe.mul(this.im.cos()), expRe.mul(this.im.sin()));
    }
    /**
     * Complex Natural Logarithm ln(z)
     * @returns {Complex}
     */
    log() {
      const rSq = this.re.mul(this.re).add(this.im.mul(this.im));
      return new _Complex(rSq.log().mul(half), this.arg());
    }
    /**
     * Trigonometric Sine sin(z)
     * sin(x+iy) = sin(x)cosh(y) + i cos(x)sinh(y)
     * @returns {Complex}
     */
    sin() {
      const x = this.re;
      const y = this.im;
      let re = x.sin();
      re = re.setmul(re, y.cosh());
      let im = x.cos();
      im.setmul(im, y.sinh());
      return new _Complex(re, im);
    }
    /**
     * Trigonometric Cosine cos(z)
     * cos(x+iy) = cos(x)cosh(y) - i sin(x)sinh(y)
     * @returns {Complex}
     */
    cos() {
      const x = this.re;
      const y = this.im;
      let re = x.cos();
      re = re.setmul(re, y.cosh());
      let im = x.sin();
      im.setmul(im, y.sinh());
      im.setneg();
      return new _Complex(re, im);
    }
    /**
     * Trigonometric Tangent tan(z)
     * @returns {Complex}
     */
    tan() {
      return this.sin().div(this.cos());
    }
    /**
     * Hyperbolic Sine sinh(z)
     * sinh(x+iy) = sinh(x)cos(y) + i cosh(x)sin(y)
     * @returns {Complex}
     */
    sinh() {
      const x = this.re;
      const y = this.im;
      let re = x.sinh();
      re.setmul(re, y.cos());
      let im = x.cosh();
      im.setmul(im, y.sin());
      return new _Complex(re, im);
    }
    /**
     * Hyperbolic Cosine cosh(z)
     * cosh(x+iy) = cosh(x)cos(y) + i sinh(x)sin(y)
     * @returns {Complex}
     */
    cosh() {
      const x = this.re;
      const y = this.im;
      let re = x.cosh();
      re.setmul(re, y.cos());
      let im = x.sinh();
      im.setmul(im, y.sin());
      return new _Complex(re, im);
    }
    /**
     * Hyperbolic Tangent tanh(z)
     * @returns {Complex}
     */
    tanh() {
      return this.sinh().div(this.cosh());
    }
    /**
     * Inverse Sine asin(z) = -i * ln(iz + sqrt(1 - z^2))
     * @returns {Complex}
     */
    asin() {
      const i = new _Complex(zero, one2);
      const one2 = new _Complex(one2, zero);
      const iz = i.mul(this);
      const sqrtPart = one2.sub(this.mul(this)).sqrt();
      return iz.add(sqrtPart).log().mul(i.neg());
    }
    /**
     * Inverse Cosine acos(z) = -i * ln(z + i*sqrt(1 - z^2))
     * @returns {Complex}
     */
    acos() {
      const i = new _Complex(zero, one2);
      const one2 = new _Complex(one2, zero);
      const sqrtPart = one2.sub(this.mul(this)).sqrt();
      return this.add(i.mul(sqrtPart)).log().mul(i.neg());
    }
    /**
     * Inverse Tangent atan(z) = (i/2) * ln((i+z)/(i-z))
     * @returns {Complex}
     */
    atan() {
      const i = new _Complex(zero, one);
      const halfI = new _Complex(zero, half);
      const numerator = i.add(this);
      const denominator = i.sub(this);
      return numerator.div(denominator).log().mul(halfI.neg());
    }
    /**
     * Inverse Hyperbolic Sine asinh(z) = ln(z + sqrt(z^2 + 1))
     * @returns {Complex}
     */
    asinh() {
      const one2 = new _Complex(one2, zero);
      return this.add(this.mul(this).add(one2).sqrt()).log();
    }
    /**
     * Inverse Hyperbolic Cosine acosh(z) = ln(z + sqrt(z^2 - 1))
     * @returns {Complex}
     */
    acosh() {
      const one2 = new _Complex(one2, zero);
      return this.add(this.mul(this).sub(one2).sqrt()).log();
    }
    /**
     * Inverse Hyperbolic Tangent atanh(z) = 0.5 * ln((1+z)/(1-z))
     * @returns {Complex}
     */
    atanh() {
      const one2 = new _Complex(one2, zero);
      const half2 = new _Complex(half2, zero);
      return one2.add(this).div(one2.sub(this)).log().mul(half2);
    }
    // --- Utilities ---
    /**
     * Returns the complex conjugate.
     * @returns {Complex}
     */
    conj() {
      return new _Complex(this.re, this.im.neg());
    }
    /**
     * Negates the complex number.
     * @returns {Complex}
     */
    neg() {
      return new _Complex(this.re.neg(), this.im.neg());
    }
    /**
     * Checks for equality with another complex number.
     * @param {Complex|number|string|BigFloat} b
     * @returns {boolean}
     */
    equals(b) {
      const other = this._wrap(b);
      return this.re.equals(other.re) && this.im.equals(other.im);
    }
    /**
     * Checks if the complex number is almost zero.
     * @returns {boolean}
     */
    isAlmostZero() {
      return this.re.isAlmostZero() && this.im.isAlmostZero();
    }
    /**
     * Checks if the complex number is exactly zero.
     * @returns {boolean}
     */
    isZero() {
      return this.re.isZero() && this.im.isZero();
    }
    /**
     * Converts the complex number to a string.
     * @param {number} [base=10]
     * @param {number} [precision=-1]
     * @param {boolean} [pretty=false] pretty print
     * @returns {string}
     */
    toString(base = 10, precision2 = -1, pretty = false) {
      let rezero = pretty ? this.re.isAlmostZero() : this.re.isZero();
      let imzero = pretty ? this.im.isAlmostZero() : this.im.isZero();
      if (rezero && imzero) {
        return "0";
      } else if (imzero) {
        return this.re.toString(base, precision2, pretty);
      } else {
        let imabs = this.im.abs();
        let imabsf = imabs.toNumber();
        const imStr = imabsf == 1 ? "" : imabs.toString(base, precision2, pretty);
        if (rezero) {
          const sign3 = this.im.cmp(zero) < 0 ? "-" : "";
          return `${sign3}${imStr}i`;
        }
        const sign2 = this.im.cmp(zero) < 0 ? "-" : "+";
        const reStr = this.re.toString(base, precision2, pretty);
        return `(${reStr} ${sign2} ${imStr}i)`;
      }
    }
    /**
     * Wraps a value in a Complex object if it isn't one already.
     * @private
     * @param {Complex|number|string|BigFloat} v
     * @returns {Complex}
     */
    _wrap(v) {
      if (v === void 0) {
        throw new Error("Operand mismatch");
      }
      if (v instanceof _Complex) return v;
      return new _Complex(v);
    }
    /**
     * Creates a complex number from polar coordinates.
     * @param {number|string|BigFloat} r - The radius.
     * @param {number|string|BigFloat} theta - The angle.
     * @returns {Complex}
     */
    static fromPolar(r, theta) {
      const R = bf(r);
      const T = bf(theta);
      return new _Complex(R.mul(T.cos()), R.mul(T.sin()));
    }
    /**
     * Creates a complex number from a string.
     * @param {string} s
     * @returns {Complex}
     */
    static fromString(s) {
      s = s.trim();
      s = s.replace(/I/g, "i");
      if (s === "i") return new _Complex(0, 1);
      if (s === "-i") return new _Complex(0, -1);
      const match = s.match(/^(.+?)([+-].*|(?=i))i$/);
      if (match) {
        const re = match[1] === "" ? 0 : match[1];
        let im = match[2];
        if (im === "+" || im === "") im = 1;
        if (im === "-") im = -1;
        return new _Complex(re, im);
      }
      if (s.endsWith("i")) {
        const im = s.slice(0, -1);
        return new _Complex(0, im === "" ? 1 : im === "-" ? -1 : im);
      }
      return new _Complex(BigFloat.fromString(s));
    }
  };
  Complex.prototype.operatorAdd = Complex.prototype.add;
  Complex.prototype.operatorSub = Complex.prototype.sub;
  Complex.prototype.operatorMul = Complex.prototype.mul;
  Complex.prototype.operatorDiv = Complex.prototype.div;
  Complex.prototype.operatorPow = Complex.prototype.pow;
  Complex.prototype.operatorNeg = Complex.prototype.neg;
  function complex(re, im) {
    return new Complex(re, im);
  }

  // src/vector.js
  var Vector = class _Vector {
    /**
     * Initializes a Vector from a size, an existing array, or another Vector.
     * @param {number|Array<number|string|BigFloat>|Vector} data 
     */
    constructor(data) {
      if (typeof data === "number") {
        this.values = new Array(data).fill(zero);
      } else if (Array.isArray(data)) {
        this.values = data.map((v) => v instanceof BigFloat ? v : bf(v));
      } else if (data instanceof _Vector) {
        this.values = [...data.values];
      } else {
        throw new Error("Vector must be initialized with a size, an array, or another Vector.");
      }
    }
    /**
     * Returns a string representation of the vector.
     * Truncates the middle part with "..." if the length exceeds maxItem.
     * 
     * @param {number} radix - The base for number representation (e.g., 10).
     * @param {number} prec - The number of decimal places for BigFloat.
     * @param {number} maxItem - Maximum number of elements to show before truncating.
     * @returns {string}
     */
    toString(radix = 10, prec = 2, maxItem = 10) {
      const len = this.length;
      const parts = [];
      const format = (i) => this.get(i).toString(radix, prec);
      if (len <= maxItem) {
        for (let i = 0; i < len; i++) {
          parts.push(format(i));
        }
      } else {
        const half2 = Math.floor(maxItem / 2);
        for (let i = 0; i < half2; i++) {
          parts.push(format(i));
        }
        parts.push("...");
        const endCount = maxItem - half2;
        for (let i = len - endCount; i < len; i++) {
          parts.push(format(i));
        }
      }
      return `Vector (length=${len}): [ ${parts.join(", ")} ]`;
    }
    /**
     * Dimension of the vector.
     * @returns {number}
     */
    get length() {
      return this.values.length;
    }
    /**
     * Gets the value at index i.
     * @param {number} i 
     * @returns {BigFloat}
     */
    get(i) {
      return this.values[i];
    }
    /**
     * Sets the value at index i.
     * @param {number} i 
     * @param {number|string|BigFloat} val 
     */
    set(i, val) {
      this.values[i] = val instanceof BigFloat ? val : bf(val);
    }
    /**
     * Adds another vector to this vector (v = this + other).
     * @param {Vector} other 
     * @returns {Vector}
     */
    add(other) {
      if (this.length !== other.length) throw new Error("Vector dimension mismatch for addition.");
      const result = new _Vector(this.length);
      for (let i = 0; i < this.length; i++) {
        result.values[i] = this.values[i].add(other.values[i]);
      }
      return result;
    }
    /**
     * Subtracts another vector from this vector (v = this - other).
     * @param {Vector} other 
     * @returns {Vector}
     */
    sub(other) {
      if (this.length !== other.length) throw new Error("Vector dimension mismatch for subtraction.");
      const result = new _Vector(this.length);
      for (let i = 0; i < this.length; i++) {
        result.values[i] = this.values[i].sub(other.values[i]);
      }
      return result;
    }
    /**
     * Multiplies the vector by a scalar (v = this * scalar).
     * @param {number|string|BigFloat} scalar 
     * @returns {Vector}
     */
    scale(scalar2) {
      const s = scalar2 instanceof BigFloat ? scalar2 : bf(scalar2);
      const result = new _Vector(this.length);
      for (let i = 0; i < this.length; i++) {
        result.values[i] = this.values[i].mul(s);
      }
      return result;
    }
    /**
     * Computes the dot product of this vector and another vector.
     * @param {Vector} other 
     * @returns {BigFloat}
     */
    dot(other) {
      if (this.length !== other.length) throw new Error("Vector dimension mismatch for dot product.");
      let sum = zero;
      for (let i = 0; i < this.length; i++) {
        sum = sum.add(this.values[i].mul(other.values[i]));
      }
      return sum;
    }
    /**
     * Computes the L2 norm (Euclidean norm) of the vector.
     * @returns {BigFloat}
     */
    norm() {
      return this.dot(this).sqrt();
    }
    /**
     * Converts the vector to a standard Javascript Array of BigFloats.
     * @returns {BigFloat[]}
     */
    toArray() {
      return [...this.values];
    }
    /**
     * Deep clones the vector.
     * @returns {Vector}
     */
    clone() {
      return new _Vector(this);
    }
  };

  // src/matrix.js
  var SparseMatrixCSC = class _SparseMatrixCSC {
    /**
     * @param {number} rows - Number of rows.
     * @param {number} cols - Number of columns.
     * @param {BigFloat[]} values - Array of non-zero BigFloat values.
     * @param {Uint32Array|number[]} rowIndices - Row indices for each non-zero value.
     * @param {Uint32Array|number[]} colPointers - Column pointers of length (cols + 1).
     */
    constructor(rows, cols, values, rowIndices, colPointers) {
      this.rows = rows;
      this.cols = cols;
      this.values = values;
      this.rowIndices = rowIndices instanceof Uint32Array ? rowIndices : new Uint32Array(rowIndices);
      this.colPointers = colPointers instanceof Uint32Array ? colPointers : new Uint32Array(colPointers);
      if (this.colPointers.length !== this.cols + 1) {
        throw new Error(`colPointers length must be cols + 1 (${this.cols + 1}), got ${this.colPointers.length}`);
      }
      if (this.values.length !== this.rowIndices.length) {
        throw new Error("values and rowIndices must have the same length.");
      }
      if (this.colPointers[this.cols] !== this.values.length) {
        throw new Error("The last element of colPointers must equal the number of non-zero elements (nnz).");
      }
    }
    /**
     * Returns a string representation of the matrix.
     * Uses the existing `get(row, col)` method and `nnz` property.
     * 
     * @param {number} radix - The base for number representation (e.g., 10).
     * @param {number} prec - The number of decimal places for BigFloat.
     * @param {number} maxRowItem - Maximum number of rows/cols to show before truncating with "...".
     * @returns {string}
     */
    toString(radix = 10, prec = 2, maxRowItem = 10) {
      const getDisplayLayout = (total) => {
        if (total <= maxRowItem) {
          return {
            indices: Array.from({ length: total }, (_, i) => i),
            isTruncated: false
          };
        }
        const half2 = Math.floor(maxRowItem / 2);
        return {
          start: Array.from({ length: half2 }, (_, i) => i),
          end: Array.from({ length: maxRowItem - half2 }, (_, i) => total - (maxRowItem - half2) + i),
          isTruncated: true
        };
      };
      const rowLayout = getDisplayLayout(this.rows);
      const colLayout = getDisplayLayout(this.cols);
      const formatRow = (r) => {
        const cells = [];
        if (!colLayout.isTruncated) {
          colLayout.indices.forEach((c) => cells.push(this.get(r, c).toString(radix, prec)));
        } else {
          colLayout.start.forEach((c) => cells.push(this.get(r, c).toString(radix, prec)));
          cells.push("...");
          colLayout.end.forEach((c) => cells.push(this.get(r, c).toString(radix, prec)));
        }
        return `[ ${cells.join(", ")} ]`;
      };
      const output = [];
      output.push(`SparseMatrixCSC (${this.rows}x${this.cols}, nnz=${this.nnz}):`);
      if (!rowLayout.isTruncated) {
        rowLayout.indices.forEach((r) => output.push(formatRow(r)));
      } else {
        rowLayout.start.forEach((r) => output.push(formatRow(r)));
        const colCount = colLayout.isTruncated ? maxRowItem + 1 : colLayout.indices.length;
        const verticalEllipsis = new Array(colCount).fill("...").join("  ");
        output.push(`  ${verticalEllipsis}  `);
        rowLayout.end.forEach((r) => output.push(formatRow(r)));
      }
      return output.join("\n");
    }
    /**
     * Number of non-zero elements.
     * @returns {number}
     */
    get nnz() {
      return this.values.length;
    }
    /**
     * Retrieves the value at the specified row and column.
     * Uses binary search within the specific column for O(log(nnz_in_col)) performance.
     * 
     * @param {number} row 
     * @param {number} col 
     * @returns {BigFloat}
     */
    get(row, col) {
      if (row < 0 || row >= this.rows || col < 0 || col >= this.cols) {
        throw new Error("Matrix indices out of bounds.");
      }
      const start = this.colPointers[col];
      const end = this.colPointers[col + 1];
      let low = start;
      let high = end - 1;
      while (low <= high) {
        const mid = low + high >>> 1;
        const r = this.rowIndices[mid];
        if (r === row) return this.values[mid];
        if (r < row) low = mid + 1;
        else high = mid - 1;
      }
      return zero;
    }
    /**
     * Sets the value at the specified row and column.
     * Warning: Modifying the structure of a CSC matrix is O(nnz). 
     * If you are building a matrix, it is highly recommended to use `fromCOO` instead.
     * 
     * @param {number} row 
     * @param {number} col 
     * @param {number|string|BigFloat} val 
     */
    set(row, col, val) {
      if (row < 0 || row >= this.rows || col < 0 || col >= this.cols) {
        throw new Error("Matrix indices out of bounds.");
      }
      const value = val instanceof BigFloat ? val : bf(val);
      const start = this.colPointers[col];
      const end = this.colPointers[col + 1];
      let low = start;
      let high = end - 1;
      let found = false;
      let insertPos = start;
      while (low <= high) {
        const mid = low + high >>> 1;
        const r = this.rowIndices[mid];
        if (r === row) {
          this.values[mid] = value;
          found = true;
          break;
        }
        if (r < row) {
          low = mid + 1;
          insertPos = low;
        } else {
          high = mid - 1;
          insertPos = mid;
        }
      }
      if (!found) {
        if (value.isZero()) return;
        const newNnz = this.nnz + 1;
        const newValues = new Array(newNnz);
        const newRowIndices = new Uint32Array(newNnz);
        for (let i = 0; i < insertPos; i++) {
          newValues[i] = this.values[i];
          newRowIndices[i] = this.rowIndices[i];
        }
        newValues[insertPos] = value;
        newRowIndices[insertPos] = row;
        for (let i = insertPos; i < this.nnz; i++) {
          newValues[i + 1] = this.values[i];
          newRowIndices[i + 1] = this.rowIndices[i];
        }
        for (let j = col + 1; j <= this.cols; j++) {
          this.colPointers[j]++;
        }
        this.values = newValues;
        this.rowIndices = newRowIndices;
      }
    }
    /**
     * Eliminates explicit structural zeros from the matrix.
     * Sparse operations might leave explicit zeros to avoid shifting arrays.
     * Call this method to compact the matrix memory.
     */
    prune() {
      let dest = 0;
      const newColPointers = new Uint32Array(this.cols + 1);
      newColPointers[0] = 0;
      for (let j = 0; j < this.cols; j++) {
        const start = this.colPointers[j];
        const end = this.colPointers[j + 1];
        for (let i = start; i < end; i++) {
          const val = this.values[i];
          if (!val.isZero()) {
            this.values[dest] = val;
            this.rowIndices[dest] = this.rowIndices[i];
            dest++;
          }
        }
        newColPointers[j + 1] = dest;
      }
      this.values.length = dest;
      this.rowIndices = this.rowIndices.slice(0, dest);
      this.colPointers = newColPointers;
    }
    /**
     * Transposes the matrix. 
     * Converts an M x N CSC matrix to an N x M CSC matrix.
     * This algorithm executes in O(nnz + max(rows, cols)) time.
     * 
     * @returns {SparseMatrixCSC}
     */
    transpose() {
      const newRows = this.cols;
      const newCols = this.rows;
      const nnz = this.nnz;
      const transposedValues = new Array(nnz);
      const transposedRowIndices = new Uint32Array(nnz);
      const transposedColPointers = new Uint32Array(newCols + 1);
      const rowCounts = new Uint32Array(newCols);
      for (let i = 0; i < nnz; i++) {
        rowCounts[this.rowIndices[i]]++;
      }
      transposedColPointers[0] = 0;
      for (let i = 0; i < newCols; i++) {
        transposedColPointers[i + 1] = transposedColPointers[i] + rowCounts[i];
      }
      const currentOffsets = new Uint32Array(transposedColPointers);
      for (let j = 0; j < this.cols; j++) {
        const start = this.colPointers[j];
        const end = this.colPointers[j + 1];
        for (let p = start; p < end; p++) {
          const r = this.rowIndices[p];
          const dest = currentOffsets[r]++;
          transposedRowIndices[dest] = j;
          transposedValues[dest] = this.values[p];
        }
      }
      return new _SparseMatrixCSC(
        newRows,
        newCols,
        transposedValues,
        transposedRowIndices,
        transposedColPointers
      );
    }
    /**
     * Creates a deep copy of the matrix.
     * @returns {SparseMatrixCSC}
     */
    clone() {
      return new _SparseMatrixCSC(
        this.rows,
        this.cols,
        [...this.values],
        // Deep enough since BigFloat is immutable mostly, or we map to clone
        new Uint32Array(this.rowIndices),
        new Uint32Array(this.colPointers)
      );
    }
    // --- Format Conversions ---
    /**
     * Creates a CSC matrix from Coordinate (COO) / Triplet format.
     * This is the recommended way to build a sparse matrix.
     * Duplicates are automatically summed.
     * 
     * @param {number} rows 
     * @param {number} cols 
     * @param {number[]} rowIdx - Array of row coordinates.
     * @param {number[]} colIdx - Array of column coordinates.
     * @param {(number|string|BigFloat)[]} vals - Array of values.
     * @returns {SparseMatrixCSC}
     */
    static fromCOO(rows, cols, rowIdx, colIdx, vals) {
      if (rowIdx.length !== colIdx.length || colIdx.length !== vals.length) {
        throw new Error("COO arrays must be of the same length.");
      }
      const n = vals.length;
      const perm = new Uint32Array(n);
      for (let i = 0; i < n; i++) perm[i] = i;
      perm.sort((a, b) => {
        if (colIdx[a] !== colIdx[b]) return colIdx[a] - colIdx[b];
        return rowIdx[a] - rowIdx[b];
      });
      const values = [];
      const rowIndices = [];
      const colPointers = new Uint32Array(cols + 1);
      colPointers.fill(0);
      let lastRow = -1;
      let lastCol = -1;
      for (let i = 0; i < n; i++) {
        const idx = perm[i];
        const r = rowIdx[idx];
        const c = colIdx[idx];
        let v = vals[idx];
        v = v instanceof BigFloat ? v : bf(v);
        if (v.isZero()) continue;
        if (c === lastCol && r === lastRow) {
          values[values.length - 1] = values[values.length - 1].add(v);
        } else {
          values.push(v);
          rowIndices.push(r);
          colPointers[c + 1]++;
          lastRow = r;
          lastCol = c;
        }
      }
      for (let i = 0; i < cols; i++) {
        colPointers[i + 1] += colPointers[i];
      }
      return new _SparseMatrixCSC(rows, cols, values, new Uint32Array(rowIndices), colPointers);
    }
    /**
     * Converts a Dense Matrix (2D Array) into a Sparse CSC Matrix.
     * @param {(number|string|BigFloat)[][]} matrix 
     * @returns {SparseMatrixCSC}
     */
    static fromDense(matrix) {
      const rows = matrix.length;
      const cols = rows > 0 ? matrix[0].length : 0;
      const rowIdx = [];
      const colIdx = [];
      const vals = [];
      for (let i = 0; i < rows; i++) {
        for (let j = 0; j < cols; j++) {
          let v = matrix[i][j];
          v = v instanceof BigFloat ? v : bf(v);
          if (!v.isZero()) {
            rowIdx.push(i);
            colIdx.push(j);
            vals.push(v);
          }
        }
      }
      return _SparseMatrixCSC.fromCOO(rows, cols, rowIdx, colIdx, vals);
    }
    /**
     * Converts the CSC Sparse Matrix back to a Dense Matrix (2D Array of BigFloat).
     * @returns {BigFloat[][]}
     */
    toDense() {
      const dense = new Array(this.rows);
      for (let i = 0; i < this.rows; i++) {
        dense[i] = new Array(this.cols).fill(zero);
      }
      for (let j = 0; j < this.cols; j++) {
        const start = this.colPointers[j];
        const end = this.colPointers[j + 1];
        for (let p = start; p < end; p++) {
          dense[this.rowIndices[p]][j] = this.values[p];
        }
      }
      return dense;
    }
    /**
     * Internal implementation for Matrix Addition and Subtraction.
     * Utilizes a highly optimized Two-Pointer Merge algorithm, running in O(nnz(A) + nnz(B)) time.
     * 
     * @private
     * @param {SparseMatrixCSC} other - The other matrix.
     * @param {boolean} isSub - True if subtraction (A - B), False if addition (A + B).
     * @returns {SparseMatrixCSC}
     */
    _addSub(other, isSub) {
      if (this.rows !== other.rows || this.cols !== other.cols) {
        throw new Error("Matrix dimensions must match for addition/subtraction.");
      }
      const cols = this.cols;
      const maxNnz = this.nnz + other.nnz;
      const newColPointers = new Uint32Array(cols + 1);
      const tempRowIndices = new Uint32Array(maxNnz);
      const tempValues = new Array(maxNnz);
      let dest = 0;
      for (let j = 0; j < cols; j++) {
        newColPointers[j] = dest;
        let pA = this.colPointers[j];
        const endA = this.colPointers[j + 1];
        let pB = other.colPointers[j];
        const endB = other.colPointers[j + 1];
        while (pA < endA && pB < endB) {
          const rA = this.rowIndices[pA];
          const rB = other.rowIndices[pB];
          if (rA < rB) {
            tempRowIndices[dest] = rA;
            tempValues[dest] = this.values[pA];
            pA++;
            dest++;
          } else if (rA > rB) {
            tempRowIndices[dest] = rB;
            tempValues[dest] = isSub ? other.values[pB].neg() : other.values[pB];
            pB++;
            dest++;
          } else {
            const valA = this.values[pA];
            const valB = other.values[pB];
            const resultVal = isSub ? valA.sub(valB) : valA.add(valB);
            if (!resultVal.isZero()) {
              tempRowIndices[dest] = rA;
              tempValues[dest] = resultVal;
              dest++;
            }
            pA++;
            pB++;
          }
        }
        while (pA < endA) {
          tempRowIndices[dest] = this.rowIndices[pA];
          tempValues[dest] = this.values[pA];
          pA++;
          dest++;
        }
        while (pB < endB) {
          tempRowIndices[dest] = other.rowIndices[pB];
          tempValues[dest] = isSub ? other.values[pB].neg() : other.values[pB];
          pB++;
          dest++;
        }
      }
      newColPointers[cols] = dest;
      return new _SparseMatrixCSC(
        this.rows,
        cols,
        tempValues.slice(0, dest),
        tempRowIndices.slice(0, dest),
        newColPointers
      );
    }
    /**
     * Adds another SparseMatrixCSC to this matrix (C = A + B).
     * @param {SparseMatrixCSC} other 
     * @returns {SparseMatrixCSC}
     */
    add(other) {
      return this._addSub(other, false);
    }
    /**
     * Subtracts another SparseMatrixCSC from this matrix (C = A - B).
     * @param {SparseMatrixCSC} other 
     * @returns {SparseMatrixCSC}
     */
    sub(other) {
      return this._addSub(other, true);
    }
    /**
     * Matrix-Matrix Multiplication (C = A * B).
     * Implements Gustavson's Algorithm (Sparse Accumulator variant).
     * Computes the product in O(flops + nnz(C)) time footprint with zero inner-loop allocation.
     * 
     * @param {SparseMatrixCSC} B - The right-hand side matrix.
     * @returns {SparseMatrixCSC}
     */
    mul(B) {
      if (this.cols !== B.rows) {
        throw new Error(`Dimension mismatch: Left matrix cols (${this.cols}) must match right matrix rows (${B.rows}).`);
      }
      const M = this.rows;
      const K = this.cols;
      const N = B.cols;
      const resultValues = [];
      const resultRowIndices = [];
      const resultColPointers = new Uint32Array(N + 1);
      const x = new Array(M);
      const marker = new Int32Array(M);
      marker.fill(-1);
      const activeRows = new Int32Array(M);
      for (let j = 0; j < N; j++) {
        resultColPointers[j] = resultValues.length;
        let activeCount = 0;
        const B_start = B.colPointers[j];
        const B_end = B.colPointers[j + 1];
        for (let p = B_start; p < B_end; p++) {
          const k = B.rowIndices[p];
          const b_kj = B.values[p];
          const A_start = this.colPointers[k];
          const A_end = this.colPointers[k + 1];
          for (let q = A_start; q < A_end; q++) {
            const i = this.rowIndices[q];
            const a_ik = this.values[q];
            const prod = a_ik.mul(b_kj);
            if (marker[i] !== j) {
              marker[i] = j;
              x[i] = prod;
              activeRows[activeCount++] = i;
            } else {
              x[i] = x[i].add(prod);
            }
          }
        }
        const colRows = activeRows.subarray(0, activeCount);
        colRows.sort();
        for (let idx = 0; idx < activeCount; idx++) {
          const r = colRows[idx];
          const val = x[r];
          if (!val.isZero()) {
            resultValues.push(val);
            resultRowIndices.push(r);
          }
        }
      }
      resultColPointers[N] = resultValues.length;
      return new _SparseMatrixCSC(
        M,
        N,
        resultValues,
        new Uint32Array(resultRowIndices),
        resultColPointers
      );
    }
    /**
     * Matrix-Vector Multiplication (y = A * x).
     * Linear time execution: O(nnz(A)).
     * 
     * @param {Vector|Array<number|string|BigFloat>} vec - The input vector.
     * @returns {Vector} - The result vector.
     */
    mulVec(vec) {
      let vArray;
      if (vec instanceof Vector) {
        vArray = vec.values;
      } else if (Array.isArray(vec)) {
        vArray = vec.map((val) => val instanceof BigFloat ? val : bf(val));
      } else {
        throw new Error("Input must be a Vector instance or an Array.");
      }
      if (this.cols !== vArray.length) {
        throw new Error(`Dimension mismatch: Matrix columns (${this.cols}) must match vector length (${vArray.length}).`);
      }
      const result = new Array(this.rows).fill(zero);
      for (let j = 0; j < this.cols; j++) {
        const xj = vArray[j];
        if (xj.isZero()) continue;
        const start = this.colPointers[j];
        const end = this.colPointers[j + 1];
        for (let p = start; p < end; p++) {
          const i = this.rowIndices[p];
          const a_ij = this.values[p];
          result[i] = result[i].add(a_ij.mul(xj));
        }
      }
      const outputVec = new Vector(this.rows);
      outputVec.values = result;
      return outputVec;
    }
    /**
     * Extracts the diagonal elements of the matrix.
     * @returns {Vector} - A vector containing the diagonal elements.
     */
    getDiagonal() {
      const minDim = Math.min(this.rows, this.cols);
      const diag = new Vector(minDim);
      for (let j = 0; j < minDim; j++) {
        const start = this.colPointers[j];
        const end = this.colPointers[j + 1];
        let low = start;
        let high = end - 1;
        while (low <= high) {
          const mid = low + high >>> 1;
          const r = this.rowIndices[mid];
          if (r === j) {
            diag.values[j] = this.values[mid];
            break;
          }
          if (r < j) low = mid + 1;
          else high = mid - 1;
        }
      }
      return diag;
    }
    /**
     * Computes the Trace of the matrix (sum of diagonal elements).
     * @returns {BigFloat}
     */
    trace() {
      let sum = zero;
      const minDim = Math.min(this.rows, this.cols);
      for (let j = 0; j < minDim; j++) {
        const val = this.get(j, j);
        if (!val.isZero()) sum = sum.add(val);
      }
      return sum;
    }
    /**
     * Computes the L1 Norm (Maximum absolute column sum).
     * Executes in O(nnz) time.
     * @returns {BigFloat}
     */
    norm1() {
      let maxNorm = zero;
      for (let j = 0; j < this.cols; j++) {
        let colSum = zero;
        const start = this.colPointers[j];
        const end = this.colPointers[j + 1];
        for (let p = start; p < end; p++) {
          colSum = colSum.add(this.values[p].abs());
        }
        if (colSum.cmp(maxNorm) > 0) maxNorm = colSum;
      }
      return maxNorm;
    }
    /**
     * Computes the Infinity Norm (Maximum absolute row sum).
     * Executes in O(nnz) time footprint.
     * @returns {BigFloat}
     */
    normInf() {
      const rowSums = new Array(this.rows).fill(zero);
      for (let j = 0; j < this.cols; j++) {
        const start = this.colPointers[j];
        const end = this.colPointers[j + 1];
        for (let p = start; p < end; p++) {
          const r = this.rowIndices[p];
          rowSums[r] = rowSums[r].add(this.values[p].abs());
        }
      }
      let maxNorm = zero;
      for (let i = 0; i < this.rows; i++) {
        if (rowSums[i].cmp(maxNorm) > 0) maxNorm = rowSums[i];
      }
      return maxNorm;
    }
    /**
     * Computes the Frobenius Norm (Square root of the sum of the squares of elements).
     * @returns {BigFloat}
     */
    normF() {
      let sumSq = zero;
      for (let i = 0; i < this.nnz; i++) {
        const val = this.values[i];
        sumSq = sumSq.add(val.mul(val));
      }
      return sumSq.sqrt();
    }
    // --- Direct Solvers for Triangular Matrices ---
    /**
     * Forward Substitution to solve L * x = b.
     * Assumes this matrix is strictly a Lower Triangular matrix.
     * Column-Oriented approach for CSC layout. O(nnz) time.
     * 
     * @param {Vector|Array<number|string|BigFloat>} b - The right-hand side vector.
     * @returns {Vector} x - The solution vector.
     */
    solveLowerTriangular(b) {
      if (this.rows !== this.cols) throw new Error("Matrix must be square.");
      const n = this.rows;
      const bVec = b instanceof Vector ? b : new Vector(b);
      if (bVec.length !== n) throw new Error("Dimension mismatch.");
      const x = bVec.toArray();
      for (let j = 0; j < n; j++) {
        if (x[j].isZero()) continue;
        const start = this.colPointers[j];
        const end = this.colPointers[j + 1];
        let diagVal = zero;
        for (let p = start; p < end; p++) {
          const r = this.rowIndices[p];
          if (r === j) {
            diagVal = this.values[p];
          } else if (r > j) {
            continue;
          }
        }
        if (diagVal.isZero()) throw new Error(`Singular matrix: zero diagonal at column ${j}`);
        x[j] = x[j].div(diagVal);
        const xj = x[j];
        for (let p = start; p < end; p++) {
          const r = this.rowIndices[p];
          if (r > j) {
            x[r] = x[r].sub(this.values[p].mul(xj));
          }
        }
      }
      const res = new Vector(n);
      res.values = x;
      return res;
    }
    /**
     * Backward Substitution to solve U * x = b.
     * Assumes this matrix is strictly an Upper Triangular matrix.
     * Column-Oriented approach for CSC layout. O(nnz) time.
     * 
     * @param {Vector|Array<number|string|BigFloat>} b - The right-hand side vector.
     * @returns {Vector} x - The solution vector.
     */
    solveUpperTriangular(b) {
      if (this.rows !== this.cols) throw new Error("Matrix must be square.");
      const n = this.rows;
      const bVec = b instanceof Vector ? b : new Vector(b);
      if (bVec.length !== n) throw new Error("Dimension mismatch.");
      const x = bVec.toArray();
      for (let j = n - 1; j >= 0; j--) {
        if (x[j].isZero()) continue;
        const start = this.colPointers[j];
        const end = this.colPointers[j + 1];
        let diagVal = zero;
        for (let p = start; p < end; p++) {
          const r = this.rowIndices[p];
          if (r === j) {
            diagVal = this.values[p];
          }
        }
        if (diagVal.isZero()) throw new Error(`Singular matrix: zero diagonal at column ${j}`);
        x[j] = x[j].div(diagVal);
        const xj = x[j];
        for (let p = start; p < end; p++) {
          const r = this.rowIndices[p];
          if (r < j) {
            x[r] = x[r].sub(this.values[p].mul(xj));
          }
        }
      }
      const res = new Vector(n);
      res.values = x;
      return res;
    }
    // --- Iterative Solvers ---
    /**
     * Conjugate Gradient (CG) Method.
     * Solves the linear system A * x = b for Symmetric Positive Definite (SPD) matrices.
     * 
     * @param {Vector|Array<number|string|BigFloat>} b - The right-hand side vector.
     * @param {number|string|BigFloat}[tol="1e-20"] - Convergence tolerance.
     * @param {number} [maxIter] - Maximum number of iterations. Defaults to matrix dimension.
     * @returns {Vector} x - The estimated solution vector.
     */
    solveCG(b, tol = "1e-20", maxIter = this.cols) {
      if (this.rows !== this.cols) throw new Error("Matrix must be square for CG.");
      const n = this.rows;
      const bVec = b instanceof Vector ? b : new Vector(b);
      const tolerance = tol instanceof BigFloat ? tol : bf(tol);
      const tolSq = tolerance.mul(tolerance);
      const xVals = new Array(n).fill(zero);
      const rVals = bVec.toArray();
      const pVals = [...rVals];
      const ApVals = new Array(n);
      let rsold = zero;
      for (let i = 0; i < n; i++) {
        rsold = rsold.add(rVals[i].mul(rVals[i]));
      }
      for (let iter = 0; iter < maxIter; iter++) {
        if (rsold.cmp(tolSq) <= 0) break;
        ApVals.fill(zero);
        for (let j = 0; j < n; j++) {
          const pj = pVals[j];
          if (pj.isZero()) continue;
          const start = this.colPointers[j];
          const end = this.colPointers[j + 1];
          for (let k = start; k < end; k++) {
            const row = this.rowIndices[k];
            ApVals[row] = ApVals[row].add(this.values[k].mul(pj));
          }
        }
        let pAp = zero;
        for (let i = 0; i < n; i++) {
          pAp = pAp.add(pVals[i].mul(ApVals[i]));
        }
        if (pAp.isZero()) break;
        const alpha = rsold.div(pAp);
        let rsnew = zero;
        for (let i = 0; i < n; i++) {
          xVals[i] = xVals[i].add(alpha.mul(pVals[i]));
          rVals[i] = rVals[i].sub(alpha.mul(ApVals[i]));
          rsnew = rsnew.add(rVals[i].mul(rVals[i]));
        }
        if (rsnew.cmp(tolSq) <= 0) break;
        const beta2 = rsnew.div(rsold);
        for (let i = 0; i < n; i++) {
          pVals[i] = rVals[i].add(beta2.mul(pVals[i]));
        }
        rsold = rsnew;
      }
      const result = new Vector(n);
      result.values = xVals;
      return result;
    }
    /**
     * Extracts the Jacobi Preconditioner (Inverse of the Diagonal).
     * This is the most memory-efficient and widely used preconditioner for diagonally dominant matrices.
     * 
     * @returns {Vector} - A vector representing the diagonal inverse M^{-1}.
     */
    getJacobiPreconditioner() {
      const n = Math.min(this.rows, this.cols);
      const invDiag = new Vector(n);
      for (let j = 0; j < n; j++) {
        const start = this.colPointers[j];
        const end = this.colPointers[j + 1];
        let diagVal = zero;
        for (let p = start; p < end; p++) {
          if (this.rowIndices[p] === j) {
            diagVal = this.values[p];
            break;
          }
        }
        if (diagVal.isZero()) {
          invDiag.values[j] = one;
        } else {
          invDiag.values[j] = one.div(diagVal);
        }
      }
      return invDiag;
    }
    /**
     * Bi-Conjugate Gradient Stabilized (BiCGSTAB) Method.
     * Solves the linear system A * x = b for non-symmetric square matrices.
     * 
     * - Fixed memory footprint (O(N) aux vectors, completely avoids GMRES memory explosion).
     * - GC-Pause Elimination: Completely pre-allocated functional closures for array vectors.
     * 
     * @param {Vector|Array<number|string|BigFloat>} b - The right-hand side vector.
     * @param {number|string|BigFloat} [tol="1e-20"] - Convergence tolerance.
     * @param {number}[maxIter] - Maximum number of iterations. Defaults to 2 * matrix dimension.
     * @param {Vector}[precond] - Optional Jacobi preconditioner vector (M^{-1}).
     * @returns {Vector} x - The estimated solution vector.
     */
    solveBiCGSTAB(b, tol = "1e-20", maxIter = this.cols * 2, precond = null) {
      if (this.rows !== this.cols) throw new Error("Matrix must be square for BiCGSTAB.");
      const n = this.rows;
      const bVec = b instanceof Vector ? b : new Vector(b);
      const tolerance = tol instanceof BigFloat ? tol : bf(tol);
      const tolSq = tolerance.mul(tolerance);
      const x = new Array(n).fill(zero);
      const r = bVec.toArray();
      const r_hat = [...r];
      const p = new Array(n).fill(zero);
      const v = new Array(n).fill(zero);
      const s = new Array(n).fill(zero);
      const t = new Array(n).fill(zero);
      let rho_prev = one;
      let alpha = one;
      let omega = one;
      const spmv = (inArr, outArr) => {
        outArr.fill(zero);
        for (let j = 0; j < n; j++) {
          const xj = inArr[j];
          if (xj.isZero()) continue;
          const start = this.colPointers[j];
          const end = this.colPointers[j + 1];
          for (let k = start; k < end; k++) {
            const i = this.rowIndices[k];
            outArr[i] = outArr[i].add(this.values[k].mul(xj));
          }
        }
      };
      const applyPrecond = (inArr, outArr) => {
        if (!precond) {
          for (let i = 0; i < n; i++) outArr[i] = inArr[i];
        } else {
          for (let i = 0; i < n; i++) outArr[i] = inArr[i].mul(precond.values[i]);
        }
      };
      const dot = (arrA, arrB) => {
        let sum = zero;
        for (let i = 0; i < n; i++) sum = sum.add(arrA[i].mul(arrB[i]));
        return sum;
      };
      const tempArr1 = new Array(n).fill(zero);
      const tempArr2 = new Array(n).fill(zero);
      for (let iter = 0; iter < maxIter; iter++) {
        const r_norm_sq = dot(r, r);
        if (r_norm_sq.cmp(tolSq) <= 0) break;
        const rho = dot(r_hat, r);
        if (rho.isZero()) break;
        if (iter > 0) {
          const beta2 = rho.div(rho_prev).mul(alpha).div(omega);
          for (let i = 0; i < n; i++) {
            const p_minus_omega_v = p[i].sub(omega.mul(v[i]));
            p[i] = r[i].add(beta2.mul(p_minus_omega_v));
          }
        } else {
          for (let i = 0; i < n; i++) p[i] = r[i];
        }
        applyPrecond(p, tempArr1);
        spmv(tempArr1, v);
        const r_hat_dot_v = dot(r_hat, v);
        if (r_hat_dot_v.isZero()) break;
        alpha = rho.div(r_hat_dot_v);
        for (let i = 0; i < n; i++) {
          s[i] = r[i].sub(alpha.mul(v[i]));
        }
        if (dot(s, s).cmp(tolSq) <= 0) {
          applyPrecond(p, tempArr2);
          for (let i = 0; i < n; i++) x[i] = x[i].add(alpha.mul(tempArr2[i]));
          break;
        }
        applyPrecond(s, tempArr1);
        spmv(tempArr1, t);
        const t_dot_t = dot(t, t);
        if (t_dot_t.isZero()) {
          omega = zero;
        } else {
          omega = dot(t, s).div(t_dot_t);
        }
        applyPrecond(p, tempArr2);
        for (let i = 0; i < n; i++) x[i] = x[i].add(alpha.mul(tempArr2[i]));
        applyPrecond(s, tempArr1);
        for (let i = 0; i < n; i++) x[i] = x[i].add(omega.mul(tempArr1[i]));
        for (let i = 0; i < n; i++) {
          r[i] = s[i].sub(omega.mul(t[i]));
        }
        rho_prev = rho;
        if (dot(r, r).cmp(tolSq) <= 0) break;
      }
      const result = new Vector(n);
      result.values = x;
      return result;
    }
    /**
     * Symmetric Rank-k Update (SYRK).
     * Computes the matrix product A * A^T.
     * 
     * @returns {SparseMatrixCSC}
     */
    syrk() {
      return this.mul(this.transpose());
    }
    /**
     * General Rank-1 Update (GER).
     * Computes A + x * y^T.
     * Note: Rank-1 updates on sparse matrices usually introduce significant fill-in (dense data).
     * 
     * @param {Vector|Array<number|string|BigFloat>} x 
     * @param {Vector|Array<number|string|BigFloat>} y 
     * @returns {SparseMatrixCSC}
     */
    ger(x, y) {
      const xVec = x instanceof Vector ? x.values : x;
      const yVec = y instanceof Vector ? y.values : y;
      if (this.rows !== xVec.length || this.cols !== yVec.length) {
        throw new Error("Dimension mismatch for GER: x must match rows, y must match cols.");
      }
      const rowIdx = [];
      const colIdx = [];
      const vals = [];
      for (let j = 0; j < yVec.length; j++) {
        const yj = yVec[j] instanceof BigFloat ? yVec[j] : bf(yVec[j]);
        if (yj.isZero()) continue;
        for (let i = 0; i < xVec.length; i++) {
          const xi = xVec[i] instanceof BigFloat ? xVec[i] : bf(xVec[i]);
          if (xi.isZero()) continue;
          rowIdx.push(i);
          colIdx.push(j);
          vals.push(xi.mul(yj));
        }
      }
      const XYt = _SparseMatrixCSC.fromCOO(this.rows, this.cols, rowIdx, colIdx, vals);
      return this.add(XYt);
    }
    /**
     * Triangular Solve with Multiple Right-Hand Sides (TRSM).
     * Solves A * X = B, where A is this triangular matrix.
     * 
     * @param {SparseMatrixCSC} B - The right-hand side sparse matrix.
     * @param {boolean} [lower=true] - True if A is lower triangular, False if upper triangular.
     * @returns {SparseMatrixCSC} X - The solution sparse matrix.
     */
    trsm(B, lower = true) {
      if (this.rows !== this.cols) throw new Error("Matrix must be square for TRSM.");
      if (this.rows !== B.rows) throw new Error("Dimension mismatch: A rows must match B rows.");
      const M = B.rows;
      const N = B.cols;
      const X_vals = [];
      const X_rowIdx = [];
      const X_colPtrs = new Uint32Array(N + 1);
      for (let j = 0; j < N; j++) {
        X_colPtrs[j] = X_vals.length;
        const bj = new Vector(M);
        const start = B.colPointers[j];
        const end = B.colPointers[j + 1];
        for (let p = start; p < end; p++) {
          bj.values[B.rowIndices[p]] = B.values[p];
        }
        const xj = lower ? this.solveLowerTriangular(bj) : this.solveUpperTriangular(bj);
        for (let i = 0; i < M; i++) {
          if (!xj.values[i].isZero()) {
            X_rowIdx.push(i);
            X_vals.push(xj.values[i]);
          }
        }
      }
      X_colPtrs[N] = X_vals.length;
      return new _SparseMatrixCSC(M, N, X_vals, new Uint32Array(X_rowIdx), X_colPtrs);
    }
    /**
     * Sparse LU Factorization (Left-Looking / Gilbert-Peierls Algorithm).
     * Computes A = L * U where L is lower triangular with unit diagonal, and U is upper triangular.
     * 
     * @returns {{L: SparseMatrixCSC, U: SparseMatrixCSC}}
     */
    lu() {
      if (this.rows !== this.cols) throw new Error("Square matrix required for LU factorization.");
      const n = this.rows;
      const L_vals = [], L_rowIdx = [];
      const L_colPtrs = new Uint32Array(n + 1);
      const U_vals = [], U_rowIdx = [];
      const U_colPtrs = new Uint32Array(n + 1);
      const x = new Array(n).fill(zero);
      const active = new Uint8Array(n);
      const activeRows = [];
      for (let j = 0; j < n; j++) {
        L_colPtrs[j] = L_vals.length;
        U_colPtrs[j] = U_vals.length;
        const A_start = this.colPointers[j];
        const A_end = this.colPointers[j + 1];
        for (let p = A_start; p < A_end; p++) {
          const r = this.rowIndices[p];
          x[r] = this.values[p];
          if (!active[r]) {
            active[r] = 1;
            activeRows.push(r);
          }
        }
        for (let i = 0; i < j; i++) {
          const xi = x[i];
          if (!xi.isZero()) {
            const L_start = L_colPtrs[i];
            const L_end = L_colPtrs[i + 1];
            for (let p = L_start; p < L_end; p++) {
              const r = L_rowIdx[p];
              if (r > i) {
                x[r] = x[r].sub(L_vals[p].mul(xi));
                if (!active[r]) {
                  active[r] = 1;
                  activeRows.push(r);
                }
              }
            }
          }
        }
        activeRows.sort((a, b) => a - b);
        const U_jj = x[j];
        if (U_jj === void 0 || U_jj.isZero()) {
          throw new Error(`Zero pivot encountered at column ${j}. Sparse LU without pivoting failed.`);
        }
        L_rowIdx.push(j);
        L_vals.push(one);
        for (let i = 0; i < activeRows.length; i++) {
          const r = activeRows[i];
          const val = x[r];
          if (!val.isZero()) {
            if (r <= j) {
              U_rowIdx.push(r);
              U_vals.push(val);
            } else {
              L_rowIdx.push(r);
              L_vals.push(val.div(U_jj));
            }
          }
          x[r] = zero;
          active[r] = 0;
        }
        activeRows.length = 0;
      }
      L_colPtrs[n] = L_vals.length;
      U_colPtrs[n] = U_vals.length;
      return {
        L: new _SparseMatrixCSC(n, n, L_vals, new Uint32Array(L_rowIdx), L_colPtrs),
        U: new _SparseMatrixCSC(n, n, U_vals, new Uint32Array(U_rowIdx), U_colPtrs)
      };
    }
    /**
     * Sparse Cholesky Factorization.
     * Computes A = L * L^T for Symmetric Positive Definite (SPD) matrices.
     * Extracts factor from the LU Decomposition by scaling L with sqrt(diag(U)).
     * 
     * @returns {SparseMatrixCSC} L - The lower triangular Cholesky factor.
     */
    cholesky() {
      const { L, U } = this.lu();
      const n = this.rows;
      const Lc_vals = new Array(L.nnz);
      const Lc_rowIdx = new Uint32Array(L.rowIndices);
      const Lc_colPtrs = new Uint32Array(L.colPointers);
      for (let j = 0; j < n; j++) {
        let U_jj = zero;
        const U_start = U.colPointers[j];
        const U_end = U.colPointers[j + 1];
        for (let p = U_start; p < U_end; p++) {
          if (U.rowIndices[p] === j) {
            U_jj = U.values[p];
            break;
          }
        }
        if (U_jj.cmp(zero) <= 0) {
          throw new Error("Matrix is not symmetric positive definite.");
        }
        const sqrt_Ujj = U_jj.sqrt();
        const L_start = L.colPointers[j];
        const L_end = L.colPointers[j + 1];
        for (let p = L_start; p < L_end; p++) {
          Lc_vals[p] = L.values[p].mul(sqrt_Ujj);
        }
      }
      return new _SparseMatrixCSC(n, n, Lc_vals, Lc_rowIdx, Lc_colPtrs);
    }
    /**
     * General Direct Solver for A * x = b.
     * Uses the exact Sparse LU Factorization to compute the solution.
     * 
     * @param {Vector|Array<number|string|BigFloat>} b 
     * @returns {Vector} x
     */
    solve(b) {
      const { L, U } = this.lu();
      const y = L.solveLowerTriangular(b);
      const x = U.solveUpperTriangular(y);
      return x;
    }
    /**
     * Computes the Determinant of the sparse matrix using LU factorization.
     * det(A) = Product of the diagonals of U.
     * 
     * @returns {BigFloat}
     */
    det() {
      if (this.rows !== this.cols) throw new Error("Square matrix required for determinant.");
      const { U } = this.lu();
      let d = one;
      for (let j = 0; j < this.cols; j++) {
        const start = U.colPointers[j];
        const end = U.colPointers[j + 1];
        let diagVal = zero;
        for (let p = start; p < end; p++) {
          if (U.rowIndices[p] === j) {
            diagVal = U.values[p];
            break;
          }
        }
        d = d.mul(diagVal);
      }
      return d;
    }
    /**
     * Computes the Log-Determinant of the matrix (useful for Gaussians and PDFs).
     * logDet(A) = Sum of the logs of the absolute diagonals of U.
     * 
     * @returns {BigFloat}
     */
    logDet() {
      if (this.rows !== this.cols) throw new Error("Square matrix required for logDet.");
      const { U } = this.lu();
      let logD = zero;
      for (let j = 0; j < this.cols; j++) {
        const start = U.colPointers[j];
        const end = U.colPointers[j + 1];
        let diagVal = zero;
        for (let p = start; p < end; p++) {
          if (U.rowIndices[p] === j) {
            diagVal = U.values[p];
            break;
          }
        }
        if (diagVal.isZero()) throw new Error("Matrix is singular, logDet is undefined.");
        logD = logD.add(diagVal.abs().log());
      }
      return logD;
    }
    // ============================================================================
    // --- Advanced Matrix Algorithms (Part 4): Inversion, QR, SVD & Eigen ---
    // ============================================================================
    /**
     * Computes the Inverse of the sparse matrix.
     * Warning: The inverse of a sparse matrix is typically dense. 
     * This uses column-by-column LU solves to construct the inverse dynamically.
     * 
     * @returns {SparseMatrixCSC}
     */
    inv() {
      if (this.rows !== this.cols) throw new Error("Matrix must be square to compute inverse.");
      const n = this.rows;
      const { L, U } = this.lu();
      const invVals = [];
      const invRowIdx = [];
      const invColPtrs = new Uint32Array(n + 1);
      const e = new Array(n).fill(zero);
      for (let j = 0; j < n; j++) {
        invColPtrs[j] = invVals.length;
        e[j] = one;
        if (j > 0) e[j - 1] = zero;
        const y = L.solveLowerTriangular(e);
        const x = U.solveUpperTriangular(y);
        for (let i = 0; i < n; i++) {
          const val = x.values[i];
          if (!val.isZero()) {
            invRowIdx.push(i);
            invVals.push(val);
          }
        }
      }
      invColPtrs[n] = invVals.length;
      return new _SparseMatrixCSC(n, n, invVals, new Uint32Array(invRowIdx), invColPtrs);
    }
    /**
     * Computes the Moore-Penrose Pseudoinverse (A^+).
     * Uses Normal Equations approach for sparse matrices to avoid full SVD overhead.
     * 
     * @returns {SparseMatrixCSC}
     */
    pinv() {
      const At = this.transpose();
      if (this.rows >= this.cols) {
        const AtA = At.mul(this);
        const invAtA = AtA.inv();
        return invAtA.mul(At);
      } else {
        const AAt = this.mul(At);
        const invAAt = AAt.inv();
        return At.mul(invAAt);
      }
    }
    /**
     * Sparse QR Factorization using Left-Looking Modified Gram-Schmidt (MGS).
     * Computes A = Q * R, where Q is orthogonal and R is upper triangular.
     * 
     * @returns {{Q: SparseMatrixCSC, R: SparseMatrixCSC}}
     */
    qr() {
      const m = this.rows;
      const n = this.cols;
      const Q_vals = [], Q_rowIdx = [];
      const Q_colPtrs = new Uint32Array(n + 1);
      const R_vals = [], R_rowIdx = [];
      const R_colPtrs = new Uint32Array(n + 1);
      const v = new Array(m).fill(zero);
      for (let j = 0; j < n; j++) {
        Q_colPtrs[j] = Q_vals.length;
        R_colPtrs[j] = R_vals.length;
        v.fill(zero);
        const startA = this.colPointers[j];
        const endA = this.colPointers[j + 1];
        for (let p = startA; p < endA; p++) {
          v[this.rowIndices[p]] = this.values[p];
        }
        for (let i = 0; i < j; i++) {
          let r_ij = zero;
          const startQ = Q_colPtrs[i];
          const endQ = Q_colPtrs[i + 1];
          for (let p = startQ; p < endQ; p++) {
            const row = Q_rowIdx[p];
            r_ij = r_ij.add(Q_vals[p].mul(v[row]));
          }
          if (!r_ij.isZero()) {
            R_rowIdx.push(i);
            R_vals.push(r_ij);
            for (let p = startQ; p < endQ; p++) {
              const row = Q_rowIdx[p];
              v[row] = v[row].sub(r_ij.mul(Q_vals[p]));
            }
          }
        }
        let normSq = zero;
        for (let i = 0; i < m; i++) {
          if (!v[i].isZero()) normSq = normSq.add(v[i].mul(v[i]));
        }
        const r_jj = normSq.sqrt();
        if (!r_jj.isZero()) {
          R_rowIdx.push(j);
          R_vals.push(r_jj);
          const inv_rjj = one.div(r_jj);
          for (let i = 0; i < m; i++) {
            if (!v[i].isZero()) {
              Q_rowIdx.push(i);
              Q_vals.push(v[i].mul(inv_rjj));
            }
          }
        } else {
          R_rowIdx.push(j);
          R_vals.push(zero);
        }
      }
      Q_colPtrs[n] = Q_vals.length;
      R_colPtrs[n] = R_vals.length;
      return {
        Q: new _SparseMatrixCSC(m, n, Q_vals, new Uint32Array(Q_rowIdx), Q_colPtrs),
        R: new _SparseMatrixCSC(n, n, R_vals, new Uint32Array(R_rowIdx), R_colPtrs)
      };
    }
    /**
     * Computes the Dominant Eigenpair using Power Iteration.
     * In sparse libraries, eigenvalue solvers extract top-K values iteratively.
     * 
     * @param {number|string|BigFloat} [tol="1e-20"] - Convergence tolerance.
     * @param {number} [maxIter=1000] - Maximum iterations.
     * @returns {{eigenvalue: BigFloat, eigenvector: Vector}}
     */
    eigen(tol = "1e-20", maxIter = 1e3) {
      if (this.rows !== this.cols) throw new Error("Matrix must be square for Eigenvalue computation.");
      const n = this.rows;
      const tolerance = tol instanceof BigFloat ? tol : bf(tol);
      let v = new Vector(n);
      let initialNormSq = zero;
      for (let i = 0; i < n; i++) {
        v.values[i] = one;
        initialNormSq = initialNormSq.add(one);
      }
      v = v.scale(one.div(initialNormSq.sqrt()));
      let eigenvalue = zero;
      let prevEigenvalue = zero;
      for (let iter = 0; iter < maxIter; iter++) {
        const w = this.mulVec(v);
        eigenvalue = v.dot(w);
        if (iter > 0 && eigenvalue.sub(prevEigenvalue).abs().cmp(tolerance) <= 0) {
          let maxResidual = zero;
          for (let i = 0; i < n; i++) {
            const diff2 = w.values[i].sub(v.values[i].mul(eigenvalue)).abs();
            if (diff2.cmp(maxResidual) > 0) {
              maxResidual = diff2;
            }
          }
          if (maxResidual.cmp(tolerance) <= 0) {
            break;
          }
        }
        prevEigenvalue = eigenvalue;
        const wNorm = w.norm();
        if (wNorm.isZero()) break;
        v = w.scale(one.div(wNorm));
      }
      return { eigenvalue, eigenvector: v };
    }
    /**
     * Computes the Dominant Singular Value and Vectors using Golub-Kahan (Power Method on A^T A).
     * Extracts the Top-1 Singular component.
     * 
     * @param {number|string|BigFloat}[tol="1e-20"]
     * @param {number} [maxIter=1000]
     * @returns {{singularValue: BigFloat, u: Vector, v: Vector}}
     */
    svd(tol = "1e-20", maxIter = 1e3) {
      const At = this.transpose();
      const AtA = At.mul(this);
      const { eigenvalue: lambda, eigenvector: v } = AtA.eigen(tol, maxIter);
      if (lambda.cmp(zero) < 0) {
        throw new Error("Numerical instability: Negative eigenvalue found for A^T A.");
      }
      const singularValue = lambda.sqrt();
      let u = new Vector(this.rows);
      if (!singularValue.isZero()) {
        u = this.mulVec(v).scale(one.div(singularValue));
      }
      return { singularValue, u, v };
    }
  };

  // src/polyfit.js
  function polyfit(x, y, order, info = {}) {
    let max_time = info.max_time || 6e4;
    let start_time = (/* @__PURE__ */ new Date()).getTime();
    if (!Array.isArray(x) || !Array.isArray(y) || x.length !== y.length || x.length === 0) {
      throw new Error("Input arrays x and y must be non-empty and of equal length.");
    }
    if (order < 0 || order >= x.length) {
      throw new Error("Polynomial degree must be non-negative and less than the number of data points.");
    }
    const X2 = x.map((val) => bf(val));
    const Y = y.map((val) => bf(val));
    const N = X2.length;
    const M = order + 1;
    info.result = null;
    info.ssr = bf(0);
    info.r_squared = bf(0);
    info.exectime = 0;
    info.toString = function() {
      if (!this.result) return "No result";
      return `degree=${order}, 
        R^2=${this.r_squared.toString(10, 6)}, 
        SSR=${this.ssr.toString(10, 6)}, 
        exectime=${this.exectime}ms`;
    };
    const powers = new Array(N);
    const max_pow = 2 * order;
    for (let i = 0; i < N; i++) {
      powers[i] = new Array(max_pow + 1);
      powers[i][0] = bf(1);
      for (let p = 1; p <= max_pow; p++) {
        powers[i][p] = powers[i][p - 1].mul(X2[i]);
      }
    }
    const A = [];
    const B = [];
    for (let j = 0; j < M; j++) {
      const row = [];
      let sumB = bf(0);
      for (let k = 0; k < M; k++) {
        let sumA = bf(0);
        for (let i = 0; i < N; i++) {
          sumA = sumA.add(powers[i][j + k]);
        }
        row.push(sumA);
      }
      for (let i = 0; i < N; i++) {
        sumB = sumB.add(powers[i][j].mul(Y[i]));
      }
      A.push(row);
      B.push(sumB);
      if ((/* @__PURE__ */ new Date()).getTime() - start_time > max_time) {
        info.exectime = (/* @__PURE__ */ new Date()).getTime() - start_time;
        return null;
      }
    }
    try {
      const coeffs = solveLinearSystem(A, B);
      info.result = coeffs.reverse();
      let ss_res = bf(0);
      let sum_y = bf(0);
      for (let i = 0; i < N; i++) sum_y = sum_y.add(Y[i]);
      const mean_y = sum_y.div(bf(N));
      let ss_tot = bf(0);
      for (let i = 0; i < N; i++) {
        let y_pred = bf(0);
        for (let k = 0; k < info.result.length; k++) {
          y_pred = y_pred.mul(X2[i]).add(info.result[k]);
        }
        const res = Y[i].sub(y_pred);
        ss_res = ss_res.add(res.mul(res));
        const dev = Y[i].sub(mean_y);
        ss_tot = ss_tot.add(dev.mul(dev));
      }
      info.ssr = ss_res;
      info.rmse = ss_res.div(bf(N)).sqrt();
      if (!ss_tot.isZero()) {
        info.r_squared = bf(1).sub(ss_res.div(ss_tot));
      } else {
        info.r_squared = bf(1);
      }
    } catch (e) {
      if (info.debug) console.error("Polyfit Solver Error:", e);
      info.result = null;
    }
    info.exectime = (/* @__PURE__ */ new Date()).getTime() - start_time;
    return info.result;
  }
  function solveLinearSystem(A, b) {
    const n = A.length;
    for (let i = 0; i < n; i++) {
      let pivotRow = i;
      let maxVal = A[i][i].abs();
      for (let k = i + 1; k < n; k++) {
        if (A[k][i].abs().cmp(maxVal) > 0) {
          maxVal = A[k][i].abs();
          pivotRow = k;
        }
      }
      if (pivotRow !== i) {
        [A[i], A[pivotRow]] = [A[pivotRow], A[i]];
        [b[i], b[pivotRow]] = [b[pivotRow], b[i]];
      }
      if (A[i][i].isZero()) {
        throw new Error("Matrix is singular or ill-conditioned.");
      }
      for (let k = i + 1; k < n; k++) {
        const factor = A[k][i].div(A[i][i]);
        b[k] = b[k].sub(factor.mul(b[i]));
        for (let j = i; j < n; j++) {
          A[k][j] = A[k][j].sub(factor.mul(A[i][j]));
        }
      }
    }
    const x = new Array(n);
    for (let i = n - 1; i >= 0; i--) {
      let sum = bf(0);
      for (let j = i + 1; j < n; j++) {
        sum = sum.add(A[i][j].mul(x[j]));
      }
      x[i] = b[i].sub(sum).div(A[i][i]);
    }
    return x;
  }
  function polyval(p, x) {
    let xv = bf(x);
    let y = bf(0);
    for (let i = 0; i < p.length; i++) {
      y = y.mul(xv).add(p[i]);
    }
    return y;
  }

  // src/ode45.js
  function ode45(odefun, tspan, y0, info = {}) {
    let _e = bfjs.bf(info._e ?? 1e-16);
    let _re = bfjs.bf(info._re ?? 1e-16);
    let max_steps_limit = info.max_step || 2e6;
    let max_time = info.max_time || 12e5;
    const start_time = (/* @__PURE__ */ new Date()).getTime();
    const bf2 = (n) => bfjs.bf(n);
    const bf_zero = bfjs.zero;
    const bf_one = bfjs.one;
    const bf_p1 = bf2(0.1);
    const bf_p8 = bf2(0.8);
    const bf_p9 = bf2(0.9);
    const bf_5 = bf2(5);
    const rat = (n, d) => bf2(n).div(bf2(d));
    const c2 = rat(1, 5), c3 = rat(3, 10), c4 = rat(4, 5), c5 = rat(8, 9);
    const a21 = rat(1, 5);
    const a31 = rat(3, 40), a32 = rat(9, 40);
    const a41 = rat(44, 45), a42 = rat(-56, 15), a43 = rat(32, 9);
    const a51 = rat(19372, 6561), a52 = rat(-25360, 2187), a53 = rat(64448, 6561), a54 = rat(-212, 729);
    const a61 = rat(9017, 3168), a62 = rat(-355, 33), a63 = rat(46732, 5247), a64 = rat(49, 176), a65 = rat(-5103, 18656);
    const a71 = rat(35, 384), a72 = bf_zero, a73 = rat(500, 1113), a74 = rat(125, 192), a75 = rat(-2187, 6784), a76 = rat(11, 84);
    const b1 = a71, b3 = a73, b4 = a74, b5 = a75, b6 = a76;
    const E1 = rat(71, 57600);
    const E3 = rat(-71, 16695);
    const E4 = rat(71, 1920);
    const E5 = rat(-17253, 339200);
    const E6 = rat(22, 525);
    const E7 = rat(-1, 40);
    let y_curr = Array.isArray(y0) ? y0.map(bf2) : [bf2(y0)];
    let dim = y_curr.length;
    const computeStageY = (y, h2, coeffs, k_vecs) => {
      let res = new Array(dim);
      for (let i = 0; i < dim; i++) {
        let sum = bf_zero;
        for (let j = 0; j < coeffs.length; j++) {
          if (coeffs[j].isZero()) continue;
          sum = sum.add(k_vecs[j][i].mul(coeffs[j]));
        }
        res[i] = y[i].add(sum.mul(h2));
      }
      return res;
    };
    let t_start = bf2(tspan[0]);
    let t_final = bf2(tspan[1]);
    let t = t_start;
    if (t_start.cmp(t_final) === 0) {
      return null;
    }
    let h = info.initial_step ? bf2(info.initial_step) : bf_zero;
    let absTol = _e;
    let relTol = _re;
    let t_span = t_final.sub(t_start);
    let direction = t_span.sign();
    let test_progress = void 0;
    if (info.progress !== void 0) {
      let progress = info.progress;
      if (progress <= 0 || progress >= 1) {
        progress = 0.1;
      }
      let last_progress = 0;
      test_progress = (t2, y) => {
        let pos = t2.sub(t_start).div(t_span).f64();
        if (pos - last_progress >= progress) {
          last_progress = pos;
          if (info.progressCb) {
            info.progressCb(pos, t2, y);
          } else {
            console.log(`Progress at progress=${(pos * 100).toFixed(1)}%, y=${(Array.isArray(y) ? y : [y]).map((x) => x.f64()).join(",")}`);
          }
        }
      };
    }
    if (h.isZero()) {
      h = t_final.sub(t_start).mul(rat(1, 100)).abs();
      let min_h = rat(1, 1e6);
      if (h.cmp(min_h) < 0) h = min_h;
      if (h.cmp(t_final.sub(t_start).abs()) > 0) h = t_final.sub(t_start).abs();
    }
    h = h.abs().mul(bf2(direction));
    info.t = [t];
    info.y = [y_curr.map((val) => val)];
    info.steps = 0;
    info.failed_steps = 0;
    info.status = "running";
    let k1 = odefun(t, y_curr);
    if (!Array.isArray(k1)) k1 = [k1];
    let done = false;
    let steps = 0;
    while (!done) {
      if (steps >= max_steps_limit) {
        console.warn(`ode45: Max steps (${max_steps_limit}) exceeded at t=${t.toString(10, 6)}.`);
        info.status = "max_steps";
        break;
      }
      if ((/* @__PURE__ */ new Date()).getTime() - start_time > max_time) {
        console.warn("ode45: Timeout reached.");
        info.status = "timeout";
        break;
      }
      let dist_to_end = t_final.sub(t);
      let dist_abs = dist_to_end.abs();
      let h_abs = h.abs();
      if (dist_abs.cmp(bf2(1e-40)) <= 0) {
        done = true;
        break;
      }
      let last_step = false;
      if (h_abs.cmp(dist_abs) >= 0) {
        h = dist_to_end;
        last_step = true;
      }
      let y_temp = computeStageY(y_curr, h, [a21], [k1]);
      let k2 = odefun(t.add(c2.mul(h)), y_temp);
      if (!Array.isArray(k2)) k2 = [k2];
      y_temp = computeStageY(y_curr, h, [a31, a32], [k1, k2]);
      let k3 = odefun(t.add(c3.mul(h)), y_temp);
      if (!Array.isArray(k3)) k3 = [k3];
      y_temp = computeStageY(y_curr, h, [a41, a42, a43], [k1, k2, k3]);
      let k4 = odefun(t.add(c4.mul(h)), y_temp);
      if (!Array.isArray(k4)) k4 = [k4];
      y_temp = computeStageY(y_curr, h, [a51, a52, a53, a54], [k1, k2, k3, k4]);
      let k5 = odefun(t.add(c5.mul(h)), y_temp);
      if (!Array.isArray(k5)) k5 = [k5];
      y_temp = computeStageY(y_curr, h, [a61, a62, a63, a64, a65], [k1, k2, k3, k4, k5]);
      let k6 = odefun(t.add(h), y_temp);
      if (!Array.isArray(k6)) k6 = [k6];
      let y_next = computeStageY(y_curr, h, [b1, bf_zero, b3, b4, b5, b6], [k1, k2, k3, k4, k5, k6]);
      let k7 = odefun(t.add(h), y_next);
      if (!Array.isArray(k7)) k7 = [k7];
      let max_norm_err = bf_zero;
      for (let i = 0; i < dim; i++) {
        let term = k1[i].mul(E1).add(k3[i].mul(E3)).add(k4[i].mul(E4)).add(k5[i].mul(E5)).add(k6[i].mul(E6)).add(k7[i].mul(E7));
        let err_abs_val = term.mul(h).abs();
        let y_max = y_curr[i].abs().cmp(y_next[i].abs()) > 0 ? y_curr[i].abs() : y_next[i].abs();
        let sc = absTol.add(y_max.mul(relTol));
        let ratio = err_abs_val.div(sc);
        if (ratio.cmp(max_norm_err) > 0) {
          max_norm_err = ratio;
        }
      }
      if (max_norm_err.cmp(bf_one) <= 0) {
        t = t.add(h);
        y_curr = y_next;
        k1 = k7;
        steps++;
        info.t.push(t);
        info.y.push(y_curr.map((v) => v));
        if (info.cb) info.cb(t, y_curr);
        if (test_progress !== void 0) test_progress(t, y_curr);
        if (last_step) {
          done = true;
        } else {
          let factor = bf_zero;
          if (max_norm_err.cmp(bf2(1e-40)) < 0) {
            factor = bf_5;
          } else {
            let inv_err = bf_one.div(max_norm_err);
            let pow_val = bfjs.exp(inv_err.log().mul(bf2(0.2)));
            factor = bf_p9.mul(pow_val);
          }
          if (factor.cmp(bf_5) > 0) factor = bf_5;
          if (factor.cmp(bf_p1) < 0) factor = bf_p1;
          h = h.mul(factor);
        }
      } else {
        info.failed_steps++;
        let inv_err = bf_one.div(max_norm_err);
        let pow_val = bfjs.exp(inv_err.log().mul(bf2(0.2)));
        let factor = bf_p9.mul(pow_val);
        if (factor.cmp(bf_p1) < 0) factor = bf_p1;
        if (factor.cmp(bf_p8) > 0) factor = bf_p8;
        h = h.mul(factor);
        if (h.abs().cmp(bf2(1e-50)) < 0) {
          console.warn("ode45: Step size underflow.");
          info.status = "underflow";
          break;
        }
      }
    }
    info.exectime = (/* @__PURE__ */ new Date()).getTime() - start_time;
    info.steps = steps;
    info.toString = function() {
      return `status=${this.status}, steps=${this.steps}, failed=${this.failed_steps}, t_final=${this.t[this.t.length - 1].toString(10, 6)}`;
    };
    if (info.status === "running") {
      info.status = "done";
      return { t: info.t, y: info.y };
    }
    return null;
  }

  // src/ode15s.js
  function ode15s(odefun, tspan, y0, info = {}) {
    const safe_bf2 = (n) => n instanceof BigFloat ? n : bf(n);
    let absTol = safe_bf2(info._e ?? "1e-15");
    let relTol = info._re === void 0 ? absTol : safe_bf2(info._re);
    let max_steps_limit = info.max_step || 2e6;
    let max_time = info.max_time || 12e5;
    const start_time = (/* @__PURE__ */ new Date()).getTime();
    let y_curr = Array.isArray(y0) ? y0.map(safe_bf2) : [safe_bf2(y0)];
    let dim = y_curr.length;
    let t_start = safe_bf2(tspan[0]);
    let t_final = safe_bf2(tspan[1]);
    let t = t_start;
    if (t_start.cmp(t_final) === 0) return null;
    let t_span = t_final.sub(t_start);
    let direction = t_span.sign();
    let test_progress = void 0;
    if (info.progress !== void 0) {
      let progress = info.progress;
      if (progress <= 0 || progress >= 1) {
        progress = 0.1;
      }
      let last_progress = 0;
      test_progress = (t2, y) => {
        let pos = t2.sub(t_start).div(t_span).f64();
        if (pos - last_progress >= progress) {
          last_progress = pos;
          if (info.progressCb) {
            info.progressCb(pos, t2, y);
          } else {
            console.log(`Progress at progress=${(pos * 100).toFixed(1)}%, y=${(Array.isArray(y) ? y : [y]).map((x) => x.f64()).join(",")}`);
          }
        }
      };
    }
    const prec = decimalPrecision();
    const machine_eps = getEpsilon();
    const time_tolerance = t_final.abs().add(one).mul(machine_eps).mul(bf("1e4"));
    const min_h_limit = bf("1e-" + Math.floor(prec * 0.9));
    const jacobian_eps = bf("1e-" + Math.max(Math.floor(prec / 2), 8));
    const tiny_lte = Math.pow(10, -Math.floor(prec * 0.6667));
    const m_zero_tol = bf("1e-" + Math.floor(prec * 0.8));
    const bf_0_2 = bf("0.2");
    const bf_0_5 = bf("0.5");
    const bf_newton_tol = bf("0.05");
    const bf_k_arr = [zero, one, bf("2"), bf("3"), bf("4"), bf("5"), bf("6")];
    let h = info.initial_step ? safe_bf2(info.initial_step).abs() : t_final.sub(t_start).abs().mul(bf("0.01"));
    if (h.cmp(bf("1e-12")) < 0) h = bf("1e-12");
    h = h.mul(bf(direction));
    const BDF = [
      null,
      // 0
      { alpha: [bf("1")], beta: bf("1") },
      // 1st Order (Backward Euler)
      { alpha: [bf("4").div(bf("3")), bf("-1").div(bf("3"))], beta: bf("2").div(bf("3")) },
      // 2nd
      { alpha: [bf("18").div(bf("11")), bf("-9").div(bf("11")), bf("2").div(bf("11"))], beta: bf("6").div(bf("11")) },
      // 3rd
      { alpha: [bf("48").div(bf("25")), bf("-36").div(bf("25")), bf("16").div(bf("25")), bf("-3").div(bf("25"))], beta: bf("12").div(bf("25")) },
      // 4th
      { alpha: [bf("300").div(bf("137")), bf("-300").div(bf("137")), bf("200").div(bf("137")), bf("-75").div(bf("137")), bf("12").div(bf("137"))], beta: bf("60").div(bf("137")) }
      // 5th
    ];
    info.t = [t];
    info.y = [y_curr.map((v) => v)];
    info.dy = [];
    info.steps = 0;
    info.failed_steps = 0;
    info.status = "running";
    let k = 1;
    let res0 = odefun(t, y_curr);
    let M0 = Array.isArray(res0) ? null : res0.M;
    let f0 = Array.isArray(res0) ? res0 : res0.f;
    if (!Array.isArray(f0)) f0 = [f0];
    let dy0 = [];
    for (let d = 0; d < dim; d++) {
      let m_val = M0 ? M0[d] : one;
      if (m_val.abs().cmp(m_zero_tol) <= 0) {
        dy0.push(zero);
      } else {
        dy0.push(f0[d].div(m_val));
      }
    }
    info.dy.push(dy0);
    let history = [{ t, y: y_curr, f: f0, M: M0 }];
    const computeDividedDifferences = (pts) => {
      let m = pts.length - 1;
      let D = [];
      for (let i = 0; i <= m; i++) D.push([pts[i].y.map((v) => v)]);
      for (let j = 1; j <= m; j++) {
        for (let i = 0; i <= m - j; i++) {
          let dx = pts[i].t.sub(pts[i + j].t);
          let inv_dx = one.div(dx);
          let diff2 = [];
          for (let d = 0; d < dim; d++) {
            diff2.push(D[i][j - 1][d].sub(D[i + 1][j - 1][d]).mul(inv_dx));
          }
          D[i].push(diff2);
        }
      }
      return D;
    };
    const evalPoly = (pts, D, t_target) => {
      let m = pts.length - 1;
      let res = new Array(dim).fill(zero);
      let term = one;
      for (let j = 0; j <= m; j++) {
        for (let d = 0; d < dim; d++) res[d] = res[d].add(D[0][j][d].mul(term));
        if (j < m) term = term.mul(t_target.sub(pts[j].t));
      }
      return res;
    };
    const getJacobian = (t_val, y_val, f_val) => {
      if (info.Jacobian) {
        let J_user = info.Jacobian(t_val, y_val, f_val);
        if (Array.isArray(J_user) && Array.isArray(J_user[0])) {
          let rowIdx2 = [], colIdx2 = [], vals2 = [];
          for (let i = 0; i < J_user.length; i++) {
            for (let j = 0; j < J_user[i].length; j++) {
              let v = J_user[i][j];
              if (!v.isZero()) {
                rowIdx2.push(i);
                colIdx2.push(j);
                vals2.push(v);
              }
            }
          }
          return { rowIdx: rowIdx2, colIdx: colIdx2, vals: vals2 };
        }
        return J_user;
      }
      let rowIdx = [];
      let colIdx = [];
      let vals = [];
      for (let j = 0; j < dim; j++) {
        let y_pert = [...y_val];
        let delta = y_val[j].abs().mul(jacobian_eps);
        if (delta.cmp(jacobian_eps) < 0) delta = jacobian_eps;
        let inv_delta = one.div(delta);
        y_pert[j] = y_pert[j].add(delta);
        let res_pert = odefun(t_val, y_pert);
        let f_pert = Array.isArray(res_pert) ? res_pert : res_pert.f;
        if (!Array.isArray(f_pert)) f_pert = [f_pert];
        for (let i = 0; i < dim; i++) {
          let diff2 = f_pert[i].sub(f_val[i]).mul(inv_delta);
          if (!diff2.isZero()) {
            rowIdx.push(i);
            colIdx.push(j);
            vals.push(diff2);
          }
        }
      }
      return { rowIdx, colIdx, vals };
    };
    let global_error;
    if (!!info.estimate_error) {
      global_error = new Array(dim).fill(zero);
      info.global_error_history = [global_error.map((v) => v.f64())];
    }
    let update_jacobian = true;
    let update_LU = true;
    let steps_since_jacobian = 0;
    let last_h_beta = zero;
    let J_M = null;
    let J_f = null;
    let lu_res = null;
    let done = false;
    let steps = 0;
    while (!done) {
      if (steps >= max_steps_limit) {
        info.status = "max_steps";
        break;
      }
      if ((/* @__PURE__ */ new Date()).getTime() - start_time > max_time) {
        info.status = "timeout";
        break;
      }
      let dist_to_end = t_final.sub(t);
      let dist_abs = dist_to_end.abs();
      if (dist_abs.cmp(time_tolerance) <= 0) {
        done = true;
        break;
      }
      let last_step = false;
      if (h.abs().cmp(dist_abs) >= 0) {
        h = dist_to_end;
        last_step = true;
        update_LU = true;
      }
      let t_next = t.add(h);
      let pts = history.slice(0, k + 1);
      let y_pred, y_star = [];
      let D_old;
      if (pts.length === 1) {
        y_pred = [];
        let M_start = pts[0].M;
        let f_start = pts[0].f;
        for (let d = 0; d < dim; d++) {
          let m_val = M_start ? M_start[d] : one;
          if (m_val.abs().cmp(m_zero_tol) <= 0) {
            y_pred.push(pts[0].y[d]);
          } else {
            y_pred.push(pts[0].y[d].add(h.mul(f_start[d].div(m_val))));
          }
        }
        y_star = [pts[0].y];
      } else {
        D_old = computeDividedDifferences(pts);
        y_pred = evalPoly(pts, D_old, t_next);
        for (let j = 1; j <= k; j++) {
          y_star.push(evalPoly(pts, D_old, t_next.sub(h.mul(bf_k_arr[j]))));
        }
      }
      let C = new Array(dim).fill(zero);
      for (let j = 1; j <= k; j++) {
        let alpha_j = BDF[k].alpha[j - 1];
        for (let d = 0; d < dim; d++) C[d] = C[d].add(alpha_j.mul(y_star[j - 1][d]));
      }
      let beta_k = BDF[k].beta;
      let h_beta = h.mul(beta_k);
      if (steps_since_jacobian >= 400) update_jacobian = true;
      if (update_jacobian) {
        let res_pred = odefun(t_next, y_pred);
        let f_pred = Array.isArray(res_pred) ? res_pred : res_pred.f;
        if (!Array.isArray(f_pred)) f_pred = [f_pred];
        J_M = Array.isArray(res_pred) ? null : res_pred.M;
        J_f = getJacobian(t_next, y_pred, f_pred);
        steps_since_jacobian = 0;
        update_LU = true;
        update_jacobian = false;
      }
      if (!update_LU) {
        let h_beta_ratio = h_beta.div(last_h_beta);
        let h_beta_ratio_n = Math.abs(h_beta_ratio.f64());
        if (h_beta_ratio_n > 1.25 || h_beta_ratio_n < 0.8) {
          update_LU = true;
        }
      }
      if (update_LU) {
        let rowIdx = [];
        let colIdx = [];
        let vals = [];
        for (let i = 0; i < dim; i++) {
          let m_val = J_M ? J_M[i] : one;
          if (!m_val.isZero()) {
            rowIdx.push(i);
            colIdx.push(i);
            vals.push(m_val);
          }
        }
        for (let i = 0; i < J_f.vals.length; i++) {
          let r = J_f.rowIdx[i];
          let c = J_f.colIdx[i];
          let v = J_f.vals[i];
          let J_term = h_beta.mul(v).neg();
          rowIdx.push(r);
          colIdx.push(c);
          vals.push(J_term);
        }
        try {
          let JG_sparse = SparseMatrixCSC.fromCOO(dim, dim, rowIdx, colIdx, vals);
          lu_res = JG_sparse.lu();
          last_h_beta = h_beta;
          update_LU = false;
        } catch (e) {
          h = h.mul(bf_0_2);
          info.failed_steps++;
          last_step = false;
          update_jacobian = true;
          continue;
        }
      }
      let y_tmp_curr = y_pred.map((v) => v);
      let newton_converged = false;
      let M_curr = null;
      let old_delta_norm = null;
      for (let iter = 0; iter < 5; iter++) {
        let res_curr = odefun(t_next, y_tmp_curr);
        M_curr = Array.isArray(res_curr) ? null : res_curr.M;
        let f_curr = Array.isArray(res_curr) ? res_curr : res_curr.f;
        if (!Array.isArray(f_curr)) f_curr = [f_curr];
        let negG = [];
        for (let d = 0; d < dim; d++) {
          let y_diff = y_tmp_curr[d].sub(C[d]);
          let m_term = M_curr ? M_curr[d].mul(y_diff) : y_diff;
          let G_val = m_term.sub(h_beta.mul(f_curr[d]));
          negG.push(G_val.neg());
        }
        let y_tmp = lu_res.L.solveLowerTriangular(negG);
        let delta_y = lu_res.U.solveUpperTriangular(y_tmp).values;
        let step_converged = true;
        let current_delta_norm = zero;
        for (let d = 0; d < dim; d++) {
          y_tmp_curr[d] = y_tmp_curr[d].add(delta_y[d]);
          let max_y = y_curr[d].abs().cmp(y_tmp_curr[d].abs()) > 0 ? y_curr[d].abs() : y_tmp_curr[d].abs();
          let sc = absTol.add(relTol.mul(max_y));
          let ratio = delta_y[d].abs().div(sc);
          if (ratio.cmp(current_delta_norm) > 0) current_delta_norm = ratio;
          if (delta_y[d].abs().cmp(sc.mul(bf_newton_tol)) > 0) {
            step_converged = false;
          }
        }
        if (step_converged) {
          newton_converged = true;
          break;
        }
        if (iter > 0 && old_delta_norm !== null) {
          let theta = current_delta_norm.div(old_delta_norm).f64();
          if (theta > 0.8) {
            break;
          }
        }
        old_delta_norm = current_delta_norm;
      }
      if (!newton_converged) {
        if (steps_since_jacobian > 0) {
          update_jacobian = true;
          continue;
        }
        h = h.mul(bf_0_5);
        info.failed_steps++;
        last_step = false;
        update_jacobian = true;
        continue;
      }
      let LTE_norm = 0;
      let inv_k_plus_1 = one.div(bf_k_arr[k + 1]);
      for (let d = 0; d < dim; d++) {
        let m_val = M_curr ? M_curr[d] : one;
        if (m_val.abs().cmp(m_zero_tol) <= 0) continue;
        let err_val = y_tmp_curr[d].sub(y_pred[d]).mul(inv_k_plus_1);
        let max_y = y_curr[d].abs().cmp(y_tmp_curr[d].abs()) > 0 ? y_curr[d].abs() : y_tmp_curr[d].abs();
        let sc = absTol.add(relTol.mul(max_y));
        let ratio = err_val.abs().div(sc).f64();
        if (ratio > LTE_norm) LTE_norm = ratio;
      }
      if (LTE_norm <= 1) {
        if (!!info.estimate_error) {
          let rhs_E = [];
          for (let d = 0; d < dim; d++) {
            let m_val = M_curr ? M_curr[d] : one;
            if (m_val.abs().cmp(m_zero_tol) <= 0) {
              rhs_E.push(global_error[d]);
            } else {
              let err_val_true = y_tmp_curr[d].sub(y_pred[d]).mul(inv_k_plus_1);
              rhs_E.push(global_error[d].add(err_val_true));
            }
          }
          try {
            let E_tmp = lu_res.L.solveLowerTriangular(rhs_E);
            global_error = lu_res.U.solveUpperTriangular(E_tmp).values;
          } catch (e) {
            for (let d = 0; d < dim; d++) global_error[d] = rhs_E[d];
          }
          info.global_error_history.push(global_error.map((v) => v.f64()));
        }
        steps_since_jacobian++;
        t = t_next;
        y_curr = y_tmp_curr;
        let res_final = odefun(t, y_curr);
        let f_final = Array.isArray(res_final) ? res_final : res_final.f;
        let M_final = Array.isArray(res_final) ? null : res_final.M;
        if (!Array.isArray(f_final)) f_final = [f_final];
        history.unshift({ t, y: y_curr, f: f_final, M: M_final });
        if (history.length > 7) history.pop();
        let dy_curr = new Array(dim);
        for (let d = 0; d < dim; d++) dy_curr[d] = y_curr[d].sub(C[d]).div(h_beta);
        info.t.push(t);
        info.y.push(y_curr.map((v) => v));
        info.dy.push(dy_curr);
        if (info.cb) info.cb(t, y_curr);
        if (test_progress !== void 0) test_progress(t, y_curr);
        steps++;
        if (last_step) {
          done = true;
          break;
        }
        let D_new = computeDividedDifferences(history);
        let L_hist = history.length;
        let max_h_opt = 0;
        let next_k = k;
        for (let m = Math.max(1, k - 1); m <= Math.min(5, k + 1); m++) {
          let err_norm_m = 0;
          if (m === k) {
            err_norm_m = LTE_norm;
          } else if (m + 1 < L_hist) {
            let term = one;
            for (let i = 1; i <= m + 1; i++) term = term.mul(history[0].t.sub(history[i].t));
            let inv_m_plus_1 = one.div(bf_k_arr[m + 1]);
            for (let d = 0; d < dim; d++) {
              let m_val = M_final ? M_final[d] : one;
              if (m_val.abs().cmp(m_zero_tol) <= 0) continue;
              let err_val = D_new[0][m + 1][d].mul(term).mul(inv_m_plus_1);
              let sc = absTol.add(relTol.mul(y_curr[d].abs()));
              let ratio = err_val.abs().div(sc).f64();
              if (ratio > err_norm_m) err_norm_m = ratio;
            }
          } else {
            continue;
          }
          if (err_norm_m < tiny_lte) err_norm_m = tiny_lte;
          let h_opt_m = 0.9 * Math.pow(err_norm_m, -1 / (m + 1));
          if (m === k - 1) h_opt_m *= 1.2;
          if (m === k + 1) h_opt_m *= 0.8;
          if (h_opt_m > max_h_opt) {
            max_h_opt = h_opt_m;
            next_k = m;
          }
        }
        let factor = Math.max(0.2, Math.min(5, max_h_opt));
        if (factor >= 1.5 || factor <= 0.8 || next_k !== k) {
          h = h.mul(bf(factor));
          k = next_k;
        }
      } else {
        info.failed_steps++;
        last_step = false;
        update_jacobian = true;
        if (LTE_norm < tiny_lte) LTE_norm = tiny_lte;
        let factor = 0.9 * Math.pow(LTE_norm, -1 / (k + 1));
        factor = Math.max(0.1, factor);
        h = h.mul(bf(factor));
        if (h.abs().cmp(min_h_limit) < 0) {
          console.warn(`ode15s: Step size underflow limit reached. Function represents aggressive structural singularities or impossibly strict relative limits.`);
          info.status = "underflow";
          break;
        }
      }
    }
    info.exectime = (/* @__PURE__ */ new Date()).getTime() - start_time;
    info.steps = steps;
    info.toString = function() {
      return `status=${this.status}, steps=${this.steps}, failed=${this.failed_steps}, final_order=${k}, t_final=${this.t[this.t.length - 1].toString(10, 6)}`;
    };
    if (!!info.estimate_error) {
      info.global_error = global_error;
    }
    if (info.status === "running") {
      info.status = "done";
      return { t: info.t, y: info.y, dy: info.dy };
    }
    return null;
  }

  // src/pdepe.js
  function pdepe(m, pdefun, icfun, bcfun, xmesh, tspan, info = {}) {
    const safe_bf2 = (n) => n instanceof BigFloat ? n : bf(n);
    const m_val = parseInt(m.toString(), 10);
    const m_plus_1 = bf(m_val + 1);
    const N = xmesh.length;
    if (N < 3) throw new Error("pdepe: xmesh must contain at least 3 spatial points.");
    const X2 = xmesh.map(safe_bf2);
    const tspan_bf = tspan.map(safe_bf2);
    const prec = decimalPrecision();
    const eps = getEpsilon();
    const bf_0_5 = bf("0.5");
    const bf_2 = bf("2");
    const bf_3 = bf("3");
    const h_min_tol = bf("1e4").mul(eps);
    const zero_tol = bf("1e-" + Math.floor(prec * 0.9));
    const q_zero_tol = bf("1e-" + Math.floor(prec * 0.8));
    const is_sym_left = m_val > 0 && X2[0].cmp(zero) === 0;
    const Xmid = new Array(N - 1);
    const dx = new Array(N - 1);
    const pre_inv_dx = new Array(N - 1);
    for (let i = 0; i < N - 1; i++) {
      Xmid[i] = X2[i].add(X2[i + 1]).mul(bf_0_5);
      dx[i] = X2[i + 1].sub(X2[i]);
      pre_inv_dx[i] = one.div(dx[i]);
    }
    const pre_inv_dx_2 = new Array(N - 1);
    for (let i = 1; i < N - 1; i++) {
      pre_inv_dx_2[i] = one.div(dx[i - 1].add(dx[i]));
    }
    const powM = (x_bf) => {
      if (m_val === 0) return one;
      if (m_val === 1) return x_bf;
      if (m_val === 2) return x_bf.mul(x_bf);
      return x_bf.pow(m_val);
    };
    const powMp1 = (x_bf) => {
      if (m_val === 0) return x_bf;
      if (m_val === 1) return x_bf.mul(x_bf);
      return x_bf.pow(m_val + 1);
    };
    const powM_X = X2.map(powM);
    const powM_Xmid = Xmid.map(powM);
    const V = new Array(N);
    const pre_inv_V = new Array(N);
    for (let i = 0; i < N; i++) {
      let left_edge = i === 0 ? X2[0] : Xmid[i - 1];
      let right_edge = i === N - 1 ? X2[N - 1] : Xmid[i];
      V[i] = powMp1(right_edge).sub(powMp1(left_edge)).div(m_plus_1);
      pre_inv_V[i] = one.div(V[i]);
    }
    const toArr = (val) => Array.isArray(val) ? val : [val];
    const _pdefun = (x_bf, t_bf, u_bf, dudx_bf) => {
      let res2 = pdefun(x_bf, t_bf, u_bf, dudx_bf);
      return { c: toArr(res2.c), f: toArr(res2.f), s: toArr(res2.s) };
    };
    const _bcfun = (xl, ul, xr, ur, t) => {
      let res2 = bcfun(xl, ul, xr, ur, t);
      return { pl: toArr(res2.pl), ql: toArr(res2.ql), pr: toArr(res2.pr), qr: toArr(res2.qr) };
    };
    let U0_flat = [];
    let D = 0;
    for (let i = 0; i < N; i++) {
      let u0_res = toArr(icfun(X2[i]));
      if (i === 0) D = u0_res.length;
      for (let d = 0; d < D; d++) U0_flat.push(u0_res[d]);
    }
    const getU = (Y, i) => Y.slice(i * D, (i + 1) * D);
    const ode_sys = (t, Y) => {
      let dY = new Array(N * D);
      let M = new Array(N * D);
      let F_mid = new Array(N - 1);
      for (let i = 0; i < N - 1; i++) {
        let u_L = getU(Y, i);
        let u_R = getU(Y, i + 1);
        let u_mid = new Array(D);
        let dudx_mid = new Array(D);
        for (let d = 0; d < D; d++) {
          u_mid[d] = u_L[d].add(u_R[d]).mul(bf_0_5);
          dudx_mid[d] = u_R[d].sub(u_L[d]).mul(pre_inv_dx[i]);
        }
        F_mid[i] = _pdefun(Xmid[i], t, u_mid, dudx_mid).f;
      }
      let C_node = new Array(N);
      let S_node = new Array(N);
      for (let i = 0; i < N; i++) {
        let u_node = getU(Y, i);
        let dudx_node = new Array(D);
        if (i === 0) {
          let u_R = getU(Y, 1);
          for (let d = 0; d < D; d++) dudx_node[d] = u_R[d].sub(u_node[d]).mul(pre_inv_dx[0]);
        } else if (i === N - 1) {
          let u_L = getU(Y, N - 2);
          for (let d = 0; d < D; d++) dudx_node[d] = u_node[d].sub(u_L[d]).mul(pre_inv_dx[N - 2]);
        } else {
          let u_L = getU(Y, i - 1);
          let u_R = getU(Y, i + 1);
          for (let d = 0; d < D; d++) dudx_node[d] = u_R[d].sub(u_L[d]).mul(pre_inv_dx_2[i]);
        }
        let pde_res = _pdefun(X2[i], t, u_node, dudx_node);
        C_node[i] = pde_res.c;
        S_node[i] = pde_res.s;
        if (i > 0 && i < N - 1) {
          for (let d = 0; d < D; d++) {
            let flux_R = powM_Xmid[i].mul(F_mid[i][d]);
            let flux_L = powM_Xmid[i - 1].mul(F_mid[i - 1][d]);
            let flux_diff = flux_R.sub(flux_L).mul(pre_inv_V[i]);
            let c_val = C_node[i][d];
            if (c_val.abs().cmp(zero_tol) < 0) {
              c_val = c_val.sign() >= 0 ? zero_tol : zero_tol.neg();
            }
            M[i * D + d] = c_val;
            dY[i * D + d] = flux_diff.add(S_node[i][d]);
          }
        }
      }
      let bc_res = _bcfun(X2[0], getU(Y, 0), X2[N - 1], getU(Y, N - 1), t);
      if (is_sym_left) {
        for (let d = 0; d < D; d++) {
          let flux_R = powM_Xmid[0].mul(F_mid[0][d]);
          let flux_L = zero;
          let flux_diff = flux_R.sub(flux_L).mul(pre_inv_V[0]);
          let c_val = C_node[0][d];
          if (c_val.abs().cmp(zero_tol) < 0) {
            c_val = c_val.sign() >= 0 ? zero_tol : zero_tol.neg();
          }
          M[d] = c_val;
          dY[d] = flux_diff.add(S_node[0][d]);
        }
      } else {
        for (let d = 0; d < D; d++) {
          if (bc_res.ql[d].abs().cmp(q_zero_tol) <= 0) {
            M[d] = zero;
            dY[d] = bc_res.pl[d];
          } else {
            let f_L = bc_res.pl[d].neg().div(bc_res.ql[d]);
            let flux_R = powM_Xmid[0].mul(F_mid[0][d]);
            let flux_L = powM_X[0].mul(f_L);
            let flux_diff = flux_R.sub(flux_L).mul(pre_inv_V[0]);
            let c_val = C_node[0][d];
            if (c_val.abs().cmp(zero_tol) < 0) {
              c_val = c_val.sign() >= 0 ? zero_tol : zero_tol.neg();
            }
            M[d] = c_val;
            dY[d] = flux_diff.add(S_node[0][d]);
          }
        }
      }
      let offset = (N - 1) * D;
      for (let d = 0; d < D; d++) {
        if (bc_res.qr[d].abs().cmp(q_zero_tol) <= 0) {
          M[offset + d] = zero;
          dY[offset + d] = bc_res.pr[d];
        } else {
          let f_R = bc_res.pr[d].neg().div(bc_res.qr[d]);
          let flux_R = powM_X[N - 1].mul(f_R);
          let flux_L = powM_Xmid[N - 2].mul(F_mid[N - 2][d]);
          let flux_diff = flux_R.sub(flux_L).mul(pre_inv_V[N - 1]);
          let c_val = C_node[N - 1][d];
          if (c_val.abs().cmp(zero_tol) < 0) {
            c_val = c_val.sign() >= 0 ? zero_tol : zero_tol.neg();
          }
          M[offset + d] = c_val;
          dY[offset + d] = flux_diff.add(S_node[N - 1][d]);
        }
      }
      return { M, f: dY };
    };
    const tmpInfo = Object.assign({
      _e: "1e-5",
      _re: "1e-4",
      max_step: 1e7,
      max_time: 1e7
    }, info);
    Object.assign(info, tmpInfo);
    if (!info.Jacobian) {
      const jacobian_eps = bf("1e-" + Math.max(Math.floor(prec / 2), 8));
      info.Jacobian = (t_val, y_val, f_val) => {
        let rowIdx = [];
        let colIdx = [];
        let vals = [];
        for (let color = 0; color < 3; color++) {
          for (let d = 0; d < D; d++) {
            let y_pert = [...y_val];
            let deltas = new Array(N).fill(zero);
            let has_pert = false;
            for (let i = color; i < N; i += 3) {
              let j = i * D + d;
              let delta = y_val[j].abs().mul(jacobian_eps);
              if (delta.cmp(jacobian_eps) < 0) delta = jacobian_eps;
              deltas[i] = delta;
              y_pert[j] = y_pert[j].add(delta);
              has_pert = true;
            }
            if (!has_pert) continue;
            let res_pert = ode_sys(t_val, y_pert);
            let f_pert = res_pert.f;
            for (let i = color; i < N; i += 3) {
              let j = i * D + d;
              let inv_delta = one.div(deltas[i]);
              let start_node = Math.max(0, i - 1);
              let end_node = Math.min(N - 1, i + 1);
              for (let node = start_node; node <= end_node; node++) {
                for (let d_aff = 0; d_aff < D; d_aff++) {
                  let r = node * D + d_aff;
                  let diff2 = f_pert[r].sub(f_val[r]).mul(inv_delta);
                  if (!diff2.isZero()) {
                    rowIdx.push(r);
                    colIdx.push(j);
                    vals.push(diff2);
                  }
                }
              }
            }
          }
        }
        return { rowIdx, colIdx, vals };
      };
    }
    let res = ode15s(ode_sys, [tspan_bf[0], tspan_bf[tspan_bf.length - 1]], U0_flat, info);
    info.Jacobian = void 0;
    if (!res) throw new Error("pdepe: Underlying ode15s integration failed catastrophically.");
    let ode_t = res.t;
    let ode_y = res.y;
    let ode_dy = res.dy;
    let sol = [];
    let k = 0;
    for (let ts of tspan_bf) {
      while (k < ode_t.length - 2 && ode_t[k + 1].cmp(ts) < 0) k++;
      let t0 = ode_t[k], t1 = ode_t[k + 1];
      let y0 = ode_y[k], y1 = ode_y[k + 1];
      let dy0 = ode_dy[k], dy1 = ode_dy[k + 1];
      let h = t1.sub(t0);
      let state_at_ts = new Array(N * D);
      if (ts.cmp(t0) === 0 || h.abs().cmp(h_min_tol) <= 0) {
        state_at_ts = y0;
      } else if (ts.cmp(t1) === 0) {
        state_at_ts = y1;
      } else {
        let s = ts.sub(t0).div(h);
        let s2 = s.mul(s), s3 = s2.mul(s);
        let h00 = one.sub(bf_3.mul(s2)).add(bf_2.mul(s3));
        let h01 = bf_3.mul(s2).sub(bf_2.mul(s3));
        let h10 = h.mul(s.sub(bf_2.mul(s2)).add(s3));
        let h11 = h.mul(s3.sub(s2));
        for (let j = 0; j < N * D; j++) {
          state_at_ts[j] = h00.mul(y0[j]).add(h01.mul(y1[j])).add(h10.mul(dy0[j])).add(h11.mul(dy1[j]));
        }
      }
      let grid_out = [];
      for (let i = 0; i < N; i++) {
        let eq_out = [];
        for (let d = 0; d < D; d++) {
          eq_out.push(state_at_ts[i * D + d]);
        }
        grid_out.push(D === 1 ? eq_out[0] : eq_out);
      }
      sol.push(grid_out);
    }
    return sol;
  }

  // src/fminbnd.js
  function fminbnd(f, _ax, _bx, info = {}) {
    let _e = info._e ?? 1e-30;
    let _re = info._re ?? _e;
    let max_step = info.max_step || 500, max_time = info.max_time || 6e4;
    if (typeof _e !== "number" || typeof _re !== "number" || typeof info !== "object") {
      throw new Error("arguments error");
    }
    const start_time = (/* @__PURE__ */ new Date()).getTime();
    const bf_zero = zero;
    const bf_one = one;
    const bf_two = two;
    const bf_half = half;
    const bf_golden = three.sub(bf(5).sqrt()).mul(bf_half);
    let a = bf(_ax);
    let b = bf(_bx);
    if (a.cmp(b) > 0) {
      let temp = a;
      a = b;
      b = temp;
    }
    let c = b.sub(a);
    let x = a.add(c.mul(bf_golden));
    let w = x;
    let v = x;
    let fx = f(x);
    let fw = fx;
    let fv = fx;
    let d = bf_zero;
    let e = bf_zero;
    let tol_act = bf(_e);
    let tol_rel = bf(_re);
    const updateInfo = (iter, errorBound, currentMinVal) => {
      info.exectime = (/* @__PURE__ */ new Date()).getTime() - start_time;
      info.steps = iter;
      info.lastresult = x;
      info.min_value = currentMinVal;
      info.error = errorBound;
      if (errorBound.cmp(bf_zero) === 0) {
        info.eff_decimal_precision = decimalPrecision();
      } else {
        info.eff_decimal_precision = Math.floor(-errorBound.log().f64() / Math.log(10));
      }
      if (info.eff_decimal_precision <= 0) {
        info.eff_decimal_precision = 0;
        info.eff_result = "";
      } else {
        let limit2 = decimalPrecision();
        let prec = info.eff_decimal_precision > limit2 ? limit2 : info.eff_decimal_precision;
        info.eff_result = x.toString(10, prec);
      }
    };
    info.toString = function() {
      return `xmin=${this.eff_result}, 
      f(xmin)=${this.min_value ? this.min_value.toString(10, 6) : "N/A"},
      steps=${this.steps}/${max_step}, 
      error=${this.error ? this.error.toString(10, 3) : "N/A"},
      exectime=${this.exectime}/${max_time}`;
    };
    for (let iter = 1; iter <= max_step; ++iter) {
      let xm = a.add(b).mul(bf_half);
      let tol1 = x.abs().mul(tol_rel).add(tol_act);
      let tol2 = tol1.mul(bf_two);
      let dist = x.sub(xm).abs().add(b.sub(a).mul(bf_half));
      if (dist.cmp(tol2) <= 0) {
        updateInfo(iter, dist, fx);
        info.result = x;
        return x;
      }
      if (info.cb) {
        updateInfo(iter, dist, fx);
        info.cb();
      }
      if ((/* @__PURE__ */ new Date()).getTime() - start_time > max_time) {
        updateInfo(iter, dist, fx);
        console.log("fminbnd: Timeout reached.");
        info.result = null;
        return null;
      }
      let p = bf_zero;
      let q = bf_zero;
      let r = bf_zero;
      let new_step = bf_zero;
      if (e.abs().cmp(tol1) > 0) {
        r = x.sub(w).mul(fx.sub(fv));
        q = x.sub(v).mul(fx.sub(fw));
        p = x.sub(v).mul(q).sub(x.sub(w).mul(r));
        q = q.sub(r).mul(bf_two);
        if (q.cmp(bf_zero) > 0) {
          p = p.neg();
        }
        q = q.abs();
        let etemp = e;
        e = d;
        let is_parabolic_valid = true;
        if (p.abs().cmp(q.mul(etemp).abs().mul(bf_half)) >= 0) {
          is_parabolic_valid = false;
        } else {
          let p_bound_a = q.mul(a.sub(x));
          let p_bound_b = q.mul(b.sub(x));
          if (p.cmp(p_bound_a) <= 0 || p.cmp(p_bound_b) >= 0) {
            is_parabolic_valid = false;
          }
        }
        if (is_parabolic_valid) {
          d = p.div(q);
          new_step = d;
          let u_tentative = x.add(d);
          if (u_tentative.sub(a).cmp(tol2) < 0 || b.sub(u_tentative).cmp(tol2) < 0) {
            let sign2 = xm.sub(x).cmp(bf_zero) >= 0 ? bf_one : bf_one.neg();
            d = tol1.mul(sign2);
          }
        } else {
          e = (x.cmp(xm) >= 0 ? a : b).sub(x);
          d = bf_golden.mul(e);
        }
      } else {
        e = (x.cmp(xm) >= 0 ? a : b).sub(x);
        d = bf_golden.mul(e);
      }
      if (d.abs().cmp(tol1) >= 0) {
        new_step = d;
      } else {
        let sign2 = d.cmp(bf_zero) >= 0 ? bf_one : bf_one.neg();
        new_step = tol1.mul(sign2);
      }
      let u = x.add(new_step);
      let fu = f(u);
      if (fu.cmp(fx) <= 0) {
        if (u.cmp(x) >= 0) {
          a = x;
        } else {
          b = x;
        }
        v = w;
        fv = fw;
        w = x;
        fw = fx;
        x = u;
        fx = fu;
      } else {
        if (u.cmp(x) < 0) {
          a = u;
        } else {
          b = u;
        }
        if (fu.cmp(fw) <= 0 || w.cmp(x) === 0) {
          v = w;
          fv = fw;
          w = u;
          fw = fu;
        } else if (fu.cmp(fv) <= 0 || v.cmp(x) === 0 || v.cmp(w) === 0) {
          v = u;
          fv = fu;
        }
      }
    }
    let final_dist = x.sub(a.add(b).mul(bf_half)).abs().add(b.sub(a).mul(bf_half));
    updateInfo(max_step, final_dist, fx);
    console.log(`fminbnd: Failed to converge after ${max_step} steps.`);
    info.result = null;
    return null;
  }

  // src/roots.js
  function roots(_coeffs, info = {}) {
    let max_step = info.max_step || 500;
    let max_time = info.max_time || 6e4;
    let tol = bf(info._e || 1e-30);
    const start_time = (/* @__PURE__ */ new Date()).getTime();
    const zero2 = zero;
    const one2 = one;
    let rawPoly = _coeffs.map((c) => new Complex(c));
    while (rawPoly.length > 0 && rawPoly[0].abs().cmp(zero2) === 0) {
      rawPoly.shift();
    }
    let n = rawPoly.length - 1;
    const updateInfo = (iter, max_err, current_roots_arr) => {
      info.steps = iter;
      info.exectime = (/* @__PURE__ */ new Date()).getTime() - start_time;
      info.error = max_err;
      let c = max_err.cmp(zero2);
      if (c === 0) {
        info.eff_decimal_precision = decimalPrecision();
      } else if (c < 0) {
        info.eff_decimal_precision = 0;
      } else {
        info.eff_decimal_precision = Math.floor(-max_err.log().f64() / Math.log(10));
      }
      if (info.eff_decimal_precision < 0) info.eff_decimal_precision = 0;
      if (current_roots_arr) {
        info.lastresult = current_roots_arr;
        let prec = Math.min(info.eff_decimal_precision, 10);
        info.eff_result = current_roots_arr[0].toString(10, prec) + ` ...(${n} roots)`;
      }
    };
    info.toString = function() {
      return `degree=${n}, 
      error=${this.error ? this.error.toString(10, 3) : "N/A"},
      steps=${this.steps}/${max_step}, 
      eff_prec=${this.eff_decimal_precision},
      exectime=${this.exectime}/${max_time}`;
    };
    if (n < 1) return [];
    if (n === 1) {
      let root = rawPoly[1].div(rawPoly[0]).neg();
      let res = [root];
      info.result = res;
      updateInfo(1, zero2, res);
      return res;
    }
    let a = [];
    let c0 = rawPoly[0];
    for (let i = 1; i <= n; i++) {
      a.push(rawPoly[i].div(c0));
    }
    let max_coeff_mag = zero2;
    for (let coeff of a) {
      let m = coeff.abs();
      if (m.cmp(max_coeff_mag) > 0) max_coeff_mag = m;
    }
    let radius = one2.add(max_coeff_mag);
    let current_roots = [];
    const pi = bf(Math.PI);
    const two_pi = pi.mul(bf(2));
    const offset = bf(0.7);
    for (let k = 0; k < n; k++) {
      let theta = two_pi.mul(bf(k)).div(bf(n)).add(offset);
      current_roots.push(Complex.fromPolar(radius, theta));
    }
    let max_change = zero2;
    for (let iter = 1; iter <= max_step; ++iter) {
      max_change = zero2;
      let next_roots = new Array(n);
      for (let i = 0; i < n; i++) {
        let z = current_roots[i];
        let p_val = z.add(a[0]);
        for (let j = 1; j < n; j++) {
          p_val = p_val.mul(z).add(a[j]);
        }
        let denom = new Complex(1, 0);
        for (let j = 0; j < n; j++) {
          if (i === j) continue;
          denom = denom.mul(z.sub(current_roots[j]));
        }
        let shift = p_val.div(denom);
        next_roots[i] = z.sub(shift);
        let change = shift.abs();
        if (change.cmp(max_change) > 0) {
          max_change = change;
        }
      }
      current_roots = next_roots;
      if (info.cb) {
        updateInfo(iter, max_change, current_roots);
        info.cb();
      }
      if ((/* @__PURE__ */ new Date()).getTime() - start_time > max_time) {
        updateInfo(iter, max_change, current_roots);
        console.log(`roots: Timeout after ${iter} steps.`);
        info.result = null;
        return null;
      }
      if (max_change.cmp(tol) <= 0) {
        updateInfo(iter, max_change, current_roots);
        info.result = current_roots;
        return current_roots;
      }
    }
    updateInfo(max_step, max_change, current_roots);
    console.log(`roots: Failed to converge. Error: ${info.error.toString(10, 3)}`);
    info.result = null;
    return null;
  }

  // src/fzero.js
  function fzero(f, _a, _b, info = {}) {
    let _e = info._e ?? 1e-30;
    let _re = info._re ?? _e;
    let max_step = info.max_step || 200, max_time = info.max_time || 6e4;
    if (typeof _e !== "number" || typeof _re !== "number" || typeof info !== "object") {
      throw new Error("arguments error");
    }
    const start_time = (/* @__PURE__ */ new Date()).getTime();
    const bf_zero = zero;
    const bf_one = one;
    const bf_two = two;
    const bf_three = three;
    const bf_half = half;
    let a = bf(_a);
    let b = bf(_b);
    let fa = f(a);
    let fb = f(b);
    if (fa.mul(fb).cmp(bf_zero) > 0) {
      throw new Error("Function values at the interval endpoints must differ in sign.");
    }
    let c = a;
    let fc = fa;
    let d = b.sub(a);
    let e = d;
    let tol_act = bf(_e);
    let tol_rel = bf(_re);
    const updateInfo = (iter, errorBound, residual) => {
      info.exectime = (/* @__PURE__ */ new Date()).getTime() - start_time;
      info.steps = iter;
      info.lastresult = b;
      info.residual = residual;
      info.error = errorBound;
      if (errorBound.cmp(bf_zero) === 0) {
        info.eff_decimal_precision = decimalPrecision();
      } else {
        info.eff_decimal_precision = Math.floor(-errorBound.log().f64() / Math.log(10));
      }
      if (info.eff_decimal_precision <= 0) {
        info.eff_decimal_precision = 0;
        info.eff_result = "";
      } else {
        let limit2 = decimalPrecision();
        let prec = info.eff_decimal_precision > limit2 ? limit2 : info.eff_decimal_precision;
        info.eff_result = b.toString(10, prec);
      }
    };
    info.toString = function() {
      return `root=${this.eff_result}, 
      residual=${this.residual ? this.residual.toString(10, 3) : "N/A"},
      steps=${this.steps}/${max_step}, 
      error=${this.error ? this.error.toString(10, 3) : "N/A"},
      eff_prec=${this.eff_decimal_precision}, 	  
      exectime=${this.exectime}/${max_time}`;
    };
    for (let iter = 1; iter <= max_step; ++iter) {
      if (fb.abs().cmp(fc.abs()) > 0) {
        a = c;
        fa = fc;
        c = b;
        fc = fb;
        b = a;
        fb = fa;
      }
      let tol1 = b.abs().mul(tol_rel).add(tol_act);
      let xm = c.sub(b).mul(bf_half);
      let xm_abs = xm.abs();
      if (xm_abs.cmp(tol1) <= 0 || fb.cmp(bf_zero) === 0) {
        updateInfo(iter, xm_abs, fb.abs());
        info.result = b;
        return b;
      }
      if (info.cb) {
        updateInfo(iter, xm_abs, fb.abs());
        info.cb();
      }
      if ((/* @__PURE__ */ new Date()).getTime() - start_time > max_time) {
        updateInfo(iter, xm_abs, fb.abs());
        console.log("fzero: Timeout reached.");
        info.result = null;
        return null;
      }
      if (e.abs().cmp(tol1) >= 0 && fa.abs().cmp(fb.abs()) > 0) {
        let s = fb.div(fa);
        let p, q;
        if (a.cmp(c) === 0) {
          p = xm.mul(bf_two).mul(s);
          q = bf_one.sub(s);
        } else {
          let q_iqi = fa.div(fc);
          let r_iqi = fb.div(fc);
          let term1 = xm.mul(bf_two).mul(q_iqi).mul(q_iqi.sub(r_iqi));
          let term2 = b.sub(a).mul(r_iqi.sub(bf_one));
          p = s.mul(term1.sub(term2));
          q = q_iqi.sub(bf_one).mul(r_iqi.sub(bf_one)).mul(s.sub(bf_one));
        }
        if (p.cmp(bf_zero) > 0) {
          q = q.neg();
        } else {
          p = p.neg();
        }
        q = q.abs();
        let cond1_bound = xm.mul(bf_three).mul(q).sub(tol1.mul(q).abs());
        let cond2_bound = e.mul(q).abs().mul(bf_half);
        if (p.mul(bf_two).cmp(cond1_bound) < 0 && p.cmp(cond2_bound) < 0) {
          e = d;
          d = p.div(q);
        } else {
          d = xm;
          e = d;
        }
      } else {
        d = xm;
        e = d;
      }
      a = b;
      fa = fb;
      if (d.abs().cmp(tol1) > 0) {
        b = b.add(d);
      } else {
        let signXm = xm.cmp(bf_zero) >= 0 ? bf_one : bf_one.neg();
        b = b.add(tol1.mul(signXm));
      }
      fb = f(b);
      if (fb.cmp(bf_zero) === 0 || fb.mul(fc).cmp(bf_zero) > 0) {
        c = a;
        fc = fa;
        d = b.sub(a);
        e = d;
      }
    }
    let final_xm = c.sub(b).mul(bf_half).abs();
    updateInfo(max_step, final_xm, fb.abs());
    console.log(`fzero: Failed to converge after ${max_step} steps. Residual: ${fb.toString(10, 3)}`);
    info.result = null;
    return null;
  }

  // src/quad.js
  function quad(f, _a, _b, info = {}) {
    function safeAdd(val1, val2) {
      if (val1 === null) return val2;
      if (val2 === null) return val1;
      if (typeof val1.im === "undefined" && typeof val2.im !== "undefined") {
        return val2.add(val1);
      }
      return val1.add(val2);
    }
    function updateInfoBase(info_obj) {
      if (!info_obj.rerror) {
        info_obj.eff_decimal_precision = 0;
        info_obj.eff_result = "";
        return;
      }
      if (info_obj.rerror.isAlmostZero && info_obj.rerror.isAlmostZero() || info_obj.rerror === 0) {
        info_obj.eff_decimal_precision = decimalPrecision();
      } else {
        let logRerr = info_obj.rerror.log();
        let logVal = logRerr.f64();
        info_obj.eff_decimal_precision = Math.floor(-logVal / Math.log(10));
      }
      if (info_obj.eff_decimal_precision <= 0) {
        info_obj.eff_decimal_precision = 0;
        info_obj.eff_result = "";
      } else {
        let lr = info_obj.lastresult;
        if (info_obj.eff_decimal_precision > decimalPrecision()) {
          info_obj.eff_result = lr.toString(10);
        } else {
          info_obj.eff_result = lr.toString(10, info_obj.eff_decimal_precision);
        }
      }
    }
    let max_step = info.max_step || 15, max_time = info.max_time || 6e4;
    let _e = info._e ?? 1e-50;
    let _re = info._re ?? _e;
    if (typeof _e != "number" || typeof _re != "number" || typeof info != "object") {
      throw new Error("arguments error");
    }
    let start_time = (/* @__PURE__ */ new Date()).getTime();
    info.toString = function() {
      return `lastresult=${this.lastresult ? this.lastresult.toString() : "N/A"}, 
        effective_result=${this.eff_result},
        steps=${this.steps}/${max_step}, 
        error=${this.error ? this.error.toString(10, 3) : "N/A"},
        rerror=${this.rerror ? this.rerror.toString(10, 3) : "N/A"},
        eff_decimal_precision=${this.eff_decimal_precision}, 	  
        exectime=${this.exectime}/${max_time}`;
    };
    if (Array.isArray(_a) && _b === void 0) {
      let points = _a;
      if (points.length < 2) return bf(0);
      let total_integral = null;
      let total_error = bf(0);
      let max_steps = 0;
      for (let i = 0; i < points.length - 1; i++) {
        let sub_info = Object.assign({}, info);
        delete sub_info.cb;
        delete sub_info.toString;
        let res = quad(f, points[i], points[i + 1], sub_info);
        if (res === null) {
          info.result = null;
          return null;
        }
        total_integral = safeAdd(total_integral, res);
        total_error = total_error.add(sub_info.error);
        if (sub_info.steps > max_steps) max_steps = sub_info.steps;
      }
      info.exectime = (/* @__PURE__ */ new Date()).getTime() - start_time;
      info.steps = max_steps;
      info.error = total_error;
      info.lastresult = total_integral;
      info.result = total_integral;
      let abs_val = total_integral.abs();
      info.rerror = abs_val.isAlmostZero() ? total_error : total_error.div(abs_val);
      updateInfoBase(info);
      return total_integral;
    }
    function parseBound(val) {
      if (val === "-Infinity" || val === "Infinity") return val;
      if (typeof val === "string") {
        if (val.includes("i") || val.includes("I")) return Complex.fromString(val);
      }
      if (typeof val === "number" || typeof val === "string") return bf(val);
      return val;
    }
    let a = parseBound(_a);
    let b = parseBound(_b);
    let a_str = a.toString();
    let b_str = b.toString();
    let a_is_inf = a_str === "-Infinity" || a_str === "Infinity";
    let b_is_inf = b_str === "-Infinity" || b_str === "Infinity";
    let e = bf(_e), re = bf(_re);
    let sign2 = 1;
    if (a_is_inf || b_is_inf) {
      let is_greater = false;
      if (a_str === "Infinity" && b_str !== "Infinity") is_greater = true;
      if (a_str !== "-Infinity" && b_str === "-Infinity") is_greater = true;
      if (a_str === "Infinity" && b_str === "-Infinity") is_greater = true;
      if (is_greater) {
        let tmp = a;
        a = b;
        b = tmp;
        let tmp_inf = a_is_inf;
        a_is_inf = b_is_inf;
        b_is_inf = tmp_inf;
        let tmp_str = a_str;
        a_str = b_str;
        b_str = tmp_str;
        sign2 = -1;
      }
    } else {
      if (a.im !== void 0 || b.im !== void 0) {
        if (a.im === void 0) a = new Complex(a);
        if (b.im === void 0) b = new Complex(b);
      }
    }
    const PI2 = PI;
    const f0p5 = half;
    const bf_0 = zero;
    const bf_1 = one;
    const bf_2 = two;
    const bf_m2 = bf(-2);
    const pi_over_2 = PI2.mul(f0p5);
    function isAtBoundary(x) {
      if (!a_is_inf && a !== void 0 && a !== null) {
        if (x.equals(a)) return true;
      }
      if (!b_is_inf && b !== void 0 && b !== null) {
        if (x.equals(b)) return true;
      }
      return false;
    }
    let calc_x_w = null;
    if (!a_is_inf && !b_is_inf) {
      let m = a.add(b).mul(f0p5);
      let c = b.sub(a).mul(f0p5);
      let c2 = b.sub(a);
      calc_x_w = (v, dv) => {
        let ch = v.cosh();
        let w = c.mul(dv).div(ch.mul(ch));
        let x;
        if (typeof v.im === "undefined" && typeof a.im === "undefined" && typeof b.im === "undefined") {
          let sign_v = v.toNumber();
          if (sign_v > 0) {
            let exp_2v = v.mul(bf_2).exp();
            let offset = c2.div(exp_2v.add(bf_1));
            x = b.sub(offset);
          } else if (sign_v < 0) {
            let exp_m2v = v.mul(bf_m2).exp();
            let offset = c2.div(exp_m2v.add(bf_1));
            x = a.add(offset);
          } else {
            x = m;
          }
        } else {
          let th = v.tanh();
          x = m.add(c.mul(th));
        }
        return { x, w };
      };
    } else if (!a_is_inf && b_str === "Infinity") {
      calc_x_w = (v, dv) => {
        let ev = v.exp();
        let x = a.add(ev);
        let w = ev.mul(dv);
        return { x, w };
      };
    } else if (a_str === "-Infinity" && !b_is_inf) {
      calc_x_w = (v, dv) => {
        let ev = v.exp();
        let x = b.sub(ev);
        let w = ev.mul(dv);
        return { x, w };
      };
    } else if (a_str === "-Infinity" && b_str === "Infinity") {
      calc_x_w = (v, dv) => {
        let x = v.sinh();
        let w = v.cosh().mul(dv);
        return { x, w };
      };
    } else {
      info.result = bf_0;
      updateInfoBase(info);
      return info.result;
    }
    function eval_sum(h, k_start, k_step) {
      let sum = null;
      let k = k_start;
      let consecutive_zeros = 0;
      if (k === 0) {
        let p0 = calc_x_w(bf_0, pi_over_2);
        if (!p0.w.isAlmostZero() && !isAtBoundary(p0.x)) {
          sum = f(p0.x).mul(p0.w);
        } else {
          sum = bf_0;
        }
        k += k_step;
      }
      while (true) {
        let t = h.mul(k);
        let v_plus = t.sinh().mul(pi_over_2);
        let dv_plus = t.cosh().mul(pi_over_2);
        let p_plus = calc_x_w(v_plus, dv_plus);
        let p_minus = calc_x_w(v_plus.neg(), dv_plus);
        let term_plus;
        if (p_plus.w.isAlmostZero() || isAtBoundary(p_plus.x)) {
          term_plus = bf_0;
        } else {
          term_plus = f(p_plus.x).mul(p_plus.w);
        }
        let term_minus;
        if (p_minus.w.isAlmostZero() || isAtBoundary(p_minus.x)) {
          term_minus = bf_0;
        } else {
          term_minus = f(p_minus.x).mul(p_minus.w);
        }
        let term_sum = safeAdd(term_plus, term_minus);
        sum = safeAdd(sum, term_sum);
        if (term_plus.isAlmostZero() && term_minus.isAlmostZero() || p_plus.w.isAlmostZero() && p_minus.w.isAlmostZero()) {
          consecutive_zeros++;
          if (consecutive_zeros > 3) break;
        } else {
          consecutive_zeros = 0;
        }
        if (k > 1e5) break;
        k += k_step;
      }
      return sum === null ? bf_0 : sum;
    }
    let h0 = one;
    let T = eval_sum(h0, 0, 1).mul(h0);
    for (let m = 1; m <= max_step; ++m) {
      let h = one.div(bf(2 ** m));
      let sum_new = eval_sum(h, 1, 2);
      let Tm = safeAdd(T.mul(f0p5), sum_new.mul(h));
      let diff2 = typeof Tm.im === "undefined" && typeof T.im !== "undefined" ? T.sub(Tm) : Tm.sub(T);
      let err = diff2.abs();
      let rerr = Tm.isAlmostZero() ? err : err.div(Tm.abs());
      if (!!info.debug && m > 2) {
        console.log("Level[" + m + "]=" + Tm.toString());
        console.log("Error: " + err.toString(10, 3));
      }
      info.exectime = (/* @__PURE__ */ new Date()).getTime() - start_time;
      info.lastresult = sign2 < 0 ? Tm.neg() : Tm;
      info.steps = m;
      info.error = err;
      info.rerror = rerr;
      if (m > 3 && (err.cmp(e) <= 0 || rerr.cmp(re) <= 0)) {
        info.result = info.lastresult;
        updateInfoBase(info);
        return info.result;
      } else if (m == max_step || info.exectime > max_time) {
        updateInfoBase(info);
        info.result = null;
        return info.result;
      }
      if (info.cb) {
        updateInfoBase(info);
        info.cb();
      }
      T = Tm;
    }
  }

  // src/romberg.js
  function romberg(f, _a, _b, info = {}) {
    let max_step = info.max_step || 25, max_acc = info.max_acc || 12, max_time = info.max_time || 6e4;
    let _e = info._e ?? 1e-30;
    let _re = info._re ?? _e;
    if (typeof _e != "number" || typeof _re != "number" || typeof info != "object") {
      throw new Error("arguments error");
    }
    let start_time = (/* @__PURE__ */ new Date()).getTime();
    info.toString = function() {
      return `lastresult=${this.lastresult}, 
        effective_result=${this.eff_result},
        steps=${this.steps}/${max_step}, 
        error=${this.error.toString(10, 3)},
        rerror=${this.rerror.toString(10, 3)},
        eff_decimal_precision=${this.eff_decimal_precision}, 	  
        exectime=${this.exectime}/${max_time}`;
    };
    let a = bf(_a), b = bf(_b), e = bf(_e), re = bf(_re);
    let sign2 = b.cmp(a);
    if (sign2 < 0) {
      let tmp = a;
      a = b;
      b = tmp;
    }
    var updateInfo = () => {
      if (info.rerror.isZero()) {
        info.eff_decimal_precision = decimalPrecision();
      } else {
        info.eff_decimal_precision = Math.floor(-info.rerror.log().f64() / Math.log(10));
      }
      if (info.eff_decimal_precision <= 0) {
        info.eff_decimal_precision = 0;
        info.eff_result = "";
      } else {
        if (info.eff_decimal_precision > decimalPrecision()) {
          info.eff_result = info.lastresult.toString(10);
        } else {
          info.eff_result = info.lastresult.toString(10, info.eff_decimal_precision);
        }
      }
    };
    const f0p5 = bf(0.5);
    const b_a_d = b.sub(a).mul(f0p5);
    let T = [0, b_a_d.mul(f(a).add(f(b)))];
    for (let m = 2; m <= max_step; ++m) {
      let Tm = [];
      let sum = bf(0);
      for (let i = 0; i < 2 ** (m - 2); ++i) {
        sum.setadd(sum, f(a.add(b_a_d.mul(i * 2 + 1))));
      }
      Tm[1] = T[1].mul(f0p5).add(b_a_d.mul(sum));
      b_a_d.setmul(b_a_d, f0p5);
      for (let j = 2; j <= max_acc && j <= m; ++j) {
        let c = bf(4 ** (j - 1)), c1 = bf(4 ** (j - 1) - 1);
        Tm[j] = Tm[j - 1].mul(c).sub(T[j - 1]).div(c1);
      }
      let err = Tm[Tm.length - 1].sub(T[T.length - 1]).abs();
      let rerr;
      if (!Tm[Tm.length - 1].isZero()) {
        rerr = err.div(Tm[Tm.length - 1].abs());
      } else {
        rerr = err;
      }
      if (!!info.debug && m > 5) {
        console.log("R[" + m + "]=" + Tm[4]);
        console.log(err.toString(10, 3));
      }
      info.exectime = (/* @__PURE__ */ new Date()).getTime() - start_time;
      info.lastresult = Tm[Tm.length - 1];
      if (sign2 < 0) {
        info.lastresult = info.lastresult.neg();
      }
      info.steps = m;
      info.error = err;
      info.rerror = rerr;
      if (m > 5 && (err.cmp(e) <= 0 || rerr.cmp(re) <= 0)) {
        info.result = info.lastresult;
        updateInfo();
        return info.result;
      } else if (m == max_step || info.exectime > max_time) {
        updateInfo();
        info.result = null;
        return info.result;
      }
      if (info.cb) {
        updateInfo();
        info.cb();
      }
      T = Tm;
    }
  }

  // src/limit.js
  function limit(f, point, info = {}) {
    let max_step = info.max_step || 100, max_acc = info.max_acc || 15, max_time = info.max_time || 6e4;
    let _e = info._e ?? 1e-30;
    let _re = info._re ?? _e;
    let useExp = !!info.useExp;
    let baseNumber = typeof info.useExp == "number" ? info.useExp : 2;
    let direction = info.direction || 1;
    if (typeof _e != "number" || typeof _re != "number" || typeof info != "object") {
      throw new Error("arguments error");
    }
    let start_time = (/* @__PURE__ */ new Date()).getTime();
    info.toString = function() {
      return `lastresult=${this.lastresult}, 
        effective_result=${this.eff_result},
        steps=${this.steps}/${max_step}, 
        error=${this.error ? this.error.toString(10, 3) : "N/A"},
        rerror=${this.rerror ? this.rerror.toString(10, 3) : "N/A"},
        eff_decimal_precision=${this.eff_decimal_precision}, 
        exectime=${this.exectime}/${max_time}`;
    };
    let target;
    let func = f;
    let h;
    let pointStr = String(point).toLowerCase();
    if (pointStr === "inf" || pointStr === "+inf" || pointStr === "infinity" || pointStr === "+infinity") {
      target = bf(0);
      if (useExp) {
        func = (t) => f(pow(baseNumber, bf(1).div(t)));
      } else {
        func = (t) => f(bf(1).div(t));
      }
      h = bf(1);
    } else if (pointStr === "-inf" || pointStr === "-infinity") {
      target = bf(0);
      if (useExp) {
        func = (t) => f(pow(baseNumber, bf(1).div(t)).neg());
      } else {
        func = (t) => f(bf(-1).div(t));
      }
      h = bf(1);
    } else {
      target = bf(point);
      let bfDirection = bf(direction);
      if (useExp) {
        func = (t) => {
          let displacement = pow(baseNumber, bf(1).div(t).neg());
          return f(target.add(bfDirection.mul(displacement)));
        };
        h = bf(1);
      } else {
        func = f;
        h = bf(direction);
      }
    }
    let e = bf(_e), re = bf(_re);
    let updateInfo = () => {
      let prec = 0;
      if (!info.rerror || info.rerror.isZero() || info.rerror.isNaN()) {
        prec = decimalPrecision();
      } else {
        try {
          let logErr = info.rerror.log();
          if (logErr.isFinite()) {
            prec = Math.floor(-logErr.f64() / Math.log(10));
          } else {
            prec = decimalPrecision();
          }
        } catch (err) {
          prec = decimalPrecision();
        }
      }
      if (!Number.isFinite(prec) || prec <= 0) {
        prec = 0;
        info.eff_decimal_precision = 0;
        info.eff_result = "";
      } else {
        info.eff_decimal_precision = prec;
        if (info.eff_decimal_precision > decimalPrecision()) {
          info.eff_result = info.lastresult.toString(10);
        } else {
          info.eff_result = info.lastresult.toString(10, info.eff_decimal_precision);
        }
      }
    };
    let T = [];
    let x0 = target.add(h);
    T[0] = func(x0);
    info.lastresult = T[0];
    info.error = bf(1e100);
    info.rerror = bf(1e100);
    let globalBest = T[0];
    let globalMinErr = bf(1e100);
    for (let m = 1; m <= max_step; ++m) {
      let Tm = [];
      h.setdiv(h, bf(2));
      let x = target.add(h);
      Tm[0] = func(x);
      for (let j = 1; j <= max_acc && j <= m; ++j) {
        let factor = bf(2).pow(j);
        let denom = factor.sub(bf(1));
        let num = Tm[j - 1].mul(factor).sub(T[j - 1]);
        Tm[j] = num.div(denom);
      }
      let bestEst = Tm[0];
      let minErr = Tm[0].sub(T[0]).abs();
      for (let j = 1; j < Tm.length; j++) {
        if (j - 1 < T.length) {
          let est = Tm[j].sub(T[j - 1]).abs();
          if (!est.isNaN() && est.cmp(minErr) < 0) {
            minErr = est;
            bestEst = Tm[j];
          }
        }
      }
      let err = minErr;
      let rerr;
      if (!bestEst.isZero()) {
        rerr = err.div(bestEst.abs());
      } else {
        rerr = err;
      }
      if (!!info.debug && m > 2) {
        console.log(`Limit[${m}]: val=${bestEst.toString(10, 10)}, err=${err.toString(10, 3)}`);
      }
      info.exectime = (/* @__PURE__ */ new Date()).getTime() - start_time;
      info.lastresult = bestEst;
      info.steps = m;
      info.error = err;
      info.rerror = rerr;
      if (err.cmp(globalMinErr) < 0) {
        globalMinErr = err;
        globalBest = bestEst;
      }
      if (m > 3 && (err.cmp(e) <= 0 || rerr.cmp(re) <= 0)) {
        info.result = info.lastresult;
        updateInfo();
        return info.result;
      } else if (m == max_step || info.exectime > max_time) {
        updateInfo();
        info.result = null;
        return info.result;
      }
      if (info.cb) {
        updateInfo();
        info.cb();
      }
      T = Tm;
    }
    return null;
  }
  function diff(f, x, n = 1, info = {}) {
    const _x = bf(x);
    const order = Math.floor(n);
    const isSingular = !!info.singular;
    if (order < 0) {
      throw new Error("Derivative order must be a non-negative integer.");
    }
    if (order === 0) return f(_x);
    const binom = [bf(1)];
    for (let i = 1; i <= order; i++) {
      let prev = binom[i - 1];
      let val = prev.mul(bf(order - i + 1)).div(bf(i));
      binom.push(val);
    }
    const differenceQuotient = (h) => {
      let sum = bf(0);
      let offset = isSingular ? 1 : 0;
      for (let k = 0; k <= order; k++) {
        let step = bf(k + offset).mul(h);
        let samplePoint = _x.add(step);
        let term = f(samplePoint).mul(binom[k]);
        if ((order - k) % 2 === 1) {
          sum.setsub(sum, term);
        } else {
          sum.setadd(sum, term);
        }
      }
      return sum.div(h.pow(bf(order)));
    };
    return limit(differenceQuotient, 0, info);
  }

  // src/nsum.js
  function nsum(f, range, info = {}) {
    let max_step = info.max_step || 20, max_acc = info.max_acc || 15, max_time = info.max_time || 6e4;
    let _e = info._e ?? 1e-30;
    let _re = info._re ?? _e;
    if (typeof _e != "number" || typeof _re != "number" || typeof info != "object" || !Array.isArray(range)) {
      throw new Error("arguments error: invalid info object or range array");
    }
    let start_time = (/* @__PURE__ */ new Date()).getTime();
    info.toString = function() {
      return `lastresult=${this.lastresult}, 
        effective_result=${this.eff_result},
        steps=${this.steps}/${max_step}, 
        terms_eval=${this.terms_count},
        error=${this.error ? this.error.toString(10, 3) : "N/A"},
        rerror=${this.rerror ? this.rerror.toString(10, 3) : "N/A"},
        eff_decimal_precision=${this.eff_decimal_precision}, 
        exectime=${this.exectime}/${max_time}`;
    };
    let n_start = bf(range[0]);
    let end_val = range[1];
    let isInfinite = false;
    let n_end = null;
    let endStr = String(end_val).toLowerCase();
    if (endStr === "inf" || endStr === "+inf" || endStr === "infinity" || endStr === "+infinity") {
      isInfinite = true;
    } else {
      n_end = bf(end_val);
    }
    let e = bf(_e), re = bf(_re);
    let updateInfo = () => {
      if (!info.rerror || info.rerror.isZero()) {
        info.eff_decimal_precision = decimalPrecision();
      } else {
        info.eff_decimal_precision = Math.floor(-info.rerror.log().f64() / Math.log(10));
      }
      if (info.eff_decimal_precision <= 0) {
        info.eff_decimal_precision = 0;
        info.eff_result = "";
      } else {
        if (info.eff_decimal_precision > decimalPrecision()) {
          info.eff_result = info.lastresult.toString(10);
        } else {
          info.eff_result = info.lastresult.toString(10, info.eff_decimal_precision);
        }
      }
    };
    let T = [];
    let current_partial_sum = bf(0);
    let current_n = bf(n_start);
    let terms_evaluated = 0;
    if (!isInfinite && current_n.cmp(n_end) > 0) {
      info.result = bf(0);
      return info.result;
    }
    let term0 = f(current_n);
    current_partial_sum = term0.clone();
    current_n.setadd(current_n, bf(1));
    terms_evaluated++;
    T[0] = current_partial_sum;
    info.lastresult = T[0];
    info.error = bf(1e100);
    info.rerror = bf(1e100);
    info.terms_count = terms_evaluated;
    if (!isInfinite && current_n.cmp(n_end) > 0) {
      info.result = current_partial_sum;
      info.steps = 0;
      info.error = bf(0);
      info.rerror = bf(0);
      updateInfo();
      return info.result;
    }
    for (let m = 1; m <= max_step; ++m) {
      let Tm = [];
      let count_to_add = Math.pow(2, m - 1);
      let stop_iteration = false;
      for (let k = 0; k < count_to_add; k++) {
        if (k % 1e3 === 0 && (/* @__PURE__ */ new Date()).getTime() - start_time > max_time) {
          break;
        }
        if (!isInfinite && current_n.cmp(n_end) > 0) {
          stop_iteration = true;
          break;
        }
        let term = f(current_n);
        current_partial_sum.setadd(current_partial_sum, term);
        current_n.setadd(current_n, one);
        terms_evaluated++;
      }
      info.terms_count = terms_evaluated;
      if (stop_iteration) {
        info.result = current_partial_sum;
        info.steps = m;
        info.error = bf(0);
        info.rerror = bf(0);
        info.exectime = (/* @__PURE__ */ new Date()).getTime() - start_time;
        updateInfo();
        return info.result;
      }
      Tm[0] = bf(current_partial_sum);
      for (let j = 1; j <= max_acc && j <= m; ++j) {
        let factor = two.pow(j);
        let denom = factor.sub(one);
        let num = Tm[j - 1].mul(factor).sub(T[j - 1]);
        Tm[j] = num.div(denom);
      }
      let lastIdx = Tm.length - 1;
      let bestEst = Tm[lastIdx];
      let err = bestEst.sub(T[T.length - 1]).abs();
      let rerr;
      if (!bestEst.isZero()) {
        rerr = err.div(bestEst.abs());
      } else {
        rerr = err;
      }
      info.exectime = (/* @__PURE__ */ new Date()).getTime() - start_time;
      info.lastresult = bestEst;
      info.steps = m;
      info.error = err;
      info.rerror = rerr;
      if (!!info.debug && m > 2) {
        console.log(`NSum[${m}]: terms=${terms_evaluated}, val=${bestEst.toString(10, 10)}, err=${err.toString(10, 3)}`);
      }
      if (m > 3 && (err.cmp(e) <= 0 || rerr.cmp(re) <= 0)) {
        info.result = info.lastresult;
        updateInfo();
        return info.result;
      } else if (m == max_step || info.exectime > max_time) {
        updateInfo();
        info.result = null;
        return info.result;
      }
      if (info.cb) {
        updateInfo();
        info.cb();
      }
      T = Tm;
    }
    return null;
  }

  // src/shanks.js
  function shanks(f, info = {}) {
    let max_step = info.max_step || 200, max_time = info.max_time || 3e4;
    let isArray = Array.isArray(f);
    if (isArray) {
      max_step = Math.min(max_step, f.length - 1);
      let fa = f;
      f = (n) => fa[n] instanceof BigFloat ? fa[n] : bf(fa[n]);
    }
    let _e = info._e ?? 1e-30;
    let _re = info._re ?? _e;
    if (typeof _e != "number" || typeof _re != "number" || typeof info != "object") {
      throw new Error("arguments error");
    }
    let start_time = (/* @__PURE__ */ new Date()).getTime();
    info.toString = function() {
      return `lastresult=${this.lastresult}, 
        effective_result=${this.eff_result},
        steps=${this.steps}/${max_step}, 
        error=${this.error ? this.error.toString(10, 3) : "N/A"},
        rerror=${this.rerror ? this.rerror.toString(10, 3) : "N/A"},
        eff_decimal_precision=${this.eff_decimal_precision}, 
        exectime=${this.exectime}/${max_time}`;
    };
    let e = bf(_e), re = bf(_re);
    let one2 = bf(1);
    let updateInfo = () => {
      let prec = 0;
      if (!info.rerror || info.rerror.isZero() || info.rerror.isNaN()) {
        prec = decimalPrecision();
      } else {
        try {
          let logErr = info.rerror.log();
          if (logErr.isFinite()) {
            prec = Math.floor(-logErr.f64() / Math.log(10));
          } else {
            prec = decimalPrecision();
          }
        } catch (err) {
          prec = decimalPrecision();
        }
      }
      if (!Number.isFinite(prec) || prec <= 0) {
        prec = 0;
        info.eff_decimal_precision = 0;
        info.eff_result = "";
      } else {
        info.eff_decimal_precision = prec;
        if (info.eff_decimal_precision > decimalPrecision()) {
          info.eff_result = info.lastresult.toString(10);
        } else {
          info.eff_result = info.lastresult.toString(10, info.eff_decimal_precision);
        }
      }
    };
    let table = [];
    info.lastresult = bf(0);
    info.error = bf(1e100);
    info.rerror = bf(1e100);
    for (let n = 0; n <= max_step; ++n) {
      let currentRow = [];
      let val = f(n);
      currentRow[0] = val;
      for (let k = 1; k <= n; ++k) {
        let prevRow = table[n - 1];
        let diff2 = currentRow[k - 1].sub(prevRow[k - 1]);
        if (diff2.isZero()) {
          currentRow[k] = currentRow[k - 1];
          break;
        }
        let term2 = one2.div(diff2);
        let term1;
        if (k === 1) {
          term1 = bf(0);
        } else {
          term1 = prevRow[k - 2];
        }
        currentRow[k] = term1.add(term2);
      }
      table[n] = currentRow;
      let bestEst = currentRow[0];
      let currentMinErr = bf(1e100);
      if (n > 0) {
        currentMinErr = currentRow[0].sub(table[n - 1][0]).abs();
      }
      for (let k = 2; k < currentRow.length; k += 2) {
        if (!currentRow[k]) continue;
        if (table[n - 1] && table[n - 1].length > k) {
          let estErr = currentRow[k].sub(table[n - 1][k]).abs();
          if (!estErr.isNaN() && estErr.cmp(currentMinErr) < 0) {
            currentMinErr = estErr;
            bestEst = currentRow[k];
          }
        }
      }
      let rerr;
      if (!bestEst.isZero()) {
        rerr = currentMinErr.div(bestEst.abs());
      } else {
        rerr = currentMinErr;
      }
      info.exectime = (/* @__PURE__ */ new Date()).getTime() - start_time;
      info.lastresult = bestEst;
      info.steps = n;
      info.error = currentMinErr;
      info.rerror = rerr;
      if (!!info.debug && n > 2) {
        console.log(`Shanks[${n}]: val=${bestEst.toString(10, 10)}, err=${currentMinErr.toString(10, 3)}`);
      }
      if (!isArray && n > 2 && (currentMinErr.cmp(e) <= 0 || rerr.cmp(re) <= 0)) {
        info.result = info.lastresult;
        updateInfo();
        return info.result;
      }
      if (info.exectime > max_time) {
        updateInfo();
        info.result = null;
        return info.result;
      }
      if (info.cb) {
        updateInfo();
        info.cb();
      }
    }
    updateInfo();
    if (isArray) {
      info.result = info.lastresult;
      return info.result;
    }
    info.result = null;
    return null;
  }

  // src/identify.js
  function identify(_x, options = {}) {
    const x = bf(_x);
    if (x.isZero()) return "0";
    const tol = bf(options.tol || "1e-25");
    const max_den = options.max_den || 1e6;
    const base_constants = options.constants || [
      { name: "PI", val: PI },
      { name: "E", val: E },
      { name: "SQRT2", val: bf(2).sqrt() },
      { name: "LN2", val: bf(2).log() },
      { name: "PHI", val: bf(5).sqrt().add(1).div(2) }
      // Golden Ratio
    ];
    function toRational(v, max_q) {
      let val = bf(v);
      let sign2 = val.cmp(0) < 0 ? -1 : 1;
      val = val.abs();
      let h0 = bf(0), h1 = bf(1);
      let k0 = bf(1), k1 = bf(0);
      let x_n = val;
      for (let i = 0; i < 50; i++) {
        let a_n = x_n.floor();
        let h_next = a_n.mul(h1).add(h0);
        let k_next = a_n.mul(k1).add(k0);
        if (k_next.cmp(max_q) > 0) break;
        h0 = h1;
        h1 = h_next;
        k0 = k1;
        k1 = k_next;
        let current_val = h1.div(k1);
        let diff2 = val.sub(current_val).abs();
        if (diff2.cmp(tol) < 0) {
          return { p: h1.mul(sign2), q: k1, diff: diff2 };
        }
        let residue = x_n.sub(a_n);
        if (residue.isZero() || i > 40) break;
        x_n = bf(1).div(residue);
      }
      return { p: h1.mul(sign2), q: k1, diff: val.sub(h1.div(k1)).abs() };
    }
    function format(p, q, constName) {
      let ps = p.toString(10);
      let qs = q.toString(10);
      let res = "";
      if (constName) {
        if (ps === "1") res = constName;
        else if (ps === "-1") res = "-" + constName;
        else res = ps + "*" + constName;
      } else {
        res = ps;
      }
      if (qs !== "1") {
        return `(${res})/${qs}`;
      }
      return res;
    }
    let rat = toRational(x, max_den);
    if (rat.diff.cmp(tol) < 0) return format(rat.p, rat.q);
    for (let c of base_constants) {
      let ratC = toRational(x.div(c.val), max_den);
      if (ratC.diff.cmp(tol) < 0) return format(ratC.p, ratC.q, c.name);
    }
    for (let c of base_constants) {
      let ratI = toRational(c.val.div(x), max_den);
      if (ratI.diff.cmp(tol) < 0) return format(ratI.q, ratI.p, c.name);
    }
    let x2 = x.mul(x);
    let rat2 = toRational(x2, max_den);
    if (rat2.diff.cmp(tol) < 0) {
      return `sqrt(${format(rat2.p, rat2.q)})`;
    }
    for (let c of base_constants) {
      for (let r = -5; r <= 5; r++) {
        if (r === 0) continue;
        let target = x.sub(r);
        let ratT = toRational(target.div(c.val), 1e3);
        if (ratT.diff.cmp(tol) < 0) {
          let term1 = format(ratT.p, ratT.q, c.name);
          return `${term1}${r > 0 ? "+" : ""}${r}`;
        }
      }
    }
    try {
      let lx = x.log();
      let ratL = toRational(lx, 1e3);
      if (ratL.diff.cmp(tol) < 0) return `exp(${format(ratL.p, ratL.q)})`;
    } catch (e) {
    }
    return x.toString(10);
  }

  // src/bernoulli.js
  var B_CACHE = [];
  function clearBernoulliCache() {
    B_CACHE = [];
  }
  function bernoulli(n) {
    if (B_CACHE.length == 0) {
      B_CACHE.push(new BigFloat(bf(1), 10, false, true));
      B_CACHE.push(new BigFloat(bf(-1).div(2), 10, false, true));
    }
    if (n < 0) throw new Error("Index must be non-negative");
    if (n > 1 && n % 2 !== 0) return bf(0);
    if (B_CACHE[n] !== void 0) return B_CACHE[n];
    if (n < 40) {
      let s = bf(0);
      const n_plus_1 = n + 1;
      let binom = bf(1);
      for (let k2 = 0; k2 < n; k2++) {
        s = s.add(binom.mul(bernoulli(k2)));
        binom = binom.mul(bf(n_plus_1 - k2)).div(bf(k2 + 1));
      }
      const res2 = s.div(bf(n_plus_1)).neg();
      B_CACHE[n] = res2;
      return res2;
    }
    const k = n / 2;
    const pi = PI;
    const twoPi = pi.mul(bf(2));
    let fact = bf(1);
    for (let i = 2; i <= n; i++) fact = fact.mul(bf(i));
    let z = bf(1);
    const eps = bf(getEpsilon() * 1e-3);
    for (let m = 2; m < 1e3; m++) {
      const term = bf(m).pow(bf(-n));
      z = z.add(term);
      if (term.cmp(eps) < 0) break;
    }
    let res = bf(2).mul(fact).mul(z).div(twoPi.pow(bf(n)));
    if ((k + 1) % 2 !== 0) res = res.neg();
    B_CACHE[n] = new BigFloat(res, 10, false, true);
    return B_CACHE[n];
  }

  // src/gamma.js
  function stirlingSeries(z, numTerms) {
    const zInv = new Complex(1).div(z);
    const zInvSq = zInv.mul(zInv);
    let termPow = zInv;
    let sum = new Complex(0);
    for (let k = 1; k <= numTerms; k++) {
      const n = 2 * k;
      const b = bernoulli(n);
      const denom = bf(n).mul(bf(n - 1));
      const term = termPow.mul(new Complex(b.div(denom)));
      sum = sum.add(term);
      termPow = termPow.mul(zInvSq);
    }
    return sum;
  }
  function logGamma(z) {
    let _z = z instanceof Complex ? z : new Complex(z);
    const prec = decimalPrecision();
    const pi = PI;
    if (_z.im.isZero() && _z.re.cmp(zero) <= 0 && _z.re.round().equals(_z.re)) {
      return new Complex(Infinity, 0);
    }
    if (_z.re.cmp(half) < 0) {
      const c_pi = new Complex(pi);
      const sinPiZ = _z.mul(c_pi).sin();
      return c_pi.log().sub(sinPiZ.log()).sub(logGamma(new Complex(1).sub(_z)));
    }
    let currentZ = _z;
    let shiftLogSum = new Complex(0);
    const threshold = bf(Math.floor(prec * 0.6) + 10);
    while (currentZ.re.cmp(threshold) < 0) {
      shiftLogSum = shiftLogSum.add(currentZ.log());
      currentZ = currentZ.add(new Complex(1));
    }
    const lnSqrt2Pi = bf(2).mul(pi).log().mul(half);
    const numTerms = Math.floor(prec * 0.4) + 2;
    let res = currentZ.sub(new Complex(0.5)).mul(currentZ.log()).sub(currentZ).add(new Complex(lnSqrt2Pi)).add(stirlingSeries(currentZ, numTerms));
    return res.sub(shiftLogSum);
  }
  function gamma(z) {
    const _z = z instanceof Complex ? z : new Complex(z);
    if (_z.im.isZero() && _z.re.cmp(zero) > 0) {
      const val = _z.re.toNumber();
      if (Number.isInteger(val) && val < 50) {
        let res = bf(1);
        for (let i = 1; i < val; i++) res = res.mul(bf(i));
        return new Complex(res);
      }
    }
    if (_z.im.isExactZero() && _z.re.cmp(zero) <= 0 && _z.re.floor().equals(_z.re)) {
      throw new Error("Gamma function pole at " + _z.re.toString());
    }
    return logGamma(_z).exp();
  }
  function factorial(n) {
    if (Number.isInteger(n) && n > 0) {
      let ret2 = bf(1);
      for (let i = 2; i <= n; ++i) {
        ret2.setmul(ret2, i);
      }
      return ret2;
    }
    const _n = n instanceof Complex ? n : new Complex(n);
    let ret = gamma(_n.add(new Complex(1)));
    if (n instanceof Complex) {
      return ret;
    }
    return ret.re;
  }
  function beta(x, y) {
    const _x = x instanceof Complex ? x : new Complex(x);
    const _y = y instanceof Complex ? y : new Complex(y);
    const logB = logGamma(_x).add(logGamma(_y)).sub(logGamma(_x.add(_y)));
    return logB.exp();
  }

  // src/zeta.js
  function zeta(s, a = "1") {
    let _s = s instanceof Complex ? s : new Complex(s);
    let _a = a instanceof Complex ? a : new Complex(a);
    const prec = decimalPrecision();
    if (_s.re.equals(one) && _s.im.isZero()) {
      return new Complex(Infinity, 0);
    }
    if (_a.re.equals(one) && _a.im.isZero() && _s.re.cmp(zero) < 0) {
      const c_pi = new Complex(PI);
      const c_one = new Complex(one);
      const log2 = new Complex(2).log();
      const logPi = c_pi.log();
      const term_sLog2 = _s.mul(log2);
      const term_sMinus1LogPi = _s.sub(c_one).mul(logPi);
      const term_logGamma = logGamma(c_one.sub(_s));
      const expTerm = term_sLog2.add(term_sMinus1LogPi).add(term_logGamma).exp();
      const sinTerm = _s.mul(c_pi.div(new Complex(2))).sin();
      const zetaTerm = zeta(c_one.sub(_s));
      return expTerm.mul(sinTerm).mul(zetaTerm);
    }
    if (_a.re.cmp(new Complex(0.5).re) === 0 && _a.im.isZero()) {
      const c_two = new Complex(2);
      const c_one = new Complex(1);
      return c_two.pow(_s).sub(c_one).mul(zeta(_s, 1));
    }
    const absS = _s.abs().toNumber();
    const N = Math.floor(prec * 0.6) + Math.floor(absS * 0.5) + 15;
    const M = Math.floor(prec * 0.4) + Math.floor(absS * 0.1) + 5;
    return zetaEulerMaclaurin(_s, _a, N, M);
  }
  function altZeta(s) {
    const _s = s instanceof Complex ? s : new Complex(s);
    const c_one = new Complex(one);
    if (_s.re.equals(one) && _s.im.isZero()) {
      return new Complex(2).log();
    }
    const c_two = new Complex(2);
    const exponent = c_one.sub(_s);
    const term = c_one.sub(c_two.pow(exponent));
    return term.mul(zeta(_s));
  }
  function zetaEulerMaclaurin(s, a, N, M) {
    const c_one = new Complex(one);
    const negS = s.neg();
    let sum1 = new Complex(0);
    for (let k = 0; k < N; k++) {
      sum1 = sum1.add(a.add(new Complex(k)).pow(negS));
    }
    const X2 = a.add(new Complex(N));
    const sMinus1 = s.sub(c_one);
    const sum2 = X2.pow(c_one.sub(s)).div(sMinus1);
    const sum3 = X2.pow(negS).mul(new Complex(0.5));
    let sum4 = new Complex(0);
    let Xpow = X2.pow(negS.sub(c_one));
    const XinvSq = c_one.div(X2.mul(X2));
    let falling = s;
    for (let k = 1; k <= M; k++) {
      const n = 2 * k;
      const bk = bernoulli(n);
      const denom = factorial(n);
      const coeff = new Complex(bk.div(denom));
      const term = Xpow.mul(coeff).mul(falling);
      sum4 = sum4.add(term);
      if (k < M) {
        falling = falling.mul(s.add(new Complex(2 * k - 1)));
        falling = falling.mul(s.add(new Complex(2 * k)));
        Xpow = Xpow.mul(XinvSq);
      }
    }
    return sum1.add(sum2).add(sum3).add(sum4);
  }
  function primeZeta(s) {
    const _s = s instanceof Complex ? s : new Complex(s);
    if (_s.re.cmp(one) <= 0) {
      throw new Error("PrimeZeta diverges for Re(s) <= 1");
    }
    const prec = decimalPrecision();
    const maxTerms = Math.floor(prec * 1.2) + 20;
    let sum = new Complex(0);
    for (let n = 1; n < maxTerms; n++) {
      const mu = getMobius(n);
      if (mu === 0) continue;
      const ns = _s.mul(new Complex(n));
      const logZ = zeta(ns).log();
      const weight = new Complex(bf(mu).div(bf(n)));
      const term = logZ.mul(weight);
      sum = sum.add(term);
      if (n > 2 && term.abs().cmp(bf(10).pow(bf(-prec))) < 0) break;
    }
    return sum;
  }
  function getMobius(n) {
    if (n === 1) return 1;
    let p = 0;
    let temp = n;
    for (let i = 2; i * i <= temp; i++) {
      if (temp % i === 0) {
        temp /= i;
        if (temp % i === 0) return 0;
        p++;
      }
    }
    if (temp > 1) p++;
    return p % 2 === 0 ? 1 : -1;
  }

  // src/lambertw.js
  function lambertw(z, k = 0) {
    let _z = z instanceof Complex ? z : new Complex(z);
    if (_z.re.isZero() && _z.im.isZero()) {
      if (k === 0) return new Complex(0);
      return new Complex(-Infinity, 0);
    }
    const E2 = new Complex(E);
    const branch_pt = new Complex(-1).div(E2);
    if (_z.re.cmp(branch_pt.re) === 0 && _z.im.isZero()) {
      if (k === 0 || k === -1) {
        return new Complex(-1);
      }
    }
    const ONE = new Complex(one);
    const TWO = new Complex(two);
    const THREE = new Complex(three);
    let w;
    let re = _z.re.toNumber();
    let im = _z.im.toNumber();
    let abs_z = Math.sqrt(re * re + im * im);
    let dist_bp = Math.sqrt((re + 0.36787944117144233) ** 2 + im * im);
    if (k === 0) {
      if (dist_bp < 0.3) {
        let p = TWO.mul(E2.mul(_z).add(ONE)).sqrt();
        w = new Complex(-1).add(p).sub(p.mul(p).div(THREE));
      } else if (re > -0.3 && abs_z < 2.5) {
        w = _z.div(_z.add(ONE));
      } else {
        let L1 = _z.log();
        let L2 = L1.log();
        w = L1.sub(L2).add(L2.div(L1));
      }
    } else if (k === -1) {
      if (dist_bp < 0.3) {
        let p = TWO.mul(E2.mul(_z).add(ONE)).sqrt().mul(new Complex(-1));
        w = new Complex(-1).add(p).sub(p.mul(p).div(THREE));
      } else if (re < 0 && abs_z < 0.5) {
        let L1 = _z.neg().log();
        let L2 = L1.neg().log();
        w = L1.sub(L2).add(L2.div(L1));
      } else {
        let L1 = _z.log().add(new Complex(0, PI.mul(bf(-2))));
        let L2 = L1.log();
        w = L1.sub(L2).add(L2.div(L1));
      }
    } else {
      let L1 = _z.log().add(new Complex(0, PI.mul(bf(2 * k))));
      let L2 = L1.log();
      w = L1.sub(L2).add(L2.div(L1));
    }
    const max_iter = 1e3;
    let prev_w = w;
    let prev2_w = w;
    for (let i = 0; i < max_iter; i++) {
      let w_re = w.re.toNumber();
      let w_next;
      let wPlus1 = w.add(ONE);
      if (wPlus1.re.isZero() && wPlus1.im.isZero()) {
        break;
      }
      let wPlus2 = w.add(TWO);
      if (w_re > 0) {
        let emw = w.neg().exp();
        let delta = w.sub(_z.mul(emw));
        let term2 = wPlus2.mul(delta).div(wPlus1.mul(TWO));
        let denom = wPlus1.sub(term2);
        if (denom.re.isZero() && denom.im.isZero()) break;
        w_next = w.sub(delta.div(denom));
      } else {
        let ew = w.exp();
        let p = w.mul(ew).sub(_z);
        let term2 = wPlus2.mul(p).div(wPlus1.mul(TWO));
        let denom = ew.mul(wPlus1).sub(term2);
        if (denom.re.isZero() && denom.im.isZero()) break;
        w_next = w.sub(p.div(denom));
      }
      if (w_next.re.cmp(w.re) === 0 && w_next.im.cmp(w.im) === 0) {
        w = w_next;
        break;
      }
      if (i > 0 && w_next.re.cmp(prev2_w.re) === 0 && w_next.im.cmp(prev2_w.im) === 0) {
        w = w_next;
        break;
      }
      prev2_w = prev_w;
      prev_w = w;
      w = w_next;
    }
    return w;
  }

  // src/bessel.js
  function isInteger(c) {
    if (!c.im.isZero()) return false;
    let num = c.re.toNumber();
    if (Math.abs(num) < Number.MAX_SAFE_INTEGER) {
      return num % 1 === 0;
    }
    return c.re.cmp(c.re.floor()) === 0;
  }
  function hyp0f1(a, z, max_iter = 1e4) {
    let _a = a instanceof Complex ? a : new Complex(a);
    let _z = z instanceof Complex ? z : new Complex(z);
    const ONE = new Complex(one);
    let term = ONE;
    let sum = ONE;
    let prev_sum = ONE;
    let prev2_sum = ONE;
    for (let k = 1; k < max_iter; k++) {
      let k_cplx = new Complex(k);
      let a_plus_k_minus_1 = _a.add(new Complex(k - 1));
      term = term.mul(_z).div(k_cplx.mul(a_plus_k_minus_1));
      let next_sum = sum.add(term);
      if (next_sum.re.cmp(sum.re) === 0 && next_sum.im.cmp(sum.im) === 0) {
        sum = next_sum;
        break;
      }
      if (k > 1 && next_sum.re.cmp(prev2_sum.re) === 0 && next_sum.im.cmp(prev2_sum.im) === 0) {
        sum = next_sum;
        break;
      }
      prev2_sum = prev_sum;
      prev_sum = sum;
      sum = next_sum;
    }
    return sum;
  }
  function besselj(nu, z) {
    let _nu = nu instanceof Complex ? nu : new Complex(nu);
    let _z = z instanceof Complex ? z : new Complex(z);
    if (_z.re.isZero() && _z.im.isZero()) {
      if (_nu.re.isZero() && _nu.im.isZero()) return new Complex(one);
      if (_nu.re.toNumber() > 0) return new Complex(0);
      return new Complex(Infinity, 0);
    }
    if (isInteger(_nu) && _nu.re.toNumber() < 0) {
      let n = Math.abs(Math.round(_nu.re.toNumber()));
      let sign2 = n % 2 === 0 ? new Complex(one) : new Complex(-1);
      return sign2.mul(besselj(_nu.neg(), _z));
    }
    const TWO = new Complex(two);
    const ONE = new Complex(one);
    let z_over_2 = _z.div(TWO);
    let prefactor = z_over_2.pow(_nu);
    let gamma_nu_plus_1 = gamma(_nu.add(ONE));
    let z_sq_over_4_neg = _z.mul(_z).div(new Complex(4)).neg();
    let h = hyp0f1(_nu.add(ONE), z_sq_over_4_neg);
    return prefactor.mul(h).div(gamma_nu_plus_1);
  }
  function besseli(nu, z) {
    let _nu = nu instanceof Complex ? nu : new Complex(nu);
    let _z = z instanceof Complex ? z : new Complex(z);
    if (_z.re.isZero() && _z.im.isZero()) {
      if (_nu.re.isZero() && _nu.im.isZero()) return new Complex(one);
      if (_nu.re.toNumber() > 0) return new Complex(0);
      return new Complex(Infinity, 0);
    }
    if (isInteger(_nu) && _nu.re.toNumber() < 0) {
      return besseli(_nu.neg(), _z);
    }
    const TWO = new Complex(two);
    const ONE = new Complex(one);
    let z_over_2 = _z.div(TWO);
    let prefactor = z_over_2.pow(_nu);
    let gamma_nu_plus_1 = gamma(_nu.add(ONE));
    let z_sq_over_4 = _z.mul(_z).div(new Complex(4));
    let h = hyp0f1(_nu.add(ONE), z_sq_over_4);
    return prefactor.mul(h).div(gamma_nu_plus_1);
  }
  function bessely(nu, z) {
    let _nu = nu instanceof Complex ? nu : new Complex(nu);
    let _z = z instanceof Complex ? z : new Complex(z);
    if (_z.re.isZero() && _z.im.isZero()) {
      return new Complex(-Infinity, 0);
    }
    if (isInteger(_nu)) {
      let eps = new Complex(1e-12);
      _nu = _nu.add(eps);
    }
    let pi = new Complex(PI || Math.PI);
    let nu_pi = _nu.mul(pi);
    let j_nu = besselj(_nu, _z);
    let j_minus_nu = besselj(_nu.neg(), _z);
    let cos_nu_pi = nu_pi.cos();
    let sin_nu_pi = nu_pi.sin();
    let num = j_nu.mul(cos_nu_pi).sub(j_minus_nu);
    return num.div(sin_nu_pi);
  }
  function besselk(nu, z) {
    let _nu = nu instanceof Complex ? nu : new Complex(nu);
    let _z = z instanceof Complex ? z : new Complex(z);
    if (_z.re.isZero() && _z.im.isZero()) {
      return new Complex(Infinity, 0);
    }
    if (isInteger(_nu)) {
      let eps = new Complex(1e-12);
      _nu = _nu.add(eps);
    }
    let pi = new Complex(PI || Math.PI);
    let nu_pi = _nu.mul(pi);
    let i_nu = besseli(_nu, _z);
    let i_minus_nu = besseli(_nu.neg(), _z);
    let sin_nu_pi = nu_pi.sin();
    let num = i_minus_nu.sub(i_nu);
    let half_pi = pi.div(new Complex(two));
    return half_pi.mul(num).div(sin_nu_pi);
  }
  function hankel1(nu, z) {
    let j = besselj(nu, z);
    let y = bessely(nu, z);
    let i = new Complex(0, 1);
    return j.add(i.mul(y));
  }
  function hankel2(nu, z) {
    let j = besselj(nu, z);
    let y = bessely(nu, z);
    let i = new Complex(0, 1);
    return j.sub(i.mul(y));
  }

  // src/frac.js
  var gcd = (a, b) => {
    return b === 0n ? a : gcd(b, a % b);
  };
  var bigIntSqrt = (value) => {
    if (value < 0n) return void 0;
    if (value < 2n) return value;
    let x0 = value;
    let x1 = x0 + value / x0 >> 1n;
    while (x1 < x0) {
      x0 = x1;
      x1 = x0 + value / x0 >> 1n;
    }
    return x0 * x0 === value ? x0 : void 0;
  };
  var _buf = new ArrayBuffer(8);
  var _f64 = new Float64Array(_buf);
  var _u64 = new BigUint64Array(_buf);
  var fromDouble = (val) => {
    _f64[0] = val;
    const bits = _u64[0];
    const sign2 = bits >> 63n ? -1n : 1n;
    const exponent = bits >> 52n & 0x7FFn;
    let mantissa = bits & 0xFFFFFFFFFFFFFn;
    if (exponent === 0n && mantissa === 0n) {
      return { n: 0n, d: 1n };
    }
    let shift;
    if (exponent === 0n) {
      shift = 1n - 1023n - 52n;
    } else {
      mantissa |= 0x10000000000000n;
      shift = exponent - 1023n - 52n;
    }
    let num, den;
    if (shift >= 0n) {
      num = mantissa << shift;
      den = 1n;
    } else {
      num = mantissa;
      den = 1n << -shift;
    }
    let resultNum = num;
    let resultDen = den;
    if (den !== 1n) {
      const absVal = Math.abs(val);
      const MAX_SAFE = BigInt(Number.MAX_SAFE_INTEGER);
      let n0 = 0n, d0 = 1n;
      let n1 = 1n, d1 = 0n;
      let remN = num;
      let remD = den;
      while (remD !== 0n) {
        const a = remN / remD;
        const nextN = remN % remD;
        const n2 = a * n1 + n0;
        const d2 = a * d1 + d0;
        if (n2 > MAX_SAFE || d2 > MAX_SAFE) {
          break;
        }
        if (Number(n2) / Number(d2) === absVal) {
          resultNum = n2;
          resultDen = d2;
          break;
        }
        remN = remD;
        remD = nextN;
        n0 = n1;
        d0 = d1;
        n1 = n2;
        d1 = d2;
      }
    }
    if (sign2 < 0n) resultNum = -resultNum;
    return { n: resultNum, d: resultDen };
  };
  var BigFraction = class _BigFraction {
    /**
     * Optimized Constructor.
     * 
     * Supports:
     * 1. Number (Float): Uses bitwise extraction (Fastest, Exact).
     * 2. Number (Integer): Direct BigInt conversion.
     * 3. String: "1.5", "1/2", "-5".
     * 4. BigInt / BigFraction.
     * @param {BigFraction | bigint | number | string | BigFloat} [n] - The numerator or the whole value.
     * @param {bigint | number | string} [d] - The denominator.
     */
    constructor(n, d) {
      if (n instanceof _BigFraction) {
        this.n = n.n;
        this.d = n.d;
        return;
      }
      let num = 0n;
      let den = 1n;
      const typeN = typeof n;
      if (typeN === "bigint") {
        num = n;
        den = d !== void 0 ? BigInt(d) : 1n;
      } else if (typeN === "number") {
        if (Number.isInteger(n)) {
          num = BigInt(n);
          den = d !== void 0 ? BigInt(d) : 1n;
        } else if (!Number.isFinite(n)) {
          this.n = 0n;
          this.d = 0n;
          return;
        } else {
          const res = fromDouble(n);
          num = res.n;
          den = res.d;
        }
      } else if (typeN === "string") {
        if (n.includes("/")) {
          const parts = n.split("/");
          num = BigInt(parts[0]);
          den = BigInt(parts[1]);
        } else if (n.includes(".")) {
          const [intPart, fracPart] = n.split(".");
          num = BigInt(intPart + fracPart);
          den = 10n ** BigInt(fracPart.length);
        } else {
          num = BigInt(n);
          den = 1n;
        }
        if (d !== void 0) {
          const den2 = BigInt(d);
          den = den * den2;
        }
      } else if (n instanceof BigFloat) {
        const str = n.toFixed(10);
        let [intPart, fracPart] = str.split(".");
        if (fracPart === void 0) {
          num = BigInt(intPart);
          den = 1n;
        } else {
          fracPart = fracPart.replace(/0+$/, "");
          num = BigInt(intPart + fracPart);
          den = 10n ** BigInt(fracPart.length);
        }
      } else if (typeN === "undefined" || n === null) {
        num = 0n;
        den = 1n;
      } else {
        throw new Error("Unsupported input type for BigFraction constructor");
      }
      if (den === 0n) {
        this.n = 0n;
        this.d = 0n;
      } else {
        if (den < 0n) {
          num = -num;
          den = -den;
        }
        const common = gcd(num > 0n ? num : -num, den);
        this.n = num / common;
        this.d = den / common;
      }
    }
    // ===========================
    // Core Arithmetic Methods
    // ===========================
    /**
     * Adds another fraction.
     * @param {BigFraction|bigint|number|string} b 
     * @returns {BigFraction} New Fraction instance
     */
    add(b) {
      const other = new _BigFraction(b);
      return new _BigFraction(
        this.n * other.d + other.n * this.d,
        this.d * other.d
      );
    }
    /**
     * Subtracts another fraction.
     * @param {BigFraction|bigint|number|string} b 
     * @returns {BigFraction}
     */
    sub(b) {
      const other = new _BigFraction(b);
      return new _BigFraction(
        this.n * other.d - other.n * this.d,
        this.d * other.d
      );
    }
    /**
     * Multiplies by another fraction.
     * @param {BigFraction|bigint|number|string} b 
     * @returns {BigFraction}
     */
    mul(b) {
      const other = new _BigFraction(b);
      return new _BigFraction(
        this.n * other.n,
        this.d * other.d
      );
    }
    /**
     * Divides by another fraction.
     * @param {BigFraction|bigint|number|string} b 
     * @returns {BigFraction}
     */
    div(b) {
      const other = new _BigFraction(b);
      if (other.isZero()) throw new Error("Division by zero");
      return new _BigFraction(
        this.n * other.d,
        this.d * other.n
      );
    }
    /**
     * Returns the integer square root of the fraction (floor(sqrt(value))).
     * Since BigInt arithmetic is integer based, exact rational roots are rare.
     * This returns a Fraction representing the integer root.
     * @returns {BigFraction|undefined}
     */
    sqrt() {
      if (this.n < 0n) throw new Error("Square root of negative number");
      const val = this.n / this.d;
      let s = bigIntSqrt(val);
      return s === void 0 ? void 0 : new _BigFraction(s, 1n);
    }
    /**
     * Raises fraction to an integer power.
     * @param {number|bigint|BigFraction} exponent 
     * @returns {BigFraction|undefined}
     */
    pow(exponent) {
      if (typeof exponent === "bigint") {
      } else if (exponent instanceof _BigFraction) {
        if (exponent.n === 0n || exponent.d === 1n) {
          exponent = exponent.n;
        } else {
          return void 0;
        }
      } else if (typeof exponent === "number") {
        if (Number.isInteger(exponent)) {
          exponent = BigInt(exponent);
        } else {
          return void 0;
        }
      }
      let exp2 = exponent;
      if (exp2 === 0n) return new _BigFraction(1n);
      let num = this.n;
      let den = this.d;
      if (exp2 < 0n) {
        let temp = num;
        num = den;
        den = temp;
        exp2 = -exp2;
      }
      return new _BigFraction(num ** exp2, den ** exp2);
    }
    /**
     * Returns the floor of the fraction (largest integer <= value).
     * @returns {BigFraction}
     */
    floor() {
      if (this.d === 0n) return this;
      let res = this.n / this.d;
      if (this.n < 0n && this.n % this.d !== 0n) {
        res -= 1n;
      }
      return new _BigFraction(res, 1n);
    }
    /**
     * Negates the value.
     * @returns {BigFraction}
     */
    neg() {
      return new _BigFraction(-this.n, this.d);
    }
    /**
     * Returns the absolute value.
     * @returns {BigFraction}
     */
    abs() {
      return new _BigFraction(this.n < 0n ? -this.n : this.n, this.d);
    }
    /**
     * Returns e^this. Placeholder, not implemented.
     * @returns {BigFraction|undefined}
     */
    exp() {
      if (this.n === 0n) return new _BigFraction(1n);
      return void 0;
    }
    /**
     * Returns log(this). Placeholder, not implemented.
     * @returns {BigFraction|undefined}
     */
    log() {
      if (this.n === this.d && this.d !== 0n) return new _BigFraction(0n);
      return void 0;
    }
    /**
     * Returns sin(this). Placeholder, not implemented.
     * @returns {BigFraction|undefined}
     */
    sin() {
      if (this.n === 0n) return new _BigFraction(0n);
      return void 0;
    }
    /**
     * Returns cos(this). Placeholder, not implemented.
     * @returns {BigFraction|undefined}
     */
    cos() {
      if (this.n === 0n) return new _BigFraction(1n);
      return void 0;
    }
    /**
     * Returns tan(this). Placeholder, not implemented.
     * @returns {BigFraction|undefined}
     */
    tan() {
      if (this.n === 0n) return new _BigFraction(0n);
      return void 0;
    }
    /**
     * Returns asin(this). Placeholder, not implemented.
     * @returns {BigFraction|undefined}
     */
    asin() {
      if (this.n === 0n) return new _BigFraction(0n);
      return void 0;
    }
    /**
     * Returns acos(this). Placeholder, not implemented.
     * @returns {BigFraction|undefined}
     */
    acos() {
      if (this.n === this.d && this.d !== 0n) return new _BigFraction(0n);
      return void 0;
    }
    /**
     * Returns atan(this). Placeholder, not implemented.
     * @returns {BigFraction|undefined}
     */
    atan() {
      if (this.n === 0n) return new _BigFraction(0n);
      return void 0;
    }
    /**
     * Returns sinh(this). Placeholder, not implemented.
     * @returns {BigFraction|undefined}
     */
    sinh() {
      if (this.n === 0n) return new _BigFraction(0n);
      return void 0;
    }
    /**
     * Returns cosh(this). Placeholder, not implemented.
     * @returns {BigFraction|undefined}
     */
    cosh() {
      if (this.n === 0n) return new _BigFraction(1n);
      return void 0;
    }
    /**
     * Returns tanh(this). Placeholder, not implemented.
     * @returns {BigFraction|undefined}
     */
    tanh() {
      if (this.n === 0n) return new _BigFraction(0n);
      return void 0;
    }
    /**
     * Returns asinh(this). Placeholder, not implemented.
     * @returns {BigFraction|undefined}
     */
    asinh() {
      if (this.n === 0n) return new _BigFraction(0n);
      return void 0;
    }
    /**
     * Returns acosh(this). Placeholder, not implemented.
     * @returns {BigFraction|undefined}
     */
    acosh() {
      if (this.n === this.d && this.d !== 0n) return new _BigFraction(0n);
      return void 0;
    }
    /**
     * Returns atanh(this). Placeholder, not implemented.
     * @returns {BigFraction|undefined}
     */
    atanh() {
      if (this.n === 0n) return new _BigFraction(0n);
      return void 0;
    }
    // ===========================
    // Status & Conversion
    // ===========================
    /**
     * Checks if the fraction is technically invalid (denominator was 0).
     * @returns {boolean}
     */
    isNaN() {
      return this.d === 0n;
    }
    /**
     * Checks if the value is zero.
     * @returns {boolean}
     */
    isZero() {
      return this.n === 0n && this.d !== 0n;
    }
    /**
     * Checks if the value is zero.
     * @returns {boolean}
     */
    isAlmostZero() {
      return this.n === 0n && this.d !== 0n;
    }
    /**
     * @returns {BigFloat}
     */
    toBigFloat() {
      return bf(this.n).div(bf(this.d));
    }
    /**
     * Converts to a standard JavaScript number (may lose precision).
     * @returns {number}
     */
    toNumber() {
      if (this.d === 0n) return NaN;
      return Number(this.n) / Number(this.d);
    }
    /**
     * Parses a string to create a fraction.
     * Supports integers "123", fractions "1/2", and decimals "1.5".
     * @param {string} str 
     * @returns {BigFraction}
     */
    static fromString(str) {
      if (typeof str != "string") {
        throw new Error("not a string");
      }
      return new _BigFraction(str);
    }
    /**
       * Converts to string.
    * @param {number} [radix=10] 
    * @param {number} [prec=-1] precision digits in radix
    * @param {boolean} [pretty=false] pretty print
       * @returns {string}
       */
    toString(radix = 10, prec = -1, pretty = false) {
      if (this.d === 1n) {
        return this.n.toString(radix, prec, pretty);
      }
      return `${this.n.toString(radix, prec, pretty)}/${this.d.toString(radix, prec, pretty)}`;
    }
    // ===========================
    // Comparisons
    // ===========================
    /**
     * Compares with another value.
     * @param {BigFraction|bigint|number|string} b 
     * @returns {number} -1 if less, 0 if equal, 1 if greater.
     */
    cmp(b) {
      const other = b instanceof _BigFraction ? b : new _BigFraction(b);
      const left = this.n * other.d;
      const right = other.n * this.d;
      if (left < right) return -1;
      if (left > right) return 1;
      return 0;
    }
    /**
     * @param {BigFraction|bigint|number|string} b 
     * @returns {boolean}
     */
    equals(b) {
      return this.cmp(b) == 0;
    }
    /**
     * @param {BigFraction|bigint|number|string} b 
     * @returns {boolean}
     */
    operatorLess(b) {
      return this.cmp(b) === -1;
    }
    /**
     * @param {BigFraction|bigint|number|string} b 
     * @returns {boolean}
     */
    operatorGreater(b) {
      return this.cmp(b) === 1;
    }
    /**
     * @param {BigFraction|bigint|number|string} b 
     * @returns {boolean}
     */
    operatorLessEqual(b) {
      return this.cmp(b) <= 0;
    }
    /**
     * @param {BigFraction|bigint|number|string} b 
     * @returns {boolean}
     */
    operatorGreaterEqual(b) {
      return this.cmp(b) >= 0;
    }
    /**
     * @param {BigFraction|bigint|number|string} b 
     * @returns {boolean}
     */
    operatorEqual(b) {
      return this.cmp(b) === 0;
    }
    /**
     * @param {BigFraction|bigint|number|string} b 
     * @returns {boolean}
     */
    operatorNotEqual(b) {
      return this.cmp(b) !== 0;
    }
    /**
     * @param {BigFraction|bigint|number|string} b 
     * @returns {BigFraction}
     */
    operatorAdd(b) {
      return this.add(b);
    }
    /**
     * @param {BigFraction|bigint|number|string} b 
     * @returns {BigFraction}
     */
    operatorSub(b) {
      return this.sub(b);
    }
    /**
     * @param {BigFraction|bigint|number|string} b 
     * @returns {BigFraction}
     */
    operatorMul(b) {
      return this.mul(b);
    }
    /**
     * @param {BigFraction|bigint|number|string} b 
     * @returns {BigFraction}
     */
    operatorDiv(b) {
      return this.div(b);
    }
    /**
     * @param {number|bigint|BigFraction} b 
     * @returns {BigFraction|undefined}
     */
    operatorPow(b) {
      return this.pow(b);
    }
    /**
     * @returns {BigFraction}
     */
    operatorNeg() {
      return this.neg();
    }
  };
  function frac(n, d = 1n) {
    return new BigFraction(n, d);
  }

  // src/poly.js
  var Poly = class _Poly {
    /**
     * @param {number[]} degs - Array of degrees (integers).
     * @param {any[]} coefs - Array of coefficients corresponding to degrees.
     * @param {number} [order=Infinity] - Truncation order O(X^n).
     * @param {Function} [coefType=bf] - the coef type
     */
    constructor(degs, coefs, order = Infinity, coefType = BigFloat) {
      this.degs = degs || [];
      this.coefs = coefs || [];
      this.o = order;
      this.coefType = coefType;
      this._normalize();
    }
    // --- Global Factory Helpers ---
    /**
     * Creates a polynomial representing X^n.
     * @param {number} [n=1] - The degree of X.
     * @param {Function} [coefType=BigFloat] - The coefficient type.
     * @returns {Poly}
     */
    static X(n = 1, coefType = BigFloat) {
      return new _Poly([n], [new coefType(1)], Infinity, coefType);
    }
    /**
     * Creates a Big-O term O(X^n).
     * @param {number} n - The order of the truncation.
     * @param {Function} [coefType=BigFloat] - The coefficient type.
     * @returns {Poly}
     */
    static O(n, coefType = BigFloat) {
      return new _Poly([], [], n, coefType);
    }
    // --- Internal Logic ---
    /**
     * Normalizes the sparse representation:
     * 1. Sorts by degree.
     * 2. Merges duplicate degrees.
     * 3. Removes zero coefficients.
     * 4. Removes terms with degree >= order.
     * @private
     */
    _normalize() {
      if (this.degs.length === 0) return;
      let terms = this.degs.map((d, i) => ({ d, c: this.coefs[i] }));
      terms.sort((a, b) => a.d - b.d);
      const newDegs = [];
      const newCoefs = [];
      if (terms.length > 0) {
        let currentD = terms[0].d;
        let currentC = terms[0].c;
        for (let i = 1; i < terms.length; i++) {
          if (terms[i].d === currentD) {
            currentC = currentC.add(terms[i].c);
          } else {
            this._pushTerm(newDegs, newCoefs, currentD, currentC);
            currentD = terms[i].d;
            currentC = terms[i].c;
          }
        }
        this._pushTerm(newDegs, newCoefs, currentD, currentC);
      }
      this.degs = newDegs;
      this.coefs = newCoefs;
    }
    /**
     * Pushes a term to the new arrays if it's valid.
     * @private
     * @param {number[]} degs
     * @param {any[]} coefs
     * @param {number} d
     * @param {any} c
     */
    _pushTerm(degs, coefs, d, c) {
      if (d >= this.o) return;
      if (!c.isZero()) {
        degs.push(d);
        coefs.push(c);
      }
    }
    /**
     * Wraps a scalar value in a Poly object.
     * @private
     * @param {any} v
     * @returns {Poly}
     */
    _wrap(v) {
      if (v instanceof _Poly) return v;
      return new _Poly([0], [this._ensureType(v)], Infinity, this.coefType);
    }
    /**
     * Ensures a value is of the correct coefficient type.
     * @private
     * @param {any} c
     * @returns {any}
     */
    _ensureType(c) {
      return c && c.add ? c : new this.coefType(c);
    }
    // --- Inspection ---
    /**
     * Returns the "Valuation" (degree of the lowest non-zero term).
     * @returns {number} - Infinity if Exact Zero, or n if O(X^n).
     */
    valuation() {
      if (this.degs.length === 0) {
        return this.o === Infinity || this.o === null ? Infinity : this.o;
      }
      return this.degs[0];
    }
    /**
     * Returns the Degree (highest non-zero term).
     * @returns {number} - -1 if zero polynomial.
     */
    degree() {
      if (this.degs.length === 0) return -1;
      return this.degs[this.degs.length - 1];
    }
    /**
     * Returns a dense array of coefficients up to the highest degree.
     * Note: Throws if polynomial contains negative powers. Use offsetCoefs for Laurent series.
     * @returns {any[]} [c0, c1, c2, ...]
     */
    get denseCoefs() {
      if (this.valuation() < 0) {
        throw new Error("denseCoefs does not support negative degrees (Laurent Series). Use offsetCoefs instead.");
      }
      const len = this.degree() + 1;
      if (len <= 0) return [];
      const zero2 = this.coefs.length > 0 ? this.coefs[0].sub(this.coefs[0]) : new this.coefType(0);
      const arr = new Array(len).fill(zero2);
      for (let i = 0; i < this.degs.length; i++) {
        arr[this.degs[i]] = this.coefs[i];
      }
      return arr;
    }
    /**
     * Returns a dense array of coefficients along with the valuation offset.
     * Supports negative degrees.
     * @returns {{val: number, coefs: any[]}} { val: starting_degree, coefs: [c_val, c_val+1, ...] }
     */
    get offsetCoefs() {
      if (this.degs.length === 0) return { val: this.o === Infinity ? 0 : this.o, coefs: [] };
      const minDeg = this.degs[0];
      const maxDeg = this.degs[this.degs.length - 1];
      const len = maxDeg - minDeg + 1;
      const zero2 = this.coefs.length > 0 ? this.coefs[0].sub(this.coefs[0]) : new this.coefType(0);
      const arr = new Array(len).fill(zero2);
      for (let i = 0; i < this.degs.length; i++) {
        arr[this.degs[i] - minDeg] = this.coefs[i];
      }
      return { val: minDeg, coefs: arr };
    }
    /**
     * Evaluates the polynomial at x.
     * P(x) = sum( c_i * x^i )
     * @param {number|BigFloat|Complex} x 
     * @returns {any}
     */
    eval(x) {
      const X2 = this._ensureType(x);
      if (this.degs.length === 0) return new this.coefType(0);
      const isZero = X2.isZero && X2.isZero() || X2 === 0;
      if (isZero && this.valuation() < 0) {
        return Infinity;
      }
      let result = new this.coefType(0);
      for (let i = 0; i < this.degs.length; i++) {
        const d = this.degs[i];
        const c = this.coefs[i];
        const term = c.mul(X2.pow(d));
        result = result.add(term);
      }
      return result;
    }
    // --- Arithmetic ---
    /**
     * Adds two polynomials.
     * @param {Poly|number|any} other
     * @returns {Poly}
     */
    add(other) {
      const B = this._wrap(other);
      const newOrder = Math.min(this.o, B.o);
      const newDegs = this.degs.concat(B.degs);
      const newCoefs = this.coefs.concat(B.coefs);
      return new _Poly(newDegs, newCoefs, newOrder, this.coefType);
    }
    /**
     * Subtracts two polynomials.
     * @param {Poly|number|any} other
     * @returns {Poly}
     */
    sub(other) {
      const B = this._wrap(other);
      const newOrder = Math.min(this.o, B.o);
      const negCoefs = B.coefs.map((c) => c.neg());
      const newDegs = this.degs.concat(B.degs);
      const newCoefs = this.coefs.concat(negCoefs);
      return new _Poly(newDegs, newCoefs, newOrder, this.coefType);
    }
    /**
     * Negates the polynomial.
     * @returns {Poly}
     */
    neg() {
      return this.mul(-1);
    }
    /**
     * Multiplies two polynomials.
     * @param {Poly|number|any} other
     * @returns {Poly}
     */
    mul(other) {
      const B = this._wrap(other);
      const vA = this.valuation();
      const vB = B.valuation();
      if (vA === Infinity || vB === Infinity) return new _Poly([], [], Infinity, this.coefType);
      const term1 = B.o === Infinity ? Infinity : vA + B.o;
      const term2 = this.o === Infinity ? Infinity : vB + this.o;
      const term3 = this.o === Infinity || B.o === Infinity ? Infinity : this.o + B.o;
      const newOrder = Math.min(term1, term2, term3);
      if (vA === Infinity || vB === Infinity) return new _Poly([], [], newOrder, this.coefType);
      const productMap = /* @__PURE__ */ new Map();
      for (let i = 0; i < this.degs.length; i++) {
        for (let j = 0; j < B.degs.length; j++) {
          const d = this.degs[i] + B.degs[j];
          if (newOrder !== Infinity && d >= newOrder) continue;
          const c = this.coefs[i].mul(B.coefs[j]);
          if (productMap.has(d)) {
            productMap.set(d, productMap.get(d).add(c));
          } else {
            productMap.set(d, c);
          }
        }
      }
      const resDegs = [];
      const resCoefs = [];
      for (const [d, c] of productMap) {
        resDegs.push(d);
        resCoefs.push(c);
      }
      return new _Poly(resDegs, resCoefs, newOrder, this.coefType);
    }
    /**
     * Power: P(x)^n
     * @param {number} n - The exponent.
     * @param {number} [d=1] - The denominator of the exponent.
     * @returns {Poly}
     */
    pow(n, d = 1) {
      if (!Number.isInteger(n) || n < 0 || d !== 1) {
        return this.powSeries(n, d);
      }
      if (n === 0) return new _Poly([0], [new this.coefType(1)], this.o, this.coefType);
      let base = this;
      let result = new _Poly([0], [new this.coefType(1)], this.o, this.coefType);
      let p = n;
      while (p > 0) {
        if (p % 2 === 1) result = result.mul(base);
        base = base.mul(base);
        p = Math.floor(p / 2);
      }
      return result;
    }
    /**
     * Division: A / B
     * Supports Exact Polynomials (Euclidean/Laurent) and Truncated Series.
     * 
     * Logic:
     * Uses "Synthetic Division" from lowest degree upwards (Series Division).
     * - If both A and B are Exact (Order=Infinity):
     *    - Tries to divide exactly. 
     *    - If remainder becomes 0, returns Exact result (Order=Infinity).
     *    - If infinite expansion (e.g. 1/(1-x)), stops at defaultLimit and returns Series (Order=Limit).
     * - If one/both are Series (Finite Order):
     *    - Calculates terms up to the theoretical precision limit.
     * 
     * @param {Poly} other
     * @param {number} [defaultLimit=100] - Max terms to calculate for infinite exact expansions.
     * @returns {Poly}
     */
    div(other, defaultLimit = 100) {
      const B = this._wrap(other);
      const vA = this.valuation();
      const vB = B.valuation();
      if (vB === Infinity) throw new Error("Division by zero");
      const a = this.o;
      const b = B.o;
      const term1 = a === Infinity ? Infinity : a - vB;
      const term2 = b === Infinity ? Infinity : b + vA - 2 * vB;
      const resOrder = Math.min(term1, term2);
      const isExactMode = resOrder === Infinity;
      const startDeg = vA === Infinity ? 0 : vA - vB;
      const limitDeg = isExactMode ? startDeg + defaultLimit : resOrder;
      const qDegs = [];
      const qCoefs = [];
      const currentRem = /* @__PURE__ */ new Map();
      for (let i = 0; i < this.degs.length; i++) {
        currentRem.set(this.degs[i], this.coefs[i]);
      }
      const bLowDeg = B.degs[0];
      const bLowCoef = B.coefs[0];
      let droppedSignificantTerm = false;
      for (let k = startDeg; k < limitDeg; k++) {
        if (isExactMode && currentRem.size === 0 && !droppedSignificantTerm) {
          return new _Poly(qDegs, qCoefs, Infinity, this.coefType);
        }
        const targetDeg = k + bLowDeg;
        let val = currentRem.get(targetDeg);
        if (!val || val.isZero()) {
          currentRem.delete(targetDeg);
          continue;
        }
        const qVal = val.div(bLowCoef);
        qDegs.push(k);
        qCoefs.push(qVal);
        currentRem.delete(targetDeg);
        const cutoffDeg = limitDeg + bLowDeg;
        for (let i = 1; i < B.degs.length; i++) {
          const bD = B.degs[i];
          const affectDeg = k + bD;
          if (affectDeg >= cutoffDeg) {
            if (!droppedSignificantTerm) {
              const termVal = qVal.mul(B.coefs[i]);
              if (!termVal.isZero()) {
                droppedSignificantTerm = true;
              }
            }
            continue;
          }
          const bC = B.coefs[i];
          const subVal = qVal.mul(bC);
          const oldRem = currentRem.get(affectDeg) || new this.coefType(0);
          const newRem = oldRem.sub(subVal);
          if (newRem.isZero()) {
            currentRem.delete(affectDeg);
          } else {
            currentRem.set(affectDeg, newRem);
          }
        }
      }
      if (isExactMode && currentRem.size === 0 && !droppedSignificantTerm) {
        return new _Poly(qDegs, qCoefs, Infinity, this.coefType);
      }
      return new _Poly(qDegs, qCoefs, limitDeg, this.coefType);
    }
    /**
      * Checks equality with another polynomial or scalar.
      * @param {Poly|number|BigFloat|Complex} other - The object to compare with.
      * @param {Function} [cmp] - Optional comparator (a, b) => boolean. 
      *                           Defaults to checking a.equals(b) or strict equality.
      * @returns {boolean}
      */
    equals(other, cmp) {
      const B = other instanceof _Poly ? other : this._wrap(other);
      if (!cmp) {
        if (this.o !== B.o) return false;
        if (this.degs.length !== B.degs.length) return false;
        for (let i2 = 0; i2 < this.degs.length; i2++) {
          if (this.degs[i2] !== B.degs[i2]) return false;
          const cA = scalar(this.coefs[i2]);
          const cB = scalar(B.coefs[i2]);
          if (cA && cA.equals) {
            if (!cA.equals(cB)) return false;
          } else {
            if (cA !== cB) return false;
          }
        }
        return true;
      }
      const limit2 = Math.min(this.o, B.o);
      let i = 0, j = 0;
      while (i < this.degs.length || j < B.degs.length) {
        const degA = i < this.degs.length ? this.degs[i] : Infinity;
        const degB = j < B.degs.length ? B.degs[j] : Infinity;
        const minDeg = Math.min(degA, degB);
        if (minDeg >= limit2) break;
        if (degA === degB) {
          if (!cmp(this.coefs[i], B.coefs[j])) return false;
          i++;
          j++;
        } else if (degA < degB) {
          if (!cmp(this.coefs[i], void 0)) return false;
          i++;
        } else {
          if (!cmp(void 0, B.coefs[j])) return false;
          j++;
        }
      }
      return true;
    }
    /** @type {function(any): Poly} */
    operatorAdd(b) {
      return this.add(b);
    }
    /** @type {function(any): Poly} */
    operatorSub(b) {
      return this.sub(b);
    }
    /** @type {function(any): Poly} */
    operatorMul(b) {
      return this.mul(b);
    }
    /** @type {function(any): Poly} */
    operatorDiv(b) {
      return this.div(b);
    }
    /** @type {function(number, number=): Poly} */
    operatorPow(b) {
      return this.pow(b);
    }
    /** @type {function(): Poly} */
    operatorNeg() {
      return this.mul(-1);
    }
    /**
     * Derivative
     * d/dx ( c * x^k ) = (c*k) * x^(k-1)
     * @returns {Poly}
     */
    deriv() {
      const newDegs = [];
      const newCoefs = [];
      for (let i = 0; i < this.degs.length; i++) {
        const d = this.degs[i];
        if (d === 0) continue;
        const k = new this.coefType(d);
        newDegs.push(d - 1);
        newCoefs.push(this.coefs[i].mul(k));
      }
      const newOrder = this.o === Infinity ? Infinity : Math.max(0, this.o - 1);
      return new _Poly(newDegs, newCoefs, newOrder, this.coefType);
    }
    /**
     * Formal Integration
     * int ( c * x^k ) = (c / (k+1)) * x^(k+1)
     * Constant term set to 0.
     * @returns {Poly}
     */
    integ() {
      const newDegs = [];
      const newCoefs = [];
      for (let i = 0; i < this.degs.length; i++) {
        const d = this.degs[i];
        if (d === -1) {
          throw new Error("Integration of 1/x term (logarithm) is not supported in Laurent series.");
        }
        if (this.o !== Infinity && d >= this.o) continue;
        const kPlus1 = new this.coefType(d + 1);
        newDegs.push(d + 1);
        newCoefs.push(this.coefs[i].div(kPlus1));
      }
      const newOrder = this.o === Infinity ? Infinity : this.o + 1;
      return new _Poly(newDegs, newCoefs, newOrder, this.coefType);
    }
    /**
     * Power for Series: P(x)^(n/d)
     * Supports negative and fractional powers.
     * 
     * Requirements:
     * 1. Must be a Series (Order != Infinity).
     * 2. Resulting valuation must be integer.
     * 
     * @param {number} n - Numerator
     * @param {number} [d=1] - Denominator
     * @returns {Poly}
     */
    powSeries(n, d = 1) {
      if (this.o === Infinity) {
        throw new Error("powSeries requires a truncated series (O(n)). Use Poly.O(k) to specify precision.");
      }
      const alpha = new this.coefType(n).div(new this.coefType(d));
      const v = this.valuation();
      if (v === Infinity || v >= this.o) {
        if (n > 0) return _Poly.O(this.o, this.coefType);
        throw new Error("Division by zero (0^neg)");
      }
      if (v * n % d !== 0) {
        throw new Error(`Resulting degree ${v * n}/${d} is not an integer.`);
      }
      const newV = v * n / d;
      const c = this.coefs[0];
      let cPow;
      if (d === 1) cPow = c.pow(new this.coefType(n));
      else {
        cPow = c.pow(alpha);
      }
      const relPrec = this.o - v;
      const deltaMap = /* @__PURE__ */ new Map();
      for (let i = 1; i < this.degs.length; i++) {
        const deg = this.degs[i] - v;
        if (deg >= relPrec) break;
        deltaMap.set(deg, this.coefs[i].div(c));
      }
      const limit2 = relPrec;
      const b = [new this.coefType(1)];
      const a = new Array(limit2 + 1).fill(null);
      for (const [deg, val] of deltaMap) {
        if (deg <= limit2) a[deg] = val;
      }
      for (let k = 1; k < limit2; k++) {
        let sum = new this.coefType(0);
        for (let j = 1; j <= k; j++) {
          if (a[j]) {
            const termScore = alpha.add(new this.coefType(1)).mul(new this.coefType(j)).sub(new this.coefType(k));
            sum = sum.add(termScore.mul(a[j]).mul(b[k - j]));
          }
        }
        b[k] = sum.div(new this.coefType(k));
      }
      const resDegs = [];
      const resCoefs = [];
      const resultOrder = this.o - v + newV;
      for (let k = 0; k < b.length; k++) {
        if (b[k].isZero()) continue;
        const finalDeg = newV + k;
        if (finalDeg >= resultOrder) break;
        resDegs.push(finalDeg);
        resCoefs.push(cPow.mul(b[k]));
      }
      return new _Poly(resDegs, resCoefs, resultOrder, this.coefType);
    }
    // --- Transcendental Functions (Series Expansion) ---
    /**
     * Checks if the polynomial has a defined order.
     * Transcendental functions require truncation to avoid infinite loops.
     * @private
     * @param {string} op - The operation name.
     */
    _checkSeries(op) {
      if (this.o === Infinity && this.degree() > 0) {
        throw new Error(`${op} requires a truncated series (O(n)). Use Poly.O(k) to specify precision.`);
      }
      if (this.valuation() < 0) {
        throw new Error(`${op} undefined for Laurent series (negative valuation/essential singularity).`);
      }
    }
    /**
     * Splits polynomial into Constant term (c0) and Variable part (V).
     * P(x) = c0 + V(x)
     * @private
     * @returns {[any, Poly]} [c0, V]
     */
    _splitConst() {
      const c0 = this.degs.length > 0 && this.degs[0] === 0 ? this.coefs[0] : new this.coefType(0);
      const V = this.sub(c0);
      return [c0, V];
    }
    /**
     * Exponential Function: e^P(x)
     * e^(c0 + V) = e^c0 * e^V
     * e^V = 1 + V + V^2/2! + V^3/3! + ...
     * @returns {Poly}
     */
    exp() {
      this._checkSeries("exp");
      const [c0, V] = this._splitConst();
      const expC0 = c0.exp();
      if (V.degs.length === 0) {
        return new _Poly([0], [expC0], this.o, this.coefType);
      }
      let result = new _Poly([0], [new this.coefType(1)], this.o, this.coefType);
      let term = new _Poly([0], [new this.coefType(1)], this.o, this.coefType);
      for (let k = 1; k < this.o + 2; k++) {
        term = term.mul(V);
        if (term.degs.length === 0) break;
        const kBf = new this.coefType(k);
        const invK = new this.coefType(1).div(kBf);
        term = term.mul(invK);
        result = result.add(term);
      }
      return result.mul(expC0);
    }
    /**
     * Natural Logarithm: ln(P(x))
     * Uses Derivative-Integration method:
     * ln(P) = int( P' / P ) dx + ln(P(0))
     * @returns {Poly}
     */
    log() {
      this._checkSeries("log");
      const c0 = this.eval(0);
      if (c0.isZero()) {
        throw new Error("log(P) undefined: Constant term is zero.");
      }
      const deriv = this.deriv();
      const quot = deriv.div(this);
      const integ = quot.integ();
      const lnC0 = c0.log();
      return integ.add(lnC0);
    }
    /**
     * Sine: sin(P(x))
     * sin(c0 + V) = sin(c0)cos(V) + cos(c0)sin(V)
     * @returns {Poly}
     */
    sin() {
      this._checkSeries("sin");
      const [c0, V] = this._splitConst();
      if (V.degs.length === 0) {
        return new _Poly([0], [c0.sin()], this.o, this.coefType);
      }
      const s0 = c0.sin();
      const c0_val = c0.cos();
      const [sinV, cosV] = this._sinCosV(V);
      const term1 = cosV.mul(s0);
      const term2 = sinV.mul(c0_val);
      return term1.add(term2);
    }
    /**
     * Cosine: cos(P(x))
     * cos(c0 + V) = cos(c0)cos(V) - sin(c0)sin(V)
     * @returns {Poly}
     */
    cos() {
      this._checkSeries("cos");
      const [c0, V] = this._splitConst();
      if (V.degs.length === 0) {
        return new _Poly([0], [c0.cos()], this.o, this.coefType);
      }
      const c0_val = c0.cos();
      const s0 = c0.sin();
      const [sinV, cosV] = this._sinCosV(V);
      const term1 = cosV.mul(c0_val);
      const term2 = sinV.mul(s0);
      return term1.sub(term2);
    }
    /**
     * Tangent: tan(P(x))
     * tan(P) = sin(P) / cos(P)
     * @returns {Poly}
     */
    tan() {
      return this.sin().div(this.cos());
    }
    /**
     * Arcsine: asin(P(x))
     * asin(P) = int( P' / sqrt(1 - P^2) ) + asin(P(0))
     * @returns {Poly}
     */
    asin() {
      this._checkSeries("asin");
      const c0 = this.eval(0);
      const dP = this.deriv();
      const pSq = this.mul(this);
      const oneMinusPSq = poly(1, this.coefType).sub(pSq);
      const denom = oneMinusPSq.powSeries(1, 2);
      const integrand = dP.div(denom);
      return integrand.integ().add(c0.asin());
    }
    /**
     * Arccosine: acos(P(x))
     * acos(P) = PI/2 - asin(P)
     * @returns {Poly}
     */
    acos() {
      const piDiv2 = new this.coefType(Math.PI).div(new this.coefType(2));
      return poly(piDiv2, this.coefType).sub(this.asin());
    }
    /**
     * Arctangent: atan(P(x))
     * atan(P) = int( P' / (1 + P^2) ) + atan(P(0))
     * @returns {Poly}
     */
    atan() {
      this._checkSeries("atan");
      const c0 = this.eval(0);
      const dP = this.deriv();
      const pSq = this.mul(this);
      const denom = poly(1, this.coefType).add(pSq);
      const integrand = dP.div(denom);
      return integrand.integ().add(c0.atan());
    }
    // --- Helper for Sin/Cos ---
    /**
     * Computes sin(V) and cos(V) for a polynomial V with no constant term.
     * Uses Taylor series optimized for simultaneous calculation.
     * sin(V) = V - V^3/3! + V^5/5! ...
     * cos(V) = 1 - V^2/2! + V^4/4! ...
     * @private
     * @param {Poly} V 
     * @returns {[Poly, Poly]} [sinV, cosV]
     */
    _sinCosV(V) {
      let sinV = V;
      let cosV = poly(1, this.coefType);
      const V2 = V.mul(V);
      if (V2.degs.length === 0) return [sinV, cosV];
      let termP = V;
      let tCos = poly(1, this.coefType);
      let tSin = V;
      for (let k = 1; k * 2 <= this.o + 2; k++) {
        const negV2 = V2.mul(new this.coefType(-1));
        const divCos = new this.coefType(2 * k).mul(new this.coefType(2 * k - 1));
        tCos = tCos.mul(negV2).mul(new this.coefType(1).div(divCos));
        if (tCos.degs.length === 0 && tSin.degs.length === 0) break;
        cosV = cosV.add(tCos);
        const divSin = new this.coefType(2 * k).mul(new this.coefType(2 * k + 1));
        tSin = tSin.mul(negV2).mul(new this.coefType(1).div(divSin));
        sinV = sinV.add(tSin);
      }
      return [sinV, cosV];
    }
    // --- Utilities ---
    /**
       * Converts the polynomial to a string representation.
       * @param {number} [radix=10]
       * @param {number} [precision=20]
    * @param {boolean} [pretty=true] pretty print
       * @returns {string}
       */
    toString(radix = 10, precision2 = 20, pretty = true) {
      let parts = [];
      for (let i = 0; i < this.degs.length; i++) {
        const d = this.degs[i];
        const c = this.coefs[i];
        if (c.isZero()) {
          continue;
        }
        const cStr = c.toString(radix, precision2, pretty);
        let part = "";
        if (d === 0) part = cStr;
        else if (d === 1) part = `${cStr}X`;
        else part = `${cStr}X^${d}`;
        parts.push(part);
      }
      if (this.o !== Infinity) {
        parts.push(`O(X^${this.o})`);
      }
      let s = parts.join(" + ");
      if (s.length == 0) {
        s = "0";
      }
      return s;
    }
  };
  function X(n = 1, coefType = BigFloat) {
    return Poly.X(n, coefType);
  }
  function O(n = 1, coefType = BigFloat) {
    return Poly.O(n, coefType);
  }
  function polyStr(v, coefType = Scalar) {
    const s = v.replace(/I/g, "i").trim();
    if (!(coefType === Scalar)) {
      if (v.indexOf("i") != -1) {
        if (!(coefType === Complex)) {
          throw new Error(`Need complex, get ${coefType.name}`);
        }
      }
      if (v.indexOf("/") != -1) {
        if (!(coefType === BigFraction)) {
          throw new Error(`Need complex, get ${coefType.name}`);
        }
      }
    }
    const len = s.length;
    let cursor = 0;
    let order = Infinity;
    const degs = [];
    const coefs = [];
    const eof = () => cursor >= len;
    const peek = () => cursor < len ? s[cursor] : null;
    const consume = () => s[cursor++];
    const error = (msg) => {
      throw new Error(`Parse Error at index ${cursor} ('${peek()}'): ${msg} in string "${s}"`);
    };
    const isSpace = (c) => /\s/.test(c);
    const isDigitStart = (c) => /[0-9.]/.test(c);
    while (!eof()) {
      let sign2 = 1;
      let hasSign = false;
      while (!eof()) {
        const c = peek();
        if (isSpace(c)) {
          consume();
        } else if (c === "+") {
          consume();
          hasSign = true;
        } else if (c === "-") {
          sign2 *= -1;
          consume();
          hasSign = true;
        } else {
          break;
        }
      }
      if (eof()) {
        if (hasSign) error("Unexpected end of string after sign.");
        break;
      }
      const char = peek();
      let currentCoefStr = "";
      let currentDeg = 0;
      let isBigO = false;
      if (char === "O") {
        isBigO = true;
      } else if (char === "X") {
        currentCoefStr = "1";
      } else if (isDigitStart(char) || char === "(" || char === "i") {
        if (char === "(") {
          let balance = 0;
          let start = cursor;
          do {
            const c = consume();
            if (c === "(") balance++;
            else if (c === ")") balance--;
          } while (!eof() && balance > 0);
          if (balance !== 0) error("Unbalanced parentheses in coefficient.");
          currentCoefStr = s.substring(start, cursor);
        } else {
          let start = cursor;
          while (!eof()) {
            const c = peek();
            if (c === "X" || c === "*" || c === "+" || c === "-") break;
            if (/[0-9.i/]/.test(c) || isSpace(c)) {
              consume();
            } else {
              error(`Unexpected character in coefficient: ${c}`);
            }
          }
          currentCoefStr = s.substring(start, cursor).replace(/\s+/g, "");
        }
      } else {
        error("Expected digit, 'X', 'O', or '(' start.");
      }
      if (isBigO) {
        consume();
        while (!eof() && isSpace(peek())) consume();
        if (consume() !== "(") error("Expected '(' after 'O'.");
        let innerContent = "";
        let balance = 1;
        while (!eof()) {
          const c = consume();
          if (c === "(") balance++;
          else if (c === ")") {
            balance--;
            if (balance === 0) break;
          }
          innerContent += c;
        }
        if (balance !== 0) error("Unclosed 'O(...)'.");
        let orderValStr = innerContent.replace(/\s+/g, "").replace("X^", "");
        if (orderValStr === "X") orderValStr = "1";
        const parsedOrder = parseInt(orderValStr, 10);
        if (isNaN(parsedOrder)) error(`Invalid number in O term: ${innerContent}`);
        order = parsedOrder;
        continue;
      }
      while (!eof() && isSpace(peek())) consume();
      if (peek() === "*") {
        consume();
        while (!eof() && isSpace(peek())) consume();
      }
      if (peek() === "X") {
        consume();
        currentDeg = 1;
        while (!eof() && isSpace(peek())) consume();
        if (peek() === "^") {
          consume();
          while (!eof() && isSpace(peek())) consume();
          let expStr = "";
          let inParen = false;
          if (peek() === "(") {
            consume();
            inParen = true;
          }
          if (peek() === "+" || peek() === "-") expStr += consume();
          while (!eof() && /[0-9]/.test(peek())) {
            expStr += consume();
          }
          if (inParen) {
            while (!eof() && isSpace(peek())) consume();
            if (consume() !== ")") error("Missing closing paren for exponent.");
          }
          if (expStr === "" || expStr === "+" || expStr === "-") {
            error("Missing value for exponent.");
          }
          currentDeg = parseInt(expStr, 10);
        }
      } else {
        currentDeg = 0;
        if (!eof() && !/[+-]/.test(peek())) {
          error(`Unexpected character after coefficient: ${peek()}`);
        }
      }
      let val;
      if (currentCoefStr.startsWith("(") && currentCoefStr.endsWith(")")) {
        currentCoefStr = currentCoefStr.substring(1, currentCoefStr.length - 1);
      }
      if (!currentCoefStr) {
        error("Empty coefficient.");
      }
      if (typeof coefType.fromString === "function") {
        val = coefType.fromString(currentCoefStr);
      } else {
        val = new coefType(currentCoefStr);
      }
      if (sign2 === -1) {
        if (typeof val.neg === "function") {
          val = val.neg();
        } else if (typeof val.mul === "function") {
          val = val.mul(new coefType(-1));
        } else {
          val = val * -1;
        }
      }
      const isZero = typeof val.isZero === "function" ? val.isZero() : val == 0;
      if (!isZero) {
        degs.push(currentDeg);
        coefs.push(val);
      }
    }
    return new Poly(degs, coefs, order, coefType);
  }
  function poly(v, coefType = BigFloat) {
    if (typeof v === "string") {
      return polyStr(v, coefType);
    }
    if (v instanceof Poly) return v;
    if (Array.isArray(v)) {
      const degs = [];
      const coefs = [];
      for (let i = 0; i < v.length; i++) {
        const val = v[i];
        const s2 = typeof val === "object" && val !== null ? val : new coefType(val);
        if (!s2.isZero()) {
          degs.push(i);
          coefs.push(s2);
        }
      }
      return new Poly(degs, coefs, Infinity, coefType);
    }
    if (typeof v === "object" && v !== null && !(v instanceof coefType) && typeof v.isZero !== "function") {
      const degs = [];
      const coefs = [];
      const entries = v instanceof Map ? v.entries() : Object.entries(v);
      for (const [key, val] of entries) {
        const d = parseInt(key, 10);
        if (isNaN(d)) continue;
        const s2 = typeof val === "object" && val !== null ? val : new coefType(val);
        if (!s2.isZero()) {
          degs.push(d);
          coefs.push(s2);
        }
      }
      return new Poly(degs, coefs, Infinity, coefType);
    }
    const s = typeof v === "object" && v !== null ? v : new coefType(v);
    if (s.isZero()) {
      return new Poly([], [], Infinity, coefType);
    }
    return new Poly([0], [s], Infinity, coefType);
  }

  // src/scalar.js
  var Scalar = class _Scalar {
    /**
     * @param {number|bigint|string|BigFraction|BigFloat|Complex|Scalar} v 
     */
    constructor(v) {
      if (v instanceof _Scalar) {
        this.value = v.value;
        this.level = v.level;
        return;
      }
      const type = typeof v;
      if (type === "number") {
        if (Number.isInteger(v)) {
          this.value = new BigFraction(v);
          this.level = 0;
        } else {
          this.value = bf(v);
          this.level = 1;
        }
      } else if (type === "bigint") {
        this.value = new BigFraction(v);
        this.level = 0;
      } else if (type === "string") {
        const parsed = _Scalar.fromString(v);
        this.value = parsed.value;
        this.level = parsed.level;
      } else {
        this.value = v;
        this.level = _Scalar.getLevel(v);
      }
    }
    /**
     * Determine the level of a raw math object.
     * @param {any} v The value to check.
     * @returns {number} The promotion level.
     */
    static getLevel(v) {
      if (v instanceof Complex) return 2;
      if (v instanceof BigFloat) return 1;
      if (v instanceof BigFraction) return 0;
      return 1;
    }
    /**
     * Promotes a raw value to the target level.
     * @param {any} v The value to promote.
     * @param {number} targetLevel The target promotion level.
     * @returns {any} The promoted value.
     */
    static promote(v, targetLevel) {
      const currentLevel = _Scalar.getLevel(v);
      if (currentLevel >= targetLevel) return v;
      if (targetLevel === 1) {
        return v.toBigFloat ? v.toBigFloat() : bf(v.toString());
      }
      if (targetLevel === 2) {
        const realPart = _Scalar.promote(v, 1);
        return new Complex(realPart, 0);
      }
      return v;
    }
    // --- Arithmetic Operators ---
    /**
     * Internal dispatcher for binary operations.
     * @private
     * @param {Scalar|any} a - The first operand.
     * @param {Scalar|any} b - The second operand.
     * @param {string} opName - The name of the operation.
     * @returns {Scalar} The result of the operation.
     */
    static _binaryOp(a, b, opName) {
      const va = a instanceof _Scalar ? a.value : a;
      const vb = b instanceof _Scalar ? b.value : b;
      const levelA = _Scalar.getLevel(va);
      const levelB = _Scalar.getLevel(vb);
      let na = va;
      let nb = vb;
      for (let targetLevel = Math.max(levelA, levelB); targetLevel <= 3; targetLevel++) {
        na = _Scalar.promote(na, targetLevel);
        nb = _Scalar.promote(nb, targetLevel);
        let v = na[opName](nb);
        if (v !== void 0) {
          return new _Scalar(v);
        }
      }
      throw new Error(`${opName} failed for ${a.toString()}, ${b.toString()}`);
    }
    /** @param {Scalar|any} other @returns {Scalar} */
    add(other) {
      return _Scalar._binaryOp(this, other, "add");
    }
    /** @param {Scalar|any} other @returns {Scalar} */
    sub(other) {
      return _Scalar._binaryOp(this, other, "sub");
    }
    /** @param {Scalar|any} other @returns {Scalar} */
    mul(other) {
      return _Scalar._binaryOp(this, other, "mul");
    }
    /** @param {Scalar|any} other @returns {Scalar} */
    div(other) {
      return _Scalar._binaryOp(this, other, "div");
    }
    /** @param {Scalar|any} other @returns {Scalar} */
    pow(other) {
      return _Scalar._binaryOp(this, other, "pow");
    }
    // Alias for operator styles
    /** @param {Scalar|any} b @returns {Scalar} */
    operatorAdd(b) {
      return this.add(b);
    }
    /** @param {Scalar|any} b @returns {Scalar} */
    operatorSub(b) {
      return this.sub(b);
    }
    /** @param {Scalar|any} b @returns {Scalar} */
    operatorMul(b) {
      return this.mul(b);
    }
    /** @param {Scalar|any} b @returns {Scalar} */
    operatorDiv(b) {
      return this.div(b);
    }
    /** @param {Scalar|any} b @returns {Scalar} */
    operatorPow(b) {
      return this.pow(b);
    }
    /** @returns {Scalar} */
    operatorNeg() {
      return this.neg();
    }
    /** @returns {Scalar} */
    neg() {
      return new _Scalar(this.value.neg());
    }
    /** @returns {boolean} */
    isZero() {
      return this.value.isZero();
    }
    /** @returns {boolean} */
    isAlmostZero() {
      return this.value.isAlmostZero();
    }
    /** @returns {Scalar} */
    abs() {
      return new _Scalar(this.value.abs());
    }
    /**
     * Internal dispatcher for functions that might return undefined for BigFraction.
     * @private
     * @param {string} opName The name of the operation.
     * @returns {Scalar}
     */
    _unary(opName) {
      if (this.level === 0) {
        const res = this.value[opName]();
        if (res !== void 0) return new _Scalar(res);
        const promoted = this.value.toBigFloat();
        return new _Scalar(promoted[opName]());
      }
      return new _Scalar(this.value[opName]());
    }
    /** @returns {Scalar} */
    exp() {
      return this._unary("exp");
    }
    /** @returns {Scalar} */
    log() {
      return this._unary("log");
    }
    /** @returns {Scalar} */
    sin() {
      return this._unary("sin");
    }
    /** @returns {Scalar} */
    cos() {
      return this._unary("cos");
    }
    /** @returns {Scalar} */
    tan() {
      return this._unary("tan");
    }
    /** @returns {Scalar} */
    asin() {
      return this._unary("asin");
    }
    /** @returns {Scalar} */
    acos() {
      return this._unary("acos");
    }
    /** @returns {Scalar} */
    atan() {
      return this._unary("atan");
    }
    /** @returns {Scalar} */
    sinh() {
      return this._unary("sinh");
    }
    /** @returns {Scalar} */
    cosh() {
      return this._unary("cosh");
    }
    /** @returns {Scalar} */
    tanh() {
      return this._unary("tanh");
    }
    /** @returns {Scalar} */
    asinh() {
      return this._unary("asinh");
    }
    /** @returns {Scalar} */
    acosh() {
      return this._unary("acosh");
    }
    /** @returns {Scalar} */
    atanh() {
      return this._unary("atanh");
    }
    /** @returns {Scalar} */
    sqrt() {
      return this._unary("sqrt");
    }
    // --- Comparison & Utilities ---
    /**
     * Compares this scalar with another value.
     * @param {Scalar|any} other
     * @returns {number} -1 if this < other, 0 if this === other, 1 if this > other.
     */
    cmp(other) {
      const vb = other instanceof _Scalar ? other.value : other;
      const targetLevel = Math.max(this.level, _Scalar.getLevel(vb));
      if (targetLevel === 2) throw new Error("Complex numbers are not ordered.");
      return _Scalar.promote(this.value, targetLevel).cmp(_Scalar.promote(vb, targetLevel));
    }
    /**
     * Checks for equality with another value.
     * @param {Scalar|any} other
     * @returns {boolean}
     */
    equals(other) {
      const vb = other instanceof _Scalar ? other.value : other;
      const targetLevel = Math.max(this.level, _Scalar.getLevel(vb));
      return _Scalar.promote(this.value, targetLevel).equals(_Scalar.promote(vb, targetLevel));
    }
    /**
     * Converts the scalar to a string.
     * @param {number} [radix=10]
     * @param {number} [precision=-1]
     * @param {boolean} [pretty=false] pretty print
     * @returns {string}
     */
    toString(radix = 10, precision2 = -1, pretty = false) {
      return this.value.toString(radix, precision2, pretty);
    }
    /**
     * Parses string and determines correct type/level.
     * @param {string} str
     * @returns {Scalar}
     */
    static fromString(str) {
      const s = str.trim();
      if (s.includes("i")) {
        return new _Scalar(Complex.fromString(s));
      }
      if (s.includes("/")) {
        return new _Scalar(BigFraction.fromString(s));
      }
      if (s.includes(".") || s.toLowerCase().includes("e")) {
        let nv = parseFloat(s);
        if (Number.isInteger(nv)) {
          return new _Scalar(BigFraction.fromString(s));
        }
        return new _Scalar(BigFloat.fromString(s));
      }
      try {
        return new _Scalar(new BigFraction(s));
      } catch (e) {
        return new _Scalar(bf(s));
      }
    }
  };
  function scalar(s) {
    if (s instanceof Scalar) {
      return s;
    }
    return new Scalar(s);
  }

  // src/bf.js
  var Flags = {};
  Flags.BF_ST_INVALID_OP = 1 << 0;
  Flags.BF_ST_DIVIDE_ZERO = 1 << 1;
  Flags.BF_ST_OVERFLOW = 1 << 2;
  Flags.BF_ST_UNDERFLOW = 1 << 3;
  Flags.BF_ST_INEXACT = 1 << 4;
  Flags.BF_ST_MEM_ERROR = 1 << 5;
  Flags.BF_RADIX_MAX = 36;
  Flags.BF_ATOF_NO_HEX = 1 << 16;
  Flags.BF_ATOF_BIN_OCT = 1 << 17;
  Flags.BF_ATOF_NO_NAN_INF = 1 << 18;
  Flags.BF_ATOF_EXPONENT = 1 << 19;
  Flags.BF_RND_MASK = 7;
  Flags.BF_FTOA_FORMAT_MASK = 3 << 16;
  Flags.BF_FTOA_FORMAT_FIXED = 0 << 16;
  Flags.BF_FTOA_FORMAT_FRAC = 1 << 16;
  Flags.BF_FTOA_FORMAT_FREE = 2 << 16;
  Flags.BF_FTOA_FORMAT_FREE_MIN = 3 << 16;
  Flags.BF_FTOA_FORCE_EXP = 1 << 20;
  Flags.BF_FTOA_ADD_PREFIX = 1 << 21;
  Flags.BF_FTOA_JS_QUIRKS = 1 << 22;
  Flags.BF_POW_JS_QUIRKS = 1 << 16;
  Flags.BF_RNDN = 0;
  Flags.BF_RNDZ = 1;
  Flags.BF_RNDD = 2;
  Flags.BF_RNDU = 3;
  Flags.BF_RNDNA = 4;
  Flags.BF_RNDA = 5;
  Flags.BF_RNDF = 6;
  var gc_array = /* @__PURE__ */ new Set();
  var gcing = false;
  function gc() {
    if (gcing) return;
    gcing = true;
    let ele = [...gc_array].sort(
      (a, b) => {
        let diff2 = b.visited - a.visited;
        if (diff2 > 2 ** 31 || diff2 < -(2 ** 31)) {
          diff2 *= -1;
        }
        return diff2;
      }
    );
    let gcstartpos = Math.floor(gc_ele_limit / 2);
    for (let i = gcstartpos; i < ele.length; ++i) {
      let e = ele[i];
      e.dispose();
    }
    gc_array = new Set(ele.slice(0, gcstartpos));
    gcing = false;
  }
  var visit_index = 1;
  function gc_track(f, addToArray = true) {
    f.visited = visit_index;
    visit_index = (visit_index + 1 & 4294967295) + 1;
    if (addToArray) {
      gc_array.add(f);
      if (gc_array.size >= gc_ele_limit) {
        gc();
      }
    }
  }
  var recyclebin = [];
  function recycle(h) {
    const recyclebin_size = gc_ele_limit;
    if (recyclebin.length < recyclebin_size) {
      recyclebin.push(h);
    } else {
      libbf._delete_(h);
    }
  }
  function get_recycled_or_new() {
    if (recyclebin.length > 0) {
      return recyclebin.pop();
    }
    return libbf._new_();
  }
  var precision = 352;
  function setPrecision(p) {
    precision = p;
  }
  function getPrecision() {
    return precision;
  }
  var precision_array = [];
  function pushPrecision(prec) {
    precision_array.push(precision);
    precision = prec;
  }
  function popPrecision() {
    if (precision.length) {
      precision = precision_array.pop();
    }
  }
  function decimalPrecision(dp) {
    if (dp != void 0) {
      precision = Math.ceil(dp * Math.log2(10));
    } else {
      return Math.ceil(precision / Math.log2(10));
    }
  }
  function pushDecimalPrecision(dp) {
    pushPrecision(0);
    decimalPrecision(dp);
  }
  var EPSILONS_cache = [];
  function getEpsilon() {
    if (void 0 === EPSILONS_cache[precision]) {
      EPSILONS_cache[precision] = bf().setEPSILON().f64();
    }
    return EPSILONS_cache[precision];
  }
  var gc_ele_limit = 4e3;
  function setGcEleLimit(l) {
    gc_ele_limit = l;
  }
  function getGcEleLimit() {
    return gc_ele_limit;
  }
  function isReady() {
    return !!libbf;
  }
  var throwExceptionOnInvalidOp = false;
  function setThrowExceptionOnInvalidOp(f) {
    throwExceptionOnInvalidOp = f;
  }
  var libbf = null;
  var globalFlag = 0;
  function getGlobalFlag() {
    return globalFlag;
  }
  function setGlobalFlag(f) {
    globalFlag = f;
  }
  function fromUint8Array(data) {
    const BF_T_STRUCT_SIZE = 20;
    const limb_size = 4;
    const dataView = new DataView(data.buffer, 0, BF_T_STRUCT_SIZE);
    const len = dataView.getUint32(12, true);
    const limbs_byte_length = len * limb_size;
    const new_h = libbf._malloc(BF_T_STRUCT_SIZE);
    let new_tab_ptr = 0;
    if (len > 0) {
      new_tab_ptr = libbf._malloc(limbs_byte_length);
    }
    libbf.HEAPU8.set(data.subarray(0, BF_T_STRUCT_SIZE), new_h);
    if (new_tab_ptr !== 0) {
      libbf.HEAPU8.set(data.subarray(BF_T_STRUCT_SIZE, BF_T_STRUCT_SIZE + limbs_byte_length), new_tab_ptr);
    }
    libbf.HEAPU32[new_h / 4 + 4] = new_tab_ptr;
    return new_h;
  }
  var formatDecimal = (str, pretty = false) => {
    if (!str.includes(".")) return str;
    const [intPart, decPart] = str.split(".");
    const newDecPart = decPart.replace(/(\d)\1+$/, (match, digit) => {
      if (digit === "0") {
        return "";
      }
      if (pretty) {
        if (match.length > 6) {
          return `${digit.repeat(5)}(${digit})`;
        }
      }
      return match;
    });
    if (newDecPart == "") {
      return intPart;
    }
    return `${intPart}.${newDecPart}`;
  };
  var BigFloat = class _BigFloat {
    /**
     * Creates a new BigFloat instance.
     * @param {string | number | bigint | BigFloat} [val] - The value to initialize the BigFloat with.
     * @param {number} [radix=10] - The radix to use if `val` is a string.
     * @param {boolean} [managed=true] - Whether the BigFloat should be managed by the garbage collector.
     * @param {boolean} [constant=false] - Whether the BigFloat is a constant.
     */
    constructor(val, radix = 10, managed = true, constant = false) {
      this.hwrapper = [get_recycled_or_new()];
      this.managed = managed;
      this.status = 0;
      switch (typeof val) {
        case "undefined":
          break;
        case "string":
          this._fromString(val, radix);
          break;
        case "number":
          this.fromNumber(val);
          break;
        case "bigint":
          this._fromString(val.toString(), 10);
          break;
        case "object":
          if (!!val && val.constructor == _BigFloat) {
            this.copy(val);
            break;
          }
        default:
          throw new Error("BigFloat: invalid constructor oprand " + typeof val);
      }
      this.visited = 0;
      this.visit();
      this.constant = constant;
    }
    /**
     * The handle to the underlying C object.
     * @type {number}
     */
    get h() {
      return this.hwrapper[0];
    }
    set h(hv) {
      this.hwrapper[0] = hv;
    }
    /**
     * Marks the BigFloat as visited by the garbage collector.
     * @param {boolean} [addToArray=true] - Whether to add the BigFloat to the garbage collector's array.
     */
    visit(addToArray = true) {
      if (this.managed) {
        gc_track(this, addToArray);
      }
    }
    /**
     * Converts the BigFloat to a Uint8Array.
     * @returns {Uint8Array}
     */
    toUint8Array() {
      const BF_T_STRUCT_SIZE = 20;
      const limb_size = 4;
      const dataView = new DataView(libbf.HEAPU8.buffer, this.h, BF_T_STRUCT_SIZE);
      const len = dataView.getUint32(12, true);
      const tab_ptr = dataView.getUint32(16, true);
      const limbs_byte_length = len * limb_size;
      const total_backup_size = BF_T_STRUCT_SIZE + limbs_byte_length;
      const h_bak_data = new Uint8Array(total_backup_size);
      h_bak_data.set(new Uint8Array(libbf.HEAPU8.buffer, this.h, BF_T_STRUCT_SIZE), 0);
      if (tab_ptr !== 0 && len > 0) {
        h_bak_data.set(new Uint8Array(libbf.HEAPU8.buffer, tab_ptr, limbs_byte_length), BF_T_STRUCT_SIZE);
      }
      return h_bak_data;
    }
    /**
     * Disposes of the BigFloat's resources.
     * @param {boolean} [recoverable=true] - Whether the BigFloat can be recovered later.
     */
    dispose(recoverable = true) {
      if (this.h != 0) {
        if (recoverable) {
          this.h_bak = this.toUint8Array();
        }
        recycle(this.h);
        this.h = 0;
      }
    }
    /**
     * Gets the handle to the underlying C object, creating it if necessary.
     * @returns {number}
     */
    geth() {
      if (this.h == 0) {
        if (this.h_bak) {
          this.h = fromUint8Array(this.h_bak);
          this.h_bak = null;
        } else {
          this.h = get_recycled_or_new();
        }
        this.visit(true);
      } else {
        this.visit(false);
      }
      return this.h;
    }
    /**
     * Checks if the last operation resulted in an inexact result.
     * @returns {boolean}
     */
    isInExact() {
      return this.checkstatus(this.status);
    }
    /**
     * Checks the status of the last operation.
     * @param {number} s - The status to check.
     * @returns {number} The status.
     */
    checkstatus(s) {
      if (s & Flags.BF_ST_DIVIDE_ZERO) {
        console.log("libbf BF_ST_DIVIDE_ZERO, status=" + s);
      }
      if (s & Flags.BF_ST_INVALID_OP) {
        if (throwExceptionOnInvalidOp) {
          throw new Error("libbf BF_ST_INVALID_OP, status=");
        }
        console.log("libbf BF_ST_INVALID_OP, status=" + s);
      }
      return s;
    }
    /**
     * Wraps the given arguments in BigFloat handles.
     * @param {...(BigFloat | string | number | bigint | object)} ar - The arguments to wrap.
     * @returns {[function(): void, ...number[]]} An array containing a cleanup function and the handles.
     */
    wraptypeh(...ar) {
      let ret = [];
      let disposes = [];
      ret.push(function() {
        for (let e of disposes) {
          e.dispose(false);
        }
      });
      for (let a of ar) {
        if (a === null) {
          ret.push(0);
        } else if (a.constructor == _BigFloat) {
          ret.push(a.geth());
        } else if (typeof a == "string" || typeof a == "number" || typeof a == "bigint") {
          let b = new _BigFloat(a, 10, false, true);
          ret.push(b.h);
          disposes.push(b);
        } else if (typeof a === "object" && a.toBigFloat) {
          let b = a.toBigFloat();
          ret.push(b.h);
          disposes.push(b);
        } else {
          throw new Error("object is not a BigFloat " + a.constructor.name);
        }
      }
      return ret;
    }
    /**
     * Converts the BigFloat to a Complex number.
     * @returns {complex}
     */
    toComplex() {
      return complex(this, zero);
    }
    /**
     * 
     * @param {string} method 
     * @param {BigFloat | number | string | bigint | null} a 
     * @param {BigFloat | number | string | bigint | null} b 
     * @param {number} prec 
     * @param {number} [flags] - if set, or with globalflag
     * @param {number} [rnd_mode] - if set, overwrite round mode in globalflag
     * @returns {BigFloat} this
     */
    calc(method, a = null, b = null, prec, flags = void 0, rnd_mode = void 0) {
      if (this.constant) {
        throw new Error("constant");
      }
      if (prec < 1) prec = precision;
      let [cleanup, ah, bh] = this.wraptypeh(a, b);
      try {
        let targetflags = globalFlag;
        if (!(flags === void 0)) {
          targetflags = targetflags | flags & ~Flags.BF_RND_MASK;
        }
        if (!(rnd_mode === void 0)) {
          targetflags = targetflags & ~Flags.BF_RND_MASK | rnd_mode & Flags.BF_RND_MASK;
        }
        this.status |= libbf._calc(method.charCodeAt(0), this.geth(), ah, bh, prec, targetflags);
      } finally {
        cleanup();
      }
      this.checkstatus(this.status);
      return this;
    }
    /**
     * 
     * @param {string} method 
     * @param {BigFloat | number | string | bigint | null} a 
     * @param {BigFloat | number | string | bigint | null} b 
     * @param {number} prec 
     * @param {number} [rnd_mode=0]
     * @param {BigFloat | null} q 
     * @returns {BigFloat} this
     */
    calc2(method, a = null, b = null, prec, rnd_mode = 0, q = null) {
      if (this.constant) {
        throw new Error("constant");
      }
      if (prec < 1) prec = precision;
      let [cleanup, ah, bh, qh] = this.wraptypeh(a, b, q);
      try {
        this.status |= libbf._calc2(method.charCodeAt(0), this.geth(), ah, bh, prec, globalFlag, rnd_mode, qh);
      } finally {
        cleanup();
      }
      this.checkstatus(this.status);
      return this;
    }
    /**
     * Checks if the given arguments are valid operands.
     * @param {...any} args - The arguments to check.
     */
    checkoprand(...args) {
      for (let a of args) {
        if (a === null || typeof a == "undefined") {
          throw new Error("oprand missmatch");
        }
      }
    }
    /**
     * Sets this BigFloat to the sum of a and b.
     * @param {BigFloat | number | string | bigint} a 
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setadd(a, b, prec = 0) {
      this.checkoprand(a, b);
      return this.calc("+", a, b, prec);
    }
    /**
     * Sets this BigFloat to the difference of a and b.
     * @param {BigFloat | number | string | bigint} a 
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setsub(a, b, prec = 0) {
      this.checkoprand(a, b);
      return this.calc("-", a, b, prec);
    }
    /**
     * Sets this BigFloat to the product of a and b.
     * @param {BigFloat | number | string | bigint} a 
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setmul(a, b, prec = 0) {
      this.checkoprand(a, b);
      return this.calc("*", a, b, prec);
    }
    /**
     * Sets this BigFloat to the division of a and b.
     * @param {BigFloat | number | string | bigint} a 
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setdiv(a, b, prec = 0) {
      this.checkoprand(a, b);
      return this.calc("/", a, b, prec);
    }
    /**
     * Sets this BigFloat to the modulus of a and b.
     * @param {BigFloat | number | string | bigint} a 
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setmod(a, b, prec = 0) {
      this.checkoprand(a, b);
      return this.calc2("%", a, b, prec, Flags.BF_RNDZ, null);
    }
    /**
     * Sets this BigFloat to the remainder of a and b.
     * @param {BigFloat | number | string | bigint} a 
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setrem(a, b, prec = 0) {
      this.checkoprand(a, b);
      return this.calc2("%", a, b, prec, Flags.BF_RNDN, null);
    }
    /**
     * Sets this BigFloat to the bitwise OR of a and b.
     * @param {BigFloat | number | string | bigint} a 
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setor(a, b, prec = 0) {
      this.checkoprand(a, b);
      return this.calc("|", a, b, prec);
    }
    /**
     * Sets this BigFloat to the bitwise XOR of a and b.
     * @param {BigFloat | number | string | bigint} a 
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setxor(a, b, prec = 0) {
      this.checkoprand(a, b);
      return this.calc("^", a, b, prec);
    }
    /**
     * Sets this BigFloat to the bitwise AND of a and b.
     * @param {BigFloat | number | string | bigint} a 
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setand(a, b, prec = 0) {
      this.checkoprand(a, b);
      return this.calc("&", a, b, prec);
    }
    /**
     * Sets this BigFloat to the square root of a.
     * @param {BigFloat | number | string | bigint} a 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setsqrt(a, prec = 0) {
      this.checkoprand(a);
      return this.calc("s", a, null, prec);
    }
    /**
     * Rounds this BigFloat to a given precision.
     * @param {number} [prec=0] 
     * @param {number} [rnd_mode=Flags.BF_RNDN] 
     * @returns {BigFloat} this
     */
    setfpround(prec = 0, rnd_mode = Flags.BF_RNDN) {
      return this.calc("r", null, null, prec, void 0, rnd_mode);
    }
    /**
     * Rounds this BigFloat to the nearest integer.
     * @returns {BigFloat} this
     */
    setround() {
      return this.calc("i", null, null, 0, void 0, Flags.BF_RNDN);
    }
    /**
     * Truncates this BigFloat to an integer.
     * @returns {BigFloat} this
     */
    settrunc() {
      return this.calc("i", null, null, 0, void 0, Flags.BF_RNDZ);
    }
    /**
     * Floors this BigFloat to an integer.
     * @returns {BigFloat} this
     */
    setfloor() {
      return this.calc("i", null, null, 0, void 0, Flags.BF_RNDD);
    }
    /**
     * Ceils this BigFloat to an integer.
     * @returns {BigFloat} this
     */
    setceil() {
      return this.calc("i", null, null, 0, void 0, Flags.BF_RNDU);
    }
    /**
     * Negates this BigFloat.
     * @returns {BigFloat} this
     */
    setneg() {
      return this.calc("n", null, null, 0);
    }
    /**
     * Sets this BigFloat to its absolute value.
     * @returns {BigFloat} this
     */
    setabs() {
      return this.calc("b", null, null, 0);
    }
    /**
     * Sets this BigFloat to the sign of a.
     * @param {BigFloat | number | string | bigint} a 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setsign(a, prec = 0) {
      this.checkoprand(a);
      return this.calc("g", a, null, prec);
    }
    /**
     * Sets this BigFloat to the value of log2(e).
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setLOG2(prec = 0) {
      return this.calc("2", null, null, prec);
    }
    /**
     * Sets this BigFloat to the value of PI.
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setPI(prec = 0) {
      return this.calc("3", null, null, prec);
    }
    /**
     * Sets this BigFloat to its minimum value.
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setMIN_VALUE(prec = 0) {
      return this.calc("z", null, null, prec);
    }
    /**
     * Sets this BigFloat to its maximum value.
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setMAX_VALUE(prec = 0) {
      return this.calc("Z", null, null, prec);
    }
    /**
     * Sets this BigFloat to its epsilon value.
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setEPSILON(prec = 0) {
      return this.calc("y", null, null, prec);
    }
    /**
     * Sets this BigFloat to e^a.
     * @param {BigFloat | number | string | bigint} a 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setexp(a, prec = 0) {
      this.checkoprand(a);
      return this.calc("E", a, null, prec);
    }
    /**
     * Sets this BigFloat to log(a).
     * @param {BigFloat | number | string | bigint} a 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setlog(a, prec = 0) {
      this.checkoprand(a);
      return this.calc("L", a, null, prec);
    }
    /**
     * Sets this BigFloat to a^b.
     * @param {BigFloat | number | string | bigint} a 
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setpow(a, b, prec = 0) {
      this.checkoprand(a, b);
      return this.calc("P", a, b, prec, Flags.BF_POW_JS_QUIRKS);
    }
    /**
     * Sets this BigFloat to cos(a).
     * @param {BigFloat | number | string | bigint} a 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setcos(a, prec = 0) {
      this.checkoprand(a);
      return this.calc("C", a, null, prec);
    }
    /**
     * Sets this BigFloat to sin(a).
     * @param {BigFloat | number | string | bigint} a 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setsin(a, prec = 0) {
      this.checkoprand(a);
      return this.calc("S", a, null, prec);
    }
    /**
     * Sets this BigFloat to tan(a).
     * @param {BigFloat | number | string | bigint} a 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    settan(a, prec = 0) {
      this.checkoprand(a);
      return this.calc("T", a, null, prec);
    }
    /**
     * Sets this BigFloat to atan(a).
     * @param {BigFloat | number | string | bigint} a 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setatan(a, prec = 0) {
      this.checkoprand(a);
      return this.calc("4", a, null, prec);
    }
    /**
     * Sets this BigFloat to atan2(a, b).
     * @param {BigFloat | number | string | bigint} a 
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setatan2(a, b, prec = 0) {
      this.checkoprand(a, b);
      return this.calc("5", a, b, prec);
    }
    /**
     * Sets this BigFloat to asin(a).
     * @param {BigFloat | number | string | bigint} a 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setasin(a, prec = 0) {
      this.checkoprand(a);
      return this.calc("6", a, null, prec);
    }
    /**
     * Sets this BigFloat to acos(a).
     * @param {BigFloat | number | string | bigint} a 
     * @param {number} [prec=0] 
     * @returns {BigFloat} this
     */
    setacos(a, prec = 0) {
      this.checkoprand(a);
      return this.calc("7", a, null, prec);
    }
    /**
     * Checks if this BigFloat is finite.
     * @returns {boolean}
     */
    isFinit() {
      return libbf._is_finite_(this.geth());
    }
    /**
     * Checks if this BigFloat is NaN.
     * @returns {boolean}
     */
    isNaN() {
      return libbf._is_nan_(this.geth());
    }
    /**
     * Checks if this BigFloat is exactly zero.
     * @returns {boolean}
     */
    isExactZero() {
      return libbf._is_zero_(this.geth());
    }
    /**
     * Checks if this BigFloat is zero.
     * @returns {boolean}
     */
    isZero() {
      return this.isExactZero();
    }
    /**
     * Checks if this BigFloat is almost zero.
     * @returns {boolean}
     */
    isAlmostZero() {
      return Math.abs(this.f64()) <= getEpsilon();
    }
    /**
     * Copies the value of another BigFloat to this one.
     * @param {BigFloat} a - The BigFloat to copy from.
     * @returns {void}
     */
    copy(a) {
      if (this.constant) {
        throw new Error("constant");
      }
      this.checkoprand(a);
      return libbf._set_(this.geth(), a.geth());
    }
    /**
     * Clones this BigFloat.
     * @returns {BigFloat}
     */
    clone() {
      return new _BigFloat(this);
    }
    /**
     * Sets the value of this BigFloat from a number.
     * @param {number} a 
     * @returns {void}
     */
    fromNumber(a) {
      if (this.constant) {
        throw new Error("constant");
      }
      return libbf._set_number_(this.geth(), a);
    }
    /**
     * Converts this BigFloat to a 64-bit float.
     * @returns {number}
     */
    f64() {
      return libbf._get_number_(this.geth());
    }
    /**
     * Converts this BigFloat to a number.
     * @returns {number}
     */
    toNumber() {
      return libbf._get_number_(this.geth());
    }
    /**
     * Compares this BigFloat with another one.
     * @param {BigFloat | number | string | bigint} b 
     * @returns {number} 0 if equal, >0 if this > b, <0 if this < b.
     */
    cmp(b) {
      this.checkoprand(b);
      let [cleanup, bh] = this.wraptypeh(b);
      let ret;
      try {
        ret = libbf._cmp_(this.geth(), bh);
      } finally {
        cleanup();
      }
      return ret;
    }
    /**
     * @param {BigFloat | number | string | bigint} b 
     * @returns {boolean}
     */
    operatorLess(b) {
      return this.cmp(b) < 0;
    }
    /**
     * @param {BigFloat | number | string | bigint} b 
     * @returns {boolean}
     */
    operatorGreater(b) {
      return this.cmp(b) > 0;
    }
    /**
     * @param {BigFloat | number | string | bigint} b 
     * @returns {boolean}
     */
    operatorLessEqual(b) {
      return this.cmp(b) <= 0;
    }
    /**
     * @param {BigFloat | number | string | bigint} b 
     * @returns {boolean}
     */
    operatorGreaterEqual(b) {
      return this.cmp(b) >= 0;
    }
    /**
     * @param {BigFloat | number | string | bigint} b 
     * @returns {boolean}
     */
    operatorEqual(b) {
      return this.cmp(b) == 0;
    }
    /**
     * @param {BigFloat | number | string | bigint} b 
     * @returns {boolean}
     */
    operatorNotEqual(b) {
      return this.cmp(b) != 0;
    }
    /**
     * @param {BigFloat | number | string | bigint} b 
     * @returns {boolean}
     */
    equals(b) {
      return this.operatorEqual(b);
    }
    /**
     * @param {string} str 
     * @param {number} [radix=10] 
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    static fromString(str, radix = 10, prec = 0) {
      return new _BigFloat()._fromString(str, radix, prec);
    }
    /**
     * @private
     * @param {string} str 
     * @param {number} [radix=10] 
     * @param {number} [prec=0] 
     * @returns {this}
     */
    _fromString(str, radix = 10, prec = 0) {
      if (this.constant) {
        throw new Error("constant");
      }
      if (radix > 64) throw new Error("radix error");
      if (prec < 1) prec = precision;
      let hstr = libbf.allocateUTF8(str);
      let ret = libbf._atof_(this.geth(), hstr, radix, prec, 0);
      libbf._free(hstr);
      this.checkstatus(ret);
      return this;
    }
    /**
     * toString
     * @param {number} [radix=10] 
     * @param {number} [prec=-1] precision digits in radix
     * @param {boolean} [pretty=false] pretty print
     * @returns {string}
     */
    toString(radix = 10, prec = -1, pretty = false) {
      if (radix > 64) throw new Error("radix error");
      if (isNaN(prec)) throw new Error("prec is NaN");
      if (prec <= 0) prec = Math.ceil(precision / Math.log2(radix));
      let flag = 0;
      flag = Flags.BF_FTOA_FORMAT_FIXED | Flags.BF_RNDZ | Flags.BF_FTOA_JS_QUIRKS;
      let ret = libbf._ftoa_(0, this.geth(), radix, prec, flag);
      let rets = libbf.AsciiToString(ret);
      libbf._free(ret);
      return formatDecimal(rets, pretty);
    }
    /**
     * 
     * @param {number} [radix=10] 
     * @param {number} [prec=-1] precision digits in radix
     * @param {number} [rnd_mode=Flags.BF_RNDNA] 
     * @returns {string}
     */
    toFixed(radix = 10, prec = -1, rnd_mode = Flags.BF_RNDNA) {
      if (radix > 64) throw new Error("radix error");
      if (prec < 0) prec = Math.floor(precision / Math.log2(radix));
      let flag = 0;
      flag = rnd_mode | Flags.BF_FTOA_FORMAT_FRAC;
      let ret = libbf._ftoa_(0, this.geth(), radix, prec, flag);
      let rets = libbf.AsciiToString(ret);
      libbf._free(ret);
      return rets;
    }
    /**
     * @returns {bigint}
     */
    toBigInt() {
      let s = this.toString(10, 0);
      return BigInt(s);
    }
    /**
     * @param {function} ofunc 
     * @param {number} numps 
     * @param {...any} args 
     * @returns {BigFloat}
     */
    callFunc(ofunc, numps, ...args) {
      if (numps == 0) {
        return ofunc.apply(new _BigFloat(this));
      } else if (numps == 2) {
        return ofunc.apply(new _BigFloat(), [this, ...args]);
      } else {
        return ofunc.apply(new _BigFloat(), [this, ...args]);
      }
    }
    /**
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    add(b, prec = 0) {
      return this.callFunc(this.setadd, 3, b, prec);
    }
    /**
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    sub(b, prec = 0) {
      return this.callFunc(this.setsub, 3, b, prec);
    }
    /**
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    mul(b, prec = 0) {
      return this.callFunc(this.setmul, 3, b, prec);
    }
    /**
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    div(b, prec = 0) {
      return this.callFunc(this.setdiv, 3, b, prec);
    }
    /**
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    mod(b, prec = 0) {
      return this.callFunc(this.setmod, 3, b, prec);
    }
    /**
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    rem(b, prec = 0) {
      return this.callFunc(this.setrem, 3, b, prec);
    }
    /**
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    or(b, prec = 0) {
      return this.callFunc(this.setor, 3, b, prec);
    }
    /**
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    xor(b, prec = 0) {
      return this.callFunc(this.setxor, 3, b, prec);
    }
    /**
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    and(b, prec = 0) {
      return this.callFunc(this.setand, 3, b, prec);
    }
    /**
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    sqrt(prec = 0) {
      return this.callFunc(this.setsqrt, 2, prec);
    }
    /**
     * @param {number} prec 
     * @param {number} rnd_mode 
     * @returns {BigFloat}
     */
    fpround(prec, rnd_mode) {
      return bf(this).setfpround(prec, rnd_mode);
    }
    /**
     * @returns {BigFloat}
     */
    round() {
      return this.callFunc(this.setround, 0);
    }
    /**
     * @returns {BigFloat}
     */
    trunc() {
      return this.callFunc(this.settrunc, 0);
    }
    /**
     * @returns {BigFloat}
     */
    floor() {
      return this.callFunc(this.setfloor, 0);
    }
    /**
     * @returns {BigFloat}
     */
    ceil() {
      return this.callFunc(this.setceil, 0);
    }
    /**
     * @returns {BigFloat}
     */
    neg() {
      return this.callFunc(this.setneg, 0);
    }
    /**
     * @returns {BigFloat}
     */
    abs() {
      return this.callFunc(this.setabs, 0);
    }
    /**
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    sign(prec = 0) {
      return this.callFunc(this.setsign, 2, prec);
    }
    /**
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    exp(prec = 0) {
      return this.callFunc(this.setexp, 2, prec);
    }
    /**
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    log(prec = 0) {
      return this.callFunc(this.setlog, 2, prec);
    }
    /**
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    pow(b, prec = 0) {
      return this.callFunc(this.setpow, 3, b, prec);
    }
    /**
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    cos(prec = 0) {
      return this.callFunc(this.setcos, 2, prec);
    }
    /**
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    sin(prec = 0) {
      return this.callFunc(this.setsin, 2, prec);
    }
    /**
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    cosh(prec = 0) {
      let exp2 = bf(void 0, 10, false, false).setexp(this, prec);
      let one_div_exp = bf(void 0, 10, false, false).setdiv(one, exp2, prec);
      let ret = bf(void 0).setadd(exp2, one_div_exp, prec);
      ret.setmul(ret, half, prec);
      exp2.dispose(false);
      one_div_exp.dispose(false);
      return ret;
    }
    /**
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    sinh(prec = 0) {
      let exp2 = bf(void 0, 10, false, false).setexp(this, prec);
      let one_div_exp = bf(void 0, 10, false, false).setdiv(one, exp2, prec);
      let ret = bf(void 0).setsub(exp2, one_div_exp, prec);
      ret.setmul(ret, half, prec);
      exp2.dispose(false);
      one_div_exp.dispose(false);
      return ret;
    }
    /**
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    tanh(prec = 0) {
      let exp2 = bf(void 0, 10, false, false).setexp(this, prec);
      let one_div_exp = bf(void 0, 10, false, false).setdiv(one, exp2, prec);
      let s = bf(void 0, 10, false, false).setsub(exp2, one_div_exp, prec);
      let c = bf(void 0, 10, false, false).setadd(exp2, one_div_exp, prec);
      let ret = bf(void 0).setdiv(s, c, prec);
      exp2.dispose(false);
      one_div_exp.dispose(false);
      s.dispose(false);
      c.dispose(false);
      return ret;
    }
    /**
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    tan(prec = 0) {
      return this.callFunc(this.settan, 2, prec);
    }
    /**
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    atan(prec = 0) {
      return this.callFunc(this.setatan, 2, prec);
    }
    /**
     * @param {BigFloat | number | string | bigint} b 
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    atan2(b, prec = 0) {
      return this.callFunc(this.setatan2, 3, b, prec);
    }
    /**
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    asin(prec = 0) {
      return this.callFunc(this.setasin, 2, prec);
    }
    /**
     * @param {number} [prec=0] 
     * @returns {BigFloat}
     */
    acos(prec = 0) {
      return this.callFunc(this.setacos, 2, prec);
    }
  };
  BigFloat.prototype.operatorAdd = BigFloat.prototype.add;
  BigFloat.prototype.operatorSub = BigFloat.prototype.sub;
  BigFloat.prototype.operatorMul = BigFloat.prototype.mul;
  BigFloat.prototype.operatorDiv = BigFloat.prototype.div;
  BigFloat.prototype.operatorPow = BigFloat.prototype.pow;
  BigFloat.prototype.operatorMod = BigFloat.prototype.mod;
  BigFloat.prototype.operatorNeg = BigFloat.prototype.neg;
  BigFloat.prototype.operatorBitwiseAnd = BigFloat.prototype.and;
  BigFloat.prototype.operatorBitwiseOr = BigFloat.prototype.or;
  BigFloat.prototype.operatorBitwiseXor = BigFloat.prototype.xor;
  BigFloat.prototype.operatorBitwiseNot = BigFloat.prototype.not;
  function bf(val, radix = 10, managed = true, constant = false) {
    return new BigFloat(val, radix, managed, constant);
  }
  var minus_one = null;
  var zero = null;
  var half = null;
  var one = null;
  var two = null;
  var three = null;
  var PI = null;
  var E = null;
  var Constants = {
    /** @type {BigFloat | null} */
    get minus_one() {
      return minus_one;
    },
    /** @type {BigFloat | null} */
    get zero() {
      return zero;
    },
    /** @type {BigFloat | null} */
    get half() {
      return half;
    },
    /** @type {BigFloat | null} */
    get one() {
      return one;
    },
    /** @type {BigFloat | null} */
    get two() {
      return two;
    },
    /** @type {BigFloat | null} */
    get three() {
      return three;
    },
    /** @type {BigFloat | null} */
    get PI() {
      return PI;
    },
    /** @type {BigFloat | null} */
    get E() {
      return E;
    }
  };
  var inited = false;
  async function init(m) {
    if (inited) {
      return;
    }
    m._init_context_();
    libbf = m;
    minus_one = bf(-1, 10, false, true);
    zero = bf(0, 10, false, true);
    half = bf(0.5, 10, false, true);
    one = bf(1, 10, false, true);
    two = bf(2, 10, false, true);
    three = bf(3, 10, false, true);
    PI = bf(bf().setPI(), 10, false, true);
    E = bf(bf(1).exp(), 10, false, true);
    inited = true;
    return true;
  }
  function max(...args) {
    args = args.map((v) => v instanceof BigFloat ? v : bf(v));
    let ret = args[0];
    for (let i = 1; i < args.length; ++i) {
      if (args[i].cmp(ret) > 0) {
        ret = args[i];
      }
    }
    return ret;
  }
  function min(...args) {
    args = args.map((v) => v instanceof BigFloat ? v : bf(v));
    let ret = args[0];
    for (let i = 1; i < args.length; ++i) {
      if (args[i].cmp(ret) < 0) {
        ret = args[i];
      }
    }
    return ret;
  }
  function sqrt(v, prec = 0) {
    return bf(v).sqrt(prec);
  }
  function fpround(v, prec = 0, rnd_mode = 0) {
    return bf(v).fpround(prec, rnd_mode);
  }
  function round(v, prec = 0) {
    return bf(v).round(prec);
  }
  function trunc(v, prec = 0) {
    return bf(v).trunc(prec);
  }
  function floor(v, prec = 0) {
    return bf(v).floor(prec);
  }
  function ceil(v, prec = 0) {
    return bf(v).ceil(prec);
  }
  function neg(v, prec = 0) {
    return bf(v).neg(prec);
  }
  function abs(v, prec = 0) {
    return bf(v).abs(prec);
  }
  function sign(v, prec = 0) {
    return bf(v).sign(prec);
  }
  function exp(v, prec = 0) {
    return bf(v).exp(prec);
  }
  function log(v, prec = 0) {
    return bf(v).log(prec);
  }
  function pow(v, b, prec = 0) {
    return bf(v).pow(b, prec);
  }
  function cos(v, prec = 0) {
    return bf(v).cos(prec);
  }
  function sin(v, prec = 0) {
    return bf(v).sin(prec);
  }
  function tan(v, prec = 0) {
    return bf(v).tan(prec);
  }
  function atan(v, prec = 0) {
    return bf(v).atan(prec);
  }
  function atan2(v, prec = 0) {
    return bf(v).atan2(prec);
  }
  function asin(v, prec = 0) {
    return bf(v).asin(prec);
  }
  function acos(v, prec = 0) {
    return bf(v).acos(prec);
  }
  function safe_bf(v) {
    return v instanceof BigFloat ? v : bf(v);
  }
  function linspace(start, end, n) {
    const arr = [];
    start = safe_bf(start);
    end = safe_bf(end);
    n = typeof n == "number" ? n : safe_bf(n).toNumber();
    const step = end.sub(start).div(n - 1);
    for (let i = 0; i < n; i++) arr.push(start.add(step.mul(i)));
    return arr;
  }
  return __toCommonJS(bf_exports);
})();
