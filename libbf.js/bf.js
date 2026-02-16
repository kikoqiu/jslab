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
    E: () => E,
    Flags: () => Flags,
    O: () => O,
    PI: () => PI,
    Poly: () => Poly,
    Scalar: () => Scalar,
    X: () => X,
    abs: () => abs,
    acos: () => acos,
    asin: () => asin,
    atan: () => atan,
    atan2: () => atan2,
    bf: () => bf,
    ceil: () => ceil,
    complex: () => complex,
    cos: () => cos,
    decimal_precision: () => decimal_precision,
    exp: () => exp,
    floor: () => floor,
    fminbnd: () => fminbnd,
    fpround: () => fpround,
    frac: () => frac,
    fzero: () => fzero,
    gc_ele_limit: () => gc_ele_limit,
    globalFlag: () => globalFlag,
    half: () => half,
    init: () => init,
    integral: () => romberg,
    is_ready: () => is_ready,
    libbf: () => libbf,
    log: () => log,
    max: () => max,
    minus_one: () => minus_one,
    neg: () => neg,
    ode45: () => ode45,
    one: () => one,
    poly: () => poly,
    polyStr: () => polyStr,
    polyfit: () => polyfit,
    polyval: () => polyval,
    pop_precision: () => pop_precision,
    pow: () => pow,
    precision: () => precision,
    push_decimal_precision: () => push_decimal_precision,
    push_precision: () => push_precision,
    romberg: () => romberg,
    roots: () => roots,
    round: () => round,
    scalar: () => scalar,
    setGlobalFlag: () => setGlobalFlag,
    setPrecision: () => setPrecision,
    setThrowExceptionOnInvalidOp: () => setThrowExceptionOnInvalidOp,
    set_gc_ele_limit: () => set_gc_ele_limit,
    sign: () => sign,
    sin: () => sin,
    solveLinearSystem: () => solveLinearSystem,
    sqrt: () => sqrt,
    tan: () => tan,
    three: () => three,
    throwExceptionOnInvalidOp: () => throwExceptionOnInvalidOp,
    trunc: () => trunc,
    two: () => two,
    zero: () => zero
  });

  // src/complex.js
  var Complex = class _Complex {
    /**
     * @param {number|string|BigFloat|Complex} re - Real part or Complex object
     * @param {number|string|BigFloat} [im=0] - Imaginary part
     */
    constructor(re, im) {
      if (re instanceof _Complex) {
        this.re = re.re;
        this.im = re.im;
      } else {
        this.re = bf(re);
        this.im = im === void 0 ? bf(0) : bf(im);
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
      const b = this._wrap(other);
      const denom = b.re.mul(b.re).add(b.im.mul(b.im));
      if (denom.cmp(bf(0)) === 0) throw new Error("Complex division by zero");
      const ac = this.re.mul(b.re);
      const bd = this.im.mul(b.im);
      const bc = this.im.mul(b.re);
      const ad = this.re.mul(b.im);
      const newRe = ac.add(bd).div(denom);
      const newIm = bc.sub(ad).div(denom);
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
      return this.re.mul(this.re).add(this.im.mul(this.im)).sqrt();
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
      const re = r.add(this.re).mul(half).sqrt();
      const im = r.sub(this.re).mul(half).sqrt();
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
      return new _Complex(
        x.sin().mul(y.cosh()),
        x.cos().mul(y.sinh())
      );
    }
    /**
     * Trigonometric Cosine cos(z)
     * cos(x+iy) = cos(x)cosh(y) - i sin(x)sinh(y)
     * @returns {Complex}
     */
    cos() {
      const x = this.re;
      const y = this.im;
      return new _Complex(
        x.cos().mul(y.cosh()),
        x.sin().mul(y.sinh()).neg()
      );
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
      return new _Complex(
        x.sinh().mul(y.cos()),
        x.cosh().mul(y.sin())
      );
    }
    /**
     * Hyperbolic Cosine cosh(z)
     * cosh(x+iy) = cosh(x)cos(y) + i sinh(x)sin(y)
     * @returns {Complex}
     */
    cosh() {
      const x = this.re;
      const y = this.im;
      return new _Complex(
        x.cosh().mul(y.cos()),
        x.sinh().mul(y.sin())
      );
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
      const i = new _Complex(0, 1);
      const one2 = new _Complex(1, 0);
      const iz = i.mul(this);
      const sqrtPart = one2.sub(this.mul(this)).sqrt();
      return iz.add(sqrtPart).log().mul(i.neg());
    }
    /**
     * Inverse Cosine acos(z) = -i * ln(z + i*sqrt(1 - z^2))
     * @returns {Complex}
     */
    acos() {
      const i = new _Complex(0, 1);
      const one2 = new _Complex(1, 0);
      const sqrtPart = one2.sub(this.mul(this)).sqrt();
      return this.add(i.mul(sqrtPart)).log().mul(i.neg());
    }
    /**
     * Inverse Tangent atan(z) = (i/2) * ln((i+z)/(i-z))
     * @returns {Complex}
     */
    atan() {
      const i = new _Complex(0, 1);
      const halfI = new _Complex(0, 0.5);
      const numerator = i.add(this);
      const denominator = i.sub(this);
      return numerator.div(denominator).log().mul(halfI.neg());
    }
    /**
     * Inverse Hyperbolic Sine asinh(z) = ln(z + sqrt(z^2 + 1))
     * @returns {Complex}
     */
    asinh() {
      const one2 = new _Complex(1, 0);
      return this.add(this.mul(this).add(one2).sqrt()).log();
    }
    /**
     * Inverse Hyperbolic Cosine acosh(z) = ln(z + sqrt(z^2 - 1))
     * @returns {Complex}
     */
    acosh() {
      const one2 = new _Complex(1, 0);
      return this.add(this.mul(this).sub(one2).sqrt()).log();
    }
    /**
     * Inverse Hyperbolic Tangent atanh(z) = 0.5 * ln((1+z)/(1-z))
     * @returns {Complex}
     */
    atanh() {
      const one2 = new _Complex(1, 0);
      const half2 = new _Complex(0.5, 0);
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
     * @param {number} [precision=20]
     * @param {boolean} [pretty=false] pretty print
     * @returns {string}
     */
    toString(base = 10, precision2 = 20, pretty = false) {
      let rezero = this.re.isZero();
      let imzero = this.im.isZero();
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
    let max_steps_limit = info.max_step || 1e5;
    let max_time = info.max_time || 6e4;
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
    let direction = t_final.sub(t_start).sign();
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
    if (info.status === "running") info.status = "done";
    info.exectime = (/* @__PURE__ */ new Date()).getTime() - start_time;
    info.steps = steps;
    info.toString = function() {
      return `status=${this.status}, steps=${this.steps}, failed=${this.failed_steps}, t_final=${this.t[this.t.length - 1].toString(10, 6)}`;
    };
    return { t: info.t, y: info.y };
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
        info.eff_decimal_precision = decimal_precision();
      } else {
        info.eff_decimal_precision = Math.floor(-errorBound.log().f64() / Math.log(10));
      }
      if (info.eff_decimal_precision <= 0) {
        info.eff_decimal_precision = 0;
        info.eff_result = "";
      } else {
        let limit = decimal_precision();
        let prec = info.eff_decimal_precision > limit ? limit : info.eff_decimal_precision;
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
        info.eff_decimal_precision = decimal_precision();
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
        info.eff_decimal_precision = decimal_precision();
      } else {
        info.eff_decimal_precision = Math.floor(-errorBound.log().f64() / Math.log(10));
      }
      if (info.eff_decimal_precision <= 0) {
        info.eff_decimal_precision = 0;
        info.eff_result = "";
      } else {
        let limit = decimal_precision();
        let prec = info.eff_decimal_precision > limit ? limit : info.eff_decimal_precision;
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
        info.eff_decimal_precision = decimal_precision();
      } else {
        info.eff_decimal_precision = Math.floor(-info.rerror.log().f64() / Math.log(10));
      }
      if (info.eff_decimal_precision <= 0) {
        info.eff_decimal_precision = 0;
        info.eff_result = "";
      } else {
        if (info.eff_decimal_precision > decimal_precision()) {
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
    if (sign2 < 0n) num = -num;
    return { n: num, d: den };
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
     * @param {BigFraction | bigint | number | string} [n] - The numerator or the whole value.
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
      } else {
        num = 0n;
        den = 1n;
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
      const limit = Math.min(this.o, B.o);
      let i = 0, j = 0;
      while (i < this.degs.length || j < B.degs.length) {
        const degA = i < this.degs.length ? this.degs[i] : Infinity;
        const degB = j < B.degs.length ? B.degs[j] : Infinity;
        const minDeg = Math.min(degA, degB);
        if (minDeg >= limit) break;
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
      const limit = relPrec;
      const b = [new this.coefType(1)];
      const a = new Array(limit + 1).fill(null);
      for (const [deg, val] of deltaMap) {
        if (deg <= limit) a[deg] = val;
      }
      for (let k = 1; k < limit; k++) {
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
        let diff = b.visited - a.visited;
        if (diff > 2 ** 31 || diff < -(2 ** 31)) {
          diff *= -1;
        }
        return diff;
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
  var visit_index = 0;
  function gc_track(f, addToArray = true) {
    f.visited = visit_index;
    visit_index = (visit_index + 1) % 2 ** 32;
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
  var precision = 500;
  function setPrecision(p) {
    precision = p;
  }
  var precision_array = [];
  function push_precision(prec) {
    precision_array.push(precision);
    precision = prec;
  }
  function pop_precision() {
    if (precision.length) {
      precision = precision_array.pop();
    }
  }
  function decimal_precision(dp) {
    if (dp != void 0) {
      precision = Math.ceil(dp * Math.log2(10));
    } else {
      return Math.ceil(precision / Math.log2(10));
    }
  }
  function push_decimal_precision(dp) {
    push_precision(0);
    decimal_precision(dp);
  }
  var gc_ele_limit = 200;
  function set_gc_ele_limit(l) {
    gc_ele_limit = l;
  }
  function is_ready() {
    return !!libbf;
  }
  var throwExceptionOnInvalidOp = false;
  function setThrowExceptionOnInvalidOp(f) {
    throwExceptionOnInvalidOp = f;
  }
  var libbf = null;
  var globalFlag = 0;
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
  var EPSILONS_cache = [];
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
     * Checks if this BigFloat is almost zero.
     * @returns {boolean}
     */
    isZero() {
      return this.isAlmostZero();
    }
    /**
     * Gets the epsilon value for the current precision.
     * @returns {number}
     */
    getEpsilon() {
      if (void 0 === EPSILONS_cache[precision]) {
        EPSILONS_cache[precision] = bf().setEPSILON().f64();
      }
      return EPSILONS_cache[precision];
    }
    /**
     * Checks if this BigFloat is almost zero.
     * @returns {boolean}
     */
    isAlmostZero() {
      return Math.abs(this.f64()) <= this.getEpsilon();
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
    let t = args.map((v) => v instanceof BigFloat ? v : bf(v));
    let ret = args[0];
    for (let i = 1; i < args.length; ++i) {
      if (args[i].cmp(ret) > 0) {
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
  return __toCommonJS(bf_exports);
})();
