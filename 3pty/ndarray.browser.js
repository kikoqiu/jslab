"use strict";
var ndarray = (() => {
  var __defProp = Object.defineProperty;
  var __getOwnPropDesc = Object.getOwnPropertyDescriptor;
  var __getOwnPropNames = Object.getOwnPropertyNames;
  var __hasOwnProp = Object.prototype.hasOwnProperty;
  var __require = /* @__PURE__ */ ((x) => typeof require !== "undefined" ? require : typeof Proxy !== "undefined" ? new Proxy(x, {
    get: (a, b) => (typeof require !== "undefined" ? require : a)[b]
  }) : x)(function(x) {
    if (typeof require !== "undefined") return require.apply(this, arguments);
    throw Error('Dynamic require of "' + x + '" is not supported');
  });
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

  // src/ndarray.js
  var ndarray_exports = {};
  __export(ndarray_exports, {
    DTYPE_MAP: () => DTYPE_MAP,
    Jit: () => Jit,
    NDArray: () => NDArray,
    NDProb: () => NDProb,
    NDWasm: () => NDWasm,
    NDWasmAnalysis: () => NDWasmAnalysis,
    NDWasmArray: () => NDWasmArray,
    NDWasmBlas: () => NDWasmBlas,
    NDWasmDecomp: () => NDWasmDecomp,
    NDWasmImage: () => NDWasmImage,
    NDWasmOptimize: () => NDWasmOptimize,
    NDWasmSignal: () => NDWasmSignal,
    WasmBuffer: () => WasmBuffer,
    WasmRuntime: () => WasmRuntime,
    analysis: () => analysis2,
    arange: () => arange,
    array: () => array,
    blas: () => blas2,
    concat: () => concat,
    decomp: () => decomp2,
    default: () => ndarray_default,
    eye: () => eye,
    float32: () => float32,
    float64: () => float64,
    fromWasm: () => fromWasm,
    full: () => full,
    help: () => help,
    image: () => image,
    init: () => init,
    int16: () => int16,
    int32: () => int32,
    int8: () => int8,
    linspace: () => linspace,
    ones: () => ones,
    optimize: () => optimize,
    random: () => random,
    signal: () => signal2,
    stack: () => stack,
    uint16: () => uint16,
    uint32: () => uint32,
    uint8: () => uint8,
    uint8c: () => uint8c,
    where: () => where,
    zeros: () => zeros
  });

  // src/ndarray_prob.js
  var NDProb = {
    /**
     * Internal helper to get a Float64 between [0, 1) using crypto.
     * Generates high-quality uniform random floats.
     * @private
     */
    _cryptoUniform01(size) {
      const uint322 = new Uint32Array(size);
      const cryptoObj = globalThis.crypto;
      const quota = 65536 / 4;
      for (let i = 0; i < size; i += quota) {
        const chunk = uint322.subarray(i, Math.min(i + quota, size));
        cryptoObj.getRandomValues(chunk);
      }
      const floats = new Float64Array(size);
      for (let i = 0; i < size; i++) {
        floats[i] = uint322[i] / 4294967296;
      }
      return floats;
    },
    /**
     * Uniform distribution over [low, high).
     * @memberof NDProb
     * @function
     * @param {Array} shape - Dimensions of the output array.
     * @param {number} [low=0] - Lower boundary.
     * @param {number} [high=1] - Upper boundary.
     * @param {string} [dtype='float64'] - Data type.
     * @returns {NDArray}
     */
    random(shape, low = 0, high = 1, dtype = "float64") {
      const size = shape.reduce((a, b) => a * b, 1);
      const data = this._cryptoUniform01(size);
      const range = high - low;
      if (range !== 1 || low !== 0) {
        for (let i = 0; i < size; i++) data[i] = data[i] * range + low;
      }
      const finalData = dtype === "float64" ? data : this._cast(data, dtype);
      return new NDArray(finalData, { shape, dtype });
    },
    /**
     * Normal (Gaussian) distribution using Box-Muller transform.
     * @param {Array} shape - Dimensions of the output array.
     * @param {number} [mean=0] - Mean of the distribution.
     * @param {number} [std=1] - Standard deviation.
     * @param {string} [dtype='float64'] - Data type.
     * @memberof NDProb
     * @returns {NDArray}
     */
    normal(shape, mean = 0, std = 1, dtype = "float64") {
      const size = shape.reduce((a, b) => a * b, 1);
      const u1 = this._cryptoUniform01(Math.ceil(size / 2) * 2);
      const u2 = this._cryptoUniform01(u1.length);
      const data = new Float64Array(size);
      for (let i = 0; i < size; i += 2) {
        const r = Math.sqrt(-2 * Math.log(Math.max(u1[i], 1e-10)));
        const theta = 2 * Math.PI * u2[i];
        data[i] = r * Math.cos(theta) * std + mean;
        if (i + 1 < size) {
          data[i + 1] = r * Math.sin(theta) * std + mean;
        }
      }
      const finalData = dtype === "float64" ? data : this._cast(data, dtype);
      return new NDArray(finalData, { shape, dtype });
    },
    /**
     * Bernoulli distribution (0 or 1 with probability p).
     * @param {Array} shape
     * @param {number} [p=0.5] - Probability of success (1).
     * @param {string} [dtype='int32']
     * @memberof NDProb
     * @returns {NDArray}
     */
    bernoulli(shape, p = 0.5, dtype = "int32") {
      const size = shape.reduce((a, b) => a * b, 1);
      const u = this._cryptoUniform01(size);
      const data = new DTYPE_MAP[dtype](size);
      for (let i = 0; i < size; i++) {
        data[i] = u[i] < p ? 1 : 0;
      }
      return new NDArray(data, { shape, dtype });
    },
    /**
     * Exponential distribution: f(x; λ) = λe^(-λx).
     * Inverse transform sampling: x = -ln(1-u) / λ.
     * @param {Array} shape
     * @param {number} [lambda=1.0] - Rate parameter.
     * @memberof NDProb
     * @returns {NDArray}
     */
    exponential(shape, lambda = 1, dtype = "float64") {
      const size = shape.reduce((a, b) => a * b, 1);
      const u = this._cryptoUniform01(size);
      const data = new Float64Array(size);
      for (let i = 0; i < size; i++) {
        data[i] = -Math.log(Math.max(1 - u[i], 1e-10)) / lambda;
      }
      const finalData = dtype === "float64" ? data : this._cast(data, dtype);
      return new NDArray(finalData, { shape, dtype });
    },
    /**
     * Poisson distribution using Knuth's algorithm.
     * Note: For very large lambda, this becomes slow; but for most use cases it's fine.
     * @param {Array} shape
     * @param {number} [lambda=1.0] - Mean of the distribution.
     * @memberof NDProb
     * @returns {NDArray}
     */
    poisson(shape, lambda = 1, dtype = "int32") {
      const size = shape.reduce((a, b) => a * b, 1);
      const data = new DTYPE_MAP[dtype](size);
      const L = Math.exp(-lambda);
      for (let i = 0; i < size; i++) {
        let k = 0;
        let p = 1;
        do {
          k++;
          p *= this._cryptoUniform01(1)[0];
        } while (p > L);
        data[i] = k - 1;
      }
      return new NDArray(data, { shape, dtype });
    },
    /** 
     * @private
     * @internal
     */
    _cast(data, dtype) {
      const out = new DTYPE_MAP[dtype](data.length);
      out.set(data);
      return out;
    }
  };

  // src/ndwasm.js
  var ndwasm_exports = {};
  __export(ndwasm_exports, {
    NDWasm: () => NDWasm,
    WasmBuffer: () => WasmBuffer,
    WasmRuntime: () => WasmRuntime,
    analysis: () => analysis,
    blas: () => blas,
    decomp: () => decomp,
    default: () => ndwasm_default,
    fromWasm: () => fromWasm,
    signal: () => signal
  });

  // src/ndwasm_blas.js
  var NDWasmBlas = {
    /**
     * Calculates the trace of a 2D square matrix (sum of diagonal elements).
     * Complexity: O(n)
     * @param {NDArray} a
     * @returns {number} The sum of the diagonal elements.
     * @throws {Error} If the array is not 2D or not a square matrix.
     */
    trace(a) {
      if (a.ndim !== 2) {
        throw new Error(`Trace is only defined for 2D matrices, but this array is ${a.ndim}D.`);
      }
      const rows = a.shape[0];
      const cols = a.shape[1];
      if (rows !== cols) {
        throw new Error(`Trace is only defined for square matrices, but this matrix is ${rows}x${cols}.`);
      }
      let sum = 0;
      const data = a.data;
      const offset = a.offset;
      const s0 = a.strides[0];
      const s1 = a.strides[1];
      const step = s0 + s1;
      for (let i = 0; i < rows; i++) {
        sum += data[offset + i * step];
      }
      return sum;
    },
    /**
     * General Matrix Multiplication (GEMM): C = A * B.
     * Complexity: O(m * n * k)
     * @memberof NDWasmBlas
     * @param {NDArray} a - Left matrix of shape [m, n].
     * @param {NDArray} b - Right matrix of shape [n, k].
     * @returns {NDArray} Result matrix of shape [m, k].
     */
    matmul(a, b) {
      if (a.shape[1] !== b.shape[0]) {
        throw new Error(`Matrix inner dimensions must match: ${a.shape[1]} != ${b.shape[0]}`);
      }
      if (b.ndim !== 2 && b.ndim !== 1) {
        throw new Error(`Right operand must be a 2D matrix (or 1D vector), but got ${b.ndim}D.`);
      }
      const m = a.shape[0];
      const n = a.shape[1];
      const k = b.ndim === 2 ? b.shape[1] : 1;
      const outShape = [m, k];
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      return NDWasm._compute([a, b], outShape, a.dtype, (aPtr, bPtr, outPtr) => {
        return NDWasm.runtime.exports[`MatMul${suffix}`](aPtr, bPtr, outPtr, m, n, k);
      });
    },
    /**
     * matPow computes A^k (Matrix Power).
     * Matrix Functions (O(n^3))
     * @memberof NDWasmBlas
     * @param {NDArray} a - Matrix of shape [n, n].
     * @returns {NDArray} Result matrix of shape [n, n].
     */
    matPow(a, k) {
      if (a.shape[0] !== a.shape[1]) {
        throw new Error(`Matrix must be square: ${a.shape[0]} != ${a.shape[1]}`);
      }
      const n = a.shape[0];
      const outShape = [n, n];
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      return NDWasm._compute([a], outShape, a.dtype, (aPtr, outPtr) => {
        return NDWasm.runtime.exports[`MatrixPower${suffix}`](aPtr, outPtr, n, k);
      });
    },
    /**
     * Batched Matrix Multiplication: C[i] = A[i] * B[i].
     * Common in deep learning inference.
     * Complexity: O(batch * m * n * k)
     * @memberof NDWasmBlas
     * @param {NDArray} a - Batch of matrices of shape [batch, m, n].
     * @param {NDArray} b - Batch of matrices of shape [batch, n, k].
     * @returns {NDArray} Result batch of shape [batch, m, k].
     */
    matmulBatch(a, b) {
      if (a.ndim !== 3 || b.ndim !== 3 || a.shape[0] !== b.shape[0]) {
        throw new Error("Input must be 3D batches with same batch size.");
      }
      if (a.shape[2] !== b.shape[1]) {
        throw new Error("Batch matrix inner dimensions must match.");
      }
      const batch = a.shape[0];
      const m = a.shape[1];
      const n = a.shape[2];
      const k = b.shape[2];
      const outShape = [batch, m, k];
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      return NDWasm._compute([a, b], outShape, a.dtype, (aPtr, bPtr, outPtr) => {
        return NDWasm.runtime.exports[`MatMulBatch${suffix}`](aPtr, bPtr, outPtr, batch, m, n, k);
      });
    },
    /**
     * Symmetric Rank-K Update: C = alpha * A * A^T + beta * C.
     * Used for efficiently computing covariance matrices or Gram matrices.
     * Complexity: O(n^2 * k)
     * @memberof NDWasmBlas
     * @param {NDArray} a - Input matrix of shape [n, k].
     * @returns {NDArray} Symmetric result matrix of shape [n, n].
     */
    syrk(a) {
      const n = a.shape[0];
      const k = a.shape[1];
      const outShape = [n, n];
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      return NDWasm._compute([a], outShape, a.dtype, (aPtr, outPtr) => {
        return NDWasm.runtime.exports[`Syrk${suffix}`](aPtr, outPtr, n, k);
      });
    },
    /**
     * Triangular System Solver: Solves A * X = B for X, where A is a triangular matrix.
     * Complexity: O(m^2 * n)
     * @memberof NDWasmBlas
     * @param {NDArray} a - Triangular matrix of shape [m, m].
     * @param {NDArray} b - Right-hand side matrix/vector of shape [m, n].
     * @returns {NDArray} Solution matrix X of shape [m, n].
     */
    trsm(a, b, lower = false) {
      if (a.shape[0] !== a.shape[1] || a.shape[0] !== b.shape[0]) {
        throw new Error("Dimension mismatch for triangular solver.");
      }
      const m = a.shape[0];
      const n = b.ndim === 1 ? 1 : b.shape[1];
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      const wa = a.toWasm(NDWasm.runtime);
      const wb = b.toWasm(NDWasm.runtime);
      try {
        NDWasm.runtime.exports[`Trsm${suffix}`](wa.ptr, wb.ptr, m, n, lower ? 1 : 0);
        return fromWasm(wb, b.shape, b.dtype);
      } finally {
        wa.dispose();
        wb.dispose();
      }
    },
    /**
     * Matrix-Vector Multiplication: y = A * x.
     * Complexity: O(m * n)
     * @memberof NDWasmBlas
     * @param {NDArray} a - Matrix of shape [m, n].
     * @param {NDArray} x - Vector of shape [n].
     * @returns {NDArray} Result vector of shape [m].
     */
    matVecMul(a, x) {
      if (a.shape[1] !== x.size) {
        throw new Error("Matrix-Vector dimension mismatch.");
      }
      const m = a.shape[0];
      const n = a.shape[1];
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      return NDWasm._compute([a, x], [m], a.dtype, (aPtr, xPtr, outPtr) => {
        return NDWasm.runtime.exports[`MatVecMul${suffix}`](aPtr, xPtr, outPtr, m, n);
      });
    },
    /**
     * Vector Outer Product (Rank-1 Update): A = x * y^T.
     * Complexity: O(m * n)
     * @memberof NDWasmBlas
     * @param {NDArray} x - Vector of shape [m].
     * @param {NDArray} y - Vector of shape [n].
     * @returns {NDArray} Result matrix of shape [m, n].
     */
    ger(x, y) {
      const m = x.size;
      const n = y.size;
      const outShape = [m, n];
      const suffix = NDWasm.runtime._getSuffix(x.dtype);
      return NDWasm._compute([x, y], outShape, x.dtype, (xPtr, yPtr, outPtr) => {
        return NDWasm.runtime.exports[`Ger${suffix}`](xPtr, yPtr, outPtr, m, n);
      });
    }
  };

  // src/ndwasm_decomp.js
  var NDWasmDecomp = {
    /**
     * Solves a system of linear equations: Ax = B for x.
     * Complexity: O(n^3)
     * @memberof NDWasmDecomp
     * @param {NDArray} a - Square coefficient matrix of shape [n, n].
     * @param {NDArray} b - Right-hand side matrix or vector of shape [n, k].
     * @returns {NDArray} Solution matrix x of shape [n, k].
     */
    solve(a, b) {
      if (a.shape[0] !== a.shape[1] || a.shape[0] !== b.shape[0]) {
        throw new Error("Dimension mismatch for linear solver: A must be square and match B's rows.");
      }
      const n = a.shape[0];
      const k = b.ndim === 1 ? 1 : b.shape[1];
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      return NDWasm._compute([a, b], [n, k], a.dtype, (aPtr, bPtr, outPtr) => {
        return NDWasm.runtime.exports[`SolveLinear${suffix}`](aPtr, bPtr, outPtr, n, k);
      });
    },
    /**
     * Computes the multiplicative inverse of a square matrix.
     * Complexity: O(n^3)
     * @memberof NDWasmDecomp
     * @param {NDArray} a - Square matrix to invert of shape [n, n].
     * @returns {NDArray} The inverted matrix of shape [n, n].
     */
    inv(a) {
      if (a.shape[0] !== a.shape[1]) throw new Error("Matrix must be square to invert.");
      const n = a.shape[0];
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      return NDWasm._compute([a], a.shape, a.dtype, (aPtr, outPtr) => {
        return NDWasm.runtime.exports[`Invert${suffix}`](aPtr, outPtr, n);
      });
    },
    /**
     * Computes the Singular Value Decomposition (SVD): A = U * S * V^T.
     * Complexity: O(m * n * min(m, n))
     * @memberof NDWasmDecomp
     * @param {NDArray} a - Input matrix of shape [m, n].
     * @returns {{u: NDArray, s: NDArray, v: NDArray}}
     */
    svd(a) {
      const [m, n] = a.shape;
      const k = Math.min(m, n);
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      const wa = a.toWasm(NDWasm.runtime);
      const ws = NDWasm.runtime.createBuffer(k, a.dtype);
      const wu = NDWasm.runtime.createBuffer(m * m, a.dtype);
      const wv = NDWasm.runtime.createBuffer(n * n, a.dtype);
      try {
        const status = NDWasm.runtime.exports[`SVD${suffix}`](wa.ptr, m, n, wu.ptr, ws.ptr, wv.ptr);
        if (status !== 0) throw new Error("SVD computation failed.");
        return {
          u: fromWasm(wu, [m, m], a.dtype),
          s: fromWasm(ws, [k], a.dtype),
          v: fromWasm(wv, [n, n], a.dtype)
        };
      } finally {
        [wa, ws, wu, wv].forEach((b) => b.dispose());
      }
    },
    /**
     * Computes the QR decomposition: A = Q * R.
     * Complexity: O(n^3)
     * @memberof NDWasmDecomp
     * @param {NDArray} a - Input matrix of shape [m, n].
     * @returns {{q: NDArray, r: NDArray}}
     */
    qr(a) {
      const [m, n] = a.shape;
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      const wa = a.toWasm(NDWasm.runtime);
      const wq = NDWasm.runtime.createBuffer(m * m, a.dtype);
      const wr = NDWasm.runtime.createBuffer(m * n, a.dtype);
      try {
        NDWasm.runtime.exports[`QR${suffix}`](wa.ptr, m, n, wq.ptr, wr.ptr);
        return {
          q: fromWasm(wq, [m, m], a.dtype),
          r: fromWasm(wr, [m, n], a.dtype)
        };
      } finally {
        [wa, wq, wr].forEach((b) => b.dispose());
      }
    },
    /**
     * Computes the Cholesky decomposition of a symmetric, positive-definite matrix: A = L * L^T.
     * Complexity: O(n^3)
     * @memberof NDWasmDecomp
     * @param {NDArray} a - Symmetric positive-definite matrix of shape [n, n].
     * @returns {NDArray} Lower triangular matrix L of shape [n, n].
     */
    cholesky(a) {
      const n = a.shape[0];
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      return NDWasm._compute([a], [n, n], a.dtype, (aPtr, outPtr) => {
        return NDWasm.runtime.exports[`Cholesky${suffix}`](aPtr, outPtr, n);
      });
    },
    /**
     * Computes the LU decomposition of a matrix: A = P * L * U.
     * The result is stored in-place in the output matrix.
     * @memberof NDWasmDecomp
     * @param {NDArray} a - Input matrix of shape [m, n].
     * @returns {NDArray} LU matrix of shape [m, n].
     */
    lu(a) {
      const [m, n] = a.shape;
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      const wa = a.toWasm(NDWasm.runtime);
      try {
        NDWasm.runtime.exports[`LU${suffix}`](wa.ptr, m, n);
        return fromWasm(wa, a.shape, a.dtype);
      } finally {
        wa.dispose();
      }
    },
    /**
     * Computes the Moore-Penrose pseudo-inverse of a matrix using SVD.
     * Complexity: O(n^3)
     * @memberof NDWasmDecomp
     * @param {NDArray} a - Input matrix of shape [m, n].
     * @returns {NDArray} Pseudo-inverted matrix of shape [n, m].
     */
    pinv(a) {
      const [m, n] = a.shape;
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      return NDWasm._compute([a], [n, m], a.dtype, (aPtr, outPtr) => {
        return NDWasm.runtime.exports[`PInverse${suffix}`](aPtr, outPtr, m, n);
      });
    },
    /**
     * Computes the determinant of a square matrix.
     * Complexity: O(n^3)
     * @memberof NDWasmDecomp
     * @param {NDArray} a - Square matrix of shape [n, n].
     * @returns {number} The determinant.
     */
    det(a) {
      if (a.shape[0] !== a.shape[1]) throw new Error("Matrix must be square.");
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      const wa = a.toWasm(NDWasm.runtime);
      try {
        return NDWasm.runtime.exports[`Det${suffix}`](wa.ptr, a.shape[0]);
      } finally {
        wa.dispose();
      }
    },
    /**
     * Computes the log-determinant for improved numerical stability.
     * Complexity: O(n^3)
     * @memberof NDWasmDecomp
     * @param {NDArray} a - Square matrix of shape [n, n].
     * @returns {{sign: number, logAbsDet: number}}
     */
    logDet(a) {
      const n = a.shape[0];
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      const wa = a.toWasm(NDWasm.runtime);
      const wDetSign = NDWasm.runtime.createBuffer(2, a.dtype);
      try {
        NDWasm.runtime.exports[`LogDet${suffix}`](wa.ptr, n, wDetSign.ptr);
        return {
          sign: wDetSign.view[1],
          logAbsDet: wDetSign.view[0]
        };
      } finally {
        [wa, wDetSign].forEach((b) => b.dispose());
      }
    },
    /**
     * Computes the eigenvalues and eigenvectors of a general square matrix.
     * Eigenvalues and eigenvectors can be complex numbers.
     * The results are returned in an interleaved format where each complex number (a + bi)
     * is represented by two consecutive float64 values (a, b).
     *
     * @param {NDArray} a - Input square matrix of shape `[n, n]`. Must be float64.
     * @returns {{values: NDArray, vectors: NDArray}} An object containing:
     *   - `values`: Complex eigenvalues as an NDArray of shape `[n, 2]`, where `[i, 0]` is real and `[i, 1]` is imaginary.
     *   - `vectors`: Complex right eigenvectors as an NDArray of shape `[n, n, 2]`, where `[i, j, 0]` is real and `[i, j, 1]` is imaginary.
     *              (Note: these are column vectors, such that `A * v = lambda * v`).
     * @throws {Error} If WASM runtime is not loaded, input is not a square matrix, or input dtype is not float64.
     * @memberof NDWasmDecomp
     */
    eigen(a) {
      if (!NDWasm.runtime?.isLoaded) throw new Error("WasmRuntime not loaded.");
      if (a.shape[0] !== a.shape[1]) throw new Error("Matrix must be square for eigen decomposition.");
      if (a.dtype !== "float64") {
        throw new Error("Eigen decomposition currently only supports 'float64' input dtype.");
      }
      const n = a.shape[0];
      let wa, weigvals, weigvecs;
      try {
        wa = a.toWasm(NDWasm.runtime);
        weigvals = NDWasm.runtime.createBuffer(n * 2, "float64");
        weigvecs = NDWasm.runtime.createBuffer(n * n * 2, "float64");
        NDWasm.runtime.exports.Eigen_F64(wa.ptr, n, weigvals.ptr, weigvecs.ptr);
        return {
          values: fromWasm(weigvals, [n, 2], "float64"),
          vectors: fromWasm(weigvecs, [n, n, 2], "float64")
        };
      } finally {
        [wa, weigvals, weigvecs].forEach((b) => b?.dispose());
      }
    }
  };

  // src/ndwasm_signal.js
  var NDWasmSignal = {
    /**
     * 1D Complex-to-Complex Fast Fourier Transform.
     * The input array must have its last dimension of size 2 (real and imaginary parts).
     * The transform is performed in-place.
     * Complexity: O(n log n)
     * @memberof NDWasmSignal
     * @param {NDArray} a - Complex input signal, with shape [..., 2].
     * @returns {NDArray} - Complex result, with the same shape as input.
     */
    fft(a) {
      if (a.ndim !== 2 || a.shape[1] !== 2) {
        throw new Error("Input to fft must be a 1D complex array with shape [n, 2].");
      }
      const n = a.size / 2;
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      const wComplex = a.toWasm(NDWasm.runtime);
      try {
        NDWasm.runtime.exports[`FFT1D${suffix}`](wComplex.ptr, n);
        return fromWasm(wComplex, a.shape, a.dtype);
      } finally {
        wComplex.dispose();
      }
    },
    /**
     * 1D Inverse Complex-to-Complex Fast Fourier Transform.
     * The input array must have its last dimension of size 2 (real and imaginary parts).
     * The transform is performed in-place.
     * Complexity: O(n log n)
     * @memberof NDWasmSignal
     * @param {NDArray} a - Complex frequency-domain signal, with shape [..., 2].
     * @returns {NDArray} - Complex time-domain result, with the same shape as input.
     */
    ifft(a) {
      if (a.ndim !== 2 || a.shape[1] !== 2) {
        throw new Error("Input to ifft must be a 1D complex array with shape [n, 2].");
      }
      const n = a.size / 2;
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      const wComplex = a.toWasm(NDWasm.runtime);
      try {
        NDWasm.runtime.exports[`IFFT1D${suffix}`](wComplex.ptr, n);
        return fromWasm(wComplex, a.shape, a.dtype);
      } finally {
        wComplex.dispose();
      }
    },
    /**
     * 1D Real-to-Complex Fast Fourier Transform (Optimized for real input).
     * The output is a complex array with shape [n/2 + 1, 2].
     * Complexity: O(n log n)
     * @memberof NDWasmSignal
     * @param {NDArray} a - Real input signal.
     * @returns {NDArray} - Complex result of shape [n/2 + 1, 2].
     */
    rfft(a) {
      if (a.ndim !== 1) {
        throw new Error("Input to rfft must be a 1D real array.");
      }
      const n = a.size;
      const outLen = Math.floor(n / 2) + 1;
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      const wa = a.toWasm(NDWasm.runtime);
      const wOut = NDWasm.runtime.createBuffer(outLen * 2, a.dtype);
      try {
        NDWasm.runtime.exports[`RFFT1D${suffix}`](wa.ptr, wOut.ptr, n);
        return fromWasm(wOut, [outLen, 2], a.dtype);
      } finally {
        wa.dispose();
        wOut.dispose();
      }
    },
    /**
     * 1D Complex-to-Real Inverse Fast Fourier Transform.
     * The input must be a complex array of shape [k, 2], where k is n/2 + 1.
     * @memberof NDWasmSignal
     * @param {NDArray} a - Complex frequency signal of shape [n/2 + 1, 2].
     * @param {number} n - Length of the original real signal.
     * @returns {NDArray} Real-valued time domain signal.
     */
    rifft(a, n) {
      if (a.ndim !== 2 || a.shape[1] !== 2) {
        throw new Error("Input to rifft must be a complex array with shape [k, 2].");
      }
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      const wa = a.toWasm(NDWasm.runtime);
      const wo = NDWasm.runtime.createBuffer(n, a.dtype);
      try {
        NDWasm.runtime.exports[`RIFFT1D${suffix}`](wa.ptr, wo.ptr, n);
        return fromWasm(wo, [n], a.dtype);
      } finally {
        wa.dispose();
        wo.dispose();
      }
    },
    /**
     * 2D Complex-to-Complex Fast Fourier Transform.
     * The input array must be 3D with shape [rows, cols, 2].
     * The transform is performed in-place.
     * Complexity: O(rows * cols * log(rows * cols))
     * @memberof NDWasmSignal
     * @param {NDArray} a - 2D Complex input signal, with shape [rows, cols, 2].
     * @returns {NDArray} - 2D Complex result, with the same shape as input.
     */
    fft2(a) {
      if (a.ndim !== 3 || a.shape[2] !== 2) {
        throw new Error("fft2 requires a 3D array with shape [rows, cols, 2].");
      }
      const [rows, cols] = a.shape;
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      const wComplex = a.toWasm(NDWasm.runtime);
      try {
        NDWasm.runtime.exports[`FFT2D${suffix}`](wComplex.ptr, rows, cols);
        return fromWasm(wComplex, a.shape, a.dtype);
      } finally {
        wComplex.dispose();
      }
    },
    /**
     * 2D Inverse Complex-to-Complex Fast Fourier Transform.
     * The input array must be 3D with shape [rows, cols, 2].
     * The transform is performed in-place.
     * @memberof NDWasmSignal
     * @param {NDArray} a - 2D Complex frequency-domain signal, with shape [rows, cols, 2].
     * @returns {NDArray} - 2D Complex time-domain result, with the same shape as input.
     */
    ifft2(a) {
      if (a.ndim !== 3 || a.shape[2] !== 2) {
        throw new Error("ifft2 requires a 3D array with shape [rows, cols, 2].");
      }
      const [rows, cols] = a.shape;
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      const wComplex = a.toWasm(NDWasm.runtime);
      try {
        NDWasm.runtime.exports[`IFFT2D${suffix}`](wComplex.ptr, rows, cols);
        return fromWasm(wComplex, a.shape, a.dtype);
      } finally {
        wComplex.dispose();
      }
    },
    /**
     * 1D Discrete Cosine Transform (Type II).
     * Complexity: O(n log n)
     * @memberof NDWasmSignal
     * @param {NDArray} a - Input signal.
     * @returns {NDArray} DCT result of same shape.
     */
    dct(a) {
      if (a.size < 2) {
        return a.copy();
      }
      const n = a.size;
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      return NDWasm._compute([a], a.shape, a.dtype, (aPtr, outPtr) => {
        return NDWasm.runtime.exports[`DCT${suffix}`](aPtr, outPtr, n);
      });
    },
    /**
     * 2D Spatial Convolution.
     * Complexity: O(img_h * img_w * kernel_h * kernel_w)
     * @memberof NDWasmSignal
     * @param {NDArray} img - 2D Image/Matrix.
     * @param {NDArray} kernel - 2D Filter kernel.
     * @param {number} stride - Step size (default 1).
     * @param {number} padding - Zero-padding size (default 0).
     * @returns {NDArray} Convolved result.
     */
    conv2d(img, kernel, stride = 1, padding = 0) {
      if (img.ndim !== 2 || kernel.ndim !== 2) throw new Error("Inputs must be 2D.");
      const [h, w] = img.shape;
      const [kh, kw] = kernel.shape;
      const oh = Math.floor((h - kh + 2 * padding) / stride) + 1;
      const ow = Math.floor((w - kw + 2 * padding) / stride) + 1;
      const suffix = NDWasm.runtime._getSuffix(img.dtype);
      return NDWasm._compute([img, kernel], [oh, ow], img.dtype, (iPtr, kPtr, outPtr) => {
        return NDWasm.runtime.exports[`Conv2D${suffix}`](iPtr, kPtr, outPtr, h, w, kh, kw, stride, padding);
      });
    },
    /**
     * 2D Spatial Cross-Correlation.
     * Similar to convolution but without flipping the kernel.
     * Complexity: O(img_h * img_w * kernel_h * kernel_w)
     * @memberof NDWasmSignal
     * @param {NDArray} img - 2D Image/Matrix.
     * @param {NDArray} kernel - 2D Filter kernel.
     * @param {number} stride - Step size.
     * @param {number} padding - Zero-padding size.
     * @returns {NDArray} Cross-correlated result.
     */
    correlate2d(img, kernel, stride = 1, padding = 0) {
      const [h, w] = img.shape;
      const [kh, kw] = kernel.shape;
      const oh = Math.floor((h - kh + 2 * padding) / stride) + 1;
      const ow = Math.floor((w - kw + 2 * padding) / stride) + 1;
      const suffix = NDWasm.runtime._getSuffix(img.dtype);
      return NDWasm._compute([img, kernel], [oh, ow], img.dtype, (iPtr, kPtr, outPtr) => {
        return NDWasm.runtime.exports[`CrossCorrelate2D${suffix}`](iPtr, kPtr, outPtr, h, w, kh, kw, stride, padding);
      });
    }
  };

  // src/ndwasm_analysis.js
  var NDWasmAnalysis = {
    // --- 1. Sorting & Searching (O(n log n)) ---
    /**
     * Returns the indices that would sort an array.
     * @memberof NDWasmAnalysis
     * @param {NDArray} a - Input array.
     * @returns {NDArray} Indices as Int32 NDArray.
     */
    argsort(a) {
      const wa = a.toWasm(NDWasm.runtime);
      const wi = NDWasm.runtime.createBuffer(a.size, "int32");
      try {
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        NDWasm.runtime.exports[`ArgSort${suffix}`](wa.ptr, wi.ptr, a.size);
        return fromWasm(wi, a.shape, "int32");
      } finally {
        wa.dispose();
        wi.dispose();
      }
    },
    /**
     * Finds the largest or smallest K elements and their indices.
     * Complexity: O(n log k)
     * @memberof NDWasmAnalysis
     * @param {NDArray} a - Input array.
     * @param {number} k - Number of elements to return.
     * @param {boolean} largest - If true, find max elements; else min.
     * @returns {{values: NDArray, indices: NDArray}}
     */
    topk(a, k, largest = true) {
      const wa = a.toWasm(NDWasm.runtime);
      const wv = NDWasm.runtime.createBuffer(k, a.dtype);
      const wi = NDWasm.runtime.createBuffer(k, "int32");
      try {
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        NDWasm.runtime.exports[`TopK${suffix}`](wa.ptr, wv.ptr, wi.ptr, a.size, k, largest ? 1 : 0);
        return {
          values: fromWasm(wv, [k], a.dtype),
          indices: fromWasm(wi, [k], "int32")
        };
      } finally {
        [wa, wv, wi].forEach((b) => b.dispose());
      }
    },
    // --- 2. Statistics & Matrix Properties (O(n^2) to O(n^3)) ---
    /**
     * Computes the covariance matrix for a dataset of shape [n_samples, n_features].
     * @memberof NDWasmAnalysis
     * @param {NDArray} a - Data matrix.
     * @returns {NDArray} Covariance matrix of shape [d, d].
     */
    cov(a) {
      const [n, d] = a.shape;
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      return NDWasm._compute([a], [d, d], a.dtype, (aP, oP) => {
        return NDWasm.runtime.exports[`Covariance${suffix}`](aP, oP, n, d);
      });
    },
    /**
     * Computes the Pearson correlation matrix for a dataset of shape [n_samples, n_features].
     * @memberof NDWasmAnalysis
     * @param {NDArray} a - Data matrix.
     * @returns {NDArray} Correlation matrix of shape [d, d].
     */
    corr(a) {
      const [n, d] = a.shape;
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      return NDWasm._compute([a], [d, d], a.dtype, (aP, oP) => {
        return NDWasm.runtime.exports[`Correlation${suffix}`](aP, oP, n, d);
      });
    },
    /**
     * Computes the matrix norm.
     * @memberof NDWasmAnalysis
     * @param {NDArray} a - Input matrix.
     * @param {number} type - 1 (The maximum absolute column sum), 2 (Frobenius), Infinity (The maximum absolute row sum)
     * @returns {number} The norm value.
     */
    norm(a, type = 2) {
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      const wa = a.toWasm(NDWasm.runtime);
      try {
        return NDWasm.runtime.exports[`MatrixNorm${suffix}`](wa.ptr, a.shape[0], a.ndim == 1 ? 1 : a.shape[1], type);
      } finally {
        wa.dispose();
      }
    },
    /**
     * Computes the rank of a matrix using SVD.
     * @memberof NDWasmAnalysis
     * @param {NDArray} a - Input matrix.
     * @param {number} tol - Tolerance for singular values (0 for 1e-14).
     * @returns {number} Integer rank of the matrix.
     */
    rank(a, tol = 0) {
      const wa = a.toWasm(NDWasm.runtime);
      try {
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        return NDWasm.runtime.exports[`Rank${suffix}`](wa.ptr, a.shape[0], a.shape[1], tol);
      } finally {
        wa.dispose();
      }
    },
    /**
     * estimates the reciprocal condition number of matrix a.
     * @memberof NDWasmAnalysis
     * @param {NDArray} a - Input matrix.
     * @param {number} norm - norm: 1 (1-norm) or Infinity (Infinity norm).
     * @returns {number} result.
    */
    cond(a, norm = 1) {
      const wa = a.toWasm(NDWasm.runtime);
      try {
        const suffix = NDWasm.runtime._getSuffix(a.dtype);
        return NDWasm.runtime.exports[`Cond${suffix}`](wa.ptr, a.shape[0], norm);
      } finally {
        wa.dispose();
      }
    },
    /**
     * Eigenvalue decomposition for symmetric matrices.
     * @memberof NDWasmAnalysis
     * @param {NDArray} a - Symmetric square matrix.
     * @param {boolean} computeVectors - Whether to return eigenvectors.
     * @returns {{values: NDArray, vectors: NDArray|null}}
     */
    eigenSym(a, computeVectors = true) {
      const n = a.shape[0];
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      const wa = a.toWasm(NDWasm.runtime);
      const wv = NDWasm.runtime.createBuffer(n, a.dtype);
      const we = computeVectors ? NDWasm.runtime.createBuffer(n * n, a.dtype) : { ptr: 0, dispose: () => {
      } };
      try {
        NDWasm.runtime.exports[`EigenSym${suffix}`](wa.ptr, n, wv.ptr, we.ptr);
        return {
          values: fromWasm(wv, [n], a.dtype),
          vectors: computeVectors ? fromWasm(we, [n, n], a.dtype) : null
        };
      } finally {
        [wa, wv, we].forEach((b) => b.dispose());
      }
    },
    // --- 3. Spatial & Iterative (O(m*n*d)) ---
    /**
     * Computes pairwise Euclidean distances between two sets of vectors.
     * @memberof NDWasmAnalysis
     * @param {NDArray} a - Matrix of shape [m, d].
     * @param {NDArray} b - Matrix of shape [n, d].
     * @returns {NDArray} Distance matrix of shape [m, n].
     */
    pairwiseDist(a, b) {
      const [m, d] = a.shape;
      const n = b.shape[0];
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      return NDWasm._compute([a, b], [m, n], a.dtype, (aP, bP, oP) => {
        return NDWasm.runtime.exports[`PairwiseDist${suffix}`](aP, bP, oP, m, n, d);
      });
    },
    /**
     * Performs K-Means clustering in WASM memory.
     * @memberof NDWasmAnalysis
     * @param {NDArray} data - Data of shape [n_samples, d_features].
     * @param {number} k - Number of clusters.
     * @param {number} maxIter - Maximum iterations.
     * @returns {{centroids: NDArray, labels: NDArray, iterations: number}}
     */
    kmeans(data, k, maxIter = 100) {
      const [n, d] = data.shape;
      const suffix = NDWasm.runtime._getSuffix(data.dtype);
      const wd = data.toWasm(NDWasm.runtime);
      const wc = NDWasm.runtime.createBuffer(k * d, data.dtype);
      const wl = NDWasm.runtime.createBuffer(n, "int32");
      try {
        const actualIters = NDWasm.runtime.exports[`KMeans${suffix}`](wd.ptr, wc.ptr, wl.ptr, n, d, k, maxIter);
        return {
          centroids: fromWasm(wc, [k, d], data.dtype),
          labels: fromWasm(wl, [n], "int32"),
          iterations: actualIters
        };
      } finally {
        [wd, wc, wl].forEach((b) => b.dispose());
      }
    },
    // --- 4. Advanced Structural (O(n^2 * m^2)) ---
    /**
     * Computes the Kronecker product C = A ⊗ B.
     * @memberof NDWasmAnalysis
     */
    kronecker(a, b) {
      const outShape = [a.shape[0] * b.shape[0], a.shape[1] * b.shape[1]];
      const suffix = NDWasm.runtime._getSuffix(a.dtype);
      return NDWasm._compute([a, b], outShape, a.dtype, (aP, bP, oP) => {
        return NDWasm.runtime.exports[`Kronecker${suffix}`](aP, bP, oP, a.shape[0], a.shape[1], b.shape[0], b.shape[1]);
      });
    }
  };

  // src/ndwasm.js
  var WasmBuffer = class {
    /**
     * @param {Object} exports - The exports object from Go WASM.
     * @param {number} size - Number of elements.
     * @param {string} dtype - Data type (float64, float32, etc.).
     */
    constructor(exports, size, dtype) {
      this.exports = exports;
      this.size = size;
      this.dtype = dtype;
      const Constructor = DTYPE_MAP[dtype];
      if (!Constructor) throw new Error(`Unsupported dtype: ${dtype}`);
      this.byteLength = size * Constructor.BYTES_PER_ELEMENT;
      if (this.byteLength === 0) {
        this.ptr = 0;
      } else {
        this.ptr = this.exports.malloc(this.byteLength);
        if (!this.ptr) throw new Error(`WASM malloc failed to allocate ${this.byteLength} bytes`);
      }
      this.refresh();
    }
    /**
     * Refreshes the view into WASM memory.
     */
    refresh() {
      const Constructor = DTYPE_MAP[this.dtype];
      this.view = new Constructor(this.exports.mem.buffer, this.ptr, this.size);
      return this;
    }
    /** Synchronizes JS data into the WASM buffer. */
    push(typedArray) {
      this.refresh();
      this.view.set(typedArray);
    }
    /** Pulls data from the WASM buffer back to JS (returns a copy).
     * @returns {NDArray}
     */
    pull() {
      this.refresh();
      return this.view.slice();
    }
    /** Disposes of the temporary buffer. */
    dispose() {
      if (this.ptr) {
        this.exports.free(this.ptr);
        this.ptr = null;
        this.view = null;
      }
    }
  };
  var WasmRuntime = class {
    constructor() {
      this.instance = null;
      this.exports = null;
      this.isLoaded = false;
    }
    /**
     * Initializes the Go WASM environment.
     * @param {object} [options] - Optional configuration.
     * @param {string} [options.wasmUrl='./ndarray_plugin.wasm'] - Path or URL to the wasm file.
     * @param {string} [options.execUrl='./wasm_exec.js'] - Path to the wasm_exec.js file (Node.js only).
     * @param {number} [options.initialMemoryPages] - Initial memory size in 64KiB pages.
     * @param {number} [options.maximumMemoryPages] - Maximum memory size in 64KiB pages.
     * @param {string} [options.baseDir='.']
     */
    async init(options = {}) {
      options = {
        initialMemoryPages: 1024 * 1024 * 64 / (64 * 1024),
        maximumMemoryPages: 1024 * 1024 * 512 / (64 * 1024),
        ...options
      };
      const wasmUrl = options.wasmUrl || `${options.baseDir || "."}/ndarray_plugin.wasm`;
      const execUrl = options.execUrl || `${options.baseDir || "."}/wasm_exec.js`;
      const wasmMemory = new WebAssembly.Memory({
        initial: options.initialMemoryPages,
        maximum: options.maximumMemoryPages
      });
      const isNode = typeof window === "undefined";
      if (isNode) {
        const fs = __require("fs");
        const path = __require("path");
        global.fs = fs;
        global.util = __require("util");
        __require(path.resolve(process.cwd(), execUrl));
        const wasmBytes = fs.readFileSync(path.resolve(process.cwd(), wasmUrl));
        const go = new Go();
        go.env["mem"] = wasmMemory;
        const { instance } = await WebAssembly.instantiate(wasmBytes, go.importObject);
        this.instance = instance;
        go.run(this.instance);
      } else {
        try {
          await importScripts(execUrl);
        } catch (e) {
          console.log(e);
        }
        if (typeof Go === "undefined") {
          console.error("Go's wasm_exec.js must be loaded with a <script> tag before initializing WasmRuntime in a browser.");
          throw new Error("Missing Go global object.");
        }
        const go = new Go();
        go.env["mem"] = wasmMemory;
        const result = await WebAssembly.instantiateStreaming(fetch(wasmUrl), go.importObject);
        this.instance = result.instance;
        go.run(this.instance);
      }
      this.exports = this.instance.exports;
      this.isLoaded = true;
    }
    /**
     * Helper method: Gets the suffix for Go export function names based on type.
     * @private
     */
    _getSuffix(dtype) {
      if (dtype === "float64") return "_F64";
      if (dtype === "float32") return "_F32";
      throw new Error(`Go-Wasm compute does not support dtype: ${dtype}. Use float64 or float32.`);
    }
    /**
     * Quickly allocates a buffer.
     * @returns {WasmBuffer}
     */
    createBuffer(size, dtype) {
      return new WasmBuffer(this.exports, size, dtype);
    }
  };
  function fromWasm(bridge, shape, dtype) {
    const data = bridge.pull();
    return new NDArray(data, { shape, dtype: dtype || bridge.dtype });
  }
  var NDWasm = {
    runtime: null,
    /** Binds a loaded WASM runtime to the bridge.
     * @param {*} runtime 
     */
    bind(runtime) {
      this.runtime = runtime;
    },
    /**
     * Init the NDWasm
     * @param {string} [baseDir='.'] 
     */
    async init(baseDir = ".") {
      const runtime = new WasmRuntime();
      await runtime.init({ baseDir });
      NDWasm.bind(runtime);
    },
    /** Internal helper: executes a computation in WASM and manages memory. */
    _compute(inputs, outShape, outDtype, computeFn) {
      if (!this.runtime || !this.runtime.isLoaded) throw new Error("WasmRuntime not initialized");
      const bridges = inputs.map((arr) => arr.toWasm(this.runtime));
      const size = outShape.reduce((a, b) => a * b, 1);
      const outBridge = this.runtime.createBuffer(size, outDtype);
      try {
        const status = computeFn(...bridges.map((b) => b.ptr), outBridge.ptr);
        if (status !== void 0 && status !== 0) {
          throw new Error(`WASM compute failed with status code: ${status}`);
        }
        return fromWasm(outBridge, outShape, outDtype);
      } finally {
        bridges.forEach((b) => b.dispose());
        outBridge.dispose();
      }
    }
  };
  var blas = NDWasmBlas;
  var decomp = NDWasmDecomp;
  var signal = NDWasmSignal;
  var analysis = NDWasmAnalysis;
  var ndwasm_default = NDWasm;

  // src/ndwasm_image.js
  var NDWasmImage = {
    /**
     * Decodes a binary image into an NDArray.
     * Supports common formats like PNG, JPEG, GIF, and WebP.
     * The resulting NDArray will have a shape of [height, width, 4] and a 'uint8c' dtype,
     * representing RGBA channels. This provides a consistent starting point for image manipulation.
     *
     * @memberof NDWasmImage
     * @param {Uint8Array} imageBytes - The raw binary data of the image file.
     * @returns {NDArray|null} A 3D NDArray representing the image, or null if decoding fails.
     * @example
     * const imageBlob = await fetch('./my-image.png').then(res => res.blob());
     * const imageBytes = new Uint8Array(await imageBlob.arrayBuffer());
     * const imageArray = NDWasmImage.decode(imageBytes);
     * // imageArray.shape is [height, width, 4]
     */
    decode(imageBytes) {
      if (!ndwasm_default.runtime || !ndwasm_default.runtime.isLoaded) {
        throw new Error("NDWasm runtime is not initialized. Call NDWasm.init() first.");
      }
      if (!(imageBytes instanceof Uint8Array)) {
        throw new Error("Input must be a Uint8Array.");
      }
      const exports = ndwasm_default.runtime.exports;
      let imagePtr = 0;
      let resultPtr = 0;
      let pixelDataPtr = 0;
      try {
        imagePtr = exports.malloc(imageBytes.length);
        if (!imagePtr) throw new Error("WASM malloc failed for image bytes.");
        new Uint8Array(exports.mem.buffer, imagePtr, imageBytes.length).set(imageBytes);
        resultPtr = exports.decode_image(imagePtr, imageBytes.length);
        if (!resultPtr) {
          console.error("Failed to decode image in WASM. The image format may be invalid or unsupported.");
          return null;
        }
        const resultView = new DataView(exports.mem.buffer, resultPtr, 16);
        pixelDataPtr = resultView.getUint32(0, true);
        const pixelDataSize = resultView.getUint32(4, true);
        const width = resultView.getUint32(8, true);
        const height = resultView.getUint32(12, true);
        if (!pixelDataPtr) {
          console.error("WASM decode_image returned a null pixel data pointer.");
          return null;
        }
        const pixelData = new Uint8ClampedArray(exports.mem.buffer, pixelDataPtr, pixelDataSize).slice();
        return new NDArray(pixelData, { shape: [height, width, 4], dtype: "uint8c" });
      } finally {
        if (imagePtr) exports.free(imagePtr, imageBytes.length);
        if (resultPtr) exports.free(resultPtr, 16);
        if (pixelDataPtr) {
          const resultView = new DataView(exports.mem.buffer, resultPtr, 16);
          const pixelDataSize = resultView.getUint32(4, true);
          exports.free(pixelDataPtr, pixelDataSize);
        }
      }
    },
    /**
     * Encodes an NDArray into a binary image format (PNG or JPEG).
     *
     * @memberof NDWasmImage
     * @param {NDArray} ndarray - The input array.
     *   Supported dtypes: 'uint8', 'uint8c', 'float32', 'float64'. Float values should be in the range [0, 1].
     *   Supported shapes: [h, w] (grayscale), [h, w, 1] (grayscale), [h, w, 3] (RGB), or [h, w, 4] (RGBA).
     * @param {object} [options={}] - Encoding options.
     * @param {string} [options.format='png'] - The target format: 'png' or 'jpeg'. It is recommended to use the `encodePng` or `encodeJpeg` helpers instead.
     * @param {number} [options.quality=90] - The quality for JPEG encoding (1-100). Ignored for PNG.
     * @returns {Uint8Array|null} A Uint8Array containing the binary data of the encoded image, or null on failure.
     * @see {@link NDWasmImage.encodePng}
     * @see {@link NDWasmImage.encodeJpeg}
     * @example
     * // Encode a 3-channel float array to a high-quality JPEG using the main function
     * const floatArr = NDArray.random([100, 150, 3]);
     * const jpegBytes = NDWasmImage.encode(floatArr, { format: 'jpeg', quality: 95 });
     */
    encode(ndarray, { format = "png", quality = 90 } = {}) {
      if (!ndwasm_default.runtime || !ndwasm_default.runtime.isLoaded) {
        throw new Error("NDWasm runtime is not initialized. Call NDWasm.init() first.");
      }
      if (!(ndarray instanceof NDArray)) {
        throw new Error("Input must be an NDArray.");
      }
      let channels;
      if (ndarray.ndim === 2) {
        channels = 1;
      } else if (ndarray.ndim === 3) {
        channels = ndarray.shape[2];
        if (channels !== 1 && channels !== 3 && channels !== 4) {
          throw new Error(`Unsupported channel count for 3D array: ${channels}. Must be 1, 3, or 4.`);
        }
      } else {
        throw new Error(`Unsupported array dimensions: ${ndarray.ndim}. Must be 2 or 3.`);
      }
      let arrayToEncode = ndarray;
      if (ndarray.dtype === "float32" || ndarray.dtype === "float64") {
        const scaledData = new Uint8Array(ndarray.size);
        let i = 0;
        ndarray.iterate((value) => {
          scaledData[i++] = Math.max(0, Math.min(255, value * 255));
        });
        arrayToEncode = new NDArray(scaledData, { shape: ndarray.shape, dtype: "uint8" });
      }
      let sourceData = arrayToEncode.data;
      if (!arrayToEncode.isContiguous) {
        const flatData = new DTYPE_MAP[arrayToEncode.dtype](arrayToEncode.size);
        let i = 0;
        arrayToEncode.iterate((v) => flatData[i++] = v);
        sourceData = flatData;
      }
      const exports = ndwasm_default.runtime.exports;
      let pixelDataPtr = 0;
      let formatPtr = 0;
      let resultPtr = 0;
      let encodedDataPtr = 0;
      const [height, width] = arrayToEncode.shape;
      const formatBytes = new TextEncoder().encode(format);
      try {
        pixelDataPtr = exports.malloc(sourceData.byteLength);
        if (!pixelDataPtr) throw new Error("WASM malloc failed for pixel data.");
        new Uint8Array(exports.mem.buffer, pixelDataPtr, sourceData.byteLength).set(sourceData);
        formatPtr = exports.malloc(formatBytes.length);
        if (!formatPtr) throw new Error("WASM malloc failed for format string.");
        new Uint8Array(exports.mem.buffer, formatPtr, formatBytes.length).set(formatBytes);
        resultPtr = exports.encode_image(pixelDataPtr, width, height, channels, quality, formatPtr, formatBytes.length);
        if (!resultPtr) {
          console.error(`Failed to encode image to ${format} in WASM.`);
          return null;
        }
        const resultView = new DataView(exports.mem.buffer, resultPtr, 8);
        encodedDataPtr = resultView.getUint32(0, true);
        const encodedDataSize = resultView.getUint32(4, true);
        if (!encodedDataPtr) {
          console.error("WASM encode_image returned a null data pointer.");
          return null;
        }
        const encodedData = new Uint8Array(exports.mem.buffer, encodedDataPtr, encodedDataSize).slice();
        return encodedData;
      } finally {
        if (pixelDataPtr) exports.free(pixelDataPtr, sourceData.byteLength);
        if (formatPtr) exports.free(formatPtr, formatBytes.length);
        if (resultPtr) exports.free(resultPtr, 8);
        if (encodedDataPtr) {
          const resultView = new DataView(exports.mem.buffer, resultPtr, 8);
          const encodedDataSize = resultView.getUint32(4, true);
          exports.free(encodedDataPtr, encodedDataSize);
        }
      }
    },
    /**
     * Encodes an NDArray into a PNG image.
     * This is a helper function that calls `encode` with `format: 'png'`.
     * @memberof NDWasmImage
     * @param {NDArray} ndarray - The input array. See `encode` for supported shapes and dtypes.
     * @returns {Uint8Array|null} A Uint8Array containing the binary data of the PNG image.
     * @example
     * const grayArr = ndarray.zeros([16, 16]);
     * const pngBytes = NDWasmImage.encodePng(grayArr);
     */
    encodePng(ndarray) {
      return this.encode(ndarray, { format: "png" });
    },
    /**
     * Encodes an NDArray into a JPEG image.
     * This is a helper function that calls `encode` with `format: 'jpeg'`.
     * @memberof NDWasmImage
     * @param {NDArray} ndarray - The input array. See `encode` for supported shapes and dtypes.
     * @param {object} [options={}] - Encoding options.
     * @param {number} [options.quality=90] - The quality for JPEG encoding (1-100).
     * @returns {Uint8Array|null} A Uint8Array containing the binary data of the JPEG image.
     * @example
     * const floatArr = NDArray.random([20, 20, 3]);
     * const jpegBytes = NDWasmImage.encodeJpeg(floatArr, { quality: 85 });
     */
    encodeJpeg(ndarray, options = {}) {
      return this.encode(ndarray, { ...options, format: "jpeg" });
    },
    /**
     * Converts a Uint8Array of binary data into a Base64 Data URL.
     * This is a utility function that runs purely in JavaScript.
     *
     * @memberof NDWasmImage
     * @param {Uint8Array} uint8array - The byte array to convert.
     * @param {string} [mimeType='image/png'] - The MIME type for the Data URL (e.g., 'image/jpeg').
     * @returns {string} The complete Data URL string.
     * @example
     * const pngBytes = NDWasmImage.encodePng(myNdarray);
     * const dataUrl = NDWasmImage.convertUint8ArrrayToDataurl(pngBytes, 'image/png');
     * // <img src={dataUrl} />
     */
    convertUint8ArrrayToDataurl(uint8array, mimeType = "image/png") {
      if (typeof window !== "undefined" && typeof window.btoa === "function") {
        let binary = "";
        const len = uint8array.byteLength;
        for (let i = 0; i < len; i++) {
          binary += String.fromCharCode(uint8array[i]);
        }
        const base64 = window.btoa(binary);
        return `data:${mimeType};base64,${base64}`;
      }
      if (typeof Buffer !== "undefined") {
        const base64 = Buffer.from(uint8array).toString("base64");
        return `data:${mimeType};base64,${base64}`;
      }
      throw new Error("Unsupported environment: btoa or Buffer not available.");
    }
  };

  // src/ndarray_factory.js
  var ndarray_factory_exports = {};
  __export(ndarray_factory_exports, {
    arange: () => arange,
    array: () => array,
    concat: () => concat,
    eye: () => eye,
    float32: () => float32,
    float64: () => float64,
    full: () => full,
    int16: () => int16,
    int32: () => int32,
    int8: () => int8,
    linspace: () => linspace,
    ones: () => ones,
    stack: () => stack,
    uint16: () => uint16,
    uint32: () => uint32,
    uint8: () => uint8,
    uint8c: () => uint8c,
    where: () => where,
    zeros: () => zeros
  });
  function array(source, dtype = "float64") {
    const Ctor = DTYPE_MAP[dtype] || Float64Array;
    const isTyped = (v) => ArrayBuffer.isView(v) && !(v instanceof DataView);
    const getShape = (v) => {
      if (isTyped(v)) return [v.length];
      if (!Array.isArray(v)) return [];
      if (v.length === 0) throw new Error("Input array cannot be empty.");
      const sub = getShape(v[0]);
      const subStr = sub.join(",");
      for (let i = 1; i < v.length; i++) {
        if (getShape(v[i]).join(",") !== subStr) {
          throw new Error(`Jagged array detected at index ${i}.`);
        }
      }
      return [v.length, ...sub];
    };
    const shape = getShape(source);
    const size = shape.reduce((a, b) => a * b, 1);
    const data = new Ctor(size);
    if (shape.length === 0) {
      data[0] = source;
    } else {
      let offset = 0;
      const lastDim = shape.length - 1;
      const fill = (v, dim) => {
        if (dim === lastDim) {
          data.set(v, offset);
          offset += v.length;
        } else {
          for (let i = 0; i < v.length; i++) {
            fill(v[i], dim + 1);
          }
        }
      };
      fill(source, 0);
    }
    return new NDArray(data, { shape, dtype });
  }
  function float64(source) {
    return array(source, "float64");
  }
  function float32(source) {
    return array(source, "float32");
  }
  function uint32(source) {
    return array(source, "uint32");
  }
  function int32(source) {
    return array(source, "int32");
  }
  function int16(source) {
    return array(source, "int16");
  }
  function uint16(source) {
    return array(source, "uint16");
  }
  function int8(source) {
    return array(source, "int8");
  }
  function uint8(source) {
    return array(source, "uint8");
  }
  function uint8c(source) {
    return array(source, "uint8c");
  }
  function zeros(shape, dtype = "float64") {
    const size = shape.reduce((a, b) => a * b, 1);
    const Constructor = DTYPE_MAP[dtype];
    return new NDArray(new Constructor(size), { shape, dtype });
  }
  function ones(shape, dtype = "float64") {
    const arr = zeros(shape, dtype);
    arr.data.fill(1);
    return arr;
  }
  function full(shape, value, dtype = "float64") {
    const arr = zeros(shape, dtype);
    arr.data.fill(value);
    return arr;
  }
  function arange(start, stop = null, step = 1, dtype = "float64") {
    if (stop === null) {
      stop = start;
      start = 0;
    }
    const size = Math.ceil((stop - start) / step);
    const Constructor = DTYPE_MAP[dtype];
    const data = new Constructor(size);
    for (let i = 0; i < size; i++) {
      data[i] = start + i * step;
    }
    return new NDArray(data, { shape: [size], dtype });
  }
  function linspace(start, stop, num = 50, dtype = "float64") {
    const step = (stop - start) / (num - 1);
    const Constructor = DTYPE_MAP[dtype];
    const data = new Constructor(num);
    for (let i = 0; i < num; i++) {
      data[i] = start + i * step;
    }
    return new NDArray(data, { shape: [num], dtype });
  }
  function where(condition, x, y) {
    const toNDArray = (val, defaultDtype = "float64") => {
      return val instanceof NDArray ? val : array(val, defaultDtype);
    };
    const cond = toNDArray(condition, "int8");
    const dtypes = [x instanceof NDArray ? x.dtype : null, y instanceof NDArray ? y.dtype : null];
    let dtype = null;
    for (let i of dtypes) {
      if (dtype === null) {
        dtype = i;
      } else if (dtype !== i && i !== null) {
        throw new Error(`dtype is different ${dtype} ${i}`);
      }
    }
    if (dtype === null) {
      dtype = "float64";
    }
    const xArr = toNDArray(x, dtype);
    const yArr = toNDArray(y, dtype);
    const outNdim = Math.max(cond.ndim, xArr.ndim, yArr.ndim);
    const outShape = new Int32Array(outNdim);
    for (let i = 1; i <= outNdim; i++) {
      const dC = cond.ndim - i >= 0 ? cond.shape[cond.ndim - i] : 1;
      const dX = xArr.ndim - i >= 0 ? xArr.shape[xArr.ndim - i] : 1;
      const dY = yArr.ndim - i >= 0 ? yArr.shape[yArr.ndim - i] : 1;
      let maxDim = 1;
      const dims = [dC, dX, dY];
      for (let j = 0; j < 3; j++) {
        const d = dims[j];
        if (d !== 1) {
          if (maxDim !== 1 && maxDim !== d) {
            throw new Error(`Incompatible shapes for where: condition [${cond.shape}], x [${xArr.shape}], y [${yArr.shape}]`);
          }
          maxDim = d;
        }
      }
      outShape[outNdim - i] = maxDim;
    }
    const outSize = outShape.reduce((a, b) => a * b, 1);
    const sC = new Int32Array(outNdim);
    const sX = new Int32Array(outNdim);
    const sY = new Int32Array(outNdim);
    const wrapC = new Float64Array(outNdim);
    const wrapX = new Float64Array(outNdim);
    const wrapY = new Float64Array(outNdim);
    for (let i = 0; i < outNdim; i++) {
      const dim = outShape[i];
      const axisFromRight = outNdim - i;
      const getStride = (arr, axisRight) => {
        const axis = arr.ndim - axisRight;
        if (axis < 0 || arr.shape[axis] === 1) return 0;
        return arr.strides[axis];
      };
      sC[i] = getStride(cond, axisFromRight);
      sX[i] = getStride(xArr, axisFromRight);
      sY[i] = getStride(yArr, axisFromRight);
      wrapC[i] = (dim - 1) * sC[i];
      wrapX[i] = (dim - 1) * sX[i];
      wrapY[i] = (dim - 1) * sY[i];
    }
    const Constructor = DTYPE_MAP[xArr.dtype] || Float64Array;
    const newData = new Constructor(outSize);
    const currentIdx = new Int32Array(outNdim);
    let pC = cond.offset;
    let pX = xArr.offset;
    let pY = yArr.offset;
    const dataC = cond.data;
    const dataX = xArr.data;
    const dataY = yArr.data;
    for (let i = 0; i < outSize; i++) {
      newData[i] = dataC[pC] ? dataX[pX] : dataY[pY];
      for (let d = outNdim - 1; d >= 0; d--) {
        if (++currentIdx[d] < outShape[d]) {
          pC += sC[d];
          pX += sX[d];
          pY += sY[d];
          break;
        } else {
          currentIdx[d] = 0;
          pC -= wrapC[d];
          pX -= wrapX[d];
          pY -= wrapY[d];
        }
      }
    }
    return new NDArray(newData, {
      shape: outShape,
      dtype: xArr.dtype
    });
  }
  function concat(arrays, axis = 0) {
    if (!arrays || arrays.length === 0) {
      throw new Error("Array list cannot be empty.");
    }
    if (arrays.length === 1) {
      return arrays[0].copy();
    }
    const firstArr = arrays[0];
    const { dtype, ndim } = firstArr;
    if (ndim === 0) {
      if (!arrays.every((arr) => arr.ndim === 0)) {
        throw new Error("All arrays must be scalars to concatenate with a scalar.");
      }
      const newData = new DTYPE_MAP[dtype](arrays.length);
      for (let i = 0; i < arrays.length; i++) {
        newData[i] = arrays[i].data[arrays[i].offset];
      }
      return new NDArray(newData, { shape: [arrays.length], dtype });
    }
    const finalAxis = axis < 0 ? ndim + axis : axis;
    if (finalAxis < 0 || finalAxis >= ndim) {
      throw new Error(`Axis ${axis} is out of bounds for array of dimension ${ndim}.`);
    }
    let newShape = [...firstArr.shape];
    let concatDimSize = 0;
    for (const arr of arrays) {
      if (arr.ndim !== ndim) throw new Error("All arrays must have same number of dimensions.");
      if (arr.dtype !== dtype) throw new Error("All arrays must have same dtype.");
      for (let j = 0; j < ndim; j++) {
        if (j !== finalAxis && arr.shape[j] !== newShape[j]) {
          throw new Error(`Dimension mismatch on axis ${j}: expected ${newShape[j]} but got ${arr.shape[j]}.`);
        }
      }
      concatDimSize += arr.shape[finalAxis];
    }
    newShape[finalAxis] = concatDimSize;
    const result = zeros(newShape, dtype);
    let destAxisOffset = 0;
    for (const arr of arrays) {
      if (arr.size === 0) continue;
      const dimSize = arr.shape[finalAxis];
      const sliceSpec = Array(ndim).fill(null);
      sliceSpec[finalAxis] = [destAxisOffset, destAxisOffset + dimSize];
      const view = result.slice(...sliceSpec);
      view.set(arr);
      destAxisOffset += dimSize;
    }
    return result;
  }
  function stack(arrays, axis = 0) {
    if (!arrays || arrays.length === 0) {
      throw new Error("Array list cannot be empty.");
    }
    const firstArr = arrays[0];
    const { shape: oldShape, ndim: oldNdim, dtype } = firstArr;
    const targetNdim = oldNdim + 1;
    let finalAxis = axis < 0 ? targetNdim + axis : axis;
    if (finalAxis < 0 || finalAxis >= targetNdim) {
      throw new Error(`Axis ${axis} is out of bounds for stack into ${targetNdim} dimensions.`);
    }
    const expandedArrays = arrays.map((arr, index) => {
      if (arr.ndim !== oldNdim) {
        throw new Error(`Array at index ${index} has ${arr.ndim} dimensions, expected ${oldNdim}.`);
      }
      if (arr.dtype !== dtype) {
        throw new Error(`Dtype mismatch at index ${index}. Expected ${dtype}.`);
      }
      for (let i = 0; i < oldNdim; i++) {
        if (arr.shape[i] !== oldShape[i]) {
          throw new Error(
            `Shape mismatch at index ${index}: expected [${oldShape}] but got [${arr.shape}].`
          );
        }
      }
      const newShape = [...arr.shape];
      newShape.splice(finalAxis, 0, 1);
      return arr.reshape(...newShape);
    });
    return concat(expandedArrays, finalAxis);
  }
  function eye(n, m, dtype = "float64") {
    const rows = n;
    const cols = m === void 0 ? n : m;
    const shape = [rows, cols];
    const res = zeros(shape, dtype);
    const offset = 0;
    const minDim = Math.min(rows, cols);
    for (let i = 0; i < minDim; i++) {
      res.data[offset + i * cols + i] = 1;
    }
    return res;
  }

  // src/ndwasm_optimize.js
  var OPTIMIZE_STATUS_MAP = [
    "NotTerminated",
    "Success",
    "FunctionThreshold",
    "FunctionConvergence",
    "GradientThreshold",
    "StepConvergence",
    "FunctionNegativeInfinity",
    "MethodConverge",
    "Failure,optimize: termination ended in failure",
    "IterationLimit,optimize: maximum number of major iterations reached",
    "RuntimeLimit,optimize: maximum runtime reached",
    "FunctionEvaluationLimit,optimize: maximum number of function evaluations reached",
    "GradientEvaluationLimit,optimize: maximum number of gradient evaluations reached",
    "HessianEvaluationLimit,optimize: maximum number of Hessian evaluations reached"
  ];
  var NDWasmOptimize = {
    /**
     * Provides Optimization capabilities by wrapping Go WASM functions.
     * minimize cᵀ * x
     * s.t      G * x <= h
     * 	        A * x = b
     *          lower <= x <= upper
     * @param {NDArray} c - Coefficient vector for the objective function (1D NDArray of float64).
     * @param {NDArray | null} G - Coefficient matrix for inequality constraints (2D NDArray of float64).
     * @param {NDArray | null} h - Right-hand side vector for inequality constraints (1D NDArray of float64).
     * @param {NDArray | null} A - Coefficient matrix for equality constraints (2D NDArray of float64).
     * @param {NDArray | null} b - Right-hand side vector for equality constraints (1D NDArray of float64).
     * @param {Array} bounds - Optional variable bounds as an array of [lower, upper] pairs. Use null for unbounded. [0, null] for all for default.
     * @returns {{x: NDArray, fun: number, status: number, message: string}} - The optimization result.
     * @throws {Error} If WASM runtime is not loaded or inputs are invalid.
     */
    linprog(c, G, h, A, b, bounds) {
      if (!NDWasm.runtime?.isLoaded) throw new Error("WasmRuntime not loaded.");
      if (c.ndim !== 1) throw new Error("c must be 1D.");
      if (G && G.ndim !== 2 || h && h.ndim !== 1) throw new Error("G must be 2D and h must be 1D.");
      if (A && A.ndim !== 2 || b && b.ndim !== 1) throw new Error("A must be 2D and b must be 1D.");
      if (G && h && G.shape[0] !== h.shape[0]) throw new Error(`Dimension mismatch: G rows (${G.shape[0]}) must match h length (${h.shape[0]}).`);
      if (G && G.shape[1] !== c.shape[0]) throw new Error(`Dimension mismatch: G cols (${G.shape[1]}) must match c length (${c.shape[0]}).`);
      if (A && b && A.shape[0] !== b.shape[0]) throw new Error(`Dimension mismatch: A rows (${A.shape[0]}) must match b length (${b.shape[0]}).`);
      if (A && A.shape[1] !== c.shape[0]) throw new Error(`Dimension mismatch: A cols (${A.shape[1]}) must match c length (${c.shape[0]}).`);
      let cWasm, GWasm, hWasm, AWasm, bWasm, boundsWasm, xResultWasm, objValWasm, statusWasm;
      try {
        const nVars = c.shape[0];
        cWasm = c.toWasm(NDWasm.runtime);
        GWasm = G ? G.toWasm(NDWasm.runtime) : null;
        hWasm = h ? h.toWasm(NDWasm.runtime) : null;
        AWasm = A ? A.toWasm(NDWasm.runtime) : null;
        bWasm = b ? b.toWasm(NDWasm.runtime) : null;
        const boundsData = Array(nVars * 2);
        for (let i = 0; i < nVars; i++) {
          let [lower, upper] = bounds && bounds[i] ? bounds[i] : [0, Infinity];
          if (lower === null) lower = -Infinity;
          if (upper === null) upper = Infinity;
          boundsData[i * 2] = lower;
          boundsData[i * 2 + 1] = upper;
        }
        boundsWasm = array(boundsData).toWasm(NDWasm.runtime);
        xResultWasm = NDWasm.runtime.createBuffer(c.size, c.dtype);
        objValWasm = NDWasm.runtime.createBuffer(1, "float64");
        statusWasm = NDWasm.runtime.createBuffer(1, "int32");
        NDWasm.runtime.exports.LinProg_F64(
          cWasm.ptr,
          cWasm.size,
          GWasm?.ptr ?? 0,
          G ? G.shape[0] : 0,
          hWasm?.ptr ?? 0,
          AWasm?.ptr ?? 0,
          A ? A.shape[0] : 0,
          bWasm?.ptr ?? 0,
          boundsWasm.ptr,
          xResultWasm.ptr,
          objValWasm.ptr,
          statusWasm.ptr
        );
        const x = fromWasm(xResultWasm, c.shape);
        const fun = objValWasm.refresh().view[0];
        const status = statusWasm.refresh().view[0];
        const message = { 0: "Optimal", 1: "Infeasible", 2: "Unbounded", [-1]: "Error" }[status] || "Unknown";
        return { x, fun, status, message };
      } finally {
        [cWasm, GWasm, hWasm, AWasm, bWasm, boundsWasm, xResultWasm, objValWasm, statusWasm].forEach((b2) => b2?.dispose());
      }
    },
    /**
     * Fits a simple linear regression model: Y = alpha + beta*X.
     * @param {NDArray} x - The independent variable (1D NDArray of float64).
     * @param {NDArray} y - The dependent variable (1D NDArray of float64).
     * @returns {{alpha: number, beta: number}} - An object containing the intercept (alpha) and slope (beta) of the fitted line.
     * @throws {Error} If WASM runtime is not loaded or inputs are invalid.
     */
    linearRegression(x, y) {
      if (!NDWasm.runtime?.isLoaded) throw new Error("WasmRuntime not loaded.");
      if (x.ndim !== 1 || y.ndim !== 1 || x.size !== y.size) throw new Error("Inputs must be 1D arrays of the same length.");
      let xWasm, yWasm, alphaWasm, betaWasm;
      try {
        xWasm = x.toWasm(NDWasm.runtime);
        yWasm = y.toWasm(NDWasm.runtime);
        alphaWasm = NDWasm.runtime.createBuffer(1, "float64");
        betaWasm = NDWasm.runtime.createBuffer(1, "float64");
        NDWasm.runtime.exports.LinearRegression_F64(xWasm.ptr, yWasm.ptr, x.size, alphaWasm.ptr, betaWasm.ptr);
        const alpha = alphaWasm.refresh().view[0];
        const beta = betaWasm.refresh().view[0];
        return { alpha, beta };
      } finally {
        [xWasm, yWasm, alphaWasm, betaWasm].forEach((b) => b?.dispose());
      }
    },
    /**
     * Finds the minimum of a scalar function of one or more variables using an L-BFGS optimizer.
     * @param {Function} func - The objective function to be minimized. It must take a 1D `Float64Array` `x` (current point) and return a single number (the function value at `x`).
     * @param {NDArray} x0 - The initial guess for the optimization (1D NDArray of float64).
     * @param {Object} [options] - Optional parameters.
     * @param {Function} [options.grad] - The gradient of the objective function. Must take `x` (a 1D `Float64Array`) and write the result into the second argument `grad_out` (a 1D `Float64Array`). This function should *not* return a value.
     * @returns {{x: NDArray, success: boolean, message: string, ...stats}} The optimization result.
     */
    minimize(func, x0, options = {}) {
      if (!NDWasm.runtime?.isLoaded) throw new Error("WasmRuntime not loaded.");
      if (typeof func !== "function") throw new Error("Objective 'func' must be a JavaScript function.");
      if (x0.ndim !== 1) throw new Error("Initial guess 'x0' must be a 1D NDArray.");
      const { grad } = options;
      let x0Wasm, resultWasm, statsWasm;
      try {
        globalThis.ndarray_minimize_func = function(xPtr, size) {
          const xArr = new Float64Array(NDWasm.runtime.exports.mem.buffer, xPtr, size);
          return func(xArr);
        };
        globalThis.ndarray_minimize_grad = !grad ? null : function(xPtr, gradPtr, size) {
          const xArr = new Float64Array(NDWasm.runtime.exports.mem.buffer, xPtr, size);
          const gradArr = new Float64Array(NDWasm.runtime.exports.mem.buffer, gradPtr, size);
          grad(xArr, gradArr);
        };
        x0Wasm = x0.toWasm(NDWasm.runtime);
        resultWasm = NDWasm.runtime.createBuffer(x0.size, "float64");
        statsWasm = NDWasm.runtime.createBuffer(6, "float64");
        NDWasm.runtime.exports.Minimize_F64(
          x0Wasm.ptr,
          x0.size,
          resultWasm.ptr,
          statsWasm.ptr
        );
        const resultArr = fromWasm(resultWasm, [x0.size], "float64");
        const stats = fromWasm(statsWasm, [6], "float64").toArray();
        const status = stats[0];
        const message = OPTIMIZE_STATUS_MAP[Math.abs(status)] || "Unknown status";
        return {
          x: resultArr,
          success: status > 0,
          // Success if status is Optimal
          status,
          message,
          fun: stats[1],
          niter: stats[2],
          nfev: stats[3],
          ngev: stats[4],
          runtime: stats[5]
        };
      } finally {
        delete globalThis.ndarray_minimize_func;
        delete globalThis.ndarray_minimize_grad;
        [x0Wasm, resultWasm, statsWasm].forEach((b) => b?.dispose());
      }
    }
  };

  // src/ndwasmarray.js
  var NDWasmArray = class _NDWasmArray {
    /**
     * @param {WasmBuffer} buffer - The WASM memory bridge (contains .ptr and .view).
     * @param {Int32Array|Array} shape - Dimensions of the array.
     * @param {string} dtype - Data type (e.g., 'float64').
     */
    constructor(buffer, shape, dtype) {
      this.buffer = buffer;
      this.shape = shape instanceof Int32Array ? shape : Int32Array.from(shape);
      this.dtype = dtype;
      this.ndim = this.shape.length;
      this.size = this.ndim === 0 ? 1 : this.shape.reduce((a, b) => a * b, 1);
    }
    /**
     * Static factory: Creates a WASM-resident array.
     * 1. If source is an NDArray, it calls .push() to move it to WASM.
     * 2. If source is a JS Array, it allocates WASM memory and fills it directly 
     *    via recursive traversal to avoid intermediate flattening.
     */
    static fromArray(source, dtype = "float64") {
      if (source instanceof NDArray) {
        return source.push();
      }
      if (!NDWasm.runtime?.isLoaded) {
        throw new Error("WasmRuntime not initialized. Call NDWasm.bind(runtime) first.");
      }
      if (Array.isArray(source)) {
        const shape = [];
        let curr = source;
        while (Array.isArray(curr)) {
          shape.push(curr.length);
          curr = curr[0];
        }
        const size = shape.length === 0 ? 0 : shape.reduce((a, b) => a * b, 1);
        const buffer = NDWasm.runtime.createBuffer(size, dtype);
        let offset = 0;
        const fill = (arr) => {
          for (let i = 0; i < arr.length; i++) {
            if (Array.isArray(arr[i])) {
              fill(arr[i]);
            } else {
              buffer.view[offset++] = arr[i];
            }
          }
        };
        fill(source);
        return new _NDWasmArray(buffer, shape, dtype);
      }
      if (typeof source === "number") {
        const buffer = NDWasm.runtime.createBuffer(1, dtype);
        buffer.view[0] = source;
        return new _NDWasmArray(buffer, [1], dtype);
      }
      throw new Error("Source must be an Array or an NDArray.");
    }
    /**
     * Pulls data from WASM to a JS-managed NDArray.
     * @param {boolean} [dispose=true] - Release WASM memory after pulling.
     */
    pull(dispose = true) {
      if (!this.buffer) throw new Error("WASM memory already disposed.");
      const data = this.buffer.pull();
      const result = new NDArray(data, { shape: this.shape, dtype: this.dtype });
      if (dispose) this.dispose();
      return result;
    }
    /**
     * Manually releases WASM heap memory.
     */
    dispose() {
      if (this.buffer) {
        this.buffer.dispose();
        this.buffer = null;
      }
    }
    /**
     * Internal helper to prepare operands for WASM operations.
     * Ensures input is converted to NDWasmArray and tracks if it needs auto-disposal.
     * @private
     */
    _prepareOperand(operand) {
      if (operand instanceof _NDWasmArray) {
        return [operand, false];
      }
      return [_NDWasmArray.fromArray(operand, this.dtype), true];
    }
    /**
     * Matrix Multiplication: C = this * other
     * @param {NDWasmArray | NDArray} other
     * @returns {NDWasmArray}
     */
    matmul(other) {
      const [right, shouldDispose] = this._prepareOperand(other);
      try {
        if (this.shape[1] !== right.shape[0]) {
          throw new Error(`Inner dimensions mismatch: ${this.shape[1]} != ${right.shape[0]}`);
        }
        const m = this.shape[0];
        const n = this.shape[1];
        const k = right.shape[1];
        const suffix = NDWasm.runtime._getSuffix(this.dtype);
        const outBuffer = NDWasm.runtime.createBuffer(m * k, this.dtype);
        const status = NDWasm.runtime.exports[`MatMul${suffix}`](
          this.buffer.ptr,
          right.buffer.ptr,
          outBuffer.ptr,
          m,
          n,
          k
        );
        if (status !== void 0 && status !== 0) throw new Error(`WASM MatMul failed with status: ${status}`);
        return new _NDWasmArray(outBuffer, [m, k], this.dtype);
      } finally {
        if (shouldDispose) right.dispose();
      }
    }
    /**
     * Batched Matrix Multiplication: C[i] = this[i] * other[i]
     * @param {NDWasmArray | NDArray}
     * @returns {NDWasmArray}
     */
    matmulBatch(other) {
      const [right, shouldDispose] = this._prepareOperand(other);
      try {
        if (this.ndim !== 3 || right.ndim !== 3 || this.shape[0] !== right.shape[0]) {
          throw new Error("Batch dimensions mismatch.");
        }
        const batch = this.shape[0];
        const m = this.shape[1];
        const n = this.shape[2];
        const k = right.shape[2];
        const suffix = NDWasm.runtime._getSuffix(this.dtype);
        const outBuffer = NDWasm.runtime.createBuffer(batch * m * k, this.dtype);
        const status = NDWasm.runtime.exports[`MatMulBatch${suffix}`](
          this.buffer.ptr,
          right.buffer.ptr,
          outBuffer.ptr,
          batch,
          m,
          n,
          k
        );
        if (status !== void 0 && status !== 0) throw new Error(`WASM MatMulBatch failed with status: ${status}`);
        return new _NDWasmArray(outBuffer, [batch, m, k], this.dtype);
      } finally {
        if (shouldDispose) right.dispose();
      }
    }
  };

  // src/ndarray_jit.js
  var Jit = { debug: false };
  function _createUnaryKernel(cacheKey, shape, sIn, fnOrStr) {
    let kernel = UNARY_KERNEL_CACHE.get(cacheKey);
    if (kernel) {
      return kernel;
    }
    const ndim = shape.length;
    const opBody = prepareUnaryOp(fnOrStr);
    let fnSource;
    if (ndim === 0) {
      fnSource = `
            return function(dataIn, dataOut, offIn, offOut) {
                "use strict";
                dataOut[offOut] = ${opBody.replace("ptrIn", "offIn")};
            }
        `;
    } else {
      let code = `
            dataOut[ptrOut++] = ${opBody};
            ptrIn += ${sIn[ndim - 1]};
        `;
      code = `for (let i${ndim - 1} = 0; i${ndim - 1} < ${shape[ndim - 1]}; i${ndim - 1}++) { ${code} }`;
      for (let d = ndim - 2; d >= 0; d--) {
        const adjIn = sIn[d] - shape[d + 1] * sIn[d + 1];
        code += ` ptrIn += ${adjIn};`;
        code = `for (let i${d} = 0; i${d} < ${shape[d]}; i${d}++) { ${code} }`;
      }
      fnSource = `
            return function(dataIn, dataOut, offIn, offOut) {
                "use strict";
                let ptrIn = offIn, ptrOut = offOut;
                ${code}
            };
        `;
    }
    if (Jit.debug) {
      console.log(cacheKey, "Unary Kernel Source:\n", fnSource);
    }
    kernel = new Function(fnSource)();
    UNARY_KERNEL_CACHE.set(cacheKey, kernel);
    return kernel;
  }
  function _createBinKernel(cacheKey, shape, sA, sB, sOut, opStr, isComparison) {
    let kernel = BIN_KERNEL_CACHE.get(cacheKey);
    if (kernel) {
      return kernel;
    }
    const ndim = shape.length;
    let body = extractOpBody(opStr);
    body = body.replace(/\ba|x\b/g, "dataA[ptrA]").replace(/\bb|y\b/g, "dataB[ptrB]");
    if (isComparison) body = `(${body}) ? 1 : 0`;
    let fnSource;
    if (ndim === 0) {
      fnSource = `
            return function(dataA, dataB, dataOut, offA, offB, offOut) {
                "use strict";
                let ptrA = offA, ptrB = offB, ptrOut = offOut;
                dataOut[ptrOut] = ${body};
            }
        `;
    } else {
      let code = `
            dataOut[ptrOut] = ${body};
            ptrA += ${sA[ndim - 1]};
            ptrB += ${sB[ndim - 1]};
            ptrOut += ${sOut[ndim - 1]};
        `;
      code = `for (let i${ndim - 1} = 0; i${ndim - 1} < ${shape[ndim - 1]}; i${ndim - 1}++) { ${code} }`;
      for (let d = ndim - 2; d >= 0; d--) {
        const adjA = sA[d] - shape[d + 1] * sA[d + 1];
        const adjB = sB[d] - shape[d + 1] * sB[d + 1];
        const adjOut = sOut[d] - shape[d + 1] * sOut[d + 1];
        code += ` ptrA += ${adjA}; ptrB += ${adjB}; ptrOut += ${adjOut};`;
        code = `for (let i${d} = 0; i${d} < ${shape[d]}; i${d}++) { ${code} }`;
      }
      fnSource = `
            return function(dataA, dataB, dataOut, offA, offB, offOut) {
                "use strict";
                let ptrA = offA, ptrB = offB, ptrOut = offOut;
                ${code}
            };
        `;
    }
    if (Jit.debug) {
      console.log(cacheKey, "Bin Kernel Source:\n", fnSource);
    }
    kernel = new Function(fnSource)();
    BIN_KERNEL_CACHE.set(cacheKey, kernel);
    return kernel;
  }
  function _createReduceKernel(cacheKey, shape, strides, iterAxes, reduceAxes, reducer, finalFn) {
    let kernel = REDUCE_KERNEL_CACHE.get(cacheKey);
    if (kernel) {
      return kernel;
    }
    const nRed = reduceAxes.length;
    const nIter = iterAxes.length;
    const redExpr = prepareReduceExpr(reducer, "reducer");
    const finalExpr = finalFn ? prepareReduceExpr(finalFn, "finalizer") : "acc";
    let redCode = `
        acc = ${redExpr};
        pIn += ${strides[reduceAxes[nRed - 1]]};
    `;
    for (let d = nRed - 1; d >= 0; d--) {
      const ax = reduceAxes[d];
      const gap = d === 0 ? 0 : strides[reduceAxes[d - 1]] - shape[ax] * strides[ax];
      redCode = `
            for (let r${d} = 0; r${d} < ${shape[ax]}; r${d}++) {
                ${redCode}
            }
            pIn += ${gap};
        `;
    }
    let fullCode = `
        let acc = initVal;
        ${redCode}
        dataOut[pOut++] = ${finalExpr};
    `;
    if (nIter > 0) {
      const innerBlockDisplacement = shape[reduceAxes[0]] * strides[reduceAxes[0]];
      for (let d = nIter - 1; d >= 0; d--) {
        const ax = iterAxes[d];
        const stride = strides[ax];
        const movedByChild = d === nIter - 1 ? innerBlockDisplacement : shape[iterAxes[d + 1]] * strides[iterAxes[d + 1]];
        const gap = stride - movedByChild;
        fullCode = `
                for (let i${d} = 0; i${d} < ${shape[ax]}; i${d}++) {
                    ${fullCode}
                    pIn += ${gap}; 
                }
            `;
      }
    }
    const fnSource = `
        return function(dataIn, dataOut, offIn, initVal, count) {
            "use strict";
            let pIn = offIn;
            let pOut = 0;
            ${fullCode}
        };
    `;
    if (Jit.debug) {
      console.log(cacheKey, "reduce Kernel Source:\n", fnSource);
    }
    kernel = new Function(fnSource)();
    REDUCE_KERNEL_CACHE.set(cacheKey, kernel);
    return kernel;
  }
  var SET_KERNEL_CACHE = /* @__PURE__ */ new Map();
  function _createSetKernel(cacheKey, ndim, targetShape, tStrides, sStrides, hasPSet, isDimReduced) {
    let kernel = SET_KERNEL_CACHE.get(cacheKey);
    if (kernel) return kernel;
    let targetDimIdx = 0;
    function buildLevel(d) {
      if (d === ndim) {
        return `dataT[pT${d}] = dataS[pS${d}];`;
      }
      const tStride = tStrides[d];
      const sStride = sStrides[d];
      const pT_prev = `pT${d}`;
      const pS_prev = `pS${d}`;
      const pT_next = `pT${d + 1}`;
      const pS_next = `pS${d + 1}`;
      if (isDimReduced[d]) {
        return `
            const ${pT_next} = ${pT_prev} + ps[${d}][0] * ${tStride};
            const ${pS_next} = ${pS_prev};
            ${buildLevel(d + 1)}`;
      } else {
        const len = targetShape[targetDimIdx++];
        if (!hasPSet[d]) {
          return `
                for (let i${d} = 0, ${pT_next} = ${pT_prev}, ${pS_next} = ${pS_prev}; i${d} < ${len}; i${d}++, ${pT_next} += ${tStride}, ${pS_next} += ${sStride}) {
                    ${buildLevel(d + 1)}
                }`;
        } else {
          return `
                for (let i${d} = 0; i${d} < ${len}; i${d}++) {
                    const ${pT_next} = ${pT_prev} + ps[${d}][i${d}] * ${tStride};
                    const ${pS_next} = ${pS_prev} + i${d} * ${sStride};
                    ${buildLevel(d + 1)}
                }`;
        }
      }
    }
    const fnSource = `
        return function(dataT, dataS, offT, offS, ps) {
            "use strict";
            const pT0 = offT;
            const pS0 = offS;
            ${buildLevel(0)}
        };
    `;
    if (Jit.debug) {
      console.log(cacheKey, "Set Kernel Source:\n", fnSource);
    }
    kernel = new Function(fnSource)();
    SET_KERNEL_CACHE.set(cacheKey, kernel);
    return kernel;
  }
  var BIN_KERNEL_CACHE = /* @__PURE__ */ new Map();
  function extractOpBody(fnStr) {
    let match = fnStr.match(/=>\s*([\s\S]+)/);
    if (match) return match[1].trim().replace(/;$/, "");
    match = fnStr.match(/\{[\s\S]*return\s+([\s\S]+?);?\s*\}/);
    if (match) return match[1].trim();
    return fnStr;
  }
  var UNARY_KERNEL_CACHE = /* @__PURE__ */ new Map();
  function prepareUnaryOp(fnOrStr) {
    if (typeof fnOrStr === "string") {
      if (fnOrStr.includes("${val}")) {
        return fnOrStr.replace(/\$\{val\}/g, "dataIn[ptrIn]");
      }
      return fnOrStr.replace(/\bval\b/g, "dataIn[ptrIn]");
    }
    const fnStr = fnOrStr.toString();
    const body = extractOpBody(fnStr);
    return body.replace(/\b(x|a|val|item)\b/g, "dataIn[ptrIn]");
  }
  var REDUCE_KERNEL_CACHE = /* @__PURE__ */ new Map();
  function prepareReduceExpr(fnOrStr, type = "reducer") {
    const s = fnOrStr.toString();
    const body = extractOpBody(s);
    if (type === "reducer") {
      return body.replace(/\bacc|a\b/g, "acc").replace(/\b(b|val|v|item)\b/g, "dataIn[pIn]");
    }
    return body.replace(/\bacc|a\b/g, "acc").replace(/\b(n|count|len)\b/g, "count");
  }
  var PICK_KERNEL_CACHE = /* @__PURE__ */ new Map();
  function _createPickKernel(cacheKey, ndim, sStrides, isFullSlice, isDimReduced, odometerShape) {
    let kernel = PICK_KERNEL_CACHE.get(cacheKey);
    if (kernel) return kernel;
    function buildLevel(d) {
      if (d === ndim) {
        return `dataOut[pOut++] = dataIn[pIn${d}];`;
      }
      const sStride = sStrides[d];
      const pIn_prev = `pIn${d}`;
      const pIn_next = `pIn${d + 1}`;
      const len = odometerShape[d];
      if (isDimReduced[d]) {
        return `
            const ${pIn_next} = ${pIn_prev} + ps[${d}][0] * ${sStride};
            ${buildLevel(d + 1)}`;
      } else if (isFullSlice[d]) {
        return `
            for (let i${d} = 0, ${pIn_next} = ${pIn_prev}; i${d} < ${len}; i${d}++, ${pIn_next} += ${sStride}) {
                ${buildLevel(d + 1)}
            }`;
      } else {
        return `
            for (let i${d} = 0; i${d} < ${len}; i${d}++) {
                const ${pIn_next} = ${pIn_prev} + ps[${d}][i${d}] * ${sStride};
                ${buildLevel(d + 1)}
            }`;
      }
    }
    const fnSource = `
        return function(dataIn, dataOut, offIn, ps) {
            "use strict";
            let pOut = 0;
            const pIn0 = offIn;
            ${buildLevel(0)}
        };
    `;
    if (Jit.debug) {
      console.log(cacheKey, "Pick Kernel Source:\n", fnSource);
    }
    kernel = new Function(fnSource)();
    PICK_KERNEL_CACHE.set(cacheKey, kernel);
    return kernel;
  }

  // src/ndarray_core.js
  var DTYPE_MAP = {
    "float64": Float64Array,
    "float32": Float32Array,
    "int32": Int32Array,
    "uint32": Uint32Array,
    "int16": Int16Array,
    "uint16": Uint16Array,
    "int8": Int8Array,
    "uint8": Uint8Array,
    "uint8c": Uint8ClampedArray
  };
  var NDArray = class _NDArray {
    /**
     * @param {TypedArray} data - The underlying physical storage.
     * @param {Object} options
     * @param {Array|Int32Array} options.shape - The dimensions of the array.
     * @param {Array|Int32Array} [options.strides] - The strides, defaults to C-style.
     * @param {number} [options.offset=0] - The view offset.
     * @param {string} [options.dtype] - The data type.
     */
    constructor(data, { shape, strides, offset = 0, dtype }) {
      this.data = data;
      this.shape = shape instanceof Int32Array ? shape : Int32Array.from(shape);
      this.ndim = this.shape.length;
      this.offset = offset;
      this.dtype = dtype || this._determineDtype(data);
      this.size = 1;
      for (let i = 0; i < this.ndim; i++) this.size *= this.shape[i];
      if (strides) {
        this.strides = strides instanceof Int32Array ? strides : Int32Array.from(strides);
      } else {
        this.strides = this._computeDefaultStrides(this.shape);
      }
      this.isContiguous = this._checkContiguity();
    }
    static random = NDProb;
    static blas = NDWasmBlas;
    static decomp = NDWasmDecomp;
    static analysis = NDWasmAnalysis;
    static image = NDWasmImage;
    static signal = NDWasmSignal;
    /**
     * Optimization module for linear programming, non-linear minimization, and linear regression.
     * @memberof NDArray
     * @type {NDWasmOptimize}
     */
    static optimize = NDWasmOptimize;
    // --- Internal private helpers ---
    _determineDtype(data) {
      for (const [name, Constructor] of Object.entries(DTYPE_MAP)) {
        if (data instanceof Constructor) return name;
      }
      return "float64";
    }
    _computeDefaultStrides(shape) {
      const strides = new Int32Array(this.ndim);
      let s = 1;
      for (let i = this.ndim - 1; i >= 0; i--) {
        strides[i] = s;
        s *= shape[i];
      }
      return strides;
    }
    _checkContiguity() {
      if (this.ndim === 0) return true;
      let expectedStride = 1;
      for (let i = this.ndim - 1; i >= 0; i--) {
        if (this.shape[i] > 1 && this.strides[i] !== expectedStride) {
          return false;
        }
        expectedStride *= this.shape[i];
      }
      return true;
    }
    /**
     * High-performance addressing: converts multidimensional indices to a physical offset.
     * @private
     * @param {Array|Int32Array} indices 
     * @param {number}
     */
    _getOffset(indices) {
      let ptr = this.offset;
      for (let i = 0; i < indices.length; i++) {
        ptr += indices[i] * this.strides[i];
      }
      return ptr;
    }
    /**
     * To JavaScript Array
     * @returns {Array|number} the array
     */
    toArray() {
      const { data, shape, strides, offset, ndim } = this;
      if (ndim === 0) {
        return data[offset];
      }
      const recurse = (dimIdx, currentOffset) => {
        const size = shape[dimIdx];
        const stride = strides[dimIdx];
        const result = new Array(size);
        if (dimIdx === ndim - 1) {
          for (let i = 0; i < size; i++) {
            result[i] = data[currentOffset + i * stride];
          }
        } else {
          for (let i = 0; i < size; i++) {
            result[i] = recurse(dimIdx + 1, currentOffset + i * stride);
          }
        }
        return result;
      };
      return recurse(0, offset);
    }
    /**
     * Returns a string representation of the ndarray.
     * Formats high-dimensional data with proper indentation and line breaks.
     * @returns {String}
     */
    toString() {
      const { shape, strides, offset, ndim, data, dtype } = this;
      if (ndim === 0) {
        return `array(${data[offset]}, dtype=${dtype})`;
      }
      const format = (dimIdx, currentOffset, indent) => {
        if (dimIdx === ndim - 1) {
          const elements = [];
          const len2 = shape[dimIdx];
          const stride2 = strides[dimIdx];
          for (let i = 0; i < len2; i++) {
            elements.push(data[currentOffset + i * stride2]);
          }
          return "[" + elements.join(", ") + "]";
        }
        const results = [];
        const nextDim = dimIdx + 1;
        const len = shape[dimIdx];
        const stride = strides[dimIdx];
        const nextIndent = indent + " ";
        for (let i = 0; i < len; i++) {
          results.push(format(nextDim, currentOffset + i * stride, nextIndent));
        }
        const depthFromBottom = ndim - dimIdx;
        const lineBreaks = "\n".repeat(Math.max(1, depthFromBottom - 1));
        const separator = "," + lineBreaks + nextIndent;
        return "[" + results.join(separator) + "]";
      };
      const prefix = "array(";
      const dataString = format(0, offset, " ".repeat(prefix.length));
      return `${prefix}${dataString}, dtype=${dtype})`;
    }
    /**
    * High-performance element-wise mapping with jit compilation.
    * @param {string | Function} fnOrStr - The function string to apply to each element, like 'Math.sqrt(${val})', or a lambda expression
    * @returns {NDArray} A new array with the results.
    * 
    */
    map(fnOrStr, dtype = void 0) {
      const outDtype = dtype || this.dtype;
      const result = zeros(this.shape, outDtype);
      const opStr = fnOrStr.toString();
      const cacheKey = `unary|${this.dtype}|${outDtype}|${opStr}|${this.shape}|${this.strides}`;
      let kernel = _createUnaryKernel(cacheKey, this.shape, this.strides, fnOrStr);
      kernel(this.data, result.data, this.offset, result.offset);
      return result;
    }
    /**
     * Generic iterator that handles stride logic. It's slow. use map if you want to use jit.
     * @param {Function} callback - A function called with `(value, index, flatPhysicalIndex)`.
     * @see NDArray#map
     */
    iterate(callback) {
      const currentIdx = new Int32Array(this.ndim);
      for (let i = 0; i < this.size; i++) {
        const ptr = this._getOffset(currentIdx);
        callback(this.data[ptr], i, ptr);
        for (let d = this.ndim - 1; d >= 0; d--) {
          if (++currentIdx[d] < this.shape[d]) break;
          currentIdx[d] = 0;
        }
      }
    }
    // --- Basic Arithmetic ---
    /**
     * Element-wise addition. Supports broadcasting.
     * @function
     * @param {NDArray|number} other - The array or scalar to add.
     * @returns {NDArray} A new array containing the results.
     * 
     */
    add(other) {
      return this._binaryOp(other, (a, b) => a + b);
    }
    /**
     * Element-wise subtraction. Supports broadcasting.
     * @function
     * @param {NDArray|number} other - The array or scalar to subtract.
     * @returns {NDArray} A new array containing the results.
     * 
     */
    sub(other) {
      return this._binaryOp(other, (a, b) => a - b);
    }
    /**
     * Element-wise multiplication. Supports broadcasting.
     * @function
     * @param {NDArray|number} other - The array or scalar to multiply by.
     * @returns {NDArray} A new array containing the results.
     * 
     */
    mul(other) {
      return this._binaryOp(other, (a, b) => a * b);
    }
    /**
     * Element-wise division. Supports broadcasting.
     * @function
     * @param {NDArray|number} other - The array or scalar to divide by.
     * @returns {NDArray} A new array containing the results.
     * 
     */
    div(other) {
      return this._binaryOp(other, (a, b) => a / b);
    }
    /**
     * Element-wise exponentiation. Supports broadcasting.
     * @function
     * @param {NDArray|number} other - The array or scalar exponent.
     * @returns {NDArray} A new array containing the results.
     * 
     */
    pow(other) {
      return this._binaryOp(other, (a, b) => a ** b);
    }
    /**
     * Element-wise modulo. Supports broadcasting.
     * @function
     * @param {NDArray|number} other - The array or scalar divisor.
     * @returns {NDArray} A new array containing the results.
     * 
     */
    mod(other) {
      return this._binaryOp(other, (a, b) => a % b);
    }
    // --- In-place Arithmetic ---
    /**
     * In-place element-wise addition.
     * @function
     * @param {NDArray|number} other - The array or scalar to add.
     * @returns {NDArray} The modified array (`this`).
     * 
     */
    iadd(other) {
      return this._binaryOp(other, (a, b) => a + b, true);
    }
    /**
     * In-place element-wise subtraction.
     * @function
     * @param {NDArray|number} other - The array or scalar to subtract.
     * @returns {NDArray} The modified array (`this`).
     * 
     */
    isub(other) {
      return this._binaryOp(other, (a, b) => a - b, true);
    }
    /**
     * In-place element-wise multiplication.
     * @function
     * @param {NDArray|number} other - The array or scalar to multiply by.
     * @returns {NDArray} The modified array (`this`).
     * 
     */
    imul(other) {
      return this._binaryOp(other, (a, b) => a * b, true);
    }
    /**
     * In-place element-wise division.
     * @function
     * @param {NDArray|number} other - The array or scalar to divide by.
     * @returns {NDArray} The modified array (`this`).
     * 
     */
    idiv(other) {
      return this._binaryOp(other, (a, b) => a / b, true);
    }
    /**
     * In-place element-wise exponentiation.
     * @function
     * @param {NDArray|number} other - The array or scalar exponent.
     * @returns {NDArray} The modified array (`this`).
     * 
     */
    ipow(other) {
      return this._binaryOp(other, (a, b) => a ** b, true);
    }
    /**
     * In-place element-wise modulo.
     * @function
     * @param {NDArray|number} other - The array or scalar divisor.
     * @returns {NDArray} The modified array (`this`).
     * 
     */
    imod(other) {
      return this._binaryOp(other, (a, b) => a % b, true);
    }
    /**
     * bitwise AND. Returns a new array.
     * @function
     * @param {NDArray|number} other - The array or scalar to perform the operation with.
     * @returns {NDArray}
     * 
     */
    bitwise_and(other) {
      return this._binaryOp(other, (a, b) => a & b, false);
    }
    /**
     * bitwise OR. Returns a new array.
     * @function
     * @param {NDArray|number} other - The array or scalar to perform the operation with.
     * @returns {NDArray}
     * 
     */
    bitwise_or(other) {
      return this._binaryOp(other, (a, b) => a | b, false);
    }
    /**
     * bitwise XOR. Returns a new array.
     * @function
     * @param {NDArray|number} other - The array or scalar to perform the operation with.
     * @returns {NDArray}
     * 
     */
    bitwise_xor(other) {
      return this._binaryOp(other, (a, b) => a ^ b, false);
    }
    /**
     * bitwise lshift. Returns a new array.
     * @function
     * @param {NDArray|number} other - The array or scalar to perform the operation with.
     * @returns {NDArray}
     * 
     */
    bitwise_lshift(other) {
      return this._binaryOp(other, (a, b) => a << b, false);
    }
    /**
     * bitwise (logical) rshift. Returns a new array.
     * @function
     * @param {NDArray|number} other - The array or scalar to perform the operation with.
     * @returns {NDArray}
     * 
     */
    bitwise_rshift(other) {
      return this._binaryOp(other, (a, b) => a >>> b, false);
    }
    /**
     * bitwise NOT. Returns a new array.
     * @function
     * @returns {NDArray}
     * 
     */
    bitwise_not() {
      return this.map("~${val}");
    }
    // --- Unary Operations ---
    /**
     * Returns a new array with the numeric negation of each element.
     * @function
     * @returns {NDArray}
     * 
     */
    neg() {
      return this.map("-${val}");
    }
    /**
     * Returns a new array with the absolute value of each element.
     * @function
     * @returns {NDArray}
     * 
     */
    abs() {
      return this.map("Math.abs(${val})");
    }
    /**
     * Returns a new array with `e` raised to the power of each element.
     * @function
     * @returns {NDArray}
     * 
     */
    exp() {
      return this.map("Math.exp(${val})");
    }
    /**
     * Returns a new array with the square root of each element.
     * @function
     * @returns {NDArray}
     * 
     */
    sqrt() {
      return this.map("Math.sqrt(${val})");
    }
    /**
    * Returns a new array with the sine of each element.
    * @function
    * @returns {NDArray}
    * 
    */
    sin() {
      return this.map("Math.sin(${val})");
    }
    /**
     * Returns a new array with the cosine of each element.
     * @function
     * @returns {NDArray}
     * 
     */
    cos() {
      return this.map("Math.cos(${val})");
    }
    /**
     * Returns a new array with the tangent of each element.
     * @function
     * @returns {NDArray}
     * 
     */
    tan() {
      return this.map("Math.tan(${val})");
    }
    /**
     * Returns a new array with the natural logarithm (base e) of each element.
     * @function
     * @returns {NDArray}
     * 
     */
    log() {
      return this.map("Math.log(${val})");
    }
    /**
     * Returns a new array with the smallest integer greater than or equal to each element.
     * @function
     * @returns {NDArray}
     * 
     */
    ceil() {
      return this.map("Math.ceil(${val})");
    }
    /**
     * Returns a new array with the largest integer less than or equal to each element.
     * @function
     * @returns {NDArray}
     * 
     */
    floor() {
      return this.map("Math.floor(${val})");
    }
    /**
     * Returns a new array with the value of each element rounded to the nearest integer.
     * @function
     * @returns {NDArray}
     * 
     */
    round() {
      return this.map("Math.round(${val})");
    }
    // --- Comparison and Logic Operators ---
    /**
     * Element-wise equality comparison. Returns a new boolean (uint8) array.
     * @function
     * @param {NDArray|number} other - The array or scalar to compare with.
     * @returns {NDArray}
     * 
     */
    eq(other) {
      return this._comparisonOp(other, (a, b) => a === b);
    }
    /**
     * Element-wise inequality comparison. Returns a new boolean (uint8) array.
     * @function
     * @param {NDArray|number} other - The array or scalar to compare with.
     * @returns {NDArray}
     * 
     */
    neq(other) {
      return this._comparisonOp(other, (a, b) => a !== b);
    }
    /**
     * Element-wise greater-than comparison. Returns a new boolean (uint8) array.
     * @function
     * @param {NDArray|number} other - The array or scalar to compare with.
     * @returns {NDArray}
     * 
     */
    gt(other) {
      return this._comparisonOp(other, (a, b) => a > b);
    }
    /**
     * Element-wise greater-than-or-equal comparison. Returns a new boolean (uint8) array.
     * @function
     * @param {NDArray|number} other - The array or scalar to compare with.
     * @returns {NDArray}
     * 
     */
    gte(other) {
      return this._comparisonOp(other, (a, b) => a >= b);
    }
    /**
     * Element-wise less-than comparison. Returns a new boolean (uint8) array.
     * @function
     * @param {NDArray|number} other - The array or scalar to compare with.
     * @returns {NDArray}
     * 
     */
    lt(other) {
      return this._comparisonOp(other, (a, b) => a < b);
    }
    /**
     * Element-wise less-than-or-equal comparison. Returns a new boolean (uint8) array.
     * @function
     * @param {NDArray|number} other - The array or scalar to compare with.
     * @returns {NDArray}
     * 
     */
    lte(other) {
      return this._comparisonOp(other, (a, b) => a <= b);
    }
    /**
     * Element-wise logical AND. Returns a new boolean (uint8) array.
     * @function
     * @param {NDArray|number} other - The array or scalar to perform the operation with.
     * @returns {NDArray}
     * 
     */
    logical_and(other) {
      return this._comparisonOp(other, (a, b) => a && b);
    }
    /**
     * Element-wise logical OR. Returns a new boolean (uint8) array.
     * @function
     * @param {NDArray|number} other - The array or scalar to perform the operation with.
     * @returns {NDArray}
     * 
     */
    logical_or(other) {
      return this._comparisonOp(other, (a, b) => a || b);
    }
    /**
     * Element-wise logical NOT. Returns a new boolean (uint8) array.
     * @function
     * @returns {NDArray}
     * 
     */
    logical_not() {
      return this.map("(${val} ? 0 : 1)");
    }
    // --- Reductions ---
    /**
     * Computes the sum of elements along the specified axis.
     * 
     * @param {number|null} [axis=null]
     * @returns {NDArray|number}
     */
    sum(axis = null) {
      return this._reduce(axis, () => 0, (a, b) => a + b);
    }
    /**
     * Computes the cumprod of elements along the specified axis.
     * 
     * @param {number|null} [axis=null]
     * @returns {NDArray|number}
     */
    cumprod(axis = null) {
      return this._reduce(axis, () => 1, (a, b) => a * b);
    }
    /**
     * Computes the arithmetic mean along the specified axis.
     * 
     * @param {number|null} [axis=null]
     * @returns {NDArray|number}
     */
    mean(axis = null) {
      return this._reduce(axis, () => 0, (a, b) => a + b, (acc, n) => acc / n);
    }
    /**
     * Returns the maximum value along the specified axis.
     * 
     * @param {number|null} [axis=null]
     * @returns {NDArray|number}
     */
    max(axis = null) {
      return this._reduce(axis, () => -Infinity, (a, b) => Math.max(a, b));
    }
    /**
     * Returns the minimum value along the specified axis.
     * 
     * @param {number|null} [axis=null]
     * @returns {NDArray|number}
     */
    min(axis = null) {
      return this._reduce(axis, () => Infinity, (a, b) => Math.min(a, b));
    }
    /**
     * Computes the variance along the specified axis.
     * Note: This implementation uses a two-pass approach (mean first, then squared differences).
     * Ensure that the `sub` method supports broadcasting if `axis` is not null.
     * 
     * 
     * @param {number|null} [axis=null] - The axis to reduce.
     * @returns {NDArray|number}
     */
    var(axis = null) {
      const mu = this.mean(axis);
      if (axis === null || this.ndim <= 1) {
        const diff2 = this.sub(mu);
        return diff2.mul(diff2).mean();
      }
      const newShape = new Int32Array(this.shape);
      newShape[axis] = 1;
      const muReshaped = mu.reshape(newShape);
      const diff = this.sub(muReshaped);
      return diff.mul(diff).mean(axis);
    }
    /**
     * Computes the standard deviation along the specified axis.
     * 
     * 
     * @param {number|null} [axis=null] - The axis to reduce.
     * @returns {NDArray|number}
     */
    std(axis = null) {
      const v = this.var(axis);
      return v instanceof _NDArray ? v.sqrt() : Math.sqrt(v);
    }
    /**
      * Internal generic reduction engine with optimizations for different memory layouts.
      * 
      * @param {number|null} axis - The axis to reduce. If null, a global reduction is performed.
      * @param {Function} initFn - Returns the initial value for the accumulator (e.g., () => 0).
      * @param {Function} reducer - The binary reduction function (e.g., (acc, val) => acc + val).
      * @param {Function} [finalFn] - Applied to the final result (e.g., (sum, count) => sum / count).
      * @returns {NDArray|number} A scalar number for global reduction, or an NDArray for axis reduction.
      * @private
      */
    _reduce(axis, initFn, reducer, finalFn = void 0) {
      const ndim = this.ndim;
      if (ndim === 0) {
        const val = this.data[this.offset];
        return finalFn ? finalFn(val, 1) : val;
      }
      let reduceAxes = [];
      if (axis === null || axis === void 0) {
        reduceAxes = Array.from({ length: ndim }, (_, i) => i);
      } else {
        let a = axis < 0 ? axis + ndim : axis;
        reduceAxes = [a];
      }
      const iterAxes = Array.from({ length: ndim }, (_, i) => i).filter((ax) => !reduceAxes.includes(ax));
      const opStr = reducer.toString() + (finalFn ? finalFn.toString() : "");
      const cacheKey = `red|${this.dtype}|${opStr}|${this.shape}|${this.strides}|${reduceAxes}`;
      let kernel = _createReduceKernel(cacheKey, this.shape, this.strides, iterAxes, reduceAxes, reducer, finalFn);
      let reduceCount = reduceAxes.reduce((a, b) => a * this.shape[b], 1);
      if (reduceCount == 0) {
        reduceCount = 1;
      }
      let outSize = this.size / reduceCount;
      if (outSize == 0) {
        outSize = 1;
      }
      const outData = new DTYPE_MAP[this.dtype](outSize);
      const initVal = initFn();
      kernel(this.data, outData, this.offset, initVal, reduceCount);
      if (axis === null || axis === void 0) return outData[0];
      const newShape = this.shape.filter((_, i) => !reduceAxes.includes(i));
      return new _NDArray(outData, { shape: newShape, dtype: this.dtype });
    }
    // --- Index Operators ---
    /**
     * Returns the index of the maximum value in a flattened array.
     * @returns {number}
     * 
     */
    argmax() {
      let maxV = -Infinity, maxIdx = -1;
      this.iterate((v, i) => {
        if (v > maxV) {
          maxV = v;
          maxIdx = i;
        }
      });
      return maxIdx;
    }
    /**
     * Returns the index of the minimum value in a flattened array.
     * @returns {number}
     * 
     */
    argmin() {
      let minV = Infinity, minIdx = -1;
      this.iterate((v, i) => {
        if (v < minV) {
          minV = v;
          minIdx = i;
        }
      });
      return minIdx;
    }
    // --- Logical tools ---
    /**
     * Checks if all elements in the array are truthy.
     * @returns {boolean}
     * 
     */
    all() {
      let result = true;
      this.iterate((v) => {
        if (!v) {
          result = false;
        }
      });
      return result;
    }
    /**
     * Checks if any element in the array is truthy.
     * @returns {boolean}
     * 
     */
    any() {
      let result = false;
      this.iterate((v) => {
        if (v) {
          result = true;
        }
      });
      return result;
    }
    //------------view functions------------- 
    /**
     * Returns a new array with a new shape, without changing data. O(1) operation.
     * This only works for contiguous arrays. If the array is not contiguous,
     * you must call .copy() first.
     * 
     * @param {...number} newShape - The new shape.
     * @returns {NDArray} NDArray view.
     */
    reshape(...newShape) {
      if (Array.isArray(newShape[0]) || ArrayBuffer.isView(newShape[0])) newShape = newShape[0];
      const newSize = newShape.reduce((a, b) => a * b, 1);
      if (newSize !== this.size) {
        throw new Error(`Cannot reshape array of size ${this.size} into shape [${newShape}] = ${newSize}`);
      }
      if (this.isContiguous) {
        return new _NDArray(this.data, {
          shape: newShape,
          offset: this.offset,
          dtype: this.dtype
        });
      }
      throw new Error("Reshape of non-contiguous view is ambiguous. Use .copy().reshape() first.");
    }
    /**
     * Returns a new view of the array with axes transposed. O(1) operation.
     * 
     * @param {...number} axes - The new order of axes, e.g., [1, 0] for a matrix transpose. If not specified, reverses the order of the axes.
     * @returns {NDArray} NDArray view.
     */
    transpose(...axes) {
      if (axes.length === 0) {
        axes = Array.from({ length: this.ndim }, (_, i) => this.ndim - 1 - i);
      } else if (Array.isArray(axes[0])) {
        axes = axes[0];
      }
      const newShape = new Int32Array(this.ndim);
      const newStrides = new Int32Array(this.ndim);
      for (let i = 0; i < this.ndim; i++) {
        const axis = axes[i];
        newShape[i] = this.shape[axis];
        newStrides[i] = this.strides[axis];
      }
      return new _NDArray(this.data, {
        shape: newShape,
        strides: newStrides,
        offset: this.offset,
        dtype: this.dtype
      });
    }
    /**
      * Returns a new view of the array sliced along each dimension.
      * This implementation strictly follows NumPy's basic slicing logic.
      * 
      * 
      * @param {...(Array|number|null|undefined)} specs - Slice parameters for each dimension.
      * - number: Scalar indexing. Picks a single element and reduces dimensionality (e.g., arr[0]).
      * - [start, end, step]: Range slicing (e.g., arr[start:end:step]).
      * - []/null/undefined: Selects the entire dimension (e.g., arr[:]).
      * 
      * @returns {NDArray} A new O(1) view of the underlying data.
      * @throws {Error} If a scalar index is out of bounds or step is zero.
      */
    slice(...specs) {
      let newOffset = this.offset;
      const newShape = [];
      const newStrides = [];
      for (let i = 0; i < this.ndim; i++) {
        const spec = i < specs.length ? specs[i] : null;
        const size_i = this.shape[i];
        const stride_i = this.strides[i];
        if (typeof spec === "number") {
          let idx = spec < 0 ? size_i + spec : spec;
          if (idx < 0 || idx >= size_i) {
            throw new Error(`Index ${spec} is out of bounds for axis ${i} with size ${size_i}`);
          }
          newOffset += idx * stride_i;
        } else {
          let start, end, step;
          if (Array.isArray(spec)) {
            [start = null, end = null, step = 1] = spec;
          } else {
            start = end = null;
            step = 1;
          }
          if (step === null || step === void 0) step = 1;
          if (step === 0) throw new Error("Slice step cannot be zero");
          if (start !== null) start = start < 0 ? size_i + start : start;
          if (end !== null) end = end < 0 ? size_i + end : end;
          if (step > 0) {
            start = start === null ? 0 : Math.max(0, Math.min(start, size_i));
            end = end === null ? size_i : Math.max(0, Math.min(end, size_i));
          } else {
            start = start === null ? size_i - 1 : Math.max(-1, Math.min(start, size_i - 1));
            end = end === null ? -1 : Math.max(-1, Math.min(end, size_i - 1));
          }
          const sliceLen = Math.max(0, Math.ceil((end - start) / step));
          newOffset += start * stride_i;
          newShape.push(sliceLen);
          newStrides.push(stride_i * step);
        }
      }
      return new _NDArray(this.data, {
        shape: newShape,
        strides: newStrides,
        offset: newOffset,
        dtype: this.dtype
      });
    }
    /**
     * Returns a 1D view of the i-th row.
     * Only applicable to 2D arrays.
     * 
     * @param {number} i - The row index.
     * @returns {NDArray} A 1D NDArray view.
     */
    rowview(i) {
      if (this.ndim !== 2) {
        throw new Error(`rowview requires 2D array, but got ${this.ndim}D`);
      }
      return this.slice(i, null);
    }
    /**
     * Returns a 1D view of the j-th column.
     * Only applicable to 2D arrays.
     * 
     * @param {number} j - The column index.
     * @returns {NDArray} A 1D NDArray view.
     */
    colview(j) {
      if (this.ndim !== 2) {
        throw new Error(`colview requires 2D array, but got ${this.ndim}D`);
      }
      return this.slice(null, j);
    }
    /**
     * Remove axes of length one from the shape. O(1) operation.
     * 
     * @param {number|null} axis - The axis to squeeze. If null, all axes of length 1 are removed.
     * @returns {NDArray} NDArray view.
     */
    squeeze(axis = null) {
      const newShape = [];
      const newStrides = [];
      for (let i = 0; i < this.ndim; i++) {
        const isTargetAxis = axis !== null && i === axis || axis === null && this.shape[i] === 1;
        if (this.shape[i] === 1 && isTargetAxis) {
        } else {
          newShape.push(this.shape[i]);
          newStrides.push(this.strides[i]);
        }
      }
      return new _NDArray(this.data, {
        shape: newShape,
        strides: newStrides,
        offset: this.offset,
        dtype: this.dtype
      });
    }
    /**
     * Returns a new, contiguous array with the same data. O(n) operation.
     * This converts any view (transposed, sliced) into a new array with a standard C-style memory layout.
     * 
     * @param {string | undefined} the target dtype
     * @returns {NDArray} NDArray view.
     */
    copy(dtype = void 0) {
      return this.map("(${val})", dtype || this.dtype);
    }
    /**
     * Ensures the returned array has a contiguous memory layout.
     * If the array is already contiguous, it returns itself. Otherwise, it returns a copy.
     * Often used as a pre-processing step before calling WASM or other libraries.
     * 
     * @returns {NDArray} NDArray view.
     */
    asContiguous() {
      return this.isContiguous ? this : this.copy();
    }
    /**
     * Gets a single element from the array.
     * Note: This has higher overhead than batch operations. Use with care in performance-critical code.
     * 
     * @param {...number} indices - The indices of the element to get.
     * @returns {number}
     */
    get(...indices) {
      if (Array.isArray(indices[0])) indices = indices[0];
      return this.data[this._getOffset(indices)];
    }
    /**
    * Sets value(s) in the array using a unified, JIT-optimized engine.
    * Supports scalar indexing, fancy (array) indexing, and NumPy-style broadcasting.
    * 
    * @param {number|Array|NDArray} value - The source data to assign.
    * @param {...(number|Array|null)} indices - Indices for each dimension.
    * @returns {NDArray}
    */
    set(value, ...indices) {
      if (this.size === 0) {
        return this;
      }
      const indexPickSets = [];
      const targetShape = [];
      const isDimReduced = new Array(this.ndim);
      const hasPSet = new Uint8Array(this.ndim);
      for (let i = 0; i < this.ndim; i++) {
        let spec = i < indices.length ? indices[i] : null;
        if (spec === null || spec === void 0) {
          indexPickSets.push(null);
          targetShape.push(this.shape[i]);
          isDimReduced[i] = false;
          hasPSet[i] = 0;
        } else if (typeof spec === "number") {
          let idx = spec < 0 ? this.shape[i] + spec : spec;
          if (idx < 0 || idx >= this.shape[i]) {
            throw new Error(`Index ${idx} out of bounds for axis ${i}`);
          }
          indexPickSets.push(new Int32Array([idx]));
          isDimReduced[i] = true;
          hasPSet[i] = 1;
        } else {
          const pSet = spec instanceof Int32Array ? spec : Int32Array.from(spec);
          for (let k = 0; k < pSet.length; k++) {
            if (pSet[k] < 0) pSet[k] += this.shape[i];
            if (pSet[k] < 0 || pSet[k] >= this.shape[i]) {
              throw new Error(`Index ${pSet[k]} out of bounds for axis ${i}`);
            }
          }
          indexPickSets.push(pSet);
          targetShape.push(pSet.length);
          isDimReduced[i] = false;
          hasPSet[i] = 1;
        }
      }
      const src = value instanceof _NDArray ? value : array(value, this.dtype);
      const alignedSrcStrides = new Int32Array(this.ndim);
      let targetDimIdx = targetShape.length - 1;
      let srcDimIdx = src.ndim - 1;
      for (let i = this.ndim - 1; i >= 0; i--) {
        if (isDimReduced[i]) {
          alignedSrcStrides[i] = 0;
        } else {
          if (srcDimIdx >= 0) {
            const sDim = src.shape[srcDimIdx];
            const tDim = targetShape[targetDimIdx];
            if (sDim !== tDim && sDim !== 1) {
              throw new Error(`Incompatible broadcast: src [${src.shape}] -> [${targetShape}]`);
            }
            alignedSrcStrides[i] = sDim === 1 ? 0 : src.strides[srcDimIdx];
            srcDimIdx--;
          } else {
            alignedSrcStrides[i] = 0;
          }
          targetDimIdx--;
        }
      }
      const totalSize = targetShape.reduce((a, b) => a * b, 1);
      if (totalSize === 0) return this;
      const cacheKey = `set|${this.ndim}|${hasPSet}|${isDimReduced}|${this.strides}|${alignedSrcStrides}|${targetShape}`;
      const kernel = _createSetKernel(
        cacheKey,
        this.ndim,
        targetShape,
        this.strides,
        alignedSrcStrides,
        hasPSet,
        isDimReduced
      );
      kernel(this.data, src.data, this.offset, src.offset, indexPickSets);
      return this;
    }
    /**
     * Advanced Indexing (Fancy Indexing).
     * Returns a physical COPY of the selected data using a JIT-compiled engine.
     * Picks elements along each dimension.
     * Note: unlike numpy, for advanced (fancy) indexing, output shape won't be reordered. 
     * Dim for 1-element advanced indexing won't be removed, either.
     * 
     * @param {...(number[]|TypedArray|number|null|undefined)} specs - Index selectors. null/undefined means select all
     */
    pick(...specs) {
      const indexPickSets = [];
      const resultShape = [];
      const isDimReduced = new Array(this.ndim);
      const isFullSlice = new Uint8Array(this.ndim);
      const odometerShape = new Int32Array(this.ndim);
      for (let i = 0; i < this.ndim; i++) {
        let spec = i < specs.length ? specs[i] : null;
        if (spec === null || spec === void 0) {
          const indices = new Int32Array(this.shape[i]);
          for (let k = 0; k < this.shape[i]; k++) indices[k] = k;
          indexPickSets.push(indices);
          resultShape.push(this.shape[i]);
          isFullSlice[i] = 1;
          isDimReduced[i] = false;
        } else if (typeof spec === "number") {
          let idx = spec < 0 ? this.shape[i] + spec : spec;
          if (idx < 0 || idx >= this.shape[i]) {
            throw new Error(`Index ${spec} out of bounds for axis ${i}`);
          }
          indexPickSets.push(new Int32Array([idx]));
          isFullSlice[i] = 0;
          isDimReduced[i] = true;
        } else {
          const indices = spec instanceof Int32Array ? spec : Int32Array.from(spec);
          for (let k = 0; k < indices.length; k++) {
            if (indices[k] < 0) indices[k] += this.shape[i];
            if (indices[k] < 0 || indices[k] >= this.shape[i]) {
              throw new Error(`Index ${indices[k]} out of bounds for axis ${i}`);
            }
          }
          indexPickSets.push(indices);
          resultShape.push(indices.length);
          isFullSlice[i] = 0;
          isDimReduced[i] = false;
        }
        odometerShape[i] = indexPickSets[i].length;
      }
      const resultSize = resultShape.reduce((a, b) => a * b, 1);
      const Constructor = DTYPE_MAP[this.dtype];
      const newData = new Constructor(resultSize);
      if (resultSize === 0) {
        return new _NDArray(newData, { shape: resultShape, dtype: this.dtype });
      }
      const cacheKey = `pick|${this.ndim}|${isFullSlice}|${isDimReduced}|${this.strides}|${odometerShape}`;
      const kernel = _createPickKernel(
        cacheKey,
        this.ndim,
        this.strides,
        isFullSlice,
        isDimReduced,
        odometerShape
      );
      kernel(this.data, newData, this.offset, indexPickSets);
      return new _NDArray(newData, {
        shape: resultShape,
        dtype: this.dtype
      });
    }
    /**
     * Responsibility: Implements element-wise filtering.
     * Returns a NEW 1D contiguous NDArray (Copy).
     * Filters elements based on a predicate function or a boolean mask.
     * 
     * 
     * @param {Function|Array|NDArray} predicateOrMask - A function returning boolean, 
     *        or an array/NDArray of the same shape/size containing truthy/falsy values.
     * @returns {NDArray} A new 1D NDArray containing the matched elements.
     */
    filter(predicateOrMask) {
      const results = [];
      const shape = this.shape;
      const strides = this.strides;
      const ndim = this.ndim;
      const isCallback = typeof predicateOrMask === "function";
      let maskData = null;
      let maskOffset = 0;
      let maskStrides = null;
      if (!isCallback) {
        if (predicateOrMask instanceof _NDArray) {
          if (predicateOrMask.size !== this.size) {
            throw new Error("Mask size must match array size");
          }
          maskData = predicateOrMask.data;
          maskOffset = predicateOrMask.offset;
          maskStrides = predicateOrMask.strides;
        } else if (Array.isArray(predicateOrMask)) {
          if (predicateOrMask.length !== this.size) {
            throw new Error("Mask length must match array size");
          }
          maskData = predicateOrMask;
        }
      }
      const currentIdx = new Int32Array(ndim);
      let currPos = this.offset;
      let currMaskPos = maskOffset;
      for (let n = 0; n < this.size; n++) {
        const val = this.data[currPos];
        let keep = false;
        if (isCallback) {
          keep = predicateOrMask(val, currentIdx, this);
        } else {
          keep = maskStrides ? maskData[currMaskPos] : maskData[n];
        }
        if (keep) {
          results.push(val);
        }
        for (let d = ndim - 1; d >= 0; d--) {
          if (++currentIdx[d] < shape[d]) {
            currPos += strides[d];
            if (maskStrides) currMaskPos += maskStrides[d];
            break;
          } else {
            currentIdx[d] = 0;
            currPos -= (shape[d] - 1) * strides[d];
            if (maskStrides) currMaskPos -= (shape[d] - 1) * maskStrides[d];
          }
        }
      }
      const Constructor = DTYPE_MAP[this.dtype];
      return new _NDArray(new Constructor(results), {
        shape: [results.length],
        dtype: this.dtype
      });
    }
    /**
     * @return {NDArray} - new flatten view to the array
     */
    flatten() {
      return this.slice().reshape(this.size);
    }
    //--------------------NDWasm---------------------
    /**
     * Projects the current ndarray to a WASM proxy (WasmBuffer).
     * 
     * @param {WasmRuntime} runtime 
     * @returns {WasmBuffer} A WasmBuffer instance representing the NDArray in WASM memory.
     */
    toWasm(runtime) {
      const bridge = runtime.createBuffer(this.size, this.dtype);
      if (this.isContiguous) {
        bridge.push(this.data.subarray(this.offset, this.offset + this.size));
      } else {
        const target = bridge.view;
        const cacheKey = `toWasm|${this.dtype}|${this.shape}|${this.strides}`;
        let kernel = _createUnaryKernel(cacheKey, this.shape, this.strides, "${val}");
        kernel(this.data, target, this.offset, 0);
      }
      return bridge;
    }
    /**
     * push to wasm
     * @returns {NDWasmArray}
     */
    push() {
      if (!NDWasm.runtime?.isLoaded) throw new Error("WasmRuntime not bound.");
      const buffer = this.toWasm(NDWasm.runtime);
      return new NDWasmArray(buffer, this.shape, this.dtype);
    }
    /**
     * Calculates the trace of a 2D square matrix (sum of diagonal elements).
     * Complexity: O(n)
     * 
     * @returns {number} The sum of the diagonal elements.
     * @throws {Error} If the array is not 2D or not a square matrix.
     */
    trace() {
      return NDWasmBlas.trace(this);
    }
    /**
     * Performs matrix multiplication. This is a wrapper around `NDWasmBlas.matmul`.
     * @param {NDArray} other The right-hand side matrix.
     * @returns {NDArray} The result of the matrix multiplication.
     * @see NDWasmBlas.matmul
     * 
     */
    matmul(other) {
      return NDWasmBlas.matmul(this, other);
    }
    /**
     * Computes the matrix power. This is a wrapper around `NDWasmBlas.matPow`.
     * @param {number} k The exponent.
     * @returns {NDArray} The result of the matrix power.
     * @see NDWasmBlas.matPow
     * 
     */
    matPow(k) {
      return NDWasmBlas.matPow(this, k);
    }
    /**
     * Performs batched matrix multiplication. This is a wrapper around `NDWasmBlas.matmulBatch`.
     * @param {NDArray} other The right-hand side batch of matrices.
     * @returns {NDArray} The result of the batched matrix multiplication.
     * @see NDWasmBlas.matmulBatch
     * 
     */
    matmulBatch(other) {
      return NDWasmBlas.matmulBatch(this, other);
    }
    /**
     * Performs matrix-vector multiplication. This is a wrapper around `NDWasmBlas.matVecMul`.
     * @param {NDArray} vec The vector to multiply by.
     * @returns {NDArray} The resulting vector.
     * @see NDWasmBlas.matVecMul
     * 
     */
    matVecMul(vec) {
      return NDWasmBlas.matVecMul(this, vec);
    }
    /**
     * Performs a symmetric rank-k update. This is a wrapper around `NDWasmBlas.syrk`.
     * @returns {NDArray} The resulting symmetric matrix.
     * @see NDWasmBlas.syrk
     * 
     */
    syrk() {
      return NDWasmBlas.syrk(this);
    }
    /**
     * Computes the vector outer product. This is a wrapper around `NDWasmBlas.ger`.
     * @param {NDArray} other The other vector.
     * @returns {NDArray} The resulting matrix.
     * @see NDWasmBlas.ger
     * 
     */
    ger(other) {
      return NDWasmBlas.ger(this, other);
    }
    /**
     * Computes the Kronecker product. This is a wrapper around `NDWasmAnalysis.kronecker`.
     * @param {NDArray} other The other matrix.
     * @returns {NDArray} The result of the Kronecker product.
     * @see NDWasmAnalysis.kronecker
     * 
     */
    kronecker(other) {
      return NDWasmAnalysis.kronecker(this, other);
    }
    // 2. Decompositions & Solvers
    /**
     * Solves a system of linear equations. This is a wrapper around `NDWasmDecomp.solve`.
     * @param {NDArray} b The right-hand side matrix or vector.
     * @returns {NDArray} The solution matrix.
     * @see NDWasmDecomp.solve
     * 
     */
    solve(b) {
      return NDWasmDecomp.solve(this, b);
    }
    /**
     * Computes the multiplicative inverse of the matrix. This is a wrapper around `NDWasmDecomp.inv`.
     * @returns {NDArray} The inverted matrix.
     * @see NDWasmDecomp.inv
     * 
     */
    inv() {
      return NDWasmDecomp.inv(this);
    }
    /**
     * Computes the Moore-Penrose pseudo-inverse of the matrix. This is a wrapper around `NDWasmDecomp.pinv`.
     * @returns {NDArray} The pseudo-inverted matrix.
     * @see NDWasmDecomp.pinv
     * 
     */
    pinv() {
      return NDWasmDecomp.pinv(this);
    }
    /**
     * Computes the Singular Value Decomposition (SVD). This is a wrapper around `NDWasmDecomp.svd`.
     * @returns {{q: NDArray, r: NDArray}} An object containing the U, S, and V matrices.
     * @see NDWasmDecomp.svd
     * 
     */
    svd() {
      return NDWasmDecomp.svd(this);
    }
    /**
     * Computes the QR decomposition. This is a wrapper around `NDWasmDecomp.qr`.
     * @returns {Object} An object containing the Q and R matrices.
     * @see NDWasmDecomp.qr
     * 
     */
    qr() {
      return NDWasmDecomp.qr(this);
    }
    /**
     * Computes the Cholesky decomposition. This is a wrapper around `NDWasmDecomp.cholesky`.
     * @returns {NDArray} The lower triangular matrix L.
     * @see NDWasmDecomp.cholesky
     * 
     */
    cholesky() {
      return NDWasmDecomp.cholesky(this);
    }
    /**
     * Computes the determinant of the matrix. This is a wrapper around `NDWasmDecomp.det`.
     * @returns {number} The determinant.
     * @see NDWasmDecomp.det
     * @memberof NDWasmDecomp.prototype
     */
    det() {
      return NDWasmDecomp.det(this);
    }
    /**
     * Computes the log-determinant of the matrix. This is a wrapper around `NDWasmDecomp.logDet`.
     * @returns {{sign: number, logAbsDet: number}} An object containing the sign and log-absolute-determinant.
     * @see NDWasmDecomp.logDet
     * @memberof NDWasmDecomp.prototype
     */
    logDet() {
      return NDWasmDecomp.logDet(this);
    }
    /**
     * Computes the LU decomposition. This is a wrapper around `NDWasmDecomp.lu`.
     * @returns {NDArray} The LU matrix.
     * @see NDWasmDecomp.lu
     * 
     */
    lu() {
      return NDWasmDecomp.lu(this);
    }
    /**
     * Computes the eigenvalues and eigenvectors of a general square matrix.
     * @returns {{values: NDArray, vectors: NDArray}} An object containing eigenvalues and eigenvectors.
     * @see NDWasmDecomp.eigen
     */
    eigen() {
      return NDWasmDecomp.eigen(this);
    }
    // 3. Signal Processing
    /**
     * Computes the 1D Fast Fourier Transform. This is a wrapper around `NDWasmSignal.fft`.
     * @this {NDArray} Complex array with shape [..., 2].
     * @returns {NDArray} Complex result with shape [..., 2].
     * @see NDWasmSignal.fft
     * 
     */
    fft() {
      return NDWasmSignal.fft(this);
    }
    /**
     * Computes the 1D Inverse Fast Fourier Transform. This is a wrapper around `NDWasmSignal.ifft`.
     * @this {NDArray} Complex array with shape [..., 2].
     * @returns {NDArray} Complex result with shape [..., 2].
     * @see NDWasmSignal.ifft
     * 
     */
    ifft() {
      return NDWasmSignal.ifft(this, imag);
    }
    /**
     * Computes the 1D Real-to-Complex Fast Fourier Transform. This is a wrapper around `NDWasmSignal.rfft`.
     * @this {NDArray} real input array.
     * @returns {NDArray} Complex result with shape [..., 2].
     * @see NDWasmSignal.rfft
     * 
     */
    rfft() {
      return NDWasmSignal.rfft(this);
    }
    /**
     * 1D Complex-to-Real Inverse Fast Fourier Transform.
     * The input must be a complex array of shape [k, 2], where k is n/2 + 1. This is a wrapper around `NDWasmSignal.rifft`.
    *  @returns {NDArray} Real-valued time domain signal.
     * @this {NDArray} Complex frequency signal of shape [n/2 + 1, 2]
     * @param {number} n - Length of the original real signal.
     * @see NDWasmSignal.rifft
     * 
     */
    rifft(n) {
      return NDWasmSignal.rifft(this, n);
    }
    /**
     * Computes the 2D Fast Fourier Transform. The input array must be 3D with shape [rows, cols, 2].
     * This is a wrapper around `NDWasmSignal.fft2`.
     * @returns {NDArray} 2D Complex result, with the same shape as input.
     * @see NDWasmSignal.fft2
     * 
     */
    fft2() {
      return NDWasmSignal.fft2(this);
    }
    /**
     * 2D Inverse Complex-to-Complex Fast Fourier Transform.
     * The input array must be 3D with shape [rows, cols, 2].
     * The transform is performed in-place.
     * This is a wrapper around `NDWasmSignal.ifft2`.
     * @returns {NDArray} 2D Complex result, with the same shape as input.
     * @see NDWasmSignal.ifft2
     * 
     */
    ifft2() {
      return NDWasmSignal.ifft2(this);
    }
    /**
     * Computes the 1D Discrete Cosine Transform. This is a wrapper around `NDWasmSignal.dct`.
     * @returns {NDArray} The result of the DCT.
     * @see NDWasmSignal.dct
     * 
     */
    dct() {
      return NDWasmSignal.dct(this);
    }
    /**
     * Performs 2D spatial convolution. This is a wrapper around `NDWasmSignal.conv2d`.
     * @param {NDArray} kernel The convolution kernel.
     * @param {number} stride The stride.
     * @param {number} padding The padding.
     * @returns {NDArray} The convolved array.
     * @see NDWasmSignal.conv2d
     * 
     */
    conv2d(kernel, stride, padding) {
      return NDWasmSignal.conv2d(this, kernel, stride, padding);
    }
    /**
     * Performs 2D spatial cross-correlation. This is a wrapper around `NDWasmSignal.correlate2d`.
     * @param {NDArray} kernel The correlation kernel.
     * @param {number} stride The stride.
     * @param {number} padding The padding.
     * @returns {NDArray} The correlated array.
     * @see NDWasmSignal.correlate2d
     * 
     */
    correlate2d(kernel, stride, padding) {
      return NDWasmSignal.correlate2d(this, kernel, stride, padding);
    }
    // 4. Analysis & Stats
    /**
     * Returns the indices that would sort the array. This is a wrapper around `NDWasmAnalysis.argsort`.
     * @returns {NDArray} An array of indices.
     * @see NDWasmAnalysis.argsort
     * 
     */
    argsort() {
      return NDWasmAnalysis.argsort(this);
    }
    /**
     * Finds the top K largest or smallest elements. This is a wrapper around `NDWasmAnalysis.topk`.
     * @param {number} k The number of elements to find.
     * @param {boolean} largest Whether to find the largest or smallest elements.
     * @returns {{values: NDArray, indices: NDArray}} An object containing the values and indices of the top K elements.
     * @see NDWasmAnalysis.topk
     * 
     */
    topk(k, largest) {
      return NDWasmAnalysis.topk(this, k, largest);
    }
    /**
     * Computes the covariance matrix. This is a wrapper around `NDWasmAnalysis.cov`.
     * @returns {NDArray} The covariance matrix.
     * @see NDWasmAnalysis.cov
     * 
     */
    cov() {
      return NDWasmAnalysis.cov(this);
    }
    /**
     * Computes the matrix norm. This is a wrapper around `NDWasmAnalysis.norm`.
     * @param {number} type The type of norm to compute.
     * @returns {number} The norm of the matrix.
     * @see NDWasmAnalysis.norm
     * 
     */
    norm(type) {
      return NDWasmAnalysis.norm(this, type);
    }
    /**
     * Computes the rank of the matrix. This is a wrapper around `NDWasmAnalysis.rank`.
     * @param {number} tol The tolerance for singular values.
     * @returns {number} The rank of the matrix.
     * @see NDWasmAnalysis.rank
     * 
     */
    rank(tol) {
      return NDWasmAnalysis.rank(this, tol);
    }
    /**
     * Computes the eigenvalue decomposition for a symmetric matrix. This is a wrapper around `NDWasmAnalysis.eigenSym`.
     * @param {boolean} vectors Whether to compute the eigenvectors.
     * @returns {{values: NDArray, vectors: NDArray|null}} An object containing the eigenvalues and eigenvectors.
     * @see NDWasmAnalysis.eigenSym
     * 
     */
    eigenSym(vectors) {
      return NDWasmAnalysis.eigenSym(this, vectors);
    }
    /**
     * Estimates the reciprocal condition number of the matrix. This is a wrapper around `NDWasmAnalysis.cond`.
     * @param {number} norm The norm type.
     * @returns {number} The reciprocal condition number.
     * @see NDWasmAnalysis.cond
     * 
     */
    cond(norm = 1) {
      return NDWasmAnalysis.cond(this, norm);
    }
    // 5. Spatial
    /**
     * Computes the pairwise distances between two sets of vectors. This is a wrapper around `NDWasmAnalysis.pairwiseDist`.
     * @param {NDArray} other The other set of vectors.
     * @returns {NDArray} The distance matrix.
     * @see NDWasmAnalysis.pairwiseDist
     * 
     */
    pairwiseDist(other) {
      return NDWasmAnalysis.pairwiseDist(this, other);
    }
    /**
     * Performs K-Means clustering. This is a wrapper around `NDWasmAnalysis.kmeans`.
     * @param {number} k The number of clusters.
     * @param {number} maxIter The maximum number of iterations.
     * @returns {{centroids: NDArray,labels: NDArray,iterations: number}} An object containing the centroids, labels, and number of iterations.
     * @see NDWasmAnalysis.kmeans
     * 
     */
    kmeans(k, maxIter) {
      return NDWasmAnalysis.kmeans(this, k, maxIter);
    }
    _binaryOp(other, opFn, isInplace = false) {
      const b = typeof other === "number" ? array([other]) : other;
      const { outShape, strideA, strideB } = broadcastShapes(this, b);
      const result = isInplace ? this : zeros(outShape, this.dtype);
      const strideOut = result.strides;
      const opStr = opFn.toString();
      const cacheKey = `bin|${this.dtype}|${result.dtype}|${opStr}|${outShape}|${strideA}|${strideB}|${strideOut}`;
      let kernel = _createBinKernel(cacheKey, outShape, strideA, strideB, strideOut, opStr, false);
      kernel(this.data, b.data, result.data, this.offset, b.offset, result.offset);
      return result;
    }
    _comparisonOp(other, compFn) {
      const b = typeof other === "number" ? array([other]) : other;
      const { outShape, strideA, strideB } = broadcastShapes(this, b);
      const result = zeros(outShape, "uint8");
      const strideOut = result.strides;
      const opStr = compFn.toString();
      const cacheKey = `comp|${this.dtype}|${opStr}|${outShape}|${strideA}|${strideB}|${strideOut}`;
      let kernel = _createBinKernel(cacheKey, outShape, strideA, strideB, strideOut, opStr, true);
      kernel(this.data, b.data, result.data, this.offset, b.offset, result.offset);
      return result;
    }
  };
  function broadcastShapes(a, b) {
    const ndim = Math.max(a.ndim, b.ndim);
    const outShape = new Int32Array(ndim);
    const strideA = new Int32Array(ndim);
    const strideB = new Int32Array(ndim);
    for (let i = 1; i <= ndim; i++) {
      const dimA = a.shape[a.ndim - i] || 1;
      const dimB = b.shape[b.ndim - i] || 1;
      const sA = a.strides[a.ndim - i] || 0;
      const sB = b.strides[b.ndim - i] || 0;
      const outDimI = ndim - i;
      if (dimA === dimB) {
        outShape[outDimI] = dimA;
        strideA[outDimI] = sA;
        strideB[outDimI] = sB;
      } else if (dimA === 1) {
        outShape[outDimI] = dimB;
        strideA[outDimI] = 0;
        strideB[outDimI] = sB;
      } else if (dimB === 1) {
        outShape[outDimI] = dimA;
        strideA[outDimI] = sA;
        strideB[outDimI] = 0;
      } else {
        throw new Error(`Incompatible shapes for broadcasting: [${a.shape}] and [${b.shape}]`);
      }
    }
    return { outShape, strideA, strideB };
  }

  // src/docs.json
  var docs_default = {
    formatDoc: {
      longname: "formatDoc",
      kind: "function",
      description: "Formats a JSDoc object into a human-readable string.",
      params: [
        {
          name: "doc",
          description: "The JSDoc object for a single symbol.",
          type: {
            names: [
              "object"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "string"
            ]
          },
          description: "A formatted string representing the documentation."
        }
      ]
    },
    formatDocHTML: {
      longname: "formatDocHTML",
      kind: "function",
      description: "Formats a JSDoc object into an HTML string.",
      params: [
        {
          name: "doc",
          description: "The JSDoc object for a single symbol.",
          type: {
            names: [
              "object"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "string"
            ]
          },
          description: "A formatted HTML string representing the documentation."
        }
      ]
    },
    help: {
      longname: "help",
      kind: "namespace"
    },
    "help.helpmap": {
      longname: "help.helpmap",
      kind: "member",
      description: "A WeakMap that maps live function/class objects back to their documentation names.\nThis map is populated by the registration logic in `index.js`."
    },
    "help.getHelpDoc": {
      longname: "help.getHelpDoc",
      kind: "function",
      description: "Returns help doc object",
      params: [
        {
          name: "target",
          description: "The name of the function/class, or the object itself.",
          type: {
            names: [
              "string",
              "object"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "object",
              "undefined"
            ]
          },
          description: "The documentation obj, or undefined if not found."
        }
      ]
    },
    "help.helpdoc": {
      longname: "help.helpdoc",
      kind: "function",
      description: "Provides help for a given class/function name or a live object.\nLogs the formatted documentation to the console.",
      params: [
        {
          name: "target",
          description: "The name of the function/class, or the object itself.",
          type: {
            names: [
              "string",
              "object"
            ]
          }
        }
      ]
    },
    "help.helphtml": {
      longname: "help.helphtml",
      kind: "function",
      description: "Returns help content as a styled HTML string for a given class/function name or a live object.",
      params: [
        {
          name: "target",
          description: "The name of the function/class, or the object itself.",
          type: {
            names: [
              "string",
              "object"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "string",
              "undefined"
            ]
          },
          description: "The documentation as an HTML string, or undefined if not found."
        }
      ]
    },
    DTYPE_MAP: {
      longname: "DTYPE_MAP",
      kind: "constant",
      description: "The TypedArray map"
    },
    NDArray: {
      longname: "NDArray",
      kind: "class",
      params: [
        {
          name: "data",
          description: "The underlying physical storage.",
          type: {
            names: [
              "TypedArray"
            ]
          }
        },
        {
          name: "options",
          type: {
            names: [
              "Object"
            ]
          }
        },
        {
          name: "options.shape",
          description: "The dimensions of the array.",
          type: {
            names: [
              "Array",
              "Int32Array"
            ]
          }
        },
        {
          name: "options.strides",
          description: "The strides, defaults to C-style.",
          type: {
            names: [
              "Array",
              "Int32Array"
            ]
          },
          optional: true
        },
        {
          name: "options.offset",
          description: "The view offset.",
          type: {
            names: [
              "number"
            ]
          },
          optional: true,
          defaultvalue: 0
        },
        {
          name: "options.dtype",
          description: "The data type.",
          type: {
            names: [
              "string"
            ]
          },
          optional: true
        }
      ]
    },
    "NDArray.prototype.NDArray": {
      longname: "NDArray#NDArray",
      kind: "class",
      params: [
        {
          name: "data",
          description: "The underlying physical storage.",
          type: {
            names: [
              "TypedArray"
            ]
          }
        },
        {
          name: "options",
          type: {
            names: [
              "Object"
            ]
          }
        },
        {
          name: "options.shape",
          description: "The dimensions of the array.",
          type: {
            names: [
              "Array",
              "Int32Array"
            ]
          }
        },
        {
          name: "options.strides",
          description: "The strides, defaults to C-style.",
          type: {
            names: [
              "Array",
              "Int32Array"
            ]
          },
          optional: true
        },
        {
          name: "options.offset",
          description: "The view offset.",
          type: {
            names: [
              "number"
            ]
          },
          optional: true,
          defaultvalue: 0
        },
        {
          name: "options.dtype",
          description: "The data type.",
          type: {
            names: [
              "string"
            ]
          },
          optional: true
        }
      ]
    },
    "NDArray.optimize": {
      longname: "NDArray.optimize",
      kind: "member",
      description: "Optimization module for linear programming, non-linear minimization, and linear regression."
    },
    "NDArray.prototype._getOffset": {
      longname: "NDArray#_getOffset",
      kind: "function",
      description: "High-performance addressing: converts multidimensional indices to a physical offset.",
      params: [
        {
          name: "indices",
          type: {
            names: [
              "Array",
              "Int32Array"
            ]
          }
        },
        {
          name: "",
          type: {
            names: [
              "number"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.toArray": {
      longname: "NDArray#toArray",
      kind: "function",
      description: "To JavaScript Array",
      params: [],
      returns: [
        {
          type: {
            names: [
              "Array",
              "number"
            ]
          },
          description: "the array"
        }
      ]
    },
    "<anonymous>~recurse": {
      longname: "<anonymous>~recurse",
      kind: "function",
      params: [
        {
          name: "dimIdx",
          type: {
            names: [
              "number"
            ]
          }
        },
        {
          name: "currentOffset",
          type: {
            names: [
              "number"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.toString": {
      longname: "NDArray#toString",
      kind: "function",
      description: "Returns a string representation of the ndarray.\rFormats high-dimensional data with proper indentation and line breaks.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "String"
            ]
          }
        }
      ]
    },
    "<anonymous>~format": {
      longname: "<anonymous>~format",
      kind: "function",
      description: "Recursive helper to format array dimensions.",
      params: [
        {
          name: "dimIdx",
          description: "Current dimension index being processed.",
          type: {
            names: [
              "number"
            ]
          }
        },
        {
          name: "currentOffset",
          description: "Current absolute memory offset in the data buffer.",
          type: {
            names: [
              "number"
            ]
          }
        },
        {
          name: "indent",
          description: "String used for vertical alignment.",
          type: {
            names: [
              "string"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.map": {
      longname: "NDArray#map",
      kind: "function",
      description: "High-performance element-wise mapping with jit compilation.",
      params: [
        {
          name: "fnOrStr",
          description: "The function string to apply to each element, like 'Math.sqrt(${val})', or a lambda expression",
          type: {
            names: [
              "string",
              "function"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "A new array with the results."
        }
      ]
    },
    "NDArray.prototype.iterate": {
      longname: "NDArray#iterate",
      kind: "function",
      description: "Generic iterator that handles stride logic. It's slow. use map if you want to use jit.",
      params: [
        {
          name: "callback",
          description: "A function called with `(value, index, flatPhysicalIndex)`.",
          type: {
            names: [
              "function"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.add": {
      longname: "NDArray#add",
      kind: "function",
      description: "Element-wise addition. Supports broadcasting.",
      params: [
        {
          name: "other",
          description: "The array or scalar to add.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "A new array containing the results."
        }
      ]
    },
    "NDArray.prototype.sub": {
      longname: "NDArray#sub",
      kind: "function",
      description: "Element-wise subtraction. Supports broadcasting.",
      params: [
        {
          name: "other",
          description: "The array or scalar to subtract.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "A new array containing the results."
        }
      ]
    },
    "NDArray.prototype.mul": {
      longname: "NDArray#mul",
      kind: "function",
      description: "Element-wise multiplication. Supports broadcasting.",
      params: [
        {
          name: "other",
          description: "The array or scalar to multiply by.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "A new array containing the results."
        }
      ]
    },
    "NDArray.prototype.div": {
      longname: "NDArray#div",
      kind: "function",
      description: "Element-wise division. Supports broadcasting.",
      params: [
        {
          name: "other",
          description: "The array or scalar to divide by.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "A new array containing the results."
        }
      ]
    },
    "NDArray.prototype.pow": {
      longname: "NDArray#pow",
      kind: "function",
      description: "Element-wise exponentiation. Supports broadcasting.",
      params: [
        {
          name: "other",
          description: "The array or scalar exponent.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "A new array containing the results."
        }
      ]
    },
    "NDArray.prototype.mod": {
      longname: "NDArray#mod",
      kind: "function",
      description: "Element-wise modulo. Supports broadcasting.",
      params: [
        {
          name: "other",
          description: "The array or scalar divisor.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "A new array containing the results."
        }
      ]
    },
    "NDArray.prototype.iadd": {
      longname: "NDArray#iadd",
      kind: "function",
      description: "In-place element-wise addition.",
      params: [
        {
          name: "other",
          description: "The array or scalar to add.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The modified array (`this`)."
        }
      ]
    },
    "NDArray.prototype.isub": {
      longname: "NDArray#isub",
      kind: "function",
      description: "In-place element-wise subtraction.",
      params: [
        {
          name: "other",
          description: "The array or scalar to subtract.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The modified array (`this`)."
        }
      ]
    },
    "NDArray.prototype.imul": {
      longname: "NDArray#imul",
      kind: "function",
      description: "In-place element-wise multiplication.",
      params: [
        {
          name: "other",
          description: "The array or scalar to multiply by.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The modified array (`this`)."
        }
      ]
    },
    "NDArray.prototype.idiv": {
      longname: "NDArray#idiv",
      kind: "function",
      description: "In-place element-wise division.",
      params: [
        {
          name: "other",
          description: "The array or scalar to divide by.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The modified array (`this`)."
        }
      ]
    },
    "NDArray.prototype.ipow": {
      longname: "NDArray#ipow",
      kind: "function",
      description: "In-place element-wise exponentiation.",
      params: [
        {
          name: "other",
          description: "The array or scalar exponent.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The modified array (`this`)."
        }
      ]
    },
    "NDArray.prototype.imod": {
      longname: "NDArray#imod",
      kind: "function",
      description: "In-place element-wise modulo.",
      params: [
        {
          name: "other",
          description: "The array or scalar divisor.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The modified array (`this`)."
        }
      ]
    },
    "NDArray.prototype.bitwise_and": {
      longname: "NDArray#bitwise_and",
      kind: "function",
      description: "bitwise AND. Returns a new array.",
      params: [
        {
          name: "other",
          description: "The array or scalar to perform the operation with.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.bitwise_or": {
      longname: "NDArray#bitwise_or",
      kind: "function",
      description: "bitwise OR. Returns a new array.",
      params: [
        {
          name: "other",
          description: "The array or scalar to perform the operation with.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.bitwise_xor": {
      longname: "NDArray#bitwise_xor",
      kind: "function",
      description: "bitwise XOR. Returns a new array.",
      params: [
        {
          name: "other",
          description: "The array or scalar to perform the operation with.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.bitwise_lshift": {
      longname: "NDArray#bitwise_lshift",
      kind: "function",
      description: "bitwise lshift. Returns a new array.",
      params: [
        {
          name: "other",
          description: "The array or scalar to perform the operation with.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.bitwise_rshift": {
      longname: "NDArray#bitwise_rshift",
      kind: "function",
      description: "bitwise (logical) rshift. Returns a new array.",
      params: [
        {
          name: "other",
          description: "The array or scalar to perform the operation with.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.bitwise_not": {
      longname: "NDArray#bitwise_not",
      kind: "function",
      description: "bitwise NOT. Returns a new array.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.neg": {
      longname: "NDArray#neg",
      kind: "function",
      description: "Returns a new array with the numeric negation of each element.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.abs": {
      longname: "NDArray#abs",
      kind: "function",
      description: "Returns a new array with the absolute value of each element.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.exp": {
      longname: "NDArray#exp",
      kind: "function",
      description: "Returns a new array with `e` raised to the power of each element.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.sqrt": {
      longname: "NDArray#sqrt",
      kind: "function",
      description: "Returns a new array with the square root of each element.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.sin": {
      longname: "NDArray#sin",
      kind: "function",
      description: "Returns a new array with the sine of each element.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.cos": {
      longname: "NDArray#cos",
      kind: "function",
      description: "Returns a new array with the cosine of each element.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.tan": {
      longname: "NDArray#tan",
      kind: "function",
      description: "Returns a new array with the tangent of each element.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.log": {
      longname: "NDArray#log",
      kind: "function",
      description: "Returns a new array with the natural logarithm (base e) of each element.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.ceil": {
      longname: "NDArray#ceil",
      kind: "function",
      description: "Returns a new array with the smallest integer greater than or equal to each element.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.floor": {
      longname: "NDArray#floor",
      kind: "function",
      description: "Returns a new array with the largest integer less than or equal to each element.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.round": {
      longname: "NDArray#round",
      kind: "function",
      description: "Returns a new array with the value of each element rounded to the nearest integer.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.eq": {
      longname: "NDArray#eq",
      kind: "function",
      description: "Element-wise equality comparison. Returns a new boolean (uint8) array.",
      params: [
        {
          name: "other",
          description: "The array or scalar to compare with.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.neq": {
      longname: "NDArray#neq",
      kind: "function",
      description: "Element-wise inequality comparison. Returns a new boolean (uint8) array.",
      params: [
        {
          name: "other",
          description: "The array or scalar to compare with.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.gt": {
      longname: "NDArray#gt",
      kind: "function",
      description: "Element-wise greater-than comparison. Returns a new boolean (uint8) array.",
      params: [
        {
          name: "other",
          description: "The array or scalar to compare with.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.gte": {
      longname: "NDArray#gte",
      kind: "function",
      description: "Element-wise greater-than-or-equal comparison. Returns a new boolean (uint8) array.",
      params: [
        {
          name: "other",
          description: "The array or scalar to compare with.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.lt": {
      longname: "NDArray#lt",
      kind: "function",
      description: "Element-wise less-than comparison. Returns a new boolean (uint8) array.",
      params: [
        {
          name: "other",
          description: "The array or scalar to compare with.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.lte": {
      longname: "NDArray#lte",
      kind: "function",
      description: "Element-wise less-than-or-equal comparison. Returns a new boolean (uint8) array.",
      params: [
        {
          name: "other",
          description: "The array or scalar to compare with.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.logical_and": {
      longname: "NDArray#logical_and",
      kind: "function",
      description: "Element-wise logical AND. Returns a new boolean (uint8) array.",
      params: [
        {
          name: "other",
          description: "The array or scalar to perform the operation with.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.logical_or": {
      longname: "NDArray#logical_or",
      kind: "function",
      description: "Element-wise logical OR. Returns a new boolean (uint8) array.",
      params: [
        {
          name: "other",
          description: "The array or scalar to perform the operation with.",
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.logical_not": {
      longname: "NDArray#logical_not",
      kind: "function",
      description: "Element-wise logical NOT. Returns a new boolean (uint8) array.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.sum": {
      longname: "NDArray#sum",
      kind: "function",
      description: "Computes the sum of elements along the specified axis.",
      params: [
        {
          name: "axis",
          type: {
            names: [
              "number",
              "null"
            ]
          },
          optional: true,
          defaultvalue: null
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.cumprod": {
      longname: "NDArray#cumprod",
      kind: "function",
      description: "Computes the cumprod of elements along the specified axis.",
      params: [
        {
          name: "axis",
          type: {
            names: [
              "number",
              "null"
            ]
          },
          optional: true,
          defaultvalue: null
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.mean": {
      longname: "NDArray#mean",
      kind: "function",
      description: "Computes the arithmetic mean along the specified axis.",
      params: [
        {
          name: "axis",
          type: {
            names: [
              "number",
              "null"
            ]
          },
          optional: true,
          defaultvalue: null
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.max": {
      longname: "NDArray#max",
      kind: "function",
      description: "Returns the maximum value along the specified axis.",
      params: [
        {
          name: "axis",
          type: {
            names: [
              "number",
              "null"
            ]
          },
          optional: true,
          defaultvalue: null
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.min": {
      longname: "NDArray#min",
      kind: "function",
      description: "Returns the minimum value along the specified axis.",
      params: [
        {
          name: "axis",
          type: {
            names: [
              "number",
              "null"
            ]
          },
          optional: true,
          defaultvalue: null
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.var": {
      longname: "NDArray#var",
      kind: "function",
      description: "Computes the variance along the specified axis.\rNote: This implementation uses a two-pass approach (mean first, then squared differences).\rEnsure that the `sub` method supports broadcasting if `axis` is not null.",
      params: [
        {
          name: "axis",
          description: "The axis to reduce.",
          type: {
            names: [
              "number",
              "null"
            ]
          },
          optional: true,
          defaultvalue: null
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.std": {
      longname: "NDArray#std",
      kind: "function",
      description: "Computes the standard deviation along the specified axis.",
      params: [
        {
          name: "axis",
          description: "The axis to reduce.",
          type: {
            names: [
              "number",
              "null"
            ]
          },
          optional: true,
          defaultvalue: null
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray",
              "number"
            ]
          }
        }
      ]
    },
    "NDArray.prototype._reduce": {
      longname: "NDArray#_reduce",
      kind: "function",
      description: "Internal generic reduction engine with optimizations for different memory layouts.",
      params: [
        {
          name: "axis",
          description: "The axis to reduce. If null, a global reduction is performed.",
          type: {
            names: [
              "number",
              "null"
            ]
          }
        },
        {
          name: "initFn",
          description: "Returns the initial value for the accumulator (e.g., () => 0).",
          type: {
            names: [
              "function"
            ]
          }
        },
        {
          name: "reducer",
          description: "The binary reduction function (e.g., (acc, val) => acc + val).",
          type: {
            names: [
              "function"
            ]
          }
        },
        {
          name: "finalFn",
          description: "Applied to the final result (e.g., (sum, count) => sum / count).",
          type: {
            names: [
              "function"
            ]
          },
          optional: true
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray",
              "number"
            ]
          },
          description: "A scalar number for global reduction, or an NDArray for axis reduction."
        }
      ]
    },
    "NDArray.prototype.argmax": {
      longname: "NDArray#argmax",
      kind: "function",
      description: "Returns the index of the maximum value in a flattened array.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "number"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.argmin": {
      longname: "NDArray#argmin",
      kind: "function",
      description: "Returns the index of the minimum value in a flattened array.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "number"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.all": {
      longname: "NDArray#all",
      kind: "function",
      description: "Checks if all elements in the array are truthy.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "boolean"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.any": {
      longname: "NDArray#any",
      kind: "function",
      description: "Checks if any element in the array is truthy.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "boolean"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.reshape": {
      longname: "NDArray#reshape",
      kind: "function",
      description: "Returns a new array with a new shape, without changing data. O(1) operation.\rThis only works for contiguous arrays. If the array is not contiguous,\ryou must call .copy() first.",
      params: [
        {
          name: "newShape",
          description: "The new shape.",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "NDArray view."
        }
      ]
    },
    "NDArray.prototype.transpose": {
      longname: "NDArray#transpose",
      kind: "function",
      description: "Returns a new view of the array with axes transposed. O(1) operation.",
      params: [
        {
          name: "axes",
          description: "The new order of axes, e.g., [1, 0] for a matrix transpose. If not specified, reverses the order of the axes.",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "NDArray view."
        }
      ]
    },
    "NDArray.prototype.slice": {
      longname: "NDArray#slice",
      kind: "function",
      description: "Returns a new view of the array sliced along each dimension.\rThis implementation strictly follows NumPy's basic slicing logic.",
      params: [
        {
          name: "specs",
          description: "Slice parameters for each dimension.\r- number: Scalar indexing. Picks a single element and reduces dimensionality (e.g., arr[0]).\r- [start, end, step]: Range slicing (e.g., arr[start:end:step]).\r- []/null/undefined: Selects the entire dimension (e.g., arr[:]).",
          type: {
            names: [
              "Array",
              "number",
              "null",
              "undefined"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "A new O(1) view of the underlying data."
        }
      ]
    },
    "NDArray.prototype.rowview": {
      longname: "NDArray#rowview",
      kind: "function",
      description: "Returns a 1D view of the i-th row.\rOnly applicable to 2D arrays.",
      params: [
        {
          name: "i",
          description: "The row index.",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "A 1D NDArray view."
        }
      ]
    },
    "NDArray.prototype.colview": {
      longname: "NDArray#colview",
      kind: "function",
      description: "Returns a 1D view of the j-th column.\rOnly applicable to 2D arrays.",
      params: [
        {
          name: "j",
          description: "The column index.",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "A 1D NDArray view."
        }
      ]
    },
    "NDArray.prototype.squeeze": {
      longname: "NDArray#squeeze",
      kind: "function",
      description: "Remove axes of length one from the shape. O(1) operation.",
      params: [
        {
          name: "axis",
          description: "The axis to squeeze. If null, all axes of length 1 are removed.",
          type: {
            names: [
              "number",
              "null"
            ]
          },
          defaultvalue: null
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "NDArray view."
        }
      ]
    },
    "NDArray.prototype.copy": {
      longname: "NDArray#copy",
      kind: "function",
      description: "Returns a new, contiguous array with the same data. O(n) operation.\rThis converts any view (transposed, sliced) into a new array with a standard C-style memory layout.",
      params: [
        {
          name: "the",
          description: "target dtype",
          type: {
            names: [
              "string",
              "undefined"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "NDArray view."
        }
      ]
    },
    "NDArray.prototype.asContiguous": {
      longname: "NDArray#asContiguous",
      kind: "function",
      description: "Ensures the returned array has a contiguous memory layout.\rIf the array is already contiguous, it returns itself. Otherwise, it returns a copy.\rOften used as a pre-processing step before calling WASM or other libraries.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "NDArray view."
        }
      ]
    },
    "NDArray.prototype.get": {
      longname: "NDArray#get",
      kind: "function",
      description: "Gets a single element from the array.\rNote: This has higher overhead than batch operations. Use with care in performance-critical code.",
      params: [
        {
          name: "indices",
          description: "The indices of the element to get.",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "number"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.set": {
      longname: "NDArray#set",
      kind: "function",
      description: "Sets value(s) in the array using a unified, JIT-optimized engine.\rSupports scalar indexing, fancy (array) indexing, and NumPy-style broadcasting.",
      params: [
        {
          name: "value",
          description: "The source data to assign.",
          type: {
            names: [
              "number",
              "Array",
              "NDArray"
            ]
          }
        },
        {
          name: "indices",
          description: "Indices for each dimension.",
          type: {
            names: [
              "number",
              "Array",
              "null"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.pick": {
      longname: "NDArray#pick",
      kind: "function",
      description: "Advanced Indexing (Fancy Indexing).\rReturns a physical COPY of the selected data using a JIT-compiled engine.\rPicks elements along each dimension.\rNote: unlike numpy, for advanced (fancy) indexing, output shape won't be reordered. \rDim for 1-element advanced indexing won't be removed, either.",
      params: [
        {
          name: "specs",
          description: "Index selectors. null/undefined means select all",
          type: {
            names: [
              "Array.<number>",
              "TypedArray",
              "number",
              "null",
              "undefined"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.filter": {
      longname: "NDArray#filter",
      kind: "function",
      description: "Responsibility: Implements element-wise filtering.\rReturns a NEW 1D contiguous NDArray (Copy).\rFilters elements based on a predicate function or a boolean mask.",
      params: [
        {
          name: "predicateOrMask",
          description: "A function returning boolean, \r       or an array/NDArray of the same shape/size containing truthy/falsy values.",
          type: {
            names: [
              "function",
              "Array",
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "A new 1D NDArray containing the matched elements."
        }
      ]
    },
    "NDArray.prototype.flatten": {
      longname: "NDArray#flatten",
      kind: "function",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "- new flatten view to the array"
        }
      ]
    },
    "NDArray.prototype.toWasm": {
      longname: "NDArray#toWasm",
      kind: "function",
      description: "Projects the current ndarray to a WASM proxy (WasmBuffer).",
      params: [
        {
          name: "runtime",
          type: {
            names: [
              "WasmRuntime"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "WasmBuffer"
            ]
          },
          description: "A WasmBuffer instance representing the NDArray in WASM memory."
        }
      ]
    },
    "NDArray.prototype.push": {
      longname: "NDArray#push",
      kind: "function",
      description: "push to wasm",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDWasmArray"
            ]
          }
        }
      ]
    },
    "NDArray.prototype.trace": {
      longname: "NDArray#trace",
      kind: "function",
      description: "Calculates the trace of a 2D square matrix (sum of diagonal elements).\rComplexity: O(n)",
      params: [],
      returns: [
        {
          type: {
            names: [
              "number"
            ]
          },
          description: "The sum of the diagonal elements."
        }
      ]
    },
    "NDArray.prototype.matmul": {
      longname: "NDArray#matmul",
      kind: "function",
      description: "Performs matrix multiplication. This is a wrapper around `NDWasmBlas.matmul`.",
      params: [
        {
          name: "other",
          description: "The right-hand side matrix.",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The result of the matrix multiplication."
        }
      ]
    },
    "NDArray.prototype.matPow": {
      longname: "NDArray#matPow",
      kind: "function",
      description: "Computes the matrix power. This is a wrapper around `NDWasmBlas.matPow`.",
      params: [
        {
          name: "k",
          description: "The exponent.",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The result of the matrix power."
        }
      ]
    },
    "NDArray.prototype.matmulBatch": {
      longname: "NDArray#matmulBatch",
      kind: "function",
      description: "Performs batched matrix multiplication. This is a wrapper around `NDWasmBlas.matmulBatch`.",
      params: [
        {
          name: "other",
          description: "The right-hand side batch of matrices.",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The result of the batched matrix multiplication."
        }
      ]
    },
    "NDArray.prototype.matVecMul": {
      longname: "NDArray#matVecMul",
      kind: "function",
      description: "Performs matrix-vector multiplication. This is a wrapper around `NDWasmBlas.matVecMul`.",
      params: [
        {
          name: "vec",
          description: "The vector to multiply by.",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The resulting vector."
        }
      ]
    },
    "NDArray.prototype.syrk": {
      longname: "NDArray#syrk",
      kind: "function",
      description: "Performs a symmetric rank-k update. This is a wrapper around `NDWasmBlas.syrk`.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The resulting symmetric matrix."
        }
      ]
    },
    "NDArray.prototype.ger": {
      longname: "NDArray#ger",
      kind: "function",
      description: "Computes the vector outer product. This is a wrapper around `NDWasmBlas.ger`.",
      params: [
        {
          name: "other",
          description: "The other vector.",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The resulting matrix."
        }
      ]
    },
    "NDArray.prototype.kronecker": {
      longname: "NDArray#kronecker",
      kind: "function",
      description: "Computes the Kronecker product. This is a wrapper around `NDWasmAnalysis.kronecker`.",
      params: [
        {
          name: "other",
          description: "The other matrix.",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The result of the Kronecker product."
        }
      ]
    },
    "NDArray.prototype.solve": {
      longname: "NDArray#solve",
      kind: "function",
      description: "Solves a system of linear equations. This is a wrapper around `NDWasmDecomp.solve`.",
      params: [
        {
          name: "b",
          description: "The right-hand side matrix or vector.",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The solution matrix."
        }
      ]
    },
    "NDArray.prototype.inv": {
      longname: "NDArray#inv",
      kind: "function",
      description: "Computes the multiplicative inverse of the matrix. This is a wrapper around `NDWasmDecomp.inv`.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The inverted matrix."
        }
      ]
    },
    "NDArray.prototype.pinv": {
      longname: "NDArray#pinv",
      kind: "function",
      description: "Computes the Moore-Penrose pseudo-inverse of the matrix. This is a wrapper around `NDWasmDecomp.pinv`.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The pseudo-inverted matrix."
        }
      ]
    },
    "NDArray.prototype.svd": {
      longname: "NDArray#svd",
      kind: "function",
      description: "Computes the Singular Value Decomposition (SVD). This is a wrapper around `NDWasmDecomp.svd`.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "Object"
            ]
          },
          description: "An object containing the U, S, and V matrices."
        }
      ]
    },
    "NDArray.prototype.qr": {
      longname: "NDArray#qr",
      kind: "function",
      description: "Computes the QR decomposition. This is a wrapper around `NDWasmDecomp.qr`.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "Object"
            ]
          },
          description: "An object containing the Q and R matrices."
        }
      ]
    },
    "NDArray.prototype.cholesky": {
      longname: "NDArray#cholesky",
      kind: "function",
      description: "Computes the Cholesky decomposition. This is a wrapper around `NDWasmDecomp.cholesky`.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The lower triangular matrix L."
        }
      ]
    },
    "NDWasmDecomp.prototype.NDArray.prototype.det": {
      longname: "NDWasmDecomp#NDArray#det",
      kind: "function",
      description: "Computes the determinant of the matrix. This is a wrapper around `NDWasmDecomp.det`.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "number"
            ]
          },
          description: "The determinant."
        }
      ]
    },
    "NDWasmDecomp.prototype.NDArray.prototype.logDet": {
      longname: "NDWasmDecomp#NDArray#logDet",
      kind: "function",
      description: "Computes the log-determinant of the matrix. This is a wrapper around `NDWasmDecomp.logDet`.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "Object"
            ]
          },
          description: "An object containing the sign and log-absolute-determinant."
        }
      ]
    },
    "NDArray.prototype.lu": {
      longname: "NDArray#lu",
      kind: "function",
      description: "Computes the LU decomposition. This is a wrapper around `NDWasmDecomp.lu`.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The LU matrix."
        }
      ]
    },
    "NDArray.prototype.eigen": {
      longname: "NDArray#eigen",
      kind: "function",
      description: "Computes the eigenvalues and eigenvectors of a general square matrix.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "Object"
            ]
          },
          description: "An object containing eigenvalues and eigenvectors."
        }
      ]
    },
    "NDArray.prototype.fft": {
      longname: "NDArray#fft",
      kind: "function",
      description: "Computes the 1D Fast Fourier Transform. This is a wrapper around `NDWasmSignal.fft`.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Complex result with shape [..., 2]."
        }
      ]
    },
    "NDArray.prototype.ifft": {
      longname: "NDArray#ifft",
      kind: "function",
      description: "Computes the 1D Inverse Fast Fourier Transform. This is a wrapper around `NDWasmSignal.ifft`.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Complex result with shape [..., 2]."
        }
      ]
    },
    "NDArray.prototype.rfft": {
      longname: "NDArray#rfft",
      kind: "function",
      description: "Computes the 1D Real-to-Complex Fast Fourier Transform. This is a wrapper around `NDWasmSignal.rfft`.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Complex result with shape [..., 2]."
        }
      ]
    },
    "NDArray.prototype.rifft": {
      longname: "NDArray#rifft",
      kind: "function",
      description: "1D Complex-to-Real Inverse Fast Fourier Transform.\rThe input must be a complex array of shape [k, 2], where k is n/2 + 1. This is a wrapper around `NDWasmSignal.rifft`.",
      params: [
        {
          name: "n",
          description: "Length of the original real signal.",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Real-valued time domain signal."
        }
      ]
    },
    "NDArray.prototype.fft2": {
      longname: "NDArray#fft2",
      kind: "function",
      description: "Computes the 2D Fast Fourier Transform. The input array must be 3D with shape [rows, cols, 2].\rThis is a wrapper around `NDWasmSignal.fft2`.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "2D Complex result, with the same shape as input."
        }
      ]
    },
    "NDArray.prototype.ifft2": {
      longname: "NDArray#ifft2",
      kind: "function",
      description: "2D Inverse Complex-to-Complex Fast Fourier Transform.\rThe input array must be 3D with shape [rows, cols, 2].\rThe transform is performed in-place.\rThis is a wrapper around `NDWasmSignal.ifft2`.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "2D Complex result, with the same shape as input."
        }
      ]
    },
    "NDArray.prototype.dct": {
      longname: "NDArray#dct",
      kind: "function",
      description: "Computes the 1D Discrete Cosine Transform. This is a wrapper around `NDWasmSignal.dct`.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The result of the DCT."
        }
      ]
    },
    "NDArray.prototype.conv2d": {
      longname: "NDArray#conv2d",
      kind: "function",
      description: "Performs 2D spatial convolution. This is a wrapper around `NDWasmSignal.conv2d`.",
      params: [
        {
          name: "kernel",
          description: "The convolution kernel.",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "stride",
          description: "The stride.",
          type: {
            names: [
              "number"
            ]
          }
        },
        {
          name: "padding",
          description: "The padding.",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The convolved array."
        }
      ]
    },
    "NDArray.prototype.correlate2d": {
      longname: "NDArray#correlate2d",
      kind: "function",
      description: "Performs 2D spatial cross-correlation. This is a wrapper around `NDWasmSignal.correlate2d`.",
      params: [
        {
          name: "kernel",
          description: "The correlation kernel.",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "stride",
          description: "The stride.",
          type: {
            names: [
              "number"
            ]
          }
        },
        {
          name: "padding",
          description: "The padding.",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The correlated array."
        }
      ]
    },
    "NDArray.prototype.argsort": {
      longname: "NDArray#argsort",
      kind: "function",
      description: "Returns the indices that would sort the array. This is a wrapper around `NDWasmAnalysis.argsort`.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "An array of indices."
        }
      ]
    },
    "NDArray.prototype.topk": {
      longname: "NDArray#topk",
      kind: "function",
      description: "Finds the top K largest or smallest elements. This is a wrapper around `NDWasmAnalysis.topk`.",
      params: [
        {
          name: "k",
          description: "The number of elements to find.",
          type: {
            names: [
              "number"
            ]
          }
        },
        {
          name: "largest",
          description: "Whether to find the largest or smallest elements.",
          type: {
            names: [
              "boolean"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "Object"
            ]
          },
          description: "An object containing the values and indices of the top K elements."
        }
      ]
    },
    "NDArray.prototype.cov": {
      longname: "NDArray#cov",
      kind: "function",
      description: "Computes the covariance matrix. This is a wrapper around `NDWasmAnalysis.cov`.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The covariance matrix."
        }
      ]
    },
    "NDArray.prototype.norm": {
      longname: "NDArray#norm",
      kind: "function",
      description: "Computes the matrix norm. This is a wrapper around `NDWasmAnalysis.norm`.",
      params: [
        {
          name: "type",
          description: "The type of norm to compute.",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "number"
            ]
          },
          description: "The norm of the matrix."
        }
      ]
    },
    "NDArray.prototype.rank": {
      longname: "NDArray#rank",
      kind: "function",
      description: "Computes the rank of the matrix. This is a wrapper around `NDWasmAnalysis.rank`.",
      params: [
        {
          name: "tol",
          description: "The tolerance for singular values.",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "number"
            ]
          },
          description: "The rank of the matrix."
        }
      ]
    },
    "NDArray.prototype.eigenSym": {
      longname: "NDArray#eigenSym",
      kind: "function",
      description: "Computes the eigenvalue decomposition for a symmetric matrix. This is a wrapper around `NDWasmAnalysis.eigenSym`.",
      params: [
        {
          name: "vectors",
          description: "Whether to compute the eigenvectors.",
          type: {
            names: [
              "boolean"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "Object"
            ]
          },
          description: "An object containing the eigenvalues and eigenvectors."
        }
      ]
    },
    "NDArray.prototype.cond": {
      longname: "NDArray#cond",
      kind: "function",
      description: "Estimates the reciprocal condition number of the matrix. This is a wrapper around `NDWasmAnalysis.cond`.",
      params: [
        {
          name: "norm",
          description: "The norm type.",
          type: {
            names: [
              "number"
            ]
          },
          defaultvalue: 1
        }
      ],
      returns: [
        {
          type: {
            names: [
              "number"
            ]
          },
          description: "The reciprocal condition number."
        }
      ]
    },
    "NDArray.prototype.pairwiseDist": {
      longname: "NDArray#pairwiseDist",
      kind: "function",
      description: "Computes the pairwise distances between two sets of vectors. This is a wrapper around `NDWasmAnalysis.pairwiseDist`.",
      params: [
        {
          name: "other",
          description: "The other set of vectors.",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The distance matrix."
        }
      ]
    },
    "NDArray.prototype.kmeans": {
      longname: "NDArray#kmeans",
      kind: "function",
      description: "Performs K-Means clustering. This is a wrapper around `NDWasmAnalysis.kmeans`.",
      params: [
        {
          name: "k",
          description: "The number of clusters.",
          type: {
            names: [
              "number"
            ]
          }
        },
        {
          name: "maxIter",
          description: "The maximum number of iterations.",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "Object"
            ]
          },
          description: "An object containing the centroids, labels, and number of iterations."
        }
      ]
    },
    broadcastShapes: {
      longname: "broadcastShapes",
      kind: "function",
      description: "Calculates the resulting shape from broadcasting two shapes and returns the\rcorresponding strides for each input array to match that new shape.",
      params: [
        {
          name: "a",
          description: "First array.",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "b",
          description: "Second array.",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "Object"
            ]
          }
        }
      ]
    },
    array: {
      longname: "array",
      kind: "function",
      description: "Creates an NDArray from a regular array or TypedArray.",
      params: [
        {
          name: "source",
          description: "The source data.",
          type: {
            names: [
              "Array",
              "TypedArray"
            ]
          }
        },
        {
          name: "dtype",
          description: "The desired data type.",
          type: {
            names: [
              "string"
            ]
          },
          optional: true,
          defaultvalue: "'float64'"
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    float64: {
      longname: "float64",
      kind: "function",
      description: "Creates an NDArray from a regular array or TypedArray.",
      params: [
        {
          name: "source",
          description: "The source data.",
          type: {
            names: [
              "Array",
              "TypedArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    float32: {
      longname: "float32",
      kind: "function",
      description: "Creates an NDArray from a regular array or TypedArray.",
      params: [
        {
          name: "source",
          description: "The source data.",
          type: {
            names: [
              "Array",
              "TypedArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    uint32: {
      longname: "uint32",
      kind: "function",
      description: "Creates an NDArray from a regular array or TypedArray.",
      params: [
        {
          name: "source",
          description: "The source data.",
          type: {
            names: [
              "Array",
              "TypedArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    int32: {
      longname: "int32",
      kind: "function",
      description: "Creates an NDArray from a regular array or TypedArray.",
      params: [
        {
          name: "source",
          description: "The source data.",
          type: {
            names: [
              "Array",
              "TypedArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    int16: {
      longname: "int16",
      kind: "function",
      description: "Creates an NDArray from a regular array or TypedArray.",
      params: [
        {
          name: "source",
          description: "The source data.",
          type: {
            names: [
              "Array",
              "TypedArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    uint16: {
      longname: "uint16",
      kind: "function",
      description: "Creates an NDArray from a regular array or TypedArray.",
      params: [
        {
          name: "source",
          description: "The source data.",
          type: {
            names: [
              "Array",
              "TypedArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    int8: {
      longname: "int8",
      kind: "function",
      description: "Creates an NDArray from a regular array or TypedArray.",
      params: [
        {
          name: "source",
          description: "The source data.",
          type: {
            names: [
              "Array",
              "TypedArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    uint8: {
      longname: "uint8",
      kind: "function",
      description: "Creates an NDArray from a regular array or TypedArray.",
      params: [
        {
          name: "source",
          description: "The source data.",
          type: {
            names: [
              "Array",
              "TypedArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    uint8c: {
      longname: "uint8c",
      kind: "function",
      description: "Creates an NDArray from a regular array or TypedArray.",
      params: [
        {
          name: "source",
          description: "The source data.",
          type: {
            names: [
              "Array",
              "TypedArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    zeros: {
      longname: "zeros",
      kind: "function",
      description: "Creates a new NDArray of the given shape, filled with zeros.",
      params: [
        {
          name: "shape",
          description: "The shape of the new array.",
          type: {
            names: [
              "Array.<number>"
            ]
          }
        },
        {
          name: "dtype",
          description: "The data type of the new array.",
          type: {
            names: [
              "string"
            ]
          },
          optional: true,
          defaultvalue: "'float64'"
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    ones: {
      longname: "ones",
      kind: "function",
      description: "Creates a new NDArray of the given shape, filled with ones.",
      params: [
        {
          name: "shape",
          description: "The shape of the new array.",
          type: {
            names: [
              "Array.<number>"
            ]
          }
        },
        {
          name: "dtype",
          description: "The data type of the new array.",
          type: {
            names: [
              "string"
            ]
          },
          optional: true,
          defaultvalue: "'float64'"
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    full: {
      longname: "full",
      kind: "function",
      description: "Creates a new NDArray of the given shape, filled with a specified value.",
      params: [
        {
          name: "shape",
          description: "The shape of the new array.",
          type: {
            names: [
              "Array.<number>"
            ]
          }
        },
        {
          name: "value",
          description: "The value to fill the array with.",
          type: {
            names: [
              "*"
            ]
          }
        },
        {
          name: "dtype",
          description: "The data type of the new array.",
          type: {
            names: [
              "string"
            ]
          },
          optional: true,
          defaultvalue: "'float64'"
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    arange: {
      longname: "arange",
      kind: "function",
      description: "Like Python's arange: [start, stop)",
      params: [
        {
          name: "start",
          description: "the starting value of the sequence.",
          type: {
            names: [
              "number"
            ]
          }
        },
        {
          name: "stop",
          description: "the end value of the sequence.",
          type: {
            names: [
              "number",
              "null"
            ]
          }
        },
        {
          name: "step",
          description: "the spacing between values.",
          type: {
            names: [
              "number",
              "null"
            ]
          },
          optional: true,
          defaultvalue: 1
        },
        {
          name: "dtype",
          description: "the data type of the resulting array.",
          type: {
            names: [
              "string"
            ]
          },
          optional: true,
          defaultvalue: "'float64'"
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    linspace: {
      longname: "linspace",
      kind: "function",
      description: "Linearly spaced points.",
      params: [
        {
          name: "start",
          description: "the starting value of the sequence.",
          type: {
            names: [
              "number"
            ]
          }
        },
        {
          name: "stop",
          description: "the end value of the sequence.",
          type: {
            names: [
              "number",
              "null"
            ]
          }
        },
        {
          name: "num",
          description: "the number of samples to generate.",
          type: {
            names: [
              "number"
            ]
          },
          optional: true,
          defaultvalue: 50
        },
        {
          name: "dtype",
          description: "the data type of the resulting array.",
          type: {
            names: [
              "string"
            ]
          },
          optional: true,
          defaultvalue: "'float64'"
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    where: {
      longname: "where",
      kind: "function",
      description: "where(condition, x, y)\rReturns a new array with elements chosen from x or y depending on condition.\rSupports NumPy-style broadcasting across all three arguments, including 0-sized dimensions.",
      params: [
        {
          name: "condition",
          description: "Where True, yield x, otherwise yield y.",
          type: {
            names: [
              "NDArray",
              "Array",
              "number"
            ]
          }
        },
        {
          name: "x",
          description: "Values from which to choose if condition is True.",
          type: {
            names: [
              "NDArray",
              "Array",
              "number"
            ]
          }
        },
        {
          name: "y",
          description: "Values from which to choose if condition is False.",
          type: {
            names: [
              "NDArray",
              "Array",
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "A new contiguous NDArray."
        }
      ]
    },
    concat: {
      longname: "concat",
      kind: "function",
      description: "Joins a sequence of arrays along an existing axis.",
      params: [
        {
          name: "arrays",
          description: "The arrays must have the same shape, except in the dimension corresponding to `axis`.",
          type: {
            names: [
              "Array.<NDArray>"
            ]
          }
        },
        {
          name: "axis",
          description: "The axis along which the arrays will be joined.",
          type: {
            names: [
              "number"
            ]
          },
          optional: true,
          defaultvalue: 0
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "A new NDArray."
        }
      ]
    },
    stack: {
      longname: "stack",
      kind: "function",
      description: "Joins a sequence of arrays along a new axis.\rThe `stack` function creates a new dimension, whereas `concat` joins along an existing one.\rAll input arrays must have the same shape and dtype.",
      params: [
        {
          name: "arrays",
          description: "The list of arrays to stack.",
          type: {
            names: [
              "Array.<NDArray>"
            ]
          }
        },
        {
          name: "axis",
          description: "The axis in the result array along which the input arrays are stacked.",
          type: {
            names: [
              "number"
            ]
          },
          optional: true,
          defaultvalue: 0
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "A new NDArray."
        }
      ]
    },
    "<anonymous>~newShape": {
      longname: "<anonymous>~newShape",
      kind: "constant",
      description: "Step 2: Reshape the array to add a singleton dimension at the target axis.\rExample: If shape is [3, 4] and axis is 1, new shape becomes [3, 1, 4].",
      params: []
    },
    eye: {
      longname: "eye",
      kind: "function",
      description: "Creates a 2D identity matrix.",
      params: [
        {
          name: "n",
          description: "Number of rows.",
          type: {
            names: [
              "number"
            ]
          }
        },
        {
          name: "m",
          description: "Number of columns. Defaults to n if not provided.",
          type: {
            names: [
              "number"
            ]
          },
          optional: true
        },
        {
          name: "dtype",
          description: "Data type of the array.",
          type: {
            names: [
              "string"
            ]
          },
          optional: true,
          defaultvalue: "'float64'"
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "A 2D NDArray with ones on the main diagonal."
        }
      ]
    },
    _createUnaryKernel: {
      longname: "_createUnaryKernel",
      kind: "function",
      description: "Generates a JIT-compiled unary kernel.\rOutput is always contiguous, so ptrOut simply increments by 1."
    },
    _createBinKernel: {
      longname: "_createBinKernel",
      kind: "function",
      description: "Generates a kernel with unrolled nested loops and cumulative pointer increments."
    },
    _createReduceKernel: {
      longname: "_createReduceKernel",
      kind: "function",
      description: "Generates a JIT kernel using nested loops and pointer gaps."
    },
    SET_KERNEL_CACHE: {
      longname: "SET_KERNEL_CACHE",
      kind: "constant",
      description: "Global cache for set JIT kernels to avoid redundant compilations.",
      params: []
    },
    _createSetKernel: {
      longname: "_createSetKernel",
      kind: "function",
      description: "Generates a JIT-compiled kernel for the set operation.\rOptimized for V8 by using incremental pointer arithmetic and static nesting.",
      params: [
        {
          name: "cacheKey",
          description: "Unique key for the specific array structure.",
          type: {
            names: [
              "string"
            ]
          }
        },
        {
          name: "ndim",
          description: "Number of dimensions of the target array.",
          type: {
            names: [
              "number"
            ]
          }
        },
        {
          name: "targetShape",
          description: "The logical shape of the selection.",
          type: {
            names: [
              "Array"
            ]
          }
        },
        {
          name: "tStrides",
          description: "Strides of the target NDArray.",
          type: {
            names: [
              "Int32Array"
            ]
          }
        },
        {
          name: "sStrides",
          description: "Pre-computed broadcasting strides for the source.",
          type: {
            names: [
              "Int32Array"
            ]
          }
        },
        {
          name: "hasPSet",
          description: "Flags indicating if a dimension uses fancy indexing or scalar.",
          type: {
            names: [
              "Uint8Array"
            ]
          }
        },
        {
          name: "isDimReduced",
          description: "Flags indicating if a dimension is collapsed by scalar indexing.",
          type: {
            names: [
              "Array.<boolean>"
            ]
          }
        }
      ]
    },
    "_createSetKernel~buildLevel": {
      longname: "_createSetKernel~buildLevel",
      kind: "function",
      description: "Recursively builds the nested loop string.",
      params: [
        {
          name: "d",
          description: "Current dimension index.",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "string"
            ]
          },
          description: "- The generated code block for this dimension."
        }
      ]
    },
    BIN_KERNEL_CACHE: {
      longname: "BIN_KERNEL_CACHE",
      kind: "constant",
      params: []
    },
    extractOpBody: {
      longname: "extractOpBody",
      kind: "function",
      description: "Extracts the expression body from a function.\rSupports both: (a, b) => a + b  AND  function(a, b) { return a + b; }",
      params: []
    },
    UNARY_KERNEL_CACHE: {
      longname: "UNARY_KERNEL_CACHE",
      kind: "constant",
      description: "Global cache for unary JIT kernels (copy/map).",
      params: []
    },
    prepareUnaryOp: {
      longname: "prepareUnaryOp",
      kind: "function",
      description: "Extracts function body or processes template strings into executable JS code.",
      params: []
    },
    REDUCE_KERNEL_CACHE: {
      longname: "REDUCE_KERNEL_CACHE",
      kind: "constant",
      params: []
    },
    prepareReduceExpr: {
      longname: "prepareReduceExpr",
      kind: "function",
      description: "Normalizes the reducer and finalizer into inline expressions.",
      params: []
    },
    PICK_KERNEL_CACHE: {
      longname: "PICK_KERNEL_CACHE",
      kind: "constant",
      description: "Global cache for pick JIT kernels.",
      params: []
    },
    _createPickKernel: {
      longname: "_createPickKernel",
      kind: "function",
      description: "Generates a JIT-compiled kernel for the pick operation.\rOptimized for V8 with incremental pointer arithmetic and contiguous output writes.",
      params: [
        {
          name: "cacheKey",
          description: "Unique key for the specific array structure.",
          type: {
            names: [
              "string"
            ]
          }
        },
        {
          name: "ndim",
          description: "Number of dimensions of the source array.",
          type: {
            names: [
              "number"
            ]
          }
        },
        {
          name: "sStrides",
          description: "Strides of the source NDArray.",
          type: {
            names: [
              "Int32Array"
            ]
          }
        },
        {
          name: "isFullSlice",
          description: 'Flags indicating if a dimension is a full ":" slice.',
          type: {
            names: [
              "Uint8Array"
            ]
          }
        },
        {
          name: "isDimReduced",
          description: "Flags indicating if a dimension is collapsed by scalar indexing.",
          type: {
            names: [
              "Array.<boolean>"
            ]
          }
        },
        {
          name: "odometerShape",
          description: "The lengths of the pick-sets for each dimension.",
          type: {
            names: [
              "Int32Array"
            ]
          }
        }
      ]
    },
    "_createPickKernel~buildLevel": {
      longname: "_createPickKernel~buildLevel",
      kind: "function",
      description: "Recursively builds the nested loop string.",
      params: [
        {
          name: "d",
          description: "Current source dimension index.",
          type: {
            names: [
              "number"
            ]
          }
        }
      ]
    },
    NDProb: {
      longname: "NDProb",
      kind: "namespace",
      description: "NDArray Probability & Random Module\rUses Cryptographically Strong Pseudo-Random Number Generator (CSPRNG).\rAll functions return a new NDArray instance."
    },
    "NDProb._cryptoUniform01": {
      longname: "NDProb._cryptoUniform01",
      kind: "function",
      description: "Internal helper to get a Float64 between [0, 1) using crypto.\rGenerates high-quality uniform random floats."
    },
    "NDProb.random": {
      longname: "NDProb.random",
      kind: "function",
      description: "Uniform distribution over [low, high).",
      params: [
        {
          name: "shape",
          description: "Dimensions of the output array.",
          type: {
            names: [
              "Array"
            ]
          }
        },
        {
          name: "low",
          description: "Lower boundary.",
          type: {
            names: [
              "number"
            ]
          },
          optional: true,
          defaultvalue: 0
        },
        {
          name: "high",
          description: "Upper boundary.",
          type: {
            names: [
              "number"
            ]
          },
          optional: true,
          defaultvalue: 1
        },
        {
          name: "dtype",
          description: "Data type.",
          type: {
            names: [
              "string"
            ]
          },
          optional: true,
          defaultvalue: "'float64'"
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDProb.normal": {
      longname: "NDProb.normal",
      kind: "function",
      description: "Normal (Gaussian) distribution using Box-Muller transform.",
      params: [
        {
          name: "shape",
          description: "Dimensions of the output array.",
          type: {
            names: [
              "Array"
            ]
          }
        },
        {
          name: "mean",
          description: "Mean of the distribution.",
          type: {
            names: [
              "number"
            ]
          },
          optional: true,
          defaultvalue: 0
        },
        {
          name: "std",
          description: "Standard deviation.",
          type: {
            names: [
              "number"
            ]
          },
          optional: true,
          defaultvalue: 1
        },
        {
          name: "dtype",
          description: "Data type.",
          type: {
            names: [
              "string"
            ]
          },
          optional: true,
          defaultvalue: "'float64'"
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDProb.bernoulli": {
      longname: "NDProb.bernoulli",
      kind: "function",
      description: "Bernoulli distribution (0 or 1 with probability p).",
      params: [
        {
          name: "shape",
          type: {
            names: [
              "Array"
            ]
          }
        },
        {
          name: "p",
          description: "Probability of success (1).",
          type: {
            names: [
              "number"
            ]
          },
          optional: true,
          defaultvalue: 0.5
        },
        {
          name: "dtype",
          type: {
            names: [
              "string"
            ]
          },
          optional: true,
          defaultvalue: "'int32'"
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDProb.exponential": {
      longname: "NDProb.exponential",
      kind: "function",
      description: "Exponential distribution: f(x; \u03BB) = \u03BBe^(-\u03BBx).\rInverse transform sampling: x = -ln(1-u) / \u03BB.",
      params: [
        {
          name: "shape",
          type: {
            names: [
              "Array"
            ]
          }
        },
        {
          name: "lambda",
          description: "Rate parameter.",
          type: {
            names: [
              "number"
            ]
          },
          optional: true,
          defaultvalue: "1.0"
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDProb.poisson": {
      longname: "NDProb.poisson",
      kind: "function",
      description: "Poisson distribution using Knuth's algorithm.\rNote: For very large lambda, this becomes slow; but for most use cases it's fine.",
      params: [
        {
          name: "shape",
          type: {
            names: [
              "Array"
            ]
          }
        },
        {
          name: "lambda",
          description: "Mean of the distribution.",
          type: {
            names: [
              "number"
            ]
          },
          optional: true,
          defaultvalue: "1.0"
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "NDProb._cast": {
      longname: "NDProb._cast",
      kind: "function"
    },
    WasmBuffer: {
      longname: "WasmBuffer",
      kind: "class",
      params: [
        {
          name: "exports",
          description: "The exports object from Go WASM.",
          type: {
            names: [
              "Object"
            ]
          }
        },
        {
          name: "size",
          description: "Number of elements.",
          type: {
            names: [
              "number"
            ]
          }
        },
        {
          name: "dtype",
          description: "Data type (float64, float32, etc.).",
          type: {
            names: [
              "string"
            ]
          }
        }
      ]
    },
    "WasmBuffer.prototype.refresh": {
      longname: "WasmBuffer#refresh",
      kind: "function",
      description: "Refreshes the view into WASM memory.",
      params: []
    },
    "WasmBuffer.prototype.push": {
      longname: "WasmBuffer#push",
      kind: "function",
      description: "Synchronizes JS data into the WASM buffer.",
      params: []
    },
    "WasmBuffer.prototype.pull": {
      longname: "WasmBuffer#pull",
      kind: "function",
      description: "Pulls data from the WASM buffer back to JS (returns a copy).",
      params: [],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ]
    },
    "WasmBuffer.prototype.dispose": {
      longname: "WasmBuffer#dispose",
      kind: "function",
      description: "Disposes of the temporary buffer.",
      params: []
    },
    WasmRuntime: {
      longname: "WasmRuntime",
      kind: "class"
    },
    "WasmRuntime.prototype.init": {
      longname: "WasmRuntime#init",
      kind: "function",
      description: "Initializes the Go WASM environment.",
      params: [
        {
          name: "options",
          description: "Optional configuration.",
          type: {
            names: [
              "object"
            ]
          },
          optional: true
        },
        {
          name: "options.wasmUrl",
          description: "Path or URL to the wasm file.",
          type: {
            names: [
              "string"
            ]
          },
          optional: true,
          defaultvalue: "'./ndarray_plugin.wasm'"
        },
        {
          name: "options.execUrl",
          description: "Path to the wasm_exec.js file (Node.js only).",
          type: {
            names: [
              "string"
            ]
          },
          optional: true,
          defaultvalue: "'./wasm_exec.js'"
        },
        {
          name: "options.initialMemoryPages",
          description: "Initial memory size in 64KiB pages.",
          type: {
            names: [
              "number"
            ]
          },
          optional: true
        },
        {
          name: "options.maximumMemoryPages",
          description: "Maximum memory size in 64KiB pages.",
          type: {
            names: [
              "number"
            ]
          },
          optional: true
        },
        {
          name: "options.baseDir",
          type: {
            names: [
              "string"
            ]
          },
          optional: true,
          defaultvalue: "'.'"
        }
      ]
    },
    "WasmRuntime.prototype._getSuffix": {
      longname: "WasmRuntime#_getSuffix",
      kind: "function",
      description: "Helper method: Gets the suffix for Go export function names based on type.",
      params: []
    },
    "WasmRuntime.prototype.createBuffer": {
      longname: "WasmRuntime#createBuffer",
      kind: "function",
      description: "Quickly allocates a buffer.",
      params: [],
      returns: [
        {
          type: {
            names: [
              "WasmBuffer"
            ]
          }
        }
      ]
    },
    fromWasm: {
      longname: "fromWasm",
      kind: "function",
      description: "Static factory: Creates a new NDArray directly from WASM computation results.",
      params: [
        {
          name: "bridge",
          description: "The WasmBuffer instance containing the result from WASM.",
          type: {
            names: [
              "WasmBuffer"
            ]
          }
        },
        {
          name: "shape",
          description: "The shape of the new NDArray.",
          type: {
            names: [
              "Array.<number>"
            ]
          }
        },
        {
          name: "dtype",
          description: "The data type of the new NDArray. If not provided, uses bridge.dtype.",
          type: {
            names: [
              "string"
            ]
          },
          optional: true
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The new NDArray."
        }
      ]
    },
    NDWasm: {
      longname: "NDWasm",
      kind: "namespace"
    },
    "NDWasm.bind": {
      longname: "NDWasm.bind",
      kind: "function",
      description: "Binds a loaded WASM runtime to the bridge.",
      params: [
        {
          name: "runtime",
          type: {
            names: [
              "*"
            ]
          }
        }
      ]
    },
    "NDWasm.init": {
      longname: "NDWasm.init",
      kind: "function",
      description: "Init the NDWasm",
      params: [
        {
          name: "baseDir",
          type: {
            names: [
              "string"
            ]
          },
          optional: true,
          defaultvalue: "'.'"
        }
      ]
    },
    "NDWasm._compute": {
      longname: "NDWasm._compute",
      kind: "function",
      description: "Internal helper: executes a computation in WASM and manages memory."
    },
    NDWasmArray: {
      longname: "NDWasmArray",
      kind: "class",
      params: [
        {
          name: "buffer",
          description: "The WASM memory bridge (contains .ptr and .view).",
          type: {
            names: [
              "WasmBuffer"
            ]
          }
        },
        {
          name: "shape",
          description: "Dimensions of the array.",
          type: {
            names: [
              "Int32Array",
              "Array"
            ]
          }
        },
        {
          name: "dtype",
          description: "Data type (e.g., 'float64').",
          type: {
            names: [
              "string"
            ]
          }
        }
      ]
    },
    "NDWasmArray.prototype.NDWasmArray": {
      longname: "NDWasmArray#NDWasmArray",
      kind: "class",
      params: [
        {
          name: "buffer",
          description: "The WASM memory bridge (contains .ptr and .view).",
          type: {
            names: [
              "WasmBuffer"
            ]
          }
        },
        {
          name: "shape",
          description: "Dimensions of the array.",
          type: {
            names: [
              "Int32Array",
              "Array"
            ]
          }
        },
        {
          name: "dtype",
          description: "Data type (e.g., 'float64').",
          type: {
            names: [
              "string"
            ]
          }
        }
      ]
    },
    "NDWasmArray.fromArray": {
      longname: "NDWasmArray.fromArray",
      kind: "function",
      description: "Static factory: Creates a WASM-resident array.\r1. If source is an NDArray, it calls .push() to move it to WASM.\r2. If source is a JS Array, it allocates WASM memory and fills it directly \r   via recursive traversal to avoid intermediate flattening.",
      params: []
    },
    "NDWasmArray.prototype.pull": {
      longname: "NDWasmArray#pull",
      kind: "function",
      description: "Pulls data from WASM to a JS-managed NDArray.",
      params: [
        {
          name: "dispose",
          description: "Release WASM memory after pulling.",
          type: {
            names: [
              "boolean"
            ]
          },
          optional: true,
          defaultvalue: true
        }
      ]
    },
    "NDWasmArray.prototype.dispose": {
      longname: "NDWasmArray#dispose",
      kind: "function",
      description: "Manually releases WASM heap memory.",
      params: []
    },
    "NDWasmArray.prototype._prepareOperand": {
      longname: "NDWasmArray#_prepareOperand",
      kind: "function",
      description: "Internal helper to prepare operands for WASM operations.\rEnsures input is converted to NDWasmArray and tracks if it needs auto-disposal.",
      params: []
    },
    "NDWasmArray.prototype.matmul": {
      longname: "NDWasmArray#matmul",
      kind: "function",
      description: "Matrix Multiplication: C = this * other",
      params: [
        {
          name: "other",
          type: {
            names: [
              "NDWasmArray",
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDWasmArray"
            ]
          }
        }
      ]
    },
    "NDWasmArray.prototype.matmulBatch": {
      longname: "NDWasmArray#matmulBatch",
      kind: "function",
      description: "Batched Matrix Multiplication: C[i] = this[i] * other[i]",
      params: [
        {
          name: "other",
          type: {
            names: [
              "NDWasmArray",
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDWasmArray"
            ]
          }
        }
      ]
    },
    NDWasmAnalysis: {
      longname: "NDWasmAnalysis",
      kind: "namespace",
      description: "NDWasmAnalysis: Advanced Analysis, Stats, Spatial & Random\rHandles O(n log n) sorting, O(n*d^2) statistics, O(n^3) matrix properties,\rspatial clustering, and high-performance random sampling."
    },
    "NDWasmAnalysis.argsort": {
      longname: "NDWasmAnalysis.argsort",
      kind: "function",
      description: "Returns the indices that would sort an array.",
      params: [
        {
          name: "a",
          description: "Input array.",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Indices as Int32 NDArray."
        }
      ]
    },
    "NDWasmAnalysis.topk": {
      longname: "NDWasmAnalysis.topk",
      kind: "function",
      description: "Finds the largest or smallest K elements and their indices.\rComplexity: O(n log k)",
      params: [
        {
          name: "a",
          description: "Input array.",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "k",
          description: "Number of elements to return.",
          type: {
            names: [
              "number"
            ]
          }
        },
        {
          name: "largest",
          description: "If true, find max elements; else min.",
          type: {
            names: [
              "boolean"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "Object"
            ]
          }
        }
      ]
    },
    "NDWasmAnalysis.cov": {
      longname: "NDWasmAnalysis.cov",
      kind: "function",
      description: "Computes the covariance matrix for a dataset of shape [n_samples, n_features].",
      params: [
        {
          name: "a",
          description: "Data matrix.",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Covariance matrix of shape [d, d]."
        }
      ]
    },
    "NDWasmAnalysis.corr": {
      longname: "NDWasmAnalysis.corr",
      kind: "function",
      description: "Computes the Pearson correlation matrix for a dataset of shape [n_samples, n_features].",
      params: [
        {
          name: "a",
          description: "Data matrix.",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Correlation matrix of shape [d, d]."
        }
      ]
    },
    "NDWasmAnalysis.norm": {
      longname: "NDWasmAnalysis.norm",
      kind: "function",
      description: "Computes the matrix norm.",
      params: [
        {
          name: "a",
          description: "Input matrix.",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "type",
          description: "1 (The maximum absolute column sum), 2 (Frobenius), Infinity (The maximum absolute row sum)",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "number"
            ]
          },
          description: "The norm value."
        }
      ]
    },
    "NDWasmAnalysis.rank": {
      longname: "NDWasmAnalysis.rank",
      kind: "function",
      description: "Computes the rank of a matrix using SVD.",
      params: [
        {
          name: "a",
          description: "Input matrix.",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "tol",
          description: "Tolerance for singular values (0 for 1e-14).",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "number"
            ]
          },
          description: "Integer rank of the matrix."
        }
      ]
    },
    "NDWasmAnalysis.cond": {
      longname: "NDWasmAnalysis.cond",
      kind: "function",
      description: "estimates the reciprocal condition number of matrix a.",
      params: [
        {
          name: "a",
          description: "Input matrix.",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "norm",
          description: "norm: 1 (1-norm) or Infinity (Infinity norm).",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "number"
            ]
          },
          description: "result."
        }
      ]
    },
    "NDWasmAnalysis.eigenSym": {
      longname: "NDWasmAnalysis.eigenSym",
      kind: "function",
      description: "Eigenvalue decomposition for symmetric matrices.",
      params: [
        {
          name: "a",
          description: "Symmetric square matrix.",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "computeVectors",
          description: "Whether to return eigenvectors.",
          type: {
            names: [
              "boolean"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "Object"
            ]
          }
        }
      ]
    },
    "NDWasmAnalysis.pairwiseDist": {
      longname: "NDWasmAnalysis.pairwiseDist",
      kind: "function",
      description: "Computes pairwise Euclidean distances between two sets of vectors.",
      params: [
        {
          name: "a",
          description: "Matrix of shape [m, d].",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "b",
          description: "Matrix of shape [n, d].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Distance matrix of shape [m, n]."
        }
      ]
    },
    "NDWasmAnalysis.kmeans": {
      longname: "NDWasmAnalysis.kmeans",
      kind: "function",
      description: "Performs K-Means clustering in WASM memory.",
      params: [
        {
          name: "data",
          description: "Data of shape [n_samples, d_features].",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "k",
          description: "Number of clusters.",
          type: {
            names: [
              "number"
            ]
          }
        },
        {
          name: "maxIter",
          description: "Maximum iterations.",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "Object"
            ]
          }
        }
      ]
    },
    "NDWasmAnalysis.kronecker": {
      longname: "NDWasmAnalysis.kronecker",
      kind: "function",
      description: "Computes the Kronecker product C = A \u2297 B."
    },
    NDWasmBlas: {
      longname: "NDWasmBlas",
      kind: "namespace",
      description: "NDWasmBlas: BLAS (Basic Linear Algebra Subprograms)\rHandles O(n^2) and O(n^3) matrix-matrix and matrix-vector operations."
    },
    "NDWasmBlas.trace": {
      longname: "NDWasmBlas.trace",
      kind: "function",
      description: "Calculates the trace of a 2D square matrix (sum of diagonal elements).\rComplexity: O(n)",
      params: [
        {
          name: "a",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "number"
            ]
          },
          description: "The sum of the diagonal elements."
        }
      ]
    },
    "NDWasmBlas.matmul": {
      longname: "NDWasmBlas.matmul",
      kind: "function",
      description: "General Matrix Multiplication (GEMM): C = A * B.\rComplexity: O(m * n * k)",
      params: [
        {
          name: "a",
          description: "Left matrix of shape [m, n].",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "b",
          description: "Right matrix of shape [n, k].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Result matrix of shape [m, k]."
        }
      ]
    },
    "NDWasmBlas.matPow": {
      longname: "NDWasmBlas.matPow",
      kind: "function",
      description: "matPow computes A^k (Matrix Power).\rMatrix Functions (O(n^3))",
      params: [
        {
          name: "a",
          description: "Matrix of shape [n, n].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Result matrix of shape [n, n]."
        }
      ]
    },
    "NDWasmBlas.matmulBatch": {
      longname: "NDWasmBlas.matmulBatch",
      kind: "function",
      description: "Batched Matrix Multiplication: C[i] = A[i] * B[i].\rCommon in deep learning inference.\rComplexity: O(batch * m * n * k)",
      params: [
        {
          name: "a",
          description: "Batch of matrices of shape [batch, m, n].",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "b",
          description: "Batch of matrices of shape [batch, n, k].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Result batch of shape [batch, m, k]."
        }
      ]
    },
    "NDWasmBlas.syrk": {
      longname: "NDWasmBlas.syrk",
      kind: "function",
      description: "Symmetric Rank-K Update: C = alpha * A * A^T + beta * C.\rUsed for efficiently computing covariance matrices or Gram matrices.\rComplexity: O(n^2 * k)",
      params: [
        {
          name: "a",
          description: "Input matrix of shape [n, k].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Symmetric result matrix of shape [n, n]."
        }
      ]
    },
    "NDWasmBlas.trsm": {
      longname: "NDWasmBlas.trsm",
      kind: "function",
      description: "Triangular System Solver: Solves A * X = B for X, where A is a triangular matrix.\rComplexity: O(m^2 * n)",
      params: [
        {
          name: "a",
          description: "Triangular matrix of shape [m, m].",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "b",
          description: "Right-hand side matrix/vector of shape [m, n].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Solution matrix X of shape [m, n]."
        }
      ]
    },
    "NDWasmBlas.matVecMul": {
      longname: "NDWasmBlas.matVecMul",
      kind: "function",
      description: "Matrix-Vector Multiplication: y = A * x.\rComplexity: O(m * n)",
      params: [
        {
          name: "a",
          description: "Matrix of shape [m, n].",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "x",
          description: "Vector of shape [n].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Result vector of shape [m]."
        }
      ]
    },
    "NDWasmBlas.ger": {
      longname: "NDWasmBlas.ger",
      kind: "function",
      description: "Vector Outer Product (Rank-1 Update): A = x * y^T.\rComplexity: O(m * n)",
      params: [
        {
          name: "x",
          description: "Vector of shape [m].",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "y",
          description: "Vector of shape [n].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Result matrix of shape [m, n]."
        }
      ]
    },
    NDWasmDecomp: {
      longname: "NDWasmDecomp",
      kind: "namespace",
      description: "NDWasmDecomp: Decompositions & Solvers\rHandles O(n^3) matrix factorizations and linear system solutions."
    },
    "NDWasmDecomp.solve": {
      longname: "NDWasmDecomp.solve",
      kind: "function",
      description: "Solves a system of linear equations: Ax = B for x.\rComplexity: O(n^3)",
      params: [
        {
          name: "a",
          description: "Square coefficient matrix of shape [n, n].",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "b",
          description: "Right-hand side matrix or vector of shape [n, k].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Solution matrix x of shape [n, k]."
        }
      ]
    },
    "NDWasmDecomp.inv": {
      longname: "NDWasmDecomp.inv",
      kind: "function",
      description: "Computes the multiplicative inverse of a square matrix.\rComplexity: O(n^3)",
      params: [
        {
          name: "a",
          description: "Square matrix to invert of shape [n, n].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "The inverted matrix of shape [n, n]."
        }
      ]
    },
    "NDWasmDecomp.svd": {
      longname: "NDWasmDecomp.svd",
      kind: "function",
      description: "Computes the Singular Value Decomposition (SVD): A = U * S * V^T.\rComplexity: O(m * n * min(m, n))",
      params: [
        {
          name: "a",
          description: "Input matrix of shape [m, n].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "Object"
            ]
          }
        }
      ]
    },
    "NDWasmDecomp.qr": {
      longname: "NDWasmDecomp.qr",
      kind: "function",
      description: "Computes the QR decomposition: A = Q * R.\rComplexity: O(n^3)",
      params: [
        {
          name: "a",
          description: "Input matrix of shape [m, n].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "Object"
            ]
          }
        }
      ]
    },
    "NDWasmDecomp.cholesky": {
      longname: "NDWasmDecomp.cholesky",
      kind: "function",
      description: "Computes the Cholesky decomposition of a symmetric, positive-definite matrix: A = L * L^T.\rComplexity: O(n^3)",
      params: [
        {
          name: "a",
          description: "Symmetric positive-definite matrix of shape [n, n].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Lower triangular matrix L of shape [n, n]."
        }
      ]
    },
    "NDWasmDecomp.lu": {
      longname: "NDWasmDecomp.lu",
      kind: "function",
      description: "Computes the LU decomposition of a matrix: A = P * L * U.\rThe result is stored in-place in the output matrix.",
      params: [
        {
          name: "a",
          description: "Input matrix of shape [m, n].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "LU matrix of shape [m, n]."
        }
      ]
    },
    "NDWasmDecomp.pinv": {
      longname: "NDWasmDecomp.pinv",
      kind: "function",
      description: "Computes the Moore-Penrose pseudo-inverse of a matrix using SVD.\rComplexity: O(n^3)",
      params: [
        {
          name: "a",
          description: "Input matrix of shape [m, n].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Pseudo-inverted matrix of shape [n, m]."
        }
      ]
    },
    "NDWasmDecomp.det": {
      longname: "NDWasmDecomp.det",
      kind: "function",
      description: "Computes the determinant of a square matrix.\rComplexity: O(n^3)",
      params: [
        {
          name: "a",
          description: "Square matrix of shape [n, n].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "number"
            ]
          },
          description: "The determinant."
        }
      ]
    },
    "NDWasmDecomp.logDet": {
      longname: "NDWasmDecomp.logDet",
      kind: "function",
      description: "Computes the log-determinant for improved numerical stability.\rComplexity: O(n^3)",
      params: [
        {
          name: "a",
          description: "Square matrix of shape [n, n].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "Object"
            ]
          }
        }
      ]
    },
    "NDWasmDecomp.eigen": {
      longname: "NDWasmDecomp.eigen",
      kind: "function",
      description: "Computes the eigenvalues and eigenvectors of a general square matrix.\rEigenvalues and eigenvectors can be complex numbers.\rThe results are returned in an interleaved format where each complex number (a + bi)\ris represented by two consecutive float64 values (a, b).",
      params: [
        {
          name: "a",
          description: "Input square matrix of shape `[n, n]`. Must be float64.",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "Object"
            ]
          },
          description: "An object containing:\r  - `values`: Complex eigenvalues as an NDArray of shape `[n, 2]`, where `[i, 0]` is real and `[i, 1]` is imaginary.\r  - `vectors`: Complex right eigenvectors as an NDArray of shape `[n, n, 2]`, where `[i, j, 0]` is real and `[i, j, 1]` is imaginary.\r             (Note: these are column vectors, such that `A * v = lambda * v`)."
        }
      ]
    },
    NDWasmImage: {
      longname: "NDWasmImage",
      kind: "namespace",
      description: "Provides WebAssembly-powered functions for image processing.\nThis module allows for efficient conversion between image binary data and NDArrays."
    },
    "NDWasmImage.decode": {
      longname: "NDWasmImage.decode",
      kind: "function",
      description: "Decodes a binary image into an NDArray.\nSupports common formats like PNG, JPEG, GIF, and WebP.\nThe resulting NDArray will have a shape of [height, width, 4] and a 'uint8c' dtype,\nrepresenting RGBA channels. This provides a consistent starting point for image manipulation.",
      params: [
        {
          name: "imageBytes",
          description: "The raw binary data of the image file.",
          type: {
            names: [
              "Uint8Array"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray",
              "null"
            ]
          },
          description: "A 3D NDArray representing the image, or null if decoding fails."
        }
      ],
      examples: [
        "const imageBlob = await fetch('./my-image.png').then(res => res.blob());\nconst imageBytes = new Uint8Array(await imageBlob.arrayBuffer());\nconst imageArray = NDWasmImage.decode(imageBytes);\n// imageArray.shape is [height, width, 4]"
      ]
    },
    "NDWasmImage.encode": {
      longname: "NDWasmImage.encode",
      kind: "function",
      description: "Encodes an NDArray into a binary image format (PNG or JPEG).",
      params: [
        {
          name: "ndarray",
          description: "The input array.\n  Supported dtypes: 'uint8', 'uint8c', 'float32', 'float64'. Float values should be in the range [0, 1].\n  Supported shapes: [h, w] (grayscale), [h, w, 1] (grayscale), [h, w, 3] (RGB), or [h, w, 4] (RGBA).",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "options",
          description: "Encoding options.",
          type: {
            names: [
              "object"
            ]
          },
          optional: true,
          defaultvalue: "{}"
        },
        {
          name: "options.format",
          description: "The target format: 'png' or 'jpeg'. It is recommended to use the `encodePng` or `encodeJpeg` helpers instead.",
          type: {
            names: [
              "string"
            ]
          },
          optional: true,
          defaultvalue: "'png'"
        },
        {
          name: "options.quality",
          description: "The quality for JPEG encoding (1-100). Ignored for PNG.",
          type: {
            names: [
              "number"
            ]
          },
          optional: true,
          defaultvalue: 90
        }
      ],
      returns: [
        {
          type: {
            names: [
              "Uint8Array",
              "null"
            ]
          },
          description: "A Uint8Array containing the binary data of the encoded image, or null on failure."
        }
      ],
      examples: [
        "// Encode a 3-channel float array to a high-quality JPEG using the main function\nconst floatArr = NDArray.random([100, 150, 3]);\nconst jpegBytes = NDWasmImage.encode(floatArr, { format: 'jpeg', quality: 95 });"
      ]
    },
    "NDWasmImage.encodePng": {
      longname: "NDWasmImage.encodePng",
      kind: "function",
      description: "Encodes an NDArray into a PNG image.\nThis is a helper function that calls `encode` with `format: 'png'`.",
      params: [
        {
          name: "ndarray",
          description: "The input array. See `encode` for supported shapes and dtypes.",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "Uint8Array",
              "null"
            ]
          },
          description: "A Uint8Array containing the binary data of the PNG image."
        }
      ],
      examples: [
        "const grayArr = ndarray.zeros([16, 16]);\nconst pngBytes = NDWasmImage.encodePng(grayArr);"
      ]
    },
    "NDWasmImage.encodeJpeg": {
      longname: "NDWasmImage.encodeJpeg",
      kind: "function",
      description: "Encodes an NDArray into a JPEG image.\nThis is a helper function that calls `encode` with `format: 'jpeg'`.",
      params: [
        {
          name: "ndarray",
          description: "The input array. See `encode` for supported shapes and dtypes.",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "options",
          description: "Encoding options.",
          type: {
            names: [
              "object"
            ]
          },
          optional: true,
          defaultvalue: "{}"
        },
        {
          name: "options.quality",
          description: "The quality for JPEG encoding (1-100).",
          type: {
            names: [
              "number"
            ]
          },
          optional: true,
          defaultvalue: 90
        }
      ],
      returns: [
        {
          type: {
            names: [
              "Uint8Array",
              "null"
            ]
          },
          description: "A Uint8Array containing the binary data of the JPEG image."
        }
      ],
      examples: [
        "const floatArr = NDArray.random([20, 20, 3]);\nconst jpegBytes = NDWasmImage.encodeJpeg(floatArr, { quality: 85 });"
      ]
    },
    "NDWasmImage.convertUint8ArrrayToDataurl": {
      longname: "NDWasmImage.convertUint8ArrrayToDataurl",
      kind: "function",
      description: "Converts a Uint8Array of binary data into a Base64 Data URL.\nThis is a utility function that runs purely in JavaScript.",
      params: [
        {
          name: "uint8array",
          description: "The byte array to convert.",
          type: {
            names: [
              "Uint8Array"
            ]
          }
        },
        {
          name: "mimeType",
          description: "The MIME type for the Data URL (e.g., 'image/jpeg').",
          type: {
            names: [
              "string"
            ]
          },
          optional: true,
          defaultvalue: "'image/png'"
        }
      ],
      returns: [
        {
          type: {
            names: [
              "string"
            ]
          },
          description: "The complete Data URL string."
        }
      ],
      examples: [
        "const pngBytes = NDWasmImage.encodePng(myNdarray);\nconst dataUrl = NDWasmImage.convertUint8ArrrayToDataurl(pngBytes, 'image/png');\n// <img src={dataUrl} />"
      ]
    },
    OPTIMIZE_STATUS_MAP: {
      longname: "OPTIMIZE_STATUS_MAP",
      kind: "constant",
      description: "Maps status codes from gonum/optimize to human-readable messages.",
      params: []
    },
    NDWasmOptimize: {
      longname: "NDWasmOptimize",
      kind: "namespace",
      description: "Namespace for Optimization functions using Go WASM."
    },
    "NDWasmOptimize.linprog": {
      longname: "NDWasmOptimize.linprog",
      kind: "function",
      description: "Provides Optimization capabilities by wrapping Go WASM functions.\nminimize c\u1D40 * x\ns.t      G * x <= h\n	        A * x = b\n         lower <= x <= upper",
      params: [
        {
          name: "c",
          description: "Coefficient vector for the objective function (1D NDArray of float64).",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "G",
          description: "Coefficient matrix for inequality constraints (2D NDArray of float64).",
          type: {
            names: [
              "NDArray",
              "null"
            ]
          }
        },
        {
          name: "h",
          description: "Right-hand side vector for inequality constraints (1D NDArray of float64).",
          type: {
            names: [
              "NDArray",
              "null"
            ]
          }
        },
        {
          name: "A",
          description: "Coefficient matrix for equality constraints (2D NDArray of float64).",
          type: {
            names: [
              "NDArray",
              "null"
            ]
          }
        },
        {
          name: "b",
          description: "Right-hand side vector for equality constraints (1D NDArray of float64).",
          type: {
            names: [
              "NDArray",
              "null"
            ]
          }
        },
        {
          name: "bounds",
          description: "Optional variable bounds as an array of [lower, upper] pairs. Use null for unbounded. [0, null] for all for default.",
          type: {
            names: [
              "Array"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "Object"
            ]
          },
          description: "- The optimization result."
        }
      ]
    },
    "NDWasmOptimize.linearRegression": {
      longname: "NDWasmOptimize.linearRegression",
      kind: "function",
      description: "Fits a simple linear regression model: Y = alpha + beta*X.",
      params: [
        {
          name: "x",
          description: "The independent variable (1D NDArray of float64).",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "y",
          description: "The dependent variable (1D NDArray of float64).",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "Object"
            ]
          },
          description: "- An object containing the intercept (alpha) and slope (beta) of the fitted line."
        }
      ]
    },
    "NDWasmOptimize.minimize": {
      longname: "NDWasmOptimize.minimize",
      kind: "function",
      description: "Finds the minimum of a scalar function of one or more variables using an L-BFGS optimizer.",
      params: [
        {
          name: "func",
          description: "The objective function to be minimized. It must take a 1D `Float64Array` `x` (current point) and return a single number (the function value at `x`).",
          type: {
            names: [
              "function"
            ]
          }
        },
        {
          name: "x0",
          description: "The initial guess for the optimization (1D NDArray of float64).",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "options",
          description: "Optional parameters.",
          type: {
            names: [
              "Object"
            ]
          },
          optional: true
        },
        {
          name: "options.grad",
          description: "The gradient of the objective function. Must take `x` (a 1D `Float64Array`) and write the result into the second argument `grad_out` (a 1D `Float64Array`). This function should *not* return a value.",
          type: {
            names: [
              "function"
            ]
          },
          optional: true
        }
      ],
      returns: [
        {
          type: {
            names: [
              "Object"
            ]
          },
          description: "The optimization result."
        }
      ]
    },
    NDWasmSignal: {
      longname: "NDWasmSignal",
      kind: "namespace",
      description: "NDWasmSignal: Signal Processing & Transformations\rHandles O(n log n) frequency domain transforms and O(n^2 * k^2) spatial filters."
    },
    "NDWasmSignal.fft": {
      longname: "NDWasmSignal.fft",
      kind: "function",
      description: "1D Complex-to-Complex Fast Fourier Transform.\rThe input array must have its last dimension of size 2 (real and imaginary parts).\rThe transform is performed in-place.\rComplexity: O(n log n)",
      params: [
        {
          name: "a",
          description: "Complex input signal, with shape [..., 2].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "- Complex result, with the same shape as input."
        }
      ]
    },
    "NDWasmSignal.ifft": {
      longname: "NDWasmSignal.ifft",
      kind: "function",
      description: "1D Inverse Complex-to-Complex Fast Fourier Transform.\rThe input array must have its last dimension of size 2 (real and imaginary parts).\rThe transform is performed in-place.\rComplexity: O(n log n)",
      params: [
        {
          name: "a",
          description: "Complex frequency-domain signal, with shape [..., 2].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "- Complex time-domain result, with the same shape as input."
        }
      ]
    },
    "NDWasmSignal.rfft": {
      longname: "NDWasmSignal.rfft",
      kind: "function",
      description: "1D Real-to-Complex Fast Fourier Transform (Optimized for real input).\rThe output is a complex array with shape [n/2 + 1, 2].\rComplexity: O(n log n)",
      params: [
        {
          name: "a",
          description: "Real input signal.",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "- Complex result of shape [n/2 + 1, 2]."
        }
      ]
    },
    "NDWasmSignal.rifft": {
      longname: "NDWasmSignal.rifft",
      kind: "function",
      description: "1D Complex-to-Real Inverse Fast Fourier Transform.\rThe input must be a complex array of shape [k, 2], where k is n/2 + 1.",
      params: [
        {
          name: "a",
          description: "Complex frequency signal of shape [n/2 + 1, 2].",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "n",
          description: "Length of the original real signal.",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Real-valued time domain signal."
        }
      ]
    },
    "NDWasmSignal.fft2": {
      longname: "NDWasmSignal.fft2",
      kind: "function",
      description: "2D Complex-to-Complex Fast Fourier Transform.\rThe input array must be 3D with shape [rows, cols, 2].\rThe transform is performed in-place.\rComplexity: O(rows * cols * log(rows * cols))",
      params: [
        {
          name: "a",
          description: "2D Complex input signal, with shape [rows, cols, 2].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "- 2D Complex result, with the same shape as input."
        }
      ]
    },
    "NDWasmSignal.ifft2": {
      longname: "NDWasmSignal.ifft2",
      kind: "function",
      description: "2D Inverse Complex-to-Complex Fast Fourier Transform.\rThe input array must be 3D with shape [rows, cols, 2].\rThe transform is performed in-place.",
      params: [
        {
          name: "a",
          description: "2D Complex frequency-domain signal, with shape [rows, cols, 2].",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "- 2D Complex time-domain result, with the same shape as input."
        }
      ]
    },
    "NDWasmSignal.dct": {
      longname: "NDWasmSignal.dct",
      kind: "function",
      description: "1D Discrete Cosine Transform (Type II).\rComplexity: O(n log n)",
      params: [
        {
          name: "a",
          description: "Input signal.",
          type: {
            names: [
              "NDArray"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "DCT result of same shape."
        }
      ]
    },
    "NDWasmSignal.conv2d": {
      longname: "NDWasmSignal.conv2d",
      kind: "function",
      description: "2D Spatial Convolution.\rComplexity: O(img_h * img_w * kernel_h * kernel_w)",
      params: [
        {
          name: "img",
          description: "2D Image/Matrix.",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "kernel",
          description: "2D Filter kernel.",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "stride",
          description: "Step size (default 1).",
          type: {
            names: [
              "number"
            ]
          }
        },
        {
          name: "padding",
          description: "Zero-padding size (default 0).",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Convolved result."
        }
      ]
    },
    "NDWasmSignal.correlate2d": {
      longname: "NDWasmSignal.correlate2d",
      kind: "function",
      description: "2D Spatial Cross-Correlation.\rSimilar to convolution but without flipping the kernel.\rComplexity: O(img_h * img_w * kernel_h * kernel_w)",
      params: [
        {
          name: "img",
          description: "2D Image/Matrix.",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "kernel",
          description: "2D Filter kernel.",
          type: {
            names: [
              "NDArray"
            ]
          }
        },
        {
          name: "stride",
          description: "Step size.",
          type: {
            names: [
              "number"
            ]
          }
        },
        {
          name: "padding",
          description: "Zero-padding size.",
          type: {
            names: [
              "number"
            ]
          }
        }
      ],
      returns: [
        {
          type: {
            names: [
              "NDArray"
            ]
          },
          description: "Cross-correlated result."
        }
      ]
    }
  };

  // src/help.js
  function formatDoc(doc) {
    let output = `[ ${doc.longname} ]

`;
    if (doc.kind === "class") {
      output = `[ class ${doc.longname} ]

`;
    }
    output += doc.description || "No description available.";
    output += "\n\n";
    if (doc.params && doc.params.length > 0) {
      output += "Parameters:\n";
      for (const param of doc.params) {
        const type = param.type ? `{${param.type.names.join("|")}}` : "";
        const optional = param.optional ? "[optional]" : "";
        const defaultValue = param.defaultvalue !== void 0 ? `(default: ${JSON.stringify(param.defaultvalue)})` : "";
        output += `  - ${param.name} ${type} ${optional} ${defaultValue}
`;
        if (param.description) {
          output += `    ${param.description || ""}
`;
        }
      }
      output += "\n";
    }
    if (doc.returns && doc.returns.length > 0) {
      output += "Returns:\n";
      for (const ret of doc.returns) {
        const type = ret.type ? `{${ret.type.names.join("|")}}` : "";
        output += `  ${type} - ${ret.description}
`;
      }
      output += "\n";
    }
    if (doc.examples) {
      output += "Examples:\n";
      for (const example of doc.examples) {
        output += "```javascript\n";
        output += example + "\n";
        output += "```\n";
      }
    }
    return output;
  }
  function formatDocHTML(doc) {
    let output = `<div class="ndarray-help-container">`;
    const kind = doc.kind === "class" ? `<span class="ndarray-help-kind">${doc.kind}</span> ` : "";
    output += `<h3 class="ndarray-help-longname">[ ${kind}${doc.longname} ]</h3>`;
    output += `<p class="ndarray-help-description">${doc.description || "No description available."}</p>`;
    if (doc.params && doc.params.length > 0) {
      output += `<div class="ndarray-help-parameters"><h4>Parameters:</h4><ul>`;
      for (const param of doc.params) {
        const type = param.type ? `<span class="ndarray-help-param-type">{${param.type.names.join("|")}}</span>` : "";
        const optional = param.optional ? `<span class="ndarray-help-param-attrs">[optional]</span>` : "";
        const defaultValue = param.defaultvalue !== void 0 ? `<span class="ndarray-help-param-attrs">(default: ${JSON.stringify(param.defaultvalue)})</span>` : "";
        output += `<li>
                <strong class="ndarray-help-param-name">${param.name}</strong>
                ${type} ${optional} ${defaultValue}
                ${param.description ? `<p class="ndarray-help-param-desc">${param.description}</p>` : ""}
            </li>`;
      }
      output += `</ul></div>`;
    }
    if (doc.returns && doc.returns.length > 0) {
      output += `<div class="ndarray-help-returns"><h4>Returns:</h4>`;
      for (const ret of doc.returns) {
        const type = ret.type ? `<span class="ndarray-help-return-type">{${ret.type.names.join("|")}}</span>` : "";
        output += `<p>${type} - ${ret.description || ""}</p>`;
      }
      output += `</div>`;
    }
    if (doc.examples) {
      output += `<div class="ndarray-help-examples"><h4>Examples:</h4>`;
      for (const example of doc.examples) {
        output += `<pre><code class="language-javascript">${example}</code></pre>`;
      }
      output += `</div>`;
    }
    output += `</div>`;
    return output;
  }
  var help = {
    /**
     * A WeakMap that maps live function/class objects back to their documentation names.
     * This map is populated by the registration logic in `index.js`.
     * @type {WeakMap<object, string>}
     */
    helpmap: /* @__PURE__ */ new WeakMap(),
    /**
     * Returns help doc object
     * @param {string | object} target - The name of the function/class, or the object itself.
     * @returns {object | undefined} The documentation obj, or undefined if not found.
     */
    getHelpDoc(target) {
      let name;
      if (typeof target === "string") {
        name = target;
      } else if (typeof target === "object" && target !== null || typeof target === "function") {
        name = help.helpmap.get(target);
      }
      if (!name) {
        return;
      }
      const doc = docs_default[name];
      return doc;
    },
    /**
     * Provides help for a given class/function name or a live object.
     * Logs the formatted documentation to the console.
     * @param {string | object} target - The name of the function/class, or the object itself.
     */
    helpdoc(target) {
      const doc = help.getHelpDoc(target);
      if (!doc) {
        const targetName = typeof target === "string" ? target : "the provided object";
        console.log(`No documentation found for '${targetName}'.`);
        if (typeof target !== "string") {
          console.log("Ensure the object is part of the ndarray library and the help system is initialized correctly.");
        }
        console.log("\nAvailable top-level names:\n" + Object.keys(docs_default).filter((k) => !k.includes(".")).sort().join("\n"));
        return;
      }
      return formatDoc(doc);
    },
    /**
     * Returns help content as a styled HTML string for a given class/function name or a live object.
     * @param {string | object} target - The name of the function/class, or the object itself.
     * @returns {string | undefined} The documentation as an HTML string, or undefined if not found.
     */
    helphtml(target) {
      const doc = help.getHelpDoc(target);
      if (!doc) {
        return;
      }
      return formatDocHTML(doc);
    }
  };

  // src/ndarray.js
  function registerAll() {
    const rootObjects = {
      NDArray,
      NDWasmArray,
      ...ndwasm_exports,
      NDProb,
      NDWasmDecomp,
      NDWasmAnalysis,
      NDWasmBlas,
      NDWasmSignal,
      NDWasmImage,
      NDWasmOptimize,
      ...ndarray_factory_exports
    };
    for (const name in docs_default) {
      const parts = name.split(".");
      const rootName = parts[0];
      if (!rootObjects[rootName]) continue;
      let obj = rootObjects[rootName];
      for (let i = 1; i < parts.length; i++) {
        obj = obj[parts[i]];
        if (!obj) break;
      }
      if (obj) {
        help.helpmap.set(obj, name);
      }
    }
  }
  registerAll();
  var random = NDProb;
  var image = NDWasmImage;
  var optimize = NDWasmOptimize;
  var decomp2 = NDWasmDecomp;
  var analysis2 = NDWasmAnalysis;
  var blas2 = NDWasmBlas;
  var signal2 = NDWasmSignal;
  function init(baseDir = ".") {
    return NDWasm.init(baseDir);
  }
  var ndarray_default = NDArray;
  return __toCommonJS(ndarray_exports);
})();
