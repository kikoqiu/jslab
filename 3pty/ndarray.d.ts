declare module "help" {
    export namespace help {
        let helpmap: WeakMap<object, string>;
        /**
         * Returns help doc object
         * @param {string | object} target - The name of the function/class, or the object itself.
         * @returns {object | undefined} The documentation obj, or undefined if not found.
         */
        function getHelpDoc(target: string | object): object | undefined;
        /**
         * Provides help for a given class/function name or a live object.
         * Logs the formatted documentation to the console.
         * @param {string | object} target - The name of the function/class, or the object itself.
         */
        function helpdoc(target: string | object): string | undefined;
        /**
         * Returns help content as a styled HTML string for a given class/function name or a live object.
         * @param {string | object} target - The name of the function/class, or the object itself.
         * @returns {string | undefined} The documentation as an HTML string, or undefined if not found.
         */
        function helphtml(target: string | object): string | undefined;
    }
}
declare module "ndarray_prob" {
    export namespace NDProb {
        /**
         * Internal helper to get a Float64 between [0, 1) using crypto.
         * Generates high-quality uniform random floats.
         * @private
         */
        function _cryptoUniform01(size: any): Float64Array<any>;
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
        function random(shape: any[], low?: number, high?: number, dtype?: string): NDArray;
        /**
         * Normal (Gaussian) distribution using Box-Muller transform.
         * @param {Array} shape - Dimensions of the output array.
         * @param {number} [mean=0] - Mean of the distribution.
         * @param {number} [std=1] - Standard deviation.
         * @param {string} [dtype='float64'] - Data type.
         * @memberof NDProb
         * @returns {NDArray}
         */
        function normal(shape: any[], mean?: number, std?: number, dtype?: string): NDArray;
        /**
         * Bernoulli distribution (0 or 1 with probability p).
         * @param {Array} shape
         * @param {number} [p=0.5] - Probability of success (1).
         * @param {string} [dtype='int32']
         * @memberof NDProb
         * @returns {NDArray}
         */
        function bernoulli(shape: any[], p?: number, dtype?: string): NDArray;
        /**
         * Exponential distribution: f(x; λ) = λe^(-λx).
         * Inverse transform sampling: x = -ln(1-u) / λ.
         * @param {Array} shape
         * @param {number} [lambda=1.0] - Rate parameter.
         * @memberof NDProb
         * @returns {NDArray}
         */
        function exponential(shape: any[], lambda?: number, dtype?: string): NDArray;
        /**
         * Poisson distribution using Knuth's algorithm.
         * Note: For very large lambda, this becomes slow; but for most use cases it's fine.
         * @param {Array} shape
         * @param {number} [lambda=1.0] - Mean of the distribution.
         * @memberof NDProb
         * @returns {NDArray}
         */
        function poisson(shape: any[], lambda?: number, dtype?: string): NDArray;
        /**
         * @private
         * @internal
         */
        function _cast(data: any, dtype: any): any;
    }
    import { NDArray } from "ndarray_core";
}
declare module "ndwasm_blas" {
    export namespace NDWasmBlas {
        /**
         * Calculates the trace of a 2D square matrix (sum of diagonal elements).
         * Complexity: O(n)
         * @param {NDArray} a
         * @returns {number} The sum of the diagonal elements.
         * @throws {Error} If the array is not 2D or not a square matrix.
         */
        function trace(a: NDArray): number;
        /**
         * General Matrix Multiplication (GEMM): C = A * B.
         * Complexity: O(m * n * k)
         * @memberof NDWasmBlas
         * @param {NDArray} a - Left matrix of shape [m, n].
         * @param {NDArray} b - Right matrix of shape [n, k].
         * @returns {NDArray} Result matrix of shape [m, k].
         */
        function matMul(a: NDArray, b: NDArray): NDArray;
        /**
         * matPow computes A^k (Matrix Power).
         * Matrix Functions (O(n^3))
         * @memberof NDWasmBlas
         * @param {NDArray} a - Matrix of shape [n, n].
         * @returns {NDArray} Result matrix of shape [n, n].
         */
        function matPow(a: NDArray, k: any): NDArray;
        /**
         * Batched Matrix Multiplication: C[i] = A[i] * B[i].
         * Common in deep learning inference.
         * Complexity: O(batch * m * n * k)
         * @memberof NDWasmBlas
         * @param {NDArray} a - Batch of matrices of shape [batch, m, n].
         * @param {NDArray} b - Batch of matrices of shape [batch, n, k].
         * @returns {NDArray} Result batch of shape [batch, m, k].
         */
        function matMulBatch(a: NDArray, b: NDArray): NDArray;
        /**
         * Symmetric Rank-K Update: C = alpha * A * A^T + beta * C.
         * Used for efficiently computing covariance matrices or Gram matrices.
         * Complexity: O(n^2 * k)
         * @memberof NDWasmBlas
         * @param {NDArray} a - Input matrix of shape [n, k].
         * @returns {NDArray} Symmetric result matrix of shape [n, n].
         */
        function syrk(a: NDArray): NDArray;
        /**
         * Triangular System Solver: Solves A * X = B for X, where A is a triangular matrix.
         * Complexity: O(m^2 * n)
         * @memberof NDWasmBlas
         * @param {NDArray} a - Triangular matrix of shape [m, m].
         * @param {NDArray} b - Right-hand side matrix/vector of shape [m, n].
         * @returns {NDArray} Solution matrix X of shape [m, n].
         */
        function trsm(a: NDArray, b: NDArray, lower?: boolean): NDArray;
        /**
         * Matrix-Vector Multiplication: y = A * x.
         * Complexity: O(m * n)
         * @memberof NDWasmBlas
         * @param {NDArray} a - Matrix of shape [m, n].
         * @param {NDArray} x - Vector of shape [n].
         * @returns {NDArray} Result vector of shape [m].
         */
        function matVecMul(a: NDArray, x: NDArray): NDArray;
        /**
         * Vector Outer Product (Rank-1 Update): A = x * y^T.
         * Complexity: O(m * n)
         * @memberof NDWasmBlas
         * @param {NDArray} x - Vector of shape [m].
         * @param {NDArray} y - Vector of shape [n].
         * @returns {NDArray} Result matrix of shape [m, n].
         */
        function ger(x: NDArray, y: NDArray): NDArray;
    }
    import { NDArray } from "ndarray_core";
}
declare module "ndwasm_decomp" {
    export namespace NDWasmDecomp {
        /**
         * Solves a system of linear equations: Ax = B for x.
         * Complexity: O(n^3)
         * @memberof NDWasmDecomp
         * @param {NDArray} a - Square coefficient matrix of shape [n, n].
         * @param {NDArray} b - Right-hand side matrix or vector of shape [n, k].
         * @returns {NDArray} Solution matrix x of shape [n, k].
         */
        function solve(a: NDArray, b: NDArray): NDArray;
        /**
         * Computes the multiplicative inverse of a square matrix.
         * Complexity: O(n^3)
         * @memberof NDWasmDecomp
         * @param {NDArray} a - Square matrix to invert of shape [n, n].
         * @returns {NDArray} The inverted matrix of shape [n, n].
         */
        function inv(a: NDArray): NDArray;
        /**
         * Computes the Singular Value Decomposition (SVD): A = U * S * V^T.
         * Complexity: O(m * n * min(m, n))
         * @memberof NDWasmDecomp
         * @param {NDArray} a - Input matrix of shape [m, n].
         * @returns {{u: NDArray, s: NDArray, v: NDArray}}
         */
        function svd(a: NDArray): {
            u: NDArray;
            s: NDArray;
            v: NDArray;
        };
        /**
         * Computes the QR decomposition: A = Q * R.
         * Complexity: O(n^3)
         * @memberof NDWasmDecomp
         * @param {NDArray} a - Input matrix of shape [m, n].
         * @returns {{q: NDArray, r: NDArray}}
         */
        function qr(a: NDArray): {
            q: NDArray;
            r: NDArray;
        };
        /**
         * Computes the Cholesky decomposition of a symmetric, positive-definite matrix: A = L * L^T.
         * Complexity: O(n^3)
         * @memberof NDWasmDecomp
         * @param {NDArray} a - Symmetric positive-definite matrix of shape [n, n].
         * @returns {NDArray} Lower triangular matrix L of shape [n, n].
         */
        function cholesky(a: NDArray): NDArray;
        /**
         * Computes the LU decomposition of a matrix: A = P * L * U.
         * The result is stored in-place in the output matrix.
         * @memberof NDWasmDecomp
         * @param {NDArray} a - Input matrix of shape [m, n].
         * @returns {NDArray} LU matrix of shape [m, n].
         */
        function lu(a: NDArray): NDArray;
        /**
         * Computes the Moore-Penrose pseudo-inverse of a matrix using SVD.
         * Complexity: O(n^3)
         * @memberof NDWasmDecomp
         * @param {NDArray} a - Input matrix of shape [m, n].
         * @returns {NDArray} Pseudo-inverted matrix of shape [n, m].
         */
        function pinv(a: NDArray): NDArray;
        /**
         * Computes the determinant of a square matrix.
         * Complexity: O(n^3)
         * @memberof NDWasmDecomp
         * @param {NDArray} a - Square matrix of shape [n, n].
         * @returns {number} The determinant.
         */
        function det(a: NDArray): number;
        /**
         * Computes the log-determinant for improved numerical stability.
         * Complexity: O(n^3)
         * @memberof NDWasmDecomp
         * @param {NDArray} a - Square matrix of shape [n, n].
         * @returns {{sign: number, logAbsDet: number}}
         */
        function logDet(a: NDArray): {
            sign: number;
            logAbsDet: number;
        };
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
        function eigen(a: NDArray): {
            values: NDArray;
            vectors: NDArray;
        };
    }
    import { NDArray } from "ndarray_core";
}
declare module "ndwasm_signal" {
    export namespace NDWasmSignal {
        /**
         * 1D Complex-to-Complex Fast Fourier Transform.
         * The input array must have its last dimension of size 2 (real and imaginary parts).
         * The transform is performed in-place.
         * Complexity: O(n log n)
         * @memberof NDWasmSignal
         * @param {NDArray} a - Complex input signal, with shape [..., 2].
         * @returns {NDArray} - Complex result, with the same shape as input.
         */
        function fft(a: NDArray): NDArray;
        /**
         * 1D Inverse Complex-to-Complex Fast Fourier Transform.
         * The input array must have its last dimension of size 2 (real and imaginary parts).
         * The transform is performed in-place.
         * Complexity: O(n log n)
         * @memberof NDWasmSignal
         * @param {NDArray} a - Complex frequency-domain signal, with shape [..., 2].
         * @returns {NDArray} - Complex time-domain result, with the same shape as input.
         */
        function ifft(a: NDArray): NDArray;
        /**
         * 1D Real-to-Complex Fast Fourier Transform (Optimized for real input).
         * The output is a complex array with shape [n/2 + 1, 2].
         * Complexity: O(n log n)
         * @memberof NDWasmSignal
         * @param {NDArray} a - Real input signal.
         * @returns {NDArray} - Complex result of shape [n/2 + 1, 2].
         */
        function rfft(a: NDArray): NDArray;
        /**
         * 1D Complex-to-Real Inverse Fast Fourier Transform.
         * The input must be a complex array of shape [k, 2], where k is n/2 + 1.
         * @memberof NDWasmSignal
         * @param {NDArray} a - Complex frequency signal of shape [n/2 + 1, 2].
         * @param {number} n - Length of the original real signal.
         * @returns {NDArray} Real-valued time domain signal.
         */
        function rifft(a: NDArray, n: number): NDArray;
        /**
         * 2D Complex-to-Complex Fast Fourier Transform.
         * The input array must be 3D with shape [rows, cols, 2].
         * The transform is performed in-place.
         * Complexity: O(rows * cols * log(rows * cols))
         * @memberof NDWasmSignal
         * @param {NDArray} a - 2D Complex input signal, with shape [rows, cols, 2].
         * @returns {NDArray} - 2D Complex result, with the same shape as input.
         */
        function fft2(a: NDArray): NDArray;
        /**
         * 2D Inverse Complex-to-Complex Fast Fourier Transform.
         * The input array must be 3D with shape [rows, cols, 2].
         * The transform is performed in-place.
         * @memberof NDWasmSignal
         * @param {NDArray} a - 2D Complex frequency-domain signal, with shape [rows, cols, 2].
         * @returns {NDArray} - 2D Complex time-domain result, with the same shape as input.
         */
        function ifft2(a: NDArray): NDArray;
        /**
         * 1D Discrete Cosine Transform (Type II).
         * Complexity: O(n log n)
         * @memberof NDWasmSignal
         * @param {NDArray} a - Input signal.
         * @returns {NDArray} DCT result of same shape.
         */
        function dct(a: NDArray): NDArray;
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
        function conv2d(img: NDArray, kernel: NDArray, stride?: number, padding?: number): NDArray;
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
        function correlate2d(img: NDArray, kernel: NDArray, stride?: number, padding?: number): NDArray;
    }
    import { NDArray } from "ndarray_core";
}
declare module "ndwasm_analysis" {
    export namespace NDWasmAnalysis {
        /**
         * Returns the indices that would sort an array.
         * @memberof NDWasmAnalysis
         * @param {NDArray} a - Input array.
         * @returns {NDArray} Indices as Int32 NDArray.
         */
        function argsort(a: NDArray): NDArray;
        /**
         * Finds the largest or smallest K elements and their indices.
         * Complexity: O(n log k)
         * @memberof NDWasmAnalysis
         * @param {NDArray} a - Input array.
         * @param {number} k - Number of elements to return.
         * @param {boolean} largest - If true, find max elements; else min.
         * @returns {{values: NDArray, indices: NDArray}}
         */
        function topk(a: NDArray, k: number, largest?: boolean): {
            values: NDArray;
            indices: NDArray;
        };
        /**
         * Computes the covariance matrix for a dataset of shape [n_samples, n_features].
         * @memberof NDWasmAnalysis
         * @param {NDArray} a - Data matrix.
         * @returns {NDArray} Covariance matrix of shape [d, d].
         */
        function cov(a: NDArray): NDArray;
        /**
         * Computes the Pearson correlation matrix for a dataset of shape [n_samples, n_features].
         * @memberof NDWasmAnalysis
         * @param {NDArray} a - Data matrix.
         * @returns {NDArray} Correlation matrix of shape [d, d].
         */
        function corr(a: NDArray): NDArray;
        /**
         * Computes the matrix norm.
         * @memberof NDWasmAnalysis
         * @param {NDArray} a - Input matrix.
         * @param {number} type - 1 (The maximum absolute column sum), 2 (Frobenius), Infinity (The maximum absolute row sum)
         * @returns {number} The norm value.
         */
        function norm(a: NDArray, type?: number): number;
        /**
         * Computes the rank of a matrix using SVD.
         * @memberof NDWasmAnalysis
         * @param {NDArray} a - Input matrix.
         * @param {number} tol - Tolerance for singular values (0 for 1e-14).
         * @returns {number} Integer rank of the matrix.
         */
        function rank(a: NDArray, tol?: number): number;
        /**
         * estimates the reciprocal condition number of matrix a.
         * @memberof NDWasmAnalysis
         * @param {NDArray} a - Input matrix.
         * @param {number} norm - norm: 1 (1-norm) or Infinity (Infinity norm).
         * @returns {number} result.
        */
        function cond(a: NDArray, norm?: number): number;
        /**
         * Eigenvalue decomposition for symmetric matrices.
         * @memberof NDWasmAnalysis
         * @param {NDArray} a - Symmetric square matrix.
         * @param {boolean} computeVectors - Whether to return eigenvectors.
         * @returns {{values: NDArray, vectors: NDArray|null}}
         */
        function eigenSym(a: NDArray, computeVectors?: boolean): {
            values: NDArray;
            vectors: NDArray | null;
        };
        /**
         * Computes pairwise Euclidean distances between two sets of vectors.
         * @memberof NDWasmAnalysis
         * @param {NDArray} a - Matrix of shape [m, d].
         * @param {NDArray} b - Matrix of shape [n, d].
         * @returns {NDArray} Distance matrix of shape [m, n].
         */
        function pairwiseDist(a: NDArray, b: NDArray): NDArray;
        /**
         * Performs K-Means clustering in WASM memory.
         * @memberof NDWasmAnalysis
         * @param {NDArray} data - Data of shape [n_samples, d_features].
         * @param {number} k - Number of clusters.
         * @param {number} maxIter - Maximum iterations.
         * @returns {{centroids: NDArray, labels: NDArray, iterations: number}}
         */
        function kmeans(data: NDArray, k: number, maxIter?: number): {
            centroids: NDArray;
            labels: NDArray;
            iterations: number;
        };
        /**
         * Computes the Kronecker product C = A ⊗ B.
         * @memberof NDWasmAnalysis
         */
        function kronecker(a: any, b: any): NDArray;
    }
    import { NDArray } from "ndarray_core";
}
declare module "ndwasm" {
    /**
     * Static factory: Creates a new NDArray directly from WASM computation results.
     * @param {WasmBuffer} bridge - The WasmBuffer instance containing the result from WASM.
     * @param {Array<number>} shape - The shape of the new NDArray.
     * @param {string} [dtype] - The data type of the new NDArray. If not provided, uses bridge.dtype.
     * @returns {NDArray} The new NDArray.
     */
    export function fromWasm(bridge: WasmBuffer, shape: Array<number>, dtype?: string): NDArray;
    export const blas: {
        trace(a: NDArray): number;
        matMul(a: NDArray, b: NDArray): NDArray;
        matPow(a: NDArray, k: any): NDArray;
        matMulBatch(a: NDArray, b: NDArray): NDArray;
        syrk(a: NDArray): NDArray;
        trsm(a: NDArray, b: NDArray, lower?: boolean): NDArray;
        matVecMul(a: NDArray, x: NDArray): NDArray;
        ger(x: NDArray, y: NDArray): NDArray;
    };
    export const decomp: {
        solve(a: NDArray, b: NDArray): NDArray;
        inv(a: NDArray): NDArray;
        svd(a: NDArray): {
            u: NDArray;
            s: NDArray;
            v: NDArray;
        };
        qr(a: NDArray): {
            q: NDArray;
            r: NDArray;
        };
        cholesky(a: NDArray): NDArray;
        lu(a: NDArray): NDArray;
        pinv(a: NDArray): NDArray;
        det(a: NDArray): number;
        logDet(a: NDArray): {
            sign: number;
            logAbsDet: number;
        };
        eigen(a: NDArray): {
            values: NDArray;
            vectors: NDArray;
        };
    };
    export const signal: {
        fft(a: NDArray): NDArray;
        ifft(a: NDArray): NDArray;
        rfft(a: NDArray): NDArray;
        rifft(a: NDArray, n: number): NDArray;
        fft2(a: NDArray): NDArray;
        ifft2(a: NDArray): NDArray;
        dct(a: NDArray): NDArray;
        conv2d(img: NDArray, kernel: NDArray, stride?: number, padding?: number): NDArray;
        correlate2d(img: NDArray, kernel: NDArray, stride?: number, padding?: number): NDArray;
    };
    export const analysis: {
        argsort(a: NDArray): NDArray;
        topk(a: NDArray, k: number, largest?: boolean): {
            values: NDArray;
            indices: NDArray;
        };
        cov(a: NDArray): NDArray;
        corr(a: NDArray): NDArray;
        norm(a: NDArray, type?: number): number;
        rank(a: NDArray, tol?: number): number;
        cond(a: NDArray, norm?: number): number;
        eigenSym(a: NDArray, computeVectors?: boolean): {
            values: NDArray;
            vectors: NDArray | null;
        };
        pairwiseDist(a: NDArray, b: NDArray): NDArray;
        kmeans(data: NDArray, k: number, maxIter?: number): {
            centroids: NDArray;
            labels: NDArray;
            iterations: number;
        };
        kronecker(a: any, b: any): NDArray;
    };
    export default NDWasm;
    /**
     * WasmBuffer
     */
    export class WasmBuffer {
        /**
         * @param {Object} exports - The exports object from Go WASM.
         * @param {number} size - Number of elements.
         * @param {string} dtype - Data type (float64, float32, etc.).
         */
        constructor(exports: Object, size: number, dtype: string);
        exports: Object;
        size: number;
        dtype: string;
        byteLength: number;
        ptr: any;
        /**
         * Refreshes the view into WASM memory.
         */
        refresh(): this;
        view: any;
        /** Synchronizes JS data into the WASM buffer. */
        push(typedArray: any): void;
        /** Pulls data from the WASM buffer back to JS (returns a copy).
         * @returns {NDArray}
         */
        pull(): NDArray;
        /** Disposes of the temporary buffer. */
        dispose(): void;
    }
    import { NDArray } from "ndarray_core";
    export namespace NDWasm {
        let runtime: null;
        /** Binds a loaded WASM runtime to the bridge.
         * @param {*} runtime
         */
        function bind(runtime: any): void;
        /**
         * Init the NDWasm
         * @param {string} [baseDir='.']
         */
        function init(baseDir?: string): Promise<void>;
        /** Internal helper: executes a computation in WASM and manages memory. */
        function _compute(inputs: any, outShape: any, outDtype: any, computeFn: any): NDArray;
    }
    /**
     * WasmRuntime
     */
    export class WasmRuntime {
        instance: WebAssembly.Instance | null;
        exports: WebAssembly.Exports | null;
        isLoaded: boolean;
        /**
         * Initializes the Go WASM environment.
         * @param {object} [options] - Optional configuration.
         * @param {string} [options.wasmUrl='./ndarray_plugin.wasm'] - Path or URL to the wasm file.
         * @param {string} [options.execUrl='./wasm_exec.js'] - Path to the wasm_exec.js file (Node.js only).
         * @param {number} [options.initialMemoryPages] - Initial memory size in 64KiB pages.
         * @param {number} [options.maximumMemoryPages] - Maximum memory size in 64KiB pages.
         * @param {string} [options.baseDir='.']
         */
        init(options?: {
            wasmUrl?: string | undefined;
            execUrl?: string | undefined;
            initialMemoryPages?: number | undefined;
            maximumMemoryPages?: number | undefined;
            baseDir?: string | undefined;
        }): Promise<void>;
        /**
         * Helper method: Gets the suffix for Go export function names based on type.
         * @private
         */
        private _getSuffix;
        /**
         * Quickly allocates a buffer.
         * @returns {WasmBuffer}
         */
        createBuffer(size: any, dtype: any): WasmBuffer;
    }
}
declare module "ndwasm_image" {
    export namespace NDWasmImage {
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
        function decode(imageBytes: Uint8Array): NDArray | null;
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
        function encode(ndarray: NDArray, { format, quality }?: {
            format?: string | undefined;
            quality?: number | undefined;
        }): Uint8Array | null;
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
        function encodePng(ndarray: NDArray): Uint8Array | null;
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
        function encodeJpeg(ndarray: NDArray, options?: {
            quality?: number | undefined;
        }): Uint8Array | null;
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
        function convertUint8ArrrayToDataurl(uint8array: Uint8Array, mimeType?: string): string;
    }
    export default NDWasmImage;
    import { NDArray } from "ndarray_core";
}
declare module "ndarray_factory" {
    /**
     * Creates an NDArray from a regular array or TypedArray.
     * @param {Array|TypedArray} source - The source data.
     * @param {string} [dtype='float64'] - The desired data type.
     * @return {NDArray}
     */
    export function array(source: any[] | TypedArray, dtype?: string): NDArray;
    /**
     * Creates an NDArray from a regular array or TypedArray.
     * @param {Array|TypedArray} source - The source data.
     * @return {NDArray}
     */
    export function float64(source: any[] | TypedArray): NDArray;
    /**
     * Creates an NDArray from a regular array or TypedArray.
     * @param {Array|TypedArray} source - The source data.
     * @return {NDArray}
     */
    export function float32(source: any[] | TypedArray): NDArray;
    /**
     * Creates an NDArray from a regular array or TypedArray.
     * @param {Array|TypedArray} source - The source data.
     * @return {NDArray}
     */
    export function uint32(source: any[] | TypedArray): NDArray;
    /**
     * Creates an NDArray from a regular array or TypedArray.
     * @param {Array|TypedArray} source - The source data.
     * @return {NDArray}
     */
    export function int32(source: any[] | TypedArray): NDArray;
    /**
     * Creates an NDArray from a regular array or TypedArray.
     * @param {Array|TypedArray} source - The source data.
     * @return {NDArray}
     */
    export function int16(source: any[] | TypedArray): NDArray;
    /**
     * Creates an NDArray from a regular array or TypedArray.
     * @param {Array|TypedArray} source - The source data.
     * @return {NDArray}
     */
    export function uint16(source: any[] | TypedArray): NDArray;
    /**
     * Creates an NDArray from a regular array or TypedArray.
     * @param {Array|TypedArray} source - The source data.
     * @return {NDArray}
     */
    export function int8(source: any[] | TypedArray): NDArray;
    /**
     * Creates an NDArray from a regular array or TypedArray.
     * @param {Array|TypedArray} source - The source data.
     * @return {NDArray}
     */
    export function uint8(source: any[] | TypedArray): NDArray;
    /**
     * Creates an NDArray from a regular array or TypedArray.
     * @param {Array|TypedArray} source - The source data.
     * @return {NDArray}
     */
    export function uint8c(source: any[] | TypedArray): NDArray;
    /**
     * Creates a new NDArray of the given shape, filled with zeros.
     * @param {Array<number>} shape - The shape of the new array.
     * @param {string} [dtype='float64'] - The data type of the new array.
     * @returns {NDArray}
     */
    export function zeros(shape: Array<number>, dtype?: string): NDArray;
    /**
     * Creates a new NDArray of the given shape, filled with ones.
     * @param {Array<number>} shape - The shape of the new array.
     * @param {string} [dtype='float64'] - The data type of the new array.
     * @returns {NDArray}
     */
    export function ones(shape: Array<number>, dtype?: string): NDArray;
    /**
     * Creates a new NDArray of the given shape, filled with a specified value.
     * @param {Array<number>} shape - The shape of the new array.
     * @param {*} value - The value to fill the array with.
     * @param {string} [dtype='float64'] - The data type of the new array.
     * @returns {NDArray}
     */
    export function full(shape: Array<number>, value: any, dtype?: string): NDArray;
    /**
     * Like Python's arange: [start, stop)
     * @param {number} start the starting value of the sequence.
     * @param {number | null} stop the end value of the sequence.
     * @param {number | null} [step=1] the spacing between values.
     * @param {string} [dtype='float64'] the data type of the resulting array.
     * @returns {NDArray}
     */
    export function arange(start: number, stop?: number | null, step?: number | null, dtype?: string): NDArray;
    /**
     * Linearly spaced points.
     * @param {number} start the starting value of the sequence.
     * @param {number | null} stop the end value of the sequence.
     * @param {number} [num=50] the number of samples to generate.
     * @param {string} [dtype='float64'] the data type of the resulting array.
     * @returns {NDArray}
     */
    export function linspace(start: number, stop: number | null, num?: number, dtype?: string): NDArray;
    /**
     * where(condition, x, y)
     * Returns a new array with elements chosen from x or y depending on condition.
     * Supports NumPy-style broadcasting across all three arguments, including 0-sized dimensions.
     *
     * @param {NDArray|Array|number} condition - Where True, yield x, otherwise yield y.
     * @param {NDArray|Array|number} x - Values from which to choose if condition is True.
     * @param {NDArray|Array|number} y - Values from which to choose if condition is False.
     * @returns {NDArray} A new contiguous NDArray.
     */
    export function where(condition: NDArray | any[] | number, x: NDArray | any[] | number, y: NDArray | any[] | number): NDArray;
    /**
     * Joins a sequence of arrays along an existing axis.
     * @param {Array<NDArray>} arrays The arrays must have the same shape, except in the dimension corresponding to `axis`.
     * @param {number} [axis=0] The axis along which the arrays will be joined.
     * @returns {NDArray} A new NDArray.
     */
    export function concat(arrays: Array<NDArray>, axis?: number): NDArray;
    /**
     * Joins a sequence of arrays along a new axis.
     * The `stack` function creates a new dimension, whereas `concat` joins along an existing one.
     * All input arrays must have the same shape and dtype.
     *
     * @param {Array<NDArray>} arrays - The list of arrays to stack.
     * @param {number} [axis=0] - The axis in the result array along which the input arrays are stacked.
     * @returns {NDArray} A new NDArray.
     */
    export function stack(arrays: Array<NDArray>, axis?: number): NDArray;
    /**
     * Creates a 2D identity matrix.
     * @param {number} n - Number of rows.
     * @param {number} [m] - Number of columns. Defaults to n if not provided.
     * @param {string} [dtype='float64'] - Data type of the array.
     * @returns {NDArray} A 2D NDArray with ones on the main diagonal.
     */
    export function eye(n: number, m?: number, dtype?: string): NDArray;
    import { NDArray } from "ndarray_core";
}
declare module "ndwasm_optimize" {
    export namespace NDWasmOptimize {
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
        function linprog(c: NDArray, G: NDArray | null, h: NDArray | null, A: NDArray | null, b: NDArray | null, bounds: any[]): {
            x: NDArray;
            fun: number;
            status: number;
            message: string;
        };
        /**
         * Fits a simple linear regression model: Y = alpha + beta*X.
         * @param {NDArray} x - The independent variable (1D NDArray of float64).
         * @param {NDArray} y - The dependent variable (1D NDArray of float64).
         * @returns {{alpha: number, beta: number}} - An object containing the intercept (alpha) and slope (beta) of the fitted line.
         * @throws {Error} If WASM runtime is not loaded or inputs are invalid.
         */
        function linearRegression(x: NDArray, y: NDArray): {
            alpha: number;
            beta: number;
        };
        /**
         * Finds the minimum of a scalar function of one or more variables using an L-BFGS optimizer.
         * @param {Function} func - The objective function to be minimized. It must take a 1D `Float64Array` `x` (current point) and return a single number (the function value at `x`).
         * @param {NDArray} x0 - The initial guess for the optimization (1D NDArray of float64).
         * @param {Object} [options] - Optional parameters.
         * @param {Function} [options.grad] - The gradient of the objective function. Must take `x` (a 1D `Float64Array`) and write the result into the second argument `grad_out` (a 1D `Float64Array`). This function should *not* return a value.
         * @returns {{x: NDArray, success: boolean, message: string, ...stats}} The optimization result.
         */
        function minimize(func: Function, x0: NDArray, options?: {
            grad?: Function | undefined;
        }): {
            x: NDArray;
            success: boolean;
            message: string;
        };
    }
    import { NDArray } from "ndarray_core";
}
declare module "ndwasmarray" {
    /**
     * NDWasmArray
     */
    export class NDWasmArray {
        /**
         * Static factory: Creates a WASM-resident array.
         * 1. If source is an NDArray, it calls .push() to move it to WASM.
         * 2. If source is a JS Array, it allocates WASM memory and fills it directly
         *    via recursive traversal to avoid intermediate flattening.
         */
        static fromArray(source: any, dtype?: string): NDWasmArray;
        /**
         * @param {WasmBuffer} buffer - The WASM memory bridge (contains .ptr and .view).
         * @param {Int32Array|Array} shape - Dimensions of the array.
         * @param {string} dtype - Data type (e.g., 'float64').
         */
        constructor(buffer: WasmBuffer, shape: Int32Array | any[], dtype: string);
        buffer: WasmBuffer;
        shape: Int32Array<ArrayBufferLike>;
        dtype: string;
        ndim: number;
        size: number;
        /**
         * Pulls data from WASM to a JS-managed NDArray.
         * @param {boolean} [dispose=true] - Release WASM memory after pulling.
         */
        pull(dispose?: boolean): NDArray;
        /**
         * Manually releases WASM heap memory.
         */
        dispose(): void;
        /**
         * Internal helper to prepare operands for WASM operations.
         * Ensures input is converted to NDWasmArray and tracks if it needs auto-disposal.
         * @private
         */
        private _prepareOperand;
        /**
         * Matrix Multiplication: C = this * other
         * @param {NDWasmArray | NDArray} other
         * @returns {NDWasmArray}
         */
        matMul(other: NDWasmArray | NDArray): NDWasmArray;
        /**
         * Batched Matrix Multiplication: C[i] = this[i] * other[i]
         * @param {NDWasmArray | NDArray}
         * @returns {NDWasmArray}
         */
        matMulBatch(other: any): NDWasmArray;
    }
    import { NDArray } from "ndarray_core";
}
declare module "ndarray_jit" {
    /**
     * Generates a JIT-compiled unary kernel.
     * Output is always contiguous, so ptrOut simply increments by 1.
     * @private
     */
    export function _createUnaryKernel(cacheKey: any, shape: any, sIn: any, fnOrStr: any): any;
    /**
     * Generates a kernel with unrolled nested loops and cumulative pointer increments.
     * @private
     */
    export function _createBinKernel(cacheKey: any, shape: any, sA: any, sB: any, sOut: any, opStr: any, isComparison: any): any;
    /**
         * Generates a JIT kernel using nested loops and pointer gaps.
         * @private
         */
    export function _createReduceKernel(cacheKey: any, shape: any, strides: any, iterAxes: any, reduceAxes: any, reducer: any, finalFn: any): any;
    /**
     * Generates a JIT-compiled kernel for the set operation.
     * Optimized for V8 by using incremental pointer arithmetic and static nesting.
     *
     * @param {string} cacheKey - Unique key for the specific array structure.
     * @param {number} ndim - Number of dimensions of the target array.
     * @param {Array} targetShape - The logical shape of the selection.
     * @param {Int32Array} tStrides - Strides of the target NDArray.
     * @param {Int32Array} sStrides - Pre-computed broadcasting strides for the source.
     * @param {Uint8Array} hasPSet - Flags indicating if a dimension uses fancy indexing or scalar.
     * @param {Array<boolean>} isDimReduced - Flags indicating if a dimension is collapsed by scalar indexing.
     * @private
     */
    export function _createSetKernel(cacheKey: string, ndim: number, targetShape: any[], tStrides: Int32Array, sStrides: Int32Array, hasPSet: Uint8Array, isDimReduced: Array<boolean>): any;
    /**
     * Generates a JIT-compiled kernel for the pick operation.
     * Optimized for V8 with incremental pointer arithmetic and contiguous output writes.
     *
     * @param {string} cacheKey - Unique key for the specific array structure.
     * @param {number} ndim - Number of dimensions of the source array.
     * @param {Int32Array} sStrides - Strides of the source NDArray.
     * @param {Uint8Array} isFullSlice - Flags indicating if a dimension is a full ":" slice.
     * @param {Array<boolean>} isDimReduced - Flags indicating if a dimension is collapsed by scalar indexing.
     * @param {Int32Array} odometerShape - The lengths of the pick-sets for each dimension.
     * @private
     */
    export function _createPickKernel(cacheKey: string, ndim: number, sStrides: Int32Array, isFullSlice: Uint8Array, isDimReduced: Array<boolean>, odometerShape: Int32Array): any;
    export namespace Jit {
        let debug: boolean;
    }
}
declare module "ndarray_core" {
    export namespace DTYPE_MAP {
        let float64: Float64ArrayConstructor;
        let float32: Float32ArrayConstructor;
        let int32: Int32ArrayConstructor;
        let uint32: Uint32ArrayConstructor;
        let int16: Int16ArrayConstructor;
        let uint16: Uint16ArrayConstructor;
        let int8: Int8ArrayConstructor;
        let uint8: Uint8ArrayConstructor;
        let uint8c: Uint8ClampedArrayConstructor;
    }
    /**
     * The NDArray class
     */
    export class NDArray {
        static random: {
            _cryptoUniform01(size: any): Float64Array<any>;
            random(shape: any[], low?: number, high?: number, dtype?: string): NDArray;
            normal(shape: any[], mean?: number, std?: number, dtype?: string): NDArray;
            bernoulli(shape: any[], p?: number, dtype?: string): NDArray;
            exponential(shape: any[], lambda?: number, dtype?: string): NDArray;
            poisson(shape: any[], lambda?: number, dtype?: string): NDArray;
            _cast(data: any, dtype: any): any;
        };
        static blas: {
            trace(a: NDArray): number;
            matMul(a: NDArray, b: NDArray): NDArray;
            matPow(a: NDArray, k: any): NDArray;
            matMulBatch(a: NDArray, b: NDArray): NDArray;
            syrk(a: NDArray): NDArray;
            trsm(a: NDArray, b: NDArray, lower?: boolean): NDArray;
            matVecMul(a: NDArray, x: NDArray): NDArray;
            ger(x: NDArray, y: NDArray): NDArray;
        };
        static decomp: {
            solve(a: NDArray, b: NDArray): NDArray;
            inv(a: NDArray): NDArray;
            svd(a: NDArray): {
                u: NDArray;
                s: NDArray;
                v: NDArray;
            };
            qr(a: NDArray): {
                q: NDArray;
                r: NDArray;
            };
            cholesky(a: NDArray): NDArray;
            lu(a: NDArray): NDArray;
            pinv(a: NDArray): NDArray;
            det(a: NDArray): number;
            logDet(a: NDArray): {
                sign: number;
                logAbsDet: number;
            };
            eigen(a: NDArray): {
                values: NDArray;
                vectors: NDArray;
            };
        };
        static analysis: {
            argsort(a: NDArray): NDArray;
            topk(a: NDArray, k: number, largest?: boolean): {
                values: NDArray;
                indices: NDArray;
            };
            cov(a: NDArray): NDArray;
            corr(a: NDArray): NDArray;
            norm(a: NDArray, type?: number): number;
            rank(a: NDArray, tol?: number): number;
            cond(a: NDArray, norm?: number): number;
            eigenSym(a: NDArray, computeVectors?: boolean): {
                values: NDArray;
                vectors: NDArray | null;
            };
            pairwiseDist(a: NDArray, b: NDArray): NDArray;
            kmeans(data: NDArray, k: number, maxIter?: number): {
                centroids: NDArray;
                labels: NDArray;
                iterations: number;
            };
            kronecker(a: any, b: any): NDArray;
        };
        static image: {
            decode(imageBytes: Uint8Array): NDArray | null;
            encode(ndarray: NDArray, { format, quality }?: {
                format?: string | undefined;
                quality?: number | undefined;
            }): Uint8Array | null;
            encodePng(ndarray: NDArray): Uint8Array | null;
            encodeJpeg(ndarray: NDArray, options?: {
                quality?: number | undefined;
            }): Uint8Array | null;
            convertUint8ArrrayToDataurl(uint8array: Uint8Array, mimeType?: string): string;
        };
        static signal: {
            fft(a: NDArray): NDArray;
            ifft(a: NDArray): NDArray;
            rfft(a: NDArray): NDArray;
            rifft(a: NDArray, n: number): NDArray;
            fft2(a: NDArray): NDArray;
            ifft2(a: NDArray): NDArray;
            dct(a: NDArray): NDArray;
            conv2d(img: NDArray, kernel: NDArray, stride?: number, padding?: number): NDArray;
            correlate2d(img: NDArray, kernel: NDArray, stride?: number, padding?: number): NDArray;
        };
        /**
         * Optimization module for linear programming, non-linear minimization, and linear regression.
         * @memberof NDArray
         * @type {NDWasmOptimize}
         */
        static optimize: {
            linprog(c: NDArray, G: NDArray | null, h: NDArray | null, A: NDArray | null, b: NDArray | null, bounds: any[]): {
                x: NDArray;
                fun: number;
                status: number;
                message: string;
            };
            linearRegression(x: NDArray, y: NDArray): {
                alpha: number;
                beta: number;
            };
            minimize(func: Function, x0: NDArray, options?: {
                grad?: Function | undefined;
            }): {
                x: NDArray;
                success: boolean;
                message: string;
            };
        };
        /**
         * @param {TypedArray} data - The underlying physical storage.
         * @param {Object} options
         * @param {Array|Int32Array} options.shape - The dimensions of the array.
         * @param {Array|Int32Array} [options.strides] - The strides, defaults to C-style.
         * @param {number} [options.offset=0] - The view offset.
         * @param {string} [options.dtype] - The data type.
         */
        constructor(data: TypedArray, { shape, strides, offset, dtype }: {
            shape: any[] | Int32Array;
            strides?: any[] | Int32Array<ArrayBufferLike> | undefined;
            offset?: number | undefined;
            dtype?: string | undefined;
        });
        data: TypedArray;
        shape: Int32Array<ArrayBufferLike>;
        ndim: number;
        offset: number;
        dtype: string;
        size: number;
        strides: Int32Array<ArrayBufferLike>;
        isContiguous: boolean;
        _determineDtype(data: any): string;
        _computeDefaultStrides(shape: any): Int32Array<ArrayBuffer>;
        _checkContiguity(): boolean;
        /**
         * High-performance addressing: converts multidimensional indices to a physical offset.
         * @private
         * @param {Array|Int32Array} indices
         * @param {number}
         */
        private _getOffset;
        /**
         * To JavaScript Array
         * @returns {Array|number} the array
         */
        toArray(): any[] | number;
        /**
         * Returns a string representation of the ndarray.
         * Formats high-dimensional data with proper indentation and line breaks.
         * @returns {String}
         */
        toString(): string;
        /**
        * High-performance element-wise mapping with jit compilation.
        * @param {string | Function} fnOrStr - The function string to apply to each element, like 'Math.sqrt(${val})', or a lambda expression
        * @returns {NDArray} A new array with the results.
        *
        */
        map(fnOrStr: string | Function, dtype?: undefined): NDArray;
        /**
         * Generic iterator that handles stride logic. It's slow. use map/reduce if you want to use jit.
         * @param {Function} callback - A function called with `(value, index, flatPhysicalIndex)`, return true to exit early
         * @see NDArray#map
         */
        iterate(callback: Function): void;
        /**
         * Element-wise addition. Supports broadcasting.
         * @function
         * @param {NDArray|number} other - The array or scalar to add.
         * @returns {NDArray} A new array containing the results.
         *
         */
        add(other: NDArray | number): NDArray;
        /**
         * Element-wise subtraction. Supports broadcasting.
         * @function
         * @param {NDArray|number} other - The array or scalar to subtract.
         * @returns {NDArray} A new array containing the results.
         *
         */
        sub(other: NDArray | number): NDArray;
        /**
         * Element-wise multiplication. Supports broadcasting.
         * @function
         * @param {NDArray|number} other - The array or scalar to multiply by.
         * @returns {NDArray} A new array containing the results.
         *
         */
        mul(other: NDArray | number): NDArray;
        /**
         * Element-wise division. Supports broadcasting.
         * @function
         * @param {NDArray|number} other - The array or scalar to divide by.
         * @returns {NDArray} A new array containing the results.
         *
         */
        div(other: NDArray | number): NDArray;
        /**
         * Element-wise exponentiation. Supports broadcasting.
         * @function
         * @param {NDArray|number} other - The array or scalar exponent.
         * @returns {NDArray} A new array containing the results.
         *
         */
        pow(other: NDArray | number): NDArray;
        /**
         * Element-wise modulo. Supports broadcasting.
         * @function
         * @param {NDArray|number} other - The array or scalar divisor.
         * @returns {NDArray} A new array containing the results.
         *
         */
        mod(other: NDArray | number): NDArray;
        /**
         * In-place element-wise addition.
         * @function
         * @param {NDArray|number} other - The array or scalar to add.
         * @returns {NDArray} The modified array (`this`).
         *
         */
        iadd(other: NDArray | number): NDArray;
        /**
         * In-place element-wise subtraction.
         * @function
         * @param {NDArray|number} other - The array or scalar to subtract.
         * @returns {NDArray} The modified array (`this`).
         *
         */
        isub(other: NDArray | number): NDArray;
        /**
         * In-place element-wise multiplication.
         * @function
         * @param {NDArray|number} other - The array or scalar to multiply by.
         * @returns {NDArray} The modified array (`this`).
         *
         */
        imul(other: NDArray | number): NDArray;
        /**
         * In-place element-wise division.
         * @function
         * @param {NDArray|number} other - The array or scalar to divide by.
         * @returns {NDArray} The modified array (`this`).
         *
         */
        idiv(other: NDArray | number): NDArray;
        /**
         * In-place element-wise exponentiation.
         * @function
         * @param {NDArray|number} other - The array or scalar exponent.
         * @returns {NDArray} The modified array (`this`).
         *
         */
        ipow(other: NDArray | number): NDArray;
        /**
         * In-place element-wise modulo.
         * @function
         * @param {NDArray|number} other - The array or scalar divisor.
         * @returns {NDArray} The modified array (`this`).
         *
         */
        imod(other: NDArray | number): NDArray;
        /**
         * bitwise AND. Returns a new array.
         * @function
         * @param {NDArray|number} other - The array or scalar to perform the operation with.
         * @returns {NDArray}
         *
         */
        bitwise_and(other: NDArray | number): NDArray;
        /**
         * bitwise OR. Returns a new array.
         * @function
         * @param {NDArray|number} other - The array or scalar to perform the operation with.
         * @returns {NDArray}
         *
         */
        bitwise_or(other: NDArray | number): NDArray;
        /**
         * bitwise XOR. Returns a new array.
         * @function
         * @param {NDArray|number} other - The array or scalar to perform the operation with.
         * @returns {NDArray}
         *
         */
        bitwise_xor(other: NDArray | number): NDArray;
        /**
         * bitwise lshift. Returns a new array.
         * @function
         * @param {NDArray|number} other - The array or scalar to perform the operation with.
         * @returns {NDArray}
         *
         */
        bitwise_lshift(other: NDArray | number): NDArray;
        /**
         * bitwise (logical) rshift. Returns a new array.
         * @function
         * @param {NDArray|number} other - The array or scalar to perform the operation with.
         * @returns {NDArray}
         *
         */
        bitwise_rshift(other: NDArray | number): NDArray;
        /**
         * bitwise NOT. Returns a new array.
         * @function
         * @returns {NDArray}
         *
         */
        bitwise_not(): NDArray;
        /**
         * Returns a new array with the numeric negation of each element.
         * @function
         * @returns {NDArray}
         *
         */
        neg(): NDArray;
        /**
         * Returns a new array with the absolute value of each element.
         * @function
         * @returns {NDArray}
         *
         */
        abs(): NDArray;
        /**
         * Returns a new array with `e` raised to the power of each element.
         * @function
         * @returns {NDArray}
         *
         */
        exp(): NDArray;
        /**
         * Returns a new array with the square root of each element.
         * @function
         * @returns {NDArray}
         *
         */
        sqrt(): NDArray;
        /**
        * Returns a new array with the sine of each element.
        * @function
        * @returns {NDArray}
        *
        */
        sin(): NDArray;
        /**
         * Returns a new array with the cosine of each element.
         * @function
         * @returns {NDArray}
         *
         */
        cos(): NDArray;
        /**
         * Returns a new array with the tangent of each element.
         * @function
         * @returns {NDArray}
         *
         */
        tan(): NDArray;
        /**
         * Returns a new array with the natural logarithm (base e) of each element.
         * @function
         * @returns {NDArray}
         *
         */
        log(): NDArray;
        /**
         * Returns a new array with the smallest integer greater than or equal to each element.
         * @function
         * @returns {NDArray}
         *
         */
        ceil(): NDArray;
        /**
         * Returns a new array with the largest integer less than or equal to each element.
         * @function
         * @returns {NDArray}
         *
         */
        floor(): NDArray;
        /**
         * Returns a new array with the value of each element rounded to the nearest integer.
         * @function
         * @returns {NDArray}
         *
         */
        round(): NDArray;
        /**
         * Element-wise equality comparison. Returns a new boolean (uint8) array.
         * @function
         * @param {NDArray|number} other - The array or scalar to compare with.
         * @returns {NDArray}
         *
         */
        eq(other: NDArray | number): NDArray;
        /**
         * Element-wise inequality comparison. Returns a new boolean (uint8) array.
         * @function
         * @param {NDArray|number} other - The array or scalar to compare with.
         * @returns {NDArray}
         *
         */
        neq(other: NDArray | number): NDArray;
        /**
         * Element-wise greater-than comparison. Returns a new boolean (uint8) array.
         * @function
         * @param {NDArray|number} other - The array or scalar to compare with.
         * @returns {NDArray}
         *
         */
        gt(other: NDArray | number): NDArray;
        /**
         * Element-wise greater-than-or-equal comparison. Returns a new boolean (uint8) array.
         * @function
         * @param {NDArray|number} other - The array or scalar to compare with.
         * @returns {NDArray}
         *
         */
        gte(other: NDArray | number): NDArray;
        /**
         * Element-wise less-than comparison. Returns a new boolean (uint8) array.
         * @function
         * @param {NDArray|number} other - The array or scalar to compare with.
         * @returns {NDArray}
         *
         */
        lt(other: NDArray | number): NDArray;
        /**
         * Element-wise less-than-or-equal comparison. Returns a new boolean (uint8) array.
         * @function
         * @param {NDArray|number} other - The array or scalar to compare with.
         * @returns {NDArray}
         *
         */
        lte(other: NDArray | number): NDArray;
        /**
         * Element-wise logical AND. Returns a new boolean (uint8) array.
         * @function
         * @param {NDArray|number} other - The array or scalar to perform the operation with.
         * @returns {NDArray}
         *
         */
        logical_and(other: NDArray | number): NDArray;
        /**
         * Element-wise logical OR. Returns a new boolean (uint8) array.
         * @function
         * @param {NDArray|number} other - The array or scalar to perform the operation with.
         * @returns {NDArray}
         *
         */
        logical_or(other: NDArray | number): NDArray;
        /**
         * Element-wise logical NOT. Returns a new boolean (uint8) array.
         * @function
         * @returns {NDArray}
         *
         */
        logical_not(): NDArray;
        /**
         * Computes the sum of elements along the specified axis.
         *
         * @param {number|null} [axis=null]
         * @returns {NDArray|number}
         */
        sum(axis?: number | null): NDArray | number;
        /**
         * Computes the cumprod of elements along the specified axis.
         *
         * @param {number|null} [axis=null]
         * @returns {NDArray|number}
         */
        cumprod(axis?: number | null): NDArray | number;
        /**
         * Computes the arithmetic mean along the specified axis.
         *
         * @param {number|null} [axis=null]
         * @returns {NDArray|number}
         */
        mean(axis?: number | null): NDArray | number;
        /**
         * Returns the maximum value along the specified axis.
         *
         * @param {number|null} [axis=null]
         * @returns {NDArray|number}
         */
        max(axis?: number | null): NDArray | number;
        /**
         * Returns the minimum value along the specified axis.
         *
         * @param {number|null} [axis=null]
         * @returns {NDArray|number}
         */
        min(axis?: number | null): NDArray | number;
        /**
         * Computes the variance along the specified axis.
         * Note: This implementation uses a two-pass approach (mean first, then squared differences).
         * Ensure that the `sub` method supports broadcasting if `axis` is not null.
         *
         *
         * @param {number|null} [axis=null] - The axis to reduce.
         * @returns {NDArray|number}
         */
        var(axis?: number | null): NDArray | number;
        /**
         * Computes the standard deviation along the specified axis.
         *
         *
         * @param {number|null} [axis=null] - The axis to reduce.
         * @returns {NDArray|number}
         */
        std(axis?: number | null): NDArray | number;
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
        private _reduce;
        /**
         * Returns the index of the maximum value in a flattened array.
         * @returns {number}
         *
         */
        argmax(): number;
        /**
         * Returns the index of the minimum value in a flattened array.
         * @returns {number}
         *
         */
        argmin(): number;
        /**
         * Checks if all elements in the array are truthy.
         * @returns {boolean}
         *
         */
        all(): boolean;
        /**
         * Checks if any element in the array is truthy.
         * @returns {boolean}
         *
         */
        any(): boolean;
        /**
         * Returns a new array with a new shape, without changing data. O(1) operation.
         * This only works for contiguous arrays. If the array is not contiguous,
         * you must call .copy() first.
         *
         * @param {...number} newShape - The new shape.
         * @returns {NDArray} NDArray view.
         */
        reshape(...newShape: number[]): NDArray;
        /**
         * Returns a new view of the array with axes transposed. O(1) operation.
         *
         * @param {...number} axes - The new order of axes, e.g., [1, 0] for a matrix transpose. If not specified, reverses the order of the axes.
         * @returns {NDArray} NDArray view.
         */
        transpose(...axes: number[]): NDArray;
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
        slice(...specs: (any[] | number | null | undefined)[]): NDArray;
        /**
         * Returns a 1D view of the i-th row.
         * Only applicable to 2D arrays.
         *
         * @param {number} i - The row index.
         * @returns {NDArray} A 1D NDArray view.
         */
        rowview(i: number): NDArray;
        /**
         * Returns a 1D view of the j-th column.
         * Only applicable to 2D arrays.
         *
         * @param {number} j - The column index.
         * @returns {NDArray} A 1D NDArray view.
         */
        colview(j: number): NDArray;
        /**
         * Remove axes of length one from the shape. O(1) operation.
         *
         * @param {number|null} axis - The axis to squeeze. If null, all axes of length 1 are removed.
         * @returns {NDArray} NDArray view.
         */
        squeeze(axis?: number | null): NDArray;
        /**
         * Returns a new, contiguous array with the same data. O(n) operation.
         * This converts any view (transposed, sliced) into a new array with a standard C-style memory layout.
         *
         * @param {string | undefined} the target dtype
         * @returns {NDArray} NDArray view.
         */
        copy(dtype?: undefined): NDArray;
        /**
         * Ensures the returned array has a contiguous memory layout.
         * If the array is already contiguous, it returns itself. Otherwise, it returns a copy.
         * Often used as a pre-processing step before calling WASM or other libraries.
         *
         * @returns {NDArray} NDArray view.
         */
        asContiguous(): NDArray;
        /**
         * Gets a single element from the array.
         * Note: This has higher overhead than batch operations. Use with care in performance-critical code.
         *
         * @param {...number} indices - The indices of the element to get.
         * @returns {number}
         */
        get(...indices: number[]): number;
        /**
        * Sets value(s) in the array using a unified, JIT-optimized engine.
        * Supports scalar indexing, fancy (array) indexing, and NumPy-style broadcasting.
        *
        * @param {number|Array|NDArray} value - The source data to assign.
        * @param {...(number|Array|null)} indices - Indices for each dimension.
        * @returns {NDArray}
        */
        set(value: number | any[] | NDArray, ...indices: (number | any[] | null)[]): NDArray;
        /**
         * Advanced Indexing (Fancy Indexing).
         * Returns a physical COPY of the selected data using a JIT-compiled engine.
         * Picks elements along each dimension.
         * Note: unlike numpy, for advanced (fancy) indexing, output shape won't be reordered.
         * Dim for 1-element advanced indexing won't be removed, either.
         *
         * @param {...(number[]|TypedArray|number|null|undefined)} specs - Index selectors. null/undefined means select all
         */
        pick(...specs: (number[] | TypedArray | number | null | undefined)[]): NDArray;
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
        filter(predicateOrMask: Function | any[] | NDArray): NDArray;
        /**
         * @return {NDArray} - new flatten view to the array
         */
        flatten(): NDArray;
        /**
         * Dot Product (Scalar Inner Product).
         * Supports 1D arrays (vectors) only.
         * @param {NDArray} other
         * @returns {number} Scalar result.
         */
        dot(other: NDArray): number;
        /**
         * Cross Product.
         * Only valid for 1D vectors of length 3.
         * @param {NDArray} other
         * @returns {NDArray} New NDArray of size 3.
         */
        cross(other: NDArray): NDArray;
        /**
         * Matrix Multiplication in js.
         * Operations: (M, K) @ (K, N) -> (M, N)
         * @param {NDArray} other
         * @returns {NDArray} New NDArray.
         */
        jsMatMul(other: NDArray): NDArray;
        /**
         * Matrix-Vector Multiplication in js.
         * Operation: (M, K) @ (K,) -> (M,)
         * @param {NDArray} vec
         * @returns {NDArray} New NDArray (Vector).
         */
        jsMatVecMul(vec: NDArray): NDArray;
        /**
         * Projects the current ndarray to a WASM proxy (WasmBuffer).
         *
         * @param {WasmRuntime} runtime
         * @returns {WasmBuffer} A WasmBuffer instance representing the NDArray in WASM memory.
         */
        toWasm(runtime: WasmRuntime): WasmBuffer;
        /**
         * push to wasm
         * @returns {NDWasmArray}
         */
        push(): NDWasmArray;
        /**
         * Calculates the trace of a 2D square matrix (sum of diagonal elements).
         * Complexity: O(n)
         *
         * @returns {number} The sum of the diagonal elements.
         * @throws {Error} If the array is not 2D or not a square matrix.
         */
        trace(): number;
        /**
         * Performs matrix multiplication. This is a wrapper around `NDWasmBlas.matMul`.
         * @param {NDArray} other The right-hand side matrix.
         * @returns {NDArray} The result of the matrix multiplication.
         * @see NDWasmBlas.matMul
         *
         */
        matMul(other: NDArray): NDArray;
        /**
         * Computes the matrix power. This is a wrapper around `NDWasmBlas.matPow`.
         * @param {number} k The exponent.
         * @returns {NDArray} The result of the matrix power.
         * @see NDWasmBlas.matPow
         *
         */
        matPow(k: number): NDArray;
        /**
         * Performs batched matrix multiplication. This is a wrapper around `NDWasmBlas.matMulBatch`.
         * @param {NDArray} other The right-hand side batch of matrices.
         * @returns {NDArray} The result of the batched matrix multiplication.
         * @see NDWasmBlas.matMulBatch
         *
         */
        matMulBatch(other: NDArray): NDArray;
        /**
         * Performs matrix-vector multiplication. This is a wrapper around `NDWasmBlas.matVecMul`.
         * @param {NDArray} vec The vector to multiply by.
         * @returns {NDArray} The resulting vector.
         * @see NDWasmBlas.matVecMul
         *
         */
        matVecMul(vec: NDArray): NDArray;
        /**
         * Performs a symmetric rank-k update. This is a wrapper around `NDWasmBlas.syrk`.
         * @returns {NDArray} The resulting symmetric matrix.
         * @see NDWasmBlas.syrk
         *
         */
        syrk(): NDArray;
        /**
         * Computes the vector outer product. This is a wrapper around `NDWasmBlas.ger`.
         * @param {NDArray} other The other vector.
         * @returns {NDArray} The resulting matrix.
         * @see NDWasmBlas.ger
         *
         */
        ger(other: NDArray): NDArray;
        /**
         * Computes the Kronecker product. This is a wrapper around `NDWasmAnalysis.kronecker`.
         * @param {NDArray} other The other matrix.
         * @returns {NDArray} The result of the Kronecker product.
         * @see NDWasmAnalysis.kronecker
         *
         */
        kronecker(other: NDArray): NDArray;
        /**
         * Solves a system of linear equations. This is a wrapper around `NDWasmDecomp.solve`.
         * @param {NDArray} b The right-hand side matrix or vector.
         * @returns {NDArray} The solution matrix.
         * @see NDWasmDecomp.solve
         *
         */
        solve(b: NDArray): NDArray;
        /**
         * Computes the multiplicative inverse of the matrix. This is a wrapper around `NDWasmDecomp.inv`.
         * @returns {NDArray} The inverted matrix.
         * @see NDWasmDecomp.inv
         *
         */
        inv(): NDArray;
        /**
         * Computes the Moore-Penrose pseudo-inverse of the matrix. This is a wrapper around `NDWasmDecomp.pinv`.
         * @returns {NDArray} The pseudo-inverted matrix.
         * @see NDWasmDecomp.pinv
         *
         */
        pinv(): NDArray;
        /**
         * Computes the Singular Value Decomposition (SVD). This is a wrapper around `NDWasmDecomp.svd`.
         * @returns {{q: NDArray, r: NDArray}} An object containing the U, S, and V matrices.
         * @see NDWasmDecomp.svd
         *
         */
        svd(): {
            q: NDArray;
            r: NDArray;
        };
        /**
         * Computes the QR decomposition. This is a wrapper around `NDWasmDecomp.qr`.
         * @returns {Object} An object containing the Q and R matrices.
         * @see NDWasmDecomp.qr
         *
         */
        qr(): Object;
        /**
         * Computes the Cholesky decomposition. This is a wrapper around `NDWasmDecomp.cholesky`.
         * @returns {NDArray} The lower triangular matrix L.
         * @see NDWasmDecomp.cholesky
         *
         */
        cholesky(): NDArray;
        /**
         * Computes the determinant of the matrix. This is a wrapper around `NDWasmDecomp.det`.
         * @returns {number} The determinant.
         * @see NDWasmDecomp.det
         * @memberof NDWasmDecomp.prototype
         */
        det(): number;
        /**
         * Computes the log-determinant of the matrix. This is a wrapper around `NDWasmDecomp.logDet`.
         * @returns {{sign: number, logAbsDet: number}} An object containing the sign and log-absolute-determinant.
         * @see NDWasmDecomp.logDet
         * @memberof NDWasmDecomp.prototype
         */
        logDet(): {
            sign: number;
            logAbsDet: number;
        };
        /**
         * Computes the LU decomposition. This is a wrapper around `NDWasmDecomp.lu`.
         * @returns {NDArray} The LU matrix.
         * @see NDWasmDecomp.lu
         *
         */
        lu(): NDArray;
        /**
         * Computes the eigenvalues and eigenvectors of a general square matrix.
         * @returns {{values: NDArray, vectors: NDArray}} An object containing eigenvalues and eigenvectors.
         * @see NDWasmDecomp.eigen
         */
        eigen(): {
            values: NDArray;
            vectors: NDArray;
        };
        /**
         * Computes the 1D Fast Fourier Transform. This is a wrapper around `NDWasmSignal.fft`.
         * @this {NDArray} Complex array with shape [..., 2].
         * @returns {NDArray} Complex result with shape [..., 2].
         * @see NDWasmSignal.fft
         *
         */
        fft(this: NDArray): NDArray;
        /**
         * Computes the 1D Inverse Fast Fourier Transform. This is a wrapper around `NDWasmSignal.ifft`.
         * @this {NDArray} Complex array with shape [..., 2].
         * @returns {NDArray} Complex result with shape [..., 2].
         * @see NDWasmSignal.ifft
         *
         */
        ifft(this: NDArray): NDArray;
        /**
         * Computes the 1D Real-to-Complex Fast Fourier Transform. This is a wrapper around `NDWasmSignal.rfft`.
         * @this {NDArray} real input array.
         * @returns {NDArray} Complex result with shape [..., 2].
         * @see NDWasmSignal.rfft
         *
         */
        rfft(this: NDArray): NDArray;
        /**
         * 1D Complex-to-Real Inverse Fast Fourier Transform.
         * The input must be a complex array of shape [k, 2], where k is n/2 + 1. This is a wrapper around `NDWasmSignal.rifft`.
        *  @returns {NDArray} Real-valued time domain signal.
         * @this {NDArray} Complex frequency signal of shape [n/2 + 1, 2]
         * @param {number} n - Length of the original real signal.
         * @see NDWasmSignal.rifft
         *
         */
        rifft(this: NDArray, n: number): NDArray;
        /**
         * Computes the 2D Fast Fourier Transform. The input array must be 3D with shape [rows, cols, 2].
         * This is a wrapper around `NDWasmSignal.fft2`.
         * @returns {NDArray} 2D Complex result, with the same shape as input.
         * @see NDWasmSignal.fft2
         *
         */
        fft2(): NDArray;
        /**
         * 2D Inverse Complex-to-Complex Fast Fourier Transform.
         * The input array must be 3D with shape [rows, cols, 2].
         * The transform is performed in-place.
         * This is a wrapper around `NDWasmSignal.ifft2`.
         * @returns {NDArray} 2D Complex result, with the same shape as input.
         * @see NDWasmSignal.ifft2
         *
         */
        ifft2(): NDArray;
        /**
         * Computes the 1D Discrete Cosine Transform. This is a wrapper around `NDWasmSignal.dct`.
         * @returns {NDArray} The result of the DCT.
         * @see NDWasmSignal.dct
         *
         */
        dct(): NDArray;
        /**
         * Performs 2D spatial convolution. This is a wrapper around `NDWasmSignal.conv2d`.
         * @param {NDArray} kernel The convolution kernel.
         * @param {number} stride The stride.
         * @param {number} padding The padding.
         * @returns {NDArray} The convolved array.
         * @see NDWasmSignal.conv2d
         *
         */
        conv2d(kernel: NDArray, stride: number, padding: number): NDArray;
        /**
         * Performs 2D spatial cross-correlation. This is a wrapper around `NDWasmSignal.correlate2d`.
         * @param {NDArray} kernel The correlation kernel.
         * @param {number} stride The stride.
         * @param {number} padding The padding.
         * @returns {NDArray} The correlated array.
         * @see NDWasmSignal.correlate2d
         *
         */
        correlate2d(kernel: NDArray, stride: number, padding: number): NDArray;
        /**
         * Returns the indices that would sort the array. This is a wrapper around `NDWasmAnalysis.argsort`.
         * @returns {NDArray} An array of indices.
         * @see NDWasmAnalysis.argsort
         *
         */
        argsort(): NDArray;
        /**
         * Finds the top K largest or smallest elements. This is a wrapper around `NDWasmAnalysis.topk`.
         * @param {number} k The number of elements to find.
         * @param {boolean} largest Whether to find the largest or smallest elements.
         * @returns {{values: NDArray, indices: NDArray}} An object containing the values and indices of the top K elements.
         * @see NDWasmAnalysis.topk
         *
         */
        topk(k: number, largest: boolean): {
            values: NDArray;
            indices: NDArray;
        };
        /**
         * Computes the covariance matrix. This is a wrapper around `NDWasmAnalysis.cov`.
         * @returns {NDArray} The covariance matrix.
         * @see NDWasmAnalysis.cov
         *
         */
        cov(): NDArray;
        /**
         * Computes the matrix norm. This is a wrapper around `NDWasmAnalysis.norm`.
         * @param {number} type The type of norm to compute.
         * @returns {number} The norm of the matrix.
         * @see NDWasmAnalysis.norm
         *
         */
        norm(type: number): number;
        /**
         * Computes the rank of the matrix. This is a wrapper around `NDWasmAnalysis.rank`.
         * @param {number} tol The tolerance for singular values.
         * @returns {number} The rank of the matrix.
         * @see NDWasmAnalysis.rank
         *
         */
        rank(tol: number): number;
        /**
         * Computes the eigenvalue decomposition for a symmetric matrix. This is a wrapper around `NDWasmAnalysis.eigenSym`.
         * @param {boolean} vectors Whether to compute the eigenvectors.
         * @returns {{values: NDArray, vectors: NDArray|null}} An object containing the eigenvalues and eigenvectors.
         * @see NDWasmAnalysis.eigenSym
         *
         */
        eigenSym(vectors: boolean): {
            values: NDArray;
            vectors: NDArray | null;
        };
        /**
         * Estimates the reciprocal condition number of the matrix. This is a wrapper around `NDWasmAnalysis.cond`.
         * @param {number} norm The norm type.
         * @returns {number} The reciprocal condition number.
         * @see NDWasmAnalysis.cond
         *
         */
        cond(norm?: number): number;
        /**
         * Computes the pairwise distances between two sets of vectors. This is a wrapper around `NDWasmAnalysis.pairwiseDist`.
         * @param {NDArray} other The other set of vectors.
         * @returns {NDArray} The distance matrix.
         * @see NDWasmAnalysis.pairwiseDist
         *
         */
        pairwiseDist(other: NDArray): NDArray;
        /**
         * Performs K-Means clustering. This is a wrapper around `NDWasmAnalysis.kmeans`.
         * @param {number} k The number of clusters.
         * @param {number} maxIter The maximum number of iterations.
         * @returns {{centroids: NDArray,labels: NDArray,iterations: number}} An object containing the centroids, labels, and number of iterations.
         * @see NDWasmAnalysis.kmeans
         *
         */
        kmeans(k: number, maxIter: number): {
            centroids: NDArray;
            labels: NDArray;
            iterations: number;
        };
        _binaryOp(other: any, opFn: any, isInplace?: boolean): NDArray;
        _comparisonOp(other: any, compFn: any): NDArray;
    }
    import { WasmRuntime } from "ndwasm";
    import { NDWasmArray } from "ndwasmarray";
}
declare module "ndarray_helpers" {
    /**
     * Element-wise addition. Supports broadcasting.
     * @param {Array|TypedArray} l
     * @param {Array|TypedArray|number} r
     * @return {NDArray}
     * @see {NDArray#add}
     */
    export function add(l: any[] | TypedArray, r: any[] | TypedArray | number): NDArray;
    /**
     * Element-wise subtraction. Supports broadcasting.
     * @param {Array|TypedArray} l
     * @param {Array|TypedArray|number} r
     * @return {NDArray}
     * @see {NDArray#sub}
     */
    export function sub(l: any[] | TypedArray, r: any[] | TypedArray | number): NDArray;
    /**
     * Element-wise multiplication. Supports broadcasting.
     * @param {Array|TypedArray} l
     * @param {Array|TypedArray|number} r
     * @return {NDArray}
     * @see {NDArray#mul}
     */
    export function mul(l: any[] | TypedArray, r: any[] | TypedArray | number): NDArray;
    /**
     * Element-wise division. Supports broadcasting.
     * @param {Array|TypedArray} l
     * @param {Array|TypedArray|number} r
     * @return {NDArray}
     * @see {NDArray#div}
     */
    export function div(l: any[] | TypedArray, r: any[] | TypedArray | number): NDArray;
    /**
     * Element-wise exponentiation. Supports broadcasting.
     * @param {Array|TypedArray} l
     * @param {Array|TypedArray|number} r
     * @return {NDArray}
     * @see {NDArray#pow}
     */
    export function pow(l: any[] | TypedArray, r: any[] | TypedArray | number): NDArray;
    /**
     * Element-wise modulo. Supports broadcasting.
     * @param {Array|TypedArray} l
     * @param {Array|TypedArray|number} r
     * @return {NDArray}
     * @see {NDArray#mod}
     */
    export function mod(l: any[] | TypedArray, r: any[] | TypedArray | number): NDArray;
    /**
     * Bitwise AND.
     * @param {Array|TypedArray} l
     * @param {Array|TypedArray|number} r
     * @return {NDArray}
     * @see {NDArray#bitwise_and}
     */
    export function bitwise_and(l: any[] | TypedArray, r: any[] | TypedArray | number): NDArray;
    /**
     * Bitwise OR.
     * @param {Array|TypedArray} l
     * @param {Array|TypedArray|number} r
     * @return {NDArray}
     * @see {NDArray#bitwise_or}
     */
    export function bitwise_or(l: any[] | TypedArray, r: any[] | TypedArray | number): NDArray;
    /**
     * Bitwise XOR.
     * @param {Array|TypedArray} l
     * @param {Array|TypedArray|number} r
     * @return {NDArray}
     * @see {NDArray#bitwise_xor}
     */
    export function bitwise_xor(l: any[] | TypedArray, r: any[] | TypedArray | number): NDArray;
    /**
     * Bitwise left shift.
     * @param {Array|TypedArray} l
     * @param {Array|TypedArray|number} r
     * @return {NDArray}
     * @see {NDArray#bitwise_lshift}
     */
    export function bitwise_lshift(l: any[] | TypedArray, r: any[] | TypedArray | number): NDArray;
    /**
     * Bitwise (logical) right shift.
     * @param {Array|TypedArray} l
     * @param {Array|TypedArray|number} r
     * @return {NDArray}
     * @see {NDArray#bitwise_rshift}
     */
    export function bitwise_rshift(l: any[] | TypedArray, r: any[] | TypedArray | number): NDArray;
    /**
     * Bitwise NOT.
     * @param {Array|TypedArray} x
     * @return {NDArray}
     * @see {NDArray#bitwise_not}
     */
    export function bitwise_not(x: any[] | TypedArray): NDArray;
    /**
     * Numeric negation.
     * @param {Array|TypedArray} x
     * @return {NDArray}
     * @see {NDArray#neg}
     */
    export function neg(x: any[] | TypedArray): NDArray;
    /**
     * Absolute value.
     * @param {Array|TypedArray} x
     * @return {NDArray}
     * @see {NDArray#abs}
     */
    export function abs(x: any[] | TypedArray): NDArray;
    /**
     * Exponential function (e^x).
     * @param {Array|TypedArray} x
     * @return {NDArray}
     * @see {NDArray#exp}
     */
    export function exp(x: any[] | TypedArray): NDArray;
    /**
     * Square root.
     * @param {Array|TypedArray} x
     * @return {NDArray}
     * @see {NDArray#sqrt}
     */
    export function sqrt(x: any[] | TypedArray): NDArray;
    /**
     * Sine.
     * @param {Array|TypedArray} x
     * @return {NDArray}
     * @see {NDArray#sin}
     */
    export function sin(x: any[] | TypedArray): NDArray;
    /**
     * Cosine.
     * @param {Array|TypedArray} x
     * @return {NDArray}
     * @see {NDArray#cos}
     */
    export function cos(x: any[] | TypedArray): NDArray;
    /**
     * Tangent.
     * @param {Array|TypedArray} x
     * @return {NDArray}
     * @see {NDArray#tan}
     */
    export function tan(x: any[] | TypedArray): NDArray;
    /**
     * Natural logarithm (base e).
     * @param {Array|TypedArray} x
     * @return {NDArray}
     * @see {NDArray#log}
     */
    export function log(x: any[] | TypedArray): NDArray;
    /**
     * Ceiling (round up).
     * @param {Array|TypedArray} x
     * @return {NDArray}
     * @see {NDArray#ceil}
     */
    export function ceil(x: any[] | TypedArray): NDArray;
    /**
     * Floor (round down).
     * @param {Array|TypedArray} x
     * @return {NDArray}
     * @see {NDArray#floor}
     */
    export function floor(x: any[] | TypedArray): NDArray;
    /**
     * Round to nearest integer.
     * @param {Array|TypedArray} x
     * @return {NDArray}
     * @see {NDArray#round}
     */
    export function round(x: any[] | TypedArray): NDArray;
    /**
     * Dot Product
     * @param {Array|TypedArray} l
     * @param {Array|TypedArray} r
     * @return {NDArray}
     * @see {NDArray#dot}
     */
    export function dot(l: any[] | TypedArray, r: any[] | TypedArray): NDArray;
    /**
     * Cross Product
     * @param {Array|TypedArray} l
     * @param {Array|TypedArray} r
     * @return {NDArray#cross}
     */
    export function cross(l: any[] | TypedArray, r: any[] | TypedArray): NDArray;
    import { NDArray } from "ndarray_core";
}
declare module "ndarray" {
    export function init(baseDir?: string): Promise<void>;
    export const random: {
        _cryptoUniform01(size: any): Float64Array<any>;
        random(shape: any[], low?: number, high?: number, dtype?: string): NDArray;
        normal(shape: any[], mean?: number, std?: number, dtype?: string): NDArray;
        bernoulli(shape: any[], p?: number, dtype?: string): NDArray;
        exponential(shape: any[], lambda?: number, dtype?: string): NDArray;
        poisson(shape: any[], lambda?: number, dtype?: string): NDArray;
        _cast(data: any, dtype: any): any;
    };
    export const image: {
        decode(imageBytes: Uint8Array): NDArray | null;
        encode(ndarray: NDArray, { format, quality }?: {
            format?: string | undefined;
            quality?: number | undefined;
        }): Uint8Array | null;
        encodePng(ndarray: NDArray): Uint8Array | null;
        encodeJpeg(ndarray: NDArray, options?: {
            quality?: number | undefined;
        }): Uint8Array | null;
        convertUint8ArrrayToDataurl(uint8array: Uint8Array, mimeType?: string): string;
    };
    export const optimize: {
        linprog(c: NDArray, G: NDArray | null, h: NDArray | null, A: NDArray | null, b: NDArray | null, bounds: any[]): {
            x: NDArray;
            fun: number;
            status: number;
            message: string;
        };
        linearRegression(x: NDArray, y: NDArray): {
            alpha: number;
            beta: number;
        };
        minimize(func: Function, x0: NDArray, options?: {
            grad?: Function | undefined;
        }): {
            x: NDArray;
            success: boolean;
            message: string;
        };
    };
    export const decomp: {
        solve(a: NDArray, b: NDArray): NDArray;
        inv(a: NDArray): NDArray;
        svd(a: NDArray): {
            u: NDArray;
            s: NDArray;
            v: NDArray;
        };
        qr(a: NDArray): {
            q: NDArray;
            r: NDArray;
        };
        cholesky(a: NDArray): NDArray;
        lu(a: NDArray): NDArray;
        pinv(a: NDArray): NDArray;
        det(a: NDArray): number;
        logDet(a: NDArray): {
            sign: number;
            logAbsDet: number;
        };
        eigen(a: NDArray): {
            values: NDArray;
            vectors: NDArray;
        };
    };
    export const analysis: {
        argsort(a: NDArray): NDArray;
        topk(a: NDArray, k: number, largest?: boolean): {
            values: NDArray;
            indices: NDArray;
        };
        cov(a: NDArray): NDArray;
        corr(a: NDArray): NDArray;
        norm(a: NDArray, type?: number): number;
        rank(a: NDArray, tol?: number): number;
        cond(a: NDArray, norm?: number): number;
        eigenSym(a: NDArray, computeVectors?: boolean): {
            values: NDArray;
            vectors: NDArray | null;
        };
        pairwiseDist(a: NDArray, b: NDArray): NDArray;
        kmeans(data: NDArray, k: number, maxIter?: number): {
            centroids: NDArray;
            labels: NDArray;
            iterations: number;
        };
        kronecker(a: any, b: any): NDArray;
    };
    export const blas: {
        trace(a: NDArray): number;
        matMul(a: NDArray, b: NDArray): NDArray;
        matPow(a: NDArray, k: any): NDArray;
        matMulBatch(a: NDArray, b: NDArray): NDArray;
        syrk(a: NDArray): NDArray;
        trsm(a: NDArray, b: NDArray, lower?: boolean): NDArray;
        matVecMul(a: NDArray, x: NDArray): NDArray;
        ger(x: NDArray, y: NDArray): NDArray;
    };
    export const signal: {
        fft(a: NDArray): NDArray;
        ifft(a: NDArray): NDArray;
        rfft(a: NDArray): NDArray;
        rifft(a: NDArray, n: number): NDArray;
        fft2(a: NDArray): NDArray;
        ifft2(a: NDArray): NDArray;
        dct(a: NDArray): NDArray;
        conv2d(img: NDArray, kernel: NDArray, stride?: number, padding?: number): NDArray;
        correlate2d(img: NDArray, kernel: NDArray, stride?: number, padding?: number): NDArray;
    };
    export { Jit } from "./ndarray_jit.js";
    export * from "ndwasm";
    export * from "ndarray_factory";
    export * from "ndarray_helpers";
    export default NDArray;
    import { help } from "help";
    import { NDProb } from "ndarray_prob";
    import { NDWasmDecomp } from "ndwasm_decomp";
    import { NDWasmAnalysis } from "ndwasm_analysis";
    import { NDWasmBlas } from "ndwasm_blas";
    import { NDWasmSignal } from "ndwasm_signal";
    import { NDWasmImage } from "ndwasm_image";
    import { NDWasmOptimize } from "ndwasm_optimize";
    import { NDArray } from "ndarray_core";
    import { DTYPE_MAP } from "ndarray_core";
    import { NDWasmArray } from "ndwasmarray";
    export { help, NDProb, NDWasmDecomp, NDWasmAnalysis, NDWasmBlas, NDWasmSignal, NDWasmImage, NDWasmOptimize, NDArray, DTYPE_MAP, NDWasmArray };
}
