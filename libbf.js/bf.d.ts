declare module "complex" {
    /**
     * Creates a new Complex instance.
     * @param {number|string|BigFloat|Complex} re - Real part or Complex object
     * @param {number|string|BigFloat} [im] - Imaginary part
     * @returns {Complex}
     */
    export function complex(re: number | string | BigFloat | Complex, im?: number | string | BigFloat): Complex;
    /**
     * High-Precision Complex Number Utility Class.
     * Provides arithmetic and transcendental functions for complex numbers using BigFloat.
     *
     * @class
     * @property {BigFloat} re - The real part.
     * @property {BigFloat} im - The imaginary part.
     */
    export class Complex {
        /**
         * Creates a complex number from polar coordinates.
         * @param {number|string|BigFloat} r - The radius.
         * @param {number|string|BigFloat} theta - The angle.
         * @returns {Complex}
         */
        static fromPolar(r: number | string | BigFloat, theta: number | string | BigFloat): Complex;
        /**
         * Creates a complex number from a string.
         * @param {string} s
         * @returns {Complex}
         */
        static fromString(s: string): Complex;
        /**
         * @param {number|string|BigFloat|Complex} re - Real part or Complex object
         * @param {number|string|BigFloat} [im=undefined] - Imaginary part
         */
        constructor(re: number | string | BigFloat | Complex, im?: number | string | BigFloat);
        re: BigFloat | undefined;
        im: BigFloat | null | undefined;
        /**
         * Adds another complex number.
         * @param {Complex|number|string|BigFloat} other
         * @returns {Complex}
         */
        add(other: Complex | number | string | BigFloat): Complex;
        /**
         * Subtracts another complex number.
         * @param {Complex|number|string|BigFloat} other
         * @returns {Complex}
         */
        sub(other: Complex | number | string | BigFloat): Complex;
        /**
         * Multiplies by another complex number.
         * @param {Complex|number|string|BigFloat} other
         * @returns {Complex}
         */
        mul(other: Complex | number | string | BigFloat): Complex;
        /**
         * Divides by another complex number.
         * @param {Complex|number|string|BigFloat} other
         * @returns {Complex}
         */
        div(other: Complex | number | string | BigFloat): Complex;
        /**
         * Raises this complex number to the power of another.
         * @param {Complex|number|string|BigFloat} other
         * @returns {Complex}
         */
        pow(other: Complex | number | string | BigFloat): Complex;
        /**
         * Magnitude (Absolute value) |z|
         * @returns {BigFloat}
         */
        abs(): BigFloat;
        /**
         * Argument (Angle) arg(z)
         * @returns {BigFloat}
         */
        arg(): BigFloat;
        /**
         * Complex Square Root sqrt(z)
         * @returns {Complex}
         */
        sqrt(): Complex;
        /**
         * Complex Exponential e^z
         * @returns {Complex}
         */
        exp(): Complex;
        /**
         * Complex Natural Logarithm ln(z)
         * @returns {Complex}
         */
        log(): Complex;
        /**
         * Trigonometric Sine sin(z)
         * sin(x+iy) = sin(x)cosh(y) + i cos(x)sinh(y)
         * @returns {Complex}
         */
        sin(): Complex;
        /**
         * Trigonometric Cosine cos(z)
         * cos(x+iy) = cos(x)cosh(y) - i sin(x)sinh(y)
         * @returns {Complex}
         */
        cos(): Complex;
        /**
         * Trigonometric Tangent tan(z)
         * @returns {Complex}
         */
        tan(): Complex;
        /**
         * Hyperbolic Sine sinh(z)
         * sinh(x+iy) = sinh(x)cos(y) + i cosh(x)sin(y)
         * @returns {Complex}
         */
        sinh(): Complex;
        /**
         * Hyperbolic Cosine cosh(z)
         * cosh(x+iy) = cosh(x)cos(y) + i sinh(x)sin(y)
         * @returns {Complex}
         */
        cosh(): Complex;
        /**
         * Hyperbolic Tangent tanh(z)
         * @returns {Complex}
         */
        tanh(): Complex;
        /**
         * Inverse Sine asin(z) = -i * ln(iz + sqrt(1 - z^2))
         * @returns {Complex}
         */
        asin(): Complex;
        /**
         * Inverse Cosine acos(z) = -i * ln(z + i*sqrt(1 - z^2))
         * @returns {Complex}
         */
        acos(): Complex;
        /**
         * Inverse Tangent atan(z) = (i/2) * ln((i+z)/(i-z))
         * @returns {Complex}
         */
        atan(): Complex;
        /**
         * Inverse Hyperbolic Sine asinh(z) = ln(z + sqrt(z^2 + 1))
         * @returns {Complex}
         */
        asinh(): Complex;
        /**
         * Inverse Hyperbolic Cosine acosh(z) = ln(z + sqrt(z^2 - 1))
         * @returns {Complex}
         */
        acosh(): Complex;
        /**
         * Inverse Hyperbolic Tangent atanh(z) = 0.5 * ln((1+z)/(1-z))
         * @returns {Complex}
         */
        atanh(): Complex;
        /**
         * Returns the complex conjugate.
         * @returns {Complex}
         */
        conj(): Complex;
        /**
         * Negates the complex number.
         * @returns {Complex}
         */
        neg(): Complex;
        /**
         * Checks for equality with another complex number.
         * @param {Complex|number|string|BigFloat} b
         * @returns {boolean}
         */
        equals(b: Complex | number | string | BigFloat): boolean;
        /**
         * Checks if the complex number is almost zero.
         * @returns {boolean}
         */
        isAlmostZero(): boolean;
        /**
         * Checks if the complex number is exactly zero.
         * @returns {boolean}
         */
        isZero(): boolean;
        /**
         * Converts the complex number to a string.
         * @param {number} [base=10]
         * @param {number} [precision=-1]
         * @param {boolean} [pretty=false] pretty print
         * @returns {string}
         */
        toString(base?: number, precision?: number, pretty?: boolean): string;
        /**
         * Wraps a value in a Complex object if it isn't one already.
         * @private
         * @param {Complex|number|string|BigFloat} v
         * @returns {Complex}
         */
        private _wrap;
        /** @type {function(Complex|number|string|BigFloat): Complex} */
        operatorAdd: (arg0: Complex | number | string | BigFloat) => Complex;
        /** @type {function(Complex|number|string|BigFloat): Complex} */
        operatorSub: (arg0: Complex | number | string | BigFloat) => Complex;
        /** @type {function(Complex|number|string|BigFloat): Complex} */
        operatorMul: (arg0: Complex | number | string | BigFloat) => Complex;
        /** @type {function(Complex|number|string|BigFloat): Complex} */
        operatorDiv: (arg0: Complex | number | string | BigFloat) => Complex;
        /** @type {function(Complex|number|string|BigFloat): Complex} */
        operatorPow: (arg0: Complex | number | string | BigFloat) => Complex;
        /** @type {function(): Complex} */
        operatorNeg: () => Complex;
    }
    import { BigFloat } from "bf";
}
declare module "vector" {
    /**
     * High-Precision Vector class optimized for BigFloat arithmetic.
     * Represents a mathematical column vector.
     * @class
     */
    export class Vector {
        /**
         * Initializes a Vector from a size, an existing array, or another Vector.
         * @param {number|Array<number|string|BigFloat>|Vector} data
         */
        constructor(data: number | Array<number | string | BigFloat> | Vector);
        values: any[];
        /**
         * Returns a string representation of the vector.
         * Truncates the middle part with "..." if the length exceeds maxItem.
         *
         * @param {number} radix - The base for number representation (e.g., 10).
         * @param {number} prec - The number of decimal places for BigFloat.
         * @param {number} maxItem - Maximum number of elements to show before truncating.
         * @returns {string}
         */
        toString(radix?: number, prec?: number, maxItem?: number): string;
        /**
         * Dimension of the vector.
         * @returns {number}
         */
        get length(): number;
        /**
         * Gets the value at index i.
         * @param {number} i
         * @returns {BigFloat}
         */
        get(i: number): BigFloat;
        /**
         * Sets the value at index i.
         * @param {number} i
         * @param {number|string|BigFloat} val
         */
        set(i: number, val: number | string | BigFloat): void;
        /**
         * Adds another vector to this vector (v = this + other).
         * @param {Vector} other
         * @returns {Vector}
         */
        add(other: Vector): Vector;
        /**
         * Subtracts another vector from this vector (v = this - other).
         * @param {Vector} other
         * @returns {Vector}
         */
        sub(other: Vector): Vector;
        /**
         * Multiplies the vector by a scalar (v = this * scalar).
         * @param {number|string|BigFloat} scalar
         * @returns {Vector}
         */
        scale(scalar: number | string | BigFloat): Vector;
        /**
         * Computes the dot product of this vector and another vector.
         * @param {Vector} other
         * @returns {BigFloat}
         */
        dot(other: Vector): BigFloat;
        /**
         * Computes the L2 norm (Euclidean norm) of the vector.
         * @returns {BigFloat}
         */
        norm(): BigFloat;
        /**
         * Converts the vector to a standard Javascript Array of BigFloats.
         * @returns {BigFloat[]}
         */
        toArray(): BigFloat[];
        /**
         * Deep clones the vector.
         * @returns {Vector}
         */
        clone(): Vector;
    }
    import { BigFloat } from "bf";
}
declare module "matrix" {
    /**
     * Compressed Sparse Column (CSC) Matrix.
     * Exclusively optimized for BigFloat arithmetic.
     *
     * In CSC format, the matrix is specified by three arrays:
     * - values: Non-zero elements of the matrix.
     * - rowIndices: The row indices corresponding to the values.
     * - colPointers: The start and end indices in `values` and `rowIndices` for each column.
     *
     * @class
     */
    export class SparseMatrixCSC {
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
        static fromCOO(rows: number, cols: number, rowIdx: number[], colIdx: number[], vals: (number | string | BigFloat)[]): SparseMatrixCSC;
        /**
         * Converts a Dense Matrix (2D Array) into a Sparse CSC Matrix.
         * @param {(number|string|BigFloat)[][]} matrix
         * @returns {SparseMatrixCSC}
         */
        static fromDense(matrix: (number | string | BigFloat)[][]): SparseMatrixCSC;
        /**
         * @param {number} rows - Number of rows.
         * @param {number} cols - Number of columns.
         * @param {BigFloat[]} values - Array of non-zero BigFloat values.
         * @param {Uint32Array|number[]} rowIndices - Row indices for each non-zero value.
         * @param {Uint32Array|number[]} colPointers - Column pointers of length (cols + 1).
         */
        constructor(rows: number, cols: number, values: BigFloat[], rowIndices: Uint32Array | number[], colPointers: Uint32Array | number[]);
        rows: number;
        cols: number;
        values: BigFloat[];
        rowIndices: Uint32Array<ArrayBufferLike>;
        colPointers: Uint32Array<ArrayBufferLike>;
        /**
         * Returns a string representation of the matrix.
         * Uses the existing `get(row, col)` method and `nnz` property.
         *
         * @param {number} radix - The base for number representation (e.g., 10).
         * @param {number} prec - The number of decimal places for BigFloat.
         * @param {number} maxRowItem - Maximum number of rows/cols to show before truncating with "...".
         * @returns {string}
         */
        toString(radix?: number, prec?: number, maxRowItem?: number): string;
        /**
         * Number of non-zero elements.
         * @returns {number}
         */
        get nnz(): number;
        /**
         * Retrieves the value at the specified row and column.
         * Uses binary search within the specific column for O(log(nnz_in_col)) performance.
         *
         * @param {number} row
         * @param {number} col
         * @returns {BigFloat}
         */
        get(row: number, col: number): BigFloat;
        /**
         * Sets the value at the specified row and column.
         * Warning: Modifying the structure of a CSC matrix is O(nnz).
         * If you are building a matrix, it is highly recommended to use `fromCOO` instead.
         *
         * @param {number} row
         * @param {number} col
         * @param {number|string|BigFloat} val
         */
        set(row: number, col: number, val: number | string | BigFloat): void;
        /**
         * Eliminates explicit structural zeros from the matrix.
         * Sparse operations might leave explicit zeros to avoid shifting arrays.
         * Call this method to compact the matrix memory.
         */
        prune(): void;
        /**
         * Transposes the matrix.
         * Converts an M x N CSC matrix to an N x M CSC matrix.
         * This algorithm executes in O(nnz + max(rows, cols)) time.
         *
         * @returns {SparseMatrixCSC}
         */
        transpose(): SparseMatrixCSC;
        /**
         * Creates a deep copy of the matrix.
         * @returns {SparseMatrixCSC}
         */
        clone(): SparseMatrixCSC;
        /**
         * Converts the CSC Sparse Matrix back to a Dense Matrix (2D Array of BigFloat).
         * @returns {BigFloat[][]}
         */
        toDense(): BigFloat[][];
        /**
         * Internal implementation for Matrix Addition and Subtraction.
         * Utilizes a highly optimized Two-Pointer Merge algorithm, running in O(nnz(A) + nnz(B)) time.
         *
         * @private
         * @param {SparseMatrixCSC} other - The other matrix.
         * @param {boolean} isSub - True if subtraction (A - B), False if addition (A + B).
         * @returns {SparseMatrixCSC}
         */
        private _addSub;
        /**
         * Adds another SparseMatrixCSC to this matrix (C = A + B).
         * @param {SparseMatrixCSC} other
         * @returns {SparseMatrixCSC}
         */
        add(other: SparseMatrixCSC): SparseMatrixCSC;
        /**
         * Subtracts another SparseMatrixCSC from this matrix (C = A - B).
         * @param {SparseMatrixCSC} other
         * @returns {SparseMatrixCSC}
         */
        sub(other: SparseMatrixCSC): SparseMatrixCSC;
        /**
         * Matrix-Matrix Multiplication (C = A * B).
         * Implements Gustavson's Algorithm (Sparse Accumulator variant).
         * Computes the product in O(flops + nnz(C)) time footprint with zero inner-loop allocation.
         *
         * @param {SparseMatrixCSC} B - The right-hand side matrix.
         * @returns {SparseMatrixCSC}
         */
        mul(B: SparseMatrixCSC): SparseMatrixCSC;
        /**
         * Matrix-Vector Multiplication (y = A * x).
         * Linear time execution: O(nnz(A)).
         *
         * @param {Vector|Array<number|string|BigFloat>} vec - The input vector.
         * @returns {Vector} - The result vector.
         */
        mulVec(vec: Vector | Array<number | string | BigFloat>): Vector;
        /**
         * Extracts the diagonal elements of the matrix.
         * @returns {Vector} - A vector containing the diagonal elements.
         */
        getDiagonal(): Vector;
        /**
         * Computes the Trace of the matrix (sum of diagonal elements).
         * @returns {BigFloat}
         */
        trace(): BigFloat;
        /**
         * Computes the L1 Norm (Maximum absolute column sum).
         * Executes in O(nnz) time.
         * @returns {BigFloat}
         */
        norm1(): BigFloat;
        /**
         * Computes the Infinity Norm (Maximum absolute row sum).
         * Executes in O(nnz) time footprint.
         * @returns {BigFloat}
         */
        normInf(): BigFloat;
        /**
         * Computes the Frobenius Norm (Square root of the sum of the squares of elements).
         * @returns {BigFloat}
         */
        normF(): BigFloat;
        /**
         * Forward Substitution to solve L * x = b.
         * Assumes this matrix is strictly a Lower Triangular matrix.
         * Column-Oriented approach for CSC layout. O(nnz) time.
         *
         * @param {Vector|Array<number|string|BigFloat>} b - The right-hand side vector.
         * @returns {Vector} x - The solution vector.
         */
        solveLowerTriangular(b: Vector | Array<number | string | BigFloat>): Vector;
        /**
         * Backward Substitution to solve U * x = b.
         * Assumes this matrix is strictly an Upper Triangular matrix.
         * Column-Oriented approach for CSC layout. O(nnz) time.
         *
         * @param {Vector|Array<number|string|BigFloat>} b - The right-hand side vector.
         * @returns {Vector} x - The solution vector.
         */
        solveUpperTriangular(b: Vector | Array<number | string | BigFloat>): Vector;
        /**
         * Conjugate Gradient (CG) Method.
         * Solves the linear system A * x = b for Symmetric Positive Definite (SPD) matrices.
         *
         * @param {Vector|Array<number|string|BigFloat>} b - The right-hand side vector.
         * @param {number|string|BigFloat}[tol="1e-20"] - Convergence tolerance.
         * @param {number} [maxIter] - Maximum number of iterations. Defaults to matrix dimension.
         * @returns {Vector} x - The estimated solution vector.
         */
        solveCG(b: Vector | Array<number | string | BigFloat>, tol?: number | string | BigFloat, maxIter?: number): Vector;
        /**
         * Extracts the Jacobi Preconditioner (Inverse of the Diagonal).
         * This is the most memory-efficient and widely used preconditioner for diagonally dominant matrices.
         *
         * @returns {Vector} - A vector representing the diagonal inverse M^{-1}.
         */
        getJacobiPreconditioner(): Vector;
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
        solveBiCGSTAB(b: Vector | Array<number | string | BigFloat>, tol?: number | string | BigFloat, maxIter?: number, precond?: Vector): Vector;
        /**
         * Symmetric Rank-k Update (SYRK).
         * Computes the matrix product A * A^T.
         *
         * @returns {SparseMatrixCSC}
         */
        syrk(): SparseMatrixCSC;
        /**
         * General Rank-1 Update (GER).
         * Computes A + x * y^T.
         * Note: Rank-1 updates on sparse matrices usually introduce significant fill-in (dense data).
         *
         * @param {Vector|Array<number|string|BigFloat>} x
         * @param {Vector|Array<number|string|BigFloat>} y
         * @returns {SparseMatrixCSC}
         */
        ger(x: Vector | Array<number | string | BigFloat>, y: Vector | Array<number | string | BigFloat>): SparseMatrixCSC;
        /**
         * Triangular Solve with Multiple Right-Hand Sides (TRSM).
         * Solves A * X = B, where A is this triangular matrix.
         *
         * @param {SparseMatrixCSC} B - The right-hand side sparse matrix.
         * @param {boolean} [lower=true] - True if A is lower triangular, False if upper triangular.
         * @returns {SparseMatrixCSC} X - The solution sparse matrix.
         */
        trsm(B: SparseMatrixCSC, lower?: boolean): SparseMatrixCSC;
        /**
         * Sparse LU Factorization (Left-Looking / Gilbert-Peierls Algorithm).
         * Computes A = L * U where L is lower triangular with unit diagonal, and U is upper triangular.
         *
         * @returns {{L: SparseMatrixCSC, U: SparseMatrixCSC}}
         */
        lu(): {
            L: SparseMatrixCSC;
            U: SparseMatrixCSC;
        };
        /**
         * Sparse Cholesky Factorization.
         * Computes A = L * L^T for Symmetric Positive Definite (SPD) matrices.
         * Extracts factor from the LU Decomposition by scaling L with sqrt(diag(U)).
         *
         * @returns {SparseMatrixCSC} L - The lower triangular Cholesky factor.
         */
        cholesky(): SparseMatrixCSC;
        /**
         * General Direct Solver for A * x = b.
         * Uses the exact Sparse LU Factorization to compute the solution.
         *
         * @param {Vector|Array<number|string|BigFloat>} b
         * @returns {Vector} x
         */
        solve(b: Vector | Array<number | string | BigFloat>): Vector;
        /**
         * Computes the Determinant of the sparse matrix using LU factorization.
         * det(A) = Product of the diagonals of U.
         *
         * @returns {BigFloat}
         */
        det(): BigFloat;
        /**
         * Computes the Log-Determinant of the matrix (useful for Gaussians and PDFs).
         * logDet(A) = Sum of the logs of the absolute diagonals of U.
         *
         * @returns {BigFloat}
         */
        logDet(): BigFloat;
        /**
         * Computes the Inverse of the sparse matrix.
         * Warning: The inverse of a sparse matrix is typically dense.
         * This uses column-by-column LU solves to construct the inverse dynamically.
         *
         * @returns {SparseMatrixCSC}
         */
        inv(): SparseMatrixCSC;
        /**
         * Computes the Moore-Penrose Pseudoinverse (A^+).
         * Uses Normal Equations approach for sparse matrices to avoid full SVD overhead.
         *
         * @returns {SparseMatrixCSC}
         */
        pinv(): SparseMatrixCSC;
        /**
         * Sparse QR Factorization using Left-Looking Modified Gram-Schmidt (MGS).
         * Computes A = Q * R, where Q is orthogonal and R is upper triangular.
         *
         * @returns {{Q: SparseMatrixCSC, R: SparseMatrixCSC}}
         */
        qr(): {
            Q: SparseMatrixCSC;
            R: SparseMatrixCSC;
        };
        /**
         * Computes the Dominant Eigenpair using Power Iteration.
         * In sparse libraries, eigenvalue solvers extract top-K values iteratively.
         *
         * @param {number|string|BigFloat} [tol="1e-20"] - Convergence tolerance.
         * @param {number} [maxIter=1000] - Maximum iterations.
         * @returns {{eigenvalue: BigFloat, eigenvector: Vector}}
         */
        eigen(tol?: number | string | BigFloat, maxIter?: number): {
            eigenvalue: BigFloat;
            eigenvector: Vector;
        };
        /**
         * Computes the Dominant Singular Value and Vectors using Golub-Kahan (Power Method on A^T A).
         * Extracts the Top-1 Singular component.
         *
         * @param {number|string|BigFloat}[tol="1e-20"]
         * @param {number} [maxIter=1000]
         * @returns {{singularValue: BigFloat, u: Vector, v: Vector}}
         */
        svd(tol?: number | string | BigFloat, maxIter?: number): {
            singularValue: BigFloat;
            u: Vector;
            v: Vector;
        };
    }
    import { BigFloat } from "bf";
    import { Vector } from "vector";
}
declare module "polyfit" {
    /**
     * High-precision Polynomial Curve Fitting using Least Squares Method.
     *
     * Finds the coefficients of a polynomial p(x) of degree n that fits the data,
     * minimizing the sum of the squared errors.
     *
     * Coefficients are returned in descending powers (MATLAB style):
     * p(x) = c[0]*x^n + c[1]*x^(n-1) + ... + c[n-1]*x + c[n]
     *
     * @param {Array<number|string|BigFloat>} x - Array of x-coordinates.
     * @param {Array<number|string|BigFloat>} y - Array of y-coordinates.
     * @param {number} order - The degree of the polynomial to fit (n).
     *
     * @param {Object} [info={}] - Configuration and Status object.
     * @param {number} [info.max_time=60000] - Maximum execution time in milliseconds.
     * @param {boolean} [info.debug] - Enable debug logging.
     *
     *        // --- Output Status Properties ---
     * @param {Array<BigFloat>|null} info.result - Array of coefficients (descending order).
     * @param {BigFloat} info.ssr - Sum of Squared Residuals (Error).
     * @param {BigFloat} info.r_squared - Coefficient of determination (0 to 1).
     * @param {BigFloat} info.rmse - Root Mean Square Error.
     * @param {number} info.exectime - Elapsed execution time.
     * @param {Function} info.toString - Helper to format the result.
     *
     * @returns {Array<BigFloat>|null}
     *        Returns array of BigFloat coefficients if successful, null otherwise.
     */
    export function polyfit(x: Array<number | string | BigFloat>, y: Array<number | string | BigFloat>, order: number, info?: {
        max_time?: number | undefined;
        debug?: boolean | undefined;
        result: Array<BigFloat> | null;
        ssr: BigFloat;
        r_squared: BigFloat;
        rmse: BigFloat;
        exectime: number;
        toString: Function;
    }): Array<BigFloat> | null;
    /**
     * Helper: Solve Ax = b using Gaussian Elimination with Partial Pivoting
     * @param {Array<Array<BigFloat>>} A - Matrix (modified in place)
     * @param {Array<BigFloat>} b - Vector (modified in place)
     * @returns {Array<BigFloat>} x - Solution vector
     */
    export function solveLinearSystem(A: Array<Array<BigFloat>>, b: Array<BigFloat>): Array<BigFloat>;
    /**
     * Helper: Evaluate polynomial at x (MATLAB polyval style)
     * @param {Array<BigFloat>} p - Coefficients [c_n, ..., c_0]
     * @param {BigFloat|number} x
     * @returns {BigFloat}
     */
    export function polyval(p: Array<BigFloat>, x: BigFloat | number): BigFloat;
}
declare module "ode45" {
    /**
     * High-precision ODE Solver using Dormand-Prince method (similar to MATLAB's ode45).
     *
     * Solves non-stiff differential equations y' = f(t, y).
     * Implementation of the explicit Runge-Kutta (4,5) formula (Dormand-Prince pair).
     * Supports adaptive step size control.
     *
     * @param {Function} odefun - The main function to integrate: dydt = odefun(t, y).
     *        - t: BigFloat (current time)
     *        - y: Array<BigFloat> (current state vector)
     *        Returns: Array<BigFloat> (derivatives)
     *
     * @param {Array<number|string|BigFloat>} tspan - Interval of integration [t0, tf].
     *
     * @param {Array<number|string|BigFloat>|BigFloat} y0 - Initial conditions.
     *        Can be a scalar (converted to array internally) or an array of values.
     *
     * @param {Object} [info={}] - Configuration and Status object.
     *        Updates in-place with solution data and statistics.
     *
     *        // --- Input Configuration Properties ---
     * @param {number|string|BigFloat} [info._e=1e-16] - Absolute Tolerence (AbsTol).
     * @param {number|string|BigFloat} [info._re=1e-16] - Relative Tolerance (RelTol).
     *        Error control: |e| <= max(RelTol * |y|, AbsTol)
     * @param {number|string|BigFloat} [info.initial_step] - Initial step size guess.
     *        If omitted, it is automatically estimated.
     * @param {number} [info.progress] - log progress every info.progress*100% progress
     * @param {number}[info.progressCb] - progress call back progressCb(pos:Number,t,y)
     * @param {number} [info.max_step=2000000] - Maximum number of steps allowed.
     * @param {number} [info.max_time=1200000] - Maximum execution time in milliseconds.
     * @param {Function} [info.cb] - Optional callback per step: cb(t, y).
     *
     *        // --- Output Status Properties ---
     * @param {Array<BigFloat>} info.t - Array of time points.
     * @param {Array<Array<BigFloat>>} info.y - Array of state vectors corresponding to info.t.
     * @param {number} info.steps - Total successful steps taken.
     * @param {number} info.failed_steps - Number of rejected steps (due to error tolerance).
     * @param {number} info.exectime - Execution time in ms.
     * @param {string} info.status - "done", "timeout", or "max_steps".
     *
     * @returns {Object|null}
     *        Returns { t, y } (references to info.t and info.y) if successful.
     *        Returns null if critical errors occur.
     */
    export function ode45(odefun: Function, tspan: Array<number | string | BigFloat>, y0: Array<number | string | BigFloat> | BigFloat, info?: {
        _e?: number | string | BigFloat;
        _re?: number | string | BigFloat;
        initial_step?: number | string | BigFloat;
        progress?: number | undefined;
        progressCb?: number | undefined;
        max_step?: number | undefined;
        max_time?: number | undefined;
        cb?: Function | undefined;
        t: Array<BigFloat>;
        y: Array<Array<BigFloat>>;
        steps: number;
        failed_steps: number;
        exectime: number;
        status: string;
    }): Object | null;
}
declare module "ode15s" {
    /**
     * High-precision Stiff ODE Solver (Equivalent to MATLAB's ode15s).
     * Variable-Order (1-5), Variable-Step Backward Differentiation Formulas (BDF).
     * Utilizes custom SparseMatrixCSC for high-performance implicit Newton-Raphson iterations.
     *
     * @param {Function} odefun - The main function: dydt = odefun(t, y) or returns { M, f }.
     * @param {Array<number|string|BigFloat>} tspan - Interval of integration [t0, tf].
     * @param {Array<number|string|BigFloat>|BigFloat} y0 - Initial conditions.
     * @param {Object}[info={}] - Configuration and Status object.
     *
     *        // --- Input Configuration Properties ---
     * @param {Function}[info.Jacobian] - Optional analytical Jacobian: J = Jac(t, y, f). Returns COO object {rowIdx, colIdx, vals} or 2D Array.
     * @param {number|string|BigFloat}[info._e="1e-15"] - Absolute Tolerance.
     * @param {number|string|BigFloat}[info._re=info._e] - Relative Tolerance.
     * @param {boolean}[info.estimate_error] - when true, info.global_error will return the estimated global error.
     * @param {number|string|BigFloat}[info.initial_step] - Initial step size guess.
     * @param {number}[info.progress] - log progress every info.progress*100% progress
     * @param {number}[info.progressCb] - progress call back progressCb(pos:Number,t,y)
     * @param {number}[info.max_step=2000000] - Maximum number of steps allowed.
     * @param {number}[info.max_time=1200000] - Maximum execution time in milliseconds.
     * @param {Function}[info.cb] - Optional callback per accepted step: cb(t, y).
     *
     * @returns {Object|null} { t, y, dy } or null on catastrophic failure.
     */
    export function ode15s(odefun: Function, tspan: Array<number | string | BigFloat>, y0: Array<number | string | BigFloat> | BigFloat, info?: {
        Jacobian?: Function | undefined;
        _e?: string | number | BigFloat | undefined;
        _re?: string | number | BigFloat | undefined;
        estimate_error?: boolean | undefined;
        initial_step?: string | number | BigFloat | undefined;
        progress?: number | undefined;
        progressCb?: number | undefined;
        max_step?: number | undefined;
        max_time?: number | undefined;
        cb?: Function | undefined;
    }): Object | null;
    import { BigFloat } from "bf";
}
declare module "pdepe" {
    /**
     * High-performance 1D Parabolic and Elliptic PDE Solver (Equivalent to MATLAB's pdepe).
     * Solves equations of the form:
     * c(x,t,u,Du/Dx) * Du/Dt = x^(-m) * D/Dx( x^m * f(x,t,u,Du/Dx) ) + s(x,t,u,Du/Dx)
     *
     * NOTE: All user-provided functions (pdefun, icfun, bcfun) MUST return `BigFloat`
     * or `Array<BigFloat>` objects directly. No implicit type conversion is performed.
     *
     * @param {number|string} m - Symmetry parameter: 0 (slab), 1 (cylinder), 2 (sphere).
     * @param {Function} pdefun - Equation definitions: {c, f, s} = pdefun(x, t, u, dudx)
     * @param {Function} icfun - Initial conditions: u0 = icfun(x)
     * @param {Function} bcfun - Boundary conditions: {pl, ql, pr, qr} = bcfun(xl, ul, xr, ur, t)
     * @param {Array<number|string|BigFloat>} xmesh - Spatial grid points[x_0, x_1, ..., x_N]
     * @param {Array<number|string|BigFloat>} tspan - Time output points[t_0, t_1, ..., t_M]
     * @param {Object} [info={}] - Configuration and Status object forwarded to ode15s.
     *        @param {number|string|BigFloat}[info._e="1e-5"] - Absolute Tolerance.
     *        @param {number|string|BigFloat}[info._re="1e-4"] - Relative Tolerance.
     *        @param {number}[info.max_step=10000000] - Maximum number of steps allowed.
     *        @param {number}[info.max_time=10000000] - Maximum execution time in milliseconds.
     *        @param {Function}[info.cb] - Optional callback per accepted ODE step: cb(t, y).
     *        @param {number}[info.progress] - Log progress every info.progress*100%.
     *        @param {number}[info.progressCb] - progress call back progressCb(pos:Number,t,y)
     *        @param {boolean}[info.estimate_error] - Enable global error estimation tracking.
     *
     * @returns {Array<Array<BigFloat|Array<BigFloat>>>} - Solution 3D Array strictly returning BigFloat instances.
     */
    export function pdepe(m: number | string, pdefun: Function, icfun: Function, bcfun: Function, xmesh: Array<number | string | BigFloat>, tspan: Array<number | string | BigFloat>, info?: {
        _e?: string | number | BigFloat | undefined;
        _re?: string | number | BigFloat | undefined;
        max_step?: number | undefined;
        max_time?: number | undefined;
        cb?: Function | undefined;
        progress?: number | undefined;
        progressCb?: number | undefined;
        estimate_error?: boolean | undefined;
    }): Array<Array<BigFloat | Array<BigFloat>>>;
    import { BigFloat } from "bf";
}
declare module "fminbnd" {
    /**
     * High-precision Function Minimization using Brent's Method (similar to MATLAB's fminbnd).
     *
     * This algorithm finds a local minimum of a function of one variable within a fixed interval.
     * It combines Golden Section Search (linear convergence) with Parabolic Interpolation
     * (superlinear convergence) for reliability and speed.
     *
     * @param {Function} f - The objective function to minimize.
     *        Must accept a BigFloat argument and return a BigFloat result.
     *
     * @param {number|string|BigFloat} _ax - The start of the search interval.
     * @param {number|string|BigFloat} _bx - The end of the search interval.
     *
     *
     * @param {Object} [info={}] - Configuration and Status object.
     *        Updates in-place with statistics (iterations, execution time, error estimate).
     *        // --- Input Configuration Properties ---
     * @param {number|string|BigFloat} [info._e=1e-30] - Absolute Error Tolerance.
     *
     * @param {number|string|BigFloat} [info._re=info._e] - Relative Error Tolerance.
     *        The convergence criteria is based on the position x, not the function value.
     *        tol = |x| * _re + _e
     * @param {number} [info.max_step=500] - Maximum number of iterations allowed.
     *        If this limit is reached without convergence, the function returns null and logs a warning.
     * @param {number} [info.max_time=60000] - Maximum execution time in milliseconds.
     *        Prevents the function from hanging in infinite loops or extremely slow computations.
     * @param {Function} [info.cb] - Optional callback function.
     *        If defined, this function is called after every iteration. Useful for updating UI progress or logging.
     * @param {boolean} [info.debug] - Optional flag to enable debug logging (implementation dependent).
     *
     *        // --- Output Status Properties (Updated during execution) ---
     * @param {BigFloat|null} info.result - The final found root.
     *        Returns a BigFloat if converged, or null if failed.
     * @param {BigFloat} info.lastresult - The result of the last iteration (current best guess `b`).
     *        Even if convergence fails, this contains the value closest to the root when execution stopped.
     * @param {string} info.eff_result - String representation of the result based on effective precision.
     *        Generated by truncating `lastresult` according to `eff_decimal_precision`.
     * @param {number} info.steps - The number of iterations currently executed.
     * @param {number} info.exectime - The elapsed execution time in milliseconds.
     * @param {BigFloat} info.error - The estimated error bound.
     *        Typically represents half the width of the current search interval (`xm`).
     * @param {BigFloat} info.residual - The absolute value of the function at the current best guess: `|f(x)|`.
     *        Ideally, this value should be close to zero.
     * @param {number} info.eff_decimal_precision - Estimated number of significant decimal digits.
     *        Calculated as `-log10(error)`.
     * @param {Function} info.toString - Helper method.
     *        Returns a formatted string containing steps, error, residual, and execution time.
     *
     * @returns {BigFloat|null}
     *        Returns the BigFloat x where f(x) is minimized.
     *        Returns `null` if max steps or time limit exceeded.
     */
    export function fminbnd(f: Function, _ax: number | string | BigFloat, _bx: number | string | BigFloat, info?: {
        _e?: number | string | BigFloat;
        _re?: number | string | BigFloat;
        max_step?: number | undefined;
        max_time?: number | undefined;
        cb?: Function | undefined;
        debug?: boolean | undefined;
        result: BigFloat | null;
        lastresult: BigFloat;
        eff_result: string;
        steps: number;
        exectime: number;
        error: BigFloat;
        residual: BigFloat;
        eff_decimal_precision: number;
        toString: Function;
    }): BigFloat | null;
}
declare module "roots" {
    /**
     * Calculates the roots of a polynomial with high precision using the Durand-Kerner method.
     *
     * This function mimics MATLAB's `roots` command but supports arbitrary precision BigFloat numbers.
     * It solves for `x` in the polynomial equation:
     * c[0]*x^n + c[1]*x^(n-1) + ... + c[n] = 0
     *
     * The algorithm iterates simultaneously towards all `n` roots, naturally handling complex conjugate pairs.
     *
     * @param {Array<number|string|BigFloat|Complex>} _coeffs - The polynomial coefficients.
     *        Must be ordered from highest degree to lowest (e.g., [1, -5, 6] for x^2 - 5x + 6).
     *        Leading zeros are automatically removed.
     *
     * @param {Object} [info={}] - Configuration and Status object.
     *
     *        // --- Input Configuration ---
     * @param {number} [info.max_step=500] - Maximum number of iterations.
     *        Durand-Kerner usually converges quadratically, so 50-100 is typically sufficient for high precision.
     * @param {number} [info.max_time=60000] - Maximum execution time in milliseconds.
     * @param {number|string|BigFloat} [info._e=1e-30] - Convergence tolerance.
     *        The loop stops when the maximum change in any root position is smaller than this value.
     * @param {Function} [info.cb] - Optional callback function executed after each iteration.
     *
     *        // --- Output Status (Updated during execution) ---
     * @param {Array<{re:BigFloat, im:BigFloat}>|null} info.result - The final array of roots.
     * @param {number} info.steps - Current iteration count.
     * @param {number} info.exectime - Elapsed time in ms.
     * @param {BigFloat} info.error - The maximum correction (shift magnitude) applied in the last step.
     *        Used as a proxy for the current error bound.
     * @param {number} info.eff_decimal_precision - Estimated significant decimal digits based on convergence error.
     * @param {string} info.eff_result - A string summary of the first root (for debugging/display).
     * @param {Function} info.toString - Helper to print status summary.
     *
     * @returns {Array<Complex>|null}
     *          Returns an array of objects representing complex numbers {re, im}.
     *          Returns `null` if the solver fails to converge within limits.
     */
    export function roots(_coeffs: Array<number | string | BigFloat | Complex>, info?: {
        max_step?: number | undefined;
        max_time?: number | undefined;
        _e?: number | string | BigFloat;
        cb?: Function | undefined;
        result: Array<{
            re: BigFloat;
            im: BigFloat;
        }> | null;
        steps: number;
        exectime: number;
        error: BigFloat;
        eff_decimal_precision: number;
        eff_result: string;
        toString: Function;
    }): Array<Complex> | null;
    import { Complex } from "complex";
}
declare module "fzero" {
    /**
     * High-precision Root Finding using the Brent-Dekker Method (similar to MATLAB's fzero).
     *
     * This algorithm combines the reliability of Bisection, the speed of the Secant method,
     * and the high-order convergence of Inverse Quadratic Interpolation (IQI).
     * It guarantees global convergence while achieving superlinear convergence rates near the root.
     *
     * @param {Function} f - The target function to find the root of.
     *        Must accept a BigFloat argument and return a BigFloat result.
     *        The function values at the endpoints must have opposite signs (f(_a) * f(_b) <= 0).
     *
     * @param {number|string|BigFloat} _a - The start of the search interval (or first initial guess).
     * @param {number|string|BigFloat} _b - The end of the search interval (or second initial guess).
     *
     *
     * @param {Object} [info={}] - Configuration and Status object.
     *        This object configures the execution parameters and is updated in-place
     *        with statistical data during and after execution.
     * @param {number|string|BigFloat} [info._e=1e-30] - Absolute Error Tolerance.
     *        Convergence is considered achieved when the interval width or step size falls below this value.
     *
     * @param {number|string|BigFloat} [info._re=info._e] - Relative Error Tolerance.
     *        Used to handle convergence for large values. Defaults to the absolute tolerance.
     *        The effective tolerance is calculated as: `tol = |b| * _re + _e`.
     *
     *        // --- Input Configuration Properties ---
     * @param {number} [info.max_step=200] - Maximum number of iterations allowed.
     *        If this limit is reached without convergence, the function returns null and logs a warning.
     * @param {number} [info.max_time=60000] - Maximum execution time in milliseconds.
     *        Prevents the function from hanging in infinite loops or extremely slow computations.
     * @param {Function} [info.cb] - Optional callback function.
     *        If defined, this function is called after every iteration. Useful for updating UI progress or logging.
     * @param {boolean} [info.debug] - Optional flag to enable debug logging (implementation dependent).
     *
     *        // --- Output Status Properties (Updated during execution) ---
     * @param {BigFloat|null} info.result - The final found root.
     *        Returns a BigFloat if converged, or null if failed.
     * @param {BigFloat} info.lastresult - The result of the last iteration (current best guess `b`).
     *        Even if convergence fails, this contains the value closest to the root when execution stopped.
     * @param {string} info.eff_result - String representation of the result based on effective precision.
     *        Generated by truncating `lastresult` according to `eff_decimal_precision`.
     * @param {number} info.steps - The number of iterations currently executed.
     * @param {number} info.exectime - The elapsed execution time in milliseconds.
     * @param {BigFloat} info.error - The estimated error bound.
     *        Typically represents half the width of the current search interval (`xm`).
     * @param {BigFloat} info.residual - The absolute value of the function at the current best guess: `|f(x)|`.
     *        Ideally, this value should be close to zero.
     * @param {number} info.eff_decimal_precision - Estimated number of significant decimal digits.
     *        Calculated as `-log10(error)`.
     * @param {Function} info.toString - Helper method.
     *        Returns a formatted string containing steps, error, residual, and execution time.
     *
     * @returns {BigFloat|null}
     *        Returns the BigFloat root if the tolerance criteria are met.
     *        Returns `null` if the maximum steps or time limit is exceeded (a warning is logged to the console).
     */
    export function fzero(f: Function, _a: number | string | BigFloat, _b: number | string | BigFloat, info?: {
        _e?: number | string | BigFloat;
        _re?: number | string | BigFloat;
        max_step?: number | undefined;
        max_time?: number | undefined;
        cb?: Function | undefined;
        debug?: boolean | undefined;
        result: BigFloat | null;
        lastresult: BigFloat;
        eff_result: string;
        steps: number;
        exectime: number;
        error: BigFloat;
        residual: BigFloat;
        eff_decimal_precision: number;
        toString: Function;
    }): BigFloat | null;
}
declare module "quad" {
    /**
     * High-precision Numerical Integration using Tanh-Sinh (Double Exponential) Quadrature.
     *
     * SOTA level implementation equivalent to mpmath.quad.
     * - Natively supports finite, half-infinite, and fully infinite bounds.
     * - Supports Complex variables, endpoints, and complex integrands.
     * - Supports contour/path integration when `_a` is an array of nodes and `_b` is undefined.
     * - Extremely robust against endpoint singularities (e.g. log(0), 1/0) through smart boundary bypassing.
     *
     * @param {Function} f - The integrand function.
     *        Must accept a BigFloat/Complex argument (z) and return a BigFloat/Complex result (f(z)).
     *
     * @param {number|string|BigFloat|Complex|Array} _a - The lower limit of integration, or an array of points for contour integration.
     * @param {number|string|BigFloat|Complex} [_b] - The upper limit of integration (leave undefined for contour path integration).
     *
     * @param {Object} [info={}] - Configuration and Status object.
     * @param {number}[info._e=1e-50] - Absolute Error Tolerance.
     * @param {number}[info._re=info._e] - Relative Error Tolerance.
     * @param {number}[info.max_step=15] - Maximum number of interval halving steps.
     * @param {number}[info.max_time=60000] - Maximum execution time in milliseconds.
     * @param {Function}[info.cb] - Optional callback function executed after each level computation.
     * @param {boolean} [info.debug] - Optional flag to enable debug logging.
     *
     * @returns {BigFloat|Complex|null} Returns the exact integral value, or `null` if failed.
     */
    export function quad(f: Function, _a: number | string | BigFloat | Complex | any[], _b?: number | string | BigFloat | Complex, info?: {
        _e?: number | undefined;
        _re?: number | undefined;
        max_step?: number | undefined;
        max_time?: number | undefined;
        cb?: Function | undefined;
        debug?: boolean | undefined;
    }): BigFloat | Complex | null;
    export { quad as integral };
    import { Complex } from "complex";
}
declare module "romberg" {
    /**
     * High-precision Numerical Integration using Romberg's Method.
     *
     * This function estimates the definite integral of `f` over the interval `[_a, _b]`
     * using Richardson extrapolation applied to the Trapezoidal rule.
     * It iteratively refines the interval width and the order of the polynomial approximation
     * to achieve high precision with relatively few function evaluations.
     *
     * @param {Function} f - The integrand function.
     *        Must accept a BigFloat argument (x) and return a BigFloat result (f(x)).
     *
     * @param {number|string|BigFloat} _a - The lower limit of integration.
     * @param {number|string|BigFloat} _b - The upper limit of integration.
     *
     *
     * @param {Object} [info={}] - Configuration and Status object.
     *        Configures execution parameters and stores statistical data during/after execution.
     * @param {number} [info._e=1e-30] - Absolute Error Tolerance.
     *        The integration stops when the estimated absolute error falls below this threshold.
    
     * @param {number} [info._re=info._e] - Relative Error Tolerance.
     *        The integration stops when the estimated relative error falls below this threshold.
     *        (Condition: error <= _e || rerror <= _re)
     *        // --- Input Configuration Properties ---
     * @param {number} [info.max_step=25] - Maximum number of interval halving steps (rows in the Romberg table).
     *        Note: The number of function evaluations grows exponentially (2^steps).
     * @param {number} [info.max_acc=12] - Maximum extrapolation order (columns in the Romberg table).
     *        Limits the depth of Richardson extrapolation to prevent numerical instability from high-order polynomials.
     * @param {number} [info.max_time=60000] - Maximum execution time in milliseconds.
     * @param {Function} [info.cb] - Optional callback function executed after each row of the table is computed.
     * @param {boolean} [info.debug] - Optional flag to enable debug logging to the console.
     *
     *        // --- Output Status Properties (Updated during execution) ---
     * @param {BigFloat|null} info.result - The final calculated integral.
     *        Returns a BigFloat if converged, or null if failed.
     * @param {BigFloat} info.lastresult - The best estimate of the integral from the most recent iteration.
     * @param {string} info.eff_result - String representation of the result based on effective precision.
     * @param {number} info.steps - Current iteration number (row index `m`).
     *        Corresponds to dividing the interval into 2^(steps-1) segments.
     * @param {number} info.exectime - Elapsed execution time in milliseconds.
     * @param {BigFloat} info.error - Estimated absolute error.
     *        Calculated as the difference between the two most accurate extrapolations in the current row.
     * @param {BigFloat} info.rerror - Estimated relative error (`error / lastresult`).
     * @param {number} info.eff_decimal_precision - Estimated number of significant decimal digits.
     *        Calculated as `-log10(rerror)`.
     * @param {Function} info.toString - Helper method.
     *        Returns a formatted string containing steps, error, result, and execution time.
     *
     * @returns {BigFloat|null}
     *        Returns the BigFloat integral value if tolerances are met.
     *        Returns `null` if `max_step` or `max_time` is reached without convergence.
     */
    export function romberg(f: Function, _a: number | string | BigFloat, _b: number | string | BigFloat, info?: {
        _e?: number | undefined;
        _re?: number | undefined;
        max_step?: number | undefined;
        max_acc?: number | undefined;
        max_time?: number | undefined;
        cb?: Function | undefined;
        debug?: boolean | undefined;
        result: BigFloat | null;
        lastresult: BigFloat;
        eff_result: string;
        steps: number;
        exectime: number;
        error: BigFloat;
        rerror: BigFloat;
        eff_decimal_precision: number;
        toString: Function;
    }): BigFloat | null;
}
declare module "limit" {
    /**
     * High-precision Numerical Limit using Richardson Extrapolation.
     *
     * This function estimates the limit of `f(x)` as `x` approaches `point`.
     * It evaluates the function at a sequence of points converging to the target
     * and uses Richardson extrapolation to eliminate error terms, achieving high precision.
     *
     * @param {Function} f - The function to evaluate.
     *        Must accept a BigFloat argument (x) and return a BigFloat result (f(x)).
     *
     * @param {number|string|BigFloat} point - The target value to approach.
     *        Can be a finite number, or 'inf', '+inf', 'infinity' for positive infinity,
     *        '-inf', '-infinity' for negative infinity.
     *
     * @param {Object} [info={}] - Configuration and Status object.
     * @param {number} [info._e=1e-30] - Absolute Error Tolerance.
     * @param {number} [info._re=info._e] - Relative Error Tolerance.
     *
     *        // --- Input Configuration Properties ---
     * @param {boolean|Number} [info.useExp=false] - If true, uses exponential coordinate transformation.
     *        Useful for slowly converging limits (e.g. logarithmic) or limits at infinity.
     *        Use info.useExp as base number when it's a Number, otherwize use 2 as base number.
     *        For x->inf, substitutes x = pow(baseNumber,1/t).
     *        For x->c, substitutes x = c + dir * pow(baseNumber,-1/t).
     * @param {number} [info.max_step=100] - Maximum number of iterations (rows in the extrapolation table).
     * @param {number} [info.max_acc=15] - Maximum extrapolation order (columns in the table).
     * @param {number} [info.max_time=60000] - Maximum execution time in milliseconds.
     * @param {number} [info.direction=1] - Direction of approach for finite limits.
     *        1 for approaching from right (c + h), -1 for approaching from left (c - h).
     *        Ignored if point is infinite.
     * @param {Function} [info.cb] - Optional callback function executed after each iteration.
     * @param {boolean} [info.debug] - Optional flag to enable debug logging.
     *
     *        // --- Output Status Properties ---
     * @param {BigFloat|null} info.result - The final calculated limit.
     * @param {BigFloat} info.lastresult - The best estimate from the most recent iteration.
     * @param {string} info.eff_result - String representation based on effective precision.
     * @param {number} info.steps - Current iteration number.
     * @param {BigFloat} info.error - Estimated absolute error.
     * @param {BigFloat} info.rerror - Estimated relative error.
     * @param {number} info.eff_decimal_precision - Estimated significant digits.
     *
     * @returns {BigFloat|null}
     *        Returns the BigFloat limit value if tolerances are met, or null otherwise.
     */
    export function limit(f: Function, point: number | string | BigFloat, info?: {
        _e?: number | undefined;
        _re?: number | undefined;
        useExp?: number | boolean | undefined;
        max_step?: number | undefined;
        max_acc?: number | undefined;
        max_time?: number | undefined;
        direction?: number | undefined;
        cb?: Function | undefined;
        debug?: boolean | undefined;
        result: BigFloat | null;
        lastresult: BigFloat;
        eff_result: string;
        steps: number;
        error: BigFloat;
        rerror: BigFloat;
        eff_decimal_precision: number;
    }): BigFloat | null;
    /**
     * High-precision Numerical Differentiation.
     *
     * Computes the n-th derivative of function `f` at point `x` using the `limit` function
     * to extrapolate the finite difference quotient as the step size `h` approaches zero.
     *
     * @param {Function} f - The function to differentiate.
     *        Must accept and return BigFloat.
     * @param {number|string|BigFloat} x - The point at which to compute the derivative.
     * @param {number} [n=1] - The order of the derivative (1st, 2nd, etc.).
     * @param {Object} [info={}] - Configuration object.
     * @param {boolean} [info.singular=false] - If true, avoids evaluating f(x) exactly
     *        by shifting the sampling points (useful if x is a singularity).
     * @param {number} [info.direction=1] - Direction of the limit: 1 for forward, -1 for backward.
     *
     * @returns {BigFloat|null} The n-th derivative at x, or null if convergence fails.
     */
    export function diff(f: Function, x: number | string | BigFloat, n?: number, info?: {
        singular?: boolean | undefined;
        direction?: number | undefined;
    }): BigFloat | null;
}
declare module "nsum" {
    /**
     * High-precision Numerical Summation (Infinite/Finite Series).
     *
     * This function estimates the sum of `f(n)` for `n` from `start` to `end`.
     * For infinite series, it calculates a sequence of partial sums (doubling the number of terms each step)
     * and uses Richardson extrapolation to accelerate convergence.
     *
     * @param {Function} f - The term function.
     *        Must accept a BigFloat argument (n) and return a BigFloat result (f(n)).
     *
     * @param {Array<number|string|BigFloat>} range - The summation interval [start, end].
     *        e.g., [0, 100] or [1, 'inf'].
     *
     * @param {Object} [info={}] - Configuration and Status object.
     * @param {number} [info._e=1e-30] - Absolute Error Tolerance.
     * @param {number} [info._re=info._e] - Relative Error Tolerance.
     *
     *        // --- Input Configuration Properties ---
     * @param {number} [info.max_step=20] - Maximum number of doubling steps.
     *        Step m involves summing up to 2^m terms. Be careful increasing this,
     *        as computational cost doubles with each step.
     * @param {number} [info.max_acc=15] - Maximum extrapolation order.
     * @param {number} [info.max_time=60000] - Maximum execution time in milliseconds.
     * @param {Function} [info.cb] - Optional callback function executed after each iteration.
     * @param {boolean} [info.debug] - Optional flag to enable debug logging.
     *
     *        // --- Output Status Properties ---
     * @param {BigFloat|null} info.result - The final calculated sum.
     * @param {BigFloat} info.lastresult - The best estimate from the most recent iteration.
     * @param {string} info.eff_result - String representation based on effective precision.
     * @param {number} info.steps - Current iteration number (row index m).
     *        Total terms summed approx 2^steps.
     * @param {number} info.terms_count - Total number of terms explicitly evaluated.
     * @param {BigFloat} info.error - Estimated absolute error.
     * @param {BigFloat} info.rerror - Estimated relative error.
     * @param {number} info.eff_decimal_precision - Estimated significant digits.
     *
     * @returns {BigFloat|null}
     *        Returns the BigFloat sum if converged/completed, or null if failed.
     */
    export function nsum(f: Function, range: Array<number | string | BigFloat>, info?: {
        _e?: number | undefined;
        _re?: number | undefined;
        max_step?: number | undefined;
        max_acc?: number | undefined;
        max_time?: number | undefined;
        cb?: Function | undefined;
        debug?: boolean | undefined;
        result: BigFloat | null;
        lastresult: BigFloat;
        eff_result: string;
        steps: number;
        terms_count: number;
        error: BigFloat;
        rerror: BigFloat;
        eff_decimal_precision: number;
    }): BigFloat | null;
}
declare module "shanks" {
    /**
     * High-precision Sequence Limit using Wynn's Epsilon Algorithm (Shanks Transformation).
     *
     * This function estimates the limit of a sequence `f(n)` as `n` approaches infinity.
     * It is particularly effective for accelerating the convergence of slowly converging
     * alternating series or sequences.
     *
     * @param {Function|Array[number|string|bfjs.BigFloat]} f - The sequence generator.
     *        Must accept an integer (n) and return a BigFloat result (the n-th partial sum or term).
     *        f(0), f(1), f(2)... should generate the sequence to be accelerated.
     *
     * @param {Object} [info={}] - Configuration and Status object.
     * @param {number} [info._e=1e-30] - Absolute Error Tolerance.
     * @param {number} [info._re=info._e] - Relative Error Tolerance.
     * @param {number} [info.max_step=500] - Maximum number of terms to evaluate.
     * @param {number} [info.max_time=60000] - Maximum execution time in milliseconds.
     * @param {Function} [info.cb] - Optional callback function executed after each iteration.
     * @param {boolean} [info.debug] - Optional flag to enable debug logging.
     *
     *        // --- Output Status Properties ---
     * @param {BigFloat|null} info.result - The final calculated limit.
     * @param {BigFloat} info.lastresult - The best estimate from the most recent iteration.
     * @param {string} info.eff_result - String representation based on effective precision.
     * @param {number} info.steps - Current iteration number (n).
     * @param {BigFloat} info.error - Estimated absolute error.
     * @param {BigFloat} info.rerror - Estimated relative error.
     * @param {number} info.eff_decimal_precision - Estimated significant digits.
     *
     * @returns {BigFloat|null}
     *        Returns the BigFloat limit value if tolerances are met, or null otherwise.
     */
    export function shanks(f: Function | any, info?: {
        _e?: number | undefined;
        _re?: number | undefined;
        max_step?: number | undefined;
        max_time?: number | undefined;
        cb?: Function | undefined;
        debug?: boolean | undefined;
        result: BigFloat | null;
        lastresult: BigFloat;
        eff_result: string;
        steps: number;
        error: BigFloat;
        rerror: BigFloat;
        eff_decimal_precision: number;
    }): BigFloat | null;
}
declare module "identify" {
    /**
     * Identify a high-precision BigFloat as a simple symbolic expression.
     * This function uses the Continued Fraction algorithm to find the best rational
     * approximation p/q for x, x/pi, x^2, etc.
     *
     * @param {BigFloat|number|string} _x - The value to identify.
     * @param {Object} [options={}] - Configuration options.
     * @param {number|BigFloat} [options.tol] - Error tolerance (defaults to current precision's epsilon).
     * @param {number} [options.max_den=1000000] - Maximum denominator allowed for rational parts.
     * @param {Array} [options.constants] - Custom constants to check against.
     * @returns {string} - A string representation of the identified expression, or the number itself.
     */
    export function identify(_x: BigFloat | number | string, options?: {
        tol?: number | BigFloat;
        max_den?: number | undefined;
        constants?: any[] | undefined;
    }): string;
}
declare module "gamma" {
    /**
     * High-precision Log-Gamma function ln(Gamma(z)).
     * Essential for large z where Gamma(z) would overflow.
     *
     * @param {Complex|number|string|BigFloat} z
     * @returns {Complex}
     */
    export function logGamma(z: Complex | number | string | BigFloat): Complex;
    /**
     * High-precision Gamma function Γ(z).
     *
     * @param {Complex|number|string|BigFloat} z
     * @returns {Complex}
     */
    export function gamma(z: Complex | number | string | BigFloat): Complex;
    /**
     * Factorial function n! = Γ(n + 1)
     * @returns {Complex|BigFloat} return Complex when n is Complex, otherwize return a BigFloat
     */
    export function factorial(n: any): Complex | BigFloat;
    /**
     * Beta function B(x, y) = Γ(x)Γ(y) / Γ(x+y)
     */
    export function beta(x: any, y: any): bfjs.Complex;
    import { Complex } from "complex";
    import { BigFloat } from "bf";
    import * as bfjs from "bf";
}
declare module "zeta" {
    /**
     * Computes the Riemann Zeta function ζ(s) or Hurwitz Zeta function ζ(s, a).
     *
     * Logic:
     * 1. Handle pole at s = 1.
     * 2. If Re(s) < 0 and a = 1, use the Reflection Formula to mirror into the right half-plane.
     * 3. Handle specific Hurwitz reductions (e.g., a = 0.5).
     * 4. Use the Euler-Maclaurin summation formula for high-precision results.
     *
     * @param {Complex|number|BigFloat} s - The complex exponent.
     * @param {Complex|number|BigFloat} [a=1] - The shift parameter (default is 1 for Riemann Zeta).
     * @returns {Complex}
     */
    export function zeta(s: Complex | number | BigFloat, a?: Complex | number | BigFloat): Complex;
    /**
     * Dirichlet Eta function η(s) = (1 - 2^(1-s)) * ζ(s).
     *
     * @param {Complex|number|BigFloat} s
     * @returns {Complex}
     */
    export function altZeta(s: Complex | number | BigFloat): Complex;
    /**
     * Prime Zeta Function P(s) = sum_{p \in primes} p^-s.
     * Calculated via Mobius inversion: P(s) = sum_{n=1}^inf (μ(n)/n) * ln(ζ(ns))
     *
     * @param {Complex|number|BigFloat} s
     * @returns {Complex}
     */
    export function primeZeta(s: Complex | number | BigFloat): Complex;
    import { Complex } from "complex";
    import { BigFloat } from "bf";
}
declare module "lambertw" {
    /**
     * High-precision Lambert W function W_k(z).
     * Computes the principal branch W_0(z) by default, or the k-th branch W_k(z).
     *
     * @param {Complex|number|string|BigFloat} z The complex argument
     * @param {number} k Branch index (default 0)
     * @returns {Complex}
     */
    export function lambertw(z: Complex | number | string | BigFloat, k?: number): Complex;
    import { Complex } from "complex";
    import { BigFloat } from "bf";
}
declare module "bessel" {
    /**
     * Confluent Hypergeometric Limit Function 0F1(; a; z)
     * Used as the primary power series expansion for Bessel functions.
     *
     * @param {Complex} a The parameter 'a' (usually nu + 1)
     * @param {Complex} z The complex argument
     * @param {Number} [max_iter=10000]
     * @returns {Complex}
     */
    export function hyp0f1(a: Complex, z: Complex, max_iter?: number): Complex;
    /**
     * Bessel function of the first kind J_v(z).
     * Defined by series: J_v(z) = (z/2)^v / gamma(v+1) * 0F1(; v+1; -z^2/4)
     *
     * @param {Complex|number} nu Order of the Bessel function
     * @param {Complex|number} z Complex argument
     * @returns {Complex}
     */
    export function besselj(nu: Complex | number, z: Complex | number): Complex;
    /**
     * Modified Bessel function of the first kind I_v(z).
     * Defined by: I_v(z) = (z/2)^v / gamma(v+1) * 0F1(; v+1; z^2/4)
     *
     * @param {Complex|number} nu Order
     * @param {Complex|number} z Complex argument
     * @returns {Complex}
     */
    export function besseli(nu: Complex | number, z: Complex | number): Complex;
    /**
     * Bessel function of the second kind Y_v(z)
     * Uses the mpmath precision trick of limit approximation (perturbation)
     * for integer orders to sidestep exact L'Hôpital evaluation over series.
     *
     * Y_v(z) =[J_v(z)cos(v*pi) - J_{-v}(z)] / sin(v*pi)
     *
     * @param {Complex|number} nu Order
     * @param {Complex|number} z Complex argument
     * @returns {Complex}
     */
    export function bessely(nu: Complex | number, z: Complex | number): Complex;
    /**
     * Modified Bessel function of the second kind K_v(z)
     *
     * K_v(z) = (pi/2) *[I_{-v}(z) - I_v(z)] / sin(v*pi)
     *
     * @param {Complex|number} nu Order
     * @param {Complex|number} z Complex argument
     * @returns {Complex}
     */
    export function besselk(nu: Complex | number, z: Complex | number): Complex;
    /**
     * Hankel function of the first kind H1_v(z) = J_v(z) + i*Y_v(z)
     */
    export function hankel1(nu: any, z: any): bfjs.Complex;
    /**
     * Hankel function of the second kind H2_v(z) = J_v(z) - i*Y_v(z)
     */
    export function hankel2(nu: any, z: any): bfjs.Complex;
    import { Complex } from "complex";
    import * as bfjs from "bf";
}
declare module "frac" {
    /**
     * Creates a new BigFraction instance.
     * @param {BigFraction | bigint | number | string} [n] - The numerator or the whole value.
     * @param {bigint | number | string} [d=1n] - The denominator.
     * @returns {BigFraction}
     */
    export function frac(n?: BigFraction | bigint | number | string, d?: bigint | number | string): BigFraction;
    /**
     * @class BigFraction
     * @description A class for arbitrary-precision rational number arithmetic.
     */
    export class BigFraction {
        /**
         * Parses a string to create a fraction.
         * Supports integers "123", fractions "1/2", and decimals "1.5".
         * @param {string} str
         * @returns {BigFraction}
         */
        static fromString(str: string): BigFraction;
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
        constructor(n?: BigFraction | bigint | number | string, d?: bigint | number | string);
        n: bigint | undefined;
        d: bigint | undefined;
        /**
         * Adds another fraction.
         * @param {BigFraction|bigint|number|string} b
         * @returns {BigFraction} New Fraction instance
         */
        add(b: BigFraction | bigint | number | string): BigFraction;
        /**
         * Subtracts another fraction.
         * @param {BigFraction|bigint|number|string} b
         * @returns {BigFraction}
         */
        sub(b: BigFraction | bigint | number | string): BigFraction;
        /**
         * Multiplies by another fraction.
         * @param {BigFraction|bigint|number|string} b
         * @returns {BigFraction}
         */
        mul(b: BigFraction | bigint | number | string): BigFraction;
        /**
         * Divides by another fraction.
         * @param {BigFraction|bigint|number|string} b
         * @returns {BigFraction}
         */
        div(b: BigFraction | bigint | number | string): BigFraction;
        /**
         * Returns the integer square root of the fraction (floor(sqrt(value))).
         * Since BigInt arithmetic is integer based, exact rational roots are rare.
         * This returns a Fraction representing the integer root.
         * @returns {BigFraction|undefined}
         */
        sqrt(): BigFraction | undefined;
        /**
         * Raises fraction to an integer power.
         * @param {number|bigint|BigFraction} exponent
         * @returns {BigFraction|undefined}
         */
        pow(exponent: number | bigint | BigFraction): BigFraction | undefined;
        /**
         * Returns the floor of the fraction (largest integer <= value).
         * @returns {BigFraction}
         */
        floor(): BigFraction;
        /**
         * Negates the value.
         * @returns {BigFraction}
         */
        neg(): BigFraction;
        /**
         * Returns the absolute value.
         * @returns {BigFraction}
         */
        abs(): BigFraction;
        /**
         * Returns e^this. Placeholder, not implemented.
         * @returns {BigFraction|undefined}
         */
        exp(): BigFraction | undefined;
        /**
         * Returns log(this). Placeholder, not implemented.
         * @returns {BigFraction|undefined}
         */
        log(): BigFraction | undefined;
        /**
         * Returns sin(this). Placeholder, not implemented.
         * @returns {BigFraction|undefined}
         */
        sin(): BigFraction | undefined;
        /**
         * Returns cos(this). Placeholder, not implemented.
         * @returns {BigFraction|undefined}
         */
        cos(): BigFraction | undefined;
        /**
         * Returns tan(this). Placeholder, not implemented.
         * @returns {BigFraction|undefined}
         */
        tan(): BigFraction | undefined;
        /**
         * Returns asin(this). Placeholder, not implemented.
         * @returns {BigFraction|undefined}
         */
        asin(): BigFraction | undefined;
        /**
         * Returns acos(this). Placeholder, not implemented.
         * @returns {BigFraction|undefined}
         */
        acos(): BigFraction | undefined;
        /**
         * Returns atan(this). Placeholder, not implemented.
         * @returns {BigFraction|undefined}
         */
        atan(): BigFraction | undefined;
        /**
         * Returns sinh(this). Placeholder, not implemented.
         * @returns {BigFraction|undefined}
         */
        sinh(): BigFraction | undefined;
        /**
         * Returns cosh(this). Placeholder, not implemented.
         * @returns {BigFraction|undefined}
         */
        cosh(): BigFraction | undefined;
        /**
         * Returns tanh(this). Placeholder, not implemented.
         * @returns {BigFraction|undefined}
         */
        tanh(): BigFraction | undefined;
        /**
         * Returns asinh(this). Placeholder, not implemented.
         * @returns {BigFraction|undefined}
         */
        asinh(): BigFraction | undefined;
        /**
         * Returns acosh(this). Placeholder, not implemented.
         * @returns {BigFraction|undefined}
         */
        acosh(): BigFraction | undefined;
        /**
         * Returns atanh(this). Placeholder, not implemented.
         * @returns {BigFraction|undefined}
         */
        atanh(): BigFraction | undefined;
        /**
         * Checks if the fraction is technically invalid (denominator was 0).
         * @returns {boolean}
         */
        isNaN(): boolean;
        /**
         * Checks if the value is zero.
         * @returns {boolean}
         */
        isZero(): boolean;
        /**
         * Checks if the value is zero.
         * @returns {boolean}
         */
        isAlmostZero(): boolean;
        /**
         * @returns {BigFloat}
         */
        toBigFloat(): BigFloat;
        /**
         * Converts to a standard JavaScript number (may lose precision).
         * @returns {number}
         */
        toNumber(): number;
        /**
         * Converts to string.
         * @param {number} [radix=10]
         * @param {number} [prec=-1] precision digits in radix
         * @param {boolean} [pretty=false] pretty print
         * @returns {string}
         */
        toString(radix?: number, prec?: number, pretty?: boolean): string;
        /**
         * Compares with another value.
         * @param {BigFraction|bigint|number|string} b
         * @returns {number} -1 if less, 0 if equal, 1 if greater.
         */
        cmp(b: BigFraction | bigint | number | string): number;
        /**
         * @param {BigFraction|bigint|number|string} b
         * @returns {boolean}
         */
        equals(b: BigFraction | bigint | number | string): boolean;
        /**
         * @param {BigFraction|bigint|number|string} b
         * @returns {boolean}
         */
        operatorLess(b: BigFraction | bigint | number | string): boolean;
        /**
         * @param {BigFraction|bigint|number|string} b
         * @returns {boolean}
         */
        operatorGreater(b: BigFraction | bigint | number | string): boolean;
        /**
         * @param {BigFraction|bigint|number|string} b
         * @returns {boolean}
         */
        operatorLessEqual(b: BigFraction | bigint | number | string): boolean;
        /**
         * @param {BigFraction|bigint|number|string} b
         * @returns {boolean}
         */
        operatorGreaterEqual(b: BigFraction | bigint | number | string): boolean;
        /**
         * @param {BigFraction|bigint|number|string} b
         * @returns {boolean}
         */
        operatorEqual(b: BigFraction | bigint | number | string): boolean;
        /**
         * @param {BigFraction|bigint|number|string} b
         * @returns {boolean}
         */
        operatorNotEqual(b: BigFraction | bigint | number | string): boolean;
        /**
         * @param {BigFraction|bigint|number|string} b
         * @returns {BigFraction}
         */
        operatorAdd(b: BigFraction | bigint | number | string): BigFraction;
        /**
         * @param {BigFraction|bigint|number|string} b
         * @returns {BigFraction}
         */
        operatorSub(b: BigFraction | bigint | number | string): BigFraction;
        /**
         * @param {BigFraction|bigint|number|string} b
         * @returns {BigFraction}
         */
        operatorMul(b: BigFraction | bigint | number | string): BigFraction;
        /**
         * @param {BigFraction|bigint|number|string} b
         * @returns {BigFraction}
         */
        operatorDiv(b: BigFraction | bigint | number | string): BigFraction;
        /**
         * @param {number|bigint|BigFraction} b
         * @returns {BigFraction|undefined}
         */
        operatorPow(b: number | bigint | BigFraction): BigFraction | undefined;
        /**
         * @returns {BigFraction}
         */
        operatorNeg(): BigFraction;
    }
    import { BigFloat } from "bf";
}
declare module "poly" {
    /**
     * Factory for creating a polynomial representing X^n.
     * @param {number} [n=1] - The degree of X.
     * @param {Function} [coefType=BigFloat] - The coefficient type.
     * @returns {Poly}
     */
    export function X(n?: number, coefType?: Function): Poly;
    /**
     * Factory for creating a Big-O term O(X^n).
     * @param {number} [n=1] - The order of the truncation.
     * @param {Function} [coefType=BigFloat] - The coefficient type.
     * @returns {Poly}
     */
    export function O(n?: number, coefType?: Function): Poly;
    /**
     * Factory function to create a Poly instance using a State Machine parser.
     * This parser avoids regex for core logic to handle complex nested structures and strict validation.
     *
     * Supported formats examples:
     *  - Integer/Fraction: "1", "-1", "+-1", "-2/3"
     *  - Variables: "X", "-X", "-2X", "-2*X"
     *  - Exponents: "X^2", "X^-1", "X^(-1)", "X^0"
     *  - Complex in parens: "(-1+i)X", "(3+2i)"
     *  - Big-O: "+O(3)", "O(X^5)"
     *
     * @param {string} v
     * @param {Function} [coefType=Scalar] - The class used to wrap/construct coefficients.
     * @returns {Poly}
     */
    export function polyStr(v: string, coefType?: Function): Poly;
    /**
     * Factory function to create a Poly instance from various inputs.
     * Now uses the Scalar class for coefficient parsing and unified representation.
     *
     * @param {string | Array<any> | Object<string, any> | Map<string, any> | number | bigint | BigFloat | BigFraction | Complex | Poly | Scalar} v
     * @param {Function} [coefType=BigFloat] - The class used to construct coefficients.
     * @returns {Poly}
     */
    export function poly(v: string | Array<any> | {
        [x: string]: any;
    } | Map<string, any> | number | bigint | BigFloat | BigFraction | Complex | Poly | Scalar, coefType?: Function): Poly;
    /**
     * Poly Class
     * Represents a Polynomial or a Truncated Power Series using sparse storage.
     *
     * Storage Strategy:
     * - Sparse representation: Two parallel arrays, `degs` (degrees) and `coefs` (coefficients).
     * - Coefficients are generic types (BigFloat, Complex) supporting arithmetic interfaces.
     * - `order` (property `o`): The truncation order O(X^n).
     *    - If `Infinity`, it represents an exact polynomial.
     *    - If a number `n`, terms with degree >= n are discarded.
     *
     * @property {number[]} degs - Array of degrees (integers).
     * @property {any[]} coefs - Array of coefficients corresponding to degrees.
     * @property {number} o - Truncation order O(X^n).
     * @property {Function} coefType - The coefficient type constructor.
     */
    export class Poly {
        /**
         * Creates a polynomial representing X^n.
         * @param {number} [n=1] - The degree of X.
         * @param {Function} [coefType=BigFloat] - The coefficient type.
         * @returns {Poly}
         */
        static X(n?: number, coefType?: Function): Poly;
        /**
         * Creates a Big-O term O(X^n).
         * @param {number} n - The order of the truncation.
         * @param {Function} [coefType=BigFloat] - The coefficient type.
         * @returns {Poly}
         */
        static O(n: number, coefType?: Function): Poly;
        /**
         * @param {number[]} degs - Array of degrees (integers).
         * @param {any[]} coefs - Array of coefficients corresponding to degrees.
         * @param {number} [order=Infinity] - Truncation order O(X^n).
         * @param {Function} [coefType=bf] - the coef type
         */
        constructor(degs: number[], coefs: any[], order?: number, coefType?: Function);
        degs: number[];
        coefs: any[];
        o: number;
        coefType: Function;
        /**
         * Normalizes the sparse representation:
         * 1. Sorts by degree.
         * 2. Merges duplicate degrees.
         * 3. Removes zero coefficients.
         * 4. Removes terms with degree >= order.
         * @private
         */
        private _normalize;
        /**
         * Pushes a term to the new arrays if it's valid.
         * @private
         * @param {number[]} degs
         * @param {any[]} coefs
         * @param {number} d
         * @param {any} c
         */
        private _pushTerm;
        /**
         * Wraps a scalar value in a Poly object.
         * @private
         * @param {any} v
         * @returns {Poly}
         */
        private _wrap;
        /**
         * Ensures a value is of the correct coefficient type.
         * @private
         * @param {any} c
         * @returns {any}
         */
        private _ensureType;
        /**
         * Returns the "Valuation" (degree of the lowest non-zero term).
         * @returns {number} - Infinity if Exact Zero, or n if O(X^n).
         */
        valuation(): number;
        /**
         * Returns the Degree (highest non-zero term).
         * @returns {number} - -1 if zero polynomial.
         */
        degree(): number;
        /**
         * Returns a dense array of coefficients up to the highest degree.
         * Note: Throws if polynomial contains negative powers. Use offsetCoefs for Laurent series.
         * @returns {any[]} [c0, c1, c2, ...]
         */
        get denseCoefs(): any[];
        /**
         * Returns a dense array of coefficients along with the valuation offset.
         * Supports negative degrees.
         * @returns {{val: number, coefs: any[]}} { val: starting_degree, coefs: [c_val, c_val+1, ...] }
         */
        get offsetCoefs(): {
            val: number;
            coefs: any[];
        };
        /**
         * Evaluates the polynomial at x.
         * P(x) = sum( c_i * x^i )
         * @param {number|BigFloat|Complex} x
         * @returns {any}
         */
        eval(x: number | BigFloat | Complex): any;
        /**
         * Adds two polynomials.
         * @param {Poly|number|any} other
         * @returns {Poly}
         */
        add(other: Poly | number | any): Poly;
        /**
         * Subtracts two polynomials.
         * @param {Poly|number|any} other
         * @returns {Poly}
         */
        sub(other: Poly | number | any): Poly;
        /**
         * Negates the polynomial.
         * @returns {Poly}
         */
        neg(): Poly;
        /**
         * Multiplies two polynomials.
         * @param {Poly|number|any} other
         * @returns {Poly}
         */
        mul(other: Poly | number | any): Poly;
        /**
         * Power: P(x)^n
         * @param {number} n - The exponent.
         * @param {number} [d=1] - The denominator of the exponent.
         * @returns {Poly}
         */
        pow(n: number, d?: number): Poly;
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
        div(other: Poly, defaultLimit?: number): Poly;
        /**
          * Checks equality with another polynomial or scalar.
          * @param {Poly|number|BigFloat|Complex} other - The object to compare with.
          * @param {Function} [cmp] - Optional comparator (a, b) => boolean.
          *                           Defaults to checking a.equals(b) or strict equality.
          * @returns {boolean}
          */
        equals(other: Poly | number | BigFloat | Complex, cmp?: Function): boolean;
        operatorAdd(arg0: any): Poly;
        operatorSub(arg0: any): Poly;
        operatorMul(arg0: any): Poly;
        operatorDiv(arg0: any): Poly;
        operatorPow(arg0: number, arg1?: number | undefined): Poly;
        operatorNeg(): Poly;
        /**
         * Derivative
         * d/dx ( c * x^k ) = (c*k) * x^(k-1)
         * @returns {Poly}
         */
        deriv(): Poly;
        /**
         * Formal Integration
         * int ( c * x^k ) = (c / (k+1)) * x^(k+1)
         * Constant term set to 0.
         * @returns {Poly}
         */
        integ(): Poly;
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
        powSeries(n: number, d?: number): Poly;
        /**
         * Checks if the polynomial has a defined order.
         * Transcendental functions require truncation to avoid infinite loops.
         * @private
         * @param {string} op - The operation name.
         */
        private _checkSeries;
        /**
         * Splits polynomial into Constant term (c0) and Variable part (V).
         * P(x) = c0 + V(x)
         * @private
         * @returns {[any, Poly]} [c0, V]
         */
        private _splitConst;
        /**
         * Exponential Function: e^P(x)
         * e^(c0 + V) = e^c0 * e^V
         * e^V = 1 + V + V^2/2! + V^3/3! + ...
         * @returns {Poly}
         */
        exp(): Poly;
        /**
         * Natural Logarithm: ln(P(x))
         * Uses Derivative-Integration method:
         * ln(P) = int( P' / P ) dx + ln(P(0))
         * @returns {Poly}
         */
        log(): Poly;
        /**
         * Sine: sin(P(x))
         * sin(c0 + V) = sin(c0)cos(V) + cos(c0)sin(V)
         * @returns {Poly}
         */
        sin(): Poly;
        /**
         * Cosine: cos(P(x))
         * cos(c0 + V) = cos(c0)cos(V) - sin(c0)sin(V)
         * @returns {Poly}
         */
        cos(): Poly;
        /**
         * Tangent: tan(P(x))
         * tan(P) = sin(P) / cos(P)
         * @returns {Poly}
         */
        tan(): Poly;
        /**
         * Arcsine: asin(P(x))
         * asin(P) = int( P' / sqrt(1 - P^2) ) + asin(P(0))
         * @returns {Poly}
         */
        asin(): Poly;
        /**
         * Arccosine: acos(P(x))
         * acos(P) = PI/2 - asin(P)
         * @returns {Poly}
         */
        acos(): Poly;
        /**
         * Arctangent: atan(P(x))
         * atan(P) = int( P' / (1 + P^2) ) + atan(P(0))
         * @returns {Poly}
         */
        atan(): Poly;
        /**
         * Computes sin(V) and cos(V) for a polynomial V with no constant term.
         * Uses Taylor series optimized for simultaneous calculation.
         * sin(V) = V - V^3/3! + V^5/5! ...
         * cos(V) = 1 - V^2/2! + V^4/4! ...
         * @private
         * @param {Poly} V
         * @returns {[Poly, Poly]} [sinV, cosV]
         */
        private _sinCosV;
        /**
         * Converts the polynomial to a string representation.
         * @param {number} [radix=10]
         * @param {number} [precision=20]
         * @param {boolean} [pretty=true] pretty print
         * @returns {string}
         */
        toString(radix?: number, precision?: number, pretty?: boolean): string;
    }
    import { BigFloat } from "bf";
    import { BigFraction } from "frac";
    import { Complex } from "complex";
    import { Scalar } from "scalar";
}
declare module "scalar" {
    /**
     * Factory function for creating Scalar instances.
     * @param {number|bigint|string|BigFraction|BigFloat|Complex|Scalar} s
     * @returns {Scalar}
     */
    export function scalar(s: number | bigint | string | BigFraction | BigFloat | Complex | Scalar): Scalar;
    /**
     * Scalar Wrapper Class.
     * Manages dispatching and type promotion between BigFraction, BigFloat, and Complex.
     *
     * Hierarchy:
     * Level 0: BigFraction (Rational)
     * Level 1: BigFloat (Real/Float)
     * Level 2: Complex (Complex)
     *
     * @property {BigFraction|BigFloat|Complex} value - The underlying numeric value.
     * @property {number} level - The type promotion level (0, 1, or 2).
     */
    export class Scalar {
        /**
         * Determine the level of a raw math object.
         * @param {any} v The value to check.
         * @returns {number} The promotion level.
         */
        static getLevel(v: any): number;
        /**
         * Promotes a raw value to the target level.
         * @param {any} v The value to promote.
         * @param {number} targetLevel The target promotion level.
         * @returns {any} The promoted value.
         */
        static promote(v: any, targetLevel: number): any;
        /**
         * Internal dispatcher for binary operations.
         * @private
         * @param {Scalar|any} a - The first operand.
         * @param {Scalar|any} b - The second operand.
         * @param {string} opName - The name of the operation.
         * @returns {Scalar} The result of the operation.
         */
        private static _binaryOp;
        /**
         * Parses string and determines correct type/level.
         * @param {string} str
         * @returns {Scalar}
         */
        static fromString(str: string): Scalar;
        /**
         * @param {number|bigint|string|BigFraction|BigFloat|Complex|Scalar} v
         */
        constructor(v: number | bigint | string | BigFraction | BigFloat | Complex | Scalar);
        value: any;
        level: any;
        /** @param {Scalar|any} other @returns {Scalar} */
        add(other: Scalar | any): Scalar;
        /** @param {Scalar|any} other @returns {Scalar} */
        sub(other: Scalar | any): Scalar;
        /** @param {Scalar|any} other @returns {Scalar} */
        mul(other: Scalar | any): Scalar;
        /** @param {Scalar|any} other @returns {Scalar} */
        div(other: Scalar | any): Scalar;
        /** @param {Scalar|any} other @returns {Scalar} */
        pow(other: Scalar | any): Scalar;
        /** @param {Scalar|any} b @returns {Scalar} */
        operatorAdd(b: Scalar | any): Scalar;
        /** @param {Scalar|any} b @returns {Scalar} */
        operatorSub(b: Scalar | any): Scalar;
        /** @param {Scalar|any} b @returns {Scalar} */
        operatorMul(b: Scalar | any): Scalar;
        /** @param {Scalar|any} b @returns {Scalar} */
        operatorDiv(b: Scalar | any): Scalar;
        /** @param {Scalar|any} b @returns {Scalar} */
        operatorPow(b: Scalar | any): Scalar;
        /** @returns {Scalar} */
        operatorNeg(): Scalar;
        /** @returns {Scalar} */
        neg(): Scalar;
        /** @returns {boolean} */
        isZero(): boolean;
        /** @returns {boolean} */
        isAlmostZero(): boolean;
        /** @returns {Scalar} */
        abs(): Scalar;
        /**
         * Internal dispatcher for functions that might return undefined for BigFraction.
         * @private
         * @param {string} opName The name of the operation.
         * @returns {Scalar}
         */
        private _unary;
        /** @returns {Scalar} */
        exp(): Scalar;
        /** @returns {Scalar} */
        log(): Scalar;
        /** @returns {Scalar} */
        sin(): Scalar;
        /** @returns {Scalar} */
        cos(): Scalar;
        /** @returns {Scalar} */
        tan(): Scalar;
        /** @returns {Scalar} */
        asin(): Scalar;
        /** @returns {Scalar} */
        acos(): Scalar;
        /** @returns {Scalar} */
        atan(): Scalar;
        /** @returns {Scalar} */
        sinh(): Scalar;
        /** @returns {Scalar} */
        cosh(): Scalar;
        /** @returns {Scalar} */
        tanh(): Scalar;
        /** @returns {Scalar} */
        asinh(): Scalar;
        /** @returns {Scalar} */
        acosh(): Scalar;
        /** @returns {Scalar} */
        atanh(): Scalar;
        /** @returns {Scalar} */
        sqrt(): Scalar;
        /**
         * Compares this scalar with another value.
         * @param {Scalar|any} other
         * @returns {number} -1 if this < other, 0 if this === other, 1 if this > other.
         */
        cmp(other: Scalar | any): number;
        /**
         * Checks for equality with another value.
         * @param {Scalar|any} other
         * @returns {boolean}
         */
        equals(other: Scalar | any): boolean;
        /**
         * Converts the scalar to a string.
         * @param {number} [radix=10]
         * @param {number} [precision=-1]
         * @param {boolean} [pretty=false] pretty print
         * @returns {string}
         */
        toString(radix?: number, precision?: number, pretty?: boolean): string;
    }
    import { BigFraction } from "frac";
    import { BigFloat } from "bf";
    import { Complex } from "complex";
}
declare module "bf" {
    /**
     * Set the global precision
     * @param {number} p
     */
    export function setPrecision(p: number): void;
    /**
     * Get the global precision
     * @returns {number}
     */
    export function getPrecision(): number;
    /**
     * Pushes the current precision to the stack and sets a new precision.
     * @param {number} prec - The new precision in bits.
     */
    export function pushPrecision(prec: number): void;
    /**
     * Pops the precision from the stack and restores the previous precision.
     */
    export function popPrecision(): void;
    /**
     * Gets or sets the precision in decimal digits.
     * @param {number} [dp] - The new precision in decimal digits. If not provided, the function returns the current precision in decimal digits.
     * @returns {number | undefined}
     */
    export function decimalPrecision(dp?: number): number | undefined;
    /**
     * Pushes the current precision to the stack and sets a new precision in decimal digits.
     * @param {number} dp - The new precision in decimal digits.
     */
    export function pushDecimalPrecision(dp: number): void;
    /**
     * Gets the epsilon value for the current precision.
     * @returns {number}
     */
    export function getEpsilon(): number;
    /**
     * Set gc_ele_limit
     * @param {number} l
     */
    export function setGcEleLimit(l: number): void;
    /**
     * Get gc_ele_limit
     * @returns {number}
     */
    export function getGcEleLimit(): number;
    /**
     * Checks if the libbf library is ready.
     * @returns {boolean}
     */
    export function isReady(): boolean;
    /**
     * Set throwExceptionOnInvalidOp
     * @param {boolean} f
     */
    export function setThrowExceptionOnInvalidOp(f: boolean): void;
    /**
     * Get the global flags for libbf operations.
     * @returns {number}
     */
    export function getGlobalFlag(): number;
    /**
     * Set the global flags for libbf operations.
     * @param {number} f
     */
    export function setGlobalFlag(f: number): void;
    /**
     * Creates a new BigFloat instance.
     * @param {string | number | bigint | BigFloat} [val] - The value to initialize the BigFloat with.
     * @param {number} [radix=10] - The radix to use if `val` is a string.
     * @param {boolean} [managed=true] - Whether the BigFloat should be managed by the garbage collector.
     * @param {boolean} [constant=false] - Whether the BigFloat is a constant.
     * @returns {BigFloat}
     */
    export function bf(val?: string | number | bigint | BigFloat, radix?: number, managed?: boolean, constant?: boolean): BigFloat;
    /**
     * Initializes the libbf library.
     * @param {any} m - The wasm module.
     * @returns {Promise<boolean>}
     */
    export function init(m: any): Promise<boolean>;
    /**
     * @param {...(BigFloat | number | string | bigint)} args
     * @returns {BigFloat}
     */
    export function max(...args: (BigFloat | number | string | bigint)[]): BigFloat;
    /**
     * @param {...(BigFloat | number | string | bigint)} args
     * @returns {BigFloat}
     */
    export function min(...args: (BigFloat | number | string | bigint)[]): BigFloat;
    /**
     * @param {BigFloat | number | string | bigint} v
     * @param {number} [prec=0]
     * @returns {BigFloat}
     */
    export function sqrt(v: BigFloat | number | string | bigint, prec?: number): BigFloat;
    /**
     * @param {BigFloat | number | string | bigint} v
     * @param {number} [prec=0]
     * @param {number} [rnd_mode=0]
     * @returns {BigFloat}
     */
    export function fpround(v: BigFloat | number | string | bigint, prec?: number, rnd_mode?: number): BigFloat;
    /**
     * @param {BigFloat | number | string | bigint} v
     * @param {number} [prec=0]
     * @returns {BigFloat}
     */
    export function round(v: BigFloat | number | string | bigint, prec?: number): BigFloat;
    /**
     * @param {BigFloat | number | string | bigint} v
     * @param {number} [prec=0]
     * @returns {BigFloat}
     */
    export function trunc(v: BigFloat | number | string | bigint, prec?: number): BigFloat;
    /**
     * @param {BigFloat | number | string | bigint} v
     * @param {number} [prec=0]
     * @returns {BigFloat}
     */
    export function floor(v: BigFloat | number | string | bigint, prec?: number): BigFloat;
    /**
     * @param {BigFloat | number | string | bigint} v
     * @param {number} [prec=0]
     * @returns {BigFloat}
     */
    export function ceil(v: BigFloat | number | string | bigint, prec?: number): BigFloat;
    /**
     * @param {BigFloat | number | string | bigint} v
     * @param {number} [prec=0]
     * @returns {BigFloat}
     */
    export function neg(v: BigFloat | number | string | bigint, prec?: number): BigFloat;
    /**
     * @param {BigFloat | number | string | bigint} v
     * @param {number} [prec=0]
     * @returns {BigFloat}
     */
    export function abs(v: BigFloat | number | string | bigint, prec?: number): BigFloat;
    /**
     * @param {BigFloat | number | string | bigint} v
     * @param {number} [prec=0]
     * @returns {BigFloat}
     */
    export function sign(v: BigFloat | number | string | bigint, prec?: number): BigFloat;
    /**
     * @param {BigFloat | number | string | bigint} v
     * @param {number} [prec=0]
     * @returns {BigFloat}
     */
    export function exp(v: BigFloat | number | string | bigint, prec?: number): BigFloat;
    /**
     * @param {BigFloat | number | string | bigint} v
     * @param {number} [prec=0]
     * @returns {BigFloat}
     */
    export function log(v: BigFloat | number | string | bigint, prec?: number): BigFloat;
    /**
     * @param {BigFloat | number | string | bigint} v
     * @param {BigFloat | number | string | bigint} b
     * @param {number} [prec=0]
     * @returns {BigFloat}
     */
    export function pow(v: BigFloat | number | string | bigint, b: BigFloat | number | string | bigint, prec?: number): BigFloat;
    /**
     * @param {BigFloat | number | string | bigint} v
     * @param {number} [prec=0]
     * @returns {BigFloat}
     */
    export function cos(v: BigFloat | number | string | bigint, prec?: number): BigFloat;
    /**
     * @param {BigFloat | number | string | bigint} v
     * @param {number} [prec=0]
     * @returns {BigFloat}
     */
    export function sin(v: BigFloat | number | string | bigint, prec?: number): BigFloat;
    /**
     * @param {BigFloat | number | string | bigint} v
     * @param {number} [prec=0]
     * @returns {BigFloat}
     */
    export function tan(v: BigFloat | number | string | bigint, prec?: number): BigFloat;
    /**
     * @param {BigFloat | number | string | bigint} v
     * @param {number} [prec=0]
     * @returns {BigFloat}
     */
    export function atan(v: BigFloat | number | string | bigint, prec?: number): BigFloat;
    /**
     * @param {BigFloat | number | string | bigint} v
     * @param {number} [prec=0]
     * @returns {BigFloat}
     */
    export function atan2(v: BigFloat | number | string | bigint, prec?: number): BigFloat;
    /**
     * @param {BigFloat | number | string | bigint} v
     * @param {number} [prec=0]
     * @returns {BigFloat}
     */
    export function asin(v: BigFloat | number | string | bigint, prec?: number): BigFloat;
    /**
     * @param {BigFloat | number | string | bigint} v
     * @param {number} [prec=0]
     * @returns {BigFloat}
     */
    export function acos(v: BigFloat | number | string | bigint, prec?: number): BigFloat;
    /**
     *
     * @param {BigFloat|Number|string} start
     * @param {BigFloat|Number|string} end
     * @param {Number} n
     * @returns
     */
    export function linspace(start: BigFloat | number | string, end: BigFloat | number | string, n: number): BigFloat[];
    export namespace Flags {
        let BF_ST_INVALID_OP: number;
        let BF_ST_DIVIDE_ZERO: number;
        let BF_ST_OVERFLOW: number;
        let BF_ST_UNDERFLOW: number;
        let BF_ST_INEXACT: number;
        let BF_ST_MEM_ERROR: number;
        let BF_RADIX_MAX: number;
        let BF_ATOF_NO_HEX: number;
        let BF_ATOF_BIN_OCT: number;
        let BF_ATOF_NO_NAN_INF: number;
        let BF_ATOF_EXPONENT: number;
        let BF_RND_MASK: number;
        let BF_FTOA_FORMAT_MASK: number;
        let BF_FTOA_FORMAT_FIXED: number;
        let BF_FTOA_FORMAT_FRAC: number;
        let BF_FTOA_FORMAT_FREE: number;
        let BF_FTOA_FORMAT_FREE_MIN: number;
        let BF_FTOA_FORCE_EXP: number;
        let BF_FTOA_ADD_PREFIX: number;
        let BF_FTOA_JS_QUIRKS: number;
        let BF_POW_JS_QUIRKS: number;
        let BF_RNDN: number;
        let BF_RNDZ: number;
        let BF_RNDD: number;
        let BF_RNDU: number;
        let BF_RNDNA: number;
        let BF_RNDA: number;
        let BF_RNDF: number;
    }
    /**
     * The current precision in bits.
     * @type {number}
     * @private
     */
    export const precision: number;
    /**
     * The maximum number of elements before garbage collection is triggered.
     * @type {number}
     * @private
     */
    export const gc_ele_limit: number;
    /**
     * If true, an exception is thrown on invalid operations.
     * @type {boolean}
     */
    export const throwExceptionOnInvalidOp: boolean;
    /**
     * The libbf instance.
     * @type {any}
     */
    export const libbf: any;
    /**
     * The global flags for libbf operations.
     * @type {number}
     * @private
     */
    export const globalFlag: number;
    /**
     * @class BigFloat
     * @description A class for arbitrary-precision floating-point arithmetic.
     */
    export class BigFloat {
        /**
         * @param {string} str
         * @param {number} [radix=10]
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        static fromString(str: string, radix?: number, prec?: number): BigFloat;
        /**
         * Creates a new BigFloat instance.
         * @param {string | number | bigint | BigFloat} [val] - The value to initialize the BigFloat with.
         * @param {number} [radix=10] - The radix to use if `val` is a string.
         * @param {boolean} [managed=true] - Whether the BigFloat should be managed by the garbage collector.
         * @param {boolean} [constant=false] - Whether the BigFloat is a constant.
         */
        constructor(val?: string | number | bigint | BigFloat, radix?: number, managed?: boolean, constant?: boolean);
        hwrapper: any[];
        managed: boolean;
        status: number;
        visited: number;
        constant: boolean;
        set h(hv: number);
        /**
         * The handle to the underlying C object.
         * @type {number}
         */
        get h(): number;
        /**
         * Marks the BigFloat as visited by the garbage collector.
         * @param {boolean} [addToArray=true] - Whether to add the BigFloat to the garbage collector's array.
         */
        visit(addToArray?: boolean): void;
        /**
         * Converts the BigFloat to a Uint8Array.
         * @returns {Uint8Array}
         */
        toUint8Array(): Uint8Array;
        /**
         * Disposes of the BigFloat's resources.
         * @param {boolean} [recoverable=true] - Whether the BigFloat can be recovered later.
         */
        dispose(recoverable?: boolean): void;
        h_bak: Uint8Array<ArrayBufferLike> | null | undefined;
        /**
         * Gets the handle to the underlying C object, creating it if necessary.
         * @returns {number}
         */
        geth(): number;
        /**
         * Checks if the last operation resulted in an inexact result.
         * @returns {boolean}
         */
        isInExact(): boolean;
        /**
         * Checks the status of the last operation.
         * @param {number} s - The status to check.
         * @returns {number} The status.
         */
        checkstatus(s: number): number;
        /**
         * Wraps the given arguments in BigFloat handles.
         * @param {...(BigFloat | string | number | bigint | object)} ar - The arguments to wrap.
         * @returns {[function(): void, ...number[]]} An array containing a cleanup function and the handles.
         */
        wraptypeh(...ar: (BigFloat | string | number | bigint | object)[]): [() => void, ...number[]];
        /**
         * Converts the BigFloat to a Complex number.
         * @returns {complex}
         */
        toComplex(): typeof complex;
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
        calc(method: string, a: (BigFloat | number | string | bigint | null) | undefined, b: (BigFloat | number | string | bigint | null) | undefined, prec: number, flags?: number, rnd_mode?: number): BigFloat;
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
        calc2(method: string, a: (BigFloat | number | string | bigint | null) | undefined, b: (BigFloat | number | string | bigint | null) | undefined, prec: number, rnd_mode?: number, q?: BigFloat | null): BigFloat;
        /**
         * Checks if the given arguments are valid operands.
         * @param {...any} args - The arguments to check.
         */
        checkoprand(...args: any[]): void;
        /**
         * Sets this BigFloat to the sum of a and b.
         * @param {BigFloat | number | string | bigint} a
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setadd(a: BigFloat | number | string | bigint, b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Sets this BigFloat to the difference of a and b.
         * @param {BigFloat | number | string | bigint} a
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setsub(a: BigFloat | number | string | bigint, b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Sets this BigFloat to the product of a and b.
         * @param {BigFloat | number | string | bigint} a
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setmul(a: BigFloat | number | string | bigint, b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Sets this BigFloat to the division of a and b.
         * @param {BigFloat | number | string | bigint} a
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setdiv(a: BigFloat | number | string | bigint, b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Sets this BigFloat to the modulus of a and b.
         * @param {BigFloat | number | string | bigint} a
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setmod(a: BigFloat | number | string | bigint, b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Sets this BigFloat to the remainder of a and b.
         * @param {BigFloat | number | string | bigint} a
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setrem(a: BigFloat | number | string | bigint, b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Sets this BigFloat to the bitwise OR of a and b.
         * @param {BigFloat | number | string | bigint} a
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setor(a: BigFloat | number | string | bigint, b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Sets this BigFloat to the bitwise XOR of a and b.
         * @param {BigFloat | number | string | bigint} a
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setxor(a: BigFloat | number | string | bigint, b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Sets this BigFloat to the bitwise AND of a and b.
         * @param {BigFloat | number | string | bigint} a
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setand(a: BigFloat | number | string | bigint, b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Sets this BigFloat to the square root of a.
         * @param {BigFloat | number | string | bigint} a
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setsqrt(a: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Rounds this BigFloat to a given precision.
         * @param {number} [prec=0]
         * @param {number} [rnd_mode=Flags.BF_RNDN]
         * @returns {BigFloat} this
         */
        setfpround(prec?: number, rnd_mode?: number): BigFloat;
        /**
         * Rounds this BigFloat to the nearest integer.
         * @returns {BigFloat} this
         */
        setround(): BigFloat;
        /**
         * Truncates this BigFloat to an integer.
         * @returns {BigFloat} this
         */
        settrunc(): BigFloat;
        /**
         * Floors this BigFloat to an integer.
         * @returns {BigFloat} this
         */
        setfloor(): BigFloat;
        /**
         * Ceils this BigFloat to an integer.
         * @returns {BigFloat} this
         */
        setceil(): BigFloat;
        /**
         * Negates this BigFloat.
         * @returns {BigFloat} this
         */
        setneg(): BigFloat;
        /**
         * Sets this BigFloat to its absolute value.
         * @returns {BigFloat} this
         */
        setabs(): BigFloat;
        /**
         * Sets this BigFloat to the sign of a.
         * @param {BigFloat | number | string | bigint} a
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setsign(a: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Sets this BigFloat to the value of log2(e).
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setLOG2(prec?: number): BigFloat;
        /**
         * Sets this BigFloat to the value of PI.
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setPI(prec?: number): BigFloat;
        /**
         * Sets this BigFloat to its minimum value.
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setMIN_VALUE(prec?: number): BigFloat;
        /**
         * Sets this BigFloat to its maximum value.
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setMAX_VALUE(prec?: number): BigFloat;
        /**
         * Sets this BigFloat to its epsilon value.
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setEPSILON(prec?: number): BigFloat;
        /**
         * Sets this BigFloat to e^a.
         * @param {BigFloat | number | string | bigint} a
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setexp(a: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Sets this BigFloat to log(a).
         * @param {BigFloat | number | string | bigint} a
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setlog(a: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Sets this BigFloat to a^b.
         * @param {BigFloat | number | string | bigint} a
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setpow(a: BigFloat | number | string | bigint, b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Sets this BigFloat to cos(a).
         * @param {BigFloat | number | string | bigint} a
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setcos(a: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Sets this BigFloat to sin(a).
         * @param {BigFloat | number | string | bigint} a
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setsin(a: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Sets this BigFloat to tan(a).
         * @param {BigFloat | number | string | bigint} a
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        settan(a: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Sets this BigFloat to atan(a).
         * @param {BigFloat | number | string | bigint} a
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setatan(a: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Sets this BigFloat to atan2(a, b).
         * @param {BigFloat | number | string | bigint} a
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setatan2(a: BigFloat | number | string | bigint, b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Sets this BigFloat to asin(a).
         * @param {BigFloat | number | string | bigint} a
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setasin(a: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Sets this BigFloat to acos(a).
         * @param {BigFloat | number | string | bigint} a
         * @param {number} [prec=0]
         * @returns {BigFloat} this
         */
        setacos(a: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * Checks if this BigFloat is finite.
         * @returns {boolean}
         */
        isFinit(): boolean;
        /**
         * Checks if this BigFloat is NaN.
         * @returns {boolean}
         */
        isNaN(): boolean;
        /**
         * Checks if this BigFloat is exactly zero.
         * @returns {boolean}
         */
        isExactZero(): boolean;
        /**
         * Checks if this BigFloat is zero.
         * @returns {boolean}
         */
        isZero(): boolean;
        /**
         * Checks if this BigFloat is almost zero.
         * @returns {boolean}
         */
        isAlmostZero(): boolean;
        /**
         * Copies the value of another BigFloat to this one.
         * @param {BigFloat} a - The BigFloat to copy from.
         * @returns {void}
         */
        copy(a: BigFloat): void;
        /**
         * Clones this BigFloat.
         * @returns {BigFloat}
         */
        clone(): BigFloat;
        /**
         * Sets the value of this BigFloat from a number.
         * @param {number} a
         * @returns {void}
         */
        fromNumber(a: number): void;
        /**
         * Converts this BigFloat to a 64-bit float.
         * @returns {number}
         */
        f64(): number;
        /**
         * Converts this BigFloat to a number.
         * @returns {number}
         */
        toNumber(): number;
        /**
         * Compares this BigFloat with another one.
         * @param {BigFloat | number | string | bigint} b
         * @returns {number} 0 if equal, >0 if this > b, <0 if this < b.
         */
        cmp(b: BigFloat | number | string | bigint): number;
        /**
         * @param {BigFloat | number | string | bigint} b
         * @returns {boolean}
         */
        operatorLess(b: BigFloat | number | string | bigint): boolean;
        /**
         * @param {BigFloat | number | string | bigint} b
         * @returns {boolean}
         */
        operatorGreater(b: BigFloat | number | string | bigint): boolean;
        /**
         * @param {BigFloat | number | string | bigint} b
         * @returns {boolean}
         */
        operatorLessEqual(b: BigFloat | number | string | bigint): boolean;
        /**
         * @param {BigFloat | number | string | bigint} b
         * @returns {boolean}
         */
        operatorGreaterEqual(b: BigFloat | number | string | bigint): boolean;
        /**
         * @param {BigFloat | number | string | bigint} b
         * @returns {boolean}
         */
        operatorEqual(b: BigFloat | number | string | bigint): boolean;
        /**
         * @param {BigFloat | number | string | bigint} b
         * @returns {boolean}
         */
        operatorNotEqual(b: BigFloat | number | string | bigint): boolean;
        /**
         * @param {BigFloat | number | string | bigint} b
         * @returns {boolean}
         */
        equals(b: BigFloat | number | string | bigint): boolean;
        /**
         * @private
         * @param {string} str
         * @param {number} [radix=10]
         * @param {number} [prec=0]
         * @returns {this}
         */
        private _fromString;
        /**
         * toString
         * @param {number} [radix=10]
         * @param {number} [prec=-1] precision digits in radix
         * @param {boolean} [pretty=false] pretty print
         * @returns {string}
         */
        toString(radix?: number, prec?: number, pretty?: boolean): string;
        /**
         *
         * @param {number} [radix=10]
         * @param {number} [prec=-1] precision digits in radix
         * @param {number} [rnd_mode=Flags.BF_RNDNA]
         * @returns {string}
         */
        toFixed(radix?: number, prec?: number, rnd_mode?: number): string;
        /**
         * @returns {bigint}
         */
        toBigInt(): bigint;
        /**
         * @param {function} ofunc
         * @param {number} numps
         * @param {...any} args
         * @returns {BigFloat}
         */
        callFunc(ofunc: Function, numps: number, ...args: any[]): BigFloat;
        /**
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        add(b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        sub(b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        mul(b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        div(b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        mod(b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        rem(b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        or(b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        xor(b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        and(b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        sqrt(prec?: number): BigFloat;
        /**
         * @param {number} prec
         * @param {number} rnd_mode
         * @returns {BigFloat}
         */
        fpround(prec: number, rnd_mode: number): BigFloat;
        /**
         * @returns {BigFloat}
         */
        round(): BigFloat;
        /**
         * @returns {BigFloat}
         */
        trunc(): BigFloat;
        /**
         * @returns {BigFloat}
         */
        floor(): BigFloat;
        /**
         * @returns {BigFloat}
         */
        ceil(): BigFloat;
        /**
         * @returns {BigFloat}
         */
        neg(): BigFloat;
        /**
         * @returns {BigFloat}
         */
        abs(): BigFloat;
        /**
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        sign(prec?: number): BigFloat;
        /**
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        exp(prec?: number): BigFloat;
        /**
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        log(prec?: number): BigFloat;
        /**
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        pow(b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        cos(prec?: number): BigFloat;
        /**
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        sin(prec?: number): BigFloat;
        /**
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        cosh(prec?: number): BigFloat;
        /**
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        sinh(prec?: number): BigFloat;
        /**
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        tanh(prec?: number): BigFloat;
        /**
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        tan(prec?: number): BigFloat;
        /**
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        atan(prec?: number): BigFloat;
        /**
         * @param {BigFloat | number | string | bigint} b
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        atan2(b: BigFloat | number | string | bigint, prec?: number): BigFloat;
        /**
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        asin(prec?: number): BigFloat;
        /**
         * @param {number} [prec=0]
         * @returns {BigFloat}
         */
        acos(prec?: number): BigFloat;
        operatorAdd: (b: BigFloat | number | string | bigint, prec?: number) => BigFloat;
        operatorSub: (b: BigFloat | number | string | bigint, prec?: number) => BigFloat;
        operatorMul: (b: BigFloat | number | string | bigint, prec?: number) => BigFloat;
        operatorDiv: (b: BigFloat | number | string | bigint, prec?: number) => BigFloat;
        operatorPow: (b: BigFloat | number | string | bigint, prec?: number) => BigFloat;
        operatorMod: (b: BigFloat | number | string | bigint, prec?: number) => BigFloat;
        operatorNeg: () => BigFloat;
        operatorBitwiseAnd: (b: BigFloat | number | string | bigint, prec?: number) => BigFloat;
        operatorBitwiseOr: (b: BigFloat | number | string | bigint, prec?: number) => BigFloat;
        operatorBitwiseXor: (b: BigFloat | number | string | bigint, prec?: number) => BigFloat;
        operatorBitwiseNot: any;
    }
    /**
     * @type {BigFloat | null}
     * @private
     */
    export const minus_one: BigFloat | null;
    /**
     * @type {BigFloat | null}
     * @private
    */
    export const zero: BigFloat | null;
    /**
     * @type {BigFloat | null}
     * @private
    */
    export const half: BigFloat | null;
    /**
     * @type {BigFloat | null}
     * @private
    */
    export const one: BigFloat | null;
    /**
     * @type {BigFloat | null}
     * @private
    */
    export const two: BigFloat | null;
    /**
     * @type {BigFloat | null}
     * @private
    */
    export const three: BigFloat | null;
    /**
     * @type {BigFloat | null}
     * @private*/
    export const PI: BigFloat | null;
    /**
     * @type {BigFloat | null}
     * @private*/
    export const E: BigFloat | null;
    export namespace Constants {
        const minus_one: BigFloat | null;
        const zero: BigFloat | null;
        const half: BigFloat | null;
        const one: BigFloat | null;
        const two: BigFloat | null;
        const three: BigFloat | null;
        const PI: BigFloat | null;
        const E: BigFloat | null;
    }
    export * from "complex";
    export * from "vector";
    export * from "matrix";
    export * from "polyfit";
    export * from "ode45";
    export * from "ode15s";
    export * from "pdepe";
    export * from "fminbnd";
    export * from "roots";
    export * from "fzero";
    export * from "quad";
    export * from "romberg";
    export * from "limit";
    export * from "nsum";
    export * from "shanks";
    export * from "identify";
    export * from "bernoulli";
    export * from "gamma";
    export * from "zeta";
    export * from "lambertw";
    export * from "bessel";
    export * from "frac";
    export * from "poly";
    export * from "scalar";
    import { complex } from "complex";
}
declare module "bernoulli" {
    export function clearBernoulliCache(): void;
    /**
     * Bernoulli numbers B_n using Ramanujan's recursive formula for small n
     * and the Zeta relationship for large n.
     *
     * Ramanujan's Formula (speeds up recursion):
     * sum_{k=0}^n [ C(6n+3, 6k) * B_{6k} ] = (2n+1)/2
     *
     * @param {number} n - The index of the Bernoulli number.
     * @returns {BigFloat}
     */
    export function bernoulli(n: number): BigFloat;
    import { BigFloat } from "bf";
}
