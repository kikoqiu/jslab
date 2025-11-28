# JSLab

JSLab is an interactive JavaScript environment designed for numerical computing, data exploration, and powerful visualizations directly in your browser. It provides a rich set of features for performing complex calculations, generating data, creating dynamic plots, and even handling high-precision arithmetic.

Try it here: [https://kikoqiu.github.io/jslab/jslab.html](https://kikoqiu.github.io/jslab/jslab.html)

All scripts run directly in your browser.

## Key Features

*   **Interactive JavaScript Environment**: Execute code in cells, similar to a notebook interface.
*   **Powerful Output Control**: Use `echo()` for displaying text and variables, and `echoHTML()` for rendering HTML content.
*   **Mathematical Computing**: Leverage `math.js` for extensive mathematical operations, including symbolic computation (derivatives, simplification).
*   **LaTeX Support**: Beautifully render mathematical formulas using `latex()` and `latexstr()`.
*   **Dynamic Data Visualization**: Create interactive 2D and 3D plots with a simplified `plot()` and `plot3d()` interface, powered by Plotly.js.
*   **High-Precision Arithmetic**:
    *   **BigInt Operations**: Full support for JavaScript's native `BigInt` for arbitrary-precision integers.
    *   **Big Float (bfjs)**: Integrate `libbf.js` for arbitrary-precision floating-point arithmetic.
*   **Data Manipulation**: Efficiently generate sequences with `range()` and `rangen()`, and unpack data with `unpack()`.
*   **File Handling**: Read user-selected files directly into the environment using `readfile()`.
*   **Code Transformation**: Internal use of Babel for JavaScript code transformation.
*   **Code Editing**: Integrated CodeMirror for an enhanced coding experience.

## Core Functions Overview

JSLab provides several global functions for convenience:

*   `echo(...o)`: Prints values to the cell's output.
*   `echoHTML(...o)`: Prints HTML content to the cell's output.
*   `latex(...ex)`: Displays mathematical expressions as rendered LaTeX.
*   `rangen(a, b, step, mapper)`: Generates a sequence of numbers.
*   `range(a, b, step, mapper)`: Returns an array from `rangen`.
*   `plot(...args)`: Simplified interface for 2D plotting.
*   `plot3d(...args)`: Simplified interface for 3D plotting.
*   `plotly(data, layout, config, style)`: Full Plotly.js interface.
*   `readfile(type, encoding)`: Reads a user-selected file.
*   `deriv(expr, variable)`: Calculates symbolic derivative using math.js.
*   `symplify(expr)`: Simplifies math.js expressions.
*   `compile_expr(e)`, `eval_expr(e, scope)`: For math.js expression evaluation.

## Integrated Libraries

JSLab extends its capabilities by integrating with these powerful libraries:

*   **Math.js**: An extensive math library for JavaScript, providing a flexible expression parser, a large set of built-in functions, and support for various data types.
*   **d3.js**: A powerful library for manipulating documents based on data, used for advanced visualizations.
*   **Plotly.js**: Used for creating interactive, publication-quality 2D and 3D graphs.
*   **LibBF.js**: Enables arbitrary-precision floating-point arithmetic for highly sensitive calculations.
*   **Babel**: Internally used for JavaScript code transformation, ensuring compatibility and advanced features.
*   **CodeMirror**: Provides the robust in-browser code editor experience for JSLab cells.

## Getting Started and Examples

To get started and explore JSLab's features interactively, navigate to the **"Help"** menu within the JSLab application:

*   Select **"Load General Usage"** to load a set of introductory examples.
*   Select **"Load Plotting Usage"** to load examples demonstrating data visualization.

These will load cells directly into your JSLab environment, allowing you to experiment and learn interactively.
