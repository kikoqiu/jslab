/**
 * ============================================================================
 * MODULE 1: Expression Compiler
 * Compiles string expressions into optimized vectorized JavaScript functions.
 * ============================================================================
 */

/**
 * Compiles a math string into a function that processes TypedArrays.
 * Normalizes all inputs to a standard form: f(x, y) < 0.
 * 
 * @param {string|Object} input - The expression string or an already compiled object.
 * @returns {Object} { batchFunc: Function, isEquality: boolean }
 */
function compileImplicitFunc(input) {
    if (typeof input !== 'string') return input; // Already compiled

    // 1. Pre-processing: Standardize operators
    // Convert power operator '^' to JS '**'
    let expr = input.replace(/\^/g, "**");

    let isEquality = true;
    let lhs = expr;
    let rhs = "0";
    let invertSign = false; // Used to convert ">" logic to "<" logic

    // 2. Detect Operator and Split
    // We normalize everything to: LHS - RHS (compare to 0)
    if (expr.includes(">=")) {
        [lhs, rhs] = expr.split(">=");
        isEquality = false;
        invertSign = true; // a >= b  ->  b - a <= 0
    } else if (expr.includes("<=")) {
        [lhs, rhs] = expr.split("<=");
        isEquality = false;
    } else if (expr.includes("==")) {
        [lhs, rhs] = expr.split("==");
        isEquality = true;
    } else if (expr.includes("=")) {
        [lhs, rhs] = expr.split("=");
        isEquality = true;
    } else if (expr.includes(">")) {
        [lhs, rhs] = expr.split(">");
        isEquality = false;
        invertSign = true; // a > b  ->  b - a < 0
    } else if (expr.includes("<")) {
        [lhs, rhs] = expr.split("<");
        isEquality = false;
    }

    // 3. Construct Function Body
    // Math form: f(x,y) = (LHS) - (RHS)
    let mathBody = invertSign 
        ? `(${rhs}) - (${lhs})` 
        : `(${lhs}) - (${rhs})`;

    // Replace standard math functions (sin, cos, etc.) with Math.sin, Math.cos
    const mathProps = Object.getOwnPropertyNames(Math);
    for (const prop of mathProps) {
        const regex = new RegExp(`\\b${prop}\\b`, 'g');
        mathBody = mathBody.replace(regex, `Math.${prop}`);
    }

    // 4. Generate Vectorized Function (Simulates SIMD)
    // Arguments:
    // xArr, yArr: Input coordinate arrays (FloatArray)
    // outArr: Output value array (FloatArray)
    const funcBody = `
        for (let j = 0; j < h; j++) {
            for(let i = 0; i < w; i++){
                const x = xArr[i];
                const y = yArr[j];
                outArr[j*w+i] = ${mathBody};
            }
        }
    `;

    try {
        const batchFunc = new Function("w", "h", "xArr", "yArr", "outArr", funcBody);
        return { batchFunc, isEquality, original: input };
    } catch (e) {
        throw new Error(`Compilation failed: ${e.message}`);
    }
}


/**
 * ============================================================================
 * MODULE 2: Core Calculation Engine
 * Pure math and buffer manipulation. No DOM access.
 * ============================================================================
 */

/**
 * Fills the pixel buffer based on the implicit function logic.
 * 
 * @param {Object} ctx - Context containing dimensions and buffers.
 * @param {Array} tasks - Array of objects { batchFunc, isEquality, colorInt }.
 */
async function computeImplicitPixels(ctx, tasks) {
    const { width, height, superSample, bounds, buffers } = ctx;
    const { xBuffer, yBuffer, valBuffer, pixelBuffer32 } = buffers;
    const { xMin, xMax, yMin, yMax } = bounds;

    // Clear buffer (set to opaque white 0xFFFFFFFF for canvas)
    // Note: In Little Endian, 0xFFFFFFFF is A=255, B=255, G=255, R=255
    pixelBuffer32.fill(0xFFFFFFFF);

    // Helper: Fast Integer Alpha Blending (Src over Dst)
    // Assumes Little Endian (ABGR) layout.
    const blendColors = (dst, src) => {
        //const srcA = src >>> 24; 

        // Extract channels
        const srcR = src & 0xFF, srcG = (src >> 8) & 0xFF, srcB = (src >> 16) & 0xFF;
        const dstR = dst & 0xFF, dstG = (dst >> 8) & 0xFF, dstB = (dst >> 16) & 0xFF;
        
        //const invA = 255 - srcA;

        // Simple approximate blending: (src * a + dst * (255-a)) / 255
        // Using >> 8 as fast division by 256 approximation
        //const r = (srcR * srcA + dstR * invA) >> 8;
        //const g = (srcG * srcA + dstG * invA) >> 8;
        //const b = (srcB * srcA + dstB * invA) >> 8;

        const r = (srcR + dstR)>>1;
        const g = (srcG + dstG)>>1;
        const b = (srcB + dstB)>>1;
        
        // Preserve opaque alpha (255) for the canvas pixel
        return (0xFF000000) | (b << 16) | (g << 8) | r;
    };

    // Sequential Rendering (Painters Algorithm)
    for (const task of tasks) {
        const { batchFunc, isEquality, colorInt } = task;
        //colorInt = 

        // === PATH A: EQUALITY (Curves) ===
        // Uses Intermediate Value Theorem (IVT) on a supersampled grid.
        if (isEquality) {
            const ss = superSample; // e.g., 2
            const gridW = width * ss;
            const gridH = height * ss;
            
            // Grid Calculation
            // We generate coordinates for grid corners.
            // Screen Y is top-to-bottom (0 to Height). Math Y is usually Max to Min.
            const dx = (xMax - xMin) / gridW;
            const dy = (yMin - yMax) / gridH;

            // Generate coordinates (Row-major order)
            for (let j = 0; j <= gridH; j++) {       
                yBuffer[j] = yMax + j * dy;
            }
            for (let i = 0; i <= gridW; i++) {
                xBuffer[i] = xMin + i * dx;
            }

            // Batch Evaluation
            await batchFunc(gridW + 1, gridH + 1, xBuffer, yBuffer, valBuffer);

            // Rendering Loop
            // Iterate over screen pixels
            const rowStride = gridW + 1;
            
            for (let py = 0; py < height; py++) {
                const gridBaseY = py * ss;
                const pixelRowOffset = py * width;

                for (let px = 0; px < width; px++) {
                    const gridBaseX = px * ss;
                    let hit = false;

                    // Supersampling Loop (Early Exit)
                // If ANY sub-cell in this pixel contains a zero-crossing, color the pixel.
                    ssLoop:
                    for (let sy = 0; sy < ss; sy++) {
                        const rowOffset = (gridBaseY + sy) * rowStride;
                        const nextRowOffset = rowOffset + rowStride;

                        for (let sx = 0; sx < ss; sx++) {
                            const idxTL = rowOffset + gridBaseX + sx;
                            const idxTR = idxTL + 1;
                            const idxBL = nextRowOffset + gridBaseX + sx;
                            
                            // IVT Check: Check signs of corners.
                        // We check min and max of the corners. If min <= 0 <= max, there is a root.
                            const vTL = valBuffer[idxTL];
                            const vTR = valBuffer[idxTR];
                            const vBL = valBuffer[idxBL];
                            
                            let min = vTL, max = vTL;
                            if (vTR < min) min = vTR; else if (vTR > max) max = vTR;
                            if (vBL < min) min = vBL; else if (vBL > max) max = vBL;

                            // Check 4th corner only if needed for precision, 
                            // but 3 points are usually enough to detect a crossing in a triangle fan.
                            // Checking 4th improves quality for thin lines.
                            const vBR = valBuffer[idxBL + 1];
                            if (vBR < min) min = vBR; else if (vBR > max) max = vBR;

                            if (min <= 0 && max >= 0) {
                                hit = true;
                            break ssLoop; // Optimization: Found it, move to next pixel
                            }
                        }
                    }

                    if (hit) {
                        const idx = pixelRowOffset + px;
                        pixelBuffer32[idx] = blendColors(pixelBuffer32[idx], colorInt);
                    }
                }
            }
        } 
        // === PATH B: INEQUALITY (Regions) ===
        // Uses 1:1 pixel center sampling (no supersampling needed for solid regions).
        else {
            const dx = (xMax - xMin) / width;
            const dy = (yMin - yMax) / height;
            const count = width * height;

            // Grid Calculation (Standard resolution)
            for (let j = 0; j <= height; j++) {       
                yBuffer[j] = yMax + j * dy;
            }
            for (let i = 0; i <= width; i++) {
                xBuffer[i] = xMin + i * dx;
            }

            // Batch Evaluation
            await batchFunc(width, height, xBuffer, yBuffer, valBuffer);

            // Rendering Loop
            for (let i = 0; i < count; i++) {
                // Normalized logic: satisfy if val < 0
                if (valBuffer[i] < 0) {
                    pixelBuffer32[i] = blendColors(pixelBuffer32[i], colorInt);
                }
            }
        }
    }
}


/**
 * ============================================================================
 * MODULE 3: UI Controller (D3 & DOM)
 * Handles interactions, axes, and resizing.
 * ============================================================================
 */

/**
 * Renders interactive implicit mathematical functions (equations and inequalities) onto a HTML5 Canvas.
 * 
 * This function initializes a high-performance plotting environment using D3.js for interaction (zoom/pan)
 * and low-level pixel manipulation for rendering. It supports plotting multiple functions simultaneously 
 * with alpha blending, custom coloring, and a dynamic legend.
 * 
 * Features:
 * - Parses and compiles math strings (e.g., "x^2 + y^2 = 25", "y < sin(x)").
 * - Supports array inputs for plotting multiple layers sequentially (Painters Algorithm).
 * - Interactive Zoom and Pan with "Sticky Axes" and dynamic grid lines.
 * - Automatic legend generation based on provided names.
 * - Optimized rendering: Equalities are super-sampled (anti-aliased), Inequalities (regions) utilize alpha blending.
 * - Debounced resizing and rendering to maintain performance.
 *
 * @async
 * @param {HTMLElement|string} container - The DOM element to render into, or a unique CSS selector string (e.g., "#myPlot").
 * @param {string|string[]|Object|Object[]} exprInput - The mathematical expression(s) to plot. 
 *        Can be:
 *        - A single string expression (e.g., "x^2 + y^2 = 100").
 *        - An array of string expressions (e.g., ["y > x", "y < 2"]).
 *        - Pre-compiled logic objects (advanced use).
 *        Supported operators include standard JS Math (sin, cos, etc.) and `^` for power.
 * @param {Object} [options={}] - Optional configuration settings.
 * @param {string[]} [options.colors] - An array of CSS color strings (e.g., ["#f00", "rgba(0,0,255,0.5)"]). 
 *        If not provided, a default D3 category10 palette is used. 
 *        Note: Inequalities default to 0.6 opacity if not explicitly defined in the color string.
 * @param {string[]} [options.names] - An array of names corresponding to `exprInput` for the legend. 
 *        Defaults to ["f1", "f2", ...].
 * @param {string} [options.gridColor="#e0e0e0"] - CSS color for the background grid lines.
 * @param {string} [options.axisColor="#444"] - CSS color for the X/Y axes and tick labels.
 * @param {number} [options.superSample=1] - Super-sampling factor for anti-aliasing curves (Equalities). 
 *        Higher values (e.g., 2 or 4) improve line quality but increase render time.
 * @param {Object} [options.bounds] - The initial viewport boundaries.
 * @param {number} [options.bounds.xMin=-10] - Minimum X value.
 * @param {number} [options.bounds.xMax=10] - Maximum X value.
 * @param {number} [options.bounds.yMin=-10] - Minimum Y value.
 * @param {number} [options.bounds.yMax=10] - Maximum Y value.
 * @returns {Promise<void>} Resolves when the initial render setup is complete.
 */
async function plotImplicitCanvas(container, exprInput, options = {}) {
    // 1. Resolve DOM Container
    const targetEl = typeof container === "string" 
        ? document.querySelector(container) 
        : container;

    if (!targetEl) throw new Error("Plot container not found.");

    // Normalize input to array
    const inputs = Array.isArray(exprInput) ? exprInput : [exprInput];
    
    // Default Colors (D3 Category10 like)
    const defaultColors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf"];

    // 2. Configuration & Defaults
    const config = {
        gridColor: "#e0e0e0",
        axisColor: "#444",
        colors: options.colors || defaultColors,
        names: options.names || inputs.map((_, i) => `Trace ${i + 1}`),
        superSample: 1,   
        bounds: { xMin: -10, xMax: 10, yMin: -10, yMax: 10 }, 
        ...options
    };

    // Compile Logic & Prepare Colors
    let compiledTasks = [];
    try {
        compiledTasks = inputs.map((input, i) => {
            const logic = (typeof input === "string") ? compileImplicitFunc(input) : input;
            
            // Resolve Color
            const colorStr = config.colors[i % config.colors.length];
            const rgb = d3.color(colorStr);
            
            // Automatic Alpha Strategy
            let alpha = rgb.opacity * 255;
            if (rgb.opacity === 1 && !logic.isEquality) {
                alpha = 0.6 * 255;
            }
            
            // Construct ABGR Int
            const colorInt = ((Math.floor(alpha) & 0xFF) << 24) | 
                             ((rgb.b & 0xFF) << 16) | 
                             ((rgb.g & 0xFF) << 8) | 
                             (rgb.r & 0xFF);
            
            return { 
                ...logic, 
                colorInt,
                colorStr, // Store string for Legend display
                name: config.names[i] 
            };
        });
    } catch (err) {
        d3.select(targetEl).style("position", "relative")
          .html(`<div style="color:red; padding:10px; font-family:sans-serif;">Error: ${err.message}</div>`);
        return;
    }

    // 3. DOM Structure Setup
    const root = d3.select(targetEl)
        .style("position", "relative")
        .style("user-select", "none");

    // Force default height if container is collapsed
    if (targetEl.clientHeight < 50) {
        root.style("height", "600px");
    }

    root.selectAll("*").remove();

    // Wrapper
    const wrapper = root.append("div")
        .style("position", "absolute")
        .style("top", 0)
        .style("left", 0)
        .style("width", "100%")
        .style("height", "100%")
        .style("overflow", "hidden")
        .style("background", "#fff");

    // Canvas
    const canvas = wrapper.append("canvas")
        .style("position", "absolute")
        .style("top", 0).style("left", 0)
        .style("transform-origin", "0 0");
    const ctx = canvas.node().getContext("2d", { alpha: false }); 

    // SVG Layers
    const svg = wrapper.append("svg")
        .style("position", "absolute")
        .style("top", 0).style("left", 0)
        .style("pointer-events", "none"); 

    const gGrid = svg.append("g").attr("class", "grid-layer");
    const gAxis = svg.append("g").attr("class", "axis-layer");

    // UI: Tooltip (Coordinates)
    const tooltip = root.append("div")
        .style("position", "absolute")
        .style("top", "10px").style("left", "10px")
        .style("padding", "6px 10px")
        .style("background", "rgba(255, 255, 255, 0.9)")
        .style("border", "1px solid #ccc")
        .style("border-radius", "4px")
        .style("font-family", "monospace")
        .style("font-size", "12px")
        .style("pointer-events", "none")
        .style("box-shadow", "0 2px 4px rgba(0,0,0,0.1)")
        .style("display", "none")
        .text("x: 0, y: 0");

    // UI: Reset Button
    root.append("button")
        .text("Reset View")
        .style("position", "absolute")
        .style("top", "10px").style("right", "10px")
        .style("padding", "6px 12px")
        .style("background", "#fff")
        .style("border", "1px solid #999")
        .style("border-radius", "4px")
        .style("cursor", "pointer")
        .style("font-size", "12px")
        .on("click", () => {
            wrapper.transition().duration(750)
                .call(zoomBehavior.transform, d3.zoomIdentity);
        });

    // UI: Legend
    if (compiledTasks.length > 0) {
        const legend = root.append("div")
            .style("position", "absolute")
            .style("top", "45px") // Below Reset Button
            .style("right", "10px")
            .style("padding", "8px 10px")
            .style("background", "rgba(255, 255, 255, 0.9)")
            .style("border", "1px solid #ccc")
            .style("border-radius", "4px")
            .style("font-family", "sans-serif")
            .style("font-size", "12px")
            .style("box-shadow", "0 2px 4px rgba(0,0,0,0.1)")
            .style("max-height", "200px")
            .style("overflow-y", "auto");

        compiledTasks.forEach(task => {
            const row = legend.append("div")
                .style("display", "flex")
                .style("align-items", "center")
                .style("margin-bottom", "4px");
            
            // Color Swatch
            row.append("span")
                .style("display", "inline-block")
                .style("width", "12px")
                .style("height", "12px")
                .style("background-color", task.colorStr)
                .style("margin-right", "8px")
                .style("border", "1px solid rgba(0,0,0,0.1)");
            
            // Name
            row.append("span")
                .text(task.name)
                .style("white-space", "nowrap");
        });
    }

    // 4. State Management
    let width = 0, height = 0;
    let buffers = null;
    let transform = d3.zoomIdentity;
    let renderedTransform = d3.zoomIdentity;
    let debounceTimer = null;
    let resizeTimer = null; // Timer for resize debounce

    const xScale = d3.scaleLinear();
    const yScale = d3.scaleLinear();

    let pending=[];
    // 5. Render Core (Calculations)
    const renderContent = async () => {
        while(pending[0]){
            await pending[0];            
        }
        let resolve_this;
        pending[0]=new Promise((resolve,reject)=>{
            resolve_this = resolve;
        });
        try{
            if (!width || !height || !buffers) return;
            //console.time("ComputeImplicit");
            
            let currentTransform = {...transform};
            currentTransform.__proto__=transform.__proto__;

            const tx = currentTransform.rescaleX(xScale);
            const ty = currentTransform.rescaleY(yScale);

            const bounds = {
                xMin: tx.invert(0),
                xMax: tx.invert(width),
                yMin: ty.invert(height),
                yMax: ty.invert(0)
            };

            const ctxParams = {
                width, height,
                superSample: config.superSample, 
                bounds,
                buffers
            };

            await computeImplicitPixels(ctxParams, compiledTasks);

            const imageData = new ImageData(
                new Uint8ClampedArray(buffers.pixelBuffer32.buffer), 
                width, height
            );
            ctx.putImageData(imageData, 0, 0);
            canvas.style("transform", "none");
            renderedTransform = currentTransform;
            //use latest transform to update
            updateCanvasTransform(transform);

            //console.timeEnd("ComputeImplicit");
        }catch(e){
            console.error(e);
        }finally{
            resolve_this();
            delete pending[0];
        }
    };

    // 6. Visual Updates (Axes, Grid, Tooltip)
    const updateVisuals = (t) => {
        const tx = t.rescaleX(xScale);
        const ty = t.rescaleY(yScale);

        // Tooltip: Dynamic Precision
        wrapper.on("mousemove", (event) => {
            const [mx, my] = d3.pointer(event);
            const mathX = tx.invert(mx);
            const mathY = ty.invert(my);

            const domainSpan = tx.domain()[1] - tx.domain()[0];
            const unitsPerPixel = domainSpan / width;
            
            let decimals = Math.max(2, Math.ceil(-Math.log10(unitsPerPixel)) + 1);
            if (decimals > 10) decimals = 10; 

            tooltip.style("display", "block")
                .text(`x: ${mathX.toFixed(decimals)}, y: ${mathY.toFixed(decimals)}`);
        }).on("mouseleave", () => {
            tooltip.style("display", "none");
        });

        // Grid
        const xGrid = d3.axisBottom(tx).ticks(width / 80).tickSize(-height).tickFormat("");
        const yGrid = d3.axisLeft(ty).ticks(height / 80).tickSize(-width).tickFormat("");

        gGrid.selectAll(".g-x").data([1]).join("g").attr("class", "g-x")
            .attr("transform", `translate(0, ${height})`)
            .call(xGrid)
            .call(g => g.selectAll("line").attr("stroke", config.gridColor).attr("stroke-opacity", 0.5))
            .call(g => g.selectAll(".domain").remove());

        gGrid.selectAll(".g-y").data([1]).join("g").attr("class", "g-y")
            .call(yGrid)
            .call(g => g.selectAll("line").attr("stroke", config.gridColor).attr("stroke-opacity", 0.5))
            .call(g => g.selectAll(".domain").remove());

        // Sticky Axes
        const zeroX = tx(0);
        const zeroY = ty(0);
        const axisXPos = Math.max(0, Math.min(height, zeroY));
        const axisYPos = Math.max(0, Math.min(width, zeroX));

        const xAxis = d3.axisBottom(tx).ticks(width / 80);
        const yAxis = d3.axisLeft(ty).ticks(height / 80);

        gAxis.selectAll(".axis-x").data([1]).join("g").attr("class", "axis-x")
            .attr("transform", `translate(0, ${axisXPos})`)
            .call(xAxis);

        gAxis.selectAll(".axis-y").data([1]).join("g").attr("class", "axis-y")
            .attr("transform", `translate(${axisYPos}, 0)`)
            .call(yAxis);

        gAxis.selectAll("text").attr("fill", config.axisColor);
        gAxis.selectAll("line").attr("stroke", "#ccc");
        gAxis.selectAll(".domain").attr("stroke", config.axisColor).attr("stroke-width", 1.5);
    };

    const updateCanvasTransform = (t)=>{
        // Canvas Transform
        if (t !== renderedTransform) {
            const k = t.k / renderedTransform.k;
            const p0_world = renderedTransform.invert([0, 0]);
            const p0_screen = t.apply(p0_world);
            
            canvas.style("transform", 
                `translate(${p0_screen[0]}px, ${p0_screen[1]}px) scale(${k})`
            );
        }
    }

    let inited = false;

    // 7. Zoom Behavior
    const zoomBehavior = d3.zoom()
        .scaleExtent([1e-10, 1e10])
        .on("start", () => {
            if (debounceTimer){
               clearTimeout(debounceTimer);
               debounceTimer=null;
            }
        })
        .on("zoom", (event) => {
            transform = event.transform;

            updateVisuals(transform);
            updateCanvasTransform(transform);
            if (debounceTimer){
                clearTimeout(debounceTimer);
                debounceTimer =null;
            }
        })
        .on("end", () => {
            // Only trigger the heavy re-render logic once the user releases the mouse
            if (debounceTimer) clearTimeout(debounceTimer);
            debounceTimer = setTimeout(renderContent, inited?300:0); 
            inited = true;
        });

    wrapper.call(zoomBehavior).on("dblclick.zoom", null);


    // 8. Resize Handler
    const onResize = () => {
        const rect = wrapper.node().getBoundingClientRect();
        width = Math.floor(rect.width);
        height = Math.floor(rect.height);

        if (width === 0 || height === 0) return;

        // A. Immediate Visual Updates
        // 1. Hide canvas to prevent distortion/lag
        canvas.style("opacity", 0);
        
        // 2. Update SVG/Canvas dimensions
        canvas.attr("width", width).attr("height", height);
        svg.attr("width", width).attr("height", height);

        // 3. Recalculate Scales to Lock Aspect Ratio (Square pixels)
        const xRange = config.bounds.xMax - config.bounds.xMin;
        const pixelsPerUnit = width / xRange;
        const yRange = height / pixelsPerUnit;
        const yMid = (config.bounds.yMin + config.bounds.yMax) / 2;
        
        xScale.domain([config.bounds.xMin, config.bounds.xMax]).range([0, width]);
        yScale.domain([yMid - yRange / 2, yMid + yRange / 2]).range([height, 0]);

        // 4. Update Axes immediately
        transform = d3.zoomIdentity;
        updateVisuals(transform);
        updateCanvasTransform(transform);

        // B. Debounced Heavy Calculation
        if (resizeTimer) clearTimeout(resizeTimer);
        resizeTimer = setTimeout(() => {
            // Allocate Buffers
            const hasEquality = compiledTasks.some(t => t.isEquality);
            const ss = hasEquality ? config.superSample : 1;
            const count = Math.max((width * ss + 1) * (height * ss + 1), width * height);

            buffers = {
                xBuffer: new Float32Array(width * ss + 1),
                yBuffer: new Float32Array(height * ss + 1),
                valBuffer: new Float32Array(count),
                pixelBuffer32: new Uint32Array(width * height)
            };

            wrapper.call(zoomBehavior.transform, d3.zoomIdentity);
            renderedTransform = d3.zoomIdentity;

            //Render and Fade In
            //renderContent();
            //render is triggered by zoom
            
            canvas.transition().duration(1000).style("opacity", 1);
        }, inited?300:0); 
    };

    const resizeObserver = new ResizeObserver(onResize);
    resizeObserver.observe(wrapper.node());
}