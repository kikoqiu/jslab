import {nodeResolve} from "@rollup/plugin-node-resolve"
import typescript from '@rollup/plugin-typescript'; 
import resolve from '@rollup/plugin-node-resolve';
import commonjs from '@rollup/plugin-commonjs';



export default {
  input: "./editor.mjs",
  output: {
    file: "../editor.bundle.js",
    format: "iife",
    inlineDynamicImports: true,
    /*sourcemap: 'inline',*/
  },
  plugins: [
    nodeResolve({extensions: ['.js', '.ts', '.json'] }),
    typescript({
      tsconfig: './tsconfig.json'
    }),
    commonjs(),
  ]
}
