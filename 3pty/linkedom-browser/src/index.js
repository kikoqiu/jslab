// src/index.js
import * as linkedom from 'linkedom';

// Expose the linkedom module to the global scope so it can be used in the browser
if (typeof globalThis !== 'undefined') {
  globalThis.linkedom = linkedom;
}
