"use strict";
var groups = (() => {
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

  // src/groups.js
  var groups_exports = {};
  __export(groups_exports, {
    AbstractGroupSet: () => AbstractGroupSet,
    IntSetUtils: () => IntSetUtils,
    PermutationRepository: () => PermutationRepository,
    PermutationSet: () => PermutationSet,
    SchreierSimsAlgorithm: () => SchreierSimsAlgorithm,
    VisualizerCayleyForceSimulator: () => VisualizerCayleyForceSimulator,
    analyzeGenerators: () => analyzeGenerators,
    areIsomorphic: () => areIsomorphic,
    createAlternating: () => createAlternating,
    createCyclic: () => createCyclic,
    createDihedral: () => createDihedral,
    createDirectProduct: () => createDirectProduct,
    createFromCycleStrings: () => createFromCycleStrings,
    createFromRawArrays: () => createFromRawArrays,
    createIcosahedral: () => createIcosahedral,
    createKleinFour: () => createKleinFour,
    createOctahedral: () => createOctahedral,
    createQuaternion: () => createQuaternion,
    createSymmetric: () => createSymmetric,
    createTetrahedral: () => createTetrahedral,
    createTrivial: () => createTrivial,
    decomposeToCycles: () => decomposeToCycles,
    generateCayleyGraphForPlotly: () => generateCayleyGraphForPlotly,
    generateGroup: () => generateGroup,
    generateMultiplicationTable: () => generateMultiplicationTable,
    generateNames: () => generateNames,
    getCommutatorSubgroup: () => getCommutatorSubgroup,
    getLowerCentralSeries: () => getLowerCentralSeries,
    getMixedCommutatorSubgroup: () => getMixedCommutatorSubgroup,
    getNormalClosure: () => getNormalClosure,
    getQuotientStructure: () => getQuotientStructure,
    getSylowSubgroup: () => getSylowSubgroup,
    globalRepo: () => globalRepo,
    isNilpotent: () => isNilpotent,
    isNormal: () => isNormal,
    isSimple: () => isSimple,
    isSolvable: () => isSolvable,
    isSubgroup: () => isSubgroup,
    parseCycles: () => parseCycles,
    resetGlobalRepo: () => resetGlobalRepo
  });

  // src/int-set-utils.js
  var IntSetUtils = {
    /**
     * Binary Search for value existence.
     * @param {Int32Array} sortedArr - The sorted array to search within.
     * @param {number} value - The value to search for.
     * @returns {boolean} True if the value is found in the array.
     */
    has(sortedArr, value) {
      let left = 0;
      let right = sortedArr.length - 1;
      while (left <= right) {
        const mid = left + right >>> 1;
        const v = sortedArr[mid];
        if (v === value) return true;
        if (v < value) left = mid + 1;
        else right = mid - 1;
      }
      return false;
    },
    /**
     * Computes the union of two sorted Int32Arrays (A U B).
     * The resulting array will contain all unique elements from both input arrays, sorted in ascending order.
     * Linear time complexity O(|A| + |B|).
     * @param {Int32Array} arrA - The first sorted Int32Array.
     * @param {Int32Array} arrB - The second sorted Int32Array.
     * @returns {Int32Array} A new sorted Int32Array containing the union of elements.
     */
    union(arrA, arrB) {
      const lenA = arrA.length;
      const lenB = arrB.length;
      if (lenA === 0) return arrB.slice();
      if (lenB === 0) return arrA.slice();
      const res = new Int32Array(lenA + lenB);
      let i = 0, j = 0, k = 0;
      while (i < lenA && j < lenB) {
        const va = arrA[i];
        const vb = arrB[j];
        if (va < vb) {
          res[k++] = va;
          i++;
        } else if (va > vb) {
          res[k++] = vb;
          j++;
        } else {
          res[k++] = va;
          i++;
          j++;
        }
      }
      while (i < lenA) res[k++] = arrA[i++];
      while (j < lenB) res[k++] = arrB[j++];
      return res.subarray(0, k);
    },
    /**
     * Computes the intersection of two sorted Int32Arrays (A ∩ B).
     * The resulting array will contain only the elements common to both input arrays, sorted in ascending order.
     * Linear time complexity O(|A| + |B|).
     * @param {Int32Array} arrA - The first sorted Int32Array.
     * @param {Int32Array} arrB - The second sorted Int32Array.
     * @returns {Int32Array} A new sorted Int32Array containing the intersection of elements.
     */
    intersection(arrA, arrB) {
      const lenA = arrA.length;
      const lenB = arrB.length;
      const res = new Int32Array(lenA < lenB ? lenA : lenB);
      let i = 0, j = 0, k = 0;
      while (i < lenA && j < lenB) {
        const va = arrA[i];
        const vb = arrB[j];
        if (va < vb) {
          i++;
        } else if (va > vb) {
          j++;
        } else {
          res[k++] = va;
          i++;
          j++;
        }
      }
      return res.subarray(0, k);
    },
    /**
     * Computes the difference of two sorted Int32Arrays (A - B).
     * The resulting array will contain elements present in `arrA` but not in `arrB`, sorted in ascending order.
     * Linear time complexity O(|A| + |B|).
     * @param {Int32Array} arrA - The minuend sorted Int32Array.
     * @param {Int32Array} arrB - The subtrahend sorted Int32Array.
     * @returns {Int32Array} A new sorted Int32Array containing the difference (A - B) of elements.
     */
    difference(arrA, arrB) {
      const lenA = arrA.length;
      const lenB = arrB.length;
      const res = new Int32Array(lenA);
      let i = 0, j = 0, k = 0;
      while (i < lenA && j < lenB) {
        const va = arrA[i];
        const vb = arrB[j];
        if (va < vb) {
          res[k++] = va;
          i++;
        } else if (va > vb) {
          j++;
        } else {
          i++;
          j++;
        }
      }
      while (i < lenA) res[k++] = arrA[i++];
      return res.subarray(0, k);
    },
    /**
     * Sorts an Int32Array in ascending order and removes duplicate elements.
     * This function mutates the input array by sorting it in-place and then returns a subarray view
     * containing only the unique elements.
     * @param {Int32Array} rawArr - The Int32Array to sort and deduplicate. This array will be mutated.
     * @returns {Int32Array} A subarray view of the input `rawArr` containing sorted unique elements.
     */
    sortAndUnique(rawArr) {
      if (rawArr.length <= 1) return rawArr;
      rawArr.sort();
      let k = 0;
      const len = rawArr.length;
      for (let i = 0; i < len; i++) {
        if (i === 0 || rawArr[i] !== rawArr[i - 1]) {
          rawArr[k++] = rawArr[i];
        }
      }
      return rawArr.subarray(0, k);
    }
  };

  // src/group-utils.js
  function parseCycles(str, degree = 0) {
    const cycleStrings = str.match(/\(([^)]+)\)/g) || [];
    const cycles = cycleStrings.map(
      (s) => s.replace(/[()]/g, "").trim().split(/\s+/).map(Number)
    );
    if (degree === 0) {
      const maxVal = cycles.length > 0 ? Math.max(...cycles.flat()) : 0;
      degree = Math.max(maxVal, 0);
    }
    const perm = new Int32Array(degree);
    for (let i = 0; i < degree; i++) perm[i] = i;
    cycles.forEach((cycle) => {
      const len = cycle.length;
      if (len < 2) return;
      for (let i = 0; i < len; i++) {
        const current = cycle[i] - 1;
        const next = cycle[(i + 1) % len] - 1;
        if (current >= degree || next >= degree) {
          continue;
        }
        perm[current] = next;
      }
    });
    return perm;
  }
  function decomposeToCycles(perm) {
    let permArr;
    if (typeof perm === "number") {
      permArr = globalRepo.get(perm);
    } else {
      permArr = perm;
    }
    const n = permArr.length;
    const visited = new Uint8Array(n);
    const cycles = [];
    for (let i = 0; i < n; i++) {
      if (visited[i] === 0) {
        let curr = i;
        if (permArr[curr] === curr) {
          visited[curr] = 1;
          continue;
        }
        const cycle = [];
        while (visited[curr] === 0) {
          visited[curr] = 1;
          cycle.push(curr + 1);
          curr = permArr[curr];
        }
        if (cycle.length > 1) {
          cycles.push(`(${cycle.join(" ")})`);
        }
      }
    }
    return cycles.join("") || "()";
  }
  function createSymmetric(n) {
    if (n <= 1) return PermutationSet.identity(n);
    const genSwap = new Int32Array(n);
    for (let i = 0; i < n; i++) genSwap[i] = i;
    genSwap[0] = 1;
    genSwap[1] = 0;
    if (n === 2) {
      return new PermutationSet([
        globalRepo.register(genSwap)
      ], true, false);
    }
    const genCycle = new Int32Array(n);
    for (let i = 0; i < n - 1; i++) genCycle[i] = i + 1;
    genCycle[n - 1] = 0;
    const ids = [
      globalRepo.register(genSwap),
      globalRepo.register(genCycle)
    ];
    return new PermutationSet(ids, true, false);
  }
  function createAlternating(n) {
    if (n <= 2) return PermutationSet.identity(n);
    const ids = [];
    for (let i = 2; i < n; i++) {
      const perm = new Int32Array(n);
      for (let k = 0; k < n; k++) perm[k] = k;
      perm[0] = 1;
      perm[1] = i;
      perm[i] = 0;
      ids.push(globalRepo.register(perm));
    }
    return new PermutationSet(ids, true, false);
  }
  function createCyclic(n) {
    if (n <= 1) return PermutationSet.identity(n);
    const perm = new Int32Array(n);
    for (let i = 0; i < n - 1; i++) perm[i] = i + 1;
    perm[n - 1] = 0;
    return new PermutationSet([
      globalRepo.register(perm)
    ], true, false);
  }
  function createDihedral(n) {
    if (n <= 2) return createSymmetric(n);
    const rot = new Int32Array(n);
    for (let i = 0; i < n - 1; i++) rot[i] = i + 1;
    rot[n - 1] = 0;
    const ref = new Int32Array(n);
    ref[0] = 0;
    for (let i = 1; i < n; i++) {
      ref[i] = n - i;
    }
    const ids = [
      globalRepo.register(rot),
      globalRepo.register(ref)
    ];
    return new PermutationSet(ids, true, false);
  }
  function createKleinFour() {
    const a = new Int32Array([1, 0, 3, 2]);
    const b = new Int32Array([2, 3, 0, 1]);
    return new PermutationSet([
      globalRepo.register(a),
      globalRepo.register(b)
    ], true, false);
  }
  function createFromCycleStrings(cyclesStrArr, degree = 0) {
    const ids = [];
    for (const str of cyclesStrArr) {
      const permArr = parseCycles(str, degree);
      ids.push(globalRepo.register(permArr));
    }
    return new PermutationSet(ids, false, false);
  }
  function createDirectProduct(groupA, groupB) {
    const getEffectiveDegree = (groupSet) => {
      let max = 0;
      for (const id of groupSet.indices) {
        const arr = globalRepo.get(id);
        for (let i = arr.length - 1; i >= 0; i--) {
          if (arr[i] !== i) {
            if (i + 1 > max) max = i + 1;
            break;
          }
        }
      }
      return max;
    };
    const degA = Math.max(getEffectiveDegree(groupA), 1);
    const degB = Math.max(getEffectiveDegree(groupB), 1);
    const totalDegree = degA + degB;
    const newIds = [];
    for (const idA of groupA.indices) {
      if (idA === globalRepo.identity) continue;
      const permA = globalRepo.get(idA);
      const newPerm = new Int32Array(totalDegree);
      const lenA = Math.min(permA.length, degA);
      for (let i = 0; i < lenA; i++) newPerm[i] = permA[i];
      for (let i = lenA; i < degA; i++) newPerm[i] = i;
      for (let i = degA; i < totalDegree; i++) newPerm[i] = i;
      newIds.push(globalRepo.register(newPerm));
    }
    for (const idB of groupB.indices) {
      if (idB === globalRepo.identity) continue;
      const permB = globalRepo.get(idB);
      const newPerm = new Int32Array(totalDegree);
      for (let i = 0; i < degA; i++) newPerm[i] = i;
      for (let i = 0; i < degB; i++) {
        const val = i < permB.length ? permB[i] : i;
        newPerm[degA + i] = degA + val;
      }
      newIds.push(globalRepo.register(newPerm));
    }
    return new PermutationSet(newIds, false, false);
  }
  function createQuaternion() {
    const i_gen = new Int32Array([1, 4, 7, 2, 5, 0, 3, 6]);
    const j_gen = new Int32Array([2, 3, 4, 5, 6, 7, 0, 1]);
    return new PermutationSet([
      globalRepo.register(i_gen),
      globalRepo.register(j_gen)
    ], true, false);
  }
  function createTrivial() {
    return PermutationSet.identity(1);
  }
  function createFromRawArrays(arrays) {
    const ids = arrays.map((arr) => globalRepo.register(arr));
    return new PermutationSet(ids, false, false);
  }
  function createTetrahedral() {
    return createAlternating(4);
  }
  function createOctahedral() {
    return createSymmetric(4);
  }
  function createIcosahedral() {
    return createAlternating(5);
  }

  // src/permutation-repository.js
  var INITIAL_TRIE_MEMORY_BYTES = 1024 * 1024;
  var BYTES_PER_INT = 4;
  var INITIAL_TRIE_SLOTS = INITIAL_TRIE_MEMORY_BYTES / BYTES_PER_INT;
  var PermutationRepository = class {
    /**
     * @param {number} [initialDegree=4] - The initial degree (number of points) for permutations.
     *                                     The repository will automatically expand if permutations with higher degrees are registered.
     * @param {number} [initialPermCapacity=1024] - The initial capacity for storing permutations.
     *                                                The capacity will automatically expand as more unique permutations are registered.
     */
    constructor(initialDegree = 4, initialPermCapacity = 1024) {
      this.globalDegree = initialDegree;
      this.count = 0;
      this.permCapacity = initialPermCapacity;
      this.permBuffer = new Int32Array(this.permCapacity * this.globalDegree);
      this.trieNodeSize = this.globalDegree + 1;
      this.trieBuffer = new Int32Array(INITIAL_TRIE_SLOTS);
      this.trieFreePtr = 0;
      this._allocateNode();
      this.identity = this.register([]);
      if (this.identity != 0) {
        throw new Error("this.identity != 0");
      }
    }
    /**
     * Allocates a new node from the Trie Buffer memory arena.
     * Auto-expands the buffer if more space is needed.
     * @returns {number} The starting index (pointer) of the newly allocated node within the `trieBuffer`.
     * @private
     */
    _allocateNode() {
      if (this.trieFreePtr + this.trieNodeSize > this.trieBuffer.length) {
        this._expandTrieBuffer();
      }
      const ptr = this.trieFreePtr;
      const size = this.trieNodeSize;
      for (let i = 0; i < size; i++) {
        this.trieBuffer[ptr + i] = -1;
      }
      this.trieFreePtr += size;
      return ptr;
    }
    /**
     * Expands the `trieBuffer` (memory arena for trie nodes) when it runs out of space.
     * Doubles the current capacity.
     * @private
     */
    _expandTrieBuffer() {
      const oldLen = this.trieBuffer.length;
      const newLen = oldLen * 2;
      const newBuf = new Int32Array(newLen);
      newBuf.set(this.trieBuffer);
      this.trieBuffer = newBuf;
    }
    /**
     * Registers a permutation (or retrieves its existing ID if already registered).
     * If the input permutation's degree is greater than the current `globalDegree`,
     * the repository will automatically upgrade its degree.
     * @param {ArrayLike<number>} rawPerm - The permutation to register, represented as an array-like object (e.g., `[0, 2, 1]`).
     * @returns {number} The unique ID assigned to the permutation.
     */
    register(rawPerm) {
      const inputLen = rawPerm.length;
      if (inputLen > this.globalDegree) {
        this._upgradeDegree(inputLen);
      }
      if (this.count >= this.permCapacity) {
        this._expandPermCapacity();
      }
      const n = this.globalDegree;
      let currNodePtr = 0;
      for (let i = 0; i < n; i++) {
        const val = i < inputLen ? rawPerm[i] : i;
        const childSlotIdx = currNodePtr + 1 + val;
        let nextNodePtr = this.trieBuffer[childSlotIdx];
        if (nextNodePtr === -1) {
          nextNodePtr = this._allocateNode();
          this.trieBuffer[childSlotIdx] = nextNodePtr;
        }
        currNodePtr = nextNodePtr;
      }
      let id = this.trieBuffer[currNodePtr];
      if (id === -1) {
        id = this.count++;
        this.trieBuffer[currNodePtr] = id;
        this._writeToPermBuffer(id, rawPerm, inputLen);
      }
      return id;
    }
    /**
     * Retrieves the permutation data for a given ID.
     * Returns a zero-copy view (subarray) of the internal `permBuffer`.
     * @param {number} id - The unique ID of the permutation to retrieve.
     * @returns {Int32Array} A subarray representing the permutation (e.g., `[0, 1, 2]`).
     */
    get(id) {
      const start = id * this.globalDegree;
      return this.permBuffer.subarray(start, start + this.globalDegree);
    }
    /**
     * Retrieves the permutation for a given ID and converts it into a 1-based cycle notation string.
     * @param {number} id - The unique ID of the permutation.
     * @returns {string} The cycle notation string (e.g., "(1 2 3)(4 5)"). Returns "()" for the identity permutation.
     */
    getAsCycles(id) {
      return decomposeToCycles(this.get(id));
    }
    /**
     * Writes a new permutation into the `permBuffer` at the specified ID's location.
     * Pads with identity mappings if `inputArr` is shorter than `globalDegree`.
     * @param {number} id - The unique ID assigned to this permutation.
     * @param {ArrayLike<number>} inputArr - The raw permutation array-like object.
     * @param {number} validLen - The actual length of the `inputArr` to copy.
     * @private
     */
    _writeToPermBuffer(id, inputArr, validLen) {
      const n = this.globalDegree;
      const offset = id * n;
      for (let i = 0; i < validLen; i++) this.permBuffer[offset + i] = inputArr[i];
      for (let i = validLen; i < n; i++) this.permBuffer[offset + i] = i;
    }
    /**
     * Expands the `permBuffer` (permutation data pool) when it runs out of space.
     * Doubles the current capacity, copying existing data to the new buffer.
     * @private
     */
    _expandPermCapacity() {
      this.permCapacity *= 2;
      const newBuf = new Int32Array(this.permCapacity * this.globalDegree);
      newBuf.set(this.permBuffer);
      this.permBuffer = newBuf;
    }
    /**
     * Upgrades the `globalDegree` of the repository.
     * This is a "stop-the-world" operation that rebuilds both the permutation pool and the trie.
     * Existing permutations are padded with identity mappings to match the new degree.
     * @param {number} newDegree - The new, larger degree to upgrade to.
     * @private
     */
    _upgradeDegree(newDegree) {
      const oldDegree = this.globalDegree;
      const oldPermBuffer = this.permBuffer;
      const totalPerms = this.count;
      this.globalDegree = newDegree;
      this.trieNodeSize = newDegree + 1;
      this.permBuffer = new Int32Array(this.permCapacity * newDegree);
      for (let i = 0; i < totalPerms; i++) {
        const oldStart = i * oldDegree;
        const newStart = i * newDegree;
        for (let k = 0; k < oldDegree; k++) this.permBuffer[newStart + k] = oldPermBuffer[oldStart + k];
        for (let k = oldDegree; k < newDegree; k++) this.permBuffer[newStart + k] = k;
      }
      this.trieFreePtr = 0;
      this._allocateNode();
      for (let id = 0; id < totalPerms; id++) {
        const permOffset = id * newDegree;
        let currNodePtr = 0;
        for (let i = 0; i < newDegree; i++) {
          const val = this.permBuffer[permOffset + i];
          const childSlotIdx = currNodePtr + 1 + val;
          let nextNodePtr = this.trieBuffer[childSlotIdx];
          if (nextNodePtr === -1) {
            nextNodePtr = this._allocateNode();
            this.trieBuffer[childSlotIdx] = nextNodePtr;
          }
          currNodePtr = nextNodePtr;
        }
        this.trieBuffer[currNodePtr] = id;
      }
    }
    /**
     * Computes the inverse of a given permutation ID.
     * If the inverse has already been registered, its ID is retrieved; otherwise, it's computed and registered.
     * @param {number} id - The ID of the permutation to invert.
     * @returns {number} The ID of the inverse permutation.
     */
    inverse(id) {
      if (id === this.identity) return this.identity;
      const N = this.globalDegree;
      const buf = this.permBuffer;
      const off = id * N;
      const res = new Int32Array(N);
      for (let k = 0; k < N; k++) {
        res[buf[off + k]] = k;
      }
      return this.register(res);
    }
    /**
     * Multiplies two permutations, `idA` and `idB`, according to the convention (A * B)(x) = A(B(x)).
     * This means permutation `idB` is applied first, then `idA`.
     * The resulting permutation is registered, and its ID is returned.
     * Exposed as Public API for solvers.
     * @param {number} idA - The ID of the first permutation (A).
     * @param {number} idB - The ID of the second permutation (B).
     * @returns {number} The ID of the resulting permutation (A * B).
     */
    multiply(idA, idB) {
      if (idA === this.identity) return idB;
      if (idB === this.identity) return idA;
      const N = this.globalDegree;
      const buf = this.permBuffer;
      const res = new Int32Array(N);
      const offA = idA * N;
      const offB = idB * N;
      for (let k = 0; k < N; k++) {
        const valB = buf[offB + k];
        res[k] = buf[offA + valB];
      }
      return this.register(res);
    }
    /**
     * Computes the conjugate of permutation `h` by `g`: `g * h * g^-1`.
     * This operation results in a permutation that has the same cycle structure as `h`.
     * @param {number} g - The ID of the conjugating permutation (g).
     * @param {number} h - The ID of the permutation to be conjugated (h).
     * @returns {number} The ID of the resulting conjugated permutation (g * h * g^-1).
     */
    conjugate(g, h) {
      const gInv = this.inverse(g);
      const gh = this.multiply(g, h);
      return this.multiply(gh, gInv);
    }
    /**
     * Computes the commutator of two permutations: `[idA, idB] = idA^-1 * idB^-1 * idA * idB`.
     * @param {number} idA - The ID of the first permutation (a).
     * @param {number} idB - The ID of the second permutation (b).
     * @returns {number} The ID of the resulting commutator permutation.
     */
    commutator(idA, idB) {
      const invA = this.inverse(idA);
      const invB = this.inverse(idB);
      const step1 = this.multiply(invA, invB);
      const step2 = this.multiply(step1, idA);
      return this.multiply(step2, idB);
    }
  };
  var globalRepo = new PermutationRepository();
  function resetGlobalRepo() {
    globalRepo = new PermutationRepository();
  }

  // src/group-engine.js
  var _tempCompositionBuffer = new Int32Array(1024);
  function _ensureTempBuffer(requiredSize) {
    if (_tempCompositionBuffer.length < requiredSize) {
      const newSize = Math.max(requiredSize, _tempCompositionBuffer.length * 2);
      _tempCompositionBuffer = new Int32Array(newSize);
    }
  }
  var AbstractGroupSet = class _AbstractGroupSet {
    constructor() {
      if (new.target === _AbstractGroupSet) {
        throw new Error("AbstractGroupSet is not instantiable.");
      }
    }
    // Abstract Methods
    /**
     * The number of elements in the set.
     * @type {number}
     * @abstract
     */
    get size() {
      throw new Error("Method not implemented.");
    }
    /**
     * Multiplies this set by another set.
     * @param {AbstractGroupSet} other - The other set to multiply by.
     * @returns {AbstractGroupSet} A new set representing the product.
     * @abstract
     */
    // eslint-disable-next-line
    multiply(other) {
      throw new Error("Method not implemented.");
    }
    /**
     * Computes the inverse of each element in the set.
     * @returns {AbstractGroupSet} A new set containing the inverses.
     * @abstract
     */
    inverse() {
      throw new Error("Method not implemented.");
    }
    /**
     * Computes the union of this set with another set.
     * @param {AbstractGroupSet} other - The other set.
     * @returns {AbstractGroupSet} A new set representing the union.
     * @abstract
     */
    // eslint-disable-next-line
    union(other) {
      throw new Error("Method not implemented.");
    }
    /**
     * Computes the intersection of this set with another set.
     * @param {AbstractGroupSet} other - The other set.
     * @returns {AbstractGroupSet} A new set representing the intersection.
     * @abstract
     */
    // eslint-disable-next-line
    intersection(other) {
      throw new Error("Method not implemented.");
    }
    /**
     * Computes the difference of this set with another set (elements in this set but not in `other`).
     * @param {AbstractGroupSet} other - The other set.
     * @returns {AbstractGroupSet} A new set representing the difference.
     * @abstract
     */
    // eslint-disable-next-line
    difference(other) {
      throw new Error("Method not implemented.");
    }
    /**
     * Checks if this set is a superset of another set.
     * @param {AbstractGroupSet} other - The other set.
     * @returns {boolean} True if this set contains all elements of `other`.
     * @abstract
     */
    // eslint-disable-next-line
    isSuperSetOf(other) {
      throw new Error("Method not implemented.");
    }
    /**
     * Creates a lightweight read-only view of a subset.
     * @param {number} start - The starting index (inclusive).
     * @param {number} end - The ending index (exclusive).
     * @returns {AbstractGroupSet} A new set representing the slice.
     * @abstract
     */
    // eslint-disable-next-line
    slice(start, end) {
      throw new Error("Method not implemented.");
    }
    /**
     * Returns an iterator for the elements in the set.
     * @returns {Iterator<number>} An iterator for the set's element IDs.
     * @abstract
     */
    [Symbol.iterator]() {
      throw new Error("Method not implemented.");
    }
  };
  var PermutationSet = class _PermutationSet extends AbstractGroupSet {
    /**
     * @param {Int32Array|Array<number>} ids - Sorted, unique IDs from the repository.
     * @param {boolean} [isTrustedSortedUnique=false] - Skip sort/dedup if true.
     * @param {boolean} [isGroup=false] - Whether this set is known to be a mathematical group.
     */
    constructor(ids, isTrustedSortedUnique = false, isGroup = false) {
      super();
      if (isTrustedSortedUnique && ids instanceof Int32Array) {
        this._ids = ids;
      } else {
        const raw = ids instanceof Int32Array ? ids : new Int32Array(ids);
        this._ids = isTrustedSortedUnique ? raw : IntSetUtils.sortAndUnique(raw);
      }
      this.isGroup = isGroup;
    }
    // ------------------------------------------------------------------------
    // Read-Only Accessors
    // ------------------------------------------------------------------------
    /**
     * The number of elements in the set.
     * @type {number}
     */
    get size() {
      return this._ids.length;
    }
    /**
     * Returns the internal Int32Array of sorted, unique permutation IDs.
     * Direct access should be read-only.
     * @returns {Int32Array}
     */
    get indices() {
      return this._ids;
    }
    /**
     * Retrieves a permutation ID at a specific index within this set.
     * @param {number} index - The 0-based index of the element to retrieve.
     * @returns {number} The permutation ID.
     */
    get(index) {
      return this._ids[index];
    }
    /**
     * Returns an iterator for the permutation IDs in this set.
     * @returns {Iterator<number>} An iterator for the `_ids` Int32Array.
     */
    [Symbol.iterator]() {
      return this._ids[Symbol.iterator]();
    }
    /**
     * Creates a lightweight read-only view of a subset.
     * @param {number} start - The starting index (inclusive).
     * @param {number} end - The ending index (exclusive).
     * @returns {PermutationSet} A new set representing the slice.
     * @abstract
     */
    slice(start, end) {
      return new _PermutationSet(this._ids.subarray(start, end), true, false);
    }
    /**
     * Returns a string representation of the PermutationSet.
     * @returns {string} A string in the format "PermSet(ids=[...], isGroup=...)".
     */
    toString() {
      let eles = Array.from(this._ids).map((id) => `${globalRepo.getAsCycles(id)}`).join(",");
      return `PermSet( {${eles}}, ids=[${this._ids.join(",")}] size=${this.size} isGroup=${this.isGroup})`;
    }
    // ------------------------------------------------------------------------
    // Core Algebra (Performance Critical)
    // ------------------------------------------------------------------------
    /**
     * Vectorized Group Multiplication: G * H = { g * h | g in G, h in H }
     * Optimized with direct heap access and loop hoisting.
     * Multiplies this set by another set.
     * @param {PermutationSet} other - The other set to multiply by.
     * @returns {PermutationSet} A new set representing the product.
     * @abstract
     */
    multiply(other) {
      if (!(other instanceof _PermutationSet)) {
        throw new Error("Type mismatch: Expected PermutationSet.");
      }
      const sizeA = this._ids.length;
      const sizeB = other._ids.length;
      if (sizeA === 0 || sizeB === 0) {
        return new _PermutationSet(new Int32Array(0), true, false);
      }
      const repo = globalRepo;
      const N = repo.globalDegree;
      const permBuffer = repo.permBuffer;
      _ensureTempBuffer(N);
      const tempBuf = _tempCompositionBuffer;
      const resultIds = new Int32Array(sizeA * sizeB);
      let ptr = 0;
      const idsA = this._ids;
      const idsB = other._ids;
      if (sizeA <= sizeB) {
        for (let i = 0; i < sizeA; i++) {
          const idA = idsA[i];
          const offsetA = idA * N;
          for (let j = 0; j < sizeB; j++) {
            const idB = idsB[j];
            const offsetB = idB * N;
            for (let k = 0; k < N; k++) {
              const valB = permBuffer[offsetB + k];
              tempBuf[k] = permBuffer[offsetA + valB];
            }
            resultIds[ptr++] = repo.register(tempBuf.subarray(0, N));
          }
        }
      } else {
        for (let j = 0; j < sizeB; j++) {
          const idB = idsB[j];
          const offsetB = idB * N;
          for (let i = 0; i < sizeA; i++) {
            const idA = idsA[i];
            const offsetA = idA * N;
            for (let k = 0; k < N; k++) {
              const valB = permBuffer[offsetB + k];
              tempBuf[k] = permBuffer[offsetA + valB];
            }
            resultIds[ptr++] = repo.register(tempBuf.subarray(0, N));
          }
        }
      }
      return new _PermutationSet(resultIds, false, false);
    }
    /**
     * Vectorized Inverse: G^-1 = { g^-1 | g in G }
     * Computes the inverse of each element in the set.
     * @returns {PermutationSet} A new set containing the inverses.
     */
    inverse() {
      const size = this._ids.length;
      if (size === 0) return new _PermutationSet(new Int32Array(0), true, this.isGroup);
      const repo = globalRepo;
      const N = repo.globalDegree;
      const permBuffer = repo.permBuffer;
      _ensureTempBuffer(N);
      const tempBuf = _tempCompositionBuffer;
      const resultIds = new Int32Array(size);
      const ids = this._ids;
      for (let i = 0; i < size; i++) {
        const offset = ids[i] * N;
        for (let k = 0; k < N; k++) {
          const val = permBuffer[offset + k];
          tempBuf[val] = k;
        }
        resultIds[i] = repo.register(tempBuf.subarray(0, N));
      }
      return new _PermutationSet(resultIds, false, this.isGroup);
    }
    // ------------------------------------------------------------------------
    // Set Operations (Delegated to IntSetUtils)
    // ------------------------------------------------------------------------
    /**
     * Computes the union of this set with another set.
     * @param {PermutationSet} other - The other set.
     * @returns {PermutationSet} A new set representing the union.
     */
    union(other) {
      this._checkType(other);
      return new _PermutationSet(
        IntSetUtils.union(this._ids, other._ids),
        true,
        false
      );
    }
    /**
     * Computes the intersection of this set with another set.
     * @param {PermutationSet} other - The other set.
     * @returns {PermutationSet} A new set representing the intersection.
     */
    intersection(other) {
      this._checkType(other);
      const resultIsGroup = this.isGroup && other.isGroup;
      return new _PermutationSet(
        IntSetUtils.intersection(this._ids, other._ids),
        true,
        resultIsGroup
      );
    }
    /**
     * Computes the difference of this set with another set (elements in this set but not in `other`).
     * @param {PermutationSet} other - The other set.
     * @returns {PermutationSet} A new set representing the difference.
     */
    difference(other) {
      this._checkType(other);
      return new _PermutationSet(
        IntSetUtils.difference(this._ids, other._ids),
        true,
        false
      );
    }
    /**
     * Checks if this set is a superset of another set.
     * @param {PermutationSet} other - The other set.
     * @returns {boolean} True if this set contains all elements of `other`.
     */
    isSuperSetOf(other) {
      this._checkType(other);
      if (this.size < other.size) return false;
      const A = this._ids;
      const B = other._ids;
      const lenB = B.length;
      for (let i = 0; i < lenB; i++) {
        if (!IntSetUtils.has(A, B[i])) return false;
      }
      return true;
    }
    /**
     * Checks if equal.
     * @param {PermutationSet} other - The other set.
     * @returns {boolean}
     */
    equals(other) {
      if (this === other) return true;
      if (this.size !== other.size) return false;
      const A = this._ids;
      const B = other._ids;
      const len = A.length;
      for (let i = 0; i < len; i++) {
        if (A[i] !== B[i]) return false;
      }
      return true;
    }
    /**
     * Internal helper to validate that the 'other' operand is a PermutationSet.
     * @param {*} other - The operand to check.
     * @throws {Error} If `other` is not an instance of PermutationSet.
     * @private
     */
    _checkType(other) {
      if (!(other instanceof _PermutationSet)) {
        throw new Error("Operation requires PermutationSet.");
      }
    }
    // ------------------------------------------------------------------------
    // Factory Methods
    // ------------------------------------------------------------------------
    /**
     * Creates a PermutationSet containing only the identity permutation.
     * This set is always considered a group.
     * @returns {PermutationSet} A PermutationSet containing only the identity permutation.
     * @static
     */
    static identity() {
      return new _PermutationSet(new Int32Array([globalRepo.identity]), true, true);
    }
    // ------------------------------------------------------------------------
    // Group Theory Algorithms
    // ------------------------------------------------------------------------
    /**
     * Checks if the set forms an Abelian (Commutative) group.
     * Logic: For all g1, g2 in G, g1 * g2 == g2 * g1.
     * Performance: O(|G|^2 * Degree). Optimized with direct memory access.
     * @returns {boolean}
     */
    isAbelian() {
      if (this.size <= 1) return true;
      const repo = globalRepo;
      const N = repo.globalDegree;
      const permBuffer = repo.permBuffer;
      const ids = this._ids;
      const count = ids.length;
      for (let i = 0; i < count; i++) {
        const idA = ids[i];
        const offsetA = idA * N;
        for (let j = i + 1; j < count; j++) {
          const idB = ids[j];
          const offsetB = idB * N;
          for (let k = 0; k < N; k++) {
            const b_val = permBuffer[offsetB + k];
            const ab_val = permBuffer[offsetA + b_val];
            const a_val = permBuffer[offsetA + k];
            const ba_val = permBuffer[offsetB + a_val];
            if (ab_val !== ba_val) {
              return false;
            }
          }
        }
      }
      return true;
    }
    /**
     * Calculates the Orbit of a point under this group.
     * Orbit(p) = { g(p) | g in G }
     * Implements BFS/Flood Fill without object allocation.
     * 
     * @param {number} point - The integer point (0..degree-1)
     * @returns {Int32Array} Sorted unique array of points in the orbit.
     */
    calculateOrbit(point) {
      const repo = globalRepo;
      const N = repo.globalDegree;
      if (point < 0 || point >= N) {
        throw new Error(`Point ${point} out of bounds (0..${N - 1})`);
      }
      const visited = new Uint8Array(N);
      visited[point] = 1;
      const result = [point];
      const ids = this._ids;
      const size = ids.length;
      const permBuffer = repo.permBuffer;
      let head = 0;
      while (head < result.length) {
        const currPoint = result[head++];
        for (let i = 0; i < size; i++) {
          const id = ids[i];
          const nextPoint = permBuffer[id * N + currPoint];
          if (visited[nextPoint] === 0) {
            visited[nextPoint] = 1;
            result.push(nextPoint);
          }
        }
      }
      const orbitArr = new Int32Array(result);
      orbitArr.sort();
      return orbitArr;
    }
    /**
     * Decomposes this group G into right cosets of a subgroup H.
     * G = U (H * g_i)
     * 
     * 
     * @param {PermutationSet} subgroupH - The subgroup H to decompose by.
     * @returns {Array<PermutationSet>} An array of disjoint right cosets.
     */
    rightCosetDecomposition(subgroupH) {
      this._checkType(subgroupH);
      if (subgroupH.size > this.size) {
        throw new Error("H cannot be larger than G.");
      }
      const cosets = [];
      const visited = new Uint8Array(globalRepo.count);
      const gIds = this._ids;
      const len = gIds.length;
      for (let i = 0; i < len; i++) {
        const gId = gIds[i];
        if (visited[gId] === 1) {
          continue;
        }
        const representative = new _PermutationSet([gId], true, false);
        const coset = subgroupH.multiply(representative);
        cosets.push(coset);
        const cosetIds = coset._ids;
        const cosetLen = cosetIds.length;
        for (let k = 0; k < cosetLen; k++) {
          visited[cosetIds[k]] = 1;
        }
      }
      return cosets;
    }
    /**
     * Generates a subgroup from this.
     * This method uses an iterative closure approach by repeatedly multiplying the current group by the generators until no new elements are found.
     * @returns {PermutationSet} The fully generated subgroup (isGroup=true).
     * @throws {Error} If `generators` is an unknown type.
     */
    generateGroupFromThis() {
      return generateGroup(this);
    }
  };
  function generateGroup(generators) {
    if (Array.isArray(generators)) {
      generators = new PermutationSet(generators);
    } else {
      if (!(generators instanceof PermutationSet)) {
        throw new Error("unknown generators type");
      }
    }
    if (generators.isGroup) {
      return generators;
    }
    let group = generators.union(generators.inverse()).union(PermutationSet.identity());
    let lastSize = 0;
    while (group.size !== lastSize) {
      lastSize = group.size;
      const nextLevel = group.multiply(generators);
      group = group.union(nextLevel);
    }
    group.isGroup = true;
    return group;
  }

  // src/schreier-sims.js
  var SchreierSimsAlgorithm = class _SchreierSimsAlgorithm {
    /**
     * Constructs a new Schreier-Sims Algorithm instance.
     * The instance can be initialized with an optional `initialBase` to prioritize certain points in the base.
     * @param {number[]} [initialBase=[]] - An optional array of points to form the initial base. These points will be stabilized first.
     */
    constructor(initialBase = []) {
      this.repo = globalRepo;
      this.base = [];
      this.transversals = [];
      this.generators = [];
      this.idIdentity = this.repo.identity;
      for (const point of initialBase) {
        this._extendBase(point);
      }
    }
    /**
     * Factory method: Computes the Base and Strong Generating Set (BSGS) for a given set of generators.
     * @param {PermutationSet} groupSet - The set of permutations that generate the group.
     * @param {number[]} [initialBase=[]] - An optional array of points to serve as a prefix for the base.
     * @returns {SchreierSimsAlgorithm} A new `SchreierSimsAlgorithm` instance with the computed BSGS.
     * @static
     */
    static compute(groupSet, initialBase = []) {
      const engine = new _SchreierSimsAlgorithm(initialBase);
      const ids = groupSet.indices;
      for (let i = 0; i < ids.length; i++) {
        engine.siftAndInsert(ids[i]);
      }
      return engine;
    }
    // ========================================================================
    // Public API
    // ========================================================================
    /**
     * Gets the current degree (number of points) on which the permutations act.
     * This value is dynamically managed by the underlying `PermutationRepository`.
     * @returns {number} The degree of the permutation group.
     */
    get degree() {
      return this.repo.globalDegree;
    }
    /**
     * Calculates the exact order (size) of the group represented by the BSGS.
     * The order is computed as the product of the sizes of the orbits (transversals) at each level of the base.
     * @returns {bigint} The order of the group as a BigInt, to support very large group orders.
     */
    get order() {
      let size = 1n;
      for (const t of this.transversals) {
        size *= BigInt(t.size);
      }
      return size;
    }
    /**
     * Checks if a given permutation is an element of the group represented by this BSGS.
     * The permutation is "sifted" through the stabilizer chain. If the process
     * reduces the permutation to the identity element, it is in the group.
     * @param {number|Int32Array|Array<number>} perm - The permutation to check, either as an ID or a raw array.
     * @returns {boolean} True if the permutation belongs to the group, false otherwise.
     */
    contains(perm) {
      let permId = perm;
      if (typeof perm !== "number") {
        permId = this.repo.register(perm);
      }
      const { residue } = this._strip(permId);
      return residue === this.idIdentity;
    }
    /**
     * Checks if the group acts transitively on a specified domain.
     * A group is transitive if, for any two points `x` and `y` in the domain,
     * there exists a group element `g` such that `g(x) = y`.
     * This implementation checks if the orbit of the first base point (`base[0]`) under the full group
     * covers the entire `domainSize`.
     * @param {number} domainSize - The size of the domain (e.g., `globalDegree`) to check for transitivity.
     * @returns {boolean} True if the group is transitive on the given domain, false otherwise.
     */
    isTransitive(domainSize) {
      if (domainSize <= 1) return true;
      if (this.base.length === 0) return false;
      return this.transversals[0].size === domainSize;
    }
    /**
     * Computes the stabilizer subgroup G_p of a specific point `p`.
     * G_p is defined as the set of all permutations `g` in the group G such that `g(p) = p`.
     * @param {number} point - The point (0-based index) for which to compute the stabilizer.
     * @returns {PermutationSet} A `PermutationSet` containing the generators of the stabilizer subgroup.
     */
    getStabilizer(point) {
      const stabSSA = new _SchreierSimsAlgorithm([point]);
      const allGens = this.generators.flat();
      for (const gen of allGens) {
        stabSSA.siftAndInsert(gen);
      }
      const stabilizerGens = stabSSA.generators.slice(1).flat();
      return new PermutationSet(stabilizerGens, false, true);
    }
    /**
     * Multiplies two permutations `idA` and `idB`.
     * The convention is `(A * B)(x) = A(B(x))`, meaning `idB` is applied first, then `idA`.
     * This method delegates to the underlying `PermutationRepository` for the actual multiplication.
     * @param {number} idA - The ID of the first permutation (A).
     * @param {number} idB - The ID of the second permutation (B).
     * @returns {number} The ID of the resulting permutation (A * B).
     * @private
     */
    multiply(idA, idB) {
      return this.repo.multiply(idA, idB);
    }
    /**
     * Computes or retrieves the inverse of a given permutation ID.
     * This method delegates to the underlying `PermutationRepository` for inversion.
     * @param {number} id - The ID of the permutation to invert.
     * @returns {number} The ID of the inverse permutation.
     * @private
     */
    inverse(id) {
      return this.repo.inverse(id);
    }
    // ========================================================================
    // Core Logic
    // ========================================================================
    /**
     * The "Strip" (or Sift) procedure is a core component of the Schreier-Sims algorithm.
     * It attempts to reduce a permutation `gId` to the identity by applying elements from the stabilizer chain.
     * At each level `i`, if `gId` moves `base[i]` to `delta`, and `delta` is in the known orbit `transversals[i]`,
     * `gId` is multiplied by the inverse of the representative `u` (where `u(base[i]) = delta`).
     * This process continues until `gId` either becomes the identity or reaches a level where it moves `base[i]`
     * to a point not in the current orbit.
     * @param {number} gId - The permutation ID to be sifted.
     * @returns {{residue: number, level: number}} An object containing:
     *   - `residue`: The permutation ID remaining after sifting (identity if `gId` is in the group).
     *   - `level`: The level in the stabilizer chain where sifting stopped (or `base.length` if sifted to identity).
     * @private
     */
    _strip(gId) {
      let curr = gId;
      const depth = this.base.length;
      const N = this.repo.globalDegree;
      for (let i = 0; i < depth; i++) {
        const beta = this.base[i];
        const offset = curr * N;
        const delta = this.repo.permBuffer[offset + beta];
        if (delta === beta) continue;
        const traversal = this.transversals[i];
        const u = traversal.get(delta);
        if (u !== void 0) {
          const uInv = this.inverse(u);
          curr = this.multiply(uInv, curr);
        } else {
          return { residue: curr, level: i };
        }
      }
      return { residue: curr, level: depth };
    }
    /**
     * The main incremental construction method for the BSGS.
     * It sifts a given permutation `gId`. If `gId` is not already in the group generated by the current BSGS,
     * the algorithm updates the stabilizer chain (`base`, `transversals`, `generators`) to include `gId`,
     * ensuring that the BSGS correctly represents the expanded group.
     * @param {number} gId - The permutation ID to be inserted into the BSGS.
     */
    siftAndInsert(gId) {
      const { residue: h, level } = this._strip(gId);
      if (h === this.idIdentity) return;
      if (level === this.base.length) {
        const movedPoint = this._findFirstMovedPoint(h);
        if (movedPoint === -1) return;
        this._extendBase(movedPoint);
      }
      this._addGeneratorToLevel(level, h);
      for (let i = 0; i < level; i++) {
        this._addGeneratorToLevel(i, h);
      }
    }
    /**
     * Get generators as PermutationSet
     * @returns {PermutationSet}
     */
    getGeneratorsAsPermutationSet() {
      const flatIds = this.generators.flat();
      return new PermutationSet(flatIds, false, false);
    }
    /**
     * Adds a new strong generator `hId` to the `generators` list at the specified `level`
     * and triggers an update of the orbit (`transversal`) at that level.
     * @param {number} level - The level in the stabilizer chain to which the generator is added.
     * @param {number} hId - The ID of the permutation to add as a strong generator.
     * @private
     */
    _addGeneratorToLevel(level, hId) {
      this.generators[level].push(hId);
      this._updateOrbit(level, hId);
    }
    /**
     * Extends the base of the stabilizer chain by adding a new point.
     * This creates a new level in the chain, initializing its generators and transversal.
     * @param {number} point - The new point to be added to the base.
     * @private
     */
    _extendBase(point) {
      this.base.push(point);
      this.generators.push([]);
      const map = /* @__PURE__ */ new Map();
      map.set(point, this.idIdentity);
      this.transversals.push(map);
    }
    /**
     * Updates the orbit (transversal) at a specified `level` of the stabilizer chain.
     * This involves performing a Breadth-First Search (BFS) starting from existing orbit representatives
     * and the new generator `hId`. During the BFS, any newly discovered Schreier generators
     * (elements that stabilize `base[level]` but are not yet in the BSGS for `G^(level+1)`) are sifted and inserted.
     * @param {number} level - The level in the stabilizer chain whose orbit needs updating.
     * @param {number} hId - The ID of a newly added strong generator at this level.
     * @private
     */
    _updateOrbit(level, hId) {
      const transversal = this.transversals[level];
      const queue = [];
      const existingReps = Array.from(transversal.values());
      for (const uRep of existingReps) {
        this._processSchreierEdge(uRep, hId, level, queue);
      }
      let ptr = 0;
      const gens = this.generators[level];
      while (ptr < queue.length) {
        const uRep = queue[ptr++];
        for (let i = 0; i < gens.length; i++) {
          this._processSchreierEdge(uRep, gens[i], level, queue);
        }
      }
    }
    /**
     * Processes a single edge (`uRep` --`sId`--> `cand`) in the Schreier graph during orbit construction.
     * If `cand` maps to a new point in the orbit, it's added to the transversal and BFS queue.
     * If `cand` maps to an already visited point, a Schreier generator (`v^-1 * cand`) is formed
     * and recursively sifted to maintain the BSGS.
     * @param {number} uRep - The permutation ID of the current orbit representative (`u`).
     * @param {number} sId - The permutation ID of the generator (`s`) being applied.
     * @param {number} level - The current level in the stabilizer chain.
     * @param {number[]} queue - The Breadth-First Search work queue, storing permutation IDs.
     * @private
     */
    _processSchreierEdge(uRep, sId, level, queue) {
      const cand = this.multiply(sId, uRep);
      const N = this.repo.globalDegree;
      const beta = this.base[level];
      const offset = cand * N;
      const img = this.repo.permBuffer[offset + beta];
      const transversal = this.transversals[level];
      if (!transversal.has(img)) {
        transversal.set(img, cand);
        queue.push(cand);
      } else {
        const v = transversal.get(img);
        if (v === cand) return;
        const vInv = this.inverse(v);
        const schreierGen = this.multiply(vInv, cand);
        if (schreierGen !== this.idIdentity) {
          this.siftAndInsert(schreierGen);
        }
      }
    }
    // ========================================================================
    // Low-Level Helpers (Direct Memory Access)
    // ========================================================================
    /**
     * Finds the smallest point (index) that is not fixed by the given permutation `permId`.
     * This is typically used to select a new base point when extending the stabilizer chain.
     * @param {number} permId - The ID of the permutation to analyze.
     * @returns {number} The 0-based index of the first point moved by the permutation, or -1 if the permutation is the identity.
     * @private
     */
    _findFirstMovedPoint(permId) {
      const N = this.repo.globalDegree;
      const offset = permId * N;
      const buf = this.repo.permBuffer;
      for (let i = 0; i < N; i++) {
        if (buf[offset + i] !== i) return i;
      }
      return -1;
    }
  };

  // src/group-structural-utils.js
  function _ensureSSA(input) {
    if (input instanceof SchreierSimsAlgorithm) {
      return input;
    }
    if (input instanceof PermutationSet) {
      return SchreierSimsAlgorithm.compute(input);
    }
    throw new Error("Input must be PermutationSet or SchreierSimsAlgorithm");
  }
  function _ensureGens(input) {
    if (input instanceof PermutationSet) {
      return input;
    }
    if (input instanceof SchreierSimsAlgorithm) {
      const flatIds = input.generators.flat();
      return new PermutationSet(flatIds, false, false);
    }
    throw new Error("Invalid Input");
  }
  function isSubgroup(superGroup, subGroup) {
    const superSSA = _ensureSSA(superGroup);
    const subGens = _ensureGens(subGroup);
    for (const h of subGens.indices) {
      if (!superSSA.contains(h)) return false;
    }
    return true;
  }
  function isNormal(superGroup, normalN) {
    const superGens = _ensureGens(superGroup);
    const subSSA = _ensureSSA(normalN);
    const subGens = _ensureGens(normalN);
    for (const g of superGens.indices) {
      const gInv = globalRepo.inverse(g);
      for (const n of subGens.indices) {
        const gn = globalRepo.multiply(g, n);
        const conj = globalRepo.multiply(gn, gInv);
        if (!subSSA.contains(conj)) return false;
      }
    }
    return true;
  }
  function getNormalClosure(groupG, subsetS) {
    const gGens = _ensureGens(groupG).indices;
    let initialIds = [];
    if (subsetS instanceof PermutationSet) initialIds = Array.from(subsetS.indices);
    else if (Array.isArray(subsetS)) initialIds = subsetS;
    else initialIds = [subsetS];
    const closureSSA = new SchreierSimsAlgorithm();
    const queue = [];
    for (const id of initialIds) {
      if (!closureSSA.contains(id)) {
        closureSSA.siftAndInsert(id);
        queue.push(id);
      }
    }
    let head = 0;
    while (head < queue.length) {
      const n = queue[head++];
      for (const g of gGens) {
        const conj = globalRepo.conjugate(g, n);
        if (!closureSSA.contains(conj)) {
          closureSSA.siftAndInsert(conj);
          queue.push(conj);
        }
      }
    }
    return closureSSA;
  }
  function getCommutatorSubgroup(group) {
    const gens = _ensureGens(group).indices;
    const commutators = [];
    const len = gens.length;
    for (let i = 0; i < len; i++) {
      for (let j = i + 1; j < len; j++) {
        const comm = globalRepo.commutator(gens[i], gens[j]);
        if (comm !== globalRepo.identity) {
          commutators.push(comm);
        }
      }
    }
    if (commutators.length === 0) {
      return SchreierSimsAlgorithm.compute(PermutationSet.identity());
    }
    return getNormalClosure(group, commutators);
  }
  function isSolvable(group) {
    let currentSSA = _ensureSSA(group);
    const limit = 20;
    for (let i = 0; i < limit; i++) {
      if (currentSSA.order === 1n) return true;
      const nextSSA = getCommutatorSubgroup(currentSSA);
      if (nextSSA.order === currentSSA.order) return false;
      currentSSA = nextSSA;
    }
    return false;
  }
  function isSimple(group, randomTests = 10) {
    const ssa = _ensureSSA(group);
    const order = ssa.order;
    if (order === 1n) return 0;
    const gens = _ensureGens(ssa);
    if (gens.isAbelian()) {
      if (order < 9007199254740991n) {
        const n = Number(order);
        if (_isSmallPrime(n)) return 1;
        return 0;
      }
      return -1;
    }
    const derivedSSA = getCommutatorSubgroup(ssa);
    if (derivedSSA.order !== order) {
      return 0;
    }
    for (const g of gens.indices) {
      if (g === globalRepo.identity) continue;
      const nc = getNormalClosure(ssa, [g]);
      if (nc.order !== order) return 0;
    }
    for (let i = 0; i < randomTests; i++) {
      const rnd = _getRandomElement(ssa);
      if (rnd === globalRepo.identity) continue;
      const nc = getNormalClosure(ssa, [rnd]);
      if (nc.order !== order) return 0;
    }
    return -1;
  }
  var QuotientGroupMap = class {
    /**
     * @param {PermutationSet} quotientGroup - A PermutationSet whose elements act on the coset indices (0-based).
     * @param {Int32Array} representatives - An Int32Array where `representatives[i]` is a chosen representative from the i-th coset.
     * @param {bigint} quotientOrder - The order of the quotient group, |G/N|.
     */
    constructor(quotientGroup, representatives, quotientOrder) {
      this.group = quotientGroup;
      this.representatives = representatives;
      this.size = quotientOrder;
    }
    /**
     * Lifts a quotient group element (represented by a permutation ID) back to
     * a specific representative element in the original group G.
     * The returned element `g` is such that the quotient element corresponds to the coset `Ng`.
     * @param {number} quotientPermId - The ID of the permutation in the quotient group.
     * @returns {number} The ID of a representative element in the original group G.
     * @throws {Error} If the `quotientPermId` maps to an invalid coset index.
     */
    lift(quotientPermId) {
      const qPerm = globalRepo.get(quotientPermId);
      const targetCosetIdx = qPerm[0];
      if (targetCosetIdx < 0 || targetCosetIdx >= this.representatives.length) {
        throw new Error("Invalid Quotient Permutation");
      }
      return this.representatives[targetCosetIdx];
    }
  };
  function getQuotientStructure(groupG, normalN, maxIndex = 2e3) {
    const ssaG = _ensureSSA(groupG);
    const ssaN = _ensureSSA(normalN);
    if (ssaG.order % ssaN.order !== 0n) throw new Error("N must divide G");
    const indexBig = ssaG.order / ssaN.order;
    if (indexBig > BigInt(maxIndex)) {
      throw new Error(`Quotient index ${indexBig} too large for explicit construction.`);
    }
    const k = Number(indexBig);
    const cosetReps = [globalRepo.identity];
    const gGens = _ensureGens(ssaG).indices;
    let head = 0;
    while (head < cosetReps.length) {
      const currRep = cosetReps[head];
      for (const gen of gGens) {
        const candidate = globalRepo.multiply(currRep, gen);
        let found = false;
        for (let i = 0; i < cosetReps.length; i++) {
          const existing = cosetReps[i];
          const exInv = globalRepo.inverse(existing);
          const check = globalRepo.multiply(candidate, exInv);
          if (ssaN.contains(check)) {
            found = true;
            break;
          }
        }
        if (!found) {
          if (cosetReps.length >= k) throw new Error("Coset Enumeration Overflow");
          cosetReps.push(candidate);
        }
      }
      head++;
    }
    const quotientGenIds = [];
    const tempArr = new Int32Array(k);
    for (const gen of gGens) {
      for (let c = 0; c < k; c++) {
        const rep = cosetReps[c];
        const result = globalRepo.multiply(rep, gen);
        let targetIdx = -1;
        for (let i = 0; i < k; i++) {
          const existing = cosetReps[i];
          const exInv = globalRepo.inverse(existing);
          const check = globalRepo.multiply(result, exInv);
          if (ssaN.contains(check)) {
            targetIdx = i;
            break;
          }
        }
        if (targetIdx === -1) throw new Error("Coset Closure Error");
        tempArr[c] = targetIdx;
      }
      quotientGenIds.push(globalRepo.register(tempArr));
    }
    const genSet = new PermutationSet(quotientGenIds, false, false);
    const qGroup = generateGroup(genSet);
    return new QuotientGroupMap(qGroup, new Int32Array(cosetReps), indexBig);
  }
  function areIsomorphic(groupA, groupB) {
    const ssaA = _ensureSSA(groupA);
    const ssaB = _ensureSSA(groupB);
    if (ssaA.order !== ssaB.order) return 0;
    const gensA = _ensureGens(ssaA);
    const gensB = _ensureGens(ssaB);
    const abA = gensA.isAbelian();
    const abB = gensB.isAbelian();
    if (abA !== abB) return 0;
    const commA = getCommutatorSubgroup(ssaA);
    const commB = getCommutatorSubgroup(ssaB);
    if (commA.order !== commB.order) return 0;
    if (commA.order > 1n) {
      const comm2A = getCommutatorSubgroup(commA);
      const comm2B = getCommutatorSubgroup(commB);
      if (comm2A.order !== comm2B.order) return 0;
    }
    return -1;
  }
  function getMixedCommutatorSubgroup(groupG, subA, subB) {
    const gensA = _ensureGens(subA).indices;
    const gensB = _ensureGens(subB).indices;
    const commutators = [];
    for (let i = 0; i < gensA.length; i++) {
      for (let j = 0; j < gensB.length; j++) {
        const comm = globalRepo.commutator(gensA[i], gensB[j]);
        if (comm !== globalRepo.identity) {
          commutators.push(comm);
        }
      }
    }
    if (commutators.length === 0) {
      return SchreierSimsAlgorithm.compute(PermutationSet.identity());
    }
    return getNormalClosure(groupG, commutators);
  }
  function getLowerCentralSeries(group) {
    const ssaG = _ensureSSA(group);
    const series = [ssaG];
    let current = ssaG;
    const limit = 20;
    for (let i = 0; i < limit; i++) {
      if (current.order === 1n) break;
      const next = getMixedCommutatorSubgroup(ssaG, current, ssaG);
      if (next.order === current.order) {
        series.push(next);
        break;
      }
      series.push(next);
      current = next;
    }
    return series;
  }
  function isNilpotent(group) {
    const ssa = _ensureSSA(group);
    if (!isSolvable(ssa)) return 0;
    if (ssa.order === 1n) return 1;
    const series = getLowerCentralSeries(ssa);
    const last = series[series.length - 1];
    return last.order === 1n ? 1 : 0;
  }
  function analyzeGenerators(candidateIds) {
    const ssa = new SchreierSimsAlgorithm();
    const fundamental = [];
    const redundant = [];
    for (const id of candidateIds) {
      if (id === globalRepo.identity) {
        redundant.push(id);
        continue;
      }
      if (ssa.contains(id)) {
        redundant.push(id);
      } else {
        ssa.siftAndInsert(id);
        fundamental.push(id);
      }
    }
    return { fundamental, redundant, ssa };
  }
  var MAX_TRIALS = 100;
  var MAX_RESTARTS = 10;
  function getSylowSubgroup(group, p) {
    const ssa = _ensureSSA(group);
    const order = ssa.order;
    const pBig = BigInt(p);
    let tempOrder = order;
    let targetOrder = 1n;
    while (tempOrder % pBig === 0n) {
      targetOrder *= pBig;
      tempOrder /= pBig;
    }
    if (targetOrder === 1n) {
      return PermutationSet.identity(globalRepo.globalDegree);
    }
    for (let restart = 0; restart < MAX_RESTARTS; restart++) {
      let currentGroup = PermutationSet.identity(globalRepo.globalDegree);
      let currentOrder = 1n;
      let failures = 0;
      while (failures < MAX_TRIALS) {
        if (currentOrder === targetOrder) {
          return currentGroup;
        }
        const g = _getRandomElement(ssa);
        const h = _getPPart(g, p);
        if (h === globalRepo.identity) {
          failures++;
          continue;
        }
        const newGens = currentGroup.union(
          new PermutationSet([h], true, false)
        );
        const newSSA = SchreierSimsAlgorithm.compute(newGens);
        const newOrder = newSSA.order;
        if (_isPowerOfP(newOrder, pBig)) {
          if (newOrder > currentOrder) {
            currentGroup = newGens;
            currentOrder = newOrder;
            failures = 0;
            continue;
          }
        }
        failures++;
      }
    }
    throw new Error(`Failed to construct Sylow ${p}-subgroup. (Random search exhausted).`);
  }
  function _isSmallPrime(n) {
    if (n <= 1) return false;
    if (n <= 3) return true;
    if (n % 2 === 0 || n % 3 === 0) return false;
    for (let i = 5; i * i <= n; i += 6) {
      if (n % i === 0 || n % (i + 2) === 0) return false;
    }
    return true;
  }
  function _getRandomElement(ssa) {
    let result = ssa.repo.identity;
    const depth = ssa.base.length;
    for (let i = 0; i < depth; i++) {
      const transversal = ssa.transversals[i];
      const size = transversal.size;
      const randIdx = Math.floor(Math.random() * size);
      let k = 0;
      for (const permId of transversal.values()) {
        if (k === randIdx) {
          result = ssa.repo.multiply(result, permId);
          break;
        }
        k++;
      }
    }
    return result;
  }
  function _pow(gId, exp) {
    if (exp === 0) return globalRepo.identity;
    if (exp === 1) return gId;
    let base = gId;
    let result = globalRepo.identity;
    let e = exp;
    while (e > 0) {
      if (e % 2 === 1) {
        result = globalRepo.multiply(result, base);
      }
      base = globalRepo.multiply(base, base);
      e = Math.floor(e / 2);
    }
    return result;
  }
  function _lcm(a, b) {
    if (a === 0 || b === 0) return 0;
    return Math.abs(a * b / _gcd(a, b));
  }
  function _gcd(a, b) {
    while (b !== 0) {
      let t = b;
      b = a % b;
      a = t;
    }
    return a;
  }
  function _isPowerOfP(n, pBig) {
    let val = n;
    while (val > 1n) {
      if (val % pBig !== 0n) return false;
      val /= pBig;
    }
    return true;
  }
  function _getPPart(gId, p) {
    const perm = globalRepo.get(gId);
    const n = perm.length;
    const visited = new Uint8Array(n);
    let order = 1;
    for (let i = 0; i < n; i++) {
      if (visited[i]) continue;
      let curr = i;
      let len = 0;
      while (!visited[curr]) {
        visited[curr] = 1;
        curr = perm[curr];
        len++;
      }
      if (len > 1) {
        order = _lcm(order, len);
      }
    }
    if (order === 1) return globalRepo.identity;
    let m = order;
    while (m % p === 0) {
      m /= p;
    }
    return _pow(gId, m);
  }

  // src/group-visualizer.js
  function generateNames(allElementIds, generatorIds, genLabels = void 0) {
    if (analyzeGenerators(generatorIds).redundant?.length > 0) {
      throw new Error("generateNames analyzeGenerators(generatorIds).redundant?.length>0");
    }
    const nameMap = /* @__PURE__ */ new Map();
    const visited = /* @__PURE__ */ new Set();
    if (genLabels) {
      if (genLabels.length < generatorIds.length) {
        throw new Error("genLabels.length<generatorIds.length");
      }
    } else {
      genLabels = generatorIds.map((id, idx) => {
        return idx < 26 ? String.fromCharCode(97 + idx) : `g${idx + 1}`;
      });
    }
    const genMap = /* @__PURE__ */ new Map();
    generatorIds.forEach((id, i) => genMap.set(id, genLabels[i]));
    const queue = [];
    const idIdentity = globalRepo.identity;
    nameMap.set(idIdentity, "e");
    visited.add(idIdentity);
    for (let i = 0; i < generatorIds.length; i++) {
      const genId = generatorIds[i];
      const label = genLabels[i];
      if (genId === idIdentity) continue;
      nameMap.set(genId, label);
      visited.add(genId);
      queue.push({ id: genId, name: label, lastGen: label, power: 1 });
    }
    let head = 0;
    while (head < queue.length) {
      const curr = queue[head++];
      for (let i = 0; i < generatorIds.length; i++) {
        const genId = generatorIds[i];
        const genLabel = genLabels[i];
        const nextId = globalRepo.multiply(curr.id, genId);
        if (!visited.has(nextId)) {
          visited.add(nextId);
          let nextName = "";
          let nextPower = 1;
          if (curr.lastGen === genLabel) {
            const baseName = curr.power > 1 ? curr.name.substring(0, curr.name.lastIndexOf("^")) : curr.name;
            nextPower = curr.power + 1;
            nextName = `${baseName}^${nextPower}`;
          } else {
            nextName = `${curr.name}${genLabel}`;
            nextPower = 1;
          }
          nameMap.set(nextId, nextName);
          queue.push({
            id: nextId,
            name: nextName,
            lastGen: genLabel,
            power: nextPower
          });
        }
      }
    }
    for (const id of allElementIds) {
      if (!nameMap.has(id)) {
        nameMap.set(id, decomposeToCycles(id));
      }
    }
    return nameMap;
  }
  function generateMultiplicationTable(inputIds, nameMap = null) {
    let tableElements;
    let finalNames = nameMap;
    const analysis = analyzeGenerators(inputIds);
    const fundamentalGens = analysis.fundamental;
    tableElements = Array.from(generateGroup(fundamentalGens).indices);
    if (!finalNames) {
      finalNames = generateNames(tableElements, fundamentalGens);
    } else {
      for (const id of tableElements) {
        if (!finalNames.has(id)) {
          throw new Error(`Insufficient names: Element ID ${id} is missing from nameMap.`);
        }
      }
    }
    const size = tableElements.length;
    const matrix = new Array(size);
    const grid = new Array(size);
    const cycleMap = /* @__PURE__ */ new Map();
    for (const id of tableElements) {
      if (!cycleMap.has(id)) {
        cycleMap.set(id, globalRepo.getAsCycles(id));
      }
    }
    for (let r = 0; r < size; r++) {
      matrix[r] = new Int32Array(size);
      grid[r] = new Array(size);
      const rowId = tableElements[r];
      for (let c = 0; c < size; c++) {
        const colId = tableElements[c];
        const resId = globalRepo.multiply(rowId, colId);
        matrix[r][c] = resId;
        grid[r][c] = finalNames.get(resId);
        if (!cycleMap.has(resId)) {
          cycleMap.set(resId, globalRepo.getAsCycles(resId));
        }
      }
    }
    const html = _renderHtmlTable(tableElements, grid, matrix, finalNames, cycleMap);
    return { matrix, grid, cycleMap, html, nameMap: finalNames };
  }
  function _renderHtmlTable(elements, grid, matrix, nameMap, cycleMap) {
    const size = elements.length;
    const idIdentity = globalRepo.identity;
    const colorMap = /* @__PURE__ */ new Map();
    colorMap.set(idIdentity, "#ffffff");
    for (let i = 0; i < size; i++) {
      const id = elements[i];
      if (id === idIdentity) continue;
      const hue = Math.floor(i / size * 360);
      colorMap.set(id, `hsl(${hue}, 80%, 85%)`);
    }
    const formatName = (name) => name.replace(/\^(\d+)/g, "<sup>$1</sup>");
    const tableStyle = "border-collapse: collapse; text-align: center; font-family: sans-serif; cursor: default;";
    const cellStyleBase = "padding: 8px; border: 1px solid #ccc; min-width: 30px;";
    let html = `<table class="cayley-table" style="${tableStyle}">
`;
    html += "  <thead>\n    <tr>\n";
    html += `      <th class="cayley-corner" style="${cellStyleBase} background-color: #f0f0f0;">\xD7</th>
`;
    for (let i = 0; i < size; i++) {
      const id = elements[i];
      const name = formatName(nameMap.get(id));
      const cycles = cycleMap.get(id);
      const bg = colorMap.get(id);
      html += `      <th class="cayley-header" title="${cycles}" style="${cellStyleBase} background-color: ${bg};">${name}</th>
`;
    }
    html += "    </tr>\n  </thead>\n";
    html += "  <tbody>\n";
    for (let r = 0; r < size; r++) {
      const rowId = elements[r];
      const rowName = formatName(nameMap.get(rowId));
      const rowCycles = cycleMap.get(rowId);
      const rowBg = colorMap.get(rowId);
      html += "    <tr>\n";
      html += `      <th class="cayley-header" title="${rowCycles}" style="${cellStyleBase} background-color: ${rowBg};">${rowName}</th>
`;
      for (let c = 0; c < size; c++) {
        const rawName = grid[r][c];
        const valName = formatName(rawName);
        const valId = matrix[r][c];
        const valCycles = cycleMap.get(valId);
        const cellBg = colorMap.get(valId);
        const isIdentity = valId === idIdentity;
        const cellClass = isIdentity ? "cayley-cell cayley-identity" : "cayley-cell";
        html += `      <td class="${cellClass}" title="${valCycles}" style="${cellStyleBase} background-color: ${cellBg};">${valName}</td>
`;
      }
      html += "    </tr>\n";
    }
    html += "  </tbody>\n</table>";
    return html;
  }

  // src/group-visualizer-cayley-graph.js
  var _DefaultCayleyGraphConfig = {
    d0: 50,
    // Base distance factor (d0 / order)
    repulsion: 300,
    // Coulomb repulsion
    edgeStrength: 1,
    // Hooke's law spring constant
    chordStrength: 0.5,
    // Geometry maintaining chord strength
    planarStrength: 0.3,
    // Force to flatten cycles onto a plane
    convexityStrength: 0.2,
    // Force pushing nodes away from cycle center (keep convex)
    initialOffsetDist: 5,
    // Magnitude of random offset per cycle group
    decay: 0.9,
    // Velocity damping
    centerPull: 0.015,
    // Gravity to origin
    timeStep: 0.3,
    jitterMax: 10,
    // Max random displacement during annealing start
    dynamicAngleUpdateRate: 1,
    // Recalculate average angles every N ticks
    warmupRuns: 2e3,
    advancedMode: false,
    nameMap: void 0
  };
  function generateCayleyGraphForPlotly(inputIds, customConfig = {}) {
    if (inputIds instanceof PermutationSet) {
      inputIds = Array.from(inputIds.indices);
    }
    const config = { ..._DefaultCayleyGraphConfig, ...customConfig };
    const { fundamental } = analyzeGenerators(inputIds);
    const generators = fundamental;
    if (generators.length === 0) {
      throw new Error("No generators provided.");
    }
    const allElements = Array.from(generateGroup(generators).indices);
    let nameMap;
    if (customConfig.nameMap) {
      nameMap = customConfig.nameMap;
      for (const id of allElements) {
        if (!nameMap.has(id)) {
          throw new Error(`Insufficient names: Element ID ${id} is missing from nameMap.`);
        }
      }
    } else {
      nameMap = generateNames(allElements, generators);
    }
    const colors = [
      "#000000",
      //  (Black)
      "#FF0000",
      //  (Red)
      "#00FF00",
      //  (Lime)
      "#0000FF",
      //  (Blue)
      "#FFFF00",
      //  (Yellow)
      "#00FFFF",
      //  (Cyan)
      "#FF00FF",
      //  (Magenta)
      "#800000",
      //  (Maroon)
      "#008000",
      //  (Green)
      "#000080",
      //  (Navy)
      "#808000",
      //  (Olive)
      "#800080",
      //  (Purple)
      "#008080",
      //  (Teal)
      "#FFA500"
      //  (Orange)
    ];
    let usedColor = [];
    const genMeta = /* @__PURE__ */ new Map();
    generators.forEach((genId) => {
      let order = 1;
      let curr = genId;
      const idIdentity = globalRepo.identity;
      while (curr !== idIdentity && order < 2e3) {
        curr = globalRepo.multiply(curr, genId);
        order++;
      }
      let cIdx = genId * 1597 % colors.length;
      while (usedColor[cIdx] && generators.length <= colors.length) {
        cIdx++;
      }
      usedColor[cIdx] = 1;
      genMeta.set(genId, {
        id: genId,
        label: nameMap.get(genId),
        color: colors[cIdx],
        order,
        isDirected: order > 2
      });
    });
    const nodeObjMap = /* @__PURE__ */ new Map();
    const nodes = allElements.map((id) => {
      const node = {
        id,
        name: nameMap.get(id),
        x: (Math.random() - 0.5) * 2,
        y: (Math.random() - 0.5) * 2,
        z: (Math.random() - 0.5) * 2,
        vx: 0,
        vy: 0,
        vz: 0
      };
      nodeObjMap.set(id, node);
      return node;
    });
    const links = [];
    const physicsConstraints = {
      edges: [],
      chords: [],
      cycles: [],
      angleTriplets: []
      // Stores {genId, center, prev, next} for measuring angles
    };
    const addedPhysicsEdges = /* @__PURE__ */ new Set();
    const getEdgeKey = (a, b) => a < b ? `${a}:${b}` : `${b}:${a}`;
    const cycleVisited = /* @__PURE__ */ new Map();
    generators.forEach((g) => cycleVisited.set(g, /* @__PURE__ */ new Set()));
    for (const elemId of allElements) {
      for (const genId of generators) {
        const targetId = globalRepo.multiply(elemId, genId);
        const meta = genMeta.get(genId);
        links.push({
          source: elemId,
          target: targetId,
          genId,
          color: meta.color,
          order: meta.order,
          isDirected: meta.isDirected
        });
        const edgeKey = getEdgeKey(elemId, targetId);
        const targetDist = config.d0 / meta.order;
        if (elemId !== targetId && !addedPhysicsEdges.has(edgeKey)) {
          addedPhysicsEdges.add(edgeKey);
          physicsConstraints.edges.push({
            source: elemId,
            target: targetId,
            dist: targetDist,
            strength: config.edgeStrength
          });
        }
        if (meta.order > 2) {
          const nextTargetId = globalRepo.multiply(targetId, genId);
          const prevId = globalRepo.multiply(elemId, globalRepo.inverse(genId));
          physicsConstraints.angleTriplets.push({
            genId,
            center: elemId,
            prev: prevId,
            // node * g^-1
            next: targetId
            // node * g
          });
          if (meta.order > 3 && elemId !== nextTargetId) {
            const chordKey = getEdgeKey(elemId, nextTargetId);
            const theta = (meta.order - 2) * Math.PI / meta.order;
            const chordLen = Math.sqrt(
              2 * (targetDist * targetDist) - 2 * (targetDist * targetDist) * Math.cos(theta)
            );
            if (!addedPhysicsEdges.has(chordKey)) {
              physicsConstraints.chords.push({
                source: elemId,
                target: nextTargetId,
                dist: chordLen,
                strength: config.chordStrength
              });
            }
          }
        }
        if (meta.order > 1) {
          const visitedSet = cycleVisited.get(genId);
          if (!visitedSet.has(elemId)) {
            const cycleIndices = [];
            let currTrace = elemId;
            for (let k = 0; k < meta.order; k++) {
              visitedSet.add(currTrace);
              cycleIndices.push(currTrace);
              currTrace = globalRepo.multiply(currTrace, genId);
            }
            if (meta.order > 2) {
              physicsConstraints.cycles.push({
                indices: cycleIndices,
                genId
              });
            }
            let os = (meta.order + 1) * (meta.order + 1);
            const dx = (Math.random() - 0.5) * config.initialOffsetDist * os;
            const dy = (Math.random() - 0.5) * config.initialOffsetDist * os;
            const dz = (Math.random() - 0.5) * config.initialOffsetDist * os;
            cycleIndices.forEach((nodeId) => {
              const n = nodeObjMap.get(nodeId);
              if (n) {
                n.x += dx;
                n.y += dy;
                n.z += dz;
              }
            });
          }
        }
      }
    }
    const legend = Array.from(genMeta.values()).map((m) => ({
      label: `${m.label} (Order ${m.order})`,
      color: m.color,
      genId: m.id
    }));
    const simulator = new VisualizerCayleyForceSimulator(nodes, physicsConstraints, generators, genMeta, config);
    if (config.warmupRuns > 0) {
      simulator.warmup(config.warmupRuns);
    }
    if (config.advancedMode) {
      return { nodes, links, legend, simulator, config, nameMap };
    } else {
      return { nameMap, ...simulator.getPlotlyFrame() };
    }
  }
  var VisualizerCayleyForceSimulator = class {
    /**
     * @param {Array<object>} nodes - An array of node objects, each with 'id', 'x', 'y', 'z', 'vx', 'vy', 'vz' properties.
     * @param {object} constraints - An object containing arrays of 'edges', 'chords', 'cycles', and 'angleTriplets'.
     * @param {number[]} generators - An array of generator IDs.
     * @param {Map<number, object>} genMeta - A map from generator ID to its metadata (e.g., color, order).
     * @param {_CayleyGraphConfig} config - The physics configuration for the simulator.
     */
    constructor(nodes, constraints, generators, genMeta, config) {
      this.nodes = nodes;
      this.edges = constraints.edges;
      this.chords = constraints.chords;
      this.cycles = constraints.cycles;
      this.angleTriplets = constraints.angleTriplets;
      this.generators = generators;
      this.genMeta = genMeta;
      this.config = config;
      this.nodeMap = /* @__PURE__ */ new Map();
      this.nodes.forEach((n) => this.nodeMap.set(n.id, n));
    }
    /**
     * Executes one step of the physics simulation.
     * Applies forces, integrates velocities, and updates node positions.
     * @param {number} [jitterFactor=0] - The magnitude of random noise to apply to node positions, used for simulated annealing during warmup.
     */
    tick(jitterFactor = 0) {
      const nCount = this.nodes.length;
      const cfg = this.config;
      if (this.config.dynamicAngleUpdateRate > 0 && this.tickCount % this.config.dynamicAngleUpdateRate == 0) {
        this._updateChordTargets();
      }
      for (let i = 0; i < nCount; i++) {
        const n = this.nodes[i];
        n.fx = -n.x * cfg.centerPull;
        n.fy = -n.y * cfg.centerPull;
        n.fz = -n.z * cfg.centerPull;
        if (jitterFactor > 0) {
          n.x += (Math.random() - 0.5) * jitterFactor;
          n.y += (Math.random() - 0.5) * jitterFactor;
          n.z += (Math.random() - 0.5) * jitterFactor;
        }
      }
      for (let i = 0; i < nCount; i++) {
        const n1 = this.nodes[i];
        for (let j = i + 1; j < nCount; j++) {
          const n2 = this.nodes[j];
          const dx = n1.x - n2.x;
          const dy = n1.y - n2.y;
          const dz = n1.z - n2.z;
          let distSq = dx * dx + dy * dy + dz * dz;
          if (distSq < 0.1) distSq = 0.1;
          const force = cfg.repulsion / distSq;
          const dist = Math.sqrt(distSq);
          const fx = dx / dist * force;
          const fy = dy / dist * force;
          const fz = dz / dist * force;
          n1.fx += fx;
          n1.fy += fy;
          n1.fz += fz;
          n2.fx -= fx;
          n2.fy -= fy;
          n2.fz -= fz;
        }
      }
      this._applySprings(this.edges);
      this._applySprings(this.chords);
      this._applyCycleForces();
      for (let i = 0; i < nCount; i++) {
        const n = this.nodes[i];
        n.vx = (n.vx + n.fx * cfg.timeStep) * cfg.decay;
        n.vy = (n.vy + n.fy * cfg.timeStep) * cfg.decay;
        n.vz = (n.vz + n.fz * cfg.timeStep) * cfg.decay;
        n.x += n.vx * cfg.timeStep;
        n.y += n.vy * cfg.timeStep;
        n.z += n.vz * cfg.timeStep;
      }
    }
    /**
     * Dynamically adjusts the ideal lengths of chord constraints based on the average observed angles within each cycle.
     * This helps maintain the geometric integrity of cycles as the graph settles.
     * @private
     */
    _updateChordTargets() {
      const stats = /* @__PURE__ */ new Map();
      for (const tri of this.angleTriplets) {
        const center = this.nodeMap.get(tri.center);
        const prev = this.nodeMap.get(tri.prev);
        const next = this.nodeMap.get(tri.next);
        if (!center || !prev || !next) continue;
        const ux = prev.x - center.x, uy = prev.y - center.y, uz = prev.z - center.z;
        const vx = next.x - center.x, vy = next.y - center.y, vz = next.z - center.z;
        const dot = ux * vx + uy * vy + uz * vz;
        const magU = Math.sqrt(ux * ux + uy * uy + uz * uz);
        const magV = Math.sqrt(vx * vx + vy * vy + vz * vz);
        if (magU > 1e-4 && magV > 1e-4) {
          let cosTheta = dot / (magU * magV);
          if (cosTheta > 1) cosTheta = 1;
          if (cosTheta < -1) cosTheta = -1;
          const angle = Math.acos(cosTheta);
          if (!stats.has(tri.genId)) stats.set(tri.genId, { sum: 0, count: 0 });
          const entry = stats.get(tri.genId);
          entry.sum += angle;
          entry.count++;
        }
      }
      const avgAngles = /* @__PURE__ */ new Map();
      for (const [genId, data] of stats.entries()) {
        if (data.count > 0) {
          avgAngles.set(genId, data.sum / data.count);
        }
      }
      for (const chord of this.chords) {
        if (avgAngles.has(chord.genId)) {
          const thetaAvg = avgAngles.get(chord.genId);
          const meta = this.genMeta.get(chord.genId);
          const idealEdgeLen = meta.targetEdgeLength;
          const newDist = Math.sqrt(
            2 * (idealEdgeLen * idealEdgeLen) * (1 - Math.cos(thetaAvg))
          );
          chord.dist = newDist;
        }
      }
    }
    /**
     * Applies Hooke's Law (spring forces) to a list of links (edges or chords).
     * @param {Array<object>} list - An array of link objects, each with 'source', 'target', 'dist', 'strength' properties.
     * @private
     */
    _applySprings(list) {
      for (const link of list) {
        const n1 = this.nodeMap.get(link.source);
        const n2 = this.nodeMap.get(link.target);
        if (!n1 || !n2) continue;
        const dx = n2.x - n1.x;
        const dy = n2.y - n1.y;
        const dz = n2.z - n1.z;
        const dist = Math.sqrt(dx * dx + dy * dy + dz * dz) || 1e-3;
        const displacement = dist - link.dist;
        const forceMag = link.strength * displacement;
        const fx = dx / dist * forceMag;
        const fy = dy / dist * forceMag;
        const fz = dz / dist * forceMag;
        n1.fx += fx;
        n1.fy += fy;
        n1.fz += fz;
        n2.fx -= fx;
        n2.fy -= fy;
        n2.fz -= fz;
      }
    }
    /**
     * Applies forces to cycle nodes to encourage planar and convex arrangements.
     * Calculates a centroid and normal vector for each cycle to guide these forces.
     * @private
     */
    _applyCycleForces() {
      const planarStr = this.config.planarStrength;
      const convexStr = this.config.convexityStrength;
      for (const cycle of this.cycles) {
        const indices = cycle.indices;
        const len = indices.length;
        if (len < 3) continue;
        let cx = 0, cy = 0, cz = 0;
        const points = [];
        for (const id of indices) {
          const n = this.nodeMap.get(id);
          points.push(n);
          cx += n.x;
          cy += n.y;
          cz += n.z;
        }
        cx /= len;
        cy /= len;
        cz /= len;
        let nx = 0, ny = 0, nz = 0;
        for (let i = 0; i < len; i++) {
          const p1 = points[i];
          const p2 = points[(i + 1) % len];
          const ax = p1.x - cx;
          const ay = p1.y - cy;
          const az = p1.z - cz;
          const bx = p2.x - cx;
          const by = p2.y - cy;
          const bz = p2.z - cz;
          nx += ay * bz - az * by;
          ny += az * bx - ax * bz;
          nz += ax * by - ay * bx;
        }
        const normLen = Math.sqrt(nx * nx + ny * ny + nz * nz);
        if (normLen > 1e-6) {
          nx /= normLen;
          ny /= normLen;
          nz /= normLen;
          for (const p of points) {
            const vx = p.x - cx;
            const vy = p.y - cy;
            const vz = p.z - cz;
            const distToPlane = vx * nx + vy * ny + vz * nz;
            const fFlat = -distToPlane * planarStr;
            p.fx += nx * fFlat;
            p.fy += ny * fFlat;
            p.fz += nz * fFlat;
            const distToCenter = Math.sqrt(vx * vx + vy * vy + vz * vz) || 0.1;
            const fConvex = convexStr;
            p.fx += vx / distToCenter * fConvex;
            p.fy += vy / distToCenter * fConvex;
            p.fz += vz / distToCenter * fConvex;
          }
        }
      }
    }
    /**
     * Runs the simulation for a specified number of iterations with simulated annealing.
     * The `jitterFactor` decays linearly during the first 90% of iterations, then remains at 0 for the last 10%.
     * @param {number} [iterations=2000] - The total number of simulation ticks to run during the warmup phase.
     */
    warmup(iterations = 2e3) {
      const startJitter = this.config.jitterMax;
      for (let i = 0; i < iterations * 9 / 10; i++) {
        const progress = i / iterations;
        const currentJitter = startJitter * (1 - progress);
        this.tick(currentJitter);
      }
      for (let i = 0; i < iterations * 1 / 10; i++) {
        this.tick(0);
      }
    }
    /**
     * Generates a Plotly-compatible data frame (traces and layout) representing the current state of the Cayley graph.
     * Includes 3D scatter plots for nodes, lines for edges, and cones for directed edges.
     * @returns {{data: Array<object>, layout: object}} An object containing Plotly trace data and layout configuration.
     */
    getPlotlyFrame() {
      const traces = [];
      this.generators.forEach((genId) => {
        const meta = this.genMeta.get(genId);
        const x = [], y = [], z = [];
        const cx = [], cy = [], cz = [];
        const cu = [], cv = [], cw = [];
        this.nodes.forEach((node) => {
          const targetId = globalRepo.multiply(node.id, genId);
          const targetNode = this.nodeMap.get(targetId);
          if (targetNode) {
            x.push(node.x, targetNode.x, null);
            y.push(node.y, targetNode.y, null);
            z.push(node.z, targetNode.z, null);
            if (meta.isDirected) {
              const ratio = 0.9;
              const mx = node.x + (targetNode.x - node.x) * ratio;
              const my = node.y + (targetNode.y - node.y) * ratio;
              const mz = node.z + (targetNode.z - node.z) * ratio;
              const dx = targetNode.x - node.x;
              const dy = targetNode.y - node.y;
              const dz = targetNode.z - node.z;
              cx.push(mx);
              cy.push(my);
              cz.push(mz);
              cu.push(dx);
              cv.push(dy);
              cw.push(dz);
            }
          }
        });
        if (x.length > 0) {
          traces.push({
            type: "scatter3d",
            mode: "lines",
            name: meta.isDirected ? `Generator ${meta.label} (${meta.order})` : `Generator ${meta.label} (${meta.order})`,
            x,
            y,
            z,
            line: { color: meta.color, width: 4 },
            hoverinfo: "none"
          });
        }
        if (cx.length > 0) {
          traces.push({
            type: "cone",
            name: `Arrows ${meta.label}`,
            x: cx,
            y: cy,
            z: cz,
            u: cu,
            v: cv,
            w: cw,
            sizemode: "absolute",
            sizeref: 2,
            // Configurable size scale
            anchor: "center",
            showscale: false,
            colorscale: [[0, meta.color], [1, meta.color]],
            hoverinfo: "none"
          });
        }
      });
      const xn = [], yn = [], zn = [], text = [], color = [];
      this.nodes.forEach((n) => {
        xn.push(n.x);
        yn.push(n.y);
        zn.push(n.z);
        text.push(n.name);
        color.push(n.id === globalRepo.identity ? "#000000" : "#888888");
      });
      traces.push({
        type: "scatter3d",
        mode: "markers",
        name: "Elements",
        x: xn,
        y: yn,
        z: zn,
        text,
        marker: {
          size: 5,
          color,
          opacity: 0.9,
          line: { color: "#ffffff", width: 1 }
        },
        hoverinfo: "text"
      });
      const layout = {
        margin: { l: 0, r: 0, b: 0, t: 0 },
        showlegend: true,
        legend: { x: 0, y: 1 },
        scene: {
          xaxis: { showgrid: false, zeroline: false, showticklabels: false, title: "" },
          yaxis: { showgrid: false, zeroline: false, showticklabels: false, title: "" },
          zaxis: { showgrid: false, zeroline: false, showticklabels: false, title: "" },
          bgcolor: "rgba(0,0,0,0)"
        }
      };
      return { data: traces, layout };
    }
  };
  return __toCommonJS(groups_exports);
})();
