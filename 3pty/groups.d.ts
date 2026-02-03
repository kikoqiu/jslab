declare module "int-set-utils" {
    export namespace IntSetUtils {
        /**
         * Binary Search for value existence.
         * @param {Int32Array} sortedArr - The sorted array to search within.
         * @param {number} value - The value to search for.
         * @returns {boolean} True if the value is found in the array.
         */
        function has(sortedArr: Int32Array, value: number): boolean;
        /**
         * Computes the union of two sorted Int32Arrays (A U B).
         * The resulting array will contain all unique elements from both input arrays, sorted in ascending order.
         * Linear time complexity O(|A| + |B|).
         * @param {Int32Array} arrA - The first sorted Int32Array.
         * @param {Int32Array} arrB - The second sorted Int32Array.
         * @returns {Int32Array} A new sorted Int32Array containing the union of elements.
         */
        function union(arrA: Int32Array, arrB: Int32Array): Int32Array;
        /**
         * Computes the intersection of two sorted Int32Arrays (A ∩ B).
         * The resulting array will contain only the elements common to both input arrays, sorted in ascending order.
         * Linear time complexity O(|A| + |B|).
         * @param {Int32Array} arrA - The first sorted Int32Array.
         * @param {Int32Array} arrB - The second sorted Int32Array.
         * @returns {Int32Array} A new sorted Int32Array containing the intersection of elements.
         */
        function intersection(arrA: Int32Array, arrB: Int32Array): Int32Array;
        /**
         * Computes the difference of two sorted Int32Arrays (A - B).
         * The resulting array will contain elements present in `arrA` but not in `arrB`, sorted in ascending order.
         * Linear time complexity O(|A| + |B|).
         * @param {Int32Array} arrA - The minuend sorted Int32Array.
         * @param {Int32Array} arrB - The subtrahend sorted Int32Array.
         * @returns {Int32Array} A new sorted Int32Array containing the difference (A - B) of elements.
         */
        function difference(arrA: Int32Array, arrB: Int32Array): Int32Array;
        /**
         * Sorts an Int32Array in ascending order and removes duplicate elements.
         * This function mutates the input array by sorting it in-place and then returns a subarray view
         * containing only the unique elements.
         * @param {Int32Array} rawArr - The Int32Array to sort and deduplicate. This array will be mutated.
         * @returns {Int32Array} A subarray view of the input `rawArr` containing sorted unique elements.
         */
        function sortAndUnique(rawArr: Int32Array): Int32Array;
    }
}
declare module "schreier-sims" {
    /**
     * A high-performance engine for computing the Base and Strong Generating Set (BSGS)
     * of a permutation group. This implementation is designed for the `group-engine` ecosystem,
     * leveraging the global PermutationRepository for efficient memory management.
     */
    export class SchreierSimsAlgorithm {
        /**
         * Factory method: Computes the Base and Strong Generating Set (BSGS) for a given set of generators.
         * @param {PermutationSet|number[]} groupSet - The set of permutations that generate the group.
         * @param {number[]} [initialBase=[]] - An optional array of points to serve as a prefix for the base.
         * @returns {SchreierSimsAlgorithm} A new `SchreierSimsAlgorithm` instance with the computed BSGS.
         * @static
         */
        static compute(groupSet: PermutationSet | number[], initialBase?: number[]): SchreierSimsAlgorithm;
        /**
         * Constructs a new Schreier-Sims Algorithm instance.
         * The instance can be initialized with an optional `initialBase` to prioritize certain points in the base.
         * @param {number[]} [initialBase=[]] - An optional array of points to form the initial base. These points will be stabilized first.
         */
        constructor(initialBase?: number[]);
        repo: import("permutation-repository").PermutationRepository;
        /**
         * The Base sequence B = [b_1, b_2, ..., b_k].
         * This is the sequence of points that define the stabilizer chain.
         * @type {number[]}
         * @protected
         */
        protected base: number[];
        /**
         * Transversals (Orbit Lookup Tables) for each level of the stabilizer chain.
         * `transversals[i]` stores the orbit of `base[i]` under the group `G^(i)` (the stabilizer of `base[0], ..., base[i-1]`).
         * Each map: `Key: Point p` in orbit, `Value: Permutation ID u` such that `u(base[i]) = p`.
         * @type {Map[]}
         * @protected
         */
        protected transversals: Map<any, any>[];
        /**
         * Strong Generators for each level of the stabilizer chain.
         * `generators[i]` is an array of permutation IDs that, along with the elements stabilizing `base[i]`,
         * generate the stabilizer `G^(i)` (the subgroup fixing `base[0], ..., base[i-1]`).
         * @type {number[][]}
         * @protected
         */
        protected generators: number[][];
        idIdentity: number;
        /**
         * Gets the current degree (number of points) on which the permutations act.
         * This value is dynamically managed by the underlying `PermutationRepository`.
         * @returns {number} The degree of the permutation group.
         */
        get degree(): number;
        /**
         * Calculates the exact order (size) of the group represented by the BSGS.
         * The order is computed as the product of the sizes of the orbits (transversals) at each level of the base.
         * @returns {bigint} The order of the group as a BigInt, to support very large group orders.
         */
        get order(): bigint;
        /**
         * Returns a string representation.
         * @returns {string} A result string.
         */
        toString(): string;
        /**
         * Checks if a given permutation is an element of the group represented by this BSGS.
         * The permutation is "sifted" through the stabilizer chain. If the process
         * reduces the permutation to the identity element, it is in the group.
         * @param {number|Int32Array|Array<number>} perm - The permutation to check, either as an ID or a raw array.
         * @returns {boolean} True if the permutation belongs to the group, false otherwise.
         */
        contains(perm: number | Int32Array | Array<number>): boolean;
        /**
         * Checks if the group acts transitively on a specified domain.
         * A group is transitive if, for any two points `x` and `y` in the domain,
         * there exists a group element `g` such that `g(x) = y`.
         * This implementation checks if the orbit of the first base point (`base[0]`) under the full group
         * covers the entire `domainSize`.
         * @param {number} domainSize - The size of the domain (e.g., `globalDegree`) to check for transitivity.
         * @returns {boolean} True if the group is transitive on the given domain, false otherwise.
         */
        isTransitive(domainSize: number): boolean;
        /**
         * Computes the stabilizer subgroup G_p of a specific point `p`.
         * G_p is defined as the set of all permutations `g` in the group G such that `g(p) = p`.
         * @param {number} point - The point (0-based index) for which to compute the stabilizer.
         * @returns {PermutationSet} A `PermutationSet` containing the generators of the stabilizer subgroup.
         */
        getStabilizer(point: number): PermutationSet;
        /**
         * Multiplies two permutations `idA` and `idB`.
         * The convention is `(A * B)(x) = A(B(x))`, meaning `idB` is applied first, then `idA`.
         * This method delegates to the underlying `PermutationRepository` for the actual multiplication.
         * @param {number} idA - The ID of the first permutation (A).
         * @param {number} idB - The ID of the second permutation (B).
         * @returns {number} The ID of the resulting permutation (A * B).
         * @private
         */
        private multiply;
        /**
         * Computes or retrieves the inverse of a given permutation ID.
         * This method delegates to the underlying `PermutationRepository` for inversion.
         * @param {number} id - The ID of the permutation to invert.
         * @returns {number} The ID of the inverse permutation.
         * @private
         */
        private inverse;
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
        private _strip;
        /**
         * The main incremental construction method for the BSGS.
         * It sifts a given permutation `gId`. If `gId` is not already in the group generated by the current BSGS,
         * the algorithm updates the stabilizer chain (`base`, `transversals`, `generators`) to include `gId`,
         * ensuring that the BSGS correctly represents the expanded group.
         * @param {number} gId - The permutation ID to be inserted into the BSGS.
         */
        siftAndInsert(gId: number): void;
        /**
         * Get generators as PermutationSet
         * @returns {PermutationSet}
         */
        getGeneratorsAsPermutationSet(): PermutationSet;
        /**
         * Adds a new strong generator `hId` to the `generators` list at the specified `level`
         * and triggers an update of the orbit (`transversal`) at that level.
         * @param {number} level - The level in the stabilizer chain to which the generator is added.
         * @param {number} hId - The ID of the permutation to add as a strong generator.
         * @private
         */
        private _addGeneratorToLevel;
        /**
         * Extends the base of the stabilizer chain by adding a new point.
         * This creates a new level in the chain, initializing its generators and transversal.
         * @param {number} point - The new point to be added to the base.
         * @private
         */
        private _extendBase;
        /**
         * Updates the orbit (transversal) at a specified `level` of the stabilizer chain.
         * This involves performing a Breadth-First Search (BFS) starting from existing orbit representatives
         * and the new generator `hId`. During the BFS, any newly discovered Schreier generators
         * (elements that stabilize `base[level]` but are not yet in the BSGS for `G^(level+1)`) are sifted and inserted.
         * @param {number} level - The level in the stabilizer chain whose orbit needs updating.
         * @param {number} hId - The ID of a newly added strong generator at this level.
         * @private
         */
        private _updateOrbit;
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
        private _processSchreierEdge;
        /**
         * Finds the smallest point (index) that is not fixed by the given permutation `permId`.
         * This is typically used to select a new base point when extending the stabilizer chain.
         * @param {number} permId - The ID of the permutation to analyze.
         * @returns {number} The 0-based index of the first point moved by the permutation, or -1 if the permutation is the identity.
         * @private
         */
        private _findFirstMovedPoint;
    }
    import { PermutationSet } from "group-engine";
}
declare module "group-utils" {
    /**
     * Parses a string in cycle notation into a flat permutation array.
     * Supports standard disjoint cycle notation, e.g., "(1 2 3)(4 5)".
     *
     * Assumptions:
     * - Input uses 1-based indexing (standard mathematical notation).
     * - Output is 0-based Int32Array.
     *
     * @param {string} str - The cycle string (e.g., "(1 2 3)").
     * @param {number} [degree=0] - The required degree (size) of the permutation.
     *                              If 0, inferred from max element.
     * @returns {Int32Array} The permutation in array form [p[0], p[1], ...].
     */
    export function parseCycles(str: string, degree?: number): Int32Array;
    /**
     * Decomposes a permutation into disjoint cycle notation string.
     * Uses 1-based indexing for the output string.
     *
     * @param {number|Int32Array|Array<number>} perm - Permutation ID (in globalRepo) or raw array.
     * @returns {string} Cycle notation, e.g., "(1 2 3)(4 5)". Returns "()" for identity.
     */
    export function decomposeToCycles(perm: number | Int32Array | Array<number>): string;
    /**
     * Creates generators for the Symmetric Group S_n.
     * Contains all n! permutations of n elements.
     *
     * Generators used:
     * 1. Transposition (1 2)  [0-based: (0 1)]
     * 2. Long Cycle (1 2 ... n) [0-based: (0 1 ... n-1)]
     *
     * @param {number} n - The degree (number of points).
     * @returns {PermutationSet} A set containing the generators.
     */
    export function createSymmetric(n: number): PermutationSet;
    /**
     * Creates generators for the Alternating Group A_n.
     * Contains even permutations. Order: n! / 2.
     *
     * Generators used:
     * 3-cycles of the form (1 2 i) for i = 3..n [0-based: (0 1 i) for i=2..n-1].
     *
     * @param {number} n - The degree.
     * @returns {PermutationSet} A set containing n-2 generators.
     */
    export function createAlternating(n: number): PermutationSet;
    /**
     * Creates generators for the Cyclic Group C_n.
     * Order: n.
     *
     * Generator used:
     * One cycle (1 2 ... n) [0-based: (0 1 ... n-1)].
     *
     * @param {number} n - The degree.
     * @returns {PermutationSet} A set containing 1 generator.
     */
    export function createCyclic(n: number): PermutationSet;
    /**
     * Creates generators for the Dihedral Group D_n.
     * Symmetries of a regular n-gon. Order: 2n.
     *
     * Generators used:
     * 1. Rotation r: (1 2 ... n)
     * 2. Reflection s: Fixes 1, maps k -> n-k+2 (mod n check)
     *    0-based logic: Fixes 0, Maps k -> -k mod n.
     *
     * @param {number} n - The number of vertices.
     * @returns {PermutationSet} A set containing 2 generators.
     */
    export function createDihedral(n: number): PermutationSet;
    /**
     * Creates generators for the Klein Four-Group V_4.
     * A subgroup of S_4 isomorphic to C_2 x C_2. Order: 4.
     *
     * Generators used:
     * 1. (1 2)(3 4) [0-based: (0 1)(2 3)]
     * 2. (1 3)(2 4) [0-based: (0 2)(1 3)]
     *
     * @returns {PermutationSet} A set containing 2 generators on 4 points.
     */
    export function createKleinFour(): PermutationSet;
    /**
     * Creates a generator set from a list of cycle strings.
     * Convenient wrapper for parsing multiple permutations and registering them.
     *
     * @param {string[]} cyclesStrArr - Array of strings, e.g. ["(1 2 3)", "(1 2)"].
     * @param {number} [degree=0] - Force a specific degree. If 0, auto-detected per string (max).
     * @returns {PermutationSet} The set of generators.
     */
    export function createFromCycleStrings(cyclesStrArr: string[], degree?: number): PermutationSet;
    /**
     * Creates the Direct Product of two groups: G x H.
     * The resulting group acts on disjoint sets of points.
     * Degree = Degree(G) + Degree(H).
     *
     * @param {PermutationSet} groupA - Generators for group G.
     * @param {PermutationSet} groupB - Generators for group H.
     * @param {...PermutationSet} extraGroups - Additional groups to include in the direct product.
     * @returns {PermutationSet} Generators for G x H.
     */
    export function createDirectProduct(groupA: PermutationSet, groupB: PermutationSet, ...extraGroups: PermutationSet[]): PermutationSet;
    /**
     * Creates generators for the Quaternion Group Q8.
     * Order: 8. Non-abelian.
     * Defined via Regular Representation in S8.
     * Elements: {1, i, j, k, -1, -i, -j, -k}
     *
     * @returns {PermutationSet}
     */
    export function createQuaternion(): PermutationSet;
    /**
     * Creates the Trivial Group (Identity).
     * @returns {PermutationSet}
     */
    export function createTrivial(): PermutationSet;
    /**
     * Creates a group from raw integer arrays.
     * Registers each raw permutation array into the global repository and returns a PermutationSet of their IDs.
     * Useful for loading from JSON or UI input.
     *
     * @param {Array<Int32Array | Array<number>>} arrays - An array of raw permutation arrays (e.g., `[[0,1,2],[1,0,2]]`).
     * @returns {PermutationSet} A PermutationSet containing the registered permutation IDs.
     */
    export function createFromRawArrays(arrays: Array<Int32Array | Array<number>>): PermutationSet;
    /**
     * Tetrahedral Group (Rotations of a regular tetrahedron).
     * Isomorphic to A4. Order 12.
     * @returns {PermutationSet} A PermutationSet representing the generators of the Tetrahedral Group.
     */
    export function createTetrahedral(): PermutationSet;
    /**
     * Octahedral Group (Rotations of a regular octahedron).
     * Isomorphic to S4. Order 24.
     * @returns {PermutationSet} A PermutationSet representing the generators of the Octahedral Group.
     */
    export function createOctahedral(): PermutationSet;
    /**
     * Icosahedral Group (Rotations of a regular icosahedron).
     * Isomorphic to A5. Order 60.
     * @returns {PermutationSet} A PermutationSet representing the generators of the Icosahedral Group.
     */
    export function createIcosahedral(): PermutationSet;
    /**
     * Attempts to find a new set of generators for the group where every generator
     * has an order less than or equal to `maxOrder`.
     *
     * It explores the group structure (via BFS) to find candidate elements of low order.
     * If it finds enough low-order elements to generate the original group (verified by SSA),
     * it returns this new set. Otherwise, returns null.
     *
     * @param {PermutationSet|Array<number>} inputGenerators - The original generators of the group.
     * @param {number} maxOrder - The maximum allowed order for the new generators.
     * @param {number} [maxSearchSize=50000] - Limit on the number of group elements to explore during search.
     * @returns {PermutationSet|null} A new PermutationSet if successful, or null if failed.
     */
    export function findLowOrderGenerators(inputGenerators: PermutationSet | Array<number>, maxOrder: number, maxSearchSize?: number): PermutationSet | null;
    import { PermutationSet } from "group-engine";
}
declare module "permutation-repository" {
    /**
     * Resets the global permutation repository.
     * This function clears all registered permutations and re-initializes
     * `globalRepo` to a new empty PermutationRepository instance.
     * Use with caution, as all previously obtained permutation IDs will become invalid.
     */
    export function resetGlobalRepo(): void;
    /**
     * Manages and stores unique permutations using a memory-optimized approach.
     * It provides a global repository (`globalRepo`) to register permutations, assign unique IDs,
     * and retrieve their data efficiently. It uses a trie-like structure for fast lookup
     * and a flat Int32Array for permutation storage, minimizing GC overhead.
     * The repository dynamically expands its capacity and can upgrade its `globalDegree`
     * if permutations with larger degrees are registered.
     */
    export class PermutationRepository {
        /**
         * @param {number} [initialDegree=4] - The initial degree (number of points) for permutations.
         *                                     The repository will automatically expand if permutations with higher degrees are registered.
         * @param {number} [initialPermCapacity=1024] - The initial capacity for storing permutations.
         *                                                The capacity will automatically expand as more unique permutations are registered.
         */
        constructor(initialDegree?: number, initialPermCapacity?: number);
        /**
         * The current maximum degree of all permutations stored in the repository.
         * Auto-expands as larger permutations are registered.
         * @type {number}
         */
        globalDegree: number;
        /**
         * The number of unique permutations currently stored in the repository.
         * Also serves as the next available ID for a new permutation.
         * @type {number}
         */
        count: number;
        /**
         * The current allocated capacity for storing permutations.
         * This defines the maximum number of unique permutations that can be stored
         * before the `permBuffer` needs to be expanded.
         * @type {number}
         */
        permCapacity: number;
        /**
         * A flat Int32Array that stores the actual permutation data.
         * Each permutation of `globalDegree` size occupies `globalDegree` contiguous slots.
         * @type {Int32Array}
         */
        permBuffer: Int32Array;
        /**
         * The size of each node in the trie buffer.
         * A node stores an ID and `globalDegree` child pointers.
         * @type {number}
         */
        trieNodeSize: number;
        /**
         * A flat Int32Array representing the memory arena for the trie nodes.
         * @type {Int32Array}
         */
        trieBuffer: Int32Array;
        /**
         * Pointer to the next available slot in the `trieBuffer` for allocating a new node.
         * @type {number}
         */
        trieFreePtr: number;
        /**
         * The unique ID for the identity permutation. This is always 0.
         * @type {number}
         * @readonly
         */
        readonly identity: number;
        /**
         * Allocates a new node from the Trie Buffer memory arena.
         * Auto-expands the buffer if more space is needed.
         * @returns {number} The starting index (pointer) of the newly allocated node within the `trieBuffer`.
         * @private
         */
        private _allocateNode;
        /**
         * Expands the `trieBuffer` (memory arena for trie nodes) when it runs out of space.
         * Doubles the current capacity.
         * @private
         */
        private _expandTrieBuffer;
        /**
         * Registers a permutation (or retrieves its existing ID if already registered).
         * If the input permutation's degree is greater than the current `globalDegree`,
         * the repository will automatically upgrade its degree.
         * @param {ArrayLike<number>} rawPerm - The permutation to register, represented as an array-like object (e.g., `[0, 2, 1]`).
         * @returns {number} The unique ID assigned to the permutation.
         */
        register(rawPerm: ArrayLike<number>): number;
        /**
         * Retrieves the permutation data for a given ID.
         * Returns a zero-copy view (subarray) of the internal `permBuffer`.
         * @param {number} id - The unique ID of the permutation to retrieve.
         * @returns {Int32Array} A subarray representing the permutation (e.g., `[0, 1, 2]`).
         */
        get(id: number): Int32Array;
        /**
         * Retrieves the permutation for a given ID and converts it into a 1-based cycle notation string.
         * @param {number} id - The unique ID of the permutation.
         * @returns {string} The cycle notation string (e.g., "(1 2 3)(4 5)"). Returns "()" for the identity permutation.
         */
        getAsCycles(id: number): string;
        /**
         * Writes a new permutation into the `permBuffer` at the specified ID's location.
         * Pads with identity mappings if `inputArr` is shorter than `globalDegree`.
         * @param {number} id - The unique ID assigned to this permutation.
         * @param {ArrayLike<number>} inputArr - The raw permutation array-like object.
         * @param {number} validLen - The actual length of the `inputArr` to copy.
         * @private
         */
        private _writeToPermBuffer;
        /**
         * Expands the `permBuffer` (permutation data pool) when it runs out of space.
         * Doubles the current capacity, copying existing data to the new buffer.
         * @private
         */
        private _expandPermCapacity;
        /**
         * Upgrades the `globalDegree` of the repository.
         * This is a "stop-the-world" operation that rebuilds both the permutation pool and the trie.
         * Existing permutations are padded with identity mappings to match the new degree.
         * @param {number} newDegree - The new, larger degree to upgrade to.
         * @private
         */
        private _upgradeDegree;
        /**
         * Computes the inverse of a given permutation ID.
         * If the inverse has already been registered, its ID is retrieved; otherwise, it's computed and registered.
         * @param {number} id - The ID of the permutation to invert.
         * @returns {number} The ID of the inverse permutation.
         */
        inverse(id: number): number;
        /**
         * Multiplies two permutations, `idA` and `idB`, according to the convention (A * B)(x) = A(B(x)).
         * This means permutation `idB` is applied first, then `idA`.
         * The resulting permutation is registered, and its ID is returned.
         * Exposed as Public API for solvers.
         * @param {number} idA - The ID of the first permutation (A).
         * @param {number} idB - The ID of the second permutation (B).
         * @returns {number} The ID of the resulting permutation (A * B).
         */
        multiply(idA: number, idB: number): number;
        /**
         * Computes the conjugate of permutation `h` by `g`: `g * h * g^-1`.
         * This operation results in a permutation that has the same cycle structure as `h`.
         * @param {number} g - The ID of the conjugating permutation (g).
         * @param {number} h - The ID of the permutation to be conjugated (h).
         * @returns {number} The ID of the resulting conjugated permutation (g * h * g^-1).
         */
        conjugate(g: number, h: number): number;
        /**
         * Computes the commutator of two permutations: `[idA, idB] = idA^-1 * idB^-1 * idA * idB`.
         * @param {number} idA - The ID of the first permutation (a).
         * @param {number} idB - The ID of the second permutation (b).
         * @returns {number} The ID of the resulting commutator permutation.
         */
        commutator(idA: number, idB: number): number;
    }
    /**
     * Singleton instance of the PermutationRepository.
     * All permutation operations should typically go through this global instance
     * to ensure consistent ID management and memory optimization.
     * @type {PermutationRepository}
     */
    export const globalRepo: PermutationRepository;
}
declare module "group-engine" {
    /**
     * Generates a subgroup from a set of generator permutations.
     * This method uses an iterative closure approach by repeatedly multiplying the current group by the generators until no new elements are found.
     * @param {PermutationSet | Array<number> | SchreierSimsAlgorithm} generators - A PermutationSet or an array of permutation IDs or a SchreierSimsAlgorithm instance to generate the group from.
     * @returns {PermutationSet} The fully generated subgroup (isGroup=true).
     * @throws {Error} If `generators` is an unknown type.
     */
    export function generateGroup(generators: PermutationSet | Array<number> | SchreierSimsAlgorithm): PermutationSet;
    /**
     * Represents a set of permutations, providing high-performance algebraic operations.
     * This class uses direct memory access and a global repository for efficient storage and computation.
     */
    export class PermutationSet {
        /**
         * Creates a PermutationSet containing only the identity permutation.
         * This set is always considered a group.
         * @returns {PermutationSet} A PermutationSet containing only the identity permutation.
         * @static
         */
        static identity(): PermutationSet;
        /**
         * @param {Int32Array|Array<number>} ids - Sorted, unique IDs from the repository.
         * @param {boolean} [isTrustedSortedUnique=false] - Skip sort/dedup if true.
         * @param {boolean} [isGroup=false] - Whether this set is known to be a mathematical group.
         */
        constructor(ids: Int32Array | Array<number>, isTrustedSortedUnique?: boolean, isGroup?: boolean);
        _ids: Int32Array<ArrayBufferLike>;
        /**
         * Flag indicating if this set satisfies group axioms.
         * @type {boolean}
         *
         */
        isGroup: boolean;
        /**
         * The number of elements in the set.
         * @type {number}
         */
        get size(): number;
        /**
         * Returns the internal Int32Array of sorted, unique permutation IDs.
         * Direct access should be read-only.
         * @returns {Int32Array}
         */
        get indices(): Int32Array;
        /**
         * Retrieves a permutation ID at a specific index within this set.
         * @param {number} index - The 0-based index of the element to retrieve.
         * @returns {number} The permutation ID.
         */
        get(index: number): number;
        /**
         * Creates a lightweight read-only view of a subset.
         * @param {number} start - The starting index (inclusive).
         * @param {number} end - The ending index (exclusive).
         * @returns {PermutationSet} A new set representing the slice.
         * @abstract
         */
        slice(start: number, end: number): PermutationSet;
        /**
         * Returns a string representation of the PermutationSet.
         * @returns {string} A string in the format "PermSet(ids=[...], isGroup=...)".
         */
        toString(): string;
        /**
         * Vectorized Group Multiplication: G * H = { g * h | g in G, h in H }
         * Optimized with direct heap access and loop hoisting.
         * Multiplies this set by another set.
         * @param {PermutationSet} other - The other set to multiply by.
         * @returns {PermutationSet} A new set representing the product.
         * @abstract
         */
        multiply(other: PermutationSet): PermutationSet;
        /**
         * Vectorized Inverse: G^-1 = { g^-1 | g in G }
         * Computes the inverse of each element in the set.
         * @returns {PermutationSet} A new set containing the inverses.
         */
        inverse(): PermutationSet;
        /**
         * Computes the union of this set with another set.
         * @param {PermutationSet} other - The other set.
         * @returns {PermutationSet} A new set representing the union.
         */
        union(other: PermutationSet): PermutationSet;
        /**
         * Computes the intersection of this set with another set.
         * @param {PermutationSet} other - The other set.
         * @returns {PermutationSet} A new set representing the intersection.
         */
        intersection(other: PermutationSet): PermutationSet;
        /**
         * Computes the difference of this set with another set (elements in this set but not in `other`).
         * @param {PermutationSet} other - The other set.
         * @returns {PermutationSet} A new set representing the difference.
         */
        difference(other: PermutationSet): PermutationSet;
        /**
         * Checks if this set is a superset of another set.
         * @param {PermutationSet} other - The other set.
         * @returns {boolean} True if this set contains all elements of `other`.
         */
        isSuperSetOf(other: PermutationSet): boolean;
        /**
         * Checks if equal.
         * @param {PermutationSet} other - The other set.
         * @returns {boolean}
         */
        equals(other: PermutationSet): boolean;
        /**
         * Internal helper to validate that the 'other' operand is a PermutationSet.
         * @param {*} other - The operand to check.
         * @throws {Error} If `other` is not an instance of PermutationSet.
         * @private
         */
        private _checkType;
        /**
         * Checks if the set forms an Abelian (Commutative) group.
         * Logic: For all g1, g2 in G, g1 * g2 == g2 * g1.
         * Performance: O(|G|^2 * Degree). Optimized with direct memory access.
         * @returns {boolean}
         */
        isAbelian(): boolean;
        /**
         * Calculates the Orbit of a point under this group.
         * Orbit(p) = { g(p) | g in G }
         * Implements BFS/Flood Fill without object allocation.
         *
         * @param {number} point - The integer point (0..degree-1)
         * @returns {Int32Array} Sorted unique array of points in the orbit.
         */
        calculateOrbit(point: number): Int32Array;
        /**
         * Decomposes this group G into right cosets of a subgroup H.
         * G = U (H * g_i)
         *
         *
         * @param {PermutationSet} subgroupH - The subgroup H to decompose by.
         * @returns {Array<PermutationSet>} An array of disjoint right cosets.
         */
        rightCosetDecomposition(subgroupH: PermutationSet): Array<PermutationSet>;
        /**
         * Generates a subgroup from this.
         * This method uses an iterative closure approach by repeatedly multiplying the current group by the generators until no new elements are found.
         * @returns {PermutationSet} The fully generated subgroup (isGroup=true).
         * @throws {Error} If `generators` is an unknown type.
         */
        generateGroupFromThis(): PermutationSet;
        /**
         * Returns an iterator for the permutation IDs in this set.
         * @returns {Iterator<number>} An iterator for the `_ids` Int32Array.
         */
        [Symbol.iterator](): Iterator<number>;
    }
    import { SchreierSimsAlgorithm } from "schreier-sims";
}
declare module "group-private-utils" {
    /**
     * Checks if a given number is a small prime.
     * Uses trial division for efficiency with smaller numbers.
     * @param {number} n - The number to check.
     * @returns {boolean} True if the number is prime, false otherwise.
     * @private
     */
    export function _isSmallPrime(n: number): boolean;
    /**
     * Generates a pseudo-random element from the group represented by the given `SchreierSimsAlgorithm` instance.
     * @param {SchreierSimsAlgorithm} ssa - The `SchreierSimsAlgorithm` instance of the group.
     * @returns {number} The ID of a randomly selected permutation from the group.
     * @private
     */
    export function _getRandomElement(ssa: SchreierSimsAlgorithm): number;
    /**
     * Computes the exponentiation of a permutation: `gId^exp`.
     * Uses binary exponentiation (exponentiation by squaring) for efficiency.
     * @param {number} gId - The ID of the permutation (g).
     * @param {number} exp - The integer exponent.
     * @returns {number} The ID of the resulting permutation `gId^exp`.
     * @private
     */
    export function _pow(gId: number, exp: number): number;
    /**
     * Computes the least common multiple (LCM) of two integers.
     * @param {number} a - The first integer.
     * @param {number} b - The second integer.
     * @returns {number} The least common multiple of a and b.
     * @private
     */
    export function _lcm(a: number, b: number): number;
    /**
     * Computes the greatest common divisor (GCD) of two integers using the Euclidean algorithm.
     * @param {number} a - The first integer.
     * @param {number} b - The second integer.
     * @returns {number} The greatest common divisor of a and b.
     * @private
     */
    export function _gcd(a: number, b: number): number;
    /**
     * Checks if a BigInt number `n` is a power of another BigInt `pBig`.
     * For example, if `pBig` is 2, it checks if `n` is 2^k for some k >= 0.
     * @param {bigint} n - The number to check.
     * @param {bigint} pBig - The base number (must be > 1).
     * @returns {boolean} True if `n` is a power of `pBig`, false otherwise.
     * @private
     */
    export function _isPowerOfP(n: bigint, pBig: bigint): boolean;
    /**
     * Extracts the p-part of a permutation `gId`.
     * Given a permutation `g` with order `|g| = p^k * m`, where `gcd(p, m) = 1`,
     * this function returns `g^m`, which is the p-part of `g` and has order `p^k`.
     * @param {number} gId - The ID of the permutation (g).
     * @param {number} p - The prime number p.
     * @returns {number} The ID of the p-part of the permutation.
     * @private
     */
    export function _getPPart(gId: number, p: number): number;
    /**
     * Computes an approximate order of a permutation by finding the LCM of its cycle lengths.
     * If the order exceeds limit, it returns limit+1 as a sentinel value.
     * @param {*} perm - The permutation array.
     * @param {number} [limit=60] - The upper limit for the order.
     * @returns {number} - The approximate order of the permutation.
     */
    export function calcApproxOrder(perm: any, limit?: number): number;
}
declare module "group-structural-utils" {
    /**
     * Checks if `subGroup` is a subgroup of `superGroup`.
     * This is determined by verifying that all generators of `subGroup` are contained within `superGroup`.
     * @param {PermutationSet|SchreierSimsAlgorithm} superGroup - The potential supergroup G.
     * @param {PermutationSet|SchreierSimsAlgorithm} subGroup - The potential subgroup H.
     * @returns {boolean} True if H is a subgroup of G, false otherwise.
     */
    export function isSubgroup(superGroup: PermutationSet | SchreierSimsAlgorithm, subGroup: PermutationSet | SchreierSimsAlgorithm): boolean;
    /**
     * Checks if `normalN` is a normal subgroup of `superGroup` (N ◁ G).
     * This is verified by checking if for every generator `g` of `superGroup` and every generator `n` of `normalN`,
     * the conjugate `g * n * g^-1` is an element of `normalN`.
     * @param {PermutationSet|SchreierSimsAlgorithm} superGroup - The supergroup G.
     * @param {PermutationSet|SchreierSimsAlgorithm} normalN - The potential normal subgroup N.
     * @returns {boolean} True if N is a normal subgroup of G, false otherwise.
     */
    export function isNormal(superGroup: PermutationSet | SchreierSimsAlgorithm, normalN: PermutationSet | SchreierSimsAlgorithm): boolean;
    /**
     * Computes the normal closure of a subset `subsetS` within the group `groupG`.
     * The normal closure is the smallest normal subgroup of `groupG` that contains `subsetS`.
     * It is generated by all conjugates of elements of `subsetS` by elements of `groupG`.
     * @param {PermutationSet|SchreierSimsAlgorithm} groupG - The containing group G.
     * @param {PermutationSet|Array<number>|number} subsetS - The subset S (generators, array of IDs, or single ID).
     * @returns {SchreierSimsAlgorithm} The `SchreierSimsAlgorithm` instance representing the normal closure.
     */
    export function getNormalClosure(groupG: PermutationSet | SchreierSimsAlgorithm, subsetS: PermutationSet | Array<number> | number): SchreierSimsAlgorithm;
    /**
     * Computes the commutator subgroup G' = [G, G] of a group G.
     * This subgroup is generated by all commutators `[g1, g2] = g1^-1 * g2^-1 * g1 * g2` for `g1, g2` in G.
     * @param {PermutationSet|SchreierSimsAlgorithm} group - The group G.
     * @returns {SchreierSimsAlgorithm} The `SchreierSimsAlgorithm` instance representing the commutator subgroup.
     */
    export function getCommutatorSubgroup(group: PermutationSet | SchreierSimsAlgorithm): SchreierSimsAlgorithm;
    /**
     * Checks if a group is solvable.
     * A group G is solvable if its derived series terminates in the trivial group {e}.
     * The derived series is G^(0) = G, G^(n+1) = [G^(n), G^(n)].
     * @param {PermutationSet|SchreierSimsAlgorithm} group - The group G to check for solvability.
     * @returns {boolean} True if the group is solvable, false otherwise.
     */
    export function isSolvable(group: PermutationSet | SchreierSimsAlgorithm): boolean;
    /**
     * Checks if a group is simple.
     * A group G is simple if its only normal subgroups are the trivial group {e} and G itself.
     * This function uses a probabilistic approach for non-abelian groups and may return "uncertain" for large groups.
     * @param {PermutationSet|SchreierSimsAlgorithm} group - The group G to check for simplicity.
     * @param {number} [randomTests=10] - Number of random conjugates to test for non-abelian groups. Higher values increase confidence but also computation time.
     * @returns {number} 1 (Proven Simple), 0 (Proven Not Simple), -1 (Uncertain - heuristically likely simple but not strictly proven).
     */
    export function isSimple(group: PermutationSet | SchreierSimsAlgorithm, randomTests?: number): number;
    /**
     * Computes the structure of the quotient group G/N, along with a mapping
     * that lifts elements from G/N back to representatives in G.
     * This function is computationally intensive and only feasible for quotient groups
     * with a small index `[G:N]`.
     * @param {PermutationSet|SchreierSimsAlgorithm} groupG - The group G.
     * @param {PermutationSet|SchreierSimsAlgorithm} normalN - The normal subgroup N of G.
     * @param {number} [maxIndex=2000] - The maximum allowed index `[G:N]` for explicit construction.
     * @returns {QuotientGroupMap} An object containing the quotient group (as PermutationSet)
     *   and an array of representatives for each coset.
     * @throws {Error} If N is not a normal subgroup of G, or if `[G:N]` exceeds `maxIndex`.
     */
    export function getQuotientStructure(groupG: PermutationSet | SchreierSimsAlgorithm, normalN: PermutationSet | SchreierSimsAlgorithm, maxIndex?: number): QuotientGroupMap;
    /**
     * Heuristically checks if two groups `groupA` and `groupB` are isomorphic.
     * This function compares structural invariants (order, abelian-ness, derived series length).
     * It cannot definitively prove isomorphism without constructing an explicit isomorphism map,
     * but it can reliably prove non-isomorphism and provide a strong indication for isomorphism.
     * @param {PermutationSet|SchreierSimsAlgorithm} groupA - The first group.
     * @param {PermutationSet|SchreierSimsAlgorithm} groupB - The second group.
     * @returns {number} 1 (Isomorphic, if strictly proven - rare), 0 (Not isomorphic, strictly proven), -1 (Uncertain - heuristically likely isomorphic but not strictly proven).
     */
    export function areIsomorphic(groupA: PermutationSet | SchreierSimsAlgorithm, groupB: PermutationSet | SchreierSimsAlgorithm): number;
    /**
     * Computes the mixed commutator subgroup `[subA, subB]` of two subgroups `subA` and `subB`
     * within a larger group `groupG`.
     * The result is the normal closure of all commutators `[a, b]` (where `a` is from `subA` and `b` is from `subB`)
     * within the group `groupG`.
     * @param {PermutationSet|SchreierSimsAlgorithm} groupG - The containing parent group G, used for computing the normal closure.
     * @param {PermutationSet|SchreierSimsAlgorithm} subA - The first subgroup A.
     * @param {PermutationSet|SchreierSimsAlgorithm} subB - The second subgroup B.
     * @returns {SchreierSimsAlgorithm} The `SchreierSimsAlgorithm` instance representing the mixed commutator subgroup `[A, B]`.
     */
    export function getMixedCommutatorSubgroup(groupG: PermutationSet | SchreierSimsAlgorithm, subA: PermutationSet | SchreierSimsAlgorithm, subB: PermutationSet | SchreierSimsAlgorithm): SchreierSimsAlgorithm;
    /**
     * Computes the lower central series of a group G.
     * The series is defined recursively as:
     * G_0 = G
     * G_{i+1} = [G_i, G] (the mixed commutator subgroup of G_i and G).
     * The series terminates when G_{i+1} = G_i or G_i = {e}.
     * @param {PermutationSet|SchreierSimsAlgorithm} group - The group G.
     * @returns {SchreierSimsAlgorithm[]} An array of `SchreierSimsAlgorithm` instances, representing the subgroups in the lower central series: `[G_0, G_1, ..., G_k]`.
     */
    export function getLowerCentralSeries(group: PermutationSet | SchreierSimsAlgorithm): SchreierSimsAlgorithm[];
    /**
     * Checks if a group is nilpotent.
     * A group G is nilpotent if its lower central series terminates at the trivial group {e}.
     * Every nilpotent group is solvable.
     * @param {PermutationSet|SchreierSimsAlgorithm} group - The group G to check for nilpotency.
     * @returns {number} 1 (Nilpotent), 0 (Not Nilpotent).
     */
    export function isNilpotent(group: PermutationSet | SchreierSimsAlgorithm): number;
    /**
     * Analyzes a list of candidate generators to determine a minimal (fundamental) generating set.
     * It uses the Schreier-Sims Algorithm to identify and separate redundant generators.
     * @param {number[]|PermutationSet} candidateIds - An array of permutation IDs that are potential generators.
     * @returns {{
     *   fundamental: number[],
     *   redundant: number[],
     *   ssa: SchreierSimsAlgorithm
     * }} An object containing:
     *   - `fundamental`: An array of permutation IDs that form a minimal generating set.
     *   - `redundant`: An array of permutation IDs that are generated by the `fundamental` set.
     *   - `ssa`: The `SchreierSimsAlgorithm` instance computed from the `fundamental` generators.
     */
    export function analyzeGenerators(candidateIds: number[] | PermutationSet): {
        fundamental: number[];
        redundant: number[];
        ssa: SchreierSimsAlgorithm;
    };
    /**
     * Computes a Sylow p-subgroup of G.
     *
     * A Sylow p-subgroup of a group G is a maximal p-subgroup of G.
     * If |G| = p^k * m where gcd(p, m) = 1, then a Sylow p-subgroup has order p^k.
     *
     * ALGORITHM STRATEGY:
     * We use a Randomized Greedy approach with restart ("Random Search").
     * 1. Compute target order p^k.
     * 2. Start with P = {e}.
     * 3. Repeatedly pick random elements g from G.
     * 4. Extract the p-part h from g (so order(h) is a power of p).
     * 5. Attempt to extend P by h: P_new = <P, h>.
     * 6. If P_new is a p-group (order is power of p), update P = P_new.
     * 7. If |P| reaches p^k, we are done.
     *
     * Note: This is a Monte Carlo Las Vegas algorithm. It is correct if it terminates,
     * but theoretically could run indefinitely (though very unlikely for standard groups).
     * We include safeguards/limits.
    */
    /**
     * Computes a Sylow p-subgroup of a given group G.
     * A Sylow p-subgroup is a maximal p-subgroup of G, with order p^k where p^k divides |G|
     * and p^(k+1) does not.
     * The algorithm uses a randomized greedy approach with restarts.
     * @param {PermutationSet|SchreierSimsAlgorithm} group - The group G for which to find a Sylow p-subgroup.
     * @param {number} p - The prime number p.
     * @returns {PermutationSet} A `PermutationSet` containing the generators of a Sylow p-subgroup.
     * @throws {Error} If the algorithm fails to construct a Sylow p-subgroup within the configured random search limits.
     */
    export function getSylowSubgroup(group: PermutationSet | SchreierSimsAlgorithm, p: number): PermutationSet;
    import { PermutationSet } from "group-engine";
    import { SchreierSimsAlgorithm } from "schreier-sims";
    /**
     * Represents a Quotient Group G/N, providing a mapping between
     * elements of the quotient group (as permutations on coset indices)
     * and their representatives in the original group G.
     */
    class QuotientGroupMap {
        /**
         * @param {PermutationSet} quotientGroup - A PermutationSet whose elements act on the coset indices (0-based).
         * @param {Int32Array} representatives - An Int32Array where `representatives[i]` is a chosen representative from the i-th coset.
         * @param {bigint} quotientOrder - The order of the quotient group, |G/N|.
         */
        constructor(quotientGroup: PermutationSet, representatives: Int32Array, quotientOrder: bigint);
        group: PermutationSet;
        representatives: Int32Array<ArrayBufferLike>;
        size: bigint;
        /**
         * Lifts a quotient group element (represented by a permutation ID) back to
         * a specific representative element in the original group G.
         * The returned element `g` is such that the quotient element corresponds to the coset `Ng`.
         * @param {number} quotientPermId - The ID of the permutation in the quotient group.
         * @returns {number} The ID of a representative element in the original group G.
         * @throws {Error} If the `quotientPermId` maps to an invalid coset index.
         */
        lift(quotientPermId: number): number;
    }
    export {};
}
declare module "group-utils-coxeter" {
    /**
     * Attempts to find a set of generators that mimic a Coxeter system (strong generating set of involutions).
     *
     * @param {PermutationSet|number[]} inputGenerators - The initial generators defining the group.
     * @param {{beamWidth:number, generations:number, forcedBase:number[]}} [options] - Configuration options.
     * @param {number} [options.beamWidth=50] - Number of candidates to keep in beam search.
     * @param {number} [options.generations=30] - Number of mixing generations.
     * @param {number[]} [options.forcedBase] - Force a specific base order.
     * @returns {PermutationSet} A new set of generators.
     */
    export function findCoxeterLikeGenerators(inputGenerators: PermutationSet | number[], options?: {
        beamWidth: number;
        generations: number;
        forcedBase: number[];
    }): PermutationSet;
    import { PermutationSet } from "group-engine";
}
declare module "group-visualizer" {
    /**
     * Generates human-readable algebraic names for all group elements (e.g., 'e', 'a', 'b', 'ab', 'a^2').
     * This function uses a Breadth-First Search (BFS) approach, starting from the identity and generators,
     * to construct the shortest and most intuitive names based on generator products.
     * @param {number[]|PermutationSet} allElementIds - A sorted list of all unique permutation IDs belonging to the group.
     * @param {number[]|PermutationSet} generatorIds - A list of permutation IDs that are the fundamental generators of the group.
     * @param {string[]} [genLabels] - A list of strings that are the labels for the generators. Default to undefined means to use a,b,c,...
     * @returns {Map<number, string>} A Map where keys are permutation IDs and values are their corresponding generated algebraic names.
     */
    export function generateNames(allElementIds: number[] | PermutationSet, generatorIds: number[] | PermutationSet, genLabels?: string[]): Map<number, string>;
    /**
     * Generates a Multiplication (Cayley) Table for a group.
     * `inputIds` are treated as candidate generators. The function will determine a fundamental set of generators, expand the group to all its elements,
     * and generate names for them. The table will represent the full group.
     *
     * return an object
     * A 2D array where `matrix[row][col]` is the permutation ID of `rowElement * colElement`.
     * A 2D array where `grid[row][col]` is the algebraic name (string) of `rowElement * colElement`.
     * A Map where keys are permutation IDs and values are their 1-based cycle notation strings (e.g., "(1 2 3)").
     * An HTML string representation of the Cayley table with semantic coloring and tooltips.
     *
     * @param {number[]} inputIds - An array of candidate generator IDs.
     * @param {Map<number, string>} [nameMap=null] - Optional. A custom map of all permutation IDs to their display names. Use generateNames to generate.
     * @see generateNames
     * @returns {{
     *   matrix: number[][],
     *   grid: string[][],
     *   cycleMap: Map<number, string>,
     *   html: string,
     *   nameMap: Map<number, string>
     * }} An object containing the generated table data.
     *
     * @throws {Error} If `nameMap` is provided in manual mode but is incomplete (missing names for `inputIds`).
     */
    export function generateMultiplicationTable(inputIds: number[], nameMap?: Map<number, string>): {
        matrix: number[][];
        grid: string[][];
        cycleMap: Map<number, string>;
        html: string;
        nameMap: Map<number, string>;
    };
    import { PermutationSet } from "group-engine";
}
declare module "group-visualizer-cayley-graph" {
    /**
     * @typedef {object} CayleyGraphData
     * @property {Array<object>} nodes - Array of node objects with id, name, x, y, z, vx, vy, vz properties.
     * @property {Array<object>} links - Array of link objects with source, target, genId, color, order, isDirected properties.
     * @property {Array<object>} legend - Array of legend objects with label, color, genId properties.
     * @property {VisualizerCayleyForceSimulator} simulator - The force simulator instance.
     * @property {_CayleyGraphConfig} config - The effective physics configuration used.
     * @property {Map<number, string>} nameMap - the used nameMap
     */
    /**
     * Generates the graph data structure for a Cayley graph, including nodes, links, and a physics simulator.
     * This function can return either a full data structure for advanced usage or a Plotly-ready frame.
     * @param {number[]|PermutationSet} inputIds - Array of generator IDs used to construct the group.
     * @param {Partial<_CayleyGraphConfig>} [customConfig={}] - Optional physics configuration overrides.
     * @param {number[]|PermutationSet} [extraGenerators=[]] - Optional additional generators to visualize but exclude from physics forces.
     * @returns {CayleyGraphData | {data: Array<object>, layout: object, nameMap: Map<number, string>}} Returns a `CayleyGraphData` object if `config.advancedMode` is true, otherwise returns a Plotly-compatible object `{data, layout, nameMap}`.
     * @see generateNames
     * @throws {Error} If no generators are provided.
     */
    export function generateCayleyGraphForPlotly(inputIds: number[] | PermutationSet, customConfig?: Partial<_CayleyGraphConfig>, extraGenerators?: number[] | PermutationSet): CayleyGraphData | {
        data: Array<object>;
        layout: object;
        nameMap: Map<number, string>;
    };
    /**
     * Specialized 3D Force Simulator implementing a physics-based layout algorithm with simulated annealing.
     * It's designed to position nodes and edges of a Cayley graph in 3D space,
     * applying forces like repulsion, spring forces, and cycle-specific planar/convexity forces.
     */
    export class VisualizerCayleyForceSimulator {
        /**
         * @param {Array<object>} nodes - An array of node objects, each with 'id', 'x', 'y', 'z', 'vx', 'vy', 'vz' properties.
         * @param {object} constraints - An object containing arrays of 'edges', 'chords', 'cycles', and 'angleTriplets'.
         * @param {number[]} generators - An array of generator IDs.
         * @param {Map<number, object>} genMeta - A map from generator ID to its metadata (e.g., color, order).
         * @param {_CayleyGraphConfig} config - The physics configuration for the simulator.
         */
        constructor(nodes: Array<object>, constraints: object, generators: number[], genMeta: Map<number, object>, config: _CayleyGraphConfig);
        nodes: object[];
        edges: any;
        chords: any;
        cycles: any;
        angleTriplets: any;
        generators: number[];
        genMeta: Map<number, object>;
        config: _CayleyGraphConfig;
        nodeMap: Map<any, any>;
        /**
         * Executes one step of the physics simulation.
         * Applies forces, integrates velocities, and updates node positions.
         * @param {number} [jitterFactor=0] - The magnitude of random noise to apply to node positions, used for simulated annealing during warmup.
         */
        tick(jitterFactor?: number): void;
        /**
         * Dynamically adjusts the ideal lengths of chord constraints based on the average observed angles within each cycle.
         * This helps maintain the geometric integrity of cycles as the graph settles.
         * @private
         */
        private _updateChordTargets;
        /**
         * Applies Hooke's Law (spring forces) to a list of links (edges or chords).
         * @param {Array<object>} list - An array of link objects, each with 'source', 'target', 'dist', 'strength' properties.
         * @private
         */
        private _applySprings;
        /**
         * Applies forces to cycle nodes to encourage planar and convex arrangements.
         * Calculates a centroid and normal vector for each cycle to guide these forces.
         * @private
         */
        private _applyCycleForces;
        /**
         * Runs the simulation for a specified number of iterations with simulated annealing.
         * The `jitterFactor` decays linearly during the first 90% of iterations, then remains at 0 for the last 10%.
         * @param {number} [iterations=2000] - The total number of simulation ticks to run during the warmup phase.
         */
        warmup(iterations?: number): void;
        /**
         * Generates a Plotly-compatible data frame (traces and layout) representing the current state of the Cayley graph.
         * Includes 3D scatter plots for nodes, lines for edges, and cones for directed edges.
         * @returns {{data: Array<object>, layout: object}} An object containing Plotly trace data and layout configuration.
         */
        getPlotlyFrame(): {
            data: Array<object>;
            layout: object;
        };
    }
    export type CayleyGraphData = {
        /**
         * - Array of node objects with id, name, x, y, z, vx, vy, vz properties.
         */
        nodes: Array<object>;
        /**
         * - Array of link objects with source, target, genId, color, order, isDirected properties.
         */
        links: Array<object>;
        /**
         * - Array of legend objects with label, color, genId properties.
         */
        legend: Array<object>;
        /**
         * - The force simulator instance.
         */
        simulator: VisualizerCayleyForceSimulator;
        /**
         * - The effective physics configuration used.
         */
        config: _CayleyGraphConfig;
        /**
         * - the used nameMap
         */
        nameMap: Map<number, string>;
    };
    export type _CayleyGraphConfig = {
        /**
         * - Base distance factor. Used to determine ideal edge length (d0 / order).
         */
        d0: number;
        /**
         * - Strength of the Coulomb-like repulsive force between all nodes.
         */
        repulsion: number;
        /**
         * - Spring constant for edges directly connecting elements (Hooke's law).
         */
        edgeStrength: number;
        /**
         * - Spring constant for 'chord' edges in cycles, maintaining their geometric shape.
         */
        chordStrength: number;
        /**
         * - Strength of the force that flattens cycles onto a plane.
         */
        planarStrength: number;
        /**
         * - Strength of the force that pushes nodes away from the center of a cycle, maintaining convexity.
         */
        convexityStrength: number;
        /**
         * - Magnitude of random initial displacement applied to cycle groups to untangle them during setup.
         */
        initialOffsetDist: number;
        /**
         * - Velocity damping factor, reducing oscillation.
         */
        decay: number;
        /**
         * - Strength of the gravitational force pulling all nodes towards the origin.
         */
        centerPull: number;
        /**
         * - Simulation time step for integration.
         */
        timeStep: number;
        /**
         * - Maximum random displacement applied to nodes during the initial annealing phase.
         */
        jitterMax: number;
        /**
         * - How often (in ticks) to recalculate average angles for chord length adjustment. Set to 0 to disable.
         */
        dynamicAngleUpdateRate: number;
        /**
         * - Number of simulation ticks to run during the warmup phase for initial layout.
         */
        warmupRuns: number;
        /**
         * - If true, returns detailed simulator objects; otherwise, returns only Plotly frame.
         */
        advancedMode: boolean;
        /**
         * - the nameMap to use for the full group elements. Use generateNames to generate.
         */
        nameMap: Map<number, string>;
        /**
         * - use SSA to rewrite generators to StrongGenerators.
         */
        rewriteToStrongGenerators: boolean;
    };
    import { PermutationSet } from "group-engine";
}
declare module "groups" {
    export * from "group-engine";
    export * from "int-set-utils";
    export * from "permutation-repository";
    export * from "schreier-sims";
    export * from "group-structural-utils";
    export * from "group-utils";
    export * from "group-visualizer";
    export * from "group-visualizer-cayley-graph";
    export * from "group-utils-coxeter";
}
