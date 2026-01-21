/**
 * SmartURLCodec
 * A utility class to compress strings into URL parameters using the shortest possible representation.
 * It compares raw URI encoding, Base64 encoding, and Deflate compression to find the optimal strategy.
 */
class SmartURLCodec {

    /**
     * Helper: Convert ArrayBuffer to URL-Safe Base64 string.
     * Replaces '+' with '-', '/' with '_', and removes padding '='.
     * @param {ArrayBuffer} buffer - The binary data.
     * @returns {string} URL-safe Base64 string.
     */
    static #bufferToUrlBase64(buffer) {
        let binary = '';
        const bytes = new Uint8Array(buffer);
        const len = bytes.byteLength;
        // Convert bytes to binary string
        for (let i = 0; i < len; i++) {
            binary += String.fromCharCode(bytes[i]);
        }
        // Standard Base64 -> URL Safe Base64
        return window.btoa(binary)
            .replace(/\+/g, '-')
            .replace(/\//g, '_')
            .replace(/=+$/, '');
    }

    /**
     * Helper: Convert URL-Safe Base64 string back to Uint8Array.
     * Restores padding and standard Base64 characters.
     * @param {string} base64Url - The encoded string.
     * @returns {Uint8Array} The binary data.
     */
    static #urlBase64ToBuffer(base64Url) {
        // Restore standard Base64 characters
        let base64 = base64Url.replace(/-/g, '+').replace(/_/g, '/');
        // Restore padding
        while (base64.length % 4) {
            base64 += '=';
        }
        const binary = window.atob(base64);
        const len = binary.length;
        const bytes = new Uint8Array(len);
        for (let i = 0; i < len; i++) {
            bytes[i] = binary.charCodeAt(i);
        }
        return bytes;
    }

    /**
     * Generates a URL with the given data encoded in the shortest format.
     * Strategies:
     * 1. 'zcode': Deflate-raw compression + Base64 (Best for long/repetitive text).
     * 2. 'bcode': UTF-8 bytes + Base64 (Good for CJK characters compared to encodeURIComponent).
     * 3. 'code': Standard encodeURIComponent (Best for very short ASCII text).
     * 
     * @param {string} baseUrl - The target URL (e.g., "http://localhost").
     * @param {string} data - The string content to encode.
     * @returns {Promise<string>} The complete URL with parameters.
     */
    static async buildUrl(baseUrl, data) {
        if (!data) return baseUrl;

        // 1. Strategy: Standard URI Component (code)
        //const codeStr = encodeURIComponent(data);

        // Prepare binary data (UTF-8)
        const encoder = new TextEncoder();
        const rawData = encoder.encode(data);

        // 2. Strategy: Uncompressed Base64 (bcode)
        const bcodeStr = this.#bufferToUrlBase64(rawData.buffer);

        // 3. Strategy: Compressed Base64 (zcode)
        // Using 'deflate-raw' to avoid zlib headers for minimal size
        const stream = new Blob([rawData]).stream().pipeThrough(new CompressionStream('deflate-raw'));
        const compressedBuffer = await new Response(stream).arrayBuffer();
        const zcodeStr = this.#bufferToUrlBase64(compressedBuffer);

        // Calculate total length including the parameter key overhead
        // "code=".length = 5
        // "bcode=".length = 6
        // "zcode=".length = 6
        //const lenCode = codeStr.length + 5;
        const lenBcode = bcodeStr.length + 6;
        const lenZcode = zcodeStr.length + 6;

        let finalQuery = '';

        // Determine the winner (shortest length)
        if (lenZcode < lenBcode) {
            finalQuery = `zcode=${zcodeStr}`;
        } else {
            finalQuery = `bcode=${bcodeStr}`;
        }

        /*if (lenZcode < lenBcode && lenZcode < lenCode) {
            finalQuery = `zcode=${zcodeStr}`;
        } else if (lenBcode < lenCode) {
            finalQuery = `bcode=${bcodeStr}`;
        } else {
            finalQuery = `code=${codeStr}`;
        }*/

        // Append to URL
        const separator = baseUrl.includes('?') ? '&' : '?';
        return `${baseUrl}${separator}${finalQuery}`;
    }

    /**
     * Decodes the data from a URL based on priority: zcode > bcode > code.
     * @param {string} [urlStr] - Optional URL string. Defaults to window.location.href.
     * @returns {Promise<string|null>} The decoded string or null if decoding fails.
     */
    static async resolveData(urlStr) {
        const targetUrl = urlStr ? new URL(urlStr) : new URL(window.location.href);
        const params = targetUrl.searchParams;

        // Priority 1: Compressed Code (zcode)
        if (params.has('zcode')) {
            try {
                const zcode = params.get('zcode');
                const compressedBytes = this.#urlBase64ToBuffer(zcode);
                
                // Decompress using deflate-raw
                const stream = new Blob([compressedBytes]).stream().pipeThrough(new DecompressionStream('deflate-raw'));
                const decompressedBuffer = await new Response(stream).arrayBuffer();
                
                return new TextDecoder().decode(decompressedBuffer);
            } catch (e) {
                console.warn("SmartURLCodec: Failed to decode 'zcode'", e);
            }
        }

        // Priority 2: Base64 Code (bcode)
        if (params.has('bcode')) {
            try {
                const bcode = params.get('bcode');
                const bytes = this.#urlBase64ToBuffer(bcode);
                return new TextDecoder().decode(bytes);
            } catch (e) {
                console.warn("SmartURLCodec: Failed to decode 'bcode'", e);
            }
        }

        // Priority 3: Standard Code (code)
        if (params.has('code')) {
            try {
                return decodeURIComponent(params.get('code'));
            } catch (e) {
                console.warn("SmartURLCodec: Failed to decode 'code'", e);
            }
        }

        return null; // No valid data found
    }
}
