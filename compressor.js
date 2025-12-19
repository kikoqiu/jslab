/**
 * BinaryCompressor
 * Helper class for compressing/decompressing text and JSON data to ArrayBuffer.
 * 
 * Features:
 * 1. Native Gzip support using CompressionStream.
 * 2. Automatic fallback to UTF-8 if decompression fails.
 * 3. Advanced MessagePack support.
 */
class BinaryCompressor {

  // ==========================================
  // Basic Mode: Text/JSON String <-> Gzip
  // ==========================================

  /**
   * Compresses a string into a Gzipped ArrayBuffer.
   * 
   * @param {string | ArrayBuffer} text - The input string.
   * @returns {Promise<ArrayBuffer>} The binary compressed data.
   */
  static async compressString(text) {
    let rawBytes;
    // 1. Convert String to Uint8Array (UTF-8)
    if(typeof(text) === "string"){
        const encoder = new TextEncoder();
        rawBytes = encoder.encode(text);
    }else{
        rawBytes = text;
    }    

    // 2. Compress using Gzip
    return await this._gzip(rawBytes);
  }

  /**
   * Decompresses an ArrayBuffer back to a string.
   * Logic: Try Gzip decompression -> If fails, treat as raw UTF-8.
   * 
   * @param {ArrayBuffer|Uint8Array} buffer - The binary data.
   * @returns {Promise<string>} The restored string.
   */
  static async decompressToString(buffer) {
    try {
      // 1. Try to decompress assuming it's Gzip
      buffer = await this._gunzip(buffer);
    } catch (error) {
      console.log("Decompression failed. Treating data as raw UTF-8 text.");
      // === Fallback Strategy ===
    }
    try {
        const decoder = new TextDecoder();
        return decoder.decode(buffer);
    } catch (e) {
        throw new Error("text decode failed.");
    }
  }

  // ==========================================
  // Advanced Mode: JSON <-> MessagePack <-> Gzip
  // Note: Requires an external MessagePack library (e.g., @msgpack/msgpack)
  // ==========================================

  /**
   * Inject the MessagePack library dependencies.
   * @param {Object} msgpackLib - The imported object from '@msgpack/msgpack'
   */
  static setMsgPackLib(msgpackLib) {
    this.msgpack = msgpackLib;
  }

  /**
   * Pipeline: JSON Object -> MessagePack Binary -> Gzip.
   * 
   * Note: We must JSON.parse the input string first. 
   * If we just MessagePack the string directly, we lose the structural compression benefits.
   * 
   * @param {string} dataObj - The input JSON.
   * @returns {Promise<ArrayBuffer>}
   */
  static async compressMsgPack(dataObj) {
    if (!this.msgpack) throw new Error("Please call setMsgPackLib() first.");

    // 2. Encode Object to MessagePack binary
    const msgpackBytes = this.msgpack.encode(dataObj);

    // 3. Compress the MessagePack binary with Gzip
    return this._gzip(msgpackBytes);
  }

  /**
   * Pipeline: Gzip -> MessagePack Binary -> Object.
   * 
   * @param {ArrayBuffer} buffer - The compressed data.
   * @returns {Promise<Object>} The restored JSON.
   */
  static async decompressMsgPack(buffer) {
    if (!this.msgpack) throw new Error("Please call setMsgPackLib() first.");

    try {
      // 1. Decompress Gzip
      const rawMsgpackBytes = await this._gunzip(buffer);

      // 2. Decode MessagePack to Object
      const obj = this.msgpack.decode(rawMsgpackBytes);

      return obj;
    } catch (error) {
      console.error("Advanced decompression failed", error);
      throw error; 
    }
  }

  // ==========================================
  // Internal Stream Helpers
  // ==========================================

  /**
   * Internal helper to run Gzip compression.
   * @param {Uint8Array} bytes 
   * @returns {Promise<ArrayBuffer>}
   */
  static async _gzip(bytes) {
    const stream = new CompressionStream('gzip');
    const writer = stream.writable.getWriter();
    
    writer.write(bytes);
    writer.close();

    // Use Response to easily convert stream to ArrayBuffer
    return new Response(stream.readable).arrayBuffer();
  }

  /**
   * Internal helper to run Gzip decompression.
   * @param {ArrayBuffer} buffer 
   * @returns {Promise<Uint8Array>}
   */
  static async _gunzip(buffer) {
    const stream = new DecompressionStream('gzip');
    const writer = stream.writable.getWriter();
    
    writer.write(buffer);
    writer.close();

    const resBuffer = await new Response(stream.readable).arrayBuffer();
    return new Uint8Array(resBuffer);
  }
}

// export default BinaryCompressor;