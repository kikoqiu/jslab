class VFS {
    constructor() {
        this.root = null;
    }

    async getRoot() {
        if (this.root) return this.root;
        this.root = await navigator.storage.getDirectory();
        return this.root;
    }

    /**
     * Gets a handle for a directory at a given path.
     * @param {string} path - The directory path, e.g., /d1/d2/
     * @param {object} options - e.g., { create: true }
     * @returns {Promise<FileSystemDirectoryHandle|null>}
     */
    async getDirectoryHandle(path, options = {}) {
        const parts = path.split('/').filter(p => p);
        let currentHandle = await this.getRoot();
        try {
            for (const part of parts) {
                currentHandle = await currentHandle.getDirectoryHandle(part, options);
            }
            return currentHandle;
        } catch (error) {
            console.error(`Error getting directory handle for "${path}":`, error);
            return null;
        }
    }
    
    /**
     * Gets a handle for a file at a given path.
     * @param {string} path - The file path, e.g., /d1/d2/file.txt
     * @param {object} options - e.g., { create: true }
     * @returns {Promise<FileSystemFileHandle|null>}
     */
    async getFileHandle(path, options = {}) {
        const parts = path.split('/').filter(p => p);
        const fileName = parts.pop();
        if (!fileName) return null;

        const dirPath = '/' + parts.join('/');
        const dirHandle = await this.getDirectoryHandle(dirPath, options);
        if (!dirHandle) return null;
        
        try {
            return await dirHandle.getFileHandle(fileName, options);
        } catch(error) {
            console.error(`Error getting file handle for "${path}":`, error);
            return null;
        }
    }

    async saveFile(path, content) {
        try{
            content=await BinaryCompressor.compressString(content);
        } catch (error) {
            console.error(`Error saving file "${path}":`, error);
            return false;
        }

        const fileHandle = await this.getFileHandle(path, { create: true });
        if (!fileHandle) return false;
        try {
            const writable = await fileHandle.createWritable();            
            await writable.write(content);
            await writable.close();
            return true;
        } catch (error) {
            console.error(`Error saving file "${path}":`, error);
            return false;
        }
    }

    async loadFile(path) {
        const fileHandle = await this.getFileHandle(path);
        if (!fileHandle) return null;
        try {
            const file = await fileHandle.getFile();
            // Handle arraybuffer if content is not text, e.g., for WebDAV files.
            // For now, assuming text content.            
            const buffer = await file.arrayBuffer();
            return await BinaryCompressor.decompressToString(buffer);
        } catch (error) {
            console.error(`Error loading file "${path}":`, error);
            return null;
        }
    }

    async listFiles(path) {
        const dirHandle = await this.getDirectoryHandle(path);
        if (!dirHandle) return { files: [], folders: [] };

        const files = [];
        const folders = [];
        
        for await (const entry of dirHandle.values()) {
            if (entry.name.startsWith('.')) continue; // Ignore hidden files
            if (entry.name === WebDAVSyncer.SERVER_MODIFIED_JSON_NAME) continue; // Ignore servermodified.json
            if (entry.kind === 'file') {
                try {
                    const fileHandle = await dirHandle.getFileHandle(entry.name);
                    const file = await fileHandle.getFile();
                    files.push({
                        name: entry.name,
                        size: file.size,
                        lastModified: file.lastModified
                    });
                } catch (e) {
                    console.warn(`Could not get file details for ${entry.name}:`, e);
                    files.push({ name: entry.name, size: 0, lastModified: 0 }); // Fallback
                }
            } else if (entry.kind === 'directory') {
                folders.push(entry.name);
            }
        }
        return { files: files.sort((a,b) => a.name.localeCompare(b.name)), folders: folders.sort() };
    }
    
    async createDirectory(path) {
        const dirHandle = await this.getDirectoryHandle(path, { create: true });
        return !!dirHandle;
    }

    async deleteFile(path) {
        const parts = path.split('/').filter(p => p);
        const fileName = parts.pop();
        const dirPath = '/' + parts.join('/');
        
        const dirHandle = await this.getDirectoryHandle(dirPath);
        if (!dirHandle) return false;

        try {
            await dirHandle.removeEntry(fileName);
            return true;
        } catch (error) {
            console.error(`Error deleting file "${path}":`, error);
            return false;
        }
    }
    
    async deleteDirectory(path) {
        const parts = path.split('/').filter(p => p);
        const dirName = parts.pop();
        const parentPath = '/' + parts.join('/');
        
        const parentHandle = await this.getDirectoryHandle(parentPath);
        if (!parentHandle) return false;

        try {
            await parentHandle.removeEntry(dirName, { recursive: true });
            return true;
        } catch (error) {
            console.error(`Error deleting directory "${path}":`, error);
            return false;
        }
    }

    async getHandle(path) {
        const parts = path.split('/').filter(p => p);
        const name = parts.pop();
        if (!name) return this.getRoot();

        const dirPath = '/' + parts.join('/');
        const dirHandle = await this.getDirectoryHandle(dirPath);
        if (!dirHandle) return null;

        try {
            return await dirHandle.getFileHandle(name);
        } catch {
            try {
                return await dirHandle.getDirectoryHandle(name);
            } catch {
                return null; 
            }
        }
    }

    async move(oldPath, newPath) {
        try {
            const oldHandle = await this.getHandle(oldPath);
            if (!oldHandle) throw new Error("Source item not found.");

            if (oldHandle.kind === 'file') {
                // 1. Read content
                const file = await oldHandle.getFile();
                const content = await file.arrayBuffer();

                // 2. Write to new path
                const newFileHandle = await this.getFileHandle(newPath, { create: true });
                if (!newFileHandle) throw new Error("Could not create destination file handle.");
                
                const writable = await newFileHandle.createWritable();
                await writable.write(content);
                await writable.close();

                // 3. On successful write, remove the old file using its own handle
                try {
                    await oldHandle.remove();
                } catch (e) {}//ignore errors on delete
                return true;

            } else { // Directory
                await this.createDirectory(newPath);
                const oldDirHandle = await this.getDirectoryHandle(oldPath);

                for await (const entry of oldDirHandle.values()) {
                    const entryOldPath = `${oldPath.replace(/\/$/, '')}/${entry.name}`;
                    const entryNewPath = `${newPath.replace(/\/$/, '')}/${entry.name}`;
                    if (!await this.move(entryOldPath, entryNewPath)) {
                        throw new Error(`Failed to move child item: ${entry.name}`);
                    }
                }
                try {
                // After moving all children, remove the now-empty old directory
                await oldDirHandle.remove(); // The handle itself can be removed
                } catch (e) {}//ignore errors on delete
                return true;
            }
        } catch (e) {
            console.error(`Move operation failed for ${oldPath} -> ${newPath}:`, e);
            return false;
        }
    }

    async rename(path, newName) {
        const parts = path.split('/').filter(p => p);
        parts.pop();
        const parentPath = '/' + parts.join('/');
        const newPath = (parentPath === '/' ? '' : parentPath) + '/' + newName;
        return this.move(path, newPath);
    }

    async setLastFile(path) {
        if (!path) return;
        window.localStorage.lastFile = path;
    }

    async getLastFile() {
        return window.localStorage.lastFile;
    }
}