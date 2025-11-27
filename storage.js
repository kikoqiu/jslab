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
            return await file.text();
        } catch (error) {
            // console.error(`Error loading file "${path}":`, error);
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
            if (entry.kind === 'file') {
                files.push(entry.name);
            } else if (entry.kind === 'directory') {
                folders.push(entry.name);
            }
        }
        return { files: files.sort(), folders: folders.sort() };
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

    async setLastFile(path) {
        if (!path) return;
        await this.saveFile('/.lastfile', path);
    }

    async getLastFile() {
        return await this.loadFile('/.lastfile');
    }
}