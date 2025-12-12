class WebDAVSyncer {
    // Define a constant for the backup directory name to ensure consistency.
    static BACKUP_ROOT_NAME = 'backup';
    static RECYCLE_ROOT_NAME = 'recycle';
    static SERVER_MODIFIED_JSON_NAME = 'servermodified.json';

    constructor(vfs, axios, config) {
        this.vfs = vfs;
        this.axios = axios;
        this.config = { ...config };
        if (!this.config.url.endsWith('/')) {
            this.config.url += '/';
        }
        this.serverModifiedTimes = null; // Cache for the sync process
    }

    _getAxiosConfig() {
        return {
            auth: {
                username: this.config.user,
                password: this.config.pass
            }
        };
    }

    _remoteUrl(path) {
        // This function correctly resolves a path relative to the configured base URL.
        // e.g., if config.url is http://ip/dav/, _remoteUrl('/backup/') returns http://ip/dav/backup/
        const cleanPath = ('/' + path).replace(/\/+/g, '/');
        return this.config.url + cleanPath.substring(1).split('/').map(encodeURIComponent).join('/');
    }

    // New helper methods for servermodified.json
    _getServerModifiedJsonPath() {
        return `/${WebDAVSyncer.SERVER_MODIFIED_JSON_NAME}`;
    }

    async _readServerModifiedTimes() {
        const path = this._getServerModifiedJsonPath();
        try {
            const content = await this.vfs.loadFile(path);
            if (content) {
                return JSON.parse(content);
            }
        } catch (e) {
            console.warn(`Could not read or parse ${WebDAVSyncer.SERVER_MODIFIED_JSON_NAME}, starting fresh.`, e);
        }
        return {};
    }

    async _getAllLocalFilePaths(path = '/') {
        const { files, folders } = await this.vfs.listFiles(path);
        let paths = files.map(f => `${path}${f.name}`.replace('//', '/'));

        for (const folder of folders) {
            if (folder === WebDAVSyncer.RECYCLE_ROOT_NAME || folder === WebDAVSyncer.BACKUP_ROOT_NAME) {
                continue;
            }
            const subPaths = await this._getAllLocalFilePaths(`${path}${folder}/`.replace('//', '/'));
            paths = paths.concat(subPaths);
        }
        return paths;
    }

    async _writeServerModifiedTimes(times, dofilter=false) {
        const path = this._getServerModifiedJsonPath();
        let filteredTimes = times;
        if(dofilter) {
            const allLocalFiles = await this._getAllLocalFilePaths();
            const localFileSet = new Set(allLocalFiles);
            
            filteredTimes = {};
            for (const p in times) {
                if (localFileSet.has(p)) {
                    filteredTimes[p] = times[p];
                }
            }
        }
        const content = JSON.stringify(filteredTimes, null, 2);
        await this.vfs.saveFile(path, new TextEncoder('utf-8').encode(content));
    }

    async _updateServerModifiedTime(path, timestamp) {
        if (path.startsWith(`/${WebDAVSyncer.RECYCLE_ROOT_NAME}/`)) return;
        const times = await this._readServerModifiedTimes();
        times[path] = timestamp;
        await this._writeServerModifiedTimes(times);
    }

    async _removeServerModifiedTime(path) {
        if (path.startsWith(`/${WebDAVSyncer.RECYCLE_ROOT_NAME}/`)) return;
        const times = await this._readServerModifiedTimes();
        delete times[path];
        await this._writeServerModifiedTimes(times);
    }
    
    async _getRemoteFileModTime(path) {
        if (path.startsWith(`/${WebDAVSyncer.RECYCLE_ROOT_NAME}/`)) return 0;
        try {
            const response = await this.axios({
                method: 'PROPFIND',
                url: this._remoteUrl(path),
                headers: { 'Depth': '0' },
                ...this._getAxiosConfig()
            });

            const parser = new DOMParser();
            const xmlDoc = parser.parseFromString(response.data, "application/xml");
            const lastModified = xmlDoc.getElementsByTagNameNS("DAV:", "getlastmodified")[0]?.textContent;

            if (lastModified) {
                return new Date(lastModified).getTime();
            }
            return 0;
        } catch (e) {
            if (e.response && e.response.status === 404) return 0; // File doesn't exist
            console.warn(`Failed to get remote mod time for ${path}:`, e);
            return 0; // Or handle error as needed
        }
    }


    // Returns 'collection', 'file', or null
    async _remoteItemType(path) {
        if (path.startsWith(`/${WebDAVSyncer.RECYCLE_ROOT_NAME}/`)) return null;
        try {
            const response = await this.axios({
                method: 'PROPFIND',
                url: this._remoteUrl(path),
                headers: { 'Depth': '0' },
                ...this._getAxiosConfig()
            });
            const parser = new DOMParser();
            const xmlDoc = parser.parseFromString(response.data, "application/xml");
            const resourcetype = xmlDoc.getElementsByTagNameNS("DAV:", "resourcetype")[0];
            if (resourcetype && resourcetype.getElementsByTagNameNS("DAV:", "collection").length > 0) {
                return 'collection';
            }
            const status = xmlDoc.getElementsByTagNameNS("DAV:", "status")[0]?.textContent || '';
            if (status.includes("200 OK")) {
                return 'file';
            }
            return null;
        } catch (e) {
            if (e.response && e.response.status === 404) return null;
            console.warn(`Failed to determine remote item type for ${path}:`, e);
            return null;
        }
    }

    async _backupRemoteFile(path) {
        if (!this.config.backup) return;
        try {
            const now = new Date();
            const yyyymmdd = now.toISOString().slice(0, 10).replace(/-/g, '');
            const timestamp = now.toISOString().replace(/[:.]/g, '-');
            const filename = path.split('/').pop();
            const backupFilename = `${filename}.${timestamp}.bak`;
            // The backup directory path is relative to the sync root.
            const backupDir = `/${WebDAVSyncer.BACKUP_ROOT_NAME}/${yyyymmdd}/`;
            const backupPath = `${backupDir}${backupFilename}`;
            // Create the directories using paths that _remoteUrl will resolve correctly.
            await this.createRemoteDir(`/${WebDAVSyncer.BACKUP_ROOT_NAME}/`);
            await this.createRemoteDir(backupDir);
            await this.moveRemote(path, backupPath, true); // Pass true to skip backup check on the backup move itself
        } catch (e) {
            console.error(`Failed to backup file ${path}:`, e);
        }
    }

    _parseXml(xml, requestUrl) {
        const parser = new DOMParser();
        const xmlDoc = parser.parseFromString(xml, "application/xml");
        const files = new Map();
        const responses = xmlDoc.getElementsByTagNameNS("DAV:", "response");

        const decodedRequestUrl = decodeURIComponent(requestUrl);
        const requestPath = new URL(decodedRequestUrl).pathname;
        const baseRequestPath = requestPath.endsWith('/') ? requestPath : requestPath + '/';

        const backupPathPrefix = new URL(this._remoteUrl(WebDAVSyncer.BACKUP_ROOT_NAME + '/')).pathname;
        const recyclePathPrefix = new URL(this._remoteUrl(WebDAVSyncer.RECYCLE_ROOT_NAME + '/')).pathname;

        for (const res of responses) {
            // 1. Extract and validate basic properties from the XML response
            const href = res.getElementsByTagNameNS("DAV:", "href")[0]?.textContent;
            const propstat = res.getElementsByTagNameNS("DAV:", "propstat")[0];
            if (!href || !propstat) continue;

            const status = propstat.getElementsByTagNameNS("DAV:", "status")[0]?.textContent || '';
            if (!status.includes("200 OK")) continue;

            // 2. Process href, filter out items, and derive the item name
            let hrefPath;
            try {
                hrefPath = new URL(decodeURIComponent(href), decodedRequestUrl).pathname;
            } catch (e) { continue; }

            // Filter out ignored directories
            if (hrefPath.startsWith(backupPathPrefix) || hrefPath.startsWith(recyclePathPrefix)) continue;
            if (hrefPath.endsWith(WebDAVSyncer.SERVER_MODIFIED_JSON_NAME)) continue;


            // Filter out the directory being listed itself
            if (hrefPath === requestPath || hrefPath + '/' === baseRequestPath) {
                continue;
            }

            let name = hrefPath;
            if (name.startsWith(baseRequestPath)) {
                name = name.substring(baseRequestPath.length);
            }
            name = name.replace(/\/$/, '');

            if (!name) continue;

            // 3. Extract final file metadata
            const prop = propstat.getElementsByTagNameNS("DAV:", "prop")[0];
            if (!prop) continue;

            const resourcetype = prop.getElementsByTagNameNS("DAV:", "resourcetype")[0];
            const isCollection = resourcetype && resourcetype.getElementsByTagNameNS("DAV:", "collection").length > 0;
            const lastModified = prop.getElementsByTagNameNS("DAV:", "getlastmodified")[0]?.textContent;
            const size = prop.getElementsByTagNameNS("DAV:", "getcontentlength")[0]?.textContent;

            // 4. Add to file list
            files.set(name, {
                name: name,
                isCollection: isCollection,
                lastModified: lastModified ? new Date(lastModified).getTime() : 0,
                size: isCollection ? 0 : parseInt(size || '0', 10),
            });
        }
        return files;
    }

    async listRemote(path) {
        if (path.startsWith(`/${WebDAVSyncer.RECYCLE_ROOT_NAME}/`) || path.startsWith(`/${WebDAVSyncer.BACKUP_ROOT_NAME}/`)) return new Map();
        const requestUrl = this._remoteUrl(path);
        try {
            const response = await this.axios({
                method: 'PROPFIND', url: requestUrl,
                headers: { 'Depth': '1' }, ...this._getAxiosConfig()
            });
            return this._parseXml(response.data, requestUrl);
        } catch (e) {
            if (e.response && e.response.status === 404) return new Map();
            console.error(`Failed to list remote files at "${path}"`, e);
            throw new Error(`Failed to list remote files: ${e.message}`);
        }
    }

    async downloadFile(remotePath, localPath, remoteModTime, onDownloadedCallback) {
        if (remotePath.startsWith(`/${WebDAVSyncer.RECYCLE_ROOT_NAME}/`)) return;
        try {
            const response = await this.axios.get(this._remoteUrl(remotePath), {
                ...this._getAxiosConfig(),
                responseType: 'arraybuffer'
            });
            await this.vfs.saveFile(localPath, response.data);
            if (remoteModTime) {
                await this._updateServerModifiedTime(localPath, remoteModTime);
            }
            if (onDownloadedCallback) onDownloadedCallback(localPath);
        } catch (e) {
            throw new Error(`Download failed for ${remotePath}: ${e.message}`); 
        }
    }

    async uploadFile(localPath, remotePath, updateModTime = true) {
        if (localPath.startsWith(`/${WebDAVSyncer.RECYCLE_ROOT_NAME}/`)) return;
        const content = await this.vfs.loadFile(localPath);
        if (content === null) return;
        if (this.config.backup && await this._remoteItemType(remotePath) !== null) {
            await this._backupRemoteFile(remotePath);
        }
        try {
            await this.axios.put(this._remoteUrl(remotePath), content, this._getAxiosConfig());
            if (updateModTime) {
                const newModTime = await this._getRemoteFileModTime(remotePath);
                if (newModTime > 0) {
                    await this._updateServerModifiedTime(localPath, newModTime);
                }
            }
        } catch (e) {
            // If upload fails, clear the timestamp to force re-sync next time.
            await this._removeServerModifiedTime(localPath);
            throw new Error(`Upload failed for ${localPath}: ${e.message}`);
        }
    }

    async deleteRemote(path) {
        if (path.startsWith(`/${WebDAVSyncer.RECYCLE_ROOT_NAME}/`)) return;
        if (this.config.backup && await this._remoteItemType(path) !== null) {
            await this._backupRemoteFile(path);
        } else {
            try {
                await this.axios.delete(this._remoteUrl(path), this._getAxiosConfig());
            } catch (e) {
                if (e.response && e.response.status === 404) return; // Not found is fine
                throw new Error(`Remote delete failed for ${path}: ${e.message}`);
            }
        }
        await this._removeServerModifiedTime(path);
    }

    async moveRemote(oldPath, newPath, skipBackup = false) {
        if (oldPath.startsWith(`/${WebDAVSyncer.RECYCLE_ROOT_NAME}/`) || newPath.startsWith(`/${WebDAVSyncer.RECYCLE_ROOT_NAME}/`)) return;
        if (!skipBackup && this.config.backup && await this._remoteItemType(newPath) !== null) {
            await this._backupRemoteFile(newPath);
        }
        try {
            const oldModTime = await this._getRemoteFileModTime(oldPath);
            const destinationHeader = new URL(this._remoteUrl(newPath)).pathname;
            await this.axios({
                method: 'MOVE', url: this._remoteUrl(oldPath),
                headers: { 'Destination': destinationHeader, 'Overwrite': 'T' },
                ...this._getAxiosConfig()
            });

            await this._removeServerModifiedTime(oldPath);
            const newModTime = await this._getRemoteFileModTime(newPath);
            await this._updateServerModifiedTime(newPath, newModTime > 0 ? newModTime : oldModTime);

        } catch (e) {
            throw new Error(`Remote move failed for ${oldPath}: ${e.message}`);
        }
    }

    async createRemoteDir(path) {
        if (path.startsWith(`/${WebDAVSyncer.RECYCLE_ROOT_NAME}/`)) return;
        if (!path.endsWith('/')) path += '/';
        const itemType = await this._remoteItemType(path);
        if (itemType === 'collection') return;
        if (itemType === 'file') throw new Error(`Cannot create directory at ${path}: A file with that name already exists.`);
        try {
            await this.axios({
                method: 'MKCOL', url: this._remoteUrl(path),
                ...this._getAxiosConfig()
            });
        } catch (e) {
            throw new Error(`Creating remote directory ${path} failed: ${e.message}`);
        }
    }

    async syncDir(path, onDownloadedCallback, counters) {
        if (path.startsWith(`/${WebDAVSyncer.RECYCLE_ROOT_NAME}/`) || path.startsWith(`/${WebDAVSyncer.BACKUP_ROOT_NAME}/`)) return;

        const localFilesData = await this.vfs.listFiles(path);
        const localFiles = new Map(localFilesData.files.map(f => [f.name, f]));
        
        const localFolders = localFilesData.folders.filter(f => f !== WebDAVSyncer.RECYCLE_ROOT_NAME);
        localFolders.forEach(f => localFiles.set(f, { name: f, isCollection: true }));

        const remoteFiles = await this.listRemote(path);

        for (const [name, localFile] of localFiles.entries()) {
            const remoteFile = remoteFiles.get(name);
            const localFullPath = `${path}${name}`.replace('//', '/');
            const serverModTime = this.serverModifiedTimes[localFullPath] || 0;

            if (localFile.isCollection) {
                if (!remoteFile) await this.createRemoteDir(localFullPath);
                continue;
            }

            // Server-side deletion: file is local and in cache, but not on server. Move to local recycle bin.
            if (!remoteFile && serverModTime > 0) {
                counters.recycle++;
                if (counters.recycle > 10 && !counters.prompted) {
                    const message = `This sync will recycle more than 10 local files because they were not found on the server. Do you want to proceed?`;
                    counters.prompted = true; // Ask only once
                    if (!confirm(message)) {
                        throw new Error("Sync cancelled by user.");
                    }
                }

                const recyclePath = `/${WebDAVSyncer.RECYCLE_ROOT_NAME}/`;
                await this.vfs.createDirectory(recyclePath);
                
                const timestamp = new Date().toISOString().replace(/[:.]/g, '-');
                const recycledFileName = `${name}.${timestamp}.bak`;
                const newRecycledPath = `${recyclePath}${recycledFileName}`;

                await this.vfs.move(localFullPath, newRecycledPath);
                delete this.serverModifiedTimes[localFullPath]; // Remove from cache
                console.log(`Moved ${localFullPath} to ${newRecycledPath} (recycled)`);
                continue;
            }

            if (!remoteFile) { // Upload new file
                await this.uploadFile(localFullPath, localFullPath);
                this.serverModifiedTimes[localFullPath] = await this._getRemoteFileModTime(localFullPath);
            } else if (!remoteFile.isCollection) { // Upload modified file
                if (localFile.lastModified > serverModTime) {
                    await this.uploadFile(localFullPath, localFullPath);
                    this.serverModifiedTimes[localFullPath] = await this._getRemoteFileModTime(localFullPath);
                }
            }
        }

        for (const [name, remoteFile] of remoteFiles.entries()) {
            const localFile = localFiles.get(name);
            const localFullPath = `${path}${name}`.replace('//', '/');
            const serverModTime = this.serverModifiedTimes[localFullPath] || 0;

            if (!localFile) {
                if (remoteFile.isCollection) {
                    await this.vfs.createDirectory(localFullPath);
                } else {
                    await this.downloadFile(localFullPath, localFullPath, 0/*disable save of server modified time */, onDownloadedCallback);
                    this.serverModifiedTimes[localFullPath] = remoteFile.lastModified;
                }
                continue;
            }

            if (!remoteFile.isCollection && localFile && !localFile.isCollection) {
                if (remoteFile.lastModified > serverModTime) {
                    await this.downloadFile(localFullPath, localFullPath, 0, onDownloadedCallback);
                    this.serverModifiedTimes[localFullPath] = remoteFile.lastModified;
                }
            }
        }
        
        for (const folder of localFolders) {
            await this.syncDir(`${path}${folder}/`.replace('//', '/'), onDownloadedCallback, counters);
        }
    }

    async sync(onDownloadedCallback) {
        if (!this.config || !this.config.url) throw new Error("WebDAV URL is not configured.");
        
        const counters = { recycle: 0, prompted: false };
        this.serverModifiedTimes = await this._readServerModifiedTimes();
        
        try {
            await this.syncDir('/', onDownloadedCallback, counters);
            await this._writeServerModifiedTimes(this.serverModifiedTimes, true); // Filter out deleted files
        } finally {
            this.serverModifiedTimes = null; // Always clear cache
        }
    }
}
