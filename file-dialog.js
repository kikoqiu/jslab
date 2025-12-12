window.FileDialogComponent = {
  props: {
    visible: { type: Boolean, default: false },
    mode: { type: String, default: 'open' }, // 'open' or 'save'
    defaultName: { type: String, default: '' },
    vfs: { type: Object, required: true }
  },
  data() {
    return {
      currentPath: '/',
      files: [],
      folders: [],
      fileName: '',
      selectedName: null,
      selectedType: null,
      renamingName: null,
      movingItem: null,
      // WebDAV related state
      showWebdavConfigModal: false,
      webdavConfig: { url: '', user: '', pass: '', backup: true, enabled: false },
      webdavSyncer: null,
      lastSyncTime: 'never'
    };
  },
  emits: ['close', 'open', 'save', 'rename', 'move', 'show-toast', 'reload-file'],
  template: `
    <div v-if="visible" class="file-dialog-overlay" @click.self="cancelActions">
      <div class="file-dialog">
        <div class="file-dialog-header">
          <span>{{ movingItem ? 'Select destination and click Move Here' : 'File Explorer' }}</span>
          <div>
            <button @click="showWebdavConfigModal = true" class="header-btn" title="WebDAV Config"><i class="bi bi-cloud-arrow-up"></i></button>
            <button @click.stop="runSync()" class="header-btn" title="Sync Now"><i class="bi bi-arrow-clockwise"></i></button>
            <span class="sync-status" title="Last successful sync time">({{ lastSyncTime }})</span>
            <button @click="cancelActions" class="close-btn" title="Close">&times;</button>
          </div>
        </div>
        <div class="file-dialog-nav">
          <button @click="goUp" :disabled="currentPath === '/' || movingItem" title="Go up a directory">&uarr; Up</button>
          <span>{{ currentPath }}</span>
        </div>
        <div class="file-dialog-body">
            <div class="file-list-header">
                <span></span><span>Name</span><span>Size</span><span>Modified</span><span>Actions</span>
            </div>
            <div class="file-list">
                <div v-for="folder in folders" :key="currentPath + folder" class="file-list-item folder" :class="{selected: selectedName === folder && selectedType === 'folder'}" @click="select(folder, 'folder')" @dblclick="enterFolder(folder)">
                    <i class="bi bi-folder"></i>
                    <span class="file-name">
                        <input v-if="renamingName === folder" type="text" :value="folder" @blur="finishRename($event, folder, 'folder')" @keydown.enter.prevent="finishRename($event, folder, 'folder')" @keydown.esc.prevent="cancelRename" v-focus>
                        <span v-else>{{ folder }}</span>
                    </span>
                    <span></span><span></span>
                    <div class="actions">
                        <button @click.stop="startRename(folder, 'folder')" title="Rename"><i class="bi bi-pencil-square"></i></button>
                        <button @click.stop="startMove(folder, 'folder')" title="Move"><i class="bi bi-folder-symlink"></i></button>
                        <button @click.stop="handleDelete(folder, 'folder')" class="danger" title="Delete"><i class="bi bi-trash"></i></button>
                    </div>
                </div>
                <div v-for="file in files" :key="currentPath + file.name" class="file-list-item file" :class="{selected: selectedName === file.name && selectedType === 'file'}" @click="select(file.name, 'file')" @dblclick="handleOpen">
                    <i class="bi bi-file-earmark-code"></i>
                    <span class="file-name">
                        <input v-if="renamingName === file.name" type="text" :value="file.name" @blur="finishRename($event, file.name, 'file')" @keydown.enter.prevent="finishRename($event, file.name, 'file')" @keydown.esc.prevent="cancelRename" v-focus>
                        <span v-else>{{ file.name }}</span>
                    </span>
                    <span class="file-size">{{ formatSize(file.size) }}</span>
                    <span class="file-date">{{ formatDate(file.lastModified) }}</span>
                    <div class="actions">
                        <button @click.stop="startRename(file.name, 'file')" title="Rename"><i class="bi bi-pencil-square"></i></button>
                        <button @click.stop="startMove(file.name, 'file')" title="Move"><i class="bi bi-folder-symlink"></i></button>
                        <button @click.stop="handleDelete(file.name, 'file')" class="danger" title="Delete"><i class="bi bi-trash"></i></button>
                    </div>
                </div>
            </div>
        </div>
        <div class="file-dialog-footer">
          <div class="file-name-input" v-if="mode === 'save'">
            <label for="filename-input">File name:</label>
            <input type="text" id="filename-input" v-model="fileName" @keyup.enter="handleSave" placeholder="File name for saving">
          </div>
          <div class="footer-buttons">
            <button @click="newFolder" class="secondary" title="New Folder"><i class="bi bi-folder-plus"></i></button>
            <div class="spacer"></div>
            <button v-if="movingItem" @click="cancelMove" class="secondary">Cancel Move</button>
            <button v-if="movingItem" @click="finishMove" class="primary">Move Here</button>
            <button @click="cancelActions">Cancel</button>
            <button v-if="mode === 'save' && !movingItem" @click="handleSave" class="primary" :disabled="!fileName">Save</button>
            <button v-if="mode === 'open' && !movingItem" @click="handleOpen" class="primary" :disabled="selectedType !== 'file'">Open</button>
          </div>
        </div>
      </div>
    </div>

    <!-- WebDAV Config Modal -->
    <div v-if="showWebdavConfigModal" class="file-dialog-overlay" @click.self="showWebdavConfigModal = false">
      <div class="file-dialog" style="height: auto; max-height: 80%; width: 500px;">
        <div class="file-dialog-header">
          <span>WebDAV Configuration</span>
          <button class="close-btn" @click="showWebdavConfigModal = false"><i class="bi-x-lg"></i></button>
        </div>
        <div class="file-dialog-body" style="padding: 20px;">
            <div class="form-group-checkbox"><input type="checkbox" id="webdav-enabled" v-model="webdavConfig.enabled"><label for="webdav-enabled">Enable WebDAV Sync</label></div>
            <div class="form-group"><label for="webdav-url">Server URL:</label><input type="text" id="webdav-url" v-model="webdavConfig.url" placeholder="https://example.com/webdav/"></div>
            <div class="form-group"><label for="webdav-user">Username:</label><input type="text" id="webdav-user" v-model="webdavConfig.user"></div>
            <div class="form-group"><label for="webdav-pass">Password:</label><input type="password" id="webdav-pass" v-model="webdavConfig.pass"></div>
            <div class="form-group-checkbox"><input type="checkbox" id="webdav-backup" v-model="webdavConfig.backup"><label for="webdav-backup">Backup remote files on delete/modify</label></div>
        </div>
        <div class="file-dialog-footer webdav-config-footer-buttons">
          <button @click="showWebdavConfigModal = false">Cancel</button>
          <button @click="handleSaveWebdavConfig" class="primary">Save and Sync</button>
        </div>
      </div>
    </div>
  `,
  directives: { focus: { mounted(el) { el.focus(); el.select(); } } },
  watch: {
    visible(newVal) {
      if (newVal) {
        this.fetchFiles();
        this.fileName = this.defaultName.split('/').pop() || '';
        this.selectedName = null; this.selectedType = null; this.movingItem = null; this.renamingName = null;
      }
    },
  },
  mounted() { this.loadWebdavConfig(); },
  methods: {
    // --- UI Actions ---
    cancelActions() {
        if (this.movingItem || this.renamingName) { this.cancelMove(); this.cancelRename(); } 
        else { this.$emit('close'); }
    },
    async fetchFiles() {
      if (!this.vfs) return;
      const { files, folders } = await this.vfs.listFiles(this.currentPath);
      this.files = files; this.folders = folders;
    },
    goUp() {
      if (this.currentPath === '/' || this.movingItem) return;
      let parts = this.currentPath.split('/').filter(p => p);
      parts.pop();
      this.currentPath = '/' + (parts.join('/') ? parts.join('/') + '/' : '');
      this.fetchFiles();
    },
    select(name, type) {
        if (this.renamingName) return;
        this.selectedName = name; this.selectedType = type;
        if (type === 'file') this.fileName = name;
    },
    enterFolder(folderName) {
        if (this.movingItem && this.movingItem.name === folderName) return;
        if (this.currentPath === '/') this.currentPath = `/${folderName}/`;
        else this.currentPath += `${folderName}/`;
        this.fetchFiles();
        this.selectedName = null;
    },
    async newFolder() {
        const folderName = prompt("Enter new folder name:");
        if (folderName && !folderName.includes('/')) {
            await this.vfs.createDirectory(this.currentPath + folderName);
            this.fetchFiles();
        } else if (folderName) { alert("Folder name cannot contain '/'"); }
    },

    // --- File Operations (with sync triggers) ---
    async handleDelete(name, type) {
        if (confirm(`Are you sure you want to delete "${name}"?`)) {
            const fullPath = (this.currentPath + name).replace(/\/\//g, '/');
            const success = (type === 'file') ? await this.vfs.deleteFile(fullPath) : await this.vfs.deleteDirectory(fullPath);
            if (success) {
                this.fetchFiles();
                if (this.webdavSyncer) {
                    this.$emit('show-toast', `Syncing...`);
                    this.webdavSyncer.deleteRemote(fullPath).catch(e => this.$emit('show-toast', `Remote delete failed: ${e.message}`));
                }
            } else { this.$emit('show-toast', `Failed to delete ${name}`); }
        }
    },
    startRename(name, type) { this.renamingName = name; this.selectedName = name; this.selectedType = type; },
    cancelRename() { this.renamingName = null; },
    async finishRename(event, oldName, type) {
        const newName = event.target.value;
        this.cancelRename();
        if (newName && newName !== oldName && !newName.includes('/')) {
            const oldPath = (this.currentPath + oldName).replace(/\/\//g, '/');
            const newPath = (this.currentPath + newName).replace(/\/\//g, '/');
            const success = await this.vfs.rename(oldPath, newName);
            if (success) {
                this.fetchFiles();
                this.$emit('rename', { oldPath, newPath }); // Inform parent of rename
                if (this.webdavSyncer) {
                    this.$emit('show-toast', `Renaming ${oldName} on remote...`);
                    this.webdavSyncer.moveRemote(oldPath, newPath).catch(e => this.$emit('show-toast', `Remote rename failed: ${e.message}`));
                }
            } else { this.$emit('show-toast', `Failed to rename ${oldName}`); }
        }
    },
    startMove(name, type) { this.movingItem = { name, type, path: (this.currentPath + name).replace(/\/\//g, '/') }; },
    cancelMove() { this.movingItem = null; },
    async finishMove() {
        if (!this.movingItem) return;
        const oldPath = this.movingItem.path;
        const newPath = (this.currentPath + this.movingItem.name).replace(/\/\//g, '/');
        this.cancelMove();
        if (newPath === oldPath) return;
        const success = await this.vfs.move(oldPath, newPath);
        if (success) {
            this.fetchFiles();
            this.$emit('move', { oldPath, newPath }); // Inform parent of move
            if (this.webdavSyncer) {
                this.$emit('show-toast', `Moving ${oldPath.split('/').pop()} on remote...`);
                this.webdavSyncer.moveRemote(oldPath, newPath).catch(e => this.$emit('show-toast', `Remote move failed: ${e.message}`));
            }
        } else { this.$emit('show-toast', `Failed to move file`); }
    },

    // --- Parent Communication ---
    handleOpen() {
      if (this.mode !== 'open' || this.selectedType !== 'file' || !this.selectedName) return;
      this.$emit('open', (this.currentPath + this.selectedName).replace(/\/\//g, '/'));
      this.$emit('close');
    },
    handleSave() {
        if (this.mode !== 'save' || !this.fileName) return;
        this.$emit('save', (this.currentPath + this.fileName).replace(/\/\//g, '/'));
        this.$emit('close');
    },

    // --- WebDAV Methods ---
    async loadWebdavConfig() {
        const config = localStorage.getItem('webdavConfig');
        if (config) {
            this.webdavConfig = JSON.parse(config);
            this.lastSyncTime = localStorage.getItem('lastSyncTime') || 'never';
            if (this.webdavConfig.url && this.webdavConfig.enabled) {
                this.webdavSyncer = new WebDAVSyncer(this.vfs, axios, this.webdavConfig);
            }
        }
    },
    async handleSaveWebdavConfig() {
        localStorage.setItem('webdavConfig', JSON.stringify(this.webdavConfig));
        if (this.webdavConfig.url && this.webdavConfig.enabled) {
            this.webdavSyncer = new WebDAVSyncer(this.vfs, axios, this.webdavConfig);
        } else {
            this.webdavSyncer = null;
        }
        this.showWebdavConfigModal = false;
        this.runSync();
    },
    runSync() {
        if (!this.webdavSyncer) {
            this.$emit('show-toast', 'WebDAV is not configured.');
            return;
        }
        this.$emit('show-toast', 'Sync started...');

        const onFileDownloaded = (downloadedPath) => {
            if (downloadedPath === this.defaultName) {
                this.$emit('reload-file', downloadedPath);
            }
        };

        this.webdavSyncer.sync(onFileDownloaded).then(() => {
            this.lastSyncTime = new Date().toLocaleString();
            localStorage.setItem('lastSyncTime', this.lastSyncTime);
            this.$emit('show-toast', 'Sync successful!');
            this.fetchFiles(); // Refresh file list after sync
        }).catch(e => {
            this.$emit('show-toast', `Sync failed: ${e.message}`);
            console.error(e);
        });
    },

    uploadSingleFile(filePath) {
        if (!this.webdavSyncer) return; // Fail silently if not configured
        if (!filePath) return;

        this.$emit('show-toast', `Uploading ${filePath.split('/').pop()}...`);
        // Use a timeout to ensure the UI can update before the async operation
        setTimeout(() => {
            this.webdavSyncer.uploadFile(filePath, filePath)
                .then(() => {
                    this.$emit('show-toast', 'Upload successful!');
                })
                .catch(e => {
                    this.$emit('show-toast', `Upload failed: ${e.message}`);
                });
        }, 100);
    },

    // --- Formatting ---
    formatSize(bytes) {
        if (bytes === 0) return '0 B';
        const k = 1024; const sizes = ['B', 'KB', 'MB', 'GB', 'TB'];
        const i = Math.floor(Math.log(bytes) / Math.log(k));
        return parseFloat((bytes / Math.pow(k, i)).toFixed(1)) + ' ' + sizes[i];
    },
    formatDate(timestamp) {
        if (!timestamp) return '';
        return new Date(timestamp).toLocaleString();
    }
  }
};

