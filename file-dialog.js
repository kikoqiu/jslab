window.FileDialogComponent = {
  props: {
    visible: { type: Boolean, default: false, },
    mode: { type: String, default: 'open', },
    defaultName: { type: String, default: '', },
    vfs: { type: Object, required: true }
  },
  data() {
    return {
      currentPath: '/',
      files: [],
      folders: [],
      fileName: '',
      selectedName: null,
      selectedType: null, // 'file' or 'folder'
    };
  },
  template: `
    <div v-if="visible" class="file-dialog-overlay" @click.self="$emit('close')">
      <div class="file-dialog">
        <div class="file-dialog-header">
          <span>{{ mode === 'open' ? 'Open' : 'Save As' }}</span>
          <button @click="$emit('close')" class="close-btn" title="Close">&times;</button>
        </div>
        <div class="file-dialog-nav">
          <button @click="goUp" :disabled="currentPath === '/'" title="Go up a directory">&uarr; Up</button>
          <span>{{ currentPath }}</span>
        </div>
        <div class="file-dialog-body">
            <div class="file-list">
                <div v-for="folder in folders" :key="currentPath + folder" class="file-list-item folder" :class="{selected: selectedName === folder && selectedType === 'folder'}" @click="select(folder, 'folder')" @dblclick="enterFolder(folder)">
                    <i class="bi bi-folder"></i> {{ folder }}
                </div>
                <div v-for="file in files" :key="currentPath + file" class="file-list-item file" :class="{selected: selectedName === file && selectedType === 'file'}" @click="select(file, 'file')" @dblclick="handleAction">
                    <i class="bi bi-file-earmark-code"></i> {{ file }}
                </div>
            </div>
        </div>
        <div class="file-dialog-footer">
          <div class="file-name-input">
            <label for="filename-input">File name:</label>
            <input type="text" id="filename-input" v-model="fileName" @keyup.enter="handleAction" placeholder="File name">
          </div>
          <div class="footer-buttons">
            <button @click="newFolder" class="secondary" title="New Folder"><i class="bi bi-folder-plus"></i></button>
            <button v-if="mode === 'open' && selectedName" @click="handleDelete" class="danger" title="Delete"><i class="bi bi-trash"></i></button>
            <div class="spacer"></div>
            <button @click="$emit('close')">Cancel</button>
            <button @click="handleAction" class="primary" :disabled="!fileName && mode === 'save'">{{ mode === 'open' ? 'Open' : 'Save' }}</button>
          </div>
        </div>
      </div>
    </div>
  `,
  watch: {
    visible(newVal) {
      if (newVal) {
        this.fetchFiles();
        // Extract only the file name from the full path
        this.fileName = this.defaultName.split('/').pop() || '';
        this.selectedName = null;
        this.selectedType = null;
      }
    },
  },
  methods: {
    async fetchFiles() {
        if (!this.vfs) return;
        const { files, folders } = await this.vfs.listFiles(this.currentPath);
        this.files = files;
        this.folders = folders;
        this.selectedName = null;
        this.selectedType = null;
    },
    goUp() {
        if (this.currentPath === '/') return;
        let parts = this.currentPath.split('/').filter(p => p);
        parts.pop();
        this.currentPath = '/' + (parts.join('/') ? parts.join('/') + '/' : '');
        this.fetchFiles();
    },
    select(name, type) {
        this.selectedName = name;
        this.selectedType = type;
        if (type === 'file') {
            this.fileName = name;
        } else {
             this.fileName = '';
        }
    },
    enterFolder(folderName) {
        if (this.currentPath === '/') {
            this.currentPath = `/${folderName}/`;
        } else {
            this.currentPath += `${folderName}/`;
        }
        this.fetchFiles();
    },
    async newFolder() {
        const folderName = prompt("Enter new folder name:");
        if (folderName && !folderName.includes('/')) {
            const newPath = this.currentPath + folderName;
            await this.vfs.createDirectory(newPath);
            this.fetchFiles();
        } else if (folderName) {
            alert("Folder name cannot contain '/'");
        }
    },
    handleAction() {
      // If a folder is selected in 'open' mode, navigate into it
      if (this.mode === 'open' && this.selectedType === 'folder' && this.selectedName) {
        this.enterFolder(this.selectedName);
        return;
      }
      // If nothing is selected or no filename is provided for saving, do nothing
      if (!this.fileName) {
        alert("Please enter or select a file name.");
        return;
      }
      let fullPath = (this.currentPath + this.fileName).replace(/\/\//g, '/');
      this.$emit(this.mode, fullPath);
      this.$emit('close');
    },
    async handleDelete() {
        if (!this.selectedName) return;
        if (confirm(`Are you sure you want to delete "${this.selectedName}"?`)) {
            let fullPath = (this.currentPath + this.selectedName).replace(/\/\//g, '/');
            if (this.selectedType === 'file') {
                await this.vfs.deleteFile(fullPath);
            } else {
                await this.vfs.deleteDirectory(fullPath);
            }
            this.fetchFiles();
        }
    }
  }
};
