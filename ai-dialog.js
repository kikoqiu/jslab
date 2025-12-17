window.AiDialogComponent = {
  props: {
    visible: { type: Boolean, default: false },
  },
  data() {
    return {
      settingManager: null,
      settings: { // Local copy of settings for the form
        enabled: false,
        apiUrl: '',
        apiKey: '',
        model: '',
        advMode: false,
      },
      usage: { // Separate usage data
        prompt_tokens: 0,
        completion_tokens: 0,
      }
    };
  },
  emits: ['close'],
  template: `
    <div v-if="visible" class="file-dialog-overlay">
      <div class="file-dialog" style="height: auto; max-height: 80%; width: 500px;">
        <div class="file-dialog-header">
          <span>AI Copilot</span>
          <button @click="$emit('close')" class="close-btn" title="Close"><i class="bi-x-lg"></i></button>
        </div>
        <div class="file-dialog-body" style="padding: 20px;">
            <p>Configure settings for the AI-powered code completion.</p>
            <div class="form-group-checkbox">
                <input type="checkbox" id="ai-enabled" v-model="settings.enabled">
                <label for="ai-enabled">Enable AI Copilot</label>
            </div>
            <div class="form-group">
                <label for="ai-api-url">API URL</label>
                <input type="text" id="ai-api-url" v-model="settings.apiUrl" placeholder="e.g., https://api.openai.com/v1/chat/completions">
            </div>
            <div class="form-group">
                <label for="ai-api-key">API Key</label>
                <input type="password" id="ai-api-key" v-model="settings.apiKey">
            </div>
            <div class="form-group">
                <label for="ai-model">Model</label>
                <input type="text" id="ai-model" v-model="settings.model" placeholder="e.g., gpt-3.5-turbo">
            </div>
            <div class="form-group">
                <label for="ai-hotkey">Hotkey</label>
                <input type="text" id="ai-hotkey" value="Alt-i" readonly>
            </div>
            <div class="form-group-checkbox">
                <input type="checkbox" id="ai-advMode" v-model="settings.advMode">
                <label for="ai-advMode">Replace Mode (not suggested)</label>
            </div>
            <div class="usage-stats">
                <p><strong>Session Usage:</strong></p>
                <p>Prompt: {{ usage.prompt_tokens }} tokens, Completion: {{ usage.completion_tokens }} tokens.</p>
            </div>
        </div>
        <div class="file-dialog-footer" style="justify-content: flex-end;">
          <button @click="$emit('close')">Cancel</button>
          <button @click="handleSave" class="primary">Save</button>
        </div>
      </div>
    </div>
  `,
  created() {
    this.settingManager = SettingManager.getInstance();
    this.loadSettings();
  },
  watch: {
    visible(newVal) {
      if (newVal) {
        this.loadSettings();
      }
    }
  },
  methods: {
    loadSettings() {
      const aiSettings = this.settingManager.get('ai');
      if (aiSettings) {
        this.settings.enabled = aiSettings.enabled;
        this.settings.apiUrl = aiSettings.apiUrl;
        this.settings.apiKey = aiSettings.apiKey;
        this.settings.model = aiSettings.model;
        this.settings.advMode = aiSettings.advMode;
        this.usage.prompt_tokens = aiSettings.usage.prompt_tokens || 0;
        this.usage.completion_tokens = aiSettings.usage.completion_tokens || 0;
      }
    },
    handleSave() {
      this.settingManager.update({
        ai: {
          enabled: this.settings.enabled,
          apiUrl: this.settings.apiUrl,
          apiKey: this.settings.apiKey,
          model: this.settings.model,
          advMode: this.settings.advMode,
        }
      });
      this.$emit('close');
      vm.showToast('AI settings saved.'); // Use the global vm instance to show toast
    },
  },
};