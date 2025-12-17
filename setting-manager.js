/**
 * Manages all application settings using localStorage, implemented as a singleton.
 */
class SettingManager {
    constructor() {
        if (SettingManager._instance) {
            throw new Error("SettingManager is a singleton. Use SettingManager.getInstance() to get the instance.");
        }
        this.storageKey = 'jslab_settings'; // Renamed for broader scope
        this.defaults = {
            ai: {
                enabled: false,
                apiUrl: '',
                apiKey: '',
                model: 'gpt-3.5-turbo',
                usage: {
                    prompt_tokens: 0,
                    completion_tokens: 0,
                },
            },
            // Future settings can be added here, e.g.
            // theme: 'light',
        };
        this.settings = this._load();
    }

    /**
     * Gets the singleton instance of the SettingManager.
     * @returns {SettingManager} The singleton instance.
     */
    static getInstance() {
        if (!SettingManager._instance) {
            SettingManager._instance = new SettingManager();
        }
        return SettingManager._instance;
    }

    _load() {
        try {
            const stored = localStorage.getItem(this.storageKey);
            if (stored) {
                const parsed = JSON.parse(stored);
                // Deep merge defaults with parsed settings to ensure structure
                return this._deepMerge(this.defaults, parsed);
            }
        } catch (e) {
            console.error('Failed to load settings from localStorage:', e);
        }
        return JSON.parse(JSON.stringify(this.defaults)); // Deep copy defaults
    }

    _save() {
        try {
            localStorage.setItem(this.storageKey, JSON.stringify(this.settings));
        } catch (e) {
            console.error('Failed to save settings to localStorage:', e);
        }
    }
    
    _isObject(item) {
        return (item && typeof item === 'object' && !Array.isArray(item));
    }

    _deepMerge(target, source) {
        let output = { ...target };
        if (this._isObject(target) && this._isObject(source)) {
            Object.keys(source).forEach(key => {
                if (this._isObject(source[key])) {
                    if (!(key in target))
                        Object.assign(output, { [key]: source[key] });
                    else
                        output[key] = this._deepMerge(target[key], source[key]);
                } else {
                    Object.assign(output, { [key]: source[key] });
                }
            });
        }
        return output;
    }

    /**
     * Get a setting value using a dot-notation path.
     * @param {string} path - The path to the setting (e.g., 'ai.enabled').
     * @returns {*} The value of the setting, or undefined if not found.
     */
    get(path) {
        return path.split('.').reduce((acc, key) => acc && acc[key], this.settings);
    }

    /**
     * Get all settings.
     * @returns {object} All current settings.
     */
    getAll() {
        return JSON.parse(JSON.stringify(this.settings)); // Return a deep copy
    }

    /**
     * Set a setting value using a dot-notation path.
     * @param {string} path - The path to the setting (e.g., 'ai.apiKey').
     * @param {*} value - The new value for the setting.
     */
    set(path, value) {
        const keys = path.split('.');
        const lastKey = keys.pop();
        const mostNested = keys.reduce((acc, key) => acc[key] = acc[key] || {}, this.settings);
        mostNested[lastKey] = value;
        this._save();
    }
    
    /**
     * Updates multiple settings at once at the root level.
     * @param {object} updatedSettings - An object with key-value pairs to update.
     */
    update(updatedSettings) {
        this.settings = this._deepMerge(this.settings, updatedSettings);
        this._save();
    }
}

// Initialize the singleton instance on load
SettingManager.getInstance();