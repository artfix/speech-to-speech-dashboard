// =============================================================
// speech-to-speech dashboard frontend
// Single-file vanilla JS. No build step. Talk to the FastAPI
// server via fetch + websocket. Render forms from the dynamic
// schema served by /api/schema so adding new CLI flags in the
// pipeline just works.
// =============================================================

const $ = (sel, root = document) => root.querySelector(sel);
const $$ = (sel, root = document) => Array.from(root.querySelectorAll(sel));

// ---- Global state ----------------------------------------------------

const state = {
    schema: null,
    settings: {},       // current form values (CLI flag -> value)
    defaults: {},       // CLI flag -> default
    themes: [],
    currentTheme: 'cyberpunk-neon',
    saved: false,       // is there a settings file on disk?
    savedPath: '',
    status: { running: false },
    logFilter: 'ALL',   // ALL | INFO | WARNING | ERROR
    ws: null,
    logIndex: 0,
    pendingRestartForVerbose: false,
    // Curated Qwen3-TTS options loaded from /static/qwen3_models.json.
    // null if the fetch fails (offline / old install) — renderField falls
    // back to the introspected free-text input in that case.
    qwen3Models: null,
    // Curated value lists for free-form `str` fields whose Python
    // annotations don't enumerate allowed values. Same pattern as
    // qwen3Models: loaded from /static/field_choices.json. If null,
    // every free-form `str` field renders as a plain text input.
    fieldChoices: null,
    // Ollama model list, fetched from GET /api/ollama/models. Shape:
    //   { models: [{id: "gpt-oss:20b"}, ...], error: null | "<reason>" }
    // null until the first fetch. The model-name field becomes a <select>
    // when this is populated AND the LLM backend is OpenAI-compat AND the
    // base URL looks like Ollama. Errors fall back to the free-text input
    // silently — we never toast a "could not list models" failure.
    ollamaModels: null,
    // The base URL the Ollama list was fetched against. Used to dedupe
    // re-renders when the user types in the base URL field.
    ollamaBaseUrlAtFetch: '',
    // In-flight Ollama fetch, so the debounce doesn't fire N requests
    // in a row while the user is still typing.
    ollamaFetchInFlight: null,
};

// ---- Utility ---------------------------------------------------------

function el(tag, attrs = {}, children = []) {
    const e = document.createElement(tag);
    for (const [k, v] of Object.entries(attrs)) {
        if (k === 'class') e.className = v;
        else if (k === 'style') Object.assign(e.style, v);
        else if (k === 'dataset') Object.assign(e.dataset, v);
        else if (k.startsWith('on') && typeof v === 'function') {
            e.addEventListener(k.slice(2).toLowerCase(), v);
        } else if (v === true) e.setAttribute(k, '');
        else if (v === false || v == null) { /* skip */ }
        else e.setAttribute(k, v);
    }
    for (const c of [].concat(children)) {
        if (c == null || c === false) continue;
        e.appendChild(typeof c === 'string' ? document.createTextNode(c) : c);
    }
    return e;
}

function toast(message, type = 'info', ttl = 4000) {
    const t = el('div', { class: `toast ${type}` }, message);
    $('#toast-container').appendChild(t);
    setTimeout(() => t.remove(), ttl);
}

function showModal(title, body, actions) {
    $('#modal-title').textContent = title;
    $('#modal-body').textContent = '';
    if (typeof body === 'string') $('#modal-body').textContent = body;
    else $('#modal-body').appendChild(body);
    const acts = $('#modal-actions');
    acts.textContent = '';
    for (const a of actions) {
        acts.appendChild(el('button', {
            class: `btn ${a.kind || ''}`,
            onclick: () => { hideModal(); a.onClick(); }
        }, a.label));
    }
    $('#modal-backdrop').classList.add('visible');
}

function hideModal() { $('#modal-backdrop').classList.remove('visible'); }

$('#modal-backdrop').addEventListener('click', (e) => {
    if (e.target.id === 'modal-backdrop') hideModal();
});

// ---- API client ------------------------------------------------------

async function api(path, opts = {}) {
    const r = await fetch(path, {
        headers: { 'Content-Type': 'application/json' },
        ...opts,
    });
    if (!r.ok) {
        const body = await r.json().catch(() => ({ detail: r.statusText }));
        // Preserve the structured detail (e.g. install_command for the
        // chatterbox-not-installed 409) so callers can show a tailored UI
        // instead of a generic "Start failed: [object Object]" toast.
        const message = typeof body.detail === 'string'
            ? body.detail
            : (body.detail && body.detail.error) || body.detail || r.statusText;
        const err = new Error(message);
        err.status = r.status;
        err.detail = body.detail;
        throw err;
    }
    return r.json();
}

const getJSON = (path) => api(path);
const postJSON = (path, body) => api(path, { method: 'POST', body: JSON.stringify(body) });

// ---- Ollama model discovery (0.3.2+) ----------------------------------
// Mirrors the URL heuristic in web_ui/server.py:_looks_like_ollama_url:
// matches Ollama's default port (11434) or an explicit "/ollama" in the
// path. Anything else (api.openai.com, a custom vLLM endpoint, …) keeps
// the model field as a free-text input so we don't lock out non-Ollama
// users.
function _looksLikeOllamaUrl(url) {
    const s = String(url || '').trim().toLowerCase();
    if (!s) return false;
    return s.includes(':11434') || s.includes('/ollama');
}

// Debounce so a user still typing the URL doesn't fire one fetch per
// keystroke. 400 ms is fast enough to feel instant and slow enough to
// coalesce normal typing bursts.
let _ollamaFetchDebounceTimer = null;
function scheduleOllamaFetch(baseUrl, apiKey) {
    if (_ollamaFetchDebounceTimer) clearTimeout(_ollamaFetchDebounceTimer);
    _ollamaFetchDebounceTimer = setTimeout(() => {
        _ollamaFetchDebounceTimer = null;
        fetchOllamaModels(baseUrl, apiKey);
    }, 400);
}

async function fetchOllamaModels(baseUrl, apiKey) {
    // Don't refetch if the URL hasn't changed since the last successful
    // fetch. The base URL is the only "identity" of an Ollama server we
    // care about — re-keying by model name would be wrong (the user can
    // change the model without re-fetching the list).
    const url = String(baseUrl || '').trim();
    if (!url) {
        state.ollamaModels = null;
        state.ollamaBaseUrlAtFetch = '';
        state.ollamaFetchInFlight = null;
        return;
    }
    if (state.ollamaBaseUrlAtFetch === url && state.ollamaModels && !state.ollamaModels.error) {
        return; // already have a good list for this URL
    }
    // Dedup in-flight fetches for the same URL.
    if (state.ollamaFetchInFlight && state.ollamaFetchInFlight.url === url) {
        return;
    }
    const key = String(apiKey || '').trim();
    const params = new URLSearchParams({ base_url: url });
    if (key) params.set('api_key', key);
    const promise = getJSON('/api/ollama/models?' + params.toString())
        .then((j) => {
            state.ollamaModels = j || { models: [], error: 'no response' };
            state.ollamaBaseUrlAtFetch = url;
        })
        .catch((e) => {
            // Silent fallback — the model field will render as free-text.
            state.ollamaModels = { models: [], error: e.message || 'fetch failed' };
            state.ollamaBaseUrlAtFetch = url;
        })
        .finally(() => {
            if (state.ollamaFetchInFlight && state.ollamaFetchInFlight.url === url) {
                state.ollamaFetchInFlight = null;
            }
            // The initial page-load fetch races the first render: the
            // LLM tab is mounted before this fetch resolves, so
            // renderField sees state.ollamaModels === null and falls
            // back to a free-text input. Re-render the LLM tab once
            // the list is in so the dropdown appears the next time
            // the user visits the LLM tab. We re-render
            // unconditionally (not only when LLM is the active tab) —
            // tabs are rendered ONCE in renderAll() and not re-painted
            // on tab-switch, so we have to refresh the LLM tab here
            // for the dropdown to ever appear, regardless of which
            // tab the user happens to be on when the fetch resolves.
            //
            // We clear the tab's existing children first because
            // renderSettingsTab appends rather than replaces — without
            // this we'd end up with two copies of the LLM tab
            // stacked, and the original (free-text) one would still
            // be at the top.
            if (state.schema && state.ollamaModels && !state.ollamaModels.error
                && Array.isArray(state.ollamaModels.models)
                && state.ollamaModels.models.length > 0) {
                const llmTab = document.getElementById('tab-llm');
                if (llmTab) {
                    llmTab.replaceChildren();
                    renderSettingsTab(llmTab, 'llm');
                    applyDisabledStates();
                }
            }
        });
    state.ollamaFetchInFlight = { url, promise };
    return promise;
}

// ---- Initial load ----------------------------------------------------

async function init() {
    try {
        const [schema, settingsR, themesR, versionR] = await Promise.all([
            getJSON('/api/schema'),
            getJSON('/api/settings'),
            getJSON('/api/themes'),
            getJSON('/api/version'),
        ]);
        // Curated qwen3 model list is best-effort. If the static file is
        // missing (older install / offline), we silently fall back to free-
        // text inputs for the qwen3 subgroup. Don't let it break init.
        try {
            const r = await fetch('/static/qwen3_models.json', { cache: 'no-cache' });
            if (r.ok) {
                const j = await r.json();
                if (j && Array.isArray(j.models) && j.models.length) {
                    state.qwen3Models = j;
                }
            }
        } catch (_) { /* offline / missing — fallback */ }
        // field_choices.json is the smaller, flat sibling of qwen3_models.json.
        // It enumerates valid values for free-form `str` fields (device, dtype,
        // attention implementation, voice name, language code) so we render
        // a <select> instead of a text input. Best-effort — if the file is
        // missing we silently fall back.
        try {
            const r = await fetch('/static/field_choices.json', { cache: 'no-cache' });
            if (r.ok) {
                const j = await r.json();
                if (j && j.choices && typeof j.choices === 'object') {
                    state.fieldChoices = j.choices;
                }
            }
        } catch (_) { /* offline / missing — fallback */ }
        // Ollama model list — only fetched if the user has configured a
        // base URL that looks like Ollama. Best-effort, silent fallback to
        // free-text on any failure (offline, network, 4xx, 5xx).
        // MUST run AFTER state.settings is assigned below — otherwise the
        // URL is read as undefined and the fetch never fires. (Fix 0.3.2.)
        state.schema = schema;
        state.defaults = settingsR.settings;
        state.settings = { ...settingsR.settings };
        state.saved = settingsR.saved;
        state.savedPath = settingsR.path;
        state.themes = themesR.themes;
        state.version = versionR.version;
        const initialBaseUrl = String(state.settings['--responses-api-base-url'] || '');
        if (initialBaseUrl) {
            fetchOllamaModels(initialBaseUrl, state.settings['--responses-api-api-key']);
        }
        state.currentTheme = state.settings.theme || 'cyberpunk-neon';
        applyTheme(state.currentTheme);
        const verEl = document.getElementById('app-version');
        if (verEl && state.version) verEl.textContent = 'v' + state.version;
        document.title = `speech-to-speech dashboard v${state.version || ''}`.trim();
        buildThemeSelect();
        buildNavAndTabs();
        renderAll();
        startLogStream();
        startStatusPoll();
    } catch (e) {
        toast('Failed to initialize: ' + e.message, 'error', 8000);
        console.error(e);
    }
}

function applyTheme(name) {
    state.currentTheme = name;
    $('#theme-stylesheet').href = `/static/themes/${name}.css`;
    state.settings.theme = name;
}

async function onThemeChange(name) {
    applyTheme(name);
    // Auto-persist so the choice survives reload / dashboard restart.
    // Other fields are not touched -- the Settings tab still owns the
    // explicit "Save Settings" flow.
    try {
        await postJSON('/api/settings/patch', { theme: name });
        state.saved = true;
    } catch (e) {
        toast('Could not save theme: ' + e.message, 'error', 4000);
    }
}

function buildThemeSelect() {
    const sel = $('#theme-select');
    sel.textContent = '';
    for (const t of state.themes) {
        const opt = el('option', { value: t }, t);
        if (t === state.currentTheme) opt.selected = true;
        sel.appendChild(opt);
    }
    sel.onchange = () => onThemeChange(sel.value);
}

// ---- Sidebar + tabs --------------------------------------------------

const TAB_DEFS = [
    { id: 'mode', label: 'Mode', icon: '~' },
    { id: 'vad', label: 'VAD', icon: 'V' },
    { id: 'stt', label: 'STT', icon: 'S' },
    { id: 'llm', label: 'LLM', icon: 'L' },
    { id: 'tts', label: 'TTS', icon: 'T' },
    { id: 'advanced', label: 'Advanced', icon: '*' },
    { id: 'status', label: 'Status & Logs', icon: '#' },
    { id: 'guide', label: 'Guide', icon: '?' },
    { id: 'settings', label: 'Settings', icon: '$' },
    { id: 'control', label: 'Control', icon: '!' },
];

function buildNavAndTabs() {
    const nav = $('#sidebar');
    nav.textContent = '';
    for (const t of TAB_DEFS) {
        const item = el('div', {
            class: 'nav-item',
            dataset: { tab: t.id },
            onclick: () => activateTab(t.id),
        }, [
            el('span', { class: 'nav-icon' }, t.icon),
            el('span', {}, t.label),
        ]);
        nav.appendChild(item);
    }

    const main = $('#main');
    main.textContent = '';
    for (const t of TAB_DEFS) {
        main.appendChild(el('section', { class: 'tab', id: `tab-${t.id}`, dataset: { tab: t.id } }));
    }
    activateTab('mode');
}

function activateTab(id) {
    $$('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.tab === id));
    $$('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === id));
}

// ---- Form rendering --------------------------------------------------

// Map a GPU compat report to a small badge shown above the voice library.
// The badge is only visible when the user picked chatterbox + a GPU device;
// for everything else it stays empty. There is no "run install" button:
// the dashboard auto-installs the matching torch wheel when the user
// clicks Start, so the badge only shows status, not actions.
async function renderGpuBadge(container) {
    container.textContent = '';
    const tts = _settingValue("tts");
    if (tts !== "chatterbox") {
        container.style.display = "none";
        return;
    }
    const device = (_settingValue("chatterbox_device") || "auto").toLowerCase();
    if (device !== "cuda" && device !== "auto") {
        container.style.display = "none";
        return;
    }
    container.style.display = "block";

    let report;
    try {
        const r = await fetch("/api/gpu/check");
        report = await r.json();
    } catch (e) {
        container.appendChild(
            el("div", { class: "gpu-badge-row" }, [
                el("span", { class: "gpu-badge-icon gpu-badge-unknown" }, "?"),
                el("span", { class: "gpu-badge-text" }, "GPU status unknown"),
            ]),
        );
        return;
    }

    if (!report.has_gpu) {
        container.appendChild(
            el("div", { class: "gpu-badge-row" }, [
                el("span", { class: "gpu-badge-icon gpu-badge-info" }, "i"),
                el("span", { class: "gpu-badge-text" }, "No GPU detected — pipeline will run on CPU"),
            ]),
        );
        return;
    }

    if (report.supported) {
        container.appendChild(
            el("div", { class: "gpu-badge-row" }, [
                el("span", { class: "gpu-badge-icon gpu-badge-ok" }, "✓"),
                el("span", { class: "gpu-badge-text" },
                    `GPU ready: ${report.gpu_name} (CC ${report.gpu_cc})`),
            ]),
        );
        return;
    }

    if (report.recommend_cpu) {
        container.appendChild(
            el("div", { class: "gpu-badge-row" }, [
                el("span", { class: "gpu-badge-icon gpu-badge-warning" }, "!"),
                el("span", { class: "gpu-badge-text" },
                    `${report.gpu_name} (CC ${report.gpu_cc}) has no torch wheel — pipeline will run on CPU`),
            ]),
        );
        return;
    }

    // Installed torch doesn't support this GPU. The dashboard auto-installs
    // the matching wheel when the user clicks Start, so we just show a
    // status pill. No "install" button — the dashboard handles it.
    container.appendChild(
        el("div", { class: "gpu-badge-row" }, [
            el("span", { class: "gpu-badge-icon gpu-badge-pending" }, "…"),
            el("span", { class: "gpu-badge-text" },
                `Will install torch ${report.suggested_wheel.torch_version} (${report.suggested_wheel.cuda_tag}) for ${report.gpu_name} (CC ${report.gpu_cc}) when you click Start`),
        ]),
    );
}

function renderAll() {
    for (const t of TAB_DEFS) {
        const tab = $(`#tab-${t.id}`);
        tab.textContent = '';
        if (['mode', 'vad', 'stt', 'llm', 'tts', 'advanced'].includes(t.id)) {
            renderSettingsTab(tab, t.id);
        } else if (t.id === 'status') {
            renderStatusTab(tab);
        } else if (t.id === 'guide') {
            renderGuideTab(tab);
        } else if (t.id === 'settings') {
            renderSettingsFileTab(tab);
        } else if (t.id === 'control') {
            renderControlTab(tab);
        }
    }
    applyDisabledStates();
}

function getGroup(id) {
    return state.schema.groups.find(g => g.id === id);
}

function renderSettingsTab(tab, groupId) {
    const group = getGroup(groupId);
    if (!group) return;
    tab.appendChild(el('div', { class: 'tab-header' }, [
        el('h1', { class: 'tab-title' }, group.title),
        el('div', { class: 'tab-subtitle' }, group.description),
    ]));
    // 0.3.2+: temporary diagnostic badge on the LLM tab so we can see
    // at a glance whether the Ollama model fetch has landed. Will be
    // removed once the dropdown is reliably rendering. (Diagnostic, no
    // behaviour change.)
    const grid = el('div', { class: 'form-grid' });
    for (const f of group.fields) {
        grid.appendChild(renderField(f, group.title));
    }
    tab.appendChild(grid);
    for (const sub of (group.subgroups || [])) {
        const subEl = el('div', { class: 'subgroup', dataset: { subgroup: sub.id } });
        subEl.appendChild(el('div', { class: 'sub-title subgroup-title' }, sub.title));
        const subGrid = el('div', { class: 'form-grid' });
        for (const f of sub.fields) {
            subGrid.appendChild(renderField(f, sub.title));
        }
        subEl.appendChild(subGrid);
        tab.appendChild(subEl);
    }
    // Dashboard-only keepalive dropdown for Ollama / vLLM / llama.cpp. The
    // setting is NOT introspected from the pipeline's argument dataclasses
    // (upstream has no such flag — CLAUDE.md forbids editing src/), so we
    // hand-roll a single dropdown inside the LLM tab's grid instead of
    // going through the full renderField() path.
    if (groupId === 'llm') {
        const grid2 = tab.querySelector('.form-grid');
        if (grid2) {
            grid2.appendChild(renderKeepaliveField());
            // 0.3.2+: dashboard-only max-wait for the one-shot Ollama
            // warmup. Default 60 s; users on a slow LAN loading a 70 B
            // model can bump it. Same hand-rendered pattern as
            // --llm-keepalive (it's not in the introspected schema).
            grid2.appendChild(renderOllamaLoadTimeoutField());
        }
    }
    // The voice library lives inside the TTS tab. It only renders when the
    // user has selected the chatterbox backend, so it sits below the
    // subgroup list and re-renders whenever the form is re-rendered or the
    // voice library mutates.
    if (groupId === 'tts') {
        // GPU status badge: shows compatibility for the user's selected
        // chatterbox device. Hidden when the user picked a non-GPU TTS.
        const gpuBadge = el('div', { id: 'gpu-status-badge', class: 'gpu-badge' });
        tab.appendChild(gpuBadge);
        const refreshGpuBadge = () => renderGpuBadge(gpuBadge);
        refreshGpuBadge();
        // Re-check when the chatterbox device changes; the user might
        // have switched from "auto" to "cpu" or vice versa.
        const chatterboxDevice = tab.querySelector('#f---chatterbox-device');
        if (chatterboxDevice) {
            chatterboxDevice.addEventListener('change', refreshGpuBadge);
        }
        // Also re-check whenever the TTS backend changes (the badge is
        // only relevant for chatterbox).
        const ttsSelect = tab.querySelector('#f---tts');
        if (ttsSelect) {
            ttsSelect.addEventListener('change', () => {
                refreshGpuBadge();
                rerenderLibrary();
            });
        }

        const mount = el('div', { id: 'voice-library-mount' });
        tab.appendChild(mount);
        // Two voice-library UIs coexist in this mount: the chatterbox
        // voice library hides itself when --tts != "chatterbox"; the
        // qwen3 voice library does the inverse. Calling both is
        // idempotent because each one early-returns + hides the
        // container when the other backend is selected. We pass the
        // same DOM node so each backend fully owns the mount when it's
        // active (the other's render() no-ops).
        const rerenderLibrary = () => {
            renderVoiceLibrary(mount, state.settings, () => renderAll());
            if (typeof window.renderQwen3VoiceLibrary === 'function') {
                window.renderQwen3VoiceLibrary(mount, state.settings, () => renderAll());
            }
        };
        // Initial paint
        rerenderLibrary();
    }
    updateSubgroupVisibility();
}

function renderField(f, parentTitle) {
    const fieldId = `f-${f.flag}`;
    // Build a short hover-tooltip preview from the full help text. Native
    // ``title=`` tooltips don't reflow, so we trim to ~220 chars and add an
    // ellipsis. Clicking the ``?`` still toggles the full inline help div.
    const hoverPreview = f.help
        ? (f.help.length > 220 ? f.help.slice(0, 217) + '…' : f.help)
        : 'Show help';

    const label = el('label', { class: 'field-label', for: fieldId }, [
        f.flag,
        el('span', { class: 'field-flag' }, ''),
        el('button', {
            type: 'button',
            class: 'help-btn',
            title: hoverPreview,
            // Scope the toggle to the button's own ``.field`` wrapper, not the
            // whole document. The same flag name (e.g. ``--model-name``) lives
            // in multiple subgroups (``responses-api``, ``chat-completions``,
            // ``transformers``, ``mlx-lm``), which means duplicate IDs in the
            // DOM. ``document.querySelector`` would always pick the first
            // match — frequently a hidden subgroup — and the click would
            // appear to do nothing. Looking up relative to the clicked button
            // guarantees we toggle the help div the user can actually see.
            onclick: (e) => {
                e.preventDefault();
                const field = e.currentTarget.closest('.field');
                const help = field && field.querySelector('.field-help');
                if (help) help.classList.toggle('visible');
            }
        }, '?'),
    ]);

    let input;
    const val = state.settings[f.flag];
    // ---- Ollama model dropdown (0.3.2+) --------------------------------
    // The --model-name field appears once per LLM-backend subgroup. For
    // the chat-completions / responses-api subgroups, when the base URL
    // looks like Ollama AND the Ollama model list is populated, render
    // a <select> instead of a free-text input. The Custom… option
    // (mirrors the qwen3 / tryCuratedSelect pattern) lets the user type
    // any model name not in the list, so saved settings on remote
    // Ollama servers without a known list still round-trip correctly.
    if (
        f.flag === '--model-name'
        && f.ui !== 'checkbox'
        && _looksLikeOllamaUrl(state.settings['--responses-api-base-url'])
        && state.ollamaModels
        && Array.isArray(state.ollamaModels.models)
        && state.ollamaModels.models.length > 0
        && !state.ollamaModels.error
    ) {
        const models = state.ollamaModels.models;
        const curVal = val == null ? '' : String(val);
        const isCurated = models.some((m) => m.id === curVal);
        const ollamaCustomSentinel = '__ollama_custom__';
        const sel = el('select', {
            class: 'field-select',
            id: fieldId,
            onchange: (e) => {
                const v = e.target.value;
                if (v === ollamaCustomSentinel) {
                    // Swap to free-text — same escape-hatch the qwen3
                    // picker uses. The Custom option is a UI affordance,
                    // not a stored value: the user's typed text is what
                    // ends up in state.settings and in the settings file.
                    const freeText = el('input', {
                        class: 'field-input',
                        id: fieldId,
                        type: 'text',
                        placeholder: 'model-name',
                        oninput: (ev) => {
                            state.settings[f.flag] = ev.target.value;
                            applyDisabledStates();
                        },
                    });
                    // Pre-fill with the current curated value so the user
                    // can tweak it; if the saved value was already custom,
                    // keep it.
                    freeText.value = isCurated ? curVal : (curVal || '');
                    sel.replaceWith(freeText);
                    freeText.focus();
                    return;
                }
                state.settings[f.flag] = v;
                applyDisabledStates();
            },
        });
        for (const m of models) {
            // Ollama's /v1/models returns {"id": "gpt-oss:20b", ...} —
            // the id IS the model name. No separate label needed.
            sel.appendChild(el('option', { value: m.id }, m.id));
        }
        sel.appendChild(el('option', { value: ollamaCustomSentinel }, 'Custom (type your own)'));
        if (isCurated) {
            sel.value = curVal;
        } else if (curVal) {
            // Saved value isn't in the current Ollama list — preserve it
            // by defaulting to Custom so the user sees their value rather
            // than us silently clamping to the first model.
            sel.value = ollamaCustomSentinel;
        }
        // Fall through to the standard field-mount path so ``label``,
        // ``help`` and the ``.field`` wrapper are all built the same way
        // as every other field. Same pattern as the qwen3 branch above.
        input = sel;
    }
    // Curated dropdown for fields whose Python annotation is plain `str`
    // (so the schema can't introspect allowed values) but we have a known
    // short whitelist. Loaded from /static/field_choices.json at startup;
    // silently falls back to free-text when the JSON or the per-flag entry
    // is missing, so a missing choices file never blocks rendering.
    const curatedChoices = state.fieldChoices && state.fieldChoices[f.flag];
    const customSentinel = '__custom__';
    // Helper to build the <select> for the curated dropdown, sharing the
    // exact shape of the qwen3-model picker below (same __custom__
    // escape-hatch). Returns the input element or null if the curated
    // path doesn't apply for this field.
    function tryCuratedSelect() {
        if (!curatedChoices || !Array.isArray(curatedChoices) || curatedChoices.length === 0) {
            return null;
        }
        if (f.ui !== 'text') {
            // For Literal-typed fields (--qwen3-tts-backend,
            // --qwen3-tts-mlx-quantization, --chatterbox-model-variant) the
            // schema already provides f.choices; we don't override that.
            return null;
        }
        const curVal = val == null ? '' : String(val);
        const isCurated = curatedChoices.some((c) => String(c) === curVal);
        const sel = el('select', {
            class: 'field-select',
            id: fieldId,
            onchange: (e) => {
                const v = e.target.value;
                if (v === customSentinel) {
                    // Swap this select out for a free-text input so the user
                    // can type any value not in the curated list. The current
                    // (now lost) selection is preserved only by intent — the
                    // user types a fresh value.
                    const freeText = el('input', {
                        class: 'field-input',
                        id: fieldId,
                        type: 'text',
                        placeholder: curatedChoices[0],
                        oninput: (ev) => {
                            state.settings[f.flag] = ev.target.value;
                            applyDisabledStates();
                        },
                    });
                    freeText.value = curVal && !isCurated ? curVal : '';
                    sel.replaceWith(freeText);
                    freeText.focus();
                    return;
                }
                state.settings[f.flag] = v;
                applyDisabledStates();
            },
        });
        for (const c of curatedChoices) {
            sel.appendChild(el('option', { value: String(c) }, String(c)));
        }
        sel.appendChild(el('option', { value: customSentinel }, 'Custom (type your own)'));
        if (isCurated) {
            sel.value = curVal;
        } else if (curVal) {
            // The current value isn't in the curated list -- switch to
            // Custom so the user clearly sees this is non-standard, and
            // they can save without us silently clamping.
            sel.value = customSentinel;
        }
        return sel;
    }
    // Curated Qwen3-TTS model picker: when the curated JSON is loaded and
    // this is the qwen3 model-name field, render a <select> of the upstream-
    // supported model variants instead of a free-text input. Users can still
    // pick "Custom…" to type any other HF Hub ID. This is purely additive —
    // if state.qwen3Models is null, fall through to the standard text input
    // below.
    if (
        f.flag === '--qwen3-tts-model-name'
        && state.qwen3Models
        && f.ui !== 'checkbox'
    ) {
        const models = state.qwen3Models.models || [];
        const curVal = val == null ? '' : String(val);
        const isCurated = models.some((m) => m.id === curVal);
        const sel = el('select', {
            class: 'field-select',
            id: fieldId,
            onchange: (e) => {
                const v = e.target.value;
                if (v === '__custom__') {
                    // Swap this select out for a free-text input so the user
                    // can type any HF Hub ID. Preserve the current curated
                    // selection as the input's starting value.
                    const freeText = el('input', {
                        class: 'field-input',
                        id: fieldId,
                        type: 'text',
                        placeholder: 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
                        oninput: (ev) => {
                            state.settings[f.flag] = ev.target.value;
                            applyDisabledStates();
                        },
                    });
                    freeText.value = '';
                    sel.replaceWith(freeText);
                    freeText.focus();
                    return;
                }
                state.settings[f.flag] = v;
                applyDisabledStates();
                updateQwen3RequiredHints();
            },
        });
        for (const m of models) {
            sel.appendChild(el('option', { value: m.id }, m.label));
        }
        sel.appendChild(el('option', { value: '__custom__' }, 'Custom (type your own HF Hub ID)'));
        // Decide which option starts selected.
        if (isCurated) {
            sel.value = curVal;
        } else if (curVal) {
            // Value not in the curated list — preserve it by selecting Custom.
            // To survive the reload we stash it on the select via a data attr
            // and the change handler treats __custom__ as a switch-to-textbox
            // signal, not a value.
            sel.value = '__custom__';
        }
        // When the user picks a different model, the voice-library
        // visibility rules change (e.g. Reference voices section only
        // appears for Base models). Trigger a library re-render.
        sel.addEventListener('change', () => {
            if (typeof window.renderQwen3VoiceLibrary === 'function') {
                const mount = document.getElementById('voice-library-mount');
                if (mount) window.renderQwen3VoiceLibrary(mount, state.settings, () => renderAll());
            }
        });
        input = sel;
    } else if (f.ui === 'checkbox') {
        input = el('label', { class: 'field-checkbox' }, [
            el('input', {
                type: 'checkbox',
                id: fieldId,
                checked: val === true,
                onchange: (e) => {
                    state.settings[f.flag] = e.target.checked;
                    applyDisabledStates();
                },
            }),
            el('span', { class: 'text-dim' }, 'enable'),
        ]);
    } else if (f.ui === 'select') {
        const sel = el('select', {
            class: 'field-select',
            id: fieldId,
            onchange: (e) => {
                state.settings[f.flag] = e.target.value;
                if (['--stt', '--llm-backend', '--tts'].includes(f.flag)) {
                    updateSubgroupVisibility();
                }
                applyDisabledStates();
            }
        });
        // Add a blank option for Optional / None
        if (f.type === 'optional_string' || f.type === 'enum' && (val == null || val === '')) {
            sel.appendChild(el('option', { value: '' }, '(not set)'));
        }
        for (const c of (f.choices || [])) {
            const opt = el('option', { value: c }, c);
            if (String(c) === String(val)) opt.selected = true;
            sel.appendChild(opt);
        }
        input = sel;
    } else if (f.ui === 'textarea') {
        input = el('textarea', {
            class: 'field-textarea',
            id: fieldId,
            oninput: (e) => {
                state.settings[f.flag] = e.target.value;
                applyDisabledStates();
            },
        });
        input.value = val == null ? '' : String(val);
    } else {
        // Generic path: a free-form text/number input by default, but
        // swap in a curated <select> for fields whose values are known
        // short whitelists (device, dtype, attention implementation,
        // voice name, language code). See web_ui/static/field_choices.json
        // for the source of truth; missing entries fall through to the
        // standard text input silently.
        // If an earlier branch (e.g. the Ollama model dropdown at L601)
        // already produced the input element, keep it. Without this guard
        // the Ollama <select> built above would be silently discarded and
        // replaced by a brand-new <input type="text">, defeating the
        // purpose of the dropdown.
        const curated = input ? null : tryCuratedSelect();
        if (curated) {
            input = curated;
        } else if (!input) {
            const isOllamaBaseUrl = f.flag === '--responses-api-base-url';
            input = el('input', {
                class: 'field-input',
                id: fieldId,
                type: f.ui === 'number' ? 'number' : 'text',
                step: f.ui === 'number' && f.type === 'float' ? 'any' : null,
                oninput: (e) => {
                    const v = e.target.value;
                    if (f.ui === 'number') {
                        state.settings[f.flag] = f.type === 'float' ? parseFloat(v) : parseInt(v, 10);
                    } else {
                        state.settings[f.flag] = v;
                    }
                    applyDisabledStates();
                    // 0.3.2+: when the user edits the Ollama base URL,
                    // debounce-refresh the model dropdown so it tracks
                    // whatever server they typed. Re-render so the
                    // --model-name field swaps in/out of dropdown mode
                    // based on whether the new URL looks like Ollama.
                    if (isOllamaBaseUrl) {
                        if (_looksLikeOllamaUrl(v)) {
                            scheduleOllamaFetch(v, state.settings['--responses-api-api-key']);
                        } else {
                            // Non-Ollama URL — clear the cache so the
                            // dropdown disappears on the next render.
                            state.ollamaModels = null;
                            state.ollamaBaseUrlAtFetch = '';
                        }
                        // Re-render the active settings tab so the
                        // model field re-evaluates the dropdown
                        // decision. Wait one tick so the debounced
                        // Ollama fetch (400 ms) has time to land;
                        // the second pass will pick up the populated
                        // ollamaModels state. We only re-render the
                        // visible subgroup to keep input focus +
                        // scroll position elsewhere. Clear the tab's
                        // existing children first so we don't double-
                        // render.
                        const activeTab = $('.tab.active');
                        if (activeTab && activeTab.dataset && activeTab.dataset.tab) {
                            const tabId = activeTab.dataset.tab;
                            setTimeout(() => {
                                if (tabId === 'llm') {
                                    activeTab.replaceChildren();
                                    renderSettingsTab(activeTab, 'llm');
                                    applyDisabledStates();
                                }
                            }, 500);
                        }
                    }
                }
            });
            input.value = val == null ? '' : String(val);
        }
    }

    const help = el('div', { class: 'field-help', id: `help-${f.flag}` }, f.help || '(no help text)');

    // For the qwen3 reference-audio field, attach an inline "Upload…" button
    // so the user can pick a file from disk and have it saved to voices/
    // automatically. The returned absolute path fills the input.
    let refAudioRow = null;
    if (f.flag === '--qwen3-tts-ref-audio') {
        const fileInput = el('input', {
            type: 'file',
            accept: 'audio/*',
            style: { display: 'none' },
            onchange: async (e) => {
                const file = e.target.files && e.target.files[0];
                if (!file) return;
                const fd = new FormData();
                fd.append('audio', file);
                try {
                    toast('Uploading reference audio…', 'info', 5000);
                    const r = await fetch('/api/qwen3_ref_audio', { method: 'POST', body: fd });
                    if (!r.ok) {
                        const detail = await r.json().catch(() => ({}));
                        toast('Upload failed: ' + (detail.detail || r.statusText), 'error', 6000);
                        return;
                    }
                    const j = await r.json();
                    state.settings[f.flag] = j.path;
                    input.value = j.path;
                    toast('Reference audio saved.', 'success');
                    applyDisabledStates();
                } catch (err) {
                    toast('Upload failed: ' + err.message, 'error', 6000);
                }
            },
        });
        const uploadBtn = el('button', {
            type: 'button',
            class: 'btn btn-small',
            style: { marginTop: '6px' },
            onclick: () => fileInput.click(),
        }, 'Upload reference audio…');
        refAudioRow = el('div', { class: 'btn-row', style: { marginTop: '4px' } }, [fileInput, uploadBtn]);
    }

    const wrapChildren = refAudioRow ? [label, input, refAudioRow, help] : [label, input, help];
    const wrap = el('div', { class: 'field' }, wrapChildren);
    // Optional String fields benefit from full width since they may be long.
    if (f.ui === 'textarea' || f.type === 'optional_string') {
        wrap.classList.add('full');
    }
    return wrap;
}

function _settingValue(fieldName) {
    // ``state.settings`` is keyed by CLI flag (``--llm-backend``, ``--tts``...)
    // but the schema uses Python field names (``llm_backend``, ``tts``). Resolve
    // either form so the visibility/disabled lookups don't silently miss.
    if (fieldName in state.settings) return state.settings[fieldName];
    const flag = '--' + String(fieldName).replace(/_/g, '-');
    return state.settings[flag];
}

function updateSubgroupVisibility() {
    for (const group of state.schema.groups) {
        if (!group.subgroups) continue;
        for (const sub of group.subgroups) {
            const sel = _settingValue(sub.visible_when.field);
            const show = sel === sub.visible_when.equals;
            const el = $(`[data-subgroup="${sub.id}"]`);
            if (el) el.style.display = show ? '' : 'none';
        }
    }
}

// Apply the `disabled_when` rules attached to chatterbox TTS fields. Each
// rule has the shape { field, in: [<values>] } and means "gray out this
// field's input when <field>'s current value is one of <in>". The form
// values are read from `state.settings`. We do not re-render the form on
// every change; we just toggle the `.field-disabled` class on each field's
// wrapper, which CSS uses to lower opacity and disable the input.
function applyDisabledStates() {
    for (const group of state.schema.groups) {
        for (const f of [...(group.fields || []), ...((group.subgroups || []).flatMap(s => s.fields || []))]) {
            if (!f.disabled_when) continue;
            const watched = _settingValue(f.disabled_when.field);
            const shouldDisable = (f.disabled_when.in || []).includes(watched);
            const wrap = $(`#f-${f.flag}`)?.closest('.field');
            if (!wrap) continue;
            wrap.classList.toggle('field-disabled', shouldDisable);
            const input = wrap.querySelector('input, select, textarea');
            if (input) input.disabled = shouldDisable;
        }
    }
    updateQwen3RequiredHints();
}

// Mark the qwen3 ref_audio / ref_text / instruct fields with a small
// "required for this model" hint when the currently-selected model variant
// needs them. Driven by the curated JSON in state.qwen3Models.
function updateQwen3RequiredHints() {
    if (!state.qwen3Models) return;
    const modelId = _settingValue('qwen3_tts_model_name');
    const model = (state.qwen3Models.models || []).find((m) => m.id === modelId);
    if (!model) return;
    const need = new Set(model.requires || []);
    const map = {
        '--qwen3-tts-ref-audio': 'ref_audio',
        '--qwen3-tts-ref-text': 'ref_text',
        '--qwen3-tts-instruct': 'instruct',
    };
    for (const [flag, key] of Object.entries(map)) {
        const wrap = $(`#f-${flag}`)?.closest('.field');
        if (!wrap) continue;
        let hint = wrap.querySelector('.field-required-hint');
        if (need.has(key)) {
            if (!hint) {
                hint = el('span', { class: 'field-required-hint', style: { marginLeft: '8px', color: 'var(--accent)', fontSize: '11px', fontWeight: '600' } }, 'required');
                const label = wrap.querySelector('.field-label');
                if (label) label.appendChild(hint);
            }
        } else if (hint) {
            hint.remove();
        }
    }
}

// ---- Ollama LLM keepalive (dashboard-only) -----------------------------
//
// The --llm-keepalive setting is dashboard-only: upstream has no such
// argument, so the schema introspection doesn't register a field for it.
// We hand-render a single <select> in the LLM tab using the curated list
// baked into ``_KEEPALIVE_CURATED`` below, with a "Custom…" escape that
// swaps in a free-text input — same pattern renderField() uses for the
// qwen3-model picker. The value lives in state.settings["--llm-keepalive"]
// and Save / Start / Restart on the server forward it to the dashboard-
// side pinger via llm_keepaliver.update_from_settings.

const _KEEPALIVE_CURATED = [
    { value: "0",   label: "(off) — unload immediately" },
    { value: "5m",  label: "5 minutes  (Ollama default)" },
    { value: "15m", label: "15 minutes" },
    { value: "30m", label: "30 minutes" },
    { value: "1h",  label: "1 hour" },
    { value: "2h",  label: "2 hours" },
    { value: "12h", label: "12 hours" },
    // "-1" is Ollama's "keep loaded forever" sentinel.
    { value: "-1",  label: "Forever  (-1)" },
];

function renderKeepaliveField() {
    const flag = '--llm-keepalive';
    const fieldId = `f-${flag}`;
    const hoverPreview =
        'How long Ollama keeps the LLM model loaded in VRAM after the last request. ' +
        'Ollama default is 5 minutes; the first reply after a longer pause pays a ~20s reload. ' +
        'Pick a longer interval (or -1 for forever) to avoid that.';
    const helpText =
        'How long Ollama should keep the LLM model loaded in VRAM after the ' +
        'last request. Ollama unloads after 5 minutes by default; the first ' +
        'reply after a longer pause pays a ~20 s reload. A longer interval ' +
        '(or "-1" for forever) avoids that at the cost of holding VRAM. ' +
        'Custom accepts any Ollama duration: e.g. 45m, 90m, 4h, 0 (off), or -1.';
    const label = el('label', { class: 'field-label', for: fieldId }, [
        flag,
        el('span', { class: 'field-flag' }, ''),
        el('button', {
            type: 'button', class: 'help-btn', title: hoverPreview,
            onclick: (e) => {
                e.preventDefault();
                const field = e.currentTarget.closest('.field');
                const help = field && field.querySelector('.field-help');
                if (help) help.classList.toggle('visible');
            },
        }, '?'),
    ]);

    const curVal = (state.settings[flag] == null ? '' : String(state.settings[flag])).trim();
    const isCurated = _KEEPALIVE_CURATED.some((c) => c.value === curVal);
    const customSentinel = '__custom__';

    const sel = el('select', {
        class: 'field-select', id: fieldId,
        onchange: (e) => {
            const v = e.target.value;
            if (v === customSentinel) {
                const freeText = el('input', {
                    class: 'field-input', id: fieldId, type: 'text',
                    placeholder: 'e.g. 45m, 90m, 4h, -1, 0',
                    oninput: (ev) => {
                        state.settings[flag] = (ev.target.value || '').trim();
                        applyDisabledStates();
                    },
                });
                freeText.value = curVal || '';
                state.settings[flag] = freeText.value;
                sel.replaceWith(freeText);
                freeText.focus();
                freeText.select();
                return;
            }
            state.settings[flag] = v;
            applyDisabledStates();
        },
    });
    for (const c of _KEEPALIVE_CURATED) {
        sel.appendChild(el('option', { value: c.value }, c.label));
    }
    sel.appendChild(el('option', { value: customSentinel }, 'Custom (type your own)'));
    if (isCurated || curVal === '') {
        sel.value = curVal || '0';
        if (curVal === '') state.settings[flag] = '0';
    } else {
        sel.value = customSentinel;
    }

    const help = el('div', { class: 'field-help', id: `help-${flag}` }, helpText);
    return el('div', { class: 'field full' }, [label, sel, help]);
}

// 0.3.2+: Ollama model dropdown now renders correctly without a
// debug badge — see the Ollama branch in renderField() (L601).


// next to --llm-keepalive on the LLM tab; the upstream pipeline has no
// such flag, so the schema introspection doesn't know about it. A
// plain number input with a curated shortcut list keeps the UI
// compact without giving up the ability to type any value in
// [5, 600].
const _OLLAMA_TIMEOUT_PRESETS = [
    { value: 30,  label: '30 s' },
    { value: 60,  label: '60 s (default)' },
    { value: 120, label: '2 min' },
    { value: 300, label: '5 min' },
    { value: 600, label: '10 min (max)' },
];
function renderOllamaLoadTimeoutField() {
    const flag = '--ollama-load-timeout-seconds';
    const fieldId = `f-${flag}`;
    const hoverPreview =
        'Max seconds the dashboard waits for Ollama to load the LLM model ' +
        'into VRAM during the one-shot warmup. Bump this up on a slow LAN ' +
        'or for very large models; 60 s covers a 20 B model on most links.';
    const helpText =
        'How long the dashboard will wait for Ollama to finish loading the ' +
        'LLM model into VRAM during the one-shot warmup (fired when you ' +
        'click Save Settings or Start Pipeline). A cold load of a 20 B model ' +
        'over a LAN typically takes 30-60 s; very large models or slow links ' +
        'may need more. Allowed range: 5 - 600 seconds. Clamped by the ' +
        'dashboard on save; values outside the range are silently clipped.';
    const label = el('label', { class: 'field-label', for: fieldId }, [
        flag,
        el('span', { class: 'field-flag' }, ''),
        el('button', {
            type: 'button', class: 'help-btn', title: hoverPreview,
            onclick: (e) => {
                e.preventDefault();
                const field = e.currentTarget.closest('.field');
                const help = field && field.querySelector('.field-help');
                if (help) help.classList.toggle('visible');
            },
        }, '?'),
    ]);
    // Coerce whatever's in state.settings to a number; fall back to 60.
    let curVal = parseInt(state.settings[flag], 10);
    if (!Number.isFinite(curVal) || curVal <= 0) curVal = 60;
    const isCurated = _OLLAMA_TIMEOUT_PRESETS.some((c) => c.value === curVal);
    const customSentinel = '__custom__';
    const sel = el('select', {
        class: 'field-select', id: fieldId,
        onchange: (e) => {
            const v = e.target.value;
            if (v === customSentinel) {
                // Swap to a free-text numeric input so the user can
                // type any value in the allowed range. The number
                // input keeps them honest — letters / decimals out.
                const freeText = el('input', {
                    class: 'field-input', id: fieldId, type: 'number',
                    min: '5', max: '600', step: '5',
                    placeholder: 'e.g. 90',
                    oninput: (ev) => {
                        const n = parseInt(ev.target.value, 10);
                        // Update state on every keystroke; the endpoint
                        // clamps to [5, 600] on its own.
                        state.settings[flag] = Number.isFinite(n) ? n : 60;
                        applyDisabledStates();
                    },
                });
                // Pre-fill with the current value (whether curated or
                // already-custom) so the user can tweak it.
                freeText.value = String(curVal);
                state.settings[flag] = curVal;
                sel.replaceWith(freeText);
                freeText.focus();
                freeText.select();
                return;
            }
            const n = parseInt(v, 10);
            state.settings[flag] = Number.isFinite(n) ? n : 60;
            applyDisabledStates();
        },
    });
    for (const c of _OLLAMA_TIMEOUT_PRESETS) {
        sel.appendChild(el('option', { value: String(c.value) }, c.label));
    }
    sel.appendChild(el('option', { value: customSentinel }, 'Custom (type your own)'));
    if (isCurated) {
        sel.value = String(curVal);
    } else {
        // Saved value isn't in the curated list (e.g. legacy save or a
        // user-typed number) — switch to Custom so the saved value is
        // visible rather than silently clamping to 60.
        sel.value = customSentinel;
    }
    // Make sure state has a sane value even if nothing else has set it.
    if (state.settings[flag] == null) state.settings[flag] = 60;
    const help = el('div', { class: 'field-help', id: `help-${flag}` }, helpText);
    return el('div', { class: 'field full' }, [label, sel, help]);
}

// ---- Status & Logs tab ----------------------------------------------

function renderStatusTab(tab) {
    tab.appendChild(el('div', { class: 'tab-header' }, [
        el('h1', { class: 'tab-title' }, 'Status & Logs'),
        el('div', { class: 'tab-subtitle' }, 'Live pipeline status, system resources, and process logs.'),
    ]));

    const statusGrid = el('div', { class: 'status-grid', id: 'status-grid' });
    tab.appendChild(statusGrid);

    // The Command line lives in its own row so a long arg list doesn't
    // stretch the State / PID / Uptime cards next to it.
    tab.appendChild(el('div', { class: 'status-section-label' }, 'Command'));
    tab.appendChild(el('div', { class: 'status-command', id: 'status-command' }));

    const logToolbar = el('div', { class: 'log-toolbar' }, [
        el('span', { class: 'text-dim' }, 'Filter:'),
        (() => {
            const s = el('select', { class: 'field-select', onchange: (e) => { state.logFilter = e.target.value; renderLogs(); } });
            for (const v of ['ALL', 'INFO', 'WARNING', 'ERROR']) {
                s.appendChild(el('option', { value: v }, v));
            }
            return s;
        })(),
        el('button', {
            class: 'btn',
            onclick: () => { $('#log-console').textContent = ''; state.logIndex = 0; }
        }, 'Clear'),
        el('button', {
            class: 'btn',
            onclick: toggleVerbose,
        }, 'Toggle Verbose'),
    ]);
    tab.appendChild(logToolbar);

    tab.appendChild(el('div', { class: 'log-console', id: 'log-console' }));

    const poolTitle = el('h3', { style: { marginTop: '24px' } }, 'Realtime Pool');
    tab.appendChild(poolTitle);
    tab.appendChild(el('pre', { class: 'log-console', id: 'pool-console', style: { height: 'auto', maxHeight: '200px' } }, 'No data yet. Start the pipeline in realtime mode to see pool status.'));
}

function renderStatusCards() {
    const s = state.status;
    const uptime = s.running ? formatUptime(s.uptime_s) : '—';
    const grid = $('#status-grid');
    if (!grid) return;

    // Build or update the small cards (State / PID / Port / Uptime / Pool)
    // in place. We never clear the grid here -- pollSystem() appends
    // CPU/RAM/GPU cards to it, and wiping the grid every 2s would cause
    // those to flicker.
    const ensure = (id, label) => {
        let card = document.getElementById(id);
        if (!card) {
            card = el('div', { class: 'status-card', id });
            card.appendChild(el('div', { class: 'status-card-label' }, label));
            card.appendChild(el('div', { class: 'status-card-value', id: `${id}-value` }));
            grid.appendChild(card);
        }
        return card;
    };
    ensure('status-state', 'State');
    ensure('status-pid', 'PID');
    ensure('status-port', 'Port');
    ensure('status-uptime', 'Uptime');
    ensure('status-pool', 'Pool');
    ensure('status-mode', 'Mode');
    ensure('status-model', 'Model');
    ensure('status-stt', 'STT');
    ensure('status-tts', 'TTS');
    ensure('status-log', 'Log Level');
    ensure('status-pool-url', 'Pool URL');
    ensure('status-llm-url', 'LLM Endpoint');

    const stateVal = document.getElementById('status-state-value');
    stateVal.textContent = s.running ? 'RUNNING' : 'STOPPED';
    stateVal.style.color = s.running ? 'var(--success)' : '';
    document.getElementById('status-pid-value').textContent = s.pid || '—';
    document.getElementById('status-uptime-value').textContent = uptime;
    document.getElementById('status-port-value').textContent = portFromCommand(s.command_line) || '—';
    document.getElementById('status-mode-value').textContent = modeFromCommand(s.command_line) || '—';
    document.getElementById('status-model-value').textContent = modelFromCommand(s.command_line) || '—';
    document.getElementById('status-stt-value').textContent = sttFromCommand(s.command_line) || '—';
    document.getElementById('status-tts-value').textContent = ttsFromCommand(s.command_line) || '—';
    document.getElementById('status-log-value').textContent = flagValueFromArgv(s.command_line, '--log-level') || 'info';
    document.getElementById('status-pool-url-value').textContent =
        s.running ? `ws://0.0.0.0:${portFromCommand(s.command_line) || 8765}/v1/realtime` : '—';
    document.getElementById('status-llm-url-value').textContent =
        flagValueFromArgv(s.command_line, '--responses-api-base-url') || '—';

    // Pool summary is updated by pollPool(); if no data yet, show "—".
    const poolVal = document.getElementById('status-pool-value');
    const lastPool = state._lastPool;
    if (lastPool && lastPool.data) {
        poolVal.textContent = `${lastPool.data.in_use}/${lastPool.data.size}`;
    } else {
        poolVal.textContent = '—';
    }

    // Command line lives in its own row so a long argv doesn't stretch
    // the small cards above.
    const cmdBox = $('#status-command');
    if (cmdBox) {
        const cmd = s.command_line || '—';
        let label = cmdBox.querySelector('.status-card-label');
        let val = cmdBox.querySelector('.status-card-value');
        if (!label) {
            label = el('div', { class: 'status-card-label' }, 'Command');
            val = el('div', { class: 'status-card-value', id: 'status-command-value' });
            const wrap = el('div', { class: 'status-card status-card-wide', id: 'status-command-card' }, [label, val]);
            cmdBox.replaceChildren(wrap);
        }
        document.getElementById('status-command-value').textContent = cmd;
    }
}

// Extract `--port 8765` (or the long-flag version) from a saved argv.
function flagValueFromArgv(argv, flag) {
    if (!argv) return null;
    const tokens = argv.split(/\s+/);
    for (let i = 0; i < tokens.length; i++) {
        if (tokens[i] === flag && i + 1 < tokens.length) return tokens[i + 1];
    }
    return null;
}
function portFromCommand(argv) {
    return flagValueFromArgv(argv, '--port') || '8765';
}
function modeFromCommand(argv) {
    return flagValueFromArgv(argv, '--mode') || '—';
}
function modelFromCommand(argv) {
    return flagValueFromArgv(argv, '--model-name') || '—';
}
function sttFromCommand(argv) {
    return flagValueFromArgv(argv, '--stt') || '—';
}
function ttsFromCommand(argv) {
    return flagValueFromArgv(argv, '--tts') || '—';
}

function renderLogs() {
    const con = $('#log-console');
    if (!con) return;
    con.textContent = '';
    const filter = state.logFilter;
    const show = (lvl) => {
        if (filter === 'ALL') return true;
        if (filter === 'ERROR') return lvl === 'ERROR' || lvl === 'CRITICAL';
        if (filter === 'WARNING') return lvl === 'WARNING' || lvl === 'ERROR' || lvl === 'CRITICAL';
        if (filter === 'INFO') return lvl !== 'DEBUG';
        return true;
    };
    for (const line of state._logBuffer || []) {
        if (!show(line.level)) continue;
        con.appendChild(el('div', { class: `log-line ${line.level}` }, line.text));
    }
    con.scrollTop = con.scrollHeight;
}

function appendLogLine(line) {
    if (!state._logBuffer) state._logBuffer = [];
    state._logBuffer.push(line);
    if (state._logBuffer.length > 5000) state._logBuffer.shift();
    const con = $('#log-console');
    if (!con) return;
    const filter = state.logFilter;
    const show = (lvl) => {
        if (filter === 'ALL') return true;
        if (filter === 'ERROR') return lvl === 'ERROR' || lvl === 'CRITICAL';
        if (filter === 'WARNING') return lvl === 'WARNING' || lvl === 'ERROR' || lvl === 'CRITICAL';
        if (filter === 'INFO') return lvl !== 'DEBUG';
        return true;
    };
    if (show(line.level)) {
        con.appendChild(el('div', { class: `log-line ${line.level}` }, line.text));
        con.scrollTop = con.scrollHeight;
    }
}

async function fetchInitialLogs() {
    try {
        const r = await getJSON('/api/logs?since=0');
        state._logBuffer = r.lines;
        state.logIndex = r.lines.length ? r.lines[r.lines.length - 1].index : 0;
        renderLogs();
    } catch (e) {
        console.warn('fetchInitialLogs:', e);
    }
}

function startLogStream() {
    fetchInitialLogs();
    const proto = location.protocol === 'https:' ? 'wss' : 'ws';
    const ws = new WebSocket(`${proto}://${location.host}/ws/logs`);
    ws.onmessage = (e) => {
        try {
            const msg = JSON.parse(e.data);
            if (msg.__ping__) return;
            if (msg.index == null) return;
            if (msg.index <= state.logIndex) return;
            state.logIndex = msg.index;
            appendLogLine(msg);
        } catch { /* ignore */ }
    };
    ws.onclose = () => {
        setTimeout(startLogStream, 2000);
    };
    state.ws = ws;
}

// ---- Status poll -----------------------------------------------------

async function startStatusPoll() {
    pollStatus();
    setInterval(pollStatus, 2000);
    setInterval(pollSystem, 3000);
    setInterval(pollPool, 5000);
}

async function pollStatus() {
    try {
        const s = await getJSON('/api/process/status');
        state.status = s;
        const dot = $('#status-dot');
        const text = $('#status-text');
        dot.classList.toggle('running', s.running);
        text.textContent = s.running ? `running (${formatUptime(s.uptime_s)})` : 'stopped';
        renderStatusCards();
    } catch (e) { /* ignore */ }
}

async function pollSystem() {
    try {
        const r = await getJSON('/api/system');
        const grid = $('#status-grid');
        if (!grid) return;
        // Append system cards to the main status grid so all 15 cards
        // share one table (5 per row × 3 rows).
        let cpu = $('#sys-cpu'), ram = $('#sys-ram'), gpu = $('#sys-gpu');
        if (!cpu) {
            cpu = el('div', { class: 'status-card', id: 'sys-cpu' }, [el('div', { class: 'status-card-label' }, 'CPU'), el('div', { class: 'status-card-value' }, '')]);
            ram = el('div', { class: 'status-card', id: 'sys-ram' }, [el('div', { class: 'status-card-label' }, 'RAM'), el('div', { class: 'status-card-value' }, '')]);
            gpu = el('div', { class: 'status-card', id: 'sys-gpu' }, [el('div', { class: 'status-card-label' }, 'GPU'), el('div', { class: 'status-card-value' }, '')]);
            grid.appendChild(cpu); grid.appendChild(ram); grid.appendChild(gpu);
        }
        cpu.querySelector('.status-card-value').textContent = r.cpu_percent.toFixed(0) + '%';
        ram.querySelector('.status-card-value').textContent = `${r.ram_used_gb}/${r.ram_total_gb} GB (${r.ram_percent.toFixed(0)}%)`;
        if (r.gpu && r.gpu.gpus && r.gpu.gpus[0]) {
            const g = r.gpu.gpus[0];
            gpu.querySelector('.status-card-value').textContent = `${g.util_percent}% (${g.mem_used_mb}MB)`;
        } else {
            gpu.querySelector('.status-card-value').textContent = 'n/a';
        }
    } catch (e) { /* ignore */ }
}

async function pollPool() {
    const con = $('#pool-console');
    if (!con) return;
    try {
        const r = await getJSON('/api/pool');
        state._lastPool = r;
        if (!r.available) {
            con.textContent = `(pool not available: ${r.reason || 'unknown'})`;
            // Pool card needs a refresh too.
            const poolVal = document.getElementById('status-pool-value');
            if (poolVal) poolVal.textContent = '—';
            return;
        }
        con.textContent = JSON.stringify(r.data, null, 2);
        // Keep the small Pool card in sync.
        const poolVal = document.getElementById('status-pool-value');
        if (poolVal && r.data) poolVal.textContent = `${r.data.in_use}/${r.data.size}`;
    } catch (e) {
        con.textContent = '(error fetching pool: ' + e.message + ')';
    }
}

// ---- Guide tab -------------------------------------------------------

async function renderGuideTab(tab) {
    tab.appendChild(el('div', { class: 'tab-header' }, [
        el('h1', { class: 'tab-title' }, 'Guide'),
        el('div', { class: 'tab-subtitle' }, 'How to use the speech-to-speech pipeline.'),
    ]));
    const content = el('div', { class: 'guide-content', id: 'guide-content' }, 'Loading...');
    tab.appendChild(content);
    try {
        const md = await fetch('/api/guide').then(r => r.text());
        content.innerHTML = renderMarkdown(md);
    } catch (e) {
        content.textContent = 'Failed to load guide: ' + e.message;
    }
}

// Minimal markdown renderer. We need headings, lists, inline code, code blocks,
// and bold/italic. No need for a library -- this is the only Markdown we'll
// render and a 30-line renderer is enough.
function renderMarkdown(md) {
    const lines = md.split('\n');
    const out = [];
    let inCode = false, codeBuf = [];
    let inList = false;
    const flushList = () => {
        if (inList) { out.push('</ul>'); inList = false; }
    };
    for (let i = 0; i < lines.length; i++) {
        const line = lines[i];
        if (line.startsWith('```')) {
            if (inCode) {
                out.push('<pre><code>' + escapeHTML(codeBuf.join('\n')) + '</code></pre>');
                inCode = false; codeBuf = [];
            } else {
                flushList();
                inCode = true;
            }
            continue;
        }
        if (inCode) { codeBuf.push(line); continue; }
        if (line.startsWith('# ')) { flushList(); out.push('<h1>' + inline(line.slice(2)) + '</h1>'); }
        else if (line.startsWith('## ')) { flushList(); out.push('<h2>' + inline(line.slice(3)) + '</h2>'); }
        else if (line.startsWith('### ')) { flushList(); out.push('<h3>' + inline(line.slice(4)) + '</h3>'); }
        else if (line.match(/^[-*] /)) {
            if (!inList) { out.push('<ul>'); inList = true; }
            out.push('<li>' + inline(line.slice(2)) + '</li>');
        }
        else if (line.trim() === '') { flushList(); out.push(''); }
        else { flushList(); out.push('<p>' + inline(line) + '</p>'); }
    }
    flushList();
    return out.join('\n');
}
function inline(s) {
    return escapeHTML(s)
        .replace(/`([^`]+)`/g, '<code>$1</code>')
        .replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>')
        .replace(/\*([^*]+)\*/g, '<em>$1</em>');
}
function escapeHTML(s) {
    return s.replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;');
}

// ---- Settings (file) tab --------------------------------------------

function renderSettingsFileTab(tab) {
    tab.appendChild(el('div', { class: 'tab-header' }, [
        el('h1', { class: 'tab-title' }, 'Settings'),
        el('div', { class: 'tab-subtitle' }, 'Save, reset, import, and export your configuration.'),
    ]));

    tab.appendChild(el('div', { class: 'tab-header', style: { borderBottom: 'none' } }, [
        el('div', { class: 'text-dim' }, [
            'Settings file: ', el('span', { class: 'text-mono' }, state.savedPath), el('br'),
            'Status: ', el('span', { class: 'text-mono' }, state.saved ? 'saved' : 'not saved (using defaults)'),
        ]),
    ]));

    const envTitle = el('h3', {}, 'Environment Variables');
    tab.appendChild(envTitle);
    tab.appendChild(el('div', { class: 'text-dim', style: { marginBottom: '12px' } }, 'Extra environment variables passed to the pipeline subprocess. Useful for HF_TOKEN, OPENAI_API_KEY, etc.'));
    const envEditor = el('div', { id: 'env-editor' });
    tab.appendChild(envEditor);
    renderEnvEditor();

    tab.appendChild(el('h3', { style: { marginTop: '24px' } }, 'Actions'));
    const row = el('div', { class: 'btn-row' }, [
        el('button', { class: 'btn btn-primary btn-large', onclick: saveSettings }, 'Save Settings'),
        el('button', { class: 'btn', onclick: resetSettings }, 'Reset to Defaults'),
        el('button', { class: 'btn', onclick: exportSettings }, 'Export JSON'),
        el('button', { class: 'btn', onclick: importSettings }, 'Import JSON'),
        el('button', { class: 'btn', onclick: loadExampleSettings }, 'Load Example'),
    ]);
    tab.appendChild(row);
}

function renderEnvEditor() {
    const ed = $('#env-editor');
    if (!ed) return;
    ed.textContent = '';
    if (!Array.isArray(state.settings.env)) state.settings.env = [];
    state.settings.env.forEach((kv, i) => {
        const [k, ...rest] = kv.split('=');
        const v = rest.join('=');
        ed.appendChild(el('div', { class: 'field', style: { display: 'grid', gridTemplateColumns: '1fr 2fr auto', gap: '8px', marginBottom: '8px' } }, [
            el('input', {
                class: 'field-input', value: k, placeholder: 'KEY',
                oninput: (e) => state.settings.env[i] = `${e.target.value}=${v}`,
            }),
            el('input', {
                class: 'field-input', value: v, placeholder: 'VALUE',
                oninput: (e) => state.settings.env[i] = `${k}=${e.target.value}`,
            }),
            el('button', {
                class: 'btn btn-danger', onclick: () => {
                    state.settings.env.splice(i, 1);
                    renderEnvEditor();
                }
            }, '×'),
        ]));
    });
    ed.appendChild(el('button', {
        class: 'btn', onclick: () => {
            state.settings.env.push('KEY=');
            renderEnvEditor();
        }
    }, '+ Add variable'));
}

async function saveSettings() {
    try {
        await postJSON('/api/settings', state.settings);
        state.saved = true;
        toast('Settings saved.', 'success');
        renderAll();
    } catch (e) {
        toast('Save failed: ' + e.message, 'error', 6000);
    }
}

async function resetSettings() {
    showModal(
        'Reset settings?',
        'This deletes the saved settings file. The form will reload with default values. Unsaved changes will be lost.',
        [
            { label: 'Cancel', kind: '', onClick: () => {} },
            { label: 'Reset', kind: 'btn-danger', onClick: async () => {
                try {
                    const r = await postJSON('/api/reset', {});
                    state.settings = r.settings;
                    state.saved = false;
                    applyTheme(state.settings.theme || 'cyberpunk-neon');
                    toast('Settings reset.', 'success');
                    renderAll();
                } catch (e) {
                    toast('Reset failed: ' + e.message, 'error');
                }
            } },
        ]
    );
}

function exportSettings() {
    const blob = new Blob([JSON.stringify(state.settings, null, 2)], { type: 'application/json' });
    const a = document.createElement('a');
    a.href = URL.createObjectURL(blob);
    a.download = 'web_ui_settings.json';
    a.click();
    URL.revokeObjectURL(a.href);
}

function importSettings() {
    const input = document.createElement('input');
    input.type = 'file';
    input.accept = 'application/json';
    input.onchange = async (e) => {
        const f = e.target.files[0];
        if (!f) return;
        try {
            const text = await f.text();
            const obj = JSON.parse(text);
            state.settings = { ...state.defaults, ...obj };
            state.saved = false;
            applyTheme(state.settings.theme || 'cyberpunk-neon');
            toast('Imported. Click Save to persist.', 'success');
            renderAll();
        } catch (err) {
            toast('Import failed: ' + err.message, 'error');
        }
    };
    input.click();
}

// One-click load of the bundled example config. Fetches the JSON from
// /static-repo/ (the dashboard's repo-root mount) and merges it onto
// the defaults, exactly like importSettings does for a user-picked file
// — but with no file picker dialog. Useful on a fresh clone where the
// user wants a real working config (Ollama + qwen3-TTS + parakeet STT,
// realtime mode) instead of bare defaults. The example file lives at
// the repo root and is also discoverable via the Settings tab's Import
// JSON button.
async function loadExampleSettings() {
    try {
        const r = await fetch('/static-repo/web_ui_settings.example.json', { cache: 'no-cache' });
        if (!r.ok) throw new Error(`HTTP ${r.status}`);
        const obj = await r.json();
        state.settings = { ...state.defaults, ...obj };
        state.saved = false;
        applyTheme(state.settings.theme || 'cyberpunk-neon');
        toast('Example loaded (Ollama + qwen3-TTS + parakeet STT). Click Save to persist.', 'success');
        renderAll();
    } catch (e) {
        toast('Could not load example: ' + e.message, 'error');
    }
}

// ---- Control tab -----------------------------------------------------

function renderControlTab(tab) {
    tab.appendChild(el('div', { class: 'tab-header' }, [
        el('h1', { class: 'tab-title' }, 'Control'),
        el('div', { class: 'tab-subtitle' }, 'Start, stop, and shut down the pipeline and the dashboard itself.'),
    ]));

    const row1 = el('div', { class: 'btn-row' }, [
        el('button', { class: 'btn btn-primary btn-large', onclick: startPipeline }, '▶ Start Pipeline'),
        el('button', { class: 'btn btn-large', onclick: stopPipeline }, '■ Stop Pipeline'),
        el('button', { class: 'btn btn-large', onclick: restartPipeline }, '↻ Restart Pipeline'),
    ]);
    tab.appendChild(row1);

    tab.appendChild(el('h3', { style: { marginTop: '32px' } }, 'Memory'));
    tab.appendChild(el('div', { class: 'text-dim', style: { marginBottom: '12px', maxWidth: '720px' } }, 'The pipeline keeps the TTS model in RAM while running. Use this to drop the TTS model from memory without killing the pipeline; the model reloads on the next TTS request (~20s on CPU, ~5s on GPU).'));
    const rowMem = el('div', { class: 'btn-row' }, [
        el('button', { class: 'btn btn-large', onclick: unloadTtsModel }, '🧹 Unload TTS Model'),
    ]);
    tab.appendChild(rowMem);

    tab.appendChild(el('h3', { style: { marginTop: '32px' } }, 'Danger Zone'));
    tab.appendChild(el('div', { class: 'text-dim', style: { marginBottom: '12px' } }, 'Stops the pipeline AND closes the dashboard web server. You will need to re-run start_web_ui.sh to come back.'));

    const row2 = el('div', { class: 'btn-row' }, [
        el('button', { class: 'btn btn-danger btn-large', onclick: shutdownAll }, '⏻ Shutdown Everything'),
    ]);
    tab.appendChild(row2);
}

async function startPipeline() {
    try {
        const res = await postJSON('/api/process/start', { settings: state.settings });
        if (res && res.restarting) {
            // The dashboard is auto-installing the matching torch wheel
            // and restarting so the new wheel is loaded. The browser will
            // briefly lose its connection; reload the page and retry.
            const wait = (res.retry_after_ms || 4000) / 1000;
            toast(
                `Installing GPU-compatible torch and restarting dashboard... ` +
                `Reloading in ${wait.toFixed(0)}s.`,
                'info',
                10000,
            );
            setTimeout(() => window.location.reload(), res.retry_after_ms || 4000);
            return;
        }
        toast('Pipeline started.', 'success');
        pollStatus();
    } catch (e) {
        if (e.status === 409 && e.detail && e.detail.error === 'chatterbox_not_installed') {
            showChatterboxInstallModal(e.detail);
            return;
        }
        toast('Start failed: ' + e.message, 'error', 6000);
    }
}

function showChatterboxInstallModal(detail) {
    const cmd = detail.install_command || '(no command available)';
    const plat = detail.platform || '';
    const body = el('div', {}, [
        el('p', {}, `Chatterbox TTS is not installed. The pipeline can't start until it is.`),
        el('p', { class: 'text-dim' }, `Detected platform: ${plat}. The install command below is tailored to that platform.`),
        el('pre', { class: 'install-cmd' }, cmd),
        el('p', { class: 'text-dim' }, 'Install logs will stream into the Status & Logs tab. This typically takes 1-3 minutes.'),
    ]);
    showModal('Install Chatterbox TTS', body, [
        { label: 'Cancel', kind: '', onClick: () => {} },
        { label: 'Run install', kind: 'btn-primary', onClick: runChatterboxInstall },
    ]);
}

async function runChatterboxInstall() {
    try {
        await postJSON('/api/install/chatterbox', {});
        toast('Install started. Watch the Status & Logs tab.', 'info', 6000);
    } catch (e) {
        toast('Install failed to start: ' + e.message, 'error', 6000);
    }
}

async function stopPipeline() {
    showModal(
        'Stop pipeline?',
        'The pipeline subprocess will be terminated. Any active conversation will be cut off.',
        [
            { label: 'Cancel', kind: '', onClick: () => {} },
            { label: 'Stop', kind: 'btn-danger', onClick: async () => {
                try {
                    await postJSON('/api/process/stop', {});
                    toast('Pipeline stopped.', 'success');
                    pollStatus();
                } catch (e) {
                    toast('Stop failed: ' + e.message, 'error');
                }
            } },
        ]
    );
}

async function unloadTtsModel() {
    try {
        const status = await getJSON('/api/process/status');
        if (!status || !status.running) {
            toast('Pipeline is not running — nothing to unload.', 'warning');
            return;
        }
        const res = await postJSON('/api/process/unload_tts', {});
        toast(`TTS model unloaded from ${res.affected}/${res.total} unit(s). Reloads on next reply.`, 'success');
    } catch (e) {
        toast('Unload failed: ' + e.message, 'error');
    }
}

async function restartPipeline() {
    try {
        await postJSON('/api/process/restart', { settings: state.settings });
        toast('Pipeline restarted.', 'success');
        pollStatus();
    } catch (e) {
        toast('Restart failed: ' + e.message, 'error');
    }
}

function shutdownAll() {
    showModal(
        'Shutdown everything?',
        'This stops the pipeline AND closes the dashboard web server. You will need to re-run start_web_ui.sh to start it again.',
        [
            { label: 'Cancel', kind: '', onClick: () => {} },
            { label: 'Shutdown', kind: 'btn-danger', onClick: async () => {
                try {
                    await postJSON('/api/shutdown', {});
                } catch (e) { /* server is dying */ }
                toast('Shutting down...', 'warning', 3000);
            } },
        ]
    );
}

function toggleVerbose() {
    const wantDebug = state.settings['--log-level'] !== 'debug';
    showModal(
        wantDebug ? 'Enable verbose logging?' : 'Disable verbose logging?',
        wantDebug
            ? 'The pipeline will be RESTARTED with --log-level debug. The current conversation will be interrupted. This is a noisy mode useful for debugging.'
            : 'The pipeline will be RESTARTED with the previous log level. The current conversation will be interrupted.',
        [
            { label: 'Cancel', kind: '', onClick: () => {} },
            { label: wantDebug ? 'Enable verbose' : 'Disable verbose', kind: 'btn-primary', onClick: async () => {
                state.settings['--log-level'] = wantDebug ? 'debug' : 'info';
                try {
                    await saveSettings();
                    await postJSON('/api/process/restart', { settings: state.settings });
                    toast(wantDebug ? 'Verbose mode enabled.' : 'Verbose mode disabled.', 'success');
                    pollStatus();
                } catch (e) {
                    toast('Failed: ' + e.message, 'error');
                }
            } },
        ]
    );
}

// ---- Helpers ---------------------------------------------------------

function formatUptime(s) {
    if (!s) return '0s';
    s = Math.floor(s);
    const h = Math.floor(s / 3600);
    const m = Math.floor((s % 3600) / 60);
    const sec = s % 60;
    if (h > 0) return `${h}h${m}m`;
    if (m > 0) return `${m}m${sec}s`;
    return `${sec}s`;
}

// ---- Go --------------------------------------------------------------

document.addEventListener('DOMContentLoaded', init);
