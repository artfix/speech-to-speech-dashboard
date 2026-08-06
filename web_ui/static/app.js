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
const putJSON = (path, body) => api(path, { method: 'PUT', body: JSON.stringify(body) });

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
    { id: 'hermes', label: 'Hermes', icon: 'H' },
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
    activateTab(_initialTabId());
}

function activateTab(id) {
    $$('.nav-item').forEach(n => n.classList.toggle('active', n.dataset.tab === id));
    $$('.tab').forEach(t => t.classList.toggle('active', t.dataset.tab === id));
    // Remember the active tab in the URL hash so a page refresh lands
    // on the same tab instead of snapping back to Mode.
    if (history.replaceState && id) {
        history.replaceState(null, '', `#${id}`);
    }
}

function _initialTabId() {
    // Hash-based tab persistence. Only accept hashes that match a real tab.
    const raw = (window.location.hash || '').replace(/^#/, '');
    if (raw && TAB_DEFS.some(t => t.id === raw)) {
        return raw;
    }
    return 'mode';
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
        } else if (t.id === 'hermes') {
            renderHermesTab(tab);
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
            // 0.4.0+: Hermes Agent backend toggle. Hand-rendered (it's
            // not in the introspected schema — dashboard-only). Sits at
            // the top of the LLM tab so the user picks the brain before
            // they see the URL/api-key fields. Auto-fills the LLM URL
            // + api_key + backend when "hermes" is selected.
            grid2.appendChild(renderHermesBackendField());
            grid2.appendChild(renderKeepaliveField());
            // 0.3.2+: dashboard-only max-wait for the one-shot Ollama
            // warmup. Default 60 s; users on a slow LAN loading a 70 B
            // model can bump it. Same hand-rendered pattern as
            // --llm-keepalive (it's not in the introspected schema).
            grid2.appendChild(renderOllamaLoadTimeoutField());
            // 0.4.2+: per-backend LLM request read-timeout dropdown
            // (overrides the pipeline's hardcoded 20 s openai-SDK
            // read timeout). Dashboard-only — upstream pipeline has
            // no such flag. Sits below the keepalive / ollama-load
            // timeout rows so the LLM tab reads top-to-bottom:
            // backend type → keepalive → ollama warmup → request
            // timeout.
            grid2.appendChild(renderLlmRequestTimeoutField());
            // 0.4.3+: Ollama model lifecycle section. Hidden when the
            // URL doesn't look like Ollama (matches the keepalive
            // heuristic). Shows live model/endpoint/context + manual
            // reload button. Auto-unloads when --responses-api-num-ctx
            // changes if the model is currently loaded at a different
            // context.
            grid2.appendChild(renderOllamaLifecycleField());
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

        // Two voice libraries (chatterbox cloned voices + qwen3 preset
        // speakers) live on the TTS tab. They used to share a single
        // #voice-library-mount, but the chatterbox renderer awaits
        // fetchVoices() while the qwen3 renderer is synchronous, so
        // qwen3's "hide-when-I'm-not-active" branch would clobber the
        // mount's display state mid-fetch and hide the chatterbox
        // library after it had populated its DOM. Each library now
        // owns its own mount so neither can stomp the other.
        const chatterboxMount = el('div', { id: 'voice-library-mount-chatterbox' });
        const qwen3Mount = el('div', { id: 'voice-library-mount-qwen3' });
        tab.appendChild(chatterboxMount);
        tab.appendChild(qwen3Mount);
        const rerenderLibrary = () => {
            renderVoiceLibrary(chatterboxMount, state.settings, () => renderAll());
            if (typeof window.renderQwen3VoiceLibrary === 'function') {
                window.renderQwen3VoiceLibrary(qwen3Mount, state.settings, () => renderAll());
            }
        };
        // Initial paint
        rerenderLibrary();
    }
    updateSubgroupVisibility();
}

// Inline banner text for flags whose pipeline-side behaviour is broken or
// limited in a way the user needs to know about before they configure the
// field. Each entry is shown as a red alert directly under the field's
// help text. Keep messages short and concrete — they appear in-form, not
// in a modal. Add new entries here when a pipeline flag needs a "this
// doesn't work the way you'd expect" callout.
const FIELD_WARNINGS = {
    "--responses-api-num-ctx": (
        "⚠ Ollama-only. This field is sent as extra_body={'options': " +
        "{'num_ctx': N}} on every request, which is the Ollama-native " +
        "key. llama.cpp / vLLM / hosted OpenAI ignore it. Use it only " +
        "when --responses-api-base-url points at an Ollama server. For " +
        "llama.cpp, set the context window with `-c N` on the llama " +
        "serve command instead."
    ),
};

function renderField(f, parentTitle) {
    const fieldId = `f-${f.flag}`;
    // ---- Hermes-backend: --model-name is read-only --------------------
    // When the user picks "Hermes Agent" in the LLM tab, the LLM URL
    // is auto-filled with the dashboard's reverse proxy and the
    // pipeline talks to hermes. The model is whatever hermes has
    // loaded — the user changes it via `hermes model` in their
    // terminal. The dashboard has no business pretending to control
    // it, so we hide the editable --model-name input and replace it
    // with a read-only label that fetches the active model from
    // /api/hermes/models. The same label is also shown on the Hermes
    // tab (this is the single source of truth for the displayed
    // model when backend === hermes).
    //
    // We DO keep the underlying state.settings["--model-name"] value
    // untouched (whatever the user typed before flipping to hermes),
    // so flipping back to "direct" restores the editable field with
    // the same value they had. The pipeline's --model-name flag
    // continues to be forwarded regardless — hermes ignores it, but
    // it's harmless and keeps the CLI argv stable.
    if (
        f.flag === '--model-name'
        && f.ui !== 'checkbox'
        && (state.settings['--llm-backend-type'] || '').toLowerCase() === 'hermes'
    ) {
        const hermesLabel = el('div', {
            id: fieldId,
            class: 'field-readonly-model',
            style: {
                padding: '6px 10px',
                background: 'var(--bg-elev, rgba(255,255,255,0.04))',
                border: '1px solid var(--border, rgba(255,255,255,0.1))',
                borderRadius: '4px',
                fontFamily: 'monospace',
                color: 'var(--accent, #6cf)',
                minHeight: '20px',
            },
        }, '(loading hermes model...)');
        const hermesHelp = el('div', { class: 'text-dim', style: { marginTop: '4px', fontSize: '12px' } },
            'Model is set via `hermes model` in your terminal. The ' +
            'dashboard just displays whatever hermes has loaded.');
        const hermesWrap = el('div', { class: 'field' }, [
            el('label', { class: 'field-label', for: fieldId }, [
                f.flag,
                el('span', { class: 'field-flag' }, ''),
            ]),
            hermesLabel,
            hermesHelp,
        ]);
        // Kick off the fetch and update the label as soon as it lands.
        // Idempotent: safe to call multiple times during re-renders.
        _fetchHermesModelInto(hermesLabel);
        return hermesWrap;
    }

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
                const mount = document.getElementById('voice-library-mount-qwen3');
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
    // If this field has an inline warning, append it as a red banner under
    // the help text. ``FIELD_WARNINGS`` is keyed by the CLI flag; missing
    // keys mean no warning is shown, so existing fields render unchanged.
    if (Object.prototype.hasOwnProperty.call(FIELD_WARNINGS, f.flag)) {
        const warning = el(
            'div',
            { class: 'field-warning' },
            FIELD_WARNINGS[f.flag]
        );
        wrapChildren.push(warning);
    }
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
    // 0.4.4+: re-evaluate the Ollama-only field guards on every state
    // change so editing --responses-api-base-url instantly hides/shows
    // --llm-keepalive, --ollama-load-timeout-seconds, --responses-api-
    // num-ctx and the Ollama lifecycle section. Keeps state in sync
    // without forcing a tab re-render.
    const ollamaUrl = _llmBaseUrlLooksLikeOllama();
    for (const id of ['llm-keepalive-section', 'ollama-load-timeout-section', 'ollama-lifecycle-section']) {
        const wrap = document.getElementById(id);
        if (wrap) wrap.style.display = ollamaUrl ? '' : 'none';
    }
    // --responses-api-num-ctx is Ollama-only too (sent as the Ollama-
    // native extra_body={"options":{"num_ctx":N}}; llama.cpp / vLLM /
    // OpenAI ignore it). Hide the whole .field wrap when the URL isn't
    // Ollama.
    const numCtxInput = document.getElementById('f---responses-api-num-ctx');
    if (numCtxInput) {
        const fieldWrap = numCtxInput.closest('.field');
        if (fieldWrap) fieldWrap.style.display = ollamaUrl ? '' : 'none';
    }
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
    // 0.4.3+: auto-unload hook for Ollama num_ctx changes is disabled
    // (v0.5.6). The lifecycle poll is off, so the automatic unload+
    // reload path is off too — avoids background network calls when
    // Ollama is not in use.
    // _maybeAutoUnloadOnNumCtxChange();
}

// 0.4.3+: auto-unload hook state. Tracks the last num_ctx we processed
// so we don't spam Ollama with unloads on every keystroke while the
// user is typing in the field.
let _lastProcessedNumCtx = null;
async function _maybeAutoUnloadOnNumCtxChange() {
    // Bail if the URL isn't Ollama. The lifecycle section itself is
    // hidden in that case, but we still gate here as a safety net.
    if (!_llmBaseUrlLooksLikeOllama()) return;
    const numCtx = _ollamaLifecycleCurrentNumCtx();
    // Skip when the value hasn't changed since the last processed
    // tick. This debounces the per-keystroke fire.
    if (numCtx === _lastProcessedNumCtx) return;
    _lastProcessedNumCtx = numCtx;
    // Skip when there's no model name to act on.
    const model = state.settings['--model-name'];
    if (!model) return;
    // Check whether Ollama is currently holding the model at a
    // *different* context. If it's already at our setting, or not
    // loaded at all, do nothing.
    const ps = await _ollamaLifecycleFetchPs();
    if (!ps || !ps.ok) return;
    const match = (ps.models || []).find((m) => m.name === model);
    if (!match) return;  // not loaded — nothing to unload
    const currentCtx = match.context;
    // If Ollama already reports the context the user wants, do nothing.
    if (currentCtx === numCtx) return;
    // Context differs → trigger an unload + reload at the new size.
    // Silent (no toast) — the user will see the result via the live
    // status badge refresh.
    const res = await _ollamaLifecycleReload(numCtx);
    if (res && res.ok) {
        // Update the lifecycle section's badge immediately so the
        // user sees the new context without waiting for the next
        // 2s tick.
        const lbl = document.getElementById('ollama-lifecycle-model');
        if (lbl) {
            const ctxMsg = res.context != null ? `ctx ${res.context}` : 'ctx (unknown)';
            lbl.textContent = `Model: ${model} · ${ctxMsg} (auto-reloaded)`;
        }
        const lastAction = document.getElementById('ollama-lifecycle-last-action');
        if (lastAction) {
            lastAction.textContent = `Last action: ok — auto-reloaded at ${res.context} ctx (${res.duration_ms}ms)`;
            lastAction.style.color = '';
        }
    } else {
        // Don't surface errors loudly — the manual button is the
        // user-facing way to retry. Just log via the existing logs
        // mechanism if anything.
        console.warn('[ollama-lifecycle] auto-reload failed:', res);
    }
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
    // 0.4.4+: hide the Ollama-only keepalive field when the user is not
    // pointing the dashboard at an Ollama server (e.g. llama.cpp,
    // vLLM, hosted OpenAI). Mirrors the guard renderOllamaLifecycleField
    // already uses. Setting id="llm-keepalive-section" lets any future
    // reactive re-render target it explicitly.
    const wrap = el('div', { class: 'field full', id: 'llm-keepalive-section' }, [label, sel, help]);
    if (!_llmBaseUrlLooksLikeOllama()) {
        wrap.style.display = 'none';
    }
    return wrap;
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
    // 0.4.4+: hide the Ollama-only load-timeout field when the user is
    // not pointing at Ollama. Mirrors the guard renderOllamaLifecycleField
    // uses. The setting itself stays in web_ui_settings.json for users
    // who switch back to Ollama later.
    const wrap = el('div', { class: 'field full', id: 'ollama-load-timeout-section' }, [label, sel, help]);
    if (!_llmBaseUrlLooksLikeOllama()) {
        wrap.style.display = 'none';
    }
    return wrap;
}


// 0.4.2+: LLM request read-timeout dropdown. Picks how long the openai
// SDK (and therefore the pipeline) waits for the LLM to start streaming
// a response before giving up. The pipeline hardcodes a 20 s read
// timeout for every OpenAI-compatible call; if the first reply after
// a model load gets cut off ("Wow I'm a bit slow today, could you
// repeat that?"), this knob lets the user bump it. Per-backend: each
// entry in ``llm_request_timeout_s[<backend>]`` is independent, so
// switching from Hermes (0 = wait forever) to Ollama (120 s by
// default) keeps each backend's tuned value.
//
// Like ``--llm-keepalive`` and ``--ollama-load-timeout-seconds``,
// this is dashboard-only -- the upstream pipeline has no such flag
// (CLAUDE.md forbids editing src/), so we hand-roll it instead of
// going through the introspected renderField path.
const _LLM_REQUEST_TIMEOUT_PRESETS = [
    { value: 0,   label: '0 s — no timeout (wait forever)' },
    { value: 30,  label: '30 s' },
    { value: 60,  label: '60 s' },
    { value: 120, label: '120 s (default for local backends)' },
    { value: 300, label: '300 s (5 min)' },
    { value: 600, label: '600 s (10 min)' },
];
function _activeLlmBackendKey() {
    // Same key resolution as ``get_llm_request_timeout_s`` on the
    // Python side: hermes-proxy wins (key = "hermes"), otherwise the
    // raw ``--llm-backend`` value. Empty string when nothing has been
    // picked yet — the dropdown still renders, it just shows the
    // curated default for an empty key.
    const backendType = (state.settings['--llm-backend-type'] || '').toLowerCase();
    if (backendType === 'hermes') return 'hermes';
    return (state.settings['--llm-backend'] || '').toLowerCase() || 'chat-completions';
}
function _resolveLlmRequestTimeoutValue() {
    // Server-side resolution lives in ``settings_schema.py``. Mirror
    // it here so the dropdown shows the right initial value without
    // a round-trip. We re-read ``state.settings.llm_request_timeout_s``
    // on every call so backend switches re-render with the right
    // value.
    const key = _activeLlmBackendKey();
    const map = state.settings.llm_request_timeout_s;
    if (map && typeof map === 'object') {
        const raw = map[key];
        const n = parseInt(raw, 10);
        if (Number.isFinite(n) && n >= 0) return n;
    }
    // Mirror of ``_LLM_REQUEST_TIMEOUT_DEFAULTS_S`` in settings_schema.py.
    // If you change one, change the other.
    const defaults = {
        hermes: 0, ollama: 120, vllm: 120, 'llama.cpp': 120,
        'chat-completions': 120, 'responses-api': 20,
        'mlx-lm': 120, transformers: 120,
    };
    return defaults[key] != null ? defaults[key] : 120;
}
function renderLlmRequestTimeoutField() {
    const flag = 'llm_request_timeout_s';
    const fieldId = `f-llm-request-timeout-s`;
    const backendKey = _activeLlmBackendKey();
    const curVal = _resolveLlmRequestTimeoutValue();
    const isCurated = _LLM_REQUEST_TIMEOUT_PRESETS.some((c) => c.value === curVal);
    const customSentinel = '__custom__';

    const hoverPreview =
        'Per-backend LLM read timeout (seconds). The pipeline hardcodes a ' +
        '20 s default; bump it if the first reply after a model load gets ' +
        'cut off ("Wow I\'m a bit slow today..."). 0 = wait forever.';
    const helpText =
        'How long the openai SDK waits for the LLM to start streaming a ' +
        'response before giving up. The pipeline hardcodes a 20 s read ' +
        'timeout for every OpenAI-compatible call; if the first reply after ' +
        'a model load gets cut off — you hear the canned "Wow I\'m a bit ' +
        'slow today, could you repeat that?" — this knob fixes it. ' +
        '0 = wait forever (recommended for local models and Hermes). ' +
        '>0 = seconds. Stored per backend so switching backends keeps ' +
        'each one tuned. Current backend: ' + backendKey + '. ' +
        'Defaults: hermes = 0; ollama / vllm / llama.cpp / mlx-lm / ' +
        'transformers = 120 s; responses-api = 20 s (hosted OpenAI / HF).';

    const label = el('label', { class: 'field-label', for: fieldId }, [
        'LLM request timeout (s)',
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

    const sel = el('select', {
        class: 'field-select', id: fieldId,
        onchange: (e) => {
            const v = e.target.value;
            if (v === customSentinel) {
                const freeText = el('input', {
                    class: 'field-input', id: fieldId, type: 'number',
                    min: '0', max: '3600', step: '5',
                    placeholder: 'e.g. 90',
                    oninput: (ev) => {
                        const n = parseInt(ev.target.value, 10);
                        if (!state.settings[flag] || typeof state.settings[flag] !== 'object') {
                            state.settings[flag] = {};
                        }
                        state.settings[flag][backendKey] =
                            Number.isFinite(n) && n >= 0 ? n : 0;
                        applyDisabledStates();
                    },
                });
                freeText.value = String(curVal);
                if (!state.settings[flag] || typeof state.settings[flag] !== 'object') {
                    state.settings[flag] = {};
                }
                state.settings[flag][backendKey] = curVal;
                sel.replaceWith(freeText);
                freeText.focus();
                freeText.select();
                return;
            }
            const n = parseInt(v, 10);
            if (!state.settings[flag] || typeof state.settings[flag] !== 'object') {
                state.settings[flag] = {};
            }
            state.settings[flag][backendKey] =
                Number.isFinite(n) && n >= 0 ? n : 0;
            applyDisabledStates();
        },
    });
    for (const c of _LLM_REQUEST_TIMEOUT_PRESETS) {
        sel.appendChild(el('option', { value: String(c.value) }, c.label));
    }
    sel.appendChild(el('option', { value: customSentinel }, 'Custom (type your own)'));
    if (isCurated) {
        sel.value = String(curVal);
    } else {
        // Saved value isn't curated — switch to Custom so the value
        // is visible rather than silently snapping to the default.
        sel.value = customSentinel;
    }
    // Make sure state.settings[flag] is an object even if it was
    // missing on first load (e.g. settings.json from a 0.4.1
    // install). Mirrors how ``state.settings.hermes`` is treated in
    // the Hermes tab.
    if (!state.settings[flag] || typeof state.settings[flag] !== 'object') {
        state.settings[flag] = {};
    }
    if (state.settings[flag][backendKey] == null) {
        state.settings[flag][backendKey] = curVal;
    }

    // Per-backend badge: tells the user which key this dropdown is
    // editing right now. Updating when the backend changes is the
    // parent's job (it calls renderAll() after a backend change,
    // which rebuilds the field). The badge makes it obvious that
    // switching from "Ollama" to "Hermes Agent" will land you on a
    // different stored value.
    const backendBadge = el('span', {
        class: 'text-dim',
        style: { marginLeft: '8px', fontSize: '12px' },
    }, ' · for backend: ' + backendKey);

    const help = el('div', { class: 'field-help', id: `help-${flag}` }, helpText);
    // ``.field.full`` matches the keepalive/ollama-load-timeout
    // rows above so the dropdown sits full-width and lines up with
    // them. Wrapping the badge in the label row keeps the layout
    // consistent with the other dashboard-only fields.
    const wrap = el('div', { class: 'field full' }, [label, sel, help]);
    label.appendChild(backendBadge);
    return wrap;
}


// 0.4.3+: Ollama model lifecycle section. Renders only when the LLM
// URL looks like Ollama (`:11434` or `/ollama` substring — matches
// the `_looks_like_ollama_url` heuristic on the server side).
//
// Surfaces:
//   - the model name + endpoint + live context window (polled every
//     2s from /api/ollama/ps, same cadence as the Hermes tab)
//   - a "🔄 Unload & reload Ollama model with current ctx" button
//     that calls /api/ollama/reload (unload + warmup at the saved
//     --responses-api-num-ctx)
//   - the result of the last reload (ok / skipped / failed)
//
// Auto-unload behavior on num_ctx change is wired in step 9 below
// (see `_maybeAutoUnloadOnNumCtxChange`).
function _llmBaseUrlLooksLikeOllama() {
    const url = (state.settings['--responses-api-base-url'] || '').toLowerCase();
    return url.includes(':11434') || url.includes('/ollama');
}
async function _ollamaLifecycleFetchPs() {
    const baseUrl = state.settings['--responses-api-base-url'] || '';
    const apiKey = state.settings['--responses-api-api-key'] || 'ollama';
    if (!baseUrl) return { ok: false, models: [], error: 'no base_url' };
    try {
        const params = new URLSearchParams({ base_url: baseUrl });
        if (apiKey) params.set('api_key', apiKey);
        return await getJSON('/api/ollama/ps?' + params.toString());
    } catch (e) {
        return { ok: false, models: [], error: String(e) };
    }
}
async function _ollamaLifecycleReload(numCtx) {
    const baseUrl = state.settings['--responses-api-base-url'] || '';
    const apiKey = state.settings['--responses-api-api-key'] || 'ollama';
    const model = state.settings['--model-name'] || '';
    const keepAlive = state.settings['--llm-keepalive'] || '-1';
    const warmupTimeout = parseInt(state.settings['--ollama-load-timeout-seconds'], 10) || 60;
    if (!model) return { ok: false, message: 'no --model-name', duration_ms: 0, context: null };
    try {
        return await postJSON('/api/ollama/reload', {
            base_url: baseUrl,
            api_key: apiKey,
            model: model,
            num_ctx: numCtx,
            keep_alive: keepAlive,
            warmup_timeout_s: warmupTimeout,
        });
    } catch (e) {
        return { ok: false, message: String(e), duration_ms: 0, context: null };
    }
}
function _ollamaLifecycleCurrentNumCtx() {
    // Same resolver logic as the openai timeout patch reads on the
    // server side: prefer the user's saved value, fall back to no
    // num_ctx (Ollama's model default).
    const raw = state.settings['--responses-api-num-ctx'];
    const n = parseInt(raw, 10);
    return (Number.isFinite(n) && n > 0) ? n : null;
}
function renderOllamaLifecycleField() {
    const wrap = el('div', { class: 'field full', id: 'ollama-lifecycle-section' });
    // Hidden when URL isn't Ollama. The heuristic mirrors the server
    // side so the section is invisible to OpenAI / HF / vLLM users.
    if (!_llmBaseUrlLooksLikeOllama()) {
        wrap.style.display = 'none';
        return wrap;
    }
    wrap.appendChild(el('h3', { style: { marginTop: '24px', borderTop: '1px solid var(--border, rgba(255,255,255,0.1))', paddingTop: '16px' } },
        'Ollama model lifecycle'));
    const info = el('div', { class: 'text-dim', style: { marginBottom: '8px', maxWidth: '720px' } },
        'Live status of the model currently loaded in Ollama. ' +
        'Use after changing --responses-api-num-ctx — Ollama ignores ' +
        'num_ctx on requests for a model that is already resident, so ' +
        'you must unload + reload to apply a new context window.');
    wrap.appendChild(info);
    // Status badge: model / endpoint / context. The context line is
    // what the user cares about most — it's the proof that num_ctx
    // actually took effect.
    const modelLabel = el('div', { id: 'ollama-lifecycle-model',
        style: { fontFamily: 'monospace', padding: '6px 10px',
                 background: 'var(--bg-elev, rgba(255,255,255,0.04))',
                 border: '1px solid var(--border, rgba(255,255,255,0.1))',
                 borderRadius: '4px', minHeight: '20px' } },
        '(loading…)');
    const endpointLabel = el('div', { class: 'text-dim',
        style: { marginTop: '4px', fontSize: '12px' } });
    endpointLabel.textContent = 'Endpoint: ' +
        (state.settings['--responses-api-base-url'] || '');
    wrap.appendChild(modelLabel);
    wrap.appendChild(endpointLabel);
    // Reload button. Disabled until the first /api/ollama/ps lands
    // (so we know Ollama is reachable); spinner while in flight.
    const btn = el('button', { class: 'btn btn-primary', id: 'ollama-lifecycle-reload',
        style: { marginTop: '12px' }, disabled: true },
        '🔄 Unload & reload Ollama model with current ctx');
    const lastAction = el('div', { class: 'text-dim',
        id: 'ollama-lifecycle-last-action',
        style: { marginTop: '8px', fontSize: '12px', minHeight: '18px' } },
        '');
    btn.onclick = async () => {
        if (btn.disabled) return;
        btn.disabled = true;
        const originalText = btn.textContent;
        btn.textContent = '⏳ Reloading…';
        lastAction.textContent = '';
        const numCtx = _ollamaLifecycleCurrentNumCtx();
        const res = await _ollamaLifecycleReload(numCtx);
        btn.textContent = originalText;
        btn.disabled = false;
        const dur = (res && res.duration_ms) ? `${res.duration_ms}ms` : '';
        if (res && res.ok) {
            const ctxMsg = res.context ? ` at ${res.context} ctx` : '';
            lastAction.textContent = `Last action: ok — reloaded${ctxMsg}${dur ? ' (' + dur + ')' : ''}`;
            lastAction.style.color = '';
        } else {
            const msg = (res && res.message) ? res.message : 'unknown error';
            lastAction.textContent = `Last action: failed — ${msg}`;
            lastAction.style.color = 'var(--danger, #d33)';
        }
        // Refresh the badge immediately so the user sees the new context
        // without waiting for the next 2s tick.
        await _ollamaLifecycleTick();
    };
    wrap.appendChild(btn);
    wrap.appendChild(lastAction);
    // Poll /api/ollama/ps every 2s while the LLM tab is visible. Same
    // cadence as the Hermes tab — matches the dashboard's "feels
    // live" target without hammering Ollama.
    let pollTimer = null;
    async function _ollamaLifecycleTick() {
        const r = await _ollamaLifecycleFetchPs();
        if (!r || !r.ok) {
            const reason = (r && r.error) ? r.error : 'unreachable';
            modelLabel.textContent = `Ollama: ${reason}`;
            modelLabel.style.color = 'var(--danger, #d33)';
            btn.disabled = true;
            return;
        }
        modelLabel.style.color = '';
        btn.disabled = false;
        const models = r.models || [];
        if (models.length === 0) {
            modelLabel.textContent = '(no models loaded — Ollama is idle)';
            return;
        }
        // Match the live model by name; if none matches, show the
        // first one with a warning so the user knows which model is
        // actually resident (might be a different one than their
        // --model-name setting).
        const wantedName = state.settings['--model-name'] || '';
        const match = models.find((m) => m.name === wantedName) || models[0];
        const ctxStr = (match.context != null) ? `ctx ${match.context}` : 'ctx (unknown)';
        const sizeStr = (match.size_vram != null) ? ` · ${(match.size_vram / 1e9).toFixed(1)} GB VRAM` : '';
        const untilStr = match.until ? ` · until ${match.until}` : '';
        const matchedHere = match.name === wantedName;
        modelLabel.textContent =
            (matchedHere ? `Model: ${match.name}` : `Model: ${match.name} (warning: --model-name is ${wantedName})`) +
            ` · ${ctxStr}${sizeStr}${untilStr}`;
    }
    // Helper exposed on window for the global visibility-check loop.
    // v0.5.6: automatic polling is disabled to avoid wasting CPU/
    // network on Ollama when the lifecycle section isn't needed. We do
    // one manual refresh when the tab becomes active so the badge isn't
    // stuck on "(loading…)"; the reload button still fetches on demand.
    function _start() {
        if (pollTimer) return;
        _ollamaLifecycleTick();
        pollTimer = true; // sentinel: no setInterval, just guard re-entry
    }
    function _stop() {
        if (pollTimer) { pollTimer = null; }
    }
    // Start polling immediately if the LLM tab is currently active.
    const llmTab = document.getElementById('tab-llm');
    if (llmTab && llmTab.classList.contains('active')) _start();
    // Wire to the visibility-check loop that already runs every few
    // seconds. We piggy-back on the existing poll by checking the
    // LLM tab's visibility inside our own interval above.
    // Stash the timers so renderAll() can stop them.
    wrap._ollamaLifecycleStart = _start;
    wrap._ollamaLifecycleStop = _stop;
    // Also stop on renderAll: each render rebuilds the field, and we
    // don't want the old interval leaking. The new render's interval
    // takes over.
    setTimeout(() => {
        // Defer one tick so renderAll's caller has a chance to wire
        // the new lifecycle section in.
        const old = document.getElementById('ollama-lifecycle-section');
        if (old && old !== wrap) {
            if (old._ollamaLifecycleStop) old._ollamaLifecycleStop();
        }
        // Restart polling on the new section if the LLM tab is active.
        const llmTab2 = document.getElementById('tab-llm');
        if (llmTab2 && llmTab2.classList.contains('active')) _start();
    }, 0);
    return wrap;
}


// 0.4.0+: "Backend type" dropdown for the LLM tab. Picks between
// "Direct backend" (today's behavior — the pipeline talks to whatever
// URL is in --responses-api-base-url, typically local Ollama) and
// "Hermes Agent" (the pipeline talks to the dashboard's reverse proxy
// at /hermes-proxy/v1, which forwards to the hermes subprocess with
// the right Authorization + X-Hermes-Session-Id headers).
//
// Switching to "hermes" auto-fills the LLM URL / api_key / backend
// fields so the user doesn't have to type them by hand. Switching
// back to "direct" leaves whatever the user typed in place — they
// can edit it again from there. The value lives in
// ``state.settings["--llm-backend-type"]`` (dashboard-only; the
// pipeline never sees it).
//
// This is hand-rolled (not via the introspected renderField path)
// because the upstream pipeline has no such flag — the auto-fill is
// pure dashboard orchestration, see docs/HERMES_TAB_PLAN.md §4.
const _HERMES_BACKEND_VALUES = [
    { value: "direct", label: "Direct backend  (ollama / vLLM / llama.cpp / OpenAI)" },
    { value: "hermes", label: "Hermes Agent    (skills + memory + HA control)" },
];

// Where the dashboard's hermes reverse proxy is reachable. Computed
// from the dashboard's own host:port so the user doesn't have to
// hard-code it. We grab the dashboard's URL via window.location so
// this works behind a reverse proxy too (the proxy preserves the
// public origin).
function _hermesProxyBaseUrl() {
    // window.location.origin is "http(s)://host:port" — we drop the
    // trailing slash and append the proxy mount prefix.
    const o = (window.location && window.location.origin) || '';
    return o.replace(/\/+$/, '') + '/hermes-proxy/v1';
}

function renderHermesBackendField() {
    const flag = '--llm-backend-type';
    const fieldId = `f-${flag}`;
    const hoverPreview =
        'Picks the LLM backend the pipeline talks to. "Direct" talks to the URL ' +
        'and api_key fields below (today\'s behavior — Ollama, vLLM, llama.cpp, ' +
        'OpenAI, …). "Hermes Agent" routes through the dashboard\'s reverse ' +
        'proxy to a hermes-agent subprocess on this machine, so the robot ' +
        'gets skills, memory, and Home Assistant control.';
    const helpText =
        '"Direct backend" (default) makes the pipeline talk directly to the ' +
        'URL + api_key + model fields below — your local Ollama, a remote ' +
        'vLLM, llama.cpp, OpenAI, etc. ' +
        '"Hermes Agent" makes the pipeline talk to the dashboard\'s reverse ' +
        'proxy at /hermes-proxy/v1, which forwards to the hermes-agent ' +
        'subprocess you manage from the Hermes tab. ' +
        'When you pick Hermes Agent, the LLM URL is auto-filled with the ' +
        'proxy URL and the api_key is auto-filled with the Hermes tab\'s ' +
        'API key — both can be edited afterwards. The pipeline gets a ' +
        'session id (X-Hermes-Session-Id) automatically so your ' +
        'conversation keeps memory across turns.';

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

    // Default to "direct" if the field hasn't been set yet — keeps
    // today's behavior for users who never touch the toggle.
    if (state.settings[flag] == null) state.settings[flag] = 'direct';
    const curVal = String(state.settings[flag]);

    const sel = el('select', {
        class: 'field-select', id: fieldId,
        onchange: (e) => {
            const v = e.target.value;
            state.settings[flag] = v;
            if (v === 'hermes') {
                // Auto-fill the LLM URL + api_key + backend so the
                // pipeline points at the dashboard's reverse proxy.
                // We pull the api_key from the persisted hermes
                // block — it's the one the dashboard generated on
                // first /api/hermes/start.
                const hermesCfg = (state.settings.hermes && typeof state.settings.hermes === 'object')
                    ? state.settings.hermes : {};
                const apiKey = hermesCfg.api_key || '';
                state.settings['--responses-api-base-url'] = _hermesProxyBaseUrl();
                state.settings['--responses-api-api-key'] = apiKey;
                state.settings['--llm-backend'] = 'chat-completions';
            }
            // Re-render the LLM tab so the URL / api_key fields show
            // the new values. Cheap (a single DOM rebuild).
            renderAll();
        },
    });
    for (const o of _HERMES_BACKEND_VALUES) {
        const opt = el('option', { value: o.value }, o.label);
        if (o.value === curVal) opt.selected = true;
        sel.appendChild(opt);
    }
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
    // User is rebuilding the view (e.g. changed the log filter) — jump
    // to the bottom regardless of where they were scrolled.
    for (const line of state._logBuffer || []) {
        if (!show(line.level)) continue;
        con.appendChild(el('div', { class: `log-line ${line.level}` }, line.text));
    }
    con.scrollTop = con.scrollHeight;
}

// True iff the log console is scrolled to (or close to) the bottom. Used
// to decide whether new log lines should auto-scroll the view. When the
// user has scrolled up to read earlier output, we leave them alone —
// pinning to the bottom is hostile to anyone trying to read a backtrace.
function isLogConsoleAtBottom(con) {
    // 24px tolerance: anything within a couple of lines of the bottom
    // counts as "at the bottom". Once the user scrolls past that
    // threshold, auto-scroll disengages until they scroll back down.
    return (con.scrollHeight - con.scrollTop - con.clientHeight) < 24;
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
        // Snapshot the user's scroll position BEFORE we mutate the DOM,
        // because appending a child changes scrollHeight and would mask
        // whether they had scrolled up to read.
        const wasAtBottom = isLogConsoleAtBottom(con);
        con.appendChild(el('div', { class: `log-line ${line.level}` }, line.text));
        if (wasAtBottom) {
            // User was following the tail — keep them there.
            con.scrollTop = con.scrollHeight;
        }
        // Otherwise: user scrolled up to read; don't yank them back down.
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

// ---- Hermes Agent tab ------------------------------------------------
//
// All hermes-agent controls live here (plan §3: "all hermes-agent
// functionality lives in this tab, no leaks to other tabs"). The
// tab shows:
//   - a status badge (running? port? model? ready?)
//   - Start / Polite Stop buttons
//   - Filler audio config (on/off + up to 20 editable phrase boxes)
//   - A text console that streams /api/hermes/chat SSE
//   - Logs filtered to source === "hermes" from the shared /ws/logs
//   - Cancel / Kill buttons in a "Emergency" section
//
// State is re-fetched from /api/hermes/status every 2s while the tab
// is active; the rest of the dashboard polls less often.
//
// The pipeline's LLM slot talks to hermes via /hermes-proxy/v1/* on
// the dashboard's own port (see web_ui/hermes_proxy.py). When the
// user picks "Hermes Agent" in the LLM tab, --responses-api-base-url
// is auto-filled with that URL.

let _hermesChatAbort = null;  // AbortController for the in-flight SSE

function renderHermesTab(tab) {
    tab.textContent = '';
    tab.appendChild(el('div', { class: 'tab-header' }, [
        el('h1', { class: 'tab-title' }, 'Hermes Agent'),
        el('div', { class: 'tab-subtitle' },
            'Manage the hermes-agent subprocess. Pick "Hermes Agent" in the LLM tab ' +
            'to route the voice pipeline through hermes (skills, memory, Home Assistant).'),
    ]));

    // ---- Status badge (top-right of the header) ----------------------
    const badge = el('div', { id: 'hermes-status-badge', class: 'hermes-badge' });
    tab.appendChild(el('div', { class: 'hermes-header-row' }, [
        el('div', {}, []),
        badge,
    ]));

    // ---- Lifecycle buttons ------------------------------------------
    const btnStart = el('button', { class: 'btn btn-primary btn-large',
        onclick: () => hermesStart(btnStart, btnStop) }, '▶ Start');
    const btnStop  = el('button', { class: 'btn btn-large',
        onclick: () => hermesStop() }, '■ Polite Stop');
    const btnCancel = el('button', { class: 'btn btn-warning btn-large',
        onclick: () => hermesCancel() }, '⏸ Cancel');
    const btnKill  = el('button', { class: 'btn btn-danger btn-large',
        onclick: () => hermesKillConfirm() }, '✖ Kill Hermes');
    const btnResetSession = el('button', { class: 'btn btn-large',
        onclick: () => hermesResetSession() }, '↻ Reset session');
    const btnOpen = el('a', {
        class: 'btn', href: 'http://127.0.0.1:9119',
        target: '_blank', rel: 'noopener noreferrer',
    }, '↗ Open Hermes Dashboard');
    tab.appendChild(el('div', { class: 'btn-row' }, [btnStart, btnStop]));

    // ---- Hermes api_server endpoint ----------------------------------
    // These three values control where the dashboard's reverse proxy
    // (and therefore the pipeline's LLM slot) talks to hermes-agent.
    // They are NOT the upstream LLM that hermes uses internally; that is
    // configured via `hermes config` in your terminal.
    tab.appendChild(el('h3', { style: { marginTop: '24px' } }, 'Hermes endpoint'));
    tab.appendChild(el('div', { class: 'text-dim', style: { marginBottom: '12px', maxWidth: '720px' } },
        'Host, port, and advertised model for the hermes api_server. ' +
        'The pipeline connects here via the Hermes Agent toggle in the LLM tab. ' +
        'The real LLM that hermes talks to is set via `hermes config` in your terminal.'));

    // Ensure state.settings.hermes is an object so the form has
    // somewhere to write. Mirrors how state.settings.env is treated
    // elsewhere — never trust the server shape.
    if (!state.settings.hermes || typeof state.settings.hermes !== 'object') {
        state.settings.hermes = {};
    }

    const _hermesCfg = () => state.settings.hermes;
    const _hermesVal = (key, fallback) => {
        const v = _hermesCfg()[key];
        return v != null ? v : fallback;
    };
    const _setHermesVal = (key, value) => { _hermesCfg()[key] = value; };

    // Persist the whole hermes block to disk on blur. We write the
    // entire object so concurrent edits to different fields don't
    // clobber each other.
    const _persistHermesConfig = async () => {
        try {
            await postJSON('/api/settings/patch', { hermes: _hermesCfg() });
            state.saved = true;
        } catch (e) {
            toast('Hermes config save failed: ' + e.message, 'error', 4000);
        }
    };

    const hostInput = el('input', {
        type: 'text', id: 'hermes-host', class: 'field-input',
        style: { width: '160px' },
        oninput: (e) => _setHermesVal('host', e.target.value),
        onblur: () => _persistHermesConfig(),
    });
    hostInput.value = _hermesVal('host', '127.0.0.1');

    const portInput = el('input', {
        type: 'number', id: 'hermes-port', class: 'field-input',
        min: '1', max: '65535', step: '1', style: { width: '100px' },
        oninput: (e) => _setHermesVal('port', parseInt(e.target.value, 10) || 8642),
        onblur: () => _persistHermesConfig(),
    });
    portInput.value = String(_hermesVal('port', 8642));

    const _hermesField = (labelText, inputEl, helpText) => {
        const hover = helpText.length > 220 ? helpText.slice(0, 217) + '…' : helpText;
        return el('div', { class: 'field' }, [
            el('label', { class: 'field-label', for: inputEl.id }, [
                labelText,
                el('span', { class: 'field-flag' }, ''),
                el('button', {
                    type: 'button', class: 'help-btn', title: hover,
                    onclick: (e) => {
                        e.preventDefault();
                        const field = e.currentTarget.closest('.field');
                        const help = field && field.querySelector('.field-help');
                        if (help) help.classList.toggle('visible');
                    },
                }, '?'),
            ]),
            inputEl,
            el('div', { class: 'field-help' }, helpText),
        ]);
    };

    tab.appendChild(_hermesField(
        'api_server host',
        hostInput,
        'Bind address for the hermes api_server. Default 127.0.0.1 keeps it on loopback. ' +
        'Change only if you run hermes on a different machine and have network routing in place.'
    ));
    tab.appendChild(_hermesField(
        'api_server port',
        portInput,
        'TCP port for the hermes api_server. Default 8642. The pipeline connects through ' +
        'the dashboard reverse proxy, so this port only needs to be reachable from the dashboard.'
    ));

    // Read-only status line: actual running model + endpoint.
    // The model is whatever hermes reports on /v1/models; the user picks
    // it via `hermes config` / `hermes model` in their terminal, not here.
    const modelLine = el('div', { class: 'text-dim', id: 'hermes-model-line' },
        'Model: (start hermes to see)');
    tab.appendChild(modelLine);
    const modelHint = el('div', { class: 'text-dim', id: 'hermes-model-hint',
        style: { fontSize: '12px', marginTop: '4px' } },
        'Model is picked inside hermes (`hermes config` / `hermes model`). ' +
        'The dashboard only shows what hermes reports.');
    tab.appendChild(modelHint);

    // ---- Hermes stderr verbosity knob --------------------------------
    const logLevelSelect = el('select', {
        id: 'hermes-log-level', class: 'field-select',
        style: { width: '160px' },
        onchange: (e) => {
            _setHermesVal('log_level', e.target.value);
            _persistHermesConfig();
        },
    }, [
        el('option', { value: 'default' }, 'default'),
        el('option', { value: 'verbose' }, 'verbose (-v)'),
        el('option', { value: 'debug' }, 'debug (-vv)'),
    ]);
    logLevelSelect.value = _hermesVal('log_level', 'verbose');
    tab.appendChild(_hermesField(
        'stderr verbosity',
        logLevelSelect,
        'How chatty hermes is on its stderr stream. Verbose sends INFO ' +
        'logs to the dashboard log panel; debug sends DEBUG. The dashboard ' +
        'always tails ~/.hermes/logs/agent.log regardless. Change requires ' +
        'Stop + Start to take effect.'
    ));
    tab.appendChild(el('div', { class: 'text-dim', style: {
        fontSize: '12px', marginTop: '-8px', marginBottom: '12px', maxWidth: '720px'
    } },
        'Restart hermes after changing this for it to take effect.'));

    // ---- LLM read timeout knob (Hermes-only) --------------------------
    // The pipeline hardcodes a 20 s read timeout for every chat /
    // response call. When a user asks hermes to read a long passage,
    // or a large model takes longer than 20 s to first byte, the
    // pipeline's httpx client times out and the canned fallback
    // ``"Wow I'm a bit slow today, could you repeat that?"`` is
    // spoken by the TTS — to the user, hermes looks stuck in a loop.
    //
    // The knob lives here (not in the LLM tab) because it only
    // matters when the LLM backend is Hermes. For other backends
    // (direct Ollama / OpenAI / vLLM / llama.cpp) the SDK defaults
    // are fine. The knob installs a monkey-patch on the openai SDK
    // when the pipeline subprocess starts (see web_ui/process_manager.py
    // and web_ui/_openai_timeout_patch.py) — and the patch is removed
    // on pipeline stop.
    //
    // 0 = no read timeout at all (httpx.Timeout(None)); the pipeline
    //     waits forever for the LLM to close the stream.
    // >0 = read timeout in seconds (httpx.Timeout(N)).
    tab.appendChild(el('h3', { style: { marginTop: '24px' } }, 'LLM read timeout'));
    const timeoutHover =
        'Max seconds the pipeline waits for the LLM (hermes) to send ' +
        'its response before giving up and playing the canned fallback ' +
        '"Wow I\'m a bit slow today...". 0 = wait forever. Increase ' +
        'this ONLY if you actually want hermes to give up faster on ' +
        'long-running prompts (rare).';
    const timeoutHelpText =
        'How long the pipeline waits for hermes to start streaming a ' +
        'response before giving up. The pipeline\'s own hardcoded ' +
        'default is 20 s — too short for large models or long passages, ' +
        'which produces the canned "Wow I\'m a bit slow today" reply. ' +
        '0 = wait forever (recommended). >0 = seconds. Only matters ' +
        'when the LLM backend is "Hermes Agent" in the LLM tab; for ' +
        'other backends the openai SDK\'s own defaults apply.';
    // Ensure state.settings.hermes is an object so the form has
    // somewhere to write. Mirrors how state.settings.env is treated
    // elsewhere — never trust the server shape.
    if (!state.settings.hermes || typeof state.settings.hermes !== 'object') {
        state.settings.hermes = {};
    }
    let timeoutVal = parseInt(state.settings.hermes.read_timeout_s, 10);
    if (!Number.isFinite(timeoutVal) || timeoutVal < 0) timeoutVal = 0;
    const timeoutInput = el('input', {
        type: 'number', id: 'hermes-read-timeout',
        class: 'field-input', min: '0', step: '5',
        style: { width: '120px' },
        oninput: (e) => {
            const raw = parseInt(e.target.value, 10);
            const v = Number.isFinite(raw) && raw >= 0 ? raw : 0;
            state.settings.hermes.read_timeout_s = v;
        },
    });
    timeoutInput.value = String(timeoutVal);
    const timeoutLabel = el('label', { class: 'field-label', for: 'hermes-read-timeout' }, [
        '--hermes.read_timeout_s',
        el('span', { class: 'field-flag' }, ''),
        el('button', {
            type: 'button', class: 'help-btn', title: timeoutHover,
            onclick: (e) => {
                e.preventDefault();
                const field = e.currentTarget.closest('.field');
                const help = field && field.querySelector('.field-help');
                if (help) help.classList.toggle('visible');
            },
        }, '?'),
    ]);
    const timeoutHelp = el('div', { class: 'field-help', id: 'help-hermes.read_timeout_s' },
        timeoutHelpText);
    // Inline note about the semantics of 0 — separate from the help
    // text so the user sees it without clicking the ? button.
    const timeoutNote = el('div', { class: 'text-dim', style: { marginTop: '4px', fontSize: '12px' } },
        '0 = wait forever (infinite). Increase only if you want hermes ' +
        'to give up faster on long-running prompts.');
    tab.appendChild(el('div', { class: 'field' }, [
        timeoutLabel,
        timeoutInput,
        timeoutHelp,
        timeoutNote,
    ]));

    // ---- Filler audio config ----------------------------------------
    tab.appendChild(el('h3', { style: { marginTop: '24px' } }, 'Filler audio'));
    tab.appendChild(el('div', { class: 'text-dim', style: { marginBottom: '8px', maxWidth: '720px' } },
        'Plays a short phrase from this list while hermes is mid-tool-call ' +
        '(>1.5s without text), so the user knows the agent is still working. ' +
        'Uses the pipeline\'s TTS — no extra config. One phrase is picked at random.'));

    const fillerToggle = el('input', {
        type: 'checkbox', id: 'hermes-filler-enabled',
        onchange: () => hermesFillerSave({ enabled: fillerToggle.checked }),
    });
    const fillerDelayInput = el('input', {
        type: 'number',
        id: 'hermes-filler-delay',
        class: 'field-input',
        min: 0,
        max: 30000,
        step: 100,
        value: (state.settings.hermes.filler_delay_ms ?? 1500),
        style: { width: '90px', marginLeft: '16px' },
        onchange: () => {
            let v = parseInt(fillerDelayInput.value, 10);
            if (Number.isNaN(v)) v = 1500;
            v = Math.max(0, Math.min(30000, v));
            fillerDelayInput.value = v;
            hermesFillerSave({ delay_ms: v });
        },
    });
    tab.appendChild(el('div', { class: 'field', style: { display: 'flex', alignItems: 'center', flexWrap: 'wrap' } }, [
        fillerToggle,
        el('label', { for: 'hermes-filler-enabled', style: { marginLeft: '8px' } },
            ' Enable filler phrases'),
        fillerDelayInput,
        el('label', { for: 'hermes-filler-delay', style: { marginLeft: '6px' } },
            'Delay (ms)'),
    ]));

    const MAX_FILLER_PHRASES = 20;
    const fillerContainer = el('div', { id: 'hermes-filler-container', style: { maxWidth: '720px' } });
    tab.appendChild(fillerContainer);

    function _getFillerPhrases() {
        const container = document.getElementById('hermes-filler-container');
        if (!container) return [];
        const out = [];
        for (const input of container.querySelectorAll('.hermes-filler-input')) {
            const t = (input.value || '').trim();
            if (t) out.push(t);
        }
        return out.slice(0, MAX_FILLER_PHRASES);
    }

    function _saveFillerPhrases() {
        const phrases = _getFillerPhrases();
        hermesFillerSave({ phrases });
        // Re-render so empty boxes collapse and we never have more than one
        // trailing empty slot.
        _renderFillerBoxes(phrases);
    }

    function _renderFillerBoxes(phrases) {
        const container = document.getElementById('hermes-filler-container');
        if (!container) return;
        container.textContent = '';

        // Always render each saved phrase plus exactly one empty slot for
        // adding a new phrase, unless we're already at the cap.
        const values = [...phrases];
        if (values.length < MAX_FILLER_PHRASES) {
            values.push('');
        }

        values.forEach((text, idx) => {
            const isEmptySlot = !text && idx === phrases.length;
            const row = el('div', { class: 'hermes-filler-row', style: {
                display: 'flex', gap: '8px', alignItems: 'center', marginBottom: '6px'
            } });
            const input = el('input', {
                type: 'text',
                class: 'field-input hermes-filler-input',
                placeholder: isEmptySlot ? 'type a filler phrase...' : '',
                style: { flex: '1' },
                value: text,
                onblur: () => _saveFillerPhrases(),
                onkeydown: (e) => {
                    if (e.key === 'Enter') {
                        e.preventDefault();
                        _saveFillerPhrases();
                    }
                },
            });
            row.appendChild(input);

            // Delete button only for non-empty saved phrases. The empty slot
            // has no delete button; it exists purely for adding.
            if (text) {
                const del = el('button', {
                    type: 'button',
                    class: 'btn btn-danger',
                    style: { padding: '2px 8px', fontSize: '12px' },
                    onclick: () => {
                        const next = phrases.filter((_, i) => i !== idx);
                        hermesFillerSave({ phrases: next });
                        _renderFillerBoxes(next);
                    },
                }, '✕');
                row.appendChild(del);
            }

            container.appendChild(row);
        });

        // Only show the explicit "Add phrase" button when there's room and
        // the last visible slot is already filled (so the user sees a clear
        // affordance).
        if (phrases.length < MAX_FILLER_PHRASES) {
            const addBtn = el('button', {
                type: 'button',
                class: 'btn',
                style: { marginTop: '6px' },
                onclick: () => {
                    const next = [...phrases, ''];
                    _renderFillerBoxes(next);
                    // Focus the new empty box.
                    const inputs = container.querySelectorAll('.hermes-filler-input');
                    const last = inputs[inputs.length - 1];
                    if (last) last.focus();
                },
            }, '+ Add phrase');
            container.appendChild(addBtn);
        }
    }

    // ---- Console (text chat with hermes) ----------------------------
    tab.appendChild(el('h3', { style: { marginTop: '24px' } }, 'Console'));
    tab.appendChild(el('div', { class: 'text-dim', style: { marginBottom: '8px' } },
        'Text chat with hermes — useful for debugging skills without ' +
        'talking to the robot. Streams over Server-Sent Events.'));
    const consoleBox = el('div', { id: 'hermes-console', class: 'log-console',
        style: { height: '260px', maxWidth: '720px' } });
    tab.appendChild(consoleBox);
    const chatInput = el('input', {
        type: 'text', class: 'field-input', id: 'hermes-chat-input',
        placeholder: 'Type a message for hermes…', style: { width: '60%', maxWidth: '500px' },
        onkeydown: (e) => { if (e.key === 'Enter') hermesSendChat(); },
    });
    const chatSend = el('button', { class: 'btn btn-primary', onclick: () => hermesSendChat() },
        'Send');
    const chatAbort = el('button', { class: 'btn', onclick: () => hermesAbortChat() },
        'Stop');
    tab.appendChild(el('div', { class: 'btn-row', style: { marginTop: '8px' } },
        [chatInput, chatSend, chatAbort]));

    // ---- Logs (filtered to source === "hermes") ----------------------
    tab.appendChild(el('h3', { style: { marginTop: '24px' } }, 'Logs'));
    tab.appendChild(el('div', { class: 'text-dim', style: { marginBottom: '8px' } },
        'Live hermes subprocess logs. The shared /ws/logs websocket ' +
        'already tags each line with `source: "hermes"` or "pipeline"; ' +
        'we filter to hermes here. The full pipeline log is still ' +
        'on the Status & Logs tab.'));
    const hermesLogBox = el('div', { id: 'hermes-log-box', class: 'log-console',
        style: { height: '460px', maxWidth: 'none', width: '100%', overflow: 'auto' } });
    tab.appendChild(hermesLogBox);
    const hermesLogFilter = el('select', {
        class: 'field-select', id: 'hermes-log-filter',
        onchange: () => _hermesRenderLogs(),
    }, []);
    for (const v of ['ALL', 'INFO', 'WARNING', 'ERROR', 'DEBUG']) {
        hermesLogFilter.appendChild(el('option', { value: v }, v));
    }
    tab.appendChild(el('div', { class: 'btn-row' }, [
        el('span', { class: 'text-dim' }, 'Filter:'),
        hermesLogFilter,
        el('button', { class: 'btn', onclick: () => {
            _hermesLogBuffer = []; _hermesRenderLogs();
        } }, 'Clear'),
    ]));

    // ---- Emergency --------------------------------------------------
    tab.appendChild(el('h3', { style: { marginTop: '24px', color: 'var(--danger, #d33)' } },
        'Emergency'));
    tab.appendChild(el('div', { class: 'text-dim', style: { marginBottom: '8px' } },
        'Cancel stops the current hermes run cleanly (context preserved). ' +
        'Kill forces SIGKILL on the subprocess (context lost, fresh start). ' +
        'Kill requires a confirmation prompt — it is never auto-triggered.'));
    tab.appendChild(el('div', { class: 'btn-row' }, [btnCancel, btnKill, btnResetSession, btnOpen]));

    // ---- Initial paint + ongoing poll -------------------------------
    // Render filler boxes from saved phrases on first paint.
    _renderFillerBoxes(state.settings.hermes.filler_phrases || []);
    _hermesRefetchLogs();
    _hermesRefreshStatus();
    if (!_hermesPollTimer) _hermesPollTimer = setInterval(_hermesTick, 2000);
}

let _hermesPollTimer = null;
async function _hermesTick() {
    // Only poll if the tab is currently visible — saves a fetch on
    // every other tab the user is on.
    const tab = document.getElementById('tab-hermes');
    if (tab && tab.classList.contains('active')) {
        await Promise.all([_hermesRefreshStatus(), _hermesRefetchLogs()]);
    }
    // If the LLM tab is active AND --llm-backend-type === 'hermes',
    // refresh the read-only model label so `hermes model` changes in
    // the user's terminal show up within a few seconds. Cheap (one
    // fetch, only when both conditions hold).
    const llmTab = document.getElementById('tab-llm');
    if (llmTab && llmTab.classList.contains('active')) {
        const labels = document.querySelectorAll('#tab-llm .field-readonly-model');
        if (labels.length > 0) {
            // Each --model-name subgroup has its own read-only label;
            // refresh them all.
            for (const lab of labels) {
                _fetchHermesModelInto(lab);
            }
        }
    }
}

async function _hermesRefreshStatus() {
    let s = {};
    try { s = await getJSON('/api/hermes/status'); } catch (e) { /* offline */ }
    const badge = document.getElementById('hermes-status-badge');
    if (badge) {
        badge.textContent = '';
        badge.className = 'hermes-badge ' +
            (s.running ? (s.ready ? 'running' : 'starting') : 'stopped');
        badge.appendChild(el('span', { class: 'hermes-badge-dot' }, ''));
        const txt = s.running
            ? `${s.ready ? '● running' : '◐ starting'} :${s.port}` +
              (s.uptime_s ? `  (${formatUptime(s.uptime_s)})` : '')
            : '○ stopped';
        badge.appendChild(el('span', {}, txt));
    }
    // The model line is read-only: it shows whatever hermes reports,
    // not whatever the dashboard's --model-name field claims. The
    // user changes the model via `hermes model` in the terminal; the
    // dashboard has no business pretending to control it.
    //
    // /api/hermes/status returns `model_name` from
    // HermesProcess._status() which already queries /v1/models, so
    // we use it as the primary source. /api/hermes/models is a
    // secondary signal — if status didn't report a model but
    // /v1/models does, prefer that (rare but possible during
    // model swap).
    let modelName = s.running ? (s.model_name || '') : '';
    try {
        const r = await getJSON('/api/hermes/models');
        if (r && r.models && r.models.length === 1 && !modelName) {
            modelName = r.models[0].id;
        }
    } catch (e) { /* offline — keep the status-derived value */ }
    const modelLine = document.getElementById('hermes-model-line');
    if (modelLine) {
        modelLine.textContent = s.running
            ? `Model: ${modelName || '(unknown)'}    Endpoint: http://${s.host}:${s.port}/v1`
            : 'Model: (start hermes to load)';
    }
    // Refresh filler enabled toggle + delay once on first paint. We do not
    // re-render the per-phrase boxes here because the 2 s poll would
    // overwrite whatever the user is currently typing.
    const fillerToggle = document.getElementById('hermes-filler-enabled');
    const fillerDelayInput = document.getElementById('hermes-filler-delay');
    if (fillerToggle && fillerToggle.dataset.loaded !== '1') {
        try {
            const f = await getJSON('/api/hermes/filler');
            fillerToggle.checked = !!f.enabled;
            if (fillerDelayInput) {
                let v = parseInt(f.delay_ms, 10);
                if (Number.isNaN(v)) v = 1500;
                v = Math.max(0, Math.min(30000, v));
                fillerDelayInput.value = v;
            }
            fillerToggle.dataset.loaded = '1';
        } catch (e) { /* offline */ }
    }
}

let _hermesLogBuffer = [];  // [{index, level, text, ts}, ...]
let _hermesLogIndex = 0;
async function _hermesRefetchLogs() {
    try {
        // We piggy-back on the shared log websocket; the ``source``
        // tag tells us which lines belong to hermes. On first load
        // we also pull the existing buffer so the panel isn't empty
        // until the next message arrives.
        const r = await getJSON('/api/logs?since=0');
        const lines = (r.lines || []).filter(l => l.source === 'hermes');
        if (lines.length) {
            const lastIdx = lines[lines.length - 1].index;
            if (lastIdx > _hermesLogIndex) {
                _hermesLogIndex = lastIdx;
                _hermesLogBuffer = _hermesLogBuffer.concat(lines);
                // Cap to last 500 lines so the DOM doesn't grow forever.
                if (_hermesLogBuffer.length > 500) {
                    _hermesLogBuffer = _hermesLogBuffer.slice(-500);
                }
                _hermesRenderLogs();
            }
        }
    } catch (e) { /* offline */ }
}

function _hermesRenderLogs() {
    const box = document.getElementById('hermes-log-box');
    if (!box) return;
    const filterEl = document.getElementById('hermes-log-filter');
    const filter = filterEl ? filterEl.value : 'ALL';
    box.textContent = '';
    for (const l of _hermesLogBuffer) {
        if (filter !== 'ALL' && l.level !== filter) continue;
        const row = el('div', { class: 'log-row' }, `[${l.level || 'INFO'}] ${l.text}`);
        box.appendChild(row);
    }
    box.scrollTop = box.scrollHeight;
}

// Wire the shared /ws/logs websocket to also feed the hermes log
// panel — without this, the hermes panel only refreshes on the 2s
// poll, which feels laggy. The pipeline tab does the same trick on
// the same websocket; we just filter to source === 'hermes'.
function _hermesSubscribeLogStream() {
    if (window._hermesLogSubscribed) return;
    window._hermesLogSubscribed = true;
    // Wait for the global ws (started by startLogStream) — it might
    // not be open yet at boot. We poll for state.ws every 500ms.
    const wire = () => {
        const ws = state && state.ws;
        if (!ws) { setTimeout(wire, 500); return; }
        const orig = ws.onmessage;
        ws.onmessage = (e) => {
            if (typeof orig === 'function') orig(e);
            try {
                const msg = JSON.parse(e.data);
                if (msg.source !== 'hermes') return;
                if (msg.index == null || msg.index <= _hermesLogIndex) return;
                _hermesLogIndex = msg.index;
                _hermesLogBuffer.push(msg);
                if (_hermesLogBuffer.length > 500) {
                    _hermesLogBuffer = _hermesLogBuffer.slice(-500);
                }
                _hermesRenderLogs();
            } catch { /* ignore non-JSON */ }
        };
    };
    wire();
}
_hermesSubscribeLogStream();

// One-shot helper to fill a `Model: …` label with whatever hermes
// currently reports. Used by the LLM tab's read-only --model-name
// field when --llm-backend-type === 'hermes'. Cheap (3-second
// timeout, one fetch per render). If the user hasn't started hermes
// yet, the label stays as "(start hermes to load)" until they do.
async function _fetchHermesModelInto(labelEl) {
    if (!labelEl) return;
    let status = {};
    try { status = await getJSON('/api/hermes/status'); } catch (e) { /* offline */ }
    if (!status.running) {
        labelEl.textContent = '(start hermes to load)';
        return;
    }
    let modelName = status.model_name || '';
    // Fall back to /api/hermes/models when status didn't report a
    // model (rare but possible mid-model-swap).
    if (!modelName) {
        try {
            const r = await getJSON('/api/hermes/models');
            if (r && r.models && r.models.length === 1) modelName = r.models[0].id;
        } catch (e) { /* keep empty */ }
    }
    labelEl.textContent = modelName ? `Model: ${modelName}` : '(unknown)';
}

async function hermesFillerSave(patch) {
    try {
        await putJSON('/api/hermes/filler', patch);
        toast('Filler saved.', 'success', 2000);
    } catch (e) {
        toast('Filler save failed: ' + e.message, 'error', 4000);
    }
}

async function hermesStart(btnStart, btnStop) {
    if (btnStart) btnStart.disabled = true;
    try {
        const cfg = state.settings.hermes || {};
        const body = {
            host: cfg.host || '127.0.0.1',
            port: Number.isFinite(parseInt(cfg.port, 10)) ? parseInt(cfg.port, 10) : 8642,
            model_name: cfg.model_name || '',
            log_level: cfg.log_level || 'verbose',
        };
        const r = await postJSON('/api/hermes/start', body);
        toast(r.status && r.status.running
            ? 'Hermes started. Model loading...'
            : 'Hermes starting...', 'info', 2500);
        _hermesRefreshStatus();
    } catch (e) {
        toast('Hermes start failed: ' + (e.detail || e.message), 'error', 6000);
    } finally {
        if (btnStart) btnStart.disabled = false;
    }
}

async function hermesStop() {
    showModal('Stop Hermes?', 'Hermes will finish its current turn, then exit. ' +
        'The dashboard keeps the session id, so the next Start resumes context.',
        [
            { label: 'Cancel', kind: '', onClick: () => {} },
            { label: 'Stop', kind: 'btn-danger', onClick: async () => {
                try {
                    await postJSON('/api/hermes/stop', {});
                    toast('Hermes stopped.', 'info');
                    _hermesRefreshStatus();
                } catch (e) {
                    toast('Stop failed: ' + e.message, 'error', 4000);
                }
            } },
        ]);
}

async function hermesCancel() {
    try {
        await postJSON('/api/hermes/cancel', {});
        toast('Cancel signal sent to hermes.', 'info');
    } catch (e) {
        toast('Cancel failed: ' + e.message, 'error', 4000);
    }
}

function hermesKillConfirm() {
    showModal('Kill Hermes?', 'This sends SIGKILL — the subprocess dies immediately. ' +
        'All context (skills, memory, conversation) is lost. The pipeline\'s ' +
        'session id will reset on next Start.',
        [
            { label: 'Cancel', kind: '', onClick: () => {} },
            { label: 'KILL', kind: 'btn-danger', onClick: async () => {
                try {
                    await postJSON('/api/hermes/kill', { confirm: true });
                    toast('Hermes killed.', 'info');
                    _hermesRefreshStatus();
                } catch (e) {
                    toast('Kill failed: ' + e.message, 'error', 4000);
                }
            } },
        ]);
}

async function hermesSendChat() {
    const input = document.getElementById('hermes-chat-input');
    const consoleBox = document.getElementById('hermes-console');
    if (!input || !consoleBox) return;
    const msg = (input.value || '').trim();
    if (!msg) return;
    if (_hermesChatAbort) {
        toast('A chat reply is already streaming — press Stop first.', 'info', 3000);
        return;
    }
    // Append the user message to the console for context.
    const userRow = el('div', { class: 'log-row', style: { color: 'var(--accent, #6cf)' } },
        `you: ${msg}`);
    consoleBox.appendChild(userRow);
    // Placeholder for the assistant reply — we update .textContent as
    // each token arrives.
    const assistantRow = el('div', { class: 'log-row', style: { color: 'var(--text, #eee)' } },
        'hermes: ');
    consoleBox.appendChild(assistantRow);
    consoleBox.scrollTop = consoleBox.scrollHeight;
    input.value = '';

    _hermesChatAbort = new AbortController();
    try {
        const r = await fetch('/api/hermes/chat', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                message: msg,
            }),
            signal: _hermesChatAbort.signal,
        });
        if (!r.ok || !r.body) {
            assistantRow.textContent = `hermes: (error ${r.status}: ${await r.text().catch(() => '')})`;
            return;
        }
        const reader = r.body.getReader();
        const dec = new TextDecoder();
        let buf = '';
        while (true) {
            const { value, done } = await reader.read();
            if (done) break;
            buf += dec.decode(value, { stream: true });
            // SSE: events separated by blank lines, each line starting
            // with "data: ". We only care about the data payload here.
            let nl;
            while ((nl = buf.indexOf('\n\n')) !== -1) {
                const ev = buf.slice(0, nl);
                buf = buf.slice(nl + 2);
                const m = ev.match(/^data:\s*(.+)$/m);
                if (!m) continue;
                const payload = m[1].trim();
                if (!payload || payload === '[DONE]') continue;
                try {
                    const obj = JSON.parse(payload);
                    const delta = obj && obj.delta
                        || (obj && obj.choices && obj.choices[0] && obj.choices[0].delta && obj.choices[0].delta.content);
                    if (delta) {
                        assistantRow.textContent += delta;
                        consoleBox.scrollTop = consoleBox.scrollHeight;
                    }
                } catch { /* skip non-JSON frames */ }
            }
        }
        // End-of-stream marker.
        assistantRow.textContent += '\n';
        consoleBox.scrollTop = consoleBox.scrollHeight;
    } catch (e) {
        if (e.name === 'AbortError') {
            assistantRow.textContent += ' [aborted]';
        } else {
            assistantRow.textContent += ` [error: ${e.message}]`;
        }
    } finally {
        _hermesChatAbort = null;
    }
}

function hermesAbortChat() {
    if (_hermesChatAbort) {
        _hermesChatAbort.abort();
        _hermesChatAbort = null;
        toast('Chat reply stopped.', 'info', 2000);
    }
}

async function hermesResetSession() {
    showModal('Reset Hermes session?',
        'The dashboard will mint a new session id. The next time the ' +
        'robot talks, hermes will treat it as a fresh conversation — no ' +
        'memory of skills, context, or previous turns. Useful when ' +
        'context has grown large and you want to start over without ' +
        'killing the hermes subprocess.',
        [
            { label: 'Cancel', kind: '', onClick: () => {} },
            { label: 'Reset', kind: 'btn-primary', onClick: async () => {
                try {
                    const r = await postJSON('/api/hermes/reset_session', {});
                    toast('Session reset. New id: ' + (r.session_id || '').slice(0, 8) + '…',
                        'success', 3000);
                    _hermesRefreshStatus();
                } catch (e) {
                    toast('Reset failed: ' + e.message, 'error', 4000);
                }
            } },
        ]);
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
