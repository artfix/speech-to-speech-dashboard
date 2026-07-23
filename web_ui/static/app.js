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
        const detail = await r.json().catch(() => ({ detail: r.statusText }));
        throw new Error(detail.detail || r.statusText);
    }
    return r.json();
}

const getJSON = (path) => api(path);
const postJSON = (path, body) => api(path, { method: 'POST', body: JSON.stringify(body) });

// ---- Initial load ----------------------------------------------------

async function init() {
    try {
        const [schema, settingsR, themesR, versionR] = await Promise.all([
            getJSON('/api/schema'),
            getJSON('/api/settings'),
            getJSON('/api/themes'),
            getJSON('/api/version'),
        ]);
        state.schema = schema;
        state.defaults = settingsR.settings;
        state.settings = { ...settingsR.settings };
        state.saved = settingsR.saved;
        state.savedPath = settingsR.path;
        state.themes = themesR.themes;
        state.version = versionR.version;
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

function buildThemeSelect() {
    const sel = $('#theme-select');
    sel.textContent = '';
    for (const t of state.themes) {
        const opt = el('option', { value: t }, t);
        if (t === state.currentTheme) opt.selected = true;
        sel.appendChild(opt);
    }
    sel.onchange = () => applyTheme(sel.value);
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
    nav.appendChild(el('div', { class: 'theme-credit' }, 'v0.1.0'));

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
    updateSubgroupVisibility();
}

function renderField(f, parentTitle) {
    const fieldId = `f-${f.flag}`;
    const label = el('label', { class: 'field-label', for: fieldId }, [
        f.flag,
        el('span', { class: 'field-flag' }, ''),
        el('button', {
            type: 'button',
            class: 'help-btn',
            title: 'Show help',
            onclick: (e) => {
                e.preventDefault();
                const help = $(`#help-${f.flag}`);
                if (help) help.classList.toggle('visible');
            }
        }, '?'),
    ]);

    let input;
    const val = state.settings[f.flag];
    if (f.ui === 'checkbox') {
        input = el('label', { class: 'field-checkbox' }, [
            el('input', {
                type: 'checkbox',
                id: fieldId,
                checked: val === true,
                onchange: (e) => state.settings[f.flag] = e.target.checked,
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
            oninput: (e) => state.settings[f.flag] = e.target.value,
        });
        input.value = val == null ? '' : String(val);
    } else {  // text or number
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
            }
        });
        input.value = val == null ? '' : String(val);
    }

    const help = el('div', { class: 'field-help', id: `help-${f.flag}` }, f.help || '(no help text)');

    const wrap = el('div', { class: 'field' }, [label, input, help]);
    // Optional String fields benefit from full width since they may be long.
    if (f.ui === 'textarea' || f.type === 'optional_string') {
        wrap.classList.add('full');
    }
    return wrap;
}

function updateSubgroupVisibility() {
    for (const group of state.schema.groups) {
        if (!group.subgroups) continue;
        for (const sub of group.subgroups) {
            const sel = state.settings[sub.visible_when.field];
            const show = sel === sub.visible_when.equals;
            const el = $(`[data-subgroup="${sub.id}"]`);
            if (el) el.style.display = show ? '' : 'none';
        }
    }
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

    tab.appendChild(el('h3', { style: { marginTop: '32px' } }, 'Danger Zone'));
    tab.appendChild(el('div', { class: 'text-dim', style: { marginBottom: '12px' } }, 'Stops the pipeline AND closes the dashboard web server. You will need to re-run start_web_ui.sh to come back.'));

    const row2 = el('div', { class: 'btn-row' }, [
        el('button', { class: 'btn btn-danger btn-large', onclick: shutdownAll }, '⏻ Shutdown Everything'),
    ]);
    tab.appendChild(row2);
}

async function startPipeline() {
    try {
        await postJSON('/api/process/start', { settings: state.settings });
        toast('Pipeline started.', 'success');
        pollStatus();
    } catch (e) {
        toast('Start failed: ' + e.message, 'error', 6000);
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
