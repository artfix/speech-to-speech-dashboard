// =============================================================
// Chatterbox voice library UI
// Rendered inside the TTS tab when the user picks --tts chatterbox.
// A small card grid showing each cloned voice (the .pt Conditionals
// on disk). Each card has:
//   - Set active: writes the voice name into --chatterbox-voice and
//     re-renders the form so the change is visible.
//   - Test: synthesizes a short preview using the current chatterbox
//     form values (so the user hears the exact voice the robot will
//     produce), and plays it back in-browser.
//   - × Delete: confirms, then removes the .pt + manifest entry.
// A "Clone new voice" button at the top opens a modal with a file
// picker, name input, and model variant dropdown.
//
// Loaded as a plain <script> before app.js so we can attach helpers
// to window. app.js calls window.renderVoiceLibrary.
// =============================================================

(function () {
'use strict';

let _voiceLibCache = [];

async function fetchVoices() {
    try {
        const r = await getJSON('/api/voices');
        _voiceLibCache = r.voices || [];
    } catch (e) {
        _voiceLibCache = [];
        toast('Could not load voice library: ' + e.message, 'error', 4000);
    }
    return _voiceLibCache;
}

function fmtBytes(n) {
    if (!n) return '0 B';
    const u = ['B', 'KB', 'MB', 'GB'];
    let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
}

function fmtDate(t) {
    if (!t) return '';
    return new Date(t * 1000).toLocaleString();
}

function voiceCard(voice, activeName, onChange) {
    const isActive = voice.name === activeName;
    const card = el('div', { class: 'voice-card' + (isActive ? ' active' : '') }, [
        el('div', { class: 'voice-card-title' }, [
            el('span', {}, voice.name),
            isActive ? el('span', { class: 'voice-card-badge' }, 'ACTIVE') : null,
        ]),
        el('div', { class: 'voice-card-meta' }, [
            el('div', {}, `Model: ${voice.model_variant}`),
            voice.sample_count
                ? el('div', {}, `Reference: ${(voice.sample_count / Math.max(voice.sample_rate, 1)).toFixed(1)}s @ ${voice.sample_rate} Hz`)
                : null,
            el('div', {}, `Conditionals: ${fmtBytes(voice.file_size_bytes)}`),
            el('div', { class: 'text-dim' }, `Cloned ${fmtDate(voice.created_at)}`),
        ]),
        el('div', { class: 'voice-card-actions' }, [
            el('button', {
                class: 'btn btn-primary btn-small',
                disabled: isActive,
                onclick: () => setActiveVoice(voice.name, onChange),
            }, 'Set active'),
            el('button', {
                class: 'btn btn-small',
                onclick: () => testVoice(voice),
            }, 'Test'),
            el('button', {
                class: 'btn btn-danger btn-small',
                onclick: () => deleteVoice(voice.name, onChange),
            }, '×'),
        ]),
    ]);
    return card;
}

async function setActiveVoice(name, onChange) {
    try {
        await postJSON(`/api/voices/${encodeURIComponent(name)}/set-active`, {});
        // The endpoint writes to the settings file. Re-fetch so the form
        // sees the change, then re-render so the active card gets the
        // highlight.
        const s = await getJSON('/api/settings');
        state.settings = { ...state.settings, '--chatterbox-voice': name, ...s.settings };
        toast(`Voice "${name}" is now active.`, 'success');
        onChange();
    } catch (e) {
        toast('Could not set active voice: ' + e.message, 'error', 4000);
    }
}

async function deleteVoice(name, onChange) {
    showModal(
        `Delete voice "${name}"?`,
        'The cloned Conditionals will be removed from disk. The original reference audio was never saved to disk, so deleting the entry loses the clone.',
        [
            { label: 'Cancel', kind: '', onClick: () => {} },
            { label: 'Delete', kind: 'btn-danger', onClick: async () => {
                try {
                    await fetch(`/api/voices/${encodeURIComponent(name)}`, { method: 'DELETE' });
                    toast(`Voice "${name}" deleted.`, 'info');
                    if (state.settings['--chatterbox-voice'] === name) {
                        state.settings['--chatterbox-voice'] = '';
                    }
                    onChange();
                } catch (e) {
                    toast('Delete failed: ' + e.message, 'error', 4000);
                }
            } },
        ]
    );
}

function testVoice(voice) {
    // Modal with a text input pre-filled with a sample phrase. Submit
    // POSTs /api/voices/test with the current chatterbox form values so
    // the user hears the exact robot voice.
    const textInput = el('input', {
        type: 'text',
        class: 'field-input',
        value: 'Hello, this is a test of my cloned voice.',
    });
    const body = el('div', {}, [
        el('p', {}, `Type a sentence to hear "${voice.name}" speak it with the current settings.`),
        el('label', { class: 'text-dim' }, 'Test text'),
        textInput,
    ]);
    showModal(`Test "${voice.name}"`, body, [
        { label: 'Cancel', kind: '', onClick: () => {} },
        { label: 'Play', kind: 'btn-primary', onClick: async () => {
            const text = textInput.value.trim() || 'Hello.';
            await playTestPreview(voice.name, text);
        } },
    ]);
}

async function playTestPreview(voiceName, text) {
    // Build the request body from the current form values so the preview
    // matches the robot's actual voice.
    const flagToName = {
        '--chatterbox-model-variant': 'model_variant',
        '--chatterbox-exaggeration': 'exaggeration',
        '--chatterbox-cfg-weight': 'cfg_weight',
        '--chatterbox-temperature': 'temperature',
        '--chatterbox-repetition-penalty': 'repetition_penalty',
        '--chatterbox-min-p': 'min_p',
        '--chatterbox-top-p': 'top_p',
        '--chatterbox-top-k': 'top_k',
        '--chatterbox-language-id': 'language_id',
    };
    const body = { voice: voiceName, text };
    for (const [flag, key] of Object.entries(flagToName)) {
        const v = state.settings[flag];
        if (v !== undefined && v !== '' && v !== null) {
            body[key] = v;
        }
    }
    try {
        const r = await fetch('/api/voices/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!r.ok) {
            const detail = await r.json().catch(() => ({}));
            const msg = (detail && detail.detail && detail.detail.error) || JSON.stringify(detail.detail || detail);
            if (msg.includes('chatterbox_not_installed')) {
                showChatterboxInstallModal(detail.detail);
                return;
            }
            toast('Test failed: ' + msg, 'error', 5000);
            return;
        }
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.onended = () => URL.revokeObjectURL(url);
        await audio.play().catch(() => {
            // Most browsers refuse autoplay; show a click-to-play toast.
            toast('Autoplay blocked. Click Play in the test modal again.', 'warning', 4000);
        });
    } catch (e) {
        toast('Test failed: ' + e.message, 'error', 5000);
    }
}

function openCloneModal(onChange) {
    const fileInput = el('input', { type: 'file', accept: 'audio/*', class: 'field-input' });
    const nameInput = el('input', { type: 'text', class: 'field-input', placeholder: 'my-voice' });
    const variantSel = el('select', { class: 'field-select' });
    for (const v of ['chatterbox-turbo', 'chatterbox-nano', 'chatterbox', 'chatterbox-multilingual']) {
        const opt = el('option', { value: v }, v);
        // Pre-select whatever the form is currently set to.
        if (state.settings['--chatterbox-model-variant'] === v) opt.selected = true;
        variantSel.appendChild(opt);
    }
    const body = el('div', {}, [
        el('p', {}, 'Pick a reference audio file from your disk. The audio is used to embed the speaker and is NOT saved on the server -- only the resulting Conditionals (.pt) are stored.'),
        el('label', { class: 'text-dim' }, 'Reference audio (WAV, MP3, FLAC, OGG...)'),
        fileInput,
        el('label', { class: 'text-dim' }, 'Voice name (letters, digits, _, -)'),
        nameInput,
        el('label', { class: 'text-dim' }, 'Model variant for the clone'),
        variantSel,
        el('p', { class: 'text-dim' }, 'A 5-15 second clear single-speaker reference works best. Longer is fine; quality matters more than length.'),
    ]);
    showModal('Clone a new voice', body, [
        { label: 'Cancel', kind: '', onClick: () => {} },
        { label: 'Clone', kind: 'btn-primary', onClick: async () => {
            const file = fileInput.files && fileInput.files[0];
            if (!file) { toast('Pick a reference audio file first.', 'warning'); return; }
            const name = nameInput.value.trim();
            if (!name) { toast('Pick a voice name.', 'warning'); return; }
            const fd = new FormData();
            fd.append('name', name);
            fd.append('model_variant', variantSel.value);
            fd.append('audio', file);
            try {
                toast('Cloning voice (this can take 10-30 seconds)...', 'info', 30000);
                const r = await fetch('/api/voices/clone', { method: 'POST', body: fd });
                if (!r.ok) {
                    const detail = await r.json().catch(() => ({}));
                    const msg = (detail && detail.detail && detail.detail.error) || JSON.stringify(detail.detail || detail);
                    if (msg.includes('chatterbox_not_installed')) {
                        showChatterboxInstallModal(detail.detail);
                        return;
                    }
                    toast('Clone failed: ' + msg, 'error', 6000);
                    return;
                }
                toast(`Voice "${name}" cloned!`, 'success');
                onChange();
            } catch (e) {
                toast('Clone failed: ' + e.message, 'error', 6000);
            }
        } },
    ]);
}

async function renderVoiceLibrary(container, settings, onChange) {
    container.textContent = '';
    // Only render when the user has selected chatterbox TTS. When they
    // pick a different backend, this mount is hidden by the parent.
    if (settings['--tts'] !== 'chatterbox') {
        container.style.display = 'none';
        return;
    }
    container.style.display = '';

    const voices = await fetchVoices();
    const activeName = settings['--chatterbox-voice'] || '';

    const header = el('div', { class: 'voice-library-header' }, [
        el('h2', {}, 'Voice library'),
        el('button', {
            class: 'btn btn-primary',
            onclick: () => openCloneModal(onChange),
        }, '+ Clone new voice'),
    ]);
    container.appendChild(header);

    if (!voices.length) {
        container.appendChild(el('p', { class: 'text-dim' },
            'No cloned voices yet. Click "Clone new voice" to embed a speaker. The default voice bundled with the model is used until you clone and set one as active.'));
        return;
    }

    const grid = el('div', { class: 'voice-card-grid' });
    for (const v of voices) {
        grid.appendChild(voiceCard(v, activeName, onChange));
    }
    container.appendChild(grid);
}

// Expose for app.js, which loads after us.
window.renderVoiceLibrary = renderVoiceLibrary;

})();
