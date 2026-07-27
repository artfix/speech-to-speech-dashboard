// =============================================================
// Qwen3-TTS voice library UI
//
// Rendered inside the TTS tab when --tts qwen3. Mirrors the
// Chatterbox voice library's card-grid pattern, but for qwen3-TTS
// the "voices" are the 9 CustomVoice preset speakers baked into
// the model weights (Vivian, Serena, Uncle_Fu, Dylan, Eric, Ryan,
// Aiden, Ono_Anna, Sohee).
//
// Each preset card has:
//   - Speaker name + native-language badge
//   - A native sample sentence the user can hear in the Test modal
//   - "Set active" → POST /api/qwen3/voice/{speaker}/set-active
//   - "Test"      → modal with sample sentence + free text → POST
//                   /api/qwen3/voice/test → audio/wav
//
// When a Base model is selected, an additional "Reference voices"
// section appears listing files under voices/qwen3_refs/ (uploaded
// via /api/qwen3_ref_audio). The Test button on those cards WILL
// FAIL with a QwenTTSError about ABI v2 on this Pascal wheel --
// the section header says so in red so the user knows what they're
// clicking.
//
// Loaded as a plain <script> before app.js. app.js calls
// window.renderQwen3VoiceLibrary.
// =============================================================

(function () {
'use strict';

let _qwen3VoiceCache = null;

async function fetchQwen3Voices() {
    try {
        const r = await getJSON('/api/qwen3/voices');
        _qwen3VoiceCache = r;
    } catch (e) {
        _qwen3VoiceCache = { presets: [], active_speaker: '', ref_audio_files: [] };
        toast('Could not load qwen3 voice library: ' + e.message, 'error', 4000);
    }
    return _qwen3VoiceCache;
}

function fmtBytes(n) {
    if (!n) return '0 B';
    const u = ['B', 'KB', 'MB', 'GB'];
    let i = 0;
    while (n >= 1024 && i < u.length - 1) { n /= 1024; i++; }
    return `${n.toFixed(n < 10 && i > 0 ? 1 : 0)} ${u[i]}`;
}

function langBadge(language) {
    const colors = {
        chinese:  '#c0392b',
        english:  '#2c3e50',
        japanese: '#16a085',
        korean:   '#8e44ad',
        auto:     '#7f8c8d',
    };
    const c = colors[language] || '#7f8c8d';
    return el('span', {
        class: 'voice-card-lang',
        style: {
            display: 'inline-block',
            padding: '2px 6px',
            borderRadius: '4px',
            background: c,
            color: '#fff',
            fontSize: '10px',
            fontWeight: '700',
            marginLeft: '6px',
            textTransform: 'uppercase',
            letterSpacing: '0.5px',
        },
    }, language);
}

function presetCard(preset, activeSpeaker, onChange) {
    const isActive = preset.name === activeSpeaker;
    const card = el('div', { class: 'voice-card' + (isActive ? ' active' : '') }, [
        el('div', { class: 'voice-card-title' }, [
            el('span', {}, preset.name),
            langBadge(preset.language),
            isActive ? el('span', { class: 'voice-card-badge' }, 'ACTIVE') : null,
        ]),
        el('div', { class: 'voice-card-meta' }, [
            el('div', { class: 'text-dim' }, preset.label),
            el('div', { style: { fontStyle: 'italic', marginTop: '6px', color: 'var(--text-secondary)' } },
                `"${preset.sample}"`),
        ]),
        el('div', { class: 'voice-card-actions' }, [
            el('button', {
                class: 'btn btn-primary btn-small',
                disabled: isActive,
                onclick: () => setActivePreset(preset.name, onChange),
            }, 'Set active'),
            el('button', {
                class: 'btn btn-small',
                onclick: () => testPreset(preset),
            }, 'Test'),
        ]),
    ]);
    return card;
}

async function setActivePreset(name, onChange) {
    try {
        const r = await postJSON(
            `/api/qwen3/voice/${encodeURIComponent(name)}/set-active`,
            {}
        );
        // The endpoint writes to the settings file and returns the
        // merged settings object. Replace state.settings so the
        // dropdown re-paints with the new value on next render.
        if (r && r.settings) {
            state.settings = { ...state.settings, ...r.settings };
        } else {
            // Older endpoint shape: re-fetch.
            const s = await getJSON('/api/settings');
            state.settings = { ...state.settings, '--qwen3-tts-speaker': name, ...s.settings };
        }
        toast(`Voice "${name}" is now active.`, 'success');
        onChange();
    } catch (e) {
        toast('Could not set active voice: ' + e.message, 'error', 4000);
    }
}

// 10 languages the qwentts_cpp binding accepts for synthesis (auto +
// 9 explicit). Order matches the dashboard's --qwen3-tts-language
// dropdown. "auto" lets the binding infer from the input text; picking
// a specific language forces the binding to that language even if the
// input text is in a different language.
const QWEN3_LANGUAGES = [
    'auto', 'english', 'chinese', 'japanese', 'korean',
    'german', 'french', 'russian', 'portuguese', 'spanish', 'italian',
];

function testPreset(preset) {
    // Modal with a text input pre-filled with the speaker's native
    // sample sentence + a language dropdown so the user can pin the
    // synthesis language per-test. Default language: whatever the
    // dashboard's --qwen3-tts-language is set to (the speaker's native
    // language when that's "auto", or the user's explicit pick).
    const textInput = el('input', {
        type: 'text',
        class: 'field-input',
        value: preset.sample,
    });
    const langSel = el('select', { class: 'field-select' });
    const dashboardLang = state.settings['--qwen3-tts-language'] || 'auto';
    // Default: dashboard setting if it's a real language; else the
    // speaker's native language. This way the Test preview matches
    // whatever the realtime pipe is using right now.
    const initialLang = (dashboardLang && dashboardLang !== 'auto')
        ? dashboardLang
        : preset.language;
    for (const l of QWEN3_LANGUAGES) {
        const opt = el('option', { value: l }, l);
        if (l === initialLang) opt.selected = true;
        langSel.appendChild(opt);
    }
    const body = el('div', {}, [
        el('p', {}, `Type a sentence to hear "${preset.name}" speak it. The native-language sample is pre-filled.`),
        el('label', { class: 'text-dim' }, 'Test text'),
        textInput,
        el('label', { class: 'text-dim', style: { marginTop: '8px', display: 'block' } }, 'Language'),
        langSel,
        el('p', {
            class: 'text-dim',
            style: { marginTop: '8px', fontSize: '11px', lineHeight: '1.4' },
        }, [
            el('strong', {}, 'Note: '),
            el('span', {}, `"${preset.name}" sounds best in ${preset.label}. Other languages work but the binding will speak them with this speaker's voice characteristics, which can sound accented.`),
        ]),
    ]);
    showModal(`Test "${preset.name}"`, body, [
        { label: 'Cancel', kind: '', onClick: () => {} },
        { label: 'Play', kind: 'btn-primary', onClick: async () => {
            const text = textInput.value.trim() || preset.sample;
            const language = langSel.value || preset.language;
            await playPresetPreview(preset.name, text, language);
        } },
    ]);
}

async function playPresetPreview(speaker, text, language) {
    // The Test endpoint loads the qwen3 model into the dashboard's own
    // process. When the pipeline subprocess is already running it
    // holds most of the GPU's VRAM (~4 GB on Pascal for the 1.7B
    // model) and a second copy of the same model would OOM. Per the
    // CLAUDE.md hard constraint we can't add a /v1/synthesize
    // endpoint to src/speech_to_speech/, so the only safe options are
    // (a) ask the user to stop the pipeline first, or (b) send the
    // request through the pipeline's /v1/realtime WebSocket as a
    // pseudo-conversation (much more code). (a) is what we ship.
    //
    // ``state.status`` is updated by the dashboard's status poller
    // every 2s; if it says the pipeline is running, surface a clear
    // error and don't burn 5+ seconds on a request that's going to
    // fail with cudaMalloc OOM.
    const running = state.status && state.status.running;
    if (running) {
        showModal(
            'Pipeline is running',
            'The Test button loads the qwen3 model into the dashboard process, which would OOM the GPU because the pipeline already holds the model. Stop the pipeline first, click Test, then restart the pipeline.',
            [
                { label: 'Cancel', kind: '', onClick: () => {} },
                { label: 'Stop pipeline & test', kind: 'btn-primary', onClick: async () => {
                    try {
                        await postJSON('/api/process/stop', {});
                        // pollStatus() runs every 2s; force an immediate
                        // refresh so the in-flight test sees the new
                        // status instead of the stale 2s-old snapshot.
                        try { state.status = await getJSON('/api/process/status'); } catch (_) {}
                        // Give the subprocess ~800ms to fully release
                        // its CUDA context before we allocate a second.
                        await new Promise(r => setTimeout(r, 800));
                        await playPresetPreview(speaker, text, language);
                    } catch (e) {
                        toast('Could not stop pipeline: ' + e.message, 'error', 4000);
                    }
                } },
            ],
        );
        return;
    }
    const flagToName = {
        '--qwen3-tts-model-name': 'model_id',
        '--qwen3-tts-device': 'device',
        '--qwen3-tts-seed': 'seed',
        '--qwen3-tts-temperature': 'temperature',
        '--qwen3-tts-top-p': 'top_p',
        '--qwen3-tts-top-k': 'top_k',
        '--qwen3-tts-repetition-penalty': 'repetition_penalty',
    };
    const body = { speaker, text, language };
    for (const [flag, key] of Object.entries(flagToName)) {
        const v = state.settings[flag];
        if (v !== undefined && v !== '' && v !== null) {
            body[key] = v;
        }
    }
    try {
        const r = await fetch('/api/qwen3/voice/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!r.ok) {
            const detail = await r.json().catch(() => ({}));
            const errDetail = detail && detail.detail ? detail.detail : detail;
            const msg = (errDetail && errDetail.message)
                || (typeof errDetail === 'string' ? errDetail : JSON.stringify(errDetail));
            if (errDetail && errDetail.error === 'qwen3_not_installed') {
                toast('qwen3-TTS wheel is not installed: ' + msg, 'error', 6000);
                return;
            }
            if (msg && /ABI v2|qt_extract_voice_ref/i.test(msg)) {
                // Reference-audio Test on Pascal ABI v1 wheel -- the
                // expected failure for ref-audio tests. The card UI
                // already warns about this, so the toast is just a
                // confirmation that yes, that path is broken.
                toast('Reference audio / voice design requires ABI v2 (not available in the bundled Pascal wheel). Switch to a CustomVoice model to test preset voices.', 'warning', 7000);
                return;
            }
            toast('Test failed: ' + msg, 'error', 6000);
            return;
        }
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.onended = () => URL.revokeObjectURL(url);
        await audio.play().catch(() => {
            toast('Autoplay blocked. Click Play in the test modal again.', 'warning', 4000);
        });
    } catch (e) {
        toast('Test failed: ' + e.message, 'error', 6000);
    }
}

function refAudioCard(file, onChange) {
    const isCurrent = state.settings['--qwen3-tts-ref-audio'] === file.path;
    const card = el('div', { class: 'voice-card' + (isCurrent ? ' active' : '') }, [
        el('div', { class: 'voice-card-title' }, [
            el('span', { class: 'voice-card-name-mono' }, file.name),
            isCurrent ? el('span', { class: 'voice-card-badge' }, 'CURRENT') : null,
        ]),
        el('div', { class: 'voice-card-meta' }, [
            el('div', {}, `Size: ${fmtBytes(file.size_bytes)}`),
            el('div', { class: 'text-dim' }, `Uploaded ${new Date(file.mtime * 1000).toLocaleString()}`),
            el('div', { class: 'text-dim', style: { fontFamily: 'monospace', fontSize: '10px', wordBreak: 'break-all', marginTop: '4px' } },
                file.path),
        ]),
        el('div', { class: 'voice-card-actions' }, [
            el('button', {
                class: 'btn btn-primary btn-small',
                disabled: isCurrent,
                onclick: () => useAsRefAudio(file, onChange),
            }, 'Use as ref audio'),
            el('button', {
                class: 'btn btn-small',
                title: 'Will fail on Pascal ABI v1 wheel — switch to a Base model and ensure ABI v2 is bundled.',
                onclick: () => testRefAudio(file),
            }, 'Test'),
            el('button', {
                class: 'btn btn-danger btn-small',
                onclick: () => deleteRefAudio(file, onChange),
            }, '×'),
        ]),
    ]);
    return card;
}

async function useAsRefAudio(file, onChange) {
    state.settings['--qwen3-tts-ref-audio'] = file.path;
    // Persist immediately so the pipeline picks it up on next start.
    try {
        await postJSON('/api/settings/patch', { '--qwen3-tts-ref-audio': file.path });
        toast(`Reference audio set to ${file.name}.`, 'success');
        onChange();
    } catch (e) {
        toast('Could not save: ' + e.message, 'error', 4000);
    }
}

function testRefAudio(file) {
    const textInput = el('input', {
        type: 'text',
        class: 'field-input',
        value: 'Hello, this is a test using the uploaded reference audio.',
    });
    const body = el('div', {}, [
        el('p', {}, `Synthesize "${file.name}" as the voice. This will FAIL on the bundled Pascal wheel -- voice cloning needs ABI v2 symbols that aren't in the public qwentts.cpp source.`),
        el('label', { class: 'text-dim' }, 'Text to speak'),
        textInput,
        el('label', { class: 'text-dim', style: { marginTop: '8px', display: 'block' } }, 'Reference text (transcript of the ref audio)'),
        el('input', {
            type: 'text',
            class: 'field-input',
            id: 'qwen3-ref-text',
            value: state.settings['--qwen3-tts-ref-text'] || '',
            placeholder: 'What was said in the reference audio',
        }),
    ]);
    showModal(`Test ref audio "${file.name}"`, body, [
        { label: 'Cancel', kind: '', onClick: () => {} },
        { label: 'Play', kind: 'btn-primary', onClick: async () => {
            const text = textInput.value.trim() || 'Hello.';
            const refText = document.getElementById('qwen3-ref-text').value.trim();
            await playRefAudioPreview(file, text, refText);
        } },
    ]);
}

async function playRefAudioPreview(file, text, refText) {
    const body = {
        text,
        ref_audio: file.path,
        ref_text: refText,
        language: 'auto',
        model_id: state.settings['--qwen3-tts-model-name'] || 'Qwen/Qwen3-TTS-12Hz-1.7B-CustomVoice',
        device: state.settings['--qwen3-tts-device'] || 'cuda',
    };
    try {
        const r = await fetch('/api/qwen3/voice/test', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        });
        if (!r.ok) {
            const detail = await r.json().catch(() => ({}));
            const errDetail = detail && detail.detail ? detail.detail : detail;
            const msg = (errDetail && errDetail.message)
                || (typeof errDetail === 'string' ? errDetail : JSON.stringify(errDetail));
            // On Pascal ABI v1 wheel this is the EXPECTED failure mode.
            // Surface a clear, friendly toast instead of a stack trace.
            if (msg && /ABI v2|qt_extract_voice_ref/i.test(msg)) {
                toast('Voice cloning requires ABI v2 (not available in the bundled Pascal wheel). The file is saved under voices/qwen3_refs/ — once upstream publishes ABI v2, cloning will light up automatically.', 'warning', 9000);
                return;
            }
            toast('Test failed: ' + msg, 'error', 6000);
            return;
        }
        const blob = await r.blob();
        const url = URL.createObjectURL(blob);
        const audio = new Audio(url);
        audio.onended = () => URL.revokeObjectURL(url);
        await audio.play().catch(() => {
            toast('Autoplay blocked. Click Play in the test modal again.', 'warning', 4000);
        });
    } catch (e) {
        toast('Test failed: ' + e.message, 'error', 6000);
    }
}

async function deleteRefAudio(file, onChange) {
    showModal(
        `Delete "${file.name}"?`,
        'The reference audio file will be removed from voices/qwen3_refs/. The original file on your disk is not affected.',
        [
            { label: 'Cancel', kind: '', onClick: () => {} },
            { label: 'Delete', kind: 'btn-danger', onClick: async () => {
                try {
                    const r = await fetch(`/api/qwen3_ref_audio/${encodeURIComponent(file.name)}`, { method: 'DELETE' });
                    if (!r.ok) {
                        const detail = await r.json().catch(() => ({}));
                        toast('Delete failed: ' + (detail.detail || r.statusText), 'error', 4000);
                        return;
                    }
                    if (state.settings['--qwen3-tts-ref-audio'] === file.path) {
                        state.settings['--qwen3-tts-ref-audio'] = '';
                    }
                    toast(`Ref audio "${file.name}" deleted.`, 'info');
                    onChange();
                } catch (e) {
                    toast('Delete failed: ' + e.message, 'error', 4000);
                }
            } },
        ]
    );
}

// True iff the user's currently-selected model needs ref_audio to
// synthesize. Used to decide whether to render the Reference voices
// section (which is meaningless for CustomVoice models -- they have
// built-in speakers, no ref_audio).
function _selectedModelNeedsRefAudio() {
    if (!state.qwen3Models) return false;
    const modelId = _settingValue('qwen3_tts_model_name');
    const model = (state.qwen3Models.models || []).find((m) => m.id === modelId);
    return !!(model && model.requires && model.requires.includes('ref_audio'));
}

// True iff the user's currently-selected model can't work end-to-end
// because of the ABI v2 limitation. Drives the info banner shown
// above the voice library.
function _selectedModelNeedsAbiV2() {
    if (!state.qwen3Models) return false;
    const modelId = _settingValue('qwen3_tts_model_name');
    const model = (state.qwen3Models.models || []).find((m) => m.id === modelId);
    return !!(model && model.needs_abi_v2);
}

async function renderQwen3VoiceLibrary(container, settings, onChange) {
    container.textContent = '';
    if (settings['--tts'] !== 'qwen3') {
        container.style.display = 'none';
        return;
    }
    container.style.display = '';

    const data = await fetchQwen3Voices();
    const activeSpeaker = data.active_speaker || settings['--qwen3-tts-speaker'] || '';

    const header = el('div', { class: 'voice-library-header' }, [
        el('h2', {}, 'Voice library'),
        el('p', { class: 'text-dim' },
            'The 9 preset speakers below are baked into the Qwen3-TTS CustomVoice model. ' +
            'Click "Set active" to make a speaker the default for all synthesis; click "Test" to ' +
            'preview the voice with a sample sentence before applying.'),
    ]);
    container.appendChild(header);

    // ABI v2 info banner: only when the user picked a Base / VoiceDesign
    // model whose runtime path needs symbols not in the bundled wheel.
    if (_selectedModelNeedsAbiV2()) {
        const banner = el('div', {
            class: 'voice-library-banner',
            style: {
                padding: '12px 14px',
                margin: '12px 0',
                borderRadius: '6px',
                background: 'rgba(255, 193, 7, 0.12)',
                border: '1px solid rgba(255, 193, 7, 0.5)',
                color: 'var(--text-primary)',
                fontSize: '13px',
                lineHeight: '1.5',
            },
        }, [
            el('strong', { style: { color: '#ffc107' } }, '⚠ ABI v2 required'),
            el('p', { style: { margin: '6px 0 0 0' } },
                'Voice cloning (ref_audio) and voice design (instruct) on this model need ABI v2 symbols ' +
                '('),
            el('code', {}, 'qt_extract_voice_ref'),
            el('span', {}, ', '),
            el('code', {}, 'qt_voice_ref_free'),
            el('span', {}, ') that are not yet in the public '),
            el('code', {}, 'andimarafioti/qwentts.cpp'),
            el('span', {}, ' source. Synthesis will raise '),
            el('code', {}, 'QwenTTSError: qt_extract_voice_ref is unavailable'),
            el('span', {}, '. Switch back to a CustomVoice model to use the 9 preset voices.'),
        ]);
        container.appendChild(banner);
    }

    // Preset voices grid
    const grid = el('div', { class: 'voice-card-grid' });
    for (const p of (data.presets || [])) {
        grid.appendChild(presetCard(p, activeSpeaker, onChange));
    }
    container.appendChild(grid);

    // Reference voices section -- only meaningful for Base models.
    // Even on Pascal where synthesis will fail, we still let the user
    // upload / manage files because the moment ABI v2 lands upstream,
    // existing uploads light up without more dashboard work.
    if (_selectedModelNeedsRefAudio()) {
        const refHeader = el('div', {
            class: 'voice-library-section-header',
            style: { marginTop: '24px', borderTop: '1px solid var(--border)', paddingTop: '16px' },
        }, [
            el('h3', {}, 'Reference voices'),
            el('p', { class: 'text-dim', style: { fontSize: '12px' } },
                'Audio files uploaded for voice cloning. Set one as the active ref audio ' +
                'and the dashboard will pass it as '),
            el('code', { style: { fontSize: '11px' } }, '--qwen3-tts-ref-audio'),
            el('span', { class: 'text-dim', style: { fontSize: '12px' } }, '.'),
            el('p', {
                class: 'voice-library-warning',
                style: {
                    marginTop: '8px',
                    padding: '8px 10px',
                    borderRadius: '4px',
                    background: 'rgba(220, 53, 69, 0.10)',
                    border: '1px solid rgba(220, 53, 69, 0.4)',
                    color: '#ff6b6b',
                    fontSize: '12px',
                },
            }, [
                el('strong', {}, '⚠ Test will fail on this GPU. '),
                el('span', {},
                    'The bundled Pascal qwentts-cpp-python wheel exposes ABI v1 only; ' +
                    'voice cloning needs ABI v2 (qt_extract_voice_ref). The file is still ' +
                    'saved on disk under voices/qwen3_refs/ and will work the moment ' +
                    'upstream publishes ABI v2 source and the dashboard rebuilds the wheel.'),
            ]),
        ]);
        container.appendChild(refHeader);

        const refFiles = data.ref_audio_files || [];
        if (!refFiles.length) {
            container.appendChild(el('p', { class: 'text-dim', style: { marginTop: '8px' } },
                'No reference audio uploaded yet. Use the "Upload reference audio…" button ' +
                'next to --qwen3-tts-ref-audio above to add one.'));
        } else {
            const refGrid = el('div', { class: 'voice-card-grid' });
            for (const f of refFiles) {
                refGrid.appendChild(refAudioCard(f, onChange));
            }
            container.appendChild(refGrid);
        }
    }
}

window.renderQwen3VoiceLibrary = renderQwen3VoiceLibrary;

})();
