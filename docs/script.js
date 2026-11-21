const $ = id => document.getElementById(id);

let dictData = [];
const meaningIndex = new Map();
const exactIndexes = { z: new Map(), p: new Map(), y: new Map(), a: new Map(), m: new Map() };
const BASE_INDEX_KEYS = ['z', 'p', 'y', 'a'];
const ALL_INDEX_KEYS = ['z', 'p', 'y', 'a', 'm'];

const DICT_CACHE = 'pboc-dict-v1';
const DICT_URLS = ['base.json.gz', 'extra.json.gz'];
const DICT_VERSION_URL = 'data-version.json';
const DICT_RELEASE_PREFIX = `${DICT_CACHE}-release-`;
const DICT_CHECK_INTERVAL = 24 * 60 * 60 * 1000;

const SEARCH_MODES = new Set(['default', 'exact', 'regex', 'meaning', 'batch']);
const MODE_LABELS = { default: '默認', exact: '精確', regex: '正則', meaning: '釋義', batch: '批量' };
const CJK_RANGE = '\\u{3400}-\\u{4DBF}\\u{4E00}-\\u{9FFF}\\u{F900}-\\u{FAFF}\\u{20000}-\\u{323AF}';
const CJK_RE = new RegExp(`[${CJK_RANGE}]`, 'u');
const CJK_TOKEN_RE = new RegExp(`[${CJK_RANGE}]+|[^${CJK_RANGE}]+`, 'gu');
const BATCH_MAX_LENGTH = 5000;
const REGEX_MAX_LEN = 100;
const REDOS_PATTERN = /(\(.*\*.*\*|\(.*\+.*\+|\(\?=.*\*|\(\?!.*\*|(\.\*){3,}|\(\w+\+\)\+|\(\w+\*\)\+)/;

const IMAGE_EXPORT_URL = 'https://cdnjs.cloudflare.com/ajax/libs/html-to-image/1.11.11/html-to-image.min.js';
const IMAGE_EXPORT_TIMEOUT = 15000;

const DEFAULT_PAGE_SIZE = 50;
const PAGE_SIZE_OPTIONS = [20, 50, 100, 200];
const PAGE_SIZE_STORAGE_KEY = 'pboc-page-size';
const HISTORY_STORAGE_KEY = 'pboc-search-history';
const HISTORY_MAX = 20;
const SORT_OPTIONS = [
    { value: 'default',  label: '默認順序', field: null },
    { value: 'pinyin',   label: '按拼音',   field: 'p' },
    { value: 'phonetic', label: '按擬音',   field: 'y' },
    { value: 'domain',   label: '按諧聲域', field: 'a' },
];
const zhCollator = new Intl.Collator('zh', { sensitivity: 'base' });

let dictionaryLoading = false;
let dictionaryRefreshPromise = null;
let updateNotified = false;
let _batchRows = null;
let imageExportLoader = null;

let currentResults = [];
let currentQuery = '';
let currentMode = 'default';
let currentSort = 'default';
let currentPage = 1;
let pendingUrlPage = 1;
let pageSize = normalizePageSize(readStorage(PAGE_SIZE_STORAGE_KEY));

function escapeHtml(unsafe) {
    if (typeof unsafe !== 'string') return unsafe;
    return unsafe
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}

function readStorage(key, fallback = null) {
    try { return localStorage.getItem(key) ?? fallback; } catch (_e) { return fallback; }
}

function writeStorage(key, value) {
    try { localStorage.setItem(key, value); } catch (_e) {}
}

function showUpdateNotification() {
    if (updateNotified) return;
    updateNotified = true;
    const bar = $('updateBar');
    if (bar) bar.style.display = 'flex';
}

function addToIndex(index, key, item) {
    if (!key) return;
    const normalized = String(key).trim();
    if (!normalized) return;
    let bucket = index.get(normalized);
    if (!bucket) index.set(normalized, (bucket = []));
    bucket.push(item);
}

function buildIndexes({ includeExtra = false } = {}) {
    ALL_INDEX_KEYS.forEach(k => exactIndexes[k].clear());
    meaningIndex.clear();

    const keys = includeExtra ? ALL_INDEX_KEYS : BASE_INDEX_KEYS;
    dictData.forEach((item, index) => {
        keys.forEach(k => addToIndex(exactIndexes[k], item[k], item));
        if (includeExtra) {
            const addMeaningText = text => {
                if (!text) return;
                for (const char of text) {
                    let set = meaningIndex.get(char);
                    if (!set) meaningIndex.set(char, (set = new Set()));
                    set.add(index);
                }
            };
            item.d?.[1]?.forEach(addMeaningText);
            addMeaningText(item.n);
        }
    });
}

function appendUnique(target, seen, items) {
    if (!items) return;
    items.forEach(item => {
        if (!seen.has(item)) {
            seen.add(item);
            target.push(item);
        }
    });
}

function regexMatches(regex, ...values) {
    return values.some(value => {
        if (!value) return false;
        regex.lastIndex = 0;
        return regex.test(value);
    });
}

function responseBodyIsAlreadyDecoded(response) {
    const encoding = response.headers.get('content-encoding');
    return !!encoding && encoding.toLowerCase() !== 'identity';
}

function formatProgress(loaded, total) {
    return total ? `${Math.round((loaded / total) * 100)}%` : `${(loaded / 1024 / 1024).toFixed(2)} MB`;
}

async function streamAndParseJson(response, progressCallback, { compressed = false } = {}) {
    const contentLength = response.headers.get('content-length');
    const total = contentLength ? parseInt(contentLength, 10) : 0;
    let loaded = 0;

    if (compressed && !('DecompressionStream' in window)) {
        throw new Error('此瀏覽器不支持DecompressionStream，無法直接讀取.json.gz。需要Chrome 80+/Firefox 113+/Safari 16.4+');
    }

    const progressStream = new TransformStream({
        transform(chunk, controller) {
            loaded += chunk.byteLength;
            if (progressCallback) progressCallback(loaded, total);
            controller.enqueue(chunk);
        }
    });

    let stream = response.body.pipeThrough(progressStream);
    if (compressed) stream = stream.pipeThrough(new DecompressionStream('gzip'));

    return JSON.parse(await new Response(stream).text());
}

function validateDictionaryPair(base, extra) {
    if (!Array.isArray(base) || !Array.isArray(extra) || !base.length || base.length !== extra.length) {
        throw new Error('基礎字庫與擴展字庫不完整或條目數不一致');
    }
}

async function fetchWithTimeout(url, options = {}) {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 10000);
    try {
        return await fetch(url, { ...options, signal: controller.signal });
    } finally {
        clearTimeout(timeout);
    }
}

async function fetchDictionaryVersion() {
    const response = await fetchWithTimeout(DICT_VERSION_URL, { cache: 'no-store' });
    if (!response.ok) throw new Error(`字庫版本檢查失敗：${response.status}`);
    const release = await response.json();
    if (!release || typeof release.version !== 'string' || !/^[\w.-]{1,80}$/.test(release.version) ||
        !Number.isFinite(Date.parse(release.updatedAt))) {
        throw new Error('字庫版本資訊無效');
    }
    return { version: release.version, updatedAt: release.updatedAt };
}

async function readCachedDictionarySource() {
    if (!('caches' in window)) return null;
    try {
        const stateCache = await caches.open(DICT_CACHE);
        const stateResponse = await stateCache.match(DICT_VERSION_URL);
        let state = null;
        try { state = stateResponse ? await stateResponse.json() : null; } catch (_error) {}
        if (!state || typeof state.version !== 'string' || typeof state.cacheName !== 'string' ||
            !state.cacheName.startsWith(DICT_RELEASE_PREFIX)) {
            return null;
        }
        try {
            const cache = await caches.open(state.cacheName);
            const responses = await Promise.all(DICT_URLS.map(url => cache.match(url)));
            if (responses.some(response => !response)) return null;
            const data = await Promise.all(responses.map(response => streamAndParseJson(
                response.clone(), null, { compressed: !responseBodyIsAlreadyDecoded(response) }
            )));
            validateDictionaryPair(...data);
            return {
                ...state,
                fromCache: true,
                responses: new Map(DICT_URLS.map((url, i) => [url, responses[i]])),
                data: new Map(DICT_URLS.map((url, i) => [url, data[i]])),
            };
        } catch (error) {
            console.warn('本地字庫無法讀取，將嘗試重新下載:', error);
        }
    } catch (error) {
        console.warn('瀏覽器無法開啟字庫緩存:', error);
    }
    return null;
}

function createDictionarySource(release) {
    return {
        ...release,
        cacheName: `${DICT_RELEASE_PREFIX}${release.version}`,
        fromCache: false,
        responses: new Map(),
        data: new Map(),
    };
}

async function getDictionarySource() {
    return (await readCachedDictionarySource()) || createDictionarySource(await fetchDictionaryVersion());
}

async function fetchGzJson(url, progressCallback, source) {
    let response = source.responses.get(url);
    if (!response) {
        // 版本號只在發佈字庫時變動，頁面、樣式更新不會使字庫重新下載。
        const requestUrl = new URL(url, document.baseURI);
        requestUrl.searchParams.set('v', source.version);
        response = await fetch(requestUrl.href);
        if (!response.ok) throw new Error(`請求${url}失敗：${response.status}`);
        source.responses.set(url, response.clone());
        source.data.set(url, await streamAndParseJson(response, progressCallback, {
            compressed: !responseBodyIsAlreadyDecoded(response),
        }));
    }
    const updatedAt = Date.parse(source.updatedAt) || Date.parse(response?.headers.get('last-modified') || '') || null;
    return { data: source.data.get(url), fromCache: source.fromCache, updatedAt };
}

async function saveDictionaryState(source) {
    const writeState = async () => {
        const cache = await caches.open(DICT_CACHE);
        let current;
        try { current = await (await cache.match(DICT_VERSION_URL))?.json(); } catch (_error) {}
        // 另一分頁可能已完成更新，較慢返回的舊版本檢查不能將它覆蓋。
        if (Date.parse(current?.updatedAt) > Date.parse(source.updatedAt)) return false;
        await cache.put(DICT_VERSION_URL, new Response(JSON.stringify({
            version: source.version,
            updatedAt: source.updatedAt,
            cacheName: source.cacheName,
            checkedAt: Date.now(),
        }), { headers: { 'Content-Type': 'application/json' } }));
        return true;
    };
    return navigator.locks?.request
        ? navigator.locks.request(`${DICT_CACHE}:${new URL(DICT_VERSION_URL, document.baseURI)}`, writeState)
        : writeState();
}

async function saveDictionarySource(source) {
    if (!('caches' in window)) return false;
    validateDictionaryPair(...DICT_URLS.map(url => source.data.get(url)));
    try {
        const cache = await caches.open(source.cacheName);
        await Promise.all(DICT_URLS.map(url => cache.put(url, source.responses.get(url).clone())));
        // 先存齊兩份資料，最後才切換指標；下載或寫入失敗時繼續使用完整舊版。
        return await saveDictionaryState(source);
    } catch (error) {
        console.warn('字庫已載入，但本地保存失敗:', error);
        return false;
    }
}

function scheduleDictionaryCacheRefresh(source) {
    const elapsed = Date.now() - source.checkedAt;
    if (!('caches' in window) || dictionaryRefreshPromise ||
        (source.checkedAt && elapsed >= 0 && elapsed < DICT_CHECK_INTERVAL)) return;
    dictionaryRefreshPromise = checkDictionaryCacheRefresh(source)
        .catch(error => console.warn('字庫更新失敗，繼續使用本地資料:', error))
        .finally(() => { dictionaryRefreshPromise = null; });
}

async function checkDictionaryCacheRefresh(source) {
    const release = await fetchDictionaryVersion();
    if (source.version === release.version) {
        await saveDictionaryState(source);
        return;
    }

    const next = createDictionarySource(release);
    await Promise.all(DICT_URLS.map(url => fetchGzJson(url, null, next)));
    validateDictionaryPair(...DICT_URLS.map(url => next.data.get(url)));
    // 若下載期間又發佈了新版，不提交可能來自不同發佈批次的檔案。
    if ((await fetchDictionaryVersion()).version !== release.version) return;
    if (await saveDictionarySource(next)) showUpdateNotification();
}

function requestDictionaryPersistence() {
    if (navigator.storage?.persisted && navigator.storage?.persist) {
        navigator.storage.persisted()
            .then(persistent => persistent || navigator.storage.persist())
            .catch(() => {});
    }
}

async function copyText(text, btnElement) {
    try {
        await navigator.clipboard.writeText(text);
        const originalText = btnElement.innerText;
        btnElement.innerText = '✔️';
        setTimeout(() => { btnElement.innerText = originalText; }, 1500);
    } catch (err) {
        console.error('複製失敗:', err);
    }
}

function loadImageExportLibrary() {
    if (window.htmlToImage) return Promise.resolve(window.htmlToImage);
    if (imageExportLoader) return imageExportLoader;

    imageExportLoader = new Promise((resolve, reject) => {
        const script = document.createElement('script');
        const timeoutId = window.setTimeout(() => {
            script.remove();
            reject(new Error('圖片導出組件加載超時'));
        }, IMAGE_EXPORT_TIMEOUT);
        script.src = IMAGE_EXPORT_URL;
        script.async = true;
        script.onload = () => {
            window.clearTimeout(timeoutId);
            if (window.htmlToImage) resolve(window.htmlToImage);
            else reject(new Error('圖片導出組件未提供 htmlToImage'));
        };
        script.onerror = () => {
            window.clearTimeout(timeoutId);
            script.remove();
            reject(new Error('圖片導出組件加載失敗'));
        };
        document.head.appendChild(script);
    }).catch(error => {
        imageExportLoader = null;
        throw error;
    });

    return imageExportLoader;
}

function safeFileName(value, fallback = 'result-card') {
    const cleaned = String(value || '')
        .replace(/[<>:"/\\|?*\x00-\x1F]/g, '_')
        .replace(/\s+/g, ' ')
        .trim()
        .slice(0, 60);
    return cleaned || fallback;
}

async function exportCard(card) {
    const btn = card?.querySelector('.export-btn');
    if (!card || !btn || btn.disabled) return;

    const originalText = btn.textContent;
    btn.disabled = true;
    btn.setAttribute('aria-busy', 'true');
    btn.textContent = '準備中…';

    try {
        const htmlToImage = await loadImageExportLibrary();
        if (document.fonts?.ready) await document.fonts.ready;
        btn.textContent = '導出中…';

        const dataUrl = await htmlToImage.toPng(card, {
            pixelRatio: Math.min(2, window.devicePixelRatio || 1),
            backgroundColor: window.getComputedStyle(card).backgroundColor,
            cacheBust: true,
            filter: node => !node.classList?.contains('copy-btn') && !node.classList?.contains('export-btn'),
            style: { margin: '0', animation: 'none', transform: 'none', opacity: '1' }
        });

        const zi = safeFileName(card.dataset.char, 'result-card');
        const phonetic = safeFileName(card.dataset.phonetic, '');
        const link = document.createElement('a');
        link.download = phonetic ? `pboc-${zi}-${phonetic}.png` : `pboc-${zi}.png`;
        link.href = dataUrl;
        document.body.appendChild(link);
        link.click();
        link.remove();
        btn.textContent = '已儲存';
    } catch (error) {
        console.error('圖片導出失敗:', error);
        btn.textContent = '導出失敗';
    } finally {
        btn.removeAttribute('aria-busy');
        window.setTimeout(() => {
            btn.textContent = originalText;
            btn.disabled = false;
        }, 1600);
    }
}

function setSearchMode(mode) {
    const radio = mode && SEARCH_MODES.has(mode)
        ? document.querySelector(`input[name="searchMode"][value="${mode}"]`)
        : null;
    if (radio) radio.checked = true;
    syncSearchModeUI();
}

function applySearch(query, mode, scrollTop = false) {
    $('searchInput').value = query;
    setSearchMode(mode);
    searchDict();
    if (scrollTop) window.scrollTo({ top: 0, behavior: 'smooth' });
}

function syncSearchModeUI() {
    const searchInput = $('searchInput');
    const mode = document.querySelector('input[name="searchMode"]:checked')?.value;
    if (!searchInput || !mode) return;
    if (mode === 'batch') {
        searchInput.maxLength = BATCH_MAX_LENGTH;
        if (!searchInput.value) searchInput.placeholder = `輸入古文文本（最多${BATCH_MAX_LENGTH}字），自動標注擬音…`;
    } else {
        searchInput.setAttribute('maxlength', '100');
        if (searchInput.placeholder.includes('標注擬音')) {
            searchInput.placeholder = dictData.length
                ? `已加載${dictData.length}條目—請輸入關鍵字`
                : '正在加載字庫…';
        }
    }
}

function handleResultsClick(event) {
    if (!(event.target instanceof Element)) return;

    const copyBtn = event.target.closest('.copy-btn');
    if (copyBtn) {
        copyText(copyBtn.dataset.copyText || '', copyBtn);
        return;
    }

    const exactSearchBtn = event.target.closest('[data-exact-search]');
    if (exactSearchBtn) {
        applySearch(exactSearchBtn.dataset.exactSearch || '', 'exact', true);
        return;
    }

    const exportBtn = event.target.closest('.export-btn');
    if (exportBtn) {
        exportCard(exportBtn.closest('.result-card'));
        return;
    }

    const batchCopyBtn = event.target.closest('[data-batch-copy]');
    if (!batchCopyBtn || !_batchRows) return;

    const format = batchCopyBtn.dataset.batchCopy;
    if (format === 'tsv') {
        const lines = ['字\t拼音\t擬音\t諧聲域', ..._batchRows.flatMap(row =>
            row.items.length
                ? row.items.map(it => `${row.char}\t${it.p || ''}\t${it.y || ''}\t${it.a || ''}`)
                : [`${row.char}\t\t\t未收錄`]
        )];
        copyText(lines.join('\n'), batchCopyBtn);
    } else if (format === 'text') {
        const seen = new Set();
        const parts = _batchRows.filter(row => !seen.has(row.char) && seen.add(row.char)).map(row =>
            `${row.char}（${row.items.length ? row.items.map(it => it.y || '?').join(' / ') : '?'}）`
        );
        copyText(parts.join('\n'), batchCopyBtn);
    }
}

function updateDataUpdatedAt(timestamps) {
    const element = $('dataUpdatedAt');
    if (!element) return;

    const latest = timestamps.filter(Number.isFinite).reduce((a, b) => Math.max(a, b), 0);
    element.textContent = latest
        ? `資料最後更新：${new Intl.DateTimeFormat(undefined, {
            year: 'numeric', month: 'long', day: 'numeric', hour: 'numeric', minute: 'numeric', second: 'numeric'
        }).format(new Date(latest))}`
        : '資料最後更新：無法取得';
}

function renderWelcomePage() {
    const resultsDiv = $('results');
    const template = $('welcomeTemplate');
    if (!resultsDiv || !template) return;

    resultsDiv.replaceChildren(template.content.cloneNode(true));
    resultsDiv.setAttribute('aria-label', 'PBOC 說明');
}

async function initDictionary() {
    if (dictionaryLoading) return;
    dictionaryLoading = true;
    const input  = $('searchInput');
    const btn    = $('searchBtn');
    const status = $('extraStatus');
    let source;
    let baseUpdatedAt = null;

    try {
        btn.dataset.action = 'loading';
        btn.disabled = true;
        btn.textContent = '加載中...';
        input.placeholder = '讀取本地字庫…';
        source = await getDictionarySource();
        input.placeholder = source.fromCache ? '讀取本地基礎字庫…' : '下載基礎字庫…';
        const { data: baseData, updatedAt } = await fetchGzJson('base.json.gz', (loaded, total) => {
            input.placeholder = `下載基礎字庫... ${formatProgress(loaded, total)}`;
        }, source);
        baseUpdatedAt = updatedAt;
        updateDataUpdatedAt([baseUpdatedAt]);
        dictData = baseData;
        buildIndexes();
        input.placeholder = `已加載${dictData.length}條目—請輸入關鍵字`;
        input.disabled = false;
        btn.disabled = false;
        btn.dataset.action = 'search';
        btn.textContent = '查詢';

        loadFromUrl();
    } catch (error) {
        console.error('基礎字典加載錯誤:', error);
        updateDataUpdatedAt([]);
        input.placeholder = '加載字典失敗，請檢查網絡';
        btn.textContent = '重試';
        btn.dataset.action = 'retry';
        btn.disabled = false;
        dictionaryLoading = false;
        return;
    }

    try {
        status.style.display = 'block';
        status.style.color = 'var(--gold)';
        status.innerText = source.fromCache ? '讀取本地釋義與注釋…' : '下載釋義與注釋…';
        const { data: extraData, updatedAt: extraUpdatedAt } = await fetchGzJson('extra.json.gz', (loaded, total) => {
            status.innerText = `下載釋義與注釋... ${formatProgress(loaded, total)}`;
        }, source);

        updateDataUpdatedAt([baseUpdatedAt, extraUpdatedAt]);

        validateDictionaryPair(dictData, extraData);
        extraData.forEach((ext, i) => { Object.assign(dictData[i], ext); });
        buildIndexes({ includeExtra: true });

        status.innerText = '擴展數據與索引加載完成！';
        if (input.value.trim()) searchDict(false);

        if (source.fromCache) {
            scheduleDictionaryCacheRefresh(source);
        } else if (!await saveDictionarySource(source)) {
            status.innerText = '字庫已載入，但無法保存至本地；下次開啟可能需要重新下載';
            status.style.color = 'var(--vermilion)';
            dictionaryLoading = false;
            return;
        }
        requestDictionaryPersistence();
        setTimeout(() => { status.style.display = 'none'; }, 2000);
    } catch (error) {
        console.error('擴充字典加載錯誤:', error);
        updateDataUpdatedAt([baseUpdatedAt]);
        status.innerText = '釋義加載失敗，目前僅提供基礎查詢';
        status.style.color = 'var(--vermilion)';
    }

    dictionaryLoading = false;
}

function loadFromUrl() {
    const params = new URLSearchParams(window.location.search);
    const q = params.get('q');
    const sort = params.get('sort');
    pageSize = normalizePageSize(params.get('size') || pageSize);
    pendingUrlPage = Math.max(1, parseInt(params.get('page'), 10) || 1);
    currentSort = SORT_OPTIONS.some(o => o.value === sort) ? sort : 'default';

    setSearchMode(params.get('mode'));
    const input = $('searchInput');
    if (q) {
        input.value = q;
        searchDict(false);
    } else {
        input.value = '';
        currentQuery = '';
        currentResults = [];
        renderWelcomePage();
        renderSearchHistory();
    }
}

function safeRegex(str, flags = '') {
    if (str.length > REGEX_MAX_LEN) return { ok: false, msg: `正則表達式過長（${str.length}/${REGEX_MAX_LEN}字元）` };
    if (REDOS_PATTERN.test(str)) return { ok: false, msg: '正則表達式含有可能造成卡頓的模式，已拒絕執行' };
    try {
        const safeFlags = Array.from(new Set(flags.replace(/g/g, ''))).join('');
        return { ok: true, regex: new RegExp(str, safeFlags) };
    } catch (e) {
        return { ok: false, msg: `正則語法錯誤：${e.message}` };
    }
}

function parseLinks(text, formatText = escapeHtml) {
    if (!text) return text;
    const urlRegex = /((?:https?:\/\/)?(?:www\.)?[a-zA-Z0-9](?:[a-zA-Z0-9.-]*[a-zA-Z0-9])?\.[a-zA-Z]{2,}(?:\/[a-zA-Z0-9._~:/?@!$&'()*+,;=%#-]*)?)/g;

    let html = '';
    let lastIndex = 0;
    for (const match of text.matchAll(urlRegex)) {
        const url = match[0];
        html += formatText(text.slice(lastIndex, match.index));
        const cleanUrl = url.replace(/[.,?!)\]]+$/, '');
        const trailingText = url.slice(cleanUrl.length);
        if (url.length < 7) {
            html += formatText(url);
        } else {
            const href = cleanUrl.startsWith('http') ? cleanUrl : `https://${cleanUrl}`;
            html += `<a href="${escapeHtml(href)}" target="_blank" rel="noopener" class="annotation-link">${escapeHtml(cleanUrl)}</a>`;
            html += formatText(trailingText);
        }
        lastIndex = match.index + url.length;
    }
    return html + formatText(text.slice(lastIndex));
}

function renderAnnotation(text) {
    const lines = String(text).split(/\r?\n|\r/).filter(line => line.trim());
    if (!lines.length) return '';
    return `<ul class="annotation-list">${lines.map(line =>
        `<li>${parseLinks(line, highlightMeaningSearchTerm)}</li>`
    ).join('')}</ul>`;
}

function highlightMeaningSearchTerm(text) {
    const escapedText = escapeHtml(text);
    if (currentMode !== 'meaning' || !currentQuery) return escapedText;
    const escapedQuery = escapeHtml(currentQuery);
    return escapedText.split(escapedQuery).join(`<mark class="meaning-search-hit">${escapedQuery}</mark>`);
}

function normalizePageSize(value) {
    const parsed = parseInt(value, 10);
    return PAGE_SIZE_OPTIONS.includes(parsed) ? parsed : DEFAULT_PAGE_SIZE;
}

function readSearchHistory() {
    try { return JSON.parse(readStorage(HISTORY_STORAGE_KEY, '[]')); } catch (_e) { return []; }
}

function updateSearchHistory(mutator, rerender = false) {
    const next = mutator(readSearchHistory());
    writeStorage(HISTORY_STORAGE_KEY, JSON.stringify(next));
    if (rerender) renderSearchHistory();
}

function renderSearchHistory() {
    const container = $('searchHistory');
    if (!container) return;
    const history = readSearchHistory();
    if (!history.length) { container.replaceChildren(); return; }

    container.innerHTML = `
        <div class="search-history-header">
            <span class="search-history-title">最近查詢</span>
            <button class="history-clear-btn" type="button">清空歷史</button>
        </div>
        <ul class="search-history-list">${history.map(h => {
            const safeQ = escapeHtml(h.query);
            const modeLabel = MODE_LABELS[h.mode] || h.mode;
            return `<li class="history-item" role="button" tabindex="0" aria-label="再次查詢：${safeQ}" data-query="${safeQ}" data-mode="${escapeHtml(h.mode)}">
                <span class="history-item-query" title="${safeQ}">${safeQ}</span>
                ${h.mode !== 'default' ? `<span class="history-item-mode">${escapeHtml(modeLabel)}</span>` : ''}
                <button class="history-item-delete" type="button" title="刪除" aria-label="刪除「${safeQ}」">&times;</button>
            </li>`;
        }).join('')}</ul>`;
}

function sortResults(results, sortKey) {
    const field = SORT_OPTIONS.find(o => o.value === sortKey)?.field;
    return (!field || !results.length)
        ? results
        : [...results].sort((a, b) => zhCollator.compare(a[field] || '', b[field] || ''));
}

function updateSearchUrl() {
    const url = new URL(window.location);
    const params = currentQuery ? {
        q: currentQuery,
        mode: currentMode,
        page: currentPage > 1 ? String(currentPage) : null,
        size: pageSize !== DEFAULT_PAGE_SIZE ? String(pageSize) : null,
        sort: currentSort !== 'default' ? currentSort : null,
    } : { q: null, mode: null, page: null, size: null, sort: null };

    for (const [k, v] of Object.entries(params)) {
        if (v == null) url.searchParams.delete(k);
        else url.searchParams.set(k, v);
    }
    if (window.location.search !== url.search) window.history.pushState({}, '', url);
}

function pageWindow(page, totalPages) {
    const pages = new Set([1, totalPages]);
    for (let p = page - 2; p <= page + 2; p++) {
        if (p >= 1 && p <= totalPages) pages.add(p);
    }
    return Array.from(pages).sort((a, b) => a - b);
}

function createPageButton(label, page, disabled = false, current = false) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.textContent = label;
    btn.disabled = disabled;
    if (current) btn.setAttribute('aria-current', 'page');
    btn.setAttribute('aria-label', current ? `第 ${page} 頁，目前頁` : `前往第 ${page} 頁`);
    btn.addEventListener('click', () => {
        if (!disabled && !current) goToPage(page);
    });
    return btn;
}

function createSelectControl(className, labelText, ariaLabel, options, selectedValue, onChange) {
    const label = document.createElement('label');
    label.className = className;
    label.textContent = labelText;
    const select = document.createElement('select');
    select.setAttribute('aria-label', ariaLabel);
    options.forEach(({ value, label: text }) => {
        const option = document.createElement('option');
        option.value = String(value);
        option.textContent = text;
        option.selected = value === selectedValue;
        select.appendChild(option);
    });
    select.addEventListener('change', () => onChange(select.value));
    label.appendChild(select);
    return label;
}

function createPaginationBar(total, position = 'top', includeOptions = position === 'top') {
    const totalPages = Math.max(1, Math.ceil(total / pageSize));
    const start = (currentPage - 1) * pageSize + 1;
    const end = Math.min(total, currentPage * pageSize);
    const bar = document.createElement('nav');
    bar.className = `pagination-bar ${position}`;
    bar.setAttribute('aria-label', position === 'top' ? '查詢結果分頁' : '查詢結果底部分頁');

    const summary = document.createElement('div');
    summary.className = 'pagination-summary';
    summary.innerHTML = `找到<b>${total}</b>條，顯示<b>${start}–${end}</b>條 / 第<b>${currentPage}</b>頁，共<b>${totalPages}</b>頁`;
    bar.appendChild(summary);

    const controls = document.createElement('div');
    controls.className = 'pagination-controls';

    if (includeOptions) {
        controls.appendChild(createSelectControl(
            'pagination-size', '每頁', '每頁顯示條數',
            PAGE_SIZE_OPTIONS.map(size => ({ value: size, label: `${size}條` })),
            pageSize, changePageSize
        ));
        controls.appendChild(createSelectControl(
            'sort-select', '排序', '結果排序方式',
            SORT_OPTIONS, currentSort, changeSort
        ));
    }

    if (totalPages > 1) {
        controls.appendChild(createPageButton('首頁', 1, currentPage === 1));
        controls.appendChild(createPageButton('上一頁', currentPage - 1, currentPage === 1));

        let last = 0;
        pageWindow(currentPage, totalPages).forEach(page => {
            if (page - last > 1) {
                const ellipsis = document.createElement('span');
                ellipsis.className = 'pagination-ellipsis';
                ellipsis.textContent = '…';
                controls.appendChild(ellipsis);
            }
            controls.appendChild(createPageButton(String(page), page, false, page === currentPage));
            last = page;
        });

        controls.appendChild(createPageButton('下一頁', currentPage + 1, currentPage === totalPages));
        controls.appendChild(createPageButton('末頁', totalPages, currentPage === totalPages));
    }

    bar.appendChild(controls);
    return bar;
}

function renderResultCard(item, idx) {
    const card = document.createElement('div');
    card.className = 'result-card';
    card.lang = document.documentElement.lang;
    card.style.animationDelay = `${Math.min(idx, 10) * 0.06}s`;
    card.dataset.char = item.z || '';
    card.dataset.phonetic = item.y || '';
    const safeZ = escapeHtml(item.z) || '？';
    const safeA = escapeHtml(item.a);
    const safeY = escapeHtml(item.y) || '—';
    const safeP = escapeHtml(item.p) || '—';
    const safeM = escapeHtml(item.m);
    const ziUrl = item.z ? encodeURIComponent(item.z) : '';
    const rhymeBase = `https://suzukish.sakura.ne.jp/search/xianqin/index.php?yunzi=${ziUrl}`;

    let tags = '';
    if (item.a) tags += `<button class="tag xiesheng-tag" type="button" data-exact-search="${safeA}" title="點擊查找同諧聲域字" aria-label="諧聲域 ${safeA}，點擊查詢同域字">諧聲域: ${safeA}</button>`;
    if (item.s === 1) tags += `<a class="tag rhyme-tag" href="${rhymeBase}" target="_blank" rel="noopener noreferrer" title="在先秦兩漢韻部資料庫查閱詩經韻">見詩經韻</a>`;
    if (item.q === 1) tags += `<a class="tag rhyme-tag" href="${rhymeBase}" target="_blank" rel="noopener noreferrer" title="在先秦兩漢韻部資料庫查閱戰國韻">見戰國韻</a>`;

    let meaningsHtml = '<p class="empty-meaning">暫無釋義</p>';
    let meaningNote = '';
    if (item.d?.[1]?.length > 0) {
        meaningNote = item.d[0] ? `<span class="section-note">（${escapeHtml(item.d[0])}）</span>` : '';
        meaningsHtml = `<ol class="meanings-list">${item.d[1].map((m, i) =>
            `<li><span class="sense-num">${i + 1}</span><span class="meaning-text">${parseLinks(m, highlightMeaningSearchTerm)}</span></li>`
        ).join('')}</ol>`;
    }

    const annotationHtml = item.n ? renderAnnotation(item.n) : '';

    card.innerHTML = `
        <div class="char-banner">
            <div class="char-glyph-wrap">
                <div class="char-glyph">${safeZ}</div>
                ${item.z ? `<a class="zi-link" href="https://zi.tools/zi/${ziUrl}" target="_blank" rel="noopener" title="在字統網查閱字源">字統↗</a>` : ''}
            </div>
            <div class="char-phonetics">
                <div class="phonetic-row">
                    <span class="phonetic-label">擬音</span>
                    <span class="phonetic-value">${safeY}</span>
                    ${item.y ? `<button class="copy-btn" type="button" data-copy-text="${safeY}" title="複製擬音" aria-label="複製擬音">📋</button>` : ''}
                </div>
                <div class="phonetic-row">
                    <span class="phonetic-label">拼音</span>
                    <span class="pinyin-value">${safeP}</span>
                    ${item.p ? `<button class="copy-btn" type="button" data-copy-text="${safeP}" title="複製拼音" aria-label="複製拼音">📋</button>` : ''}
                </div>
                ${item.m ? `
                <div class="phonetic-row">
                    <span class="phonetic-label">聲首</span>
                    <button class="initial-value initial-search" type="button" data-exact-search="${safeM}" title="點擊查找同聲首字" aria-label="聲首 ${safeM}，點擊查詢同聲首字">${safeM}</button>
                </div>` : ''}
                ${tags ? `<div class="char-tags">${tags}</div>` : ''}
            </div>
            <button class="export-btn" type="button" title="導出爲圖片" aria-label="導出此條目爲圖片">📷 導出</button>
        </div>
        <div class="card-body">
            <div class="section">
                <div class="section-header">
                    <span class="section-label">釋義</span>${meaningNote}
                </div>
                ${meaningsHtml}
            </div>
            ${annotationHtml ? `
            <div class="section-divider"></div>
            <div class="section">
                <div class="section-header">
                    <span class="section-label">注釋</span>
                </div>
                <div class="text-block">${annotationHtml}</div>
            </div>` : ''}
        </div>`;
    return card;
}

function renderPaginatedResults(updateUrl = true, shouldScroll = false) {
    const resultsDiv = $('results');
    resultsDiv.replaceChildren();
    if (!currentQuery) return;

    if (!currentResults.length) {
        resultsDiv.setAttribute('aria-label', '查詢結果：無結果');
        resultsDiv.innerHTML = `<div class="no-result" role="status">未找到「${escapeHtml(currentQuery)}」的相關條目<br><span>可嘗試輸入單個漢字、拼音、諧聲域或使用正則</span></div>`;
        if (updateUrl) updateSearchUrl();
        return;
    }

    const sortedResults = sortResults(currentResults, currentSort);
    const totalPages = Math.max(1, Math.ceil(sortedResults.length / pageSize));
    currentPage = Math.min(Math.max(1, currentPage), totalPages);
    const startIndex = (currentPage - 1) * pageSize;
    const displayResults = sortedResults.slice(startIndex, startIndex + pageSize);
    resultsDiv.setAttribute('aria-label', `查詢結果：共 ${sortedResults.length} 條，第 ${currentPage} 頁，共 ${totalPages} 頁`);

    resultsDiv.appendChild(createPaginationBar(sortedResults.length, 'top'));

    const fragment = document.createDocumentFragment();
    displayResults.forEach((item, idx) => fragment.appendChild(renderResultCard(item, idx)));
    resultsDiv.appendChild(fragment);

    if (currentResults.length > pageSize) {
        resultsDiv.appendChild(createPaginationBar(sortedResults.length, 'bottom', false));
    }

    if (updateUrl) updateSearchUrl();
    if (shouldScroll) {
        document.querySelector('.search-container')?.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
}

function goToPage(page) {
    const totalPages = Math.max(1, Math.ceil(currentResults.length / pageSize));
    const nextPage = Math.min(Math.max(1, parseInt(page, 10) || 1), totalPages);
    if (nextPage === currentPage) return;
    currentPage = nextPage;
    renderPaginatedResults(true, true);
}

function changePageSize(value) {
    pageSize = normalizePageSize(value);
    writeStorage(PAGE_SIZE_STORAGE_KEY, String(pageSize));
    currentPage = 1;
    renderPaginatedResults(true, true);
}

function changeSort(value) {
    if (currentSort === value) return;
    currentSort = value;
    currentPage = 1;
    renderPaginatedResults(true, false);
}

function searchDict(updateUrl = true) {
    const query = $('searchInput').value.trim();
    const resultsDiv = $('results');
    const regexErrorDiv = $('regexError');
    const mode = document.querySelector('input[name="searchMode"]:checked').value;

    currentQuery = query;
    currentMode = mode;
    currentResults = [];
    currentPage = updateUrl ? 1 : pendingUrlPage;

    resultsDiv.replaceChildren();
    regexErrorDiv.style.display = 'none';
    regexErrorDiv.textContent = '';
    if (!query) {
        if (updateUrl) updateSearchUrl();
        renderWelcomePage();
        renderSearchHistory();
        return;
    }

    $('searchHistory')?.replaceChildren();

    // Batch mode
    if (mode === 'batch') {
        renderBatchResults(query);
        if (updateUrl) updateSearchUrl();
        return;
    }

    let results = [];
    if (mode === 'meaning') {
        const [firstChar] = query;
        results = Array.from(meaningIndex.get(firstChar) || [], i => dictData[i])
            .filter(item => item.d?.[1]?.some(m => m.includes(query)) || item.n?.includes(query));
    } else {
        const isRegexMode = mode === 'regex' || (query.startsWith('/') && query.endsWith('/') && query.length > 2);
        const isExactMode = mode === 'exact' || (query.startsWith('"') && query.endsWith('"') && query.length > 2);

        if (isRegexMode) {
            const match = query.match(/^\/(.+)\/([gimsuy]*)$/);
            const result = safeRegex(match ? match[1] : query, match ? match[2] : '');
            if (!result.ok) {
                regexErrorDiv.textContent = result.msg;
                regexErrorDiv.style.display = 'block';
                return;
            }
            results = dictData.filter(item => regexMatches(result.regex, item.z, item.y, item.p, item.a));
        } else if (isExactMode) {
            const exactStr = (query.startsWith('"') && query.endsWith('"')) ? query.slice(1, -1) : query;
            const seen = new Set();
            ALL_INDEX_KEYS.forEach(k => appendUnique(results, seen, exactIndexes[k].get(exactStr)));
        } else {
            const tokens = query.split(/\s+/).flatMap(p =>
                (p.match(CJK_TOKEN_RE) || []).flatMap(m => CJK_RE.test(m) ? Array.from(m) : [m])
            );
            if (!tokens.length) tokens.push(query);

            const seen = new Set();
            tokens.forEach(val => {
                const isUpper = /^[A-Z0-9]+$/.test(val);
                const isHan = CJK_RE.test(val);
                BASE_INDEX_KEYS.forEach(k => appendUnique(results, seen, exactIndexes[k].get(val)));

                if (isUpper) {
                    for (const [domain, items] of exactIndexes.a) {
                        if (domain.includes(val)) appendUnique(results, seen, items);
                    }
                }
                if (!isHan) {
                    dictData.forEach(item => {
                        if (!seen.has(item) && (item.y?.includes(val) || item.p?.includes(val))) {
                            seen.add(item);
                            results.push(item);
                        }
                    });
                }
            });
        }
    }

    currentResults = results;
    if (results.length) {
        updateSearchHistory(history => [
            { query, mode },
            ...history.filter(h => !(h.query === query && h.mode === mode)),
        ].slice(0, HISTORY_MAX));
    }
    renderPaginatedResults(updateUrl);
}

function renderBatchResults(text) {
    const resultsDiv = $('results');
    const allChars = Array.from(text);
    const chars = allChars.slice(0, BATCH_MAX_LENGTH).filter(ch => CJK_RE.test(ch));
    if (!chars.length) {
        resultsDiv.innerHTML = '<div class="no-result" role="status">未找到可標音的漢字<br><span>請輸入包含漢字的文本</span></div>';
        return;
    }

    const rows = chars.map(char => ({ char, items: exactIndexes.z.get(char) || [] }));
    _batchRows = rows;

    const totalChars = rows.length;
    const foundChars = rows.filter(r => r.items.length > 0).length;
    const totalRows = rows.reduce((sum, r) => sum + Math.max(1, r.items.length), 0);

    let tableRows = '';
    let idx = 0;
    rows.forEach(row => {
        if (!row.items.length) {
            idx++;
            tableRows += `<tr><td>${idx}</td><td class="batch-char">${escapeHtml(row.char)}</td><td class="batch-not-found" colspan="3">未收錄</td></tr>`;
        } else {
            const span = row.items.length > 1 ? ` rowspan="${row.items.length}"` : '';
            row.items.forEach((item, i) => {
                idx++;
                const lead = i === 0 ? `<td${span}>${idx}</td><td class="batch-char"${span}>${escapeHtml(row.char)}</td>` : '';
                tableRows += `<tr>${lead}
                    <td class="batch-pinyin">${escapeHtml(item.p || '—')}</td>
                    <td class="batch-phonetic">${escapeHtml(item.y || '—')}</td>
                    <td class="batch-domain">${escapeHtml(item.a || '—')}</td></tr>`;
            });
        }
    });

    const truncationNote = allChars.length > BATCH_MAX_LENGTH ? `，已截取前${BATCH_MAX_LENGTH}字` : '';
    resultsDiv.innerHTML = `
        <div class="batch-container">
            <div class="batch-actions">
                <button type="button" data-batch-copy="tsv" title="複製為 TSV 格式（可貼入 Excel）">📋 複製為 TSV</button>
                <button type="button" data-batch-copy="text" title="複製為文本格式（字＋擬音）">📋 複製為標音文本</button>
            </div>
            <div class="batch-table-wrap">
                <table class="batch-table">
                    <thead><tr><th>#</th><th>字</th><th>拼音</th><th>擬音</th><th>諧聲域</th></tr></thead>
                    <tbody>${tableRows}</tbody>
                </table>
            </div>
            <p class="batch-summary">共 ${totalChars} 字，${foundChars} 字有擬音，${totalChars - foundChars} 字未收錄，展開 ${totalRows} 行${truncationNote}</p>
        </div>`;
}

function setupEventListeners() {
    // 全域快捷鍵：按下 / 快速 focus 搜尋框
    document.addEventListener('keydown', e => {
        const t = e.target;
        const isTyping = t instanceof HTMLElement &&
            (t.tagName === 'INPUT' || t.tagName === 'TEXTAREA' || t.tagName === 'SELECT' || t.isContentEditable);
        if (e.key === '/' && !e.metaKey && !e.ctrlKey && !e.altKey && !isTyping) {
            e.preventDefault();
            $('searchInput')?.focus();
        }
    });

    $('searchInput')?.addEventListener('keydown', e => {
        if (e.key === 'Enter' && !e.isComposing) searchDict();
    });
    $('searchBtn')?.addEventListener('click', e => {
        if (e.currentTarget.dataset.action === 'retry') initDictionary();
        else searchDict();
    });
    document.querySelectorAll('input[name="searchMode"]').forEach(input => {
        input.addEventListener('change', () => {
            syncSearchModeUI();
            searchDict();
        });
    });
    syncSearchModeUI();

    $('reloadBtn')?.addEventListener('click', () => location.reload());
    $('dismissUpdateBtn')?.addEventListener('click', () => {
        const bar = $('updateBar');
        if (bar) bar.style.display = 'none';
    });
    $('results')?.addEventListener('click', handleResultsClick);

    const historyEl = $('searchHistory');
    historyEl?.addEventListener('click', e => {
        if (!(e.target instanceof Element)) return;
        if (e.target.closest('.history-clear-btn')) {
            updateSearchHistory(() => [], true);
            return;
        }
        const item = e.target.closest('.history-item');
        if (!item) return;
        if (e.target.closest('.history-item-delete')) {
            updateSearchHistory(h => h.filter(x => !(x.query === item.dataset.query && x.mode === item.dataset.mode)), true);
        } else {
            applySearch(item.dataset.query, item.dataset.mode);
        }
    });
    historyEl?.addEventListener('keydown', e => {
        if ((e.key !== 'Enter' && e.key !== ' ') || !(e.target instanceof Element)) return;
        const item = e.target.closest('.history-item');
        if (!item || e.target.closest('button')) return;
        e.preventDefault();
        applySearch(item.dataset.query, item.dataset.mode);
    });

    const backToTop = $('backToTop');
    const searchContainer = document.querySelector('.search-container');
    if (backToTop && searchContainer) {
        new IntersectionObserver(
            ([entry]) => backToTop.classList.toggle('visible', !entry.isIntersecting),
            { threshold: 0, rootMargin: '0px' }
        ).observe(searchContainer);

        backToTop.addEventListener('click', () => {
            searchContainer.scrollIntoView({ behavior: 'smooth', block: 'start' });
            $('searchInput')?.focus({ preventScroll: true });
        });
    }

    window.addEventListener('popstate', loadFromUrl);
}

setupEventListeners();
renderWelcomePage();
initDictionary();

if ('serviceWorker' in navigator) {
    if (['localhost', '127.0.0.1', '[::1]'].includes(window.location.hostname)) {
        navigator.serviceWorker.getRegistrations()
            .then(regs => Promise.all(regs.map(r => r.unregister())))
            .catch(() => {});
    } else {
        window.addEventListener('load', () => {
            let hadController = !!navigator.serviceWorker.controller;
            navigator.serviceWorker.register('sw.js', { scope: './' }).catch(() => {});
            navigator.serviceWorker.addEventListener('controllerchange', () => {
                if (hadController) showUpdateNotification();
                hadController = true;
            });
        });
    }
}
