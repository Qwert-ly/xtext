let dictData = [];
const meaningIndex = new Map();
const zIndex = new Map();
const pinyinIndex = new Map();
const phoneticIndex = new Map();
const domainExactIndex = new Map();
const DICT_CACHE = 'poc-dict-v3';
const DICT_URLS = ['base.json.gz', 'extra.json.gz'];
const SEARCH_MODES = new Set(['default', 'exact', 'regex', 'meaning']);
const supportsGzipStreams = 'DecompressionStream' in window;
let _updatedFiles = false;
let dictionaryLoading = false;

// 全域快捷鍵：按下 / 快速 focus 搜尋框
document.addEventListener('keydown', (e) => {
    const target = e.target;
    const isTyping = target instanceof HTMLElement && (
        target.tagName === 'INPUT' ||
        target.tagName === 'TEXTAREA' ||
        target.tagName === 'SELECT' ||
        target.isContentEditable
    );

    if (e.key === '/' && !e.metaKey && !e.ctrlKey && !e.altKey && !isTyping) {
        e.preventDefault();
        document.getElementById('searchInput').focus();
    }
});

function escapeHtml(unsafe) {
    if (typeof unsafe !== 'string') return unsafe;
    return unsafe
         .replace(/&/g, "&amp;")
         .replace(/</g, "&lt;")
         .replace(/>/g, "&gt;")
         .replace(/"/g, "&quot;")
         .replace(/'/g, "&#039;");
}

let updateNotified = false;
function showUpdateNotification() {
    if (updateNotified) return;
    updateNotified = true;
    const bar = document.getElementById('updateBar');
    if (bar) bar.style.display = 'flex';
}

function addToIndex(index, key, item) {
    if (!key) return;
    const normalized = String(key).trim();
    if (!normalized) return;
    if (!index.has(normalized)) index.set(normalized, []);
    index.get(normalized).push(item);
}

function buildBaseIndexes() {
    zIndex.clear();
    pinyinIndex.clear();
    phoneticIndex.clear();
    domainExactIndex.clear();

    dictData.forEach(item => {
        addToIndex(zIndex, item.z, item);
        addToIndex(pinyinIndex, item.p, item);
        addToIndex(phoneticIndex, item.y, item);
        addToIndex(domainExactIndex, item.a, item);
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

async function streamAndParseJson(response, progressCallback, { compressed = false } = {}) {
    const contentLength = response.headers.get('content-length');
    const total = contentLength ? parseInt(contentLength, 10) : 0;
    let loaded = 0;

    if (compressed && !supportsGzipStreams) {
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
    if (compressed) {
        stream = stream.pipeThrough(new DecompressionStream('gzip'));
    }

    return JSON.parse(await new Response(stream).text());
}

async function streamAndParseGz(response, progressCallback) {
    return streamAndParseJson(response, progressCallback, {
        compressed: !responseBodyIsAlreadyDecoded(response),
    });
}

function sameCachedPayload(cachedResp, netResp) {
    const etag = cachedResp.headers.get('ETag');
    const newEtag = netResp.headers.get('ETag');
    if (etag && newEtag) return etag === newEtag;

    const lastMod = cachedResp.headers.get('Last-Modified');
    const newLastMod = netResp.headers.get('Last-Modified');
    if (lastMod && newLastMod) return lastMod === newLastMod;

    if (etag || newEtag || lastMod || newLastMod) return false;

    const oldLen = cachedResp.headers.get('Content-Length');
    const newLen = netResp.headers.get('Content-Length');
    return !!oldLen && !!newLen && oldLen === newLen;
}

let dictionaryRefreshPromise = null;

function scheduleDictionaryCacheRefresh() {
    if (!('caches' in window) || dictionaryRefreshPromise) return;
    dictionaryRefreshPromise = checkDictionaryCacheRefresh()
        .catch(error => console.warn('字典背景更新检查失败:', error))
        .finally(() => { dictionaryRefreshPromise = null; });
}

async function cleanupLegacyDictionaryCaches() {
    if (!('caches' in window)) return;
    const keys = await caches.keys();
    await Promise.all(
        keys
            .filter(key => (
                (key.startsWith('poc-dict-') || key.startsWith('nocm-dict-')) &&
                key !== DICT_CACHE
            ))
            .map(key => caches.delete(key))
    );
}

async function checkDictionaryCacheRefresh() {
    const cache = await caches.open(DICT_CACHE);
    const cachedEntries = await Promise.all(DICT_URLS.map(async url => ({
        url,
        response: await cache.match(url),
    })));

    if (cachedEntries.some(entry => !entry.response)) return;

    const refreshResults = await Promise.all(cachedEntries.map(async ({ url, response: cachedResp }) => {
        const condHeaders = {};
        const etag = cachedResp.headers.get('ETag');
        const lastMod = cachedResp.headers.get('Last-Modified');
        if (etag) condHeaders['If-None-Match'] = etag;
        if (lastMod) condHeaders['If-Modified-Since'] = lastMod;

        const netResp = await fetch(url, { cache: 'no-cache', headers: condHeaders });

        if (netResp.status === 304) return { url, changed: false };

        if (!netResp.ok) {
            throw new Error(`${url} 更新检查返回 ${netResp.status}`);
        }

        if (sameCachedPayload(cachedResp, netResp)) {
            return { url, changed: false };
        }

        return {
            url,
            changed: true,
            response: netResp,
        };
    }));

    const changedResults = refreshResults.filter(result => result.changed);
    if (!changedResults.length) return;

    const nextResponses = new Map();
    refreshResults.forEach(result => {
        nextResponses.set(result.url, result.changed ? result.response : cachedEntries.find(entry => entry.url === result.url).response);
    });

    const [nextBase, nextExtra] = await Promise.all(
        DICT_URLS.map(url => streamAndParseGz(nextResponses.get(url).clone()))
    );

    if (!Array.isArray(nextBase) || !Array.isArray(nextExtra) || nextBase.length !== nextExtra.length) {
        console.warn('字典更新已检测到，但 base/extra 条目数不一致，本次跳过回写。');
        return;
    }

    await Promise.all(changedResults.map(result => cache.put(result.url, result.response)));
    if (!_updatedFiles) {
        _updatedFiles = true;
        showUpdateNotification();
    }
}

async function fetchGzJson(url, progressCallback) {
    const fetchFromNetwork = async () => {
        const resp = await fetch(url, { cache: 'no-cache' });
        if (!resp.ok) throw new Error(`請求${url}失敗`);

        const respForCache = ('caches' in window) ? resp.clone() : null;
        const data = await streamAndParseGz(resp, progressCallback);

        if (respForCache) {
            const cache = await caches.open(DICT_CACHE);
            await cache.put(url, respForCache);
        }
        return { data, fromCache: false };
    };

    if (!('caches' in window)) return fetchFromNetwork();

    const cache = await caches.open(DICT_CACHE);
    const cachedResp = await cache.match(url);

    if (cachedResp) {
        const data = await streamAndParseGz(cachedResp.clone(), progressCallback);
        return { data, fromCache: true };
    }

    return fetchFromNetwork();
}

async function copyText(text, btnElement) {
    try {
        await navigator.clipboard.writeText(text);
        const originalText = btnElement.innerText;
        btnElement.innerText = '✔️';
        setTimeout(() => btnElement.innerText = originalText, 1500);
    } catch (err) {
        console.error('複製失敗:', err);
    }
}

function hideUpdateNotification() {
    const bar = document.getElementById('updateBar');
    if (bar) bar.style.display = 'none';
}

function handleResultsClick(event) {
    if (!(event.target instanceof Element)) return;

    const copyBtn = event.target.closest('.copy-btn');
    if (copyBtn) {
        copyText(copyBtn.dataset.copyText || '', copyBtn);
        return;
    }

    const domainBtn = event.target.closest('.xiesheng-tag');
    if (domainBtn) {
        searchByDomain(domainBtn.dataset.domain || '');
        return;
    }

    const exportBtn = event.target.closest('.export-btn');
    if (exportBtn) {
        exportCard(exportBtn.closest('.result-card'));
    }
}

function setupEventListeners() {
    document.getElementById('searchInput')?.addEventListener('keydown', handleEnter);
    document.getElementById('searchBtn')?.addEventListener('click', () => {
        const btn = document.getElementById('searchBtn');
        if (btn?.dataset.action === 'retry') {
            initDictionary();
            return;
        }
        searchDict();
    });
    document.querySelectorAll('input[name="searchMode"]').forEach(input => {
        input.addEventListener('change', () => searchDict());
    });
    document.getElementById('reloadBtn')?.addEventListener('click', () => location.reload());
    document.getElementById('dismissUpdateBtn')?.addEventListener('click', hideUpdateNotification);
    document.getElementById('results')?.addEventListener('click', handleResultsClick);
}

function sanitizeFilename(name) {
    if (!name) return '';
    return name.replace(/[<>:"/\\|?*\x00-\x1F]/g, '_').trim();
}

async function exportCard(card) {
    if (!card) return;
    const btn = card.querySelector('.export-btn');
    const originalText = btn.innerText;

    if (!window.htmlToImage) {
        await new Promise((resolve, reject) => {
            const s = document.createElement('script');
            s.src = 'https://cdnjs.cloudflare.com/ajax/libs/html-to-image/1.11.11/html-to-image.min.js';
            s.onload = resolve; s.onerror = reject;
            document.head.appendChild(s);
        });
    }
    try {
        btn.innerText = '⏳ 處理中...';
        btn.disabled = true;
        card.classList.add('exporting');

        const bgColor = window.getComputedStyle(card).backgroundColor;
        const dataUrl = await htmlToImage.toPng(card, {
            pixelRatio: 2,
            backgroundColor: bgColor,
            style: { margin: '0' }
        });

        card.classList.remove('exporting');

        const zi = sanitizeFilename(card.dataset.char || 'result-card');
        const phonetic = sanitizeFilename(card.dataset.phonetic || '');
        const filename = phonetic ? `poc-${zi}-${phonetic}.png` : `poc-${zi}.png`;

        const link = document.createElement('a');
        link.download = filename;
        link.href = dataUrl;
        document.body.appendChild(link);
        link.click();
        document.body.removeChild(link);

        btn.innerText = '✔️ 已儲存';
    } catch (err) {
        console.error('汇出图片失败:', err);
        btn.innerText = '❌ 失敗';
        card.classList.remove('exporting');
    } finally {
        setTimeout(() => {
            btn.innerText = originalText;
            btn.disabled = false;
        }, 1500);
    }
}

async function initDictionary() {
    if (dictionaryLoading) return;
    dictionaryLoading = true;
    cleanupLegacyDictionaryCaches().catch(() => {});

    const input  = document.getElementById('searchInput');
    const btn    = document.getElementById('searchBtn');
    const status = document.getElementById('extraStatus');
    let baseFromCache = false;

    try {
        btn.dataset.action = 'loading';
        btn.disabled = true;
        btn.textContent = '加載中...';
        const { data: baseData, fromCache } = await fetchGzJson('base.json.gz', (loaded, total) => {
            const pct = total ? Math.round((loaded / total) * 100) + '%' : (loaded / 1024 / 1024).toFixed(2) + ' MB';
            input.placeholder = `加載基礎字庫... ${pct}`;
        });
        baseFromCache = fromCache;
        dictData = baseData;
        buildBaseIndexes();
        input.placeholder = `已加載${dictData.length}條目—請輸入關鍵字`;
        input.disabled = false;
        btn.disabled = false;
        btn.dataset.action = 'search';
        btn.textContent = '查詢';

        loadFromUrl();
    } catch (error) {
        console.error('基礎字典加載錯誤:', error);
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
        const { data: extraData, fromCache: extraFromCache } = await fetchGzJson('extra.json.gz', (loaded, total) => {
            const pct = total ? Math.round((loaded / total) * 100) + '%' : (loaded / 1024 / 1024).toFixed(2) + ' MB';
            status.innerText = `獲取釋義與注釋... ${pct}`;
        });

        if (baseFromCache || extraFromCache) scheduleDictionaryCacheRefresh();
        if (dictData.length === extraData.length) {
            extraData.forEach((ext, i) => { Object.assign(dictData[i], ext); });

            meaningIndex.clear();
            dictData.forEach((item, i) => {
                const addTextToIndex = (text) => {
                    if (!text) return;
                    for (const char of text) {
                        if (!meaningIndex.has(char)) meaningIndex.set(char, new Set());
                        meaningIndex.get(char).add(i);
                    }
                };
                item.d?.[1]?.forEach(m => addTextToIndex(m));
                addTextToIndex(item.e);
                addTextToIndex(item.n);
            });

            status.innerText = '擴展數據與索引加載完成！';
            const searchBtn = document.getElementById('searchBtn');
            searchBtn.dataset.action = 'search';
            searchBtn.textContent = '查詢';
        } else {
            console.error('資料長度不匹配，跳過擴展資料合併');
            status.innerText = '擴展數據異常，僅提供基礎查詢';
            status.style.color = 'var(--vermilion)';
        }
        setTimeout(() => { status.style.display = 'none'; }, 2000);

        if (input.value.trim()) searchDict(false);
    } catch (error) {
        console.error('擴充字典加載錯誤:', error);
        status.innerText = '釋義加載失敗，目前僅提供基礎查詢';
        status.style.color = 'var(--vermilion)';
    }

    dictionaryLoading = false;
}

// 啟動
setupEventListeners();
initDictionary();

function loadFromUrl() {
    const params = new URLSearchParams(window.location.search);
    const q = params.get('q');
    const mode = params.get('mode');
    pageSize = normalizePageSize(params.get('size') || pageSize);
    pendingUrlPage = Math.max(1, parseInt(params.get('page'), 10) || 1);

    if (mode && SEARCH_MODES.has(mode)) {
        const radio = document.querySelector(`input[name="searchMode"][value="${mode}"]`);
        if (radio) radio.checked = true;
    }
    const input = document.getElementById('searchInput');
    if (q) {
        input.value = q;
        searchDict(false);
    } else {
        input.value = '';
        currentQuery = '';
        currentResults = [];
        document.getElementById('results').replaceChildren();
    }
}

window.addEventListener('popstate', loadFromUrl);

function handleEnter(e) {
    if (e.key === 'Enter' && !e.isComposing) searchDict();
}

function searchByDomain(domain) {
    document.getElementById('searchInput').value = domain;
    document.querySelector('input[name="searchMode"][value="exact"]').checked = true;
    searchDict();
    window.scrollTo({ top: 0, behavior: 'smooth' });
}

const REGEX_MAX_LEN = 100;
const REDOS_PATTERN = /(\(.*\*.*\*|\(.*\+.*\+|\(\?=.*\*|\(\?!.*\*|(\.\*){3,}|\(\w+\+\)\+|\(\w+\*\)\+)/;

function safeRegex(str, flags = "") {
    if (str.length > REGEX_MAX_LEN) return { ok: false, msg: `正則表達式過長（${str.length}/${REGEX_MAX_LEN}字元）` };
    if (REDOS_PATTERN.test(str)) return { ok: false, msg: '正則表達式含有可能造成卡頓的模式，已拒絕執行' };
    try {
        const safeFlags = Array.from(new Set(flags.replace(/g/g, '').split(''))).join('');
        const r = new RegExp(str, safeFlags);
        return { ok: true, regex: r };
    } catch(e) {
        return { ok: false, msg: `正則語法錯誤：${e.message}` };
    }
}

function parseLinks(text) {
    if (!text) return text;
    const urlRegex = /((https?:\/\/)?(?:www\.)?[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}(?:\/[^\s（）()，。；：「」『』、]*)?)/g;

    return text.replace(urlRegex, function(url) {
        let cleanUrl = url.replace(/[.,?!)\]]+$/, '');
        let href = cleanUrl.startsWith('http') ? cleanUrl : 'https://' + cleanUrl;
        return `<a href="${href}" target="_blank" rel="noopener" class="annotation-link">${cleanUrl}</a>`;
    });
}

const DEFAULT_PAGE_SIZE = 50;
const PAGE_SIZE_OPTIONS = [20, 50, 100, 200];
const PAGE_SIZE_STORAGE_KEY = 'poc-page-size';
let currentResults = [];
let currentQuery = '';
let currentMode = 'default';
let currentPage = 1;
let pendingUrlPage = 1;
let pageSize = readStoredPageSize();

function normalizePageSize(value) {
    const parsed = parseInt(value, 10);
    return PAGE_SIZE_OPTIONS.includes(parsed) ? parsed : DEFAULT_PAGE_SIZE;
}

function readStoredPageSize() {
    try {
        return normalizePageSize(localStorage.getItem(PAGE_SIZE_STORAGE_KEY) || DEFAULT_PAGE_SIZE);
    } catch (_e) {
        return DEFAULT_PAGE_SIZE;
    }
}

function persistPageSize(value) {
    try { localStorage.setItem(PAGE_SIZE_STORAGE_KEY, String(value)); } catch (_e) {}
}

function updateSearchUrl() {
    const url = new URL(window.location);
    if (currentQuery) {
        url.searchParams.set('q', currentQuery);
        url.searchParams.set('mode', currentMode);
        if (currentPage > 1) url.searchParams.set('page', String(currentPage));
        else url.searchParams.delete('page');
        if (pageSize !== DEFAULT_PAGE_SIZE) url.searchParams.set('size', String(pageSize));
        else url.searchParams.delete('size');
    } else {
        url.searchParams.delete('q');
        url.searchParams.delete('mode');
        url.searchParams.delete('page');
        url.searchParams.delete('size');
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

function createPaginationBar(total, position = 'top') {
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

    const sizeLabel = document.createElement('label');
    sizeLabel.className = 'pagination-size';
    sizeLabel.textContent = '每頁';
    const select = document.createElement('select');
    select.setAttribute('aria-label', '每頁顯示條數');
    PAGE_SIZE_OPTIONS.forEach(size => {
        const option = document.createElement('option');
        option.value = String(size);
        option.textContent = `${size}條`;
        option.selected = size === pageSize;
        select.appendChild(option);
    });
    select.addEventListener('change', () => changePageSize(select.value));
    sizeLabel.appendChild(select);
    controls.appendChild(sizeLabel);

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
    card.style.animationDelay = (Math.min(idx, 10) * 0.06) + 's';
    card.dataset.char = item.z || '';
    card.dataset.phonetic = item.y || '';
    const safeZ = escapeHtml(item.z) || '？';
    const safeA = escapeHtml(item.a);
    const safeY = escapeHtml(item.y) || '—';
    const safeP = escapeHtml(item.p) || '—';
    const ziUrl = item.z ? encodeURIComponent(item.z) : '';

    let tags = '';
    if (item.a) tags += `<button class="tag xiesheng-tag" type="button" data-domain="${safeA}" title="點擊查找同諧聲域字" aria-label="諧聲域 ${safeA}，點擊查詢同域字">諧聲域: ${safeA}</button>`;
    if (item.s === 1) tags += `<span class="tag">見詩經韻</span>`;
    if (item.q === 1) tags += `<span class="tag">見戰國韻</span>`;

    let meaningsHtml = '<p class="empty-meaning">暫無釋義</p>';
    let meaningNote = '';
    if (item.d?.[1]?.length > 0) {
        meaningNote = item.d[0] ? `<span class="section-note">（${escapeHtml(item.d[0])}）</span>` : '';
        meaningsHtml = `<ol class="meanings-list">${item.d[1].map((m, i) =>
            `<li><span class="sense-num">${i + 1}</span><span>${parseLinks(escapeHtml(m))}</span></li>`
        ).join('')}</ol>`;
    }

    const safeN = item.n ? parseLinks(escapeHtml(item.n)) : '';

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
                ${tags ? `<div class="char-tags">${tags}</div>` : ''}
            </div>
            <button class="export-btn" type="button" title="導出爲圖片">📷 導出</button>
        </div>
        <div class="card-body">
            <div class="section">
                <div class="section-header">
                    <span class="section-label">釋義</span>${meaningNote}
                </div>
                ${meaningsHtml}
            </div>
            ${safeN ? `
            <div class="section-divider"></div>
            <div class="section">
                <div class="section-header">
                    <span class="section-label">注釋</span>
                </div>
                <div class="text-block annotation">${safeN}</div>
            </div>` : ''}
        </div>`;
    return card;
}

function renderPaginatedResults(updateUrl = true, shouldScroll = false) {
    const resultsDiv = document.getElementById('results');
    resultsDiv.replaceChildren();

    if (!currentQuery) return;

    if (currentResults.length === 0) {
        resultsDiv.setAttribute('aria-label', '查詢結果：無結果');
        resultsDiv.innerHTML = `<div class="no-result" role="status">未找到「${escapeHtml(currentQuery)}」的相關條目<br><span>可嘗試輸入單個漢字、拼音、諧聲域或使用正則</span></div>`;
        if (updateUrl) updateSearchUrl();
        return;
    }

    const totalPages = Math.max(1, Math.ceil(currentResults.length / pageSize));
    currentPage = Math.min(Math.max(1, currentPage), totalPages);
    const startIndex = (currentPage - 1) * pageSize;
    const displayResults = currentResults.slice(startIndex, startIndex + pageSize);
    resultsDiv.setAttribute('aria-label', `查詢結果：共 ${currentResults.length} 條，第 ${currentPage} 頁，共 ${totalPages} 頁`);

    resultsDiv.appendChild(createPaginationBar(currentResults.length, 'top'));

    const fragment = document.createDocumentFragment();
    displayResults.forEach((item, idx) => fragment.appendChild(renderResultCard(item, idx)));
    resultsDiv.appendChild(fragment);

    if (currentResults.length > pageSize) {
        resultsDiv.appendChild(createPaginationBar(currentResults.length, 'bottom'));
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
    persistPageSize(pageSize);
    currentPage = 1;
    renderPaginatedResults(true, true);
}

function searchDict(updateUrl = true) {
    const query = document.getElementById('searchInput').value.trim();
    const resultsDiv = document.getElementById('results');
    const regexErrorDiv = document.getElementById('regexError');
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
        return;
    }

    let isRegexMode = mode === 'regex' || (query.startsWith('/') && query.endsWith('/') && query.length > 2);
    let isExactMode = mode === 'exact' || (query.startsWith('"') && query.endsWith('"') && query.length > 2);
    let regex = null;
    let exactStr = "";

    if (isRegexMode) {
        let regexStr = query;
        let flags = "";
        const match = query.match(/^\/(.+)\/([gimsuy]*)$/);
        if (match) {
            regexStr = match[1];
            flags = match[2];
        }
        const result = safeRegex(regexStr, flags);
        if (!result.ok) {
            regexErrorDiv.textContent = result.msg;
            regexErrorDiv.style.display = 'block';
            return;
        }
        regex = result.regex;
    }

    if (isExactMode) {
        exactStr = (query.startsWith('"') && query.endsWith('"')) ? query.slice(1, -1) : query;
    }

    let results;

    if (mode === 'default' && !isRegexMode && !isExactMode) {
        let tokens = [];
        const parts = query.split(/\s+/);
        for (let p of parts) {
            if (!p) continue;
            const matches = p.match(/[\u4E00-\u9FFF\u3400-\u4DBF]+|[^ \u4E00-\u9FFF\u3400-\u4DBF]+/g);
            if (matches) {
                matches.forEach(match => {
                    if (/[\u4E00-\u9FFF\u3400-\u4DBF]/.test(match)) {
                        tokens.push(...match.split(''));
                    } else {
                        tokens.push(match);
                    }
                });
            }
        }
        if (tokens.length === 0) tokens = [query];
        const parsedTokens = tokens.map(t => ({ val: t, isUpper: /^[A-Z0-9]+$/.test(t), isHan: /[\u4E00-\u9FFF\u3400-\u4DBF]/.test(t) }));

        results = [];
        const seen = new Set();

        parsedTokens.forEach(token => {
            appendUnique(results, seen, zIndex.get(token.val));
            appendUnique(results, seen, pinyinIndex.get(token.val));
            appendUnique(results, seen, phoneticIndex.get(token.val));
            appendUnique(results, seen, domainExactIndex.get(token.val));

            if (token.isUpper) {
                for (const [domain, items] of domainExactIndex) {
                    if (domain.includes(token.val)) appendUnique(results, seen, items);
                }
            }

            if (!token.isHan) {
                dictData.forEach(item => {
                    if (!seen.has(item) && ((item.y && item.y.includes(token.val)) || (item.p && item.p.includes(token.val)))) {
                        seen.add(item);
                        results.push(item);
                    }
                });
            }
        });
    } else if (mode === 'meaning') {
        if (query.length > 0 && meaningIndex.has(query[0])) {
            const candidateIndices = Array.from(meaningIndex.get(query[0]));
            results = candidateIndices.map(i => dictData[i]).filter(item => {
                const inMeaning = item.d?.[1]?.some(m => m.includes(query));
                const inEtym    = item.e?.includes(query);
                const inNote    = item.n?.includes(query);
                return inMeaning || inEtym || inNote;
            });
        } else {
            results = [];
        }
    } else {
        if (isRegexMode && regex) {
            results = dictData.filter(item => regexMatches(regex, item.z, item.y, item.p, item.a));
        } else if (isExactMode) {
            const seen = new Set();
            results = [];
            appendUnique(results, seen, zIndex.get(exactStr));
            appendUnique(results, seen, pinyinIndex.get(exactStr));
            appendUnique(results, seen, phoneticIndex.get(exactStr));
            appendUnique(results, seen, domainExactIndex.get(exactStr));
        } else {
            results = [];
        }
    }

    currentResults = results;
    renderPaginatedResults(updateUrl);
}

if ('serviceWorker' in navigator) {
  window.addEventListener('load', () => {
    let hadController = !!navigator.serviceWorker.controller;
    navigator.serviceWorker.register('sw.js', { scope: './' }).catch(() => {});
    navigator.serviceWorker.addEventListener('controllerchange', () => {
        if (hadController) showUpdateNotification();
        hadController = true;
    });
  });
}
