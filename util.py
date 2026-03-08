import os
from string import punctuation
import json, mmap, math, re
from time import time
from heapq import nlargest
import pickle
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from scipy.spatial.distance import cosine, squareform, cdist
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict, Counter
from sklearn.feature_extraction.text import TfidfTransformer
from tqdm import tqdm, trange
from functools import lru_cache, wraps
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from hashlib import md5
from concurrent.futures import ThreadPoolExecutor
import linecache
import matplotlib as mpl
import matplotlib.pyplot as plt
from typing import List, Dict
from scipy.sparse import csr_matrix

try:
    from numba import jit, prange, njit, vectorize

    HAS_NUMBA = True
except ImportError:
    HAS_NUMBA = False


    def jit(*args):
        def decorator(func):
            return func

        return decorator if args and callable(args[0]) else decorator


    prange = range
    njit = jit


    def vectorize():
        def decorator(func):
            return np.vectorize(func)

        return decorator


try:
    from datasketch import MinHash, MinHashLSH

    HAS_DATASKETCH = True
except ImportError:
    HAS_DATASKETCH = False


mpl.use('pgf')
mpl.rcParams.update({
    'font.family': 'serif',
    'font.size': 10.5,
    'pgf.rcfonts': False,
    'text.usetex': True,
    'pgf.preamble': '\n'.join([
        r"\usepackage{unicode-math}",
        r"\setmathfont{XITS Math}",
        r"\setmainfont{Times New Roman}",
        r"\usepackage{xeCJK}",
        r"\xeCJKsetup{CJKmath=true}",
        r"\setCJKmainfont{SimSun}", ])})

CP = set(
    '\x11\r\n#①②③④⑤⑥⑦⑧⑨⑩�□█●○﹦︰︰，。！？；：→↓"\'（）、…《》【】〈〉「」『』0123456789·​—“”䷀䷁䷂䷃䷄䷅䷆䷇䷈䷉䷊䷋䷌䷍䷎䷏䷐䷑䷒䷓䷔䷕䷖䷗䷘䷙䷚䷛䷜䷝䷞䷟䷠䷡䷢䷣䷤䷥䷦䷧䷨䷩䷪䷫䷬䷭䷮䷯䷰䷱䷲䷳䷴䷵䷶䷷䷸䷹䷺䷻䷼䷽䷾䷿'
    'ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzá  ꞎ°±²µ½ÄÈÉ×ÜàâäæçèéëìíîðòóöùúüýþÿāČčēěīńŊŋōőœśŠšūƆƉƐƔƩƱǎǐǒǔǚɑɔɕɖəɛɣɤɥɨɯʂʃʉʊʔʰʷʿˊ˞ˤ̈ΑΓΔΕΖΗΙΚΛΜΝΟΠΡΣΤΥΧίαβγδεικλμξοπρςχόВЖИЙЛМНОПСТУЦЧЯабеийкмноръةتقیᶑṓἀ‎–―‖‘’†•‧‬′※℃ⅡⅢⅥⅨⅪ∅−√∴∵∶≈≠≥⋯─■▲☆♀♬♭♯⟨⟩⸨⸩⿰〃〇〔〕〜おとのゆりろわァアイウエオカキクケゲコゴシスズソゾタダチッヅテトドナノパフプホマメャユラリルレロヰヱンヴㄏㄒㄞㄢㄦㄨㄩ\ue158\ue190\ue415\ue473兀﹋﹐﹑﹔﹣＃％＆＋－．／０１２３４５６７８９＝ＡＢＣＤＥＦＧＨＪＫＬＭＮＯＰＱＲＳＴＶＷＸＹＺ［］ａｂｃｄｅｆｇｈｉｌｍｎｏｐｒｓｔｕｖｗｘｙｚ～￥𝄇𝑝🜨')
P = CP.union(punctuation)

PP = set(
    '是此何乎或也兮于与於與歈俞歟耶即卽既旣莫乃其且然而若粤粵如所雖為爲維惟唯焉以已矣哉則者之彼非匪不否弗未勿亡厥伊靡無亡毋誰爰在暨曁斯兹玆噫嘻咨嗟虖歑吁殹')


@lru_cache(maxsize=128)
def f_hash(content: str) -> str:
    return md5(content.encode()).hexdigest()


def func_timer(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        st = time()
        result = func(*args, **kwargs)
        print(f'{func.__name__}()耗时{time() - st:.4f}秒')
        return result

    return wrapper


def profil(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        from pyinstrument import Profiler
        p = Profiler()
        p.start()
        r = func(*args, **kwargs)
        p.stop()
        p.print()
        return r

    return wrapper


def perf_check(func):
    from heartrate import trace, files
    trace(browser=True, files=files.all)

    @func_timer
    @profil
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


@njit(cache=True)
def _t2v_numba(text_freqs, all_chars_list, char_freqs_keys, char_freqs_values):
    result = np.zeros(len(all_chars_list), dtype=np.float64)
    for i, char in enumerate(all_chars_list):
        for j in range(len(char_freqs_keys)):
            if char_freqs_keys[j] == char:
                result[i] = char_freqs_values[j] * text_freqs
                break
    return result


def t2v(text: str, all_chars: List[str], char_freqs: Dict[str, float]) -> np.ndarray:
    freq = char_freqs[text]
    if HAS_NUMBA and len(text) > 100:  # 长文本 只用numba
        char_freqs_keys = np.array(list(freq.keys()))
        char_freqs_values = np.array(list(freq.values()))
        return _t2v_numba(1.0, all_chars, char_freqs_keys, char_freqs_values)
    else:
        return np.array([freq.get(char, 0) for char in all_chars])


@lru_cache(maxsize=1024)
def de_p(text):
    return ''.join(c for c in text if not c.isspace() and c not in P)


def read(dir):
    try:
        with open(dir, 'r', encoding='utf-8') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                return m.read().decode('utf-8')
    except (IOError, ValueError) as e:
        print(f'读取文件出错{dir}: {e}')
        return ''


def read_files(dir, lst=False):
    files = [f for f in os.listdir(dir) if f.endswith('.txt')]

    with ThreadPoolExecutor() as executor:
        if lst:
            file_paths = [os.path.join(dir, f) for f in files]
            return [de_p(content) for content in list(executor.map(read, file_paths))]
        else:
            results = {}

            def process_file(f):
                return f[:-4], de_p(read(os.path.join(dir, f)))

            future_to_file = {executor.submit(process_file, f): f for f in files}
            for future in tqdm(future_to_file, desc='读取中'):
                name, content = future.result()
                results[name] = content
            all_chars = set()
            for text in results.values():
                all_chars.update(text)

            return results, all_chars


def create_index(path, folder_path):
    idx = {}

    def process_file(file_name):
        with open(os.path.join(folder_path, file_name), 'r', encoding='utf-8') as file:
            for line_num, line in enumerate(file, 1):
                for c in set(line)-P:
                    if (c not in idx) and (c not in P):
                        idx[c] = []
                    idx[c].append((file_name, line_num))

    with ThreadPoolExecutor() as executor:
        executor.map(process_file, os.listdir(folder_path))
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(idx, f, ensure_ascii=False)
    return idx


def display_results(idx, char, texts_dir, df=None):
    results = idx.get(char, [])
    if not results:
        print('未找到结果')
        return
    results_by_file = {}
    for f, l in results:
        if f not in results_by_file:
            results_by_file[f] = []
        results_by_file[f].append(l)
    for f, line_nums in results_by_file.items():
        f_path = os.path.join(texts_dir, f)
        for l in line_nums:
            print(f'\n{f[:-4]}：{l}\n{linecache.getline(f_path, l).strip()}')
    print()
    if df is not None:
        for _, r in df.iterrows():
            print(f'{r.iloc[0]}{r.iloc[1]}：{"".join(map(str, r.iloc[list(range(5, 10)) + [4]]))}\t{r.iloc[10]}')
    print(f'找到{len(results)}个结果\n')


@lru_cache(maxsize=128)
def process_text(text_path, char_list):
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return set(de_p(text)) - set(char_list)


def load_char_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return tuple(f.read().strip())


def silhouette_analysis(X, max_clusters=50, random_state=2):
    from sklearn.metrics import silhouette_score
    score = []
    max_clusters = min(max_clusters, X.shape[0] - 1)  # 确保max_clusters有效

    def compute_score(n_clusters):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
        labels = kmeans.fit_predict(X)
        try:
            return n_clusters, silhouette_score(X, labels)
        except:
            return n_clusters, -1

    with ThreadPoolExecutor() as executor:
        results = list(executor.map(compute_score, range(2, max_clusters + 1)))

    for n_clusters, s_score in results:
        score.append(s_score)
        print(f"聚{n_clusters}类\t平均轮廓分数(silhouette score){s_score:.6f}")

    plt.figure(figsize=(10, 6))
    plt.plot(range(2, max_clusters + 1), score)
    plt.title('轮廓分析')
    plt.xlabel('聚类数')
    plt.ylabel('轮廓分数')
    plt.grid(True)
    plt.show()
    return score.index(max(score)) + 2


def get_top_chars(char_f, n=10):
    return dict(nlargest(n, char_f.items(), key=lambda x: x[1]))


def create_minhash(text, num_perm=256):
    m = MinHash(num_perm=num_perm)
    for char in text:
        m.update(char.encode('utf8'))
    return m


@lru_cache(maxsize=1024)
def char_freq(text):
    char_counts = Counter(text)
    return {c: count / sum(char_counts.values()) for c, count in char_counts.items()}


def check(target, texts, all_chars, num_neighbors=5):
    def find_nearest_neighbors(char_freqs, vec1):
        neighbors = []
        vec2 = np.zeros(len(all_chars))
        for n, n_freq in char_freqs.items():
            for i, char in enumerate(all_chars):
                vec2[i] = n_freq.get(char, 0)
            neighbors.append((n, cosine(vec1, vec2)))
        return sorted(neighbors, key=lambda x: x[1])

    char_freqs = {n: char_freq(texts[n]) for n in list(texts.keys()) if n != target}
    target_freq = char_freq(texts[target])
    vec1 = np.zeros(len(all_chars))
    for i, char in enumerate(all_chars):
        vec1[i] = target_freq.get(char, 0)

    result = find_nearest_neighbors(char_freqs, vec1)
    print(f"离{target}最近的{num_neighbors}个文本是：")

    for neighbor, d in result[:num_neighbors]:
        print(f"{neighbor}\t距离：{d * 100:.4f}%")

    if len(result) < num_neighbors:
        print(f'字数太少或太生僻，阈值内的文本不到{num_neighbors}个')
    print('————')

    max_distance = max(d for _, d in result)
    print(f'平均距离: {sum(d for _, d in result) / len(result) * 100:.4f}%')
    print(f'最远的文本: {next((n for n, d in result if d == max_distance), "")}, 距离: {max_distance * 100:.4f}%')
    print('————')


def clust(n_clusters, texts, model, X):
    res = {i + 1: [] for i in range(n_clusters)}
    for name, cluster in zip(texts.keys(), model.labels_):
        res[cluster + 1].append(name)
        print(f"{name}属于第 {cluster + 1} 类")
    result = pd.DataFrame([res[i + 1] for i in range(n_clusters)])
    result.index = [f"类别{i + 1}" for i in range(n_clusters)]
    result.columns = [f"文本{i + 1}" for i in range(1, result.shape[1] + 1)]
    print("\n聚类结果:")
    print(result)
    return result, visualise(n_clusters, texts, model, X, res, save=False)


def build_lsh(texts, dir, load=False, save=False, threshold=0.25, num_perm=256):
    if load:
        with open(dir, 'rb') as f:
            return pickle.load(f)

    lsh = MinHashLSH(threshold=threshold, num_perm=num_perm)
    for name, text in tqdm(texts.items(), desc='正在创建LSH索引'):
        lsh.insert(name, create_minhash(text, num_perm))

    if save:
        with open(dir, 'wb') as f:
            pickle.dump(lsh, f)

    return lsh


def organize_files_by_chapter(dir):
    chapters_by_title = ''

    for fn in os.listdir(dir):
        if fn.endswith('.txt'):
            title = fn[:-4]

            # 读取文件内容并添加到对应篇名的章节内容中
            with open(os.path.join(dir, fn), 'r', encoding='utf-8') as file:
                chapter_content = file.read()
                chapters_by_title += f"{chapter_content}\n\n"

    with open(os.path.join(dir, f"qqqqtt.txt"), 'w', encoding='utf-8') as file:
        file.write(chapters_by_title)


@func_timer
def create_idx(dir, INDEX_FILE, save=False, load=False):
    if load and os.path.exists(INDEX_FILE):
        print('正在加载索引...')
        with open(INDEX_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    if save or not os.path.exists(INDEX_FILE):
        print("正在创建索引...")
        return create_index(INDEX_FILE, dir)


def dbscan_clustering(distance_matrix, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    cluster_labels = dbscan.fit_predict(distance_matrix)

    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(distance_matrix.index[i])

    return clusters


def visualise(n_c, texts, model, X, res, title='字频向量聚类结果 (PCA降维)', figsize=(14, 12), dpi=500):
    d = model.transform(X)

    closest_texts = {}
    for i in range(n_c):
        cluster_t = [name for name, label in zip(texts.keys(), model.labels_) if label == i]
        closest_texts[i] = cluster_t[d[model.labels_ == i][:, i].argmin()]

    f = []
    for i in range(n_c):
        cluster_t = [texts[n] for n in res[i + 1]]
        top_chars = get_top_chars(char_freq(''.join(cluster_t)))
        f.append({"类别": f"{i + 1}",
                  "文本数": len(res[i + 1]),
                  "总字数": sum(len(t) for t in cluster_t),
                  "代表性文本": closest_texts[i],
                  "常见字符": ', '.join(f"{char}({freq:.2%})" for char, freq in top_chars.items())})
    f = pd.DataFrame(f)
    print("\n聚类特征:")
    print(f)

    plt.rc('font', family='SimHei')
    plt.rc('axes', unicode_minus=False)
    x = PCA(n_components=2).fit_transform(X)
    plt.figure(figsize=figsize, dpi=dpi)
    for i, color in enumerate(plt.cm.rainbow(np.linspace(0, 1, n_c))):
        cluster_mask = model.labels_ == i
        plt.scatter(x[cluster_mask, 0], x[cluster_mask, 1], s=3, c=[color], label=f'类别 {i + 1}')

    # 标注代表性文本
    for i, t in closest_texts.items():
        text_i = list(texts.keys()).index(t)
        plt.annotate(t, (x[text_i, 0], x[text_i, 1]), xytext=(5, 5), textcoords='offset points',
                     bbox=dict(boxstyle="round", fc="w", ec="0.5", alpha=0.5),
                     arrowprops=dict(arrowstyle="->", connectionstyle="arc3,rad=0"),
                     fontsize=9, alpha=0.5)

    plt.title('字频向量聚类结果 (PCA降维)')
    plt.xlabel('PC1')
    plt.ylabel('PC2')
    plt.tight_layout()
    plt.savefig(fname=title, dpi=dpi)
    plt.show()
    return f


def visualize_clusters_2d(d_M, clusters):
    tsne = TSNE(n_components=2, random_state=2)
    pos = tsne.fit_transform(d_M)

    plt.figure(figsize=(12, 8))
    colors = plt.cm.rainbow(np.linspace(0, 1, len(set(clusters.keys())) - 1))
    for label, color in zip(set(clusters.keys()) - {-1}, colors):
        mask = [i for i, text in enumerate(d_M.index) if text in clusters[label]]
        plt.scatter(pos[mask, 0], pos[mask, 1], c=[color], label=f'Cluster {label}')

    noise_mask = [i for i, text in enumerate(d_M.index) if text in clusters.get(-1, [])]
    plt.scatter(pos[noise_mask, 0], pos[noise_mask, 1], c='gray', label='Noise', alpha=0.5)

    for label, texts in clusters.items():
        if label != -1:  # Not noise
            representative_text = texts[np.argmin(d_M.loc[texts, texts].sum())]
            idx = list(d_M.index).index(representative_text)
            plt.annotate(representative_text, (pos[idx, 0], pos[idx, 1]))

    plt.title('DBSCAN Clustering Visualization (t-SNE)')
    plt.legend()
    plt.savefig('DBSCAN.png')
    plt.show()


def h_cluster(distance_matrix,
              method='ward'):  # 链接方法 ('single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward')
    return linkage(squareform(distance_matrix), method=method)


def plot_dendrogram(d_M, width=600, method='ward', title='交互式层次聚类树形图',
                    file_name='interactive_dendrogram.html'):
    import plotly.figure_factory as ff
    linkage_matrix = h_cluster(d_M, method)
    fig = ff.create_dendrogram(d_M,
                               orientation='left',
                               labels=d_M.index,
                               linkagefun=lambda x: linkage_matrix)

    fig.update_layout(title=title,
                      width=width,
                      height=len(d_M) * 14,
                      yaxis_title='篇目',
                      xaxis_title='距离',
                      font=dict(size=7))

    fig.write_html(file_name)
    print(f'交互式树状图已保存为{file_name}')


def spectral_clustering(M, n_clusters=5, affinity='precomputed', random_state=2):
    sc = SpectralClustering(n_clusters=n_clusters, affinity=affinity, random_state=random_state)
    return sc.fit_predict(1 - M)


def tsne_reduction(M, n_components=2, random_state=2):
    tsne = TSNE(n_components=n_components, random_state=random_state)
    return tsne.fit_transform(M.values)


def visualize_clusters(M, tsne_result, cluster_labels):
    text_names = M.M.columns.tolist()
    plt.figure(figsize=(12, 10))
    scatter = plt.scatter(tsne_result[:, 0], tsne_result[:, 1], alpha=0.5, s=5, c=cluster_labels, cmap='viridis')
    plt.colorbar(scatter)
    plt.title('t-SNE谱聚类可视化')
    plt.xlabel('t-SNE维1')
    plt.ylabel('t-SNE维2')

    summary = []
    for cls in np.unique(cluster_labels):
        cls_i = np.where(cluster_labels == cls)[0]
        cls_t = [text_names[i] for i in cls_i]

        p = tsne_result[cluster_labels == cls]
        rep_index = np.argmin(cdist([np.mean(p, axis=0)], p)[0])
        rep_text = text_names[np.where(cluster_labels == cls)[0][rep_index]]
        plt.annotate(rep_text, p[rep_index], xytext=(5, 5), textcoords='offset points', fontsize=8)
        summary.append({'类别': cls,
                        '文本数': len(cls_t),
                        '总字数': sum([M.txt_len[i] for i in cls_i]),
                        '代表性文本': cls_t[rep_index],
                        '所有文本': ', '.join(cls_t)})
    plt.tight_layout()
    plt.show()
    return pd.DataFrame(summary)


def count_vectorizer(texts, min_df=2, top_n_drop=20):
    """
    字频矩阵生成器/sklearn.CountVectorizer

    参数:
        texts: dict, {文本名: 文本内容}
        min_df: int, 最小文档频率。一个字如果出现的篇目数少于这个值，就会被过滤掉。
        top_n_drop: int, 剔除所有文本中绝对字频最高的前N个字，减少常用词干扰。
    返回:
        X: scipy.sparse.csr_matrix, 稀疏文档-字频矩阵
        feature_chars: list, 提取出的词汇表（所有非生僻特征字）
    """
    doc_freq = Counter()
    total_char_freq = Counter()

    # 每个字在多少篇文档中出现过
    for text in texts.values():
        doc_freq.update(set(text))  # 单篇文档去重，只看是否出现过
        total_char_freq.update(text)  # 绝对总字频
    # 获取 Top N 高频字集合
    top_chars = set()
    if top_n_drop > 0:
        top_chars = set(char for char, count in total_char_freq.most_common(top_n_drop))
        print(f"剔除全局Top {top_n_drop}高频字：{''.join(top_chars)}")

    # 去除罕见字、Top N 高频字
    chars = []
    for c, count in doc_freq.items():
        if count >= min_df:
            if c in top_chars: continue  # 过滤高频字
            chars.append(c)

    char_to_index = {char: idx for idx, char in enumerate(chars)}
    print(f"原始字库大小: {len(doc_freq)}，过滤后特征字库大小: {len(chars)}")

    rows, cols, data = [], [], []
    for doc_idx, text in enumerate(texts.values()):
        # 统计单篇文档内各个字的绝对出现次数
        char_counts = Counter(text)
        for c, count in char_counts.items():
            if c in char_to_index:  # 只保留特征字库里的字
                rows.append(doc_idx)
                cols.append(char_to_index[c])
                data.append(count)

    return csr_matrix((data, (rows, cols)), shape=(len(texts), len(chars))), chars


def tfidf_vectorizer(texts, min_df=2, top_n_drop=20):
    """
    基于count_vectorizer的过滤结果，计算 TF-IDF 权重
    """
    X_counts, chars = count_vectorizer(texts, min_df=min_df, top_n_drop=top_n_drop)
    transformer = TfidfTransformer()
    X_tfidf = transformer.fit_transform(X_counts)
    return X_tfidf, chars


def analyze_topo_features(M, title="Persistence Diagram"):
    from persim import plot_diagrams
    from ripser import ripser
    def persistent_homology(M, max_dimension=1, threshold=np.inf):
        return ripser((M + M.T) / 2, distance_matrix=True, maxdim=max_dimension, thresh=threshold)['dgms']

    diagrams = persistent_homology(M, max_dimension=1)
    plot_diagrams(diagrams, show=False)
    plt.title(title)
    plt.tight_layout()
    plt.show()
    return diagrams


def interpret_results(diagrams):
    summary = []
    for dim, diagram in enumerate(diagrams):
        if len(diagram) > 0:
            lifetimes = diagram[:, 1] - diagram[:, 0]
            summary.append(f"{dim}维：{np.sum(lifetimes > np.mean(lifetimes))}个重要特征")

    return "\n".join(summary)


def main_topo_analysis(M: np.array):
    print("Performing persistent homology analysis...")
    d = analyze_topo_features(M)

    print("\nInterpreting results:")
    interp = interpret_results(d)
    print(interp)
    return d


def unzip(path, zipf, zip_=False, format='zip'):
    if not zip_:
        os.makedirs(path, exist_ok=True)

    if format == 'zip':
        import zipfile
        if zip_:
            with zipfile.ZipFile(zipf, 'w', zipfile.ZIP_DEFLATED) as zf:
                for root, _, files in os.walk(path):
                    for file in files:
                        zf.write(os.path.join(root, file),
                                 os.path.relpath(os.path.join(root, file), path))
        else:
            with zipfile.ZipFile(zipf, 'r') as zf:
                zf.extractall(path)

    elif format == '7z':
        import py7zr
        if zip_:
            with py7zr.SevenZipFile(zipf, 'w') as z:
                z.writeall(path)
        else:
            with py7zr.SevenZipFile(zipf, 'r') as z:
                z.extractall(path)

    elif format == 'rar':
        if zip_:
            import shutil
            shutil.make_archive(zipf, 'rar', path)
        else:
            import rarfile
            with rarfile.RarFile(zipf) as rf:
                rf.extractall(path)

    else:
        raise ValueError(f"不支持的格式: {format}")

    print(f"{'压缩' if zip_ else '解压缩'}完成: {zipf}")


class HMMSegmenter:
    def __init__(self):
        self.states = ['B', 'M', 'E', 'S']
        self.MIN_FLOAT = -1e100
        self.start_p = {}  # 初始状态概率pi
        self.trans_p = {}  # 状态转移概率A
        self.emit_p = {}  # 发射概率B

    def _make_label(self, word):
        """将词转换为 BMES 标签序列"""
        if len(word) == 1:
            return ['S']
        return ['B'] + ['M'] * (len(word) - 2) + ['E']

    @func_timer
    def train(self, corpus_path):
        """
        训练 HMM 模型
        假设语料库是已经分好词的文本，词与词之间用空格隔开
        """
        print(f"开始训练 HMM 模型，读取语料: {corpus_path}")

        # 频数统计字典
        start_count = defaultdict(int)
        trans_count = {s: defaultdict(int) for s in self.states}
        emit_count = {s: defaultdict(int) for s in self.states}
        state_count = defaultdict(int)

        with open(corpus_path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, desc='统计频次'):
                line = line.strip()
                if not line:
                    continue

                # 假设预料格式为: "學 而 時 習 之" 或 "君子 坦 蕩蕩"
                words = line.split()
                char_seq = []
                label_seq = []

                for word in words:
                    char_seq.extend(list(word))
                    label_seq.extend(self._make_label(word))

                if not label_seq: continue

                # 统计初始状态
                start_count[label_seq[0]] += 1

                # 统计转移和发射状态
                for i in range(len(label_seq)):
                    state = label_seq[i]
                    char = char_seq[i]

                    state_count[state] += 1
                    emit_count[state][char] += 1

                    if i > 0:
                        prev_state = label_seq[i - 1]
                        trans_count[prev_state][state] += 1

        # 计算对数概率 (加 1 平滑处理)
        total_start = sum(start_count.values())
        for state in self.states:
            self.start_p[state] = math.log((start_count[state] + 1) / (total_start + 4))

            total_trans = sum(trans_count[state].values())
            self.trans_p[state] = {}
            for next_state in self.states:
                self.trans_p[state][next_state] = math.log((trans_count[state][next_state] + 1) / (total_trans + 4))

            total_emit = sum(emit_count[state].values())
            self.emit_p[state] = {char: math.log((count + 1) / (total_emit + len(emit_count[state])))
                                  for char, count in emit_count[state].items()}
            # 记录该状态下的默认发射概率 (用于处理未登录字 OOV)
            self.emit_p[state]['<UNK>'] = math.log(1.0 / (total_emit + len(emit_count[state])))

        print("HMM 模型训练完成！")

    def viterbi(self, text):
        """Viterbi 动态规划解码"""
        if not text: return []

        # V[t][state] 表示时刻 t 到达状态 state 的最大对数概率
        V = [{}]
        # path[state] 保存到达当前状态的最优路径
        path = {}

        # 1. 初始化 t=0
        char = text[0]
        for state in self.states:
            emit_prob = self.emit_p[state].get(char, self.emit_p[state]['<UNK>'])
            V[0][state] = self.start_p[state] + emit_prob
            path[state] = [state]

        # 2. 递推 t > 0
        for t in range(1, len(text)):
            V.append({})
            new_path = {}
            char = text[t]

            for state in self.states:
                emit_prob = self.emit_p[state].get(char, self.emit_p[state]['<UNK>'])

                # 寻找从上一时刻 y0 转移到当前 state 的最大概率
                (prob, best_prev_state) = max(
                    [(V[t - 1][y0] + self.trans_p[y0][state] + emit_prob, y0) for y0 in self.states]
                )

                V[t][state] = prob
                new_path[state] = path[best_prev_state] + [state]

            path = new_path

        # 3. 终止条件: 最后一个字通常是 E 或 S
        (prob, best_final_state) = max([(V[len(text) - 1][state], state) for state in ('E', 'S')])

        return path[best_final_state]

    def cut(self, text):
        """对输入文本进行分词"""
        if not text: return []

        labels = self.viterbi(text)
        words = []
        word = ""

        for i, char in enumerate(text):
            word += char
            if labels[i] in ('E', 'S'):
                words.append(word)
                word = ""

        # 处理异常截断
        if word: words.append(word)
        return words

    def save(self, filepath='hmm_model.json'):
        model_data = {
            'start_p': self.start_p,
            'trans_p': self.trans_p,
            'emit_p': self.emit_p
        }
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(model_data, f, ensure_ascii=False)
        print(f"模型已保存至 {filepath}")

    def load(self, filepath='hmm_model.json'):
        with open(filepath, 'r', encoding='utf-8') as f:
            model_data = json.load(f)
        self.start_p = model_data['start_p']
        self.trans_p = model_data['trans_p']
        self.emit_p = model_data['emit_p']
        print(f"模型已从 {filepath} 加载")


class CharNGram:
    def __init__(self, n=2):
        """
        初始化 N-Gram 模型
        :param n: N的维度，默认2为Bi-gram (基于前1个字预测当前字)
        """
        self.n = n
        self.vocab = set()
        self.ngram_counts = defaultdict(int)
        self.context_counts = defaultdict(int)
        self.vocab_size = 0

    @func_timer
    def train(self, texts):
        """
        训练 N-Gram 模型，统计词频
        :param texts: 字典 {文件名: 文本内容} 或 文本列表
        """
        text_list = texts.values() if isinstance(texts, dict) else texts

        for text in tqdm(text_list, desc=f'训练 {self.n}-Gram 模型'):
            clean_text = de_p(text)
            self.vocab.update(clean_text)

            # 统计 N-Gram 和 (N-1)-Gram 的频次
            for i in range(len(clean_text) - self.n + 1):
                ngram = clean_text[i:i + self.n]
                context = clean_text[i:i + self.n - 1]

                self.ngram_counts[ngram] += 1
                self.context_counts[context] += 1

        self.vocab_size = len(self.vocab)
        print(f"训练完成：字表大小 {self.vocab_size}，收集到 {len(self.ngram_counts)} 种不同的 {self.n}-Gram。")

    def perplexity(self, sentence):
        """
        计算单句的困惑度
        :param sentence: 输入的古汉语句子
        :return: 困惑度数值 (越低越通顺)
        """
        clean_sentence = de_p(sentence)
        N_len = len(clean_sentence) - self.n + 1

        if N_len <= 0:
            return float('inf')

        log_prob_sum = 0.0

        for i in range(N_len):
            ngram = clean_sentence[i:i + self.n]
            context = clean_sentence[i:i + self.n - 1]

            # 拉普拉斯平滑计算概率
            count_ngram = self.ngram_counts.get(ngram, 0)
            count_context = self.context_counts.get(context, 0)

            prob = (count_ngram + 1) / (count_context + self.vocab_size)
            log_prob_sum += math.log(prob)

        return math.exp(-log_prob_sum / N_len)


def split_sentences(text):
    text = re.sub(f'[{re.escape("".join(P))}\\s]+', '|', text)
    return [s for s in text.split('|') if s]


def entropy(neighbor_dict):
    total = sum(neighbor_dict.values())
    if total == 0:
        return 0.0

    entropy = 0.0
    for count in neighbor_dict.values():
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


class WordDiscoverer:
    def __init__(self, max_word_len=4, min_freq=5, min_pmi=3.0, min_entropy=1.2):
        self.max_word_len = max_word_len
        self.min_freq = min_freq
        self.min_pmi = min_pmi
        self.min_entropy = min_entropy

        # 统计频次
        self.word_counts = defaultdict(int)
        self.left_neighbors = defaultdict(lambda: defaultdict(int))
        self.right_neighbors = defaultdict(lambda: defaultdict(int))
        self.total_chars = 0

    def fit(self, texts):
        """扫描文本，统计 N-Gram 频次和左右邻字"""
        print("统计 N-Gram 频次...")
        for name, content in tqdm(texts.items(), desc='处理文档'):
            sentences = split_sentences(content)
            for s in sentences:
                length = len(s)
                self.total_chars += length

                for i in range(length):
                    for j in range(1, self.max_word_len + 1):
                        if i+j <= length:
                            w = s[i:i + j]
                            self.word_counts[w] += 1
                            if i > 0: self.left_neighbors[w][s[i - 1]] += 1  # 左邻字
                            if i+j < length: self.right_neighbors[w][s[i + j]] += 1  # 右邻字

    def _pmi(self, word):
        """计算内部凝固度(PMI)"""
        if len(word) < 2:
            return 0.0

        p_word = self.word_counts[word] / self.total_chars
        min_pmi = float('inf')

        for i in range(1, len(word)):
            left_part = word[:i]
            right_part = word[i:]
            p_left = self.word_counts[left_part] / self.total_chars
            p_right = self.word_counts[right_part] / self.total_chars

            pmi = math.log2(p_word / (p_left * p_right))
            min_pmi = min(min_pmi, pmi)

        return min_pmi

    def extract_words(self):
        """计算指标并提取最终词汇表"""
        print("第二遍扫描：计算凝固度与自由度...")
        extracted_words = {}

        for word, freq in tqdm(self.word_counts.items(), desc='筛选词汇'):
            if len(word) < 2 or freq < self.min_freq:
                continue

            pmi = self._pmi(word)
            if pmi < self.min_pmi:
                continue

            left_entropy = entropy(self.left_neighbors[word])
            right_entropy = entropy(self.right_neighbors[word])
            min_entropy = min(left_entropy, right_entropy)

            if min_entropy >= self.min_entropy:
                extracted_words[word] = {
                    'freq': freq,
                    'pmi': pmi,
                    'entropy': min_entropy
                }

        # 按频次和凝固度排序
        return dict(sorted(extracted_words.items(), key=lambda x: (x[1]['freq'], x[1]['pmi']), reverse=True))


class MaxMatchSegmenter:
    """基于提取出的词汇表，进行逆向最大匹配分词，生成 HMM 训练语料"""

    def __init__(self, vocab_set):
        self.vocab = vocab_set
        self.max_len = max([len(w) for w in vocab_set]) if vocab_set else 1

    def cut_sentence(self, sentence):
        result = []
        i = len(sentence)
        while i > 0:
            word_found = False
            for j in range(max(0, i - self.max_len), i):
                word = sentence[j:i]
                if word in self.vocab or len(word) == 1:
                    result.insert(0, word)
                    i = j
                    word_found = True
                    break
            if not word_found:  # Fallback (理论上走不到这里，因为单字在字典里兜底)
                result.insert(0, sentence[i - 1])
                i -= 1
        return result


if __name__ == "__main__":
    print("正在加载语料...")
    texts, _ = read_files(r'xtext\ctext - all - slice')
    model = CharNGram(n=2)
    model.train(texts)

    test_sentences = [
        "學而時習之，不亦說乎？",
        "關關雎鳩，在河之洲。",
        "之乎者也矣焉哉不亦。",
        "現代計算機科學技術。",
        "秦王掃六合虎視何雄哉"
    ]

    for s in test_sentences:
        pp = model.perplexity(s)
        print(f"[{s}] \t困惑度: {pp:.2f}")

    while True:
        user_input = input("\n请输入要测试的句子 (打 'q' 退出): ")
        if user_input.lower() == 'q':
            break
        print(f"困惑度: {model.perplexity(user_input):.2f}")
