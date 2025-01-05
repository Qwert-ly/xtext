import os
from string import punctuation
import json
import mmap
from time import time
from heapq import nlargest
import pickle
import numpy as np
import pandas as pd
from datasketch import MinHash, MinHashLSH
from scipy.spatial.distance import cosine, squareform, cdist
from scipy.cluster.hierarchy import dendrogram, linkage
from collections import defaultdict, Counter
from tqdm import tqdm, trange
from sklearn.cluster import KMeans, DBSCAN, SpectralClustering
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from hashlib import md5
from concurrent.futures import ThreadPoolExecutor

import matplotlib as mpl
import matplotlib.pyplot as plt

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

PP = set('是此何乎或也兮于与於與歈俞歟耶即卽既旣莫乃其且然而若粤粵如所雖為爲維惟唯焉以已矣哉則者之彼非匪不否弗未勿亡厥伊靡無亡毋誰爰在暨曁斯兹玆噫嘻咨嗟虖歑吁殹')


def f_hash(content):
    return md5(content.encode()).hexdigest()


def func_timer(func):
    def wrapper(*args, **kwargs):
        st = time()
        result = func(*args, **kwargs)
        print(f'{func.__name__}()耗时{time() - st:.4f}秒')
        return result
    return wrapper


def profil(func):
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


def t2v(text, all_chars, char_freqs):
    freq = char_freqs[text]
    return np.array([freq.get(char, 0) for char in all_chars])


def de_p(text):
    return ''.join(c for c in text if not c.isspace() and c not in P)


def read(dir):
    with open(dir, 'r', encoding='utf-8') as f:
        with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
            return m.read().decode('utf-8')


def read_files(dir, lst=False):
    if lst:
        return [de_p(read(os.path.join(dir, f))) for f in tqdm(os.listdir(dir), desc='读取中') if f.endswith('.txt')]
    else:
        texts = {f[:-4]: de_p(read(os.path.join(dir, f))) for f in tqdm(os.listdir(dir), desc='读取中') if f.endswith('.txt')}
        return texts, set().union(*texts.values())


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


def process_text(text_path, char_list):
    with open(text_path, 'r', encoding='utf-8') as f:
        text = f.read()
    return set(de_p(text)) - char_list


def load_char_list(file_path):
    with open(file_path, 'r', encoding='utf-8') as f:
        return set(f.read().strip())


def silhouette_analysis(X, max_clusters=50, random_state=2):
    from sklearn.metrics import silhouette_score
    score = []
    for n_clusters in range(2, max_clusters + 1):
        kmeans = KMeans(n_clusters=n_clusters, random_state=random_state)
        scr = silhouette_score(X, kmeans.fit_predict(X))
        score.append(scr)
        print(f"聚{n_clusters}类\t平均轮廓分数(silhouette score){scr:.6f}")

    plt.plot(range(2, max_clusters + 1), score)
    plt.title('轮廓分析')
    plt.xlabel('聚类数')
    plt.ylabel('轮廓分数')
    plt.show()

    return score.index(max(score)) + 2


def get_top_chars(char_f, n=10):
    return dict(nlargest(n, char_f.items(), key=lambda x: x[1]))


def create_minhash(text, num_perm=256):
    m = MinHash(num_perm=num_perm)
    for char in text:
        m.update(char.encode('utf8'))
    return m


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

    print(f"离{target}最近的{num_neighbors}个文本是：")
    result = find_nearest_neighbors(char_freqs, vec1)

    max_distance = max(d for _, d in result)
    for neighbor, d in result[:num_neighbors]:
        print(f"{neighbor}\t距离：{d * 100:.4f}%")

    if len(result) < num_neighbors:
        print(f'字数太少或太生僻，阈值内的文本不到{num_neighbors}个')
    print('————')
    print(f'平均距离: {sum(d for _, d in result) / len(result) * 100:.4f}%')
    print(f'最远的文本: {next((n for n, d in result if d == max_distance), "")}, 距离: {max_distance * 100:.4f}%')
    print('————')


def clust(n_clusters, texts, model, X):
    res = {i+1: [] for i in range(n_clusters)}
    for name, cluster in zip(texts.keys(), model.labels_):
        res[cluster + 1].append(name)
        print(f"{name}属于第 {cluster+1} 类")
    result = pd.DataFrame([res[i+1] for i in range(n_clusters)])
    result.index = [f"类别{i+1}" for i in range(n_clusters)]
    result.columns = [f"文本{i+1}" for i in range(1, result.shape[1] + 1)]
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

    with open(os.path.join(dir, f"小雅.txt"), 'w', encoding='utf-8') as file:
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


# @perf_check
def display_results(idx, char, texts_dir, df=None):
    results = idx.get(char, [])
    if not results:
        print('未找到结果')
        return

    for file, line_num in results:
        with open(os.path.join(texts_dir, file), 'r', encoding='utf-8') as f:
            with mmap.mmap(f.fileno(), 0, access=mmap.ACCESS_READ) as m:
                for _ in range(line_num - 1):
                    m.readline()
                line = m.readline().decode('utf-8').strip()
        print(f"\n{file[:-4]}：{line_num}\n{line}")
    print()

    if df is not None:
        row = df[df.iloc[:, 10].str.contains(char, na=False)]
        for _, r in row.iterrows():
            print(f"{r.iloc[0]}{r.iloc[1]}：{''.join(map(str, r.iloc[5:10].tolist() + [r.iloc[4]]))}")
    print(f'找到{len(results)}个结果\n')


def dbscan_clustering(distance_matrix, eps, min_samples):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples, metric='precomputed')
    cluster_labels = dbscan.fit_predict(distance_matrix)

    clusters = {}
    for i, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(distance_matrix.index[i])

    return clusters


def visualise(n_c, texts, model, X, res, save=None, title='字频向量聚类结果 (PCA降维)', figsize=(14, 12), dpi=500):
    d = model.transform(X)

    closest_texts = {}
    for i in range(n_c):
        cluster_t = [name for name, label in zip(texts.keys(), model.labels_) if label == i]
        closest_texts[i] = cluster_t[d[model.labels_ == i][:, i].argmin()]

    f = []
    for i in range(n_c):
        cluster_t = [texts[n] for n in res[i+1]]
        top_chars = get_top_chars(char_freq(''.join(cluster_t)))
        f.append({"类别": f"{i+1}",
                  "文本数": len(res[i+1]),
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

    if save:
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
    plt.show()


def h_cluster(distance_matrix, method='ward'):  # 链接方法 ('single', 'complete', 'average', 'weighted', 'centroid', 'median', 'ward')
    return linkage(squareform(distance_matrix), method=method)


def plot_dendrogram(d_M, width=600, method='ward', title='交互式层次聚类树形图', file_name='interactive_dendrogram.html'):
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
    return sc.fit_predict(1-M)


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




if __name__ == '__main__':
    if not os.path.exists('ctext - all - slice'):
        unzip('ctext - all - slice', 'ctext - all - slice.7z', format='7z')
    if not os.path.exists('ctext - 副本 - 副本'):
        unzip('ctext - 副本 - 副本', 'ctext - 副本 - 副本.7z', format='7z')
