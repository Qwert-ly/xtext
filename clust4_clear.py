from util import *
import json
import gzip


def calculate_distance_matrix(texts, all_chars):
    import torch
    @func_timer
    def sim_F(F):
        F_norm = F / torch.norm(F, dim=1, keepdim=True)  # 归一化
        d_M = torch.mm(F_norm, F_norm.t())
        print('完成')
        return 1 - d_M

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    char_freqs = {t: char_freq(content) for t, content in texts.items()}
    Fm = [t2v(t, all_chars, char_freqs) for t in tqdm(texts, desc='生成特征矩阵')]
    F = torch.tensor(np.array(Fm), dtype=torch.float32, device=device)
    print('计算距离矩阵...')
    d_M = np.nan_to_num(sim_F(F).cpu().numpy(), nan=0, posinf=0, neginf=0)   # 将结果转移到CPU并转换为NumPy数组
    return pd.DataFrame(d_M, index=texts.keys(), columns=texts.keys()), [len(t) for t in list(texts.values())]


def embed_len(d_M: pd.DataFrame, txt_len=None, remove=False):
    d_M_copy = d_M.copy()
    diag = np.diag(d_M_copy).astype(int)
    fill_value = np.zeros(d_M.shape[0]) if remove else txt_len

    if not remove and txt_len is None:
        raise ValueError('remove=False时，须提供txt_len。')

    embed_M = d_M_copy.copy()
    np.fill_diagonal(embed_M.values, fill_value)
    return embed_M, diag


class SimilarityMatrix:
    def __init__(self, M, file_hash, txt_len):
        self.M = M
        self.file_hash = file_hash
        self.txt_len = txt_len

    @staticmethod
    def compress(data):
        return gzip.compress(json.dumps(data).encode('utf-8'))

    @staticmethod
    def decompress(compressed):
        return json.loads(gzip.decompress(compressed).decode('utf-8'))

    @classmethod
    def create(cls, texts, all_chars):
        M, txt_len = calculate_distance_matrix(texts, all_chars)
        return cls(M, {t: f_hash(c) for t, c in texts.items()}, txt_len)

    @classmethod
    def load(cls, filename):
        def to_symmetric(upper_T):
            return upper_T + np.triu(upper_T, k=1).T

        with gzip.open(filename, 'rb') as f:
            data = cls.decompress(f.read())
        M = pd.DataFrame(to_symmetric(np.array(data['matrix'])), index=data['index'], columns=data['columns'])
        M, txt_len = embed_len(M, remove=True)
        return cls(M, data['file_hashes'], txt_len)

    def save(self, filename):
        embed_M = embed_len(self.M, self.txt_len)[0]
        data = {'matrix': np.triu(embed_M.values).tolist(),
                'index': self.M.index.tolist(),
                'columns': self.M.columns.tolist(),
                'file_hashes': self.file_hash}
        with gzip.open(filename, 'wb') as f:
            f.write(self.compress(data))
        self.M = embed_len(self.M, remove=True)[0]


@func_timer
def main_run(dir, M_dir, load=False, save=False):
    M = None
    if load:
        try:
            M = SimilarityMatrix.load(M_dir)
            print(f'从{M_dir}加载距离矩阵')
        except FileNotFoundError:
            print(f'{M_dir}不存在，创建中...')

    if M is None:
        texts, all_chars = read_files(dir)
        M = SimilarityMatrix.create(texts, all_chars)

    if save:
        M.save(M_dir)
        print(f'距离矩阵已保存至{M_dir}')

    return M


if __name__ == '__main__':
    M = main_run('ctext - 副本 - 副本', 'similarity_matrix.json', save=True)

    clusters = dbscan_clustering(M.M, eps=0.1, min_samples=5)
    visualize_clusters_2d(M.M, clusters)
    plot_dendrogram(M.M.clip(lower=0.), width=700, method='ward', title='', file_name='dendrogram_ward.html')
    diagrams = main_topo_analysis(M.M.to_numpy())
    cls_df = visualize_clusters(M, tsne_reduction(M.M), spectral_clustering(M.M, n_clusters=19))



    clusters = M.dbscan_clustering(eps=0.4, min_samples=2)

    summary_df = M.get_cluster_summary(clusters)
    print(summary_df)
    M.visualize_clusters_2d(clusters)
