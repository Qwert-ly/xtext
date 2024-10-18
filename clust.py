from util import *
from scipy.sparse import csr_matrix


def create_feature_matrix(texts, all_chars):
    char_to_i = {char: i for i, char in enumerate(all_chars)}
    rows, cols, data = [], [], []
    for i, text in enumerate(texts.values()):
        freq = char_freq(text)
        for c, f in freq.items():
            if c in char_to_i:
                rows.append(i)
                cols.append(char_to_i[c])
                data.append(f)
    return csr_matrix((data, (rows, cols)), shape=(len(texts), len(all_chars)))


texts, all_chars = read_files("ctext - all - slice")
X = create_feature_matrix(texts, all_chars)

# best_n_clusters = silhouette_analysis(X, random_state=2)
# print(f"最佳聚类数量: \t{best_n_clusters}")
best_n_clusters = 9
kmeans = KMeans(n_clusters=best_n_clusters, random_state=2)
kmeans.fit(X)

result, cls_f = clust(best_n_clusters, texts, kmeans, X)

check(input(), texts, all_chars, num_neighbors=15)  # 输入ctext - all - slice（即read_files()读的）内的文本名，实时查询最邻近的num_neighbors个文本
