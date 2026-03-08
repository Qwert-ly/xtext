from sklearn.decomposition import NMF
from util import *


@func_timer
def train_nmf(texts, n_topics=5, max_iter=200, random_state=2, min_df=2, top_n_drop=20):
    text_names = list(texts.keys())
    X_tfidf, chars = tfidf_vectorizer(texts, min_df=min_df, top_n_drop=top_n_drop, use_stopwords=True)

    print(f"正在训练 NMF 模型 (主题数={n_topics})...")
    nmf = NMF(
        n_components=n_topics,
        random_state=random_state,
        init='nndsvd',  # 适合稀疏的 TF-IDF 矩阵，能加速收敛且结果更稳定
        max_iter=max_iter
    )

    # 拟合模型，得到 W 矩阵（文档-主题权重）
    W = nmf.fit_transform(X_tfidf)

    # 归一化 W 矩阵，使每行（每篇文档）的主题权重之和为 1
    row_sums = W.sum(axis=1, keepdims=True)
    W_normalized = np.divide(W, row_sums, out=np.zeros_like(W), where=row_sums != 0)

    return nmf, X_tfidf, W_normalized, text_names, chars


def display_topics(nmf_model, feature_chars, n_top_words=30):
    for topic_idx, topic in enumerate(nmf_model.components_):
        # 提取权重最高的前 n_top_words 个字 (H 矩阵)
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_chars[i] for i in top_features_ind]
        print(f"主题 {topic_idx + 1}: {' '.join(top_features)}")


def analyze_soft_clustering(doc_topic_dist, text_names, n_topics):
    df_doc_topic = pd.DataFrame(doc_topic_dist, index=text_names, columns=[f"主题 {i + 1}" for i in range(n_topics)])

    print(df_doc_topic.head(10).style.format("{:.2%}").to_string())
    return df_doc_topic


def plot_topic_distribution(df_doc_topic, top_n_texts=15, figsize=(12, 6), random_state=None):
    sample_size = min(top_n_texts, len(df_doc_topic))
    plot_df = df_doc_topic.sample(n=sample_size, random_state=random_state)

    plt.rc('font', family='SimHei')
    plt.rc('axes', unicode_minus=False)

    ax = plot_df.plot(kind='bar', stacked=True, figsize=figsize, colormap='tab20')
    plt.title('软聚类主题分布 (NMF提取)', fontsize=16)
    plt.xlabel('篇目', fontsize=12)
    plt.ylabel('主题权重占比', fontsize=12)
    plt.legend(title='隐含主题', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('NMF_Soft_Clustering.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    texts, all_chars = read_files('ctext - all - slice')

    nmf_model, X_tfidf, doc_topic_dist, text_names, feature_chars = train_nmf(
        texts, n_topics=20, max_iter=200, min_df=2, top_n_drop=20
    )

    display_topics(nmf_model, feature_chars, n_top_words=20)
    df_result = analyze_soft_clustering(doc_topic_dist, text_names, 20)
    plot_topic_distribution(df_result, top_n_texts=20)
