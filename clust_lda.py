from sklearn.decomposition import LatentDirichletAllocation
from util import *


@func_timer
def train_lda(texts, n_topics=5, max_iter=20, random_state=2, min_df=2):
    text_names = list(texts.keys())
    X, chars = count_vectorizer(texts, min_df=min_df)

    print(f"正在训练 LDA 模型 (主题数={n_topics})...")
    lda = LatentDirichletAllocation(
        n_components=n_topics,
        max_iter=max_iter,
        learning_method='batch',  # 数据量不大时用 batch 效果更稳定
        random_state=random_state,
        n_jobs=-1  # 开启多核并行
    )

    return lda, X, lda.fit_transform(X), text_names, chars


def display_topics(lda, feature_chars, n_top_words=30):
    for topic_idx, topic in enumerate(lda.components_):
        # 提取概率最高的前 n_top_words 个字
        top_features_ind = topic.argsort()[:-n_top_words - 1:-1]
        top_features = [feature_chars[i] for i in top_features_ind]
        print(f"主题 {topic_idx + 1}: {' '.join(top_features)}")


def analyze_soft_clustering(doc_topic_dist, text_names, n_topics):
    """
    分析文档的软聚类分布
    """
    df_doc_topic = pd.DataFrame(doc_topic_dist, index=text_names, columns=[f"主题 {i + 1}" for i in range(n_topics)])

    print("文档-主题分布 (软聚类结果) - 前10篇")
    print(df_doc_topic.head(10).style.format("{:.2%}").to_string())  # 百分比

    return df_doc_topic


def plot_topic_distribution(df_doc_topic, top_n_texts=15, figsize=(12, 6), random_state=None):
    sample_size = min(top_n_texts, len(df_doc_topic))
    plot_df = df_doc_topic.sample(n=sample_size, random_state=random_state)

    plt.rc('font', family='SimHei')
    plt.rc('axes', unicode_minus=False)

    ax = plot_df.plot(kind='bar', stacked=True, figsize=figsize, colormap='tab20')
    plt.title('古籍文档的软聚类主题分布', fontsize=16)
    plt.xlabel('篇目', fontsize=12)
    plt.ylabel('主题概率占比', fontsize=12)
    plt.legend(title='隐含主题', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('LDA_Soft_Clustering_Random.png', dpi=300)
    plt.show()


if __name__ == "__main__":
    texts, all_chars = read_files('ctext - all - slice')

    # 2. 训练模型
    lda_model, X, doc_topic_dist, text_names, feature_chars = train_lda(texts, n_topics=20)

    # 3. 展示每个隐含主题的代表字（比如：战争主题、祭祀主题等）
    display_topics(lda_model, feature_chars, n_top_words=20)

    # 4. 分析具体某篇文本的主题概率（软聚类）
    df_result = analyze_soft_clustering(doc_topic_dist, text_names, 20)

    # 5. 可视化分布
    plot_topic_distribution(df_result, top_n_texts=20)

