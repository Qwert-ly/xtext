from gensim.models import Word2Vec
from util import *


@func_timer
def build_w2v(path, save=False, text='', load=False):
    if load:
        w2v = Word2Vec.load(path)
        print(f"模型已从{path}加载")
        return w2v
    print('正在训练模型...')
    w2v = Word2Vec(read_files(text, lst=True),
                   vector_size=400, window=7, min_count=2, workers=32, epochs=50, sg=1, hs=1, sample=1e-4, max_vocab_size=None)

    if save:
        w2v.save(path)
        print(f"模型已保存到{path}")
    return w2v


def find_synonym(model, word, top_n=5):
    try:
        return model.wv.most_similar(word, topn=top_n)
    except KeyError:
        return None


def visualize_words(model, words, perplexity=10, n_iter=1000):
    word_vec = [model.wv[w] for w in words if w in model.wv]

    if not word_vec:
        print('模型中无匹配的词。')
        return

    word_vec = np.array(word_vec)
    tsne = TSNE(perplexity=perplexity, n_components=2, init='pca', max_iter=n_iter, random_state=2)
    embd = tsne.fit_transform(word_vec)

    plt.figure(figsize=(16, 9))
    for i, w in enumerate(words):
        if w in model.wv:
            plt.scatter(embd[i, 0], embd[i, 1])
            plt.annotate(w, (embd[i, 0], embd[i, 1]))
    plt.title('词向量可视化')
    plt.savefig('词向量可视化.png')
    plt.show()


def cluster_words(model, num_clusters=10, top_n=5000):
    w = model.wv.index_to_key[:top_n]
    kmeans = KMeans(n_clusters=num_clusters)
    kmeans.fit([model.wv[wd] for wd in w])

    word_cls = {}
    for word, cl in zip(w, kmeans.labels_):
        if cl not in word_cls:
            word_cls[cl] = []
        word_cls[cl].append(word)
    return word_cls


def sentence2vec(model, sentence):
    word_vec = [model.wv[c] for c in sentence if c in model.wv]
    if not word_vec:
        return None
    return np.mean(word_vec, axis=0)


def sentence_sim(model, sentence1, sentence2):
    vec1 = sentence2vec(model, sentence1)
    vec2 = sentence2vec(model, sentence2)
    if vec1 is None or vec2 is None:
        return None
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))


folder = 'all - slice'
model = build_w2v(f'w2v_model_{folder}.bin', text=f'ctext - {folder}', load=True)

while True:
    word = input('请输入要查询的词语 (打"q"退出): ')
    if word.lower() == 'q':
        break
    syn = find_synonym(model, word, top_n=15)
    for w, sim in syn or []:
        print(f'"{w}"\t相似度{sim:.6f}')
    else:
        print(f'{word}不在词汇表中') if not syn else None

# 词向量可视化(t-SNE)
visualize_words(model, ['州', '侯', '城', '郡', '縣', '晉', '楚', '午', '宋', '伐', '鄭', '敗', '攻', '衛', '葬',
                        '秦', '燕', '齊', '蔡', '韓', '趙', '魏', '奔', '圍', '襄', '帥', '討', '項', '葉', '觶',
                        '鵫', '韘', '瓞', '釳'])

# 语素聚类
for cl, w in cluster_words(model, num_clusters=50, top_n=10000).items():
    print(f"聚类{cl}: {', '.join(w[:25])}")

# 句子相似度
s1 = '關關雎鳩，在河之洲。窈窕淑女，君子好逑。'
s2 = '學而時習之，不亦說乎？有朋自遠方來，不亦樂乎？人不知而不慍，不亦君子乎？'
print(f'句1\t{s1}')
print(f'句2\t{s2}')
print(f'相似度\t{sentence_sim(model, s1, s2): .6f}')
