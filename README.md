# 说明
- `ctext - all - slice.7z`：自主收集的文言文繁体语料库，主要爬取自[中國哲學書電子化計劃](ctext.org/zh)、[全上古三代秦漢三國六朝文](zh.wikisource.org/wiki/全上古三代秦漢三國六朝文)、[漢川草廬](www.sidneyluo.net/index.php)等处。可以自主解压，也可以用`util.py`中`unzip('ctext - all - slice', 'ctext - all - slice.7z', format='7z')`解压
- `ctext - 副本 - 副本.7z`：`ctext - all - slice`的先秦部分，另附有《说文解字》（大徐本）
- `util.py`： 全局工具性函数
- `Word2vec.py`：用`gensim.models.Word2Vec()`进行简单nlp研究，附使用示例。预训练的Word2Vec模型：[访问码eah1](cloud.189.cn/t/eQZvIzBR3Ana)
- `ctext-scraper(2,3).py`：用于[中國哲學書電子化計劃](ctext.org/zh)的爬虫，按页面格式不同暂分为3份文件
- `sidneykuo-scraper.py`：用于[漢川草廬](www.sidneyluo.net/index.php)的爬虫，需要另按页面格式自主调整
- `wikipedia-scraper.py`：用于[维基百科](zh.wikipedia.org)的爬虫
- `wikisource-scraper(2).py`：用于[维基文库](zh.qikisource.org)爬虫
- `dict.py`：比照并找出`dir`中不属于`字.txt`的字，将按格式保存于`new_chars.txt`
- `search.py`：在`texts_dir`中查找字符，将打印所在的文件名及所在的行。所用的索引：`ctext - all - slice`对应[访问码：n9jg](cloud.189.cn/t/ZnArInRzQJNv)，`ctext - 副本 - 副本`对应[访问码：kj7x](cloud.189.cn/t/NziMJ3BJ7fQj)
- `data`：《廣韻》形聲考（Sliark再整理, 2024.4.25），在`search.py`中用于展示中古音韵地位
- 

## 其他语料库(github)
- [殆知阁古代文献](github.com/garychowcmu/daizhigev20)
- [近代汉语语料库数据集(蒋彦廷, 潘雨婷, 杨乐. 基于统计与词嵌入的近代汉语动量结构研究. 2020)](github.com/JiangYanting/Pre-modern_Chinese_corpus_dataset)
- 


