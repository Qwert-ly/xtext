import requests
import sys
from bs4 import BeautifulSoup
from requests import RequestException


def extract_text(element):
    return ''.join(text for text in element.stripped_strings)


# class Ref:
#     def __init__(self, ref):
#         self.ref = ref
#         self.author = self.get_author()
#
#     def get_author(self):
#         return self.ref.contents[0]



class WikipediaItem:
    def __init__(self, url: str, header, content: BeautifulSoup, classification: list[str]):
        """
            维基百科条目类

            :param url : str 条目网址
            :param content : str 条目正文
            :param classification : list[str] 分类列表
            """
        self.url = url
        self.header = header
        self.content = content
        self.classification = classification
        # self.lang_dict = self.getLangDict()
        self.text = self.text_wrap()

    def trans_title(self, target_lang):
        """
        获取目标语言版本对应的条目名称

        :param target_lang : str 目标语言代码
        :return: target_title: str 目标语言条目名
        """
        return self.getLangDict()[target_lang]

    @classmethod
    def getWikiItem(cls, url):
        def find_content(soup):
            content = soup.find('div', class_='mw-content-ltr mw-parser-output')
            for i in content.find_all('div', id='headerContainer'):
                i.extract()
            for i in content.find_all('div', class_='licenseContainer'):
                i.extract()
            for i in content.find_all('ul', id='plainSister'):
                i.extract()
            for i in content.find_all('small'):
                i.extract()
            return content

        try:
            r = requests.get(url)
            r.raise_for_status()  # 如果状态码不是200，抛出异常
            soup = BeautifulSoup(r.content.decode('utf-8'), 'html.parser')

            HEADER = soup.find('header', class_='mw-body-header vector-page-titlebar')
            CONTENT = find_content(soup)
            CLASS = None#find_cls(soup)

            return cls(url, HEADER, CONTENT, CLASS)
        except RequestException as e:
            print('网络请求错误：', e)
            return []

    def getLangDict(self):
        """
        返回语言代码(str)为键，条目名(str)为值的字典

        :return: dict
        """
        langList = {}
        for a in self.header.find_all('a'):
            if a.has_attr('href') and a['href'].startswith('https://'):
                try:
                    langList[a['hreflang']] = a['data-title']
                except KeyError:
                    continue
        return langList

    def text_wrap(self, mode='txt', delimiter='\n\n', include_ref=False):
        """
        提取全文文本（不含模板、图片）

        :return: str
        """
        elements = ''

        if mode == 'txt' or 'TXT' or 'text':
            for t in self.content.find_all(['p', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
                for unwanted in t.find_all(['span', 'sup']):
                    unwanted.extract()
                if not include_ref:
                    for r in t.find_all('cite', class_='citation conference'):
                        r.extract()
                text = extract_text(t)
                if t and text:
                    elements += (text + delimiter)
        elif mode == 'md' or 'markdown':
            pass
        elif mode == 'html' or 'HTML' or 'htm':
            pass

        return elements

    def save_to_txt(self, content, to, mode=None, save_mode='w', encoding='utf-8'):
        """
        保存内容(str)到路径(to)

        :param content: str
        :param mode: str，可选。默认保存content(str)的内容
        :param save_mode: str，可选。即open()的打开模式，默认'w'
        :return: 无，但会保存/覆盖一个txt文件
        """
        import os
        dir_path = os.path.dirname(to)
        if mode:
            pass  # TODO
        else:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
            with open(to, save_mode, encoding=encoding) as f:
                f.write(content)


# @func_timer
# def main_run(url, dir, M_dir, load=False, save=False):
#     M = None
#     if load:
#         try:
#             M = WikipediaItem.getWikiItem(url)
#             print(f'从{M_dir}加载距离矩阵')
#         except FileNotFoundError:
#             print(f'{M_dir}不存在，创建中...')
#
#     if M is None:
#         texts, all_chars = read_files(dir)
#         M = SimilarityMatrix.create(texts, all_chars)
#
#     if save:
#         M.save(M_dir)
#         print(f'距离矩阵已保存至{M_dir}')
#
#     return M

M = WikipediaItem.getWikiItem('https://zh.wikisource.org/wiki/茶馆')
for a in M.content.find_all('a'):
    if a.has_attr('title') and a['href'].startswith('/wiki/%'):
        title = a["title"]
        m = WikipediaItem.getWikiItem(f'https://zh.wikisource.org/zh-hant/{title}')
        m.save_to_txt(m.text, f'{sys.path[0]}\\老舍\\{title.split("/")[0]}.{title.split("/")[1]}.txt')
