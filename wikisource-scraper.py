import requests
import sys
from bs4 import BeautifulSoup
from requests import RequestException


def extract_text(element):
    return ''.join(t for t in element.stripped_strings)


# class Ref:
#     def __init__(self, ref):
#         self.ref = ref
#         self.author = self.get_author()
#
#     def get_author(self):
#         return self.ref.contents[0]



class WikisourceItem:
    def __init__(self, url: str):
        self.url = url
        self.header, self.content, self.classification = self._fetch_page_data()
        self.lang_dict = self._get_lang_dict()
        self.text = self._extract_text()

    @staticmethod
    def _extract(soup):
        for link in soup.find_all('a'):
            link.replace_with(link.get_text())
        return soup

    def _fetch_page_data(self):
        try:
            r = requests.get(self.url)
            r.raise_for_status()
            soup = self._extract(BeautifulSoup(r.content.decode('utf-8'), 'html.parser'))

            HEADER = soup.find('header', class_='mw-body-header vector-page-titlebar')
            CONTENT = self._find_content(soup)
            CLASS = self._find_classification(soup)

            return HEADER, CONTENT, CLASS
        except RequestException as e:
            print(f'网络请求错误：{e}')
            return None, None, []

    @staticmethod
    def _find_content(soup):
        content = soup.find('div', class_='mw-content-ltr mw-parser-output')
        if content:
            for selector in ['div#headerContainer', 'div.licenseContainer', 'ul#plainSister', 'small']:
                for i in content.select(selector):
                    i.extract()
        return content

    @staticmethod
    def _find_classification(soup):
        f = soup.find('div', class_='mw-normal-catlinks')
        return [a.text for a in f.find_all('a', href=lambda href: href and href.startswith('/wiki'))] if f else []

    def _get_lang_dict(self):
        return {a['hreflang']: a['data-title']
            for a in self.header.find_all('a', href=lambda href: href and href.startswith('https://')) if 'hreflang' in a.attrs and 'data-title' in a.attrs}

    def _extract_text(self, delimiter='\n\n'):
        def process_span(span):
            if a := span.find('a'):
                span.string = a.text
                a.extract()

        elements = []
        for tag in self.content.find_all(['p', 'h2', 'h3', 'h4', 'h5', 'h6', 'li']):
            for span in tag.find_all('span'):
                process_span(span)
            for unwanted in tag.find_all(['sup', 'a']):
                unwanted.extract()
            if text := tag.get_text(strip=True):
                elements.append(text)

        return delimiter.join(elements)

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

    def trans_title(self, target_lang):
        return self.lang_dict.get(target_lang)

    def save_to_txt(self, content, path, mode='w', encoding='utf-8'):
        """Save content to a text file."""
        import os
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, mode, encoding=encoding) as f:
            f.write(content)

    @classmethod
    def get_wiki_item(cls, url):
        return cls(url)


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

def save_url(url, to):
    M = WikisourceItem.get_wiki_item(f'https://zh.wikisource.org/zh-hant/{url}')
    M.save_to_txt(M.text, f'{sys.path[0]}\\{to}.txt')


save_url('先唐文', 'wiki')
