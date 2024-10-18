import requests
import sys
from bs4 import BeautifulSoup
from requests import RequestException
from time import sleep
from tqdm import tqdm
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException

Site1 = 'i/i04'
Site2 = 'i0408.htm'
Saveas = '東坡樂府'
def extract_text(element):
    return ' '.join(text for text in element.stripped_strings)


def getURL(url):
    try:
        r = requests.get(url)
        r.raise_for_status()  # 如果状态码不是200，抛出异常
        soup = BeautifulSoup(r.content.decode('utf-8'), 'html.parser')
        tb = soup.find_all('table', class_='tableb')
    except RequestException as e:
        print('网络请求错误：', e)
        return []
    # except BeautifulSoup.exceptions.BeautifulSoupParseError as e:
    #     print('HTML解析错误：', e)
    #     return []

    href = {}
    for t in tb:
        for a in t.find_all('a'):
            if a.has_attr('href'):
                href[a.contents[0]] = a['href']
    return href






def getTXT(l, folder='A', by卷=True):

    content = ''
    旧卷 = ''
    driver = webdriver.Chrome()
    for url in tqdm(l, desc='正在爬取'):
        try:
            driver.get(f'http://www.sidneyluo.net/{Site1}/{l[url]}')
            WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.CLASS_NAME, "main")))
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            try:
                新卷 = soup.find('h3').contents[0]
            except AttributeError:
                新卷 = soup.find('span', class_='style9').contents[0]
            if by卷:
                if (新卷 != 旧卷) and len(旧卷):
                    save(content.strip(), f'{sys.path[0]}\\{folder}\\{Saveas}.{旧卷}.txt')
                content = ''
            旧卷 = 新卷
            elements = []
            for tag in soup.find_all(['ul', 'p']):
                if 'style7' not in tag.get('class', []):
                    for unwanted in tag.find_all(['span', 'a']):
                        unwanted.extract()
                    if tag.name == 'ul':
                        for li in tag.find_all('li', recursive=False):
                            text = extract_text(li)
                            if text:
                                elements.append(text)
                    else:  # tag is a <p>
                        text = extract_text(tag)
                        if text:
                            elements.append(text)
        except RequestException as e:
            print('网络请求错误：', e)
            return []
        except TimeoutException as e:
            print('加载超时：', e)
            driver.get(f'http://www.sidneyluo.net/{Site1}/{l[url]}')
            WebDriverWait(driver, 30).until(EC.visibility_of_element_located((By.CLASS_NAME, "main")))
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            新卷 = soup.find('h3').contents[0]
            if (新卷 != 旧卷) and len(旧卷):
                save(content.strip(), f'{sys.path[0]}\\{folder}\\{Saveas}.{旧卷}.txt')
            content = ''
            旧卷 = 新卷
            elements = []
            for tag in soup.find_all(['ul', 'p']):
                if 'style7' not in tag.get('class', []):
                    for unwanted in tag.find_all(['span', 'a']):
                        unwanted.extract()
                    if tag.name == 'ul':
                        for li in tag.find_all('li', recursive=False):
                            text = extract_text(li)
                            if text:
                                elements.append(text)
                    else:  # tag is a <p>
                        text = extract_text(tag)
                        if text:
                            elements.append(text)
        for t in elements:
            content += f'{t.lstrip()}\n\n'
    if not by卷:
        save(content.strip(), f'{sys.path[0]}\\{folder}\\{Saveas}.txt')
    # save(content.strip(), f'{sys.path[0]}\\{folder}\\{Saveas}.{新卷}.txt')
    return content


def save(content, to):
    import os
    dir_path = os.path.dirname(to)
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)
    with open(to, 'w', encoding='utf-8') as f:
        f.write(content)


href_list = getURL(f'http://www.sidneyluo.net/{Site1}/{Site2}')
txt = getTXT(href_list, folder='诗经', by卷=True)
