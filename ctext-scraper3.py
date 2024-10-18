'''请先确保Selenium所用的chromedriver安装正确'''
from requests import RequestException
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
from selenium.common.exceptions import TimeoutException
import requests
import os
from urllib.parse import urlparse, parse_qs
from bs4 import BeautifulSoup
from time import sleep

Site = 'book-of-poetry'
Saveas = '诗经'


def getURL(url):
    try:
        r = requests.get(url)
        r.raise_for_status()  # 如果状态码不是200，抛出异常
        soup = BeautifulSoup(r.text, 'html.parser')
        div = soup.find_all('div', id='content3')
    except RequestException as e:
        print('网络请求错误：', e)
        return []

    href = []
    for d in div:
        for a in d.find_all('a'):
            if a.has_attr('href') and a['href'].startswith('text.pl'):
                href.append('n'+parse_qs(urlparse(a['href']).query).get('node', [''])[0])
    return href


def getTXT(href_list, by章=False):
    content = ''
    driver = webdriver.Chrome()
    for href in href_list:
        print(href)
        try:
            driver.get('http://ctext.org/plugins/textexport/#zh||ctp:' + href)
            WebDriverWait(driver, 20).until(EC.visibility_of_element_located((By.ID, "text")))
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            t = soup.find_all('textarea', id='text')[0].get_text() or ''  # 需要授权时find_all将返回None，而None无法与str拼接
            content += t
            print(t)
            if by章:
                file_path = f'C:\\Users\\13261\\Desktop\\xtext\\诗经\\{Saveas}.{t.splitlines()[0]}.txt'
                os.makedirs(os.path.dirname(file_path), exist_ok=True)
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(t)
        except TimeoutException:
            print(f'需授权，跳过')
        finally:
            sleep(1)

    driver.quit()
    return content


href_list = getURL('https://ctext.org/' + Site + '/zh')
txt = getTXT(href_list, by章=True)

# with open(Saveas, 'w', encoding='utf-8') as f:
#     f.write(txt)
