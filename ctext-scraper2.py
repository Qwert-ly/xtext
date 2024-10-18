'''请先确保Selenium所用的chromedriver安装正确'''

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


Site = 'xijing-zaji'
Saveas = '西京雜記'


def getURL(url):
    try:
        r = requests.get(url)
        r.raise_for_status()  # 如果状态码不是200，抛出异常
        soup = BeautifulSoup(r.text, 'html.parser')
        div = soup.find_all('div', id='content2')
    except RequestException as e:
        print('网络请求错误：', e)
        return []
    except BeautifulSoup4.exceptions.BeautifulSoupParseError as e:
        print('HTML解析错误：', e)
        return []

    href = []
    for d in div:
        for a in d.find_all('a'):
            if a.has_attr('href') and a['href'].startswith(Site):
                href.append(a['href'][:-3])  # 去掉结尾的"/zh"
    return href


def emp(lis):
    lis = lis.splitlines()
    return lis[0] if len(lis) else ''


def getTXT(href_list, by章=False):
    def save_file(file_path, t):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, 'w', encoding='utf-8') as f:
            f.write(t)
    driver = webdriver.Chrome()
    content = ''
    t = ''
    for href in href_list:
        print(href)
        try:
            driver.get('http://ctext.org/plugins/textexport/#zh||ctp:' + href)
            WebDriverWait(driver, 4).until(EC.visibility_of_element_located((By.ID, "text")))
            soup = BeautifulSoup(driver.page_source, 'html.parser')
            t = soup.find_all('textarea', id='text')[0].get_text() or ''  # 需要授权时find_all将返回None，而None无法与str拼接
            content += t
            print(emp(t))
            if by章:
                save_file(f'C:\\Users\\13261\\Desktop\\xtext\\ctext - all - slice\\{Saveas}.{emp(t)}.txt', t)

        except TimeoutException:
            print(f'需授权，跳过')
            if not by章:
                save_file(
                    f'C:\\Users\\13261\\Desktop\\xtext\\ctext - all - slice\\{Saveas}.{emp(t)}.txt',
                    content)

    driver.quit()
    return content


href_list = getURL('https://ctext.org/' + Site + '/zh')
txt = getTXT(href_list, by章=True)

with open(f'C:\\Users\\13261\\Desktop\\xtext\\ctext - all - slice\\{Saveas}.txt', 'w', encoding='utf-8') as f:
    f.write(txt)
