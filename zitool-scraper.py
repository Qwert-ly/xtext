'''请先确保Selenium所用的chromedriver安装正确'''
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC
import pandas as pd
import re
import threading
import time
from tqdm import tqdm
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
import queue
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
pattern = r'\n(.)\n'

# 全局变量
processed_count = 0
lock = threading.Lock()
results_queue = queue.Queue()
save_interval = 50  # 每处理50个字符保存一次


def create_driver():
    """创建WebDriver实例"""
    options = webdriver.ChromeOptions()
    options.add_argument('--headless')
    options.add_argument('--disable-gpu')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    options.add_argument('--blink-settings=imagesEnabled=false')
    options.add_experimental_option("prefs", {"profile.managed_default_content_settings.images": 2})
    options.add_argument('--disable-extensions')
    options.add_argument('--disable-infobars')
    options.add_argument('--disable-notifications')
    options.add_argument('--disable-blink-features=AutomationControlled')
    options.add_argument('--disable-logging')
    options.add_argument('--log-level=3')
    return webdriver.Chrome(options=options)


def getTXT(driver, c, max_retries=5):
    """获取字符信息，带重试机制"""
    for attempt in range(max_retries):
        try:
            url = 'https://zi.tools/zi/' + c
            driver.get(url)
            tbl = WebDriverWait(driver, 7).until(
                EC.presence_of_all_elements_located((By.CSS_SELECTOR, 'table[data-v-0186dc5c]')))
            driver.execute_script("""
                        var elements = document.querySelectorAll('.mobile-row.ant-col-6');
                        for (var i = 0; i < elements.length; i++) {
                            elements[i].remove();
                        }
                    """)
            c_result = tbl[-1].text
            return re.sub(pattern, r'\1', c_result)
        except Exception as e:
            # 如果还未达到最大重试次数，则等一下并继续
            if attempt < max_retries - 1:
                logging.debug(f"字符 '{c}' 抓取失败，正在进行第 {attempt + 1} 次重试...")
                time.sleep(5)
                continue
            else:
                # 达到最大重试次数后仍然失败，返回空字符串
                logging.warning(f"字符 '{c}' 抓取失败，已重试 {max_retries} 次")
                return ''
    return ''


def worker_thread(char_data_chunk, thread_id, progress_bar):
    """工作线程函数"""
    global processed_count
    driver = create_driver()
    local_results = []

    try:
        for index, char in char_data_chunk:
            try:
                result = getTXT(driver, char)
                local_results.append((index, result))
                print(f"{char=}")
                print(result)

                with lock:
                    processed_count += 1
                    progress_bar.update(1)
                    progress_bar.set_description(f"线程{thread_id}: 处理字符 '{char}'")

                # 添加到结果队列
                results_queue.put((index, result))

            except Exception as e:
                logging.error(f"线程{thread_id}处理字符'{char}'时出错: {str(e)[:100]}")
                local_results.append((index, ''))
                results_queue.put((index, ''))

    finally:
        driver.quit()

    return local_results


def save_progress(df, filename='progress_save.xlsx'):
    """保存进度"""
    try:
        df.to_excel(filename, sheet_name='字典表', index=False)
        logging.info(f"进度已保存到 {filename}")
    except Exception as e:
        logging.error(f"保存进度失败: {str(e)}")


def update_dataframe_from_queue(df):
    """从队列中更新DataFrame"""
    updates = 0
    while not results_queue.empty():
        try:
            index, result = results_queue.get_nowait()
            df.at[index, '字統'] = result
            updates += 1
        except queue.Empty:
            break
    return updates


def main():
    global processed_count

    # 检查是否存在进度文件
    progress_file = 'progress_save.xlsx'
    if os.path.exists(progress_file):
        response = input(f"发现进度文件 {progress_file}，是否从上次进度继续？(y/n): ")
        if response.lower() == 'y':
            df = pd.read_excel(progress_file, sheet_name='字典表')
        else:
            df = pd.read_excel('上古汉语音节表.xlsx', sheet_name='字典表')
            df['字統'] = None
    else:
        df = pd.read_excel('上古汉语音节表.xlsx', sheet_name='字典表')
        df['字統'] = None

    # 找出需要处理的字符（未处理或处理失败的）
    mask = (df['字統'].isna()) | (df['字統'] == '') | (df['字統'].isnull())
    to_process = df[mask].copy()

    if len(to_process) == 0:
        print("所有字符都已处理完成！")
        return

    print(f"总共需要处理 {len(to_process)} 个字符")

    num_threads = 16
    print(f"使用 {num_threads} 个线程并发处理")

    # 准备数据块
    char_data = [(idx, row['字']) for idx, row in to_process.iterrows()]
    chunk_size = max(1, len(char_data) // num_threads)
    chunks = [char_data[i:i + chunk_size] for i in range(0, len(char_data), chunk_size)]

    # 创建进度条
    progress_bar = tqdm(total=len(to_process), desc="处理进度", unit="字")

    # 启动定时保存线程
    def auto_save():
        last_save_count = 0
        while processed_count < len(to_process):
            time.sleep(30)  # 每30秒检查一次
            if processed_count - last_save_count >= save_interval:
                update_count = update_dataframe_from_queue(df)
                if update_count > 0:
                    save_progress(df)
                    last_save_count = processed_count

    save_thread = threading.Thread(target=auto_save, daemon=True)
    save_thread.start()

    # 使用ThreadPoolExecutor管理线程
    start_time = time.time()
    with ThreadPoolExecutor(max_workers=num_threads) as executor:
        future_to_thread = {
            executor.submit(worker_thread, chunk, i + 1, progress_bar): i + 1
            for i, chunk in enumerate(chunks)
        }

        try:
            for future in as_completed(future_to_thread):
                thread_id = future_to_thread[future]
                try:
                    results = future.result()
                    logging.info(f"线程{thread_id}完成，处理了{len(results)}个字符")
                except Exception as e:
                    logging.error(f"线程{thread_id}执行出错: {str(e)}")

        except KeyboardInterrupt:
            logging.info("接收到中断信号，正在保存当前进度...")
            executor.shutdown(wait=False)

    progress_bar.close()

    # 最终更新DataFrame
    final_updates = update_dataframe_from_queue(df)
    logging.info(f"最终更新了 {final_updates} 个结果")

    # 保存最终结果
    df.to_excel('Done.xlsx', sheet_name='字典表', index=False)

    # 清理进度文件
    if os.path.exists(progress_file):
        os.remove(progress_file)

    end_time = time.time()
    total_time = end_time - start_time

    print(f"\n处理完成！")
    print(f"总耗时: {total_time:.2f}秒")
    print(f"平均每个字符: {total_time / len(to_process):.2f}秒")
    print(f"结果已保存到 Done.xlsx")


if __name__ == "__main__":
    main()
