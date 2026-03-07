import json
import os
from openai import OpenAI
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor

CONFIG_FILE = "config.json"
DEFAULT_API_URL = "https://api.deepseek.com"
INPUT_FILE = "上古汉语音节表.json"
OUTPUT_FILE = "proofread_result.json"


class ConfigManager:
    @staticmethod
    def load_config():
        try:
            with open(CONFIG_FILE, 'r', encoding='utf-8') as f:
                return json.load(f) if os.path.exists(CONFIG_FILE) else {}
        except Exception as e:
            print(f"配置文件读取失败: {e}")
            return {}


class DeepSeekProofreader:
    def __init__(self):
        self.config = ConfigManager.load_config()
        self.api_key = self.config.get("api_key", os.environ.get("DEEPSEEK_API_KEY", ""))

        if not self.api_key:
            raise ValueError("未找到 API Key，请在 config.json 中配置 'api_key'")

        self.client = OpenAI(api_key=self.api_key, base_url=DEFAULT_API_URL)
        # 推荐使用 deepseek-chat (V3)，对于此类逻辑分析任务性价比和速度最优
        # 如果需要极强的推理能力，可以改为 deepseek-reasoner (R1)
        self.model = self.config.get("model", "deepseek-chat")

    def construct_prompt(self, entry):
        """构建校对用的 Prompt"""
        system_prompt = """你是一位资深的古汉语学家、训诂学专家和辞书编纂专家。你的任务是校对字典的json条目。
按以下三点检查：
1. “義項”是否有格式错误
2. “小韻表注釋”是否合理：
   - 如果注释是单个字（如“亮”的注释是“諒”），请分析是否属于合理的通假字/异体字/古今字
   - 如果注释是多个词组（如“標記、記住、識別”），请分析它是否错误反映了“義項”
注意：不需要指出信息本身是否错漏

返回json格式的反馈：
{
  "status": "通过" | "需人工复核" | "存在明显错误",
  "issues_found": ["具体问题1", "具体问题2..."] (没问题则为空列表),
  "suggested_fix": {
     // 如有建议的修改，给出完整的“釋義”或“小韻表注釋”结构；无需修改可不填
  }
}
务必确保输出是合法的json字符串"""

        # 为了节省 Token，只传入需要校对的关键字段
        check_data = {
            "字": entry.get("字"),
            "釋義": entry.get("釋義"),
            "注釋": entry.get("注釋"),
            "小韻表注釋": entry.get("小韻表注釋")
        }

        user_prompt = f"请校对以下字典条目：\n```json\n{json.dumps(check_data, ensure_ascii=False, indent=2)}\n```"

        return system_prompt, user_prompt

    def process_single_entry(self, task_data):
        """处理单个条目"""
        i, entry, total = task_data
        print(f"分发 [{i + 1}/{total}]: {entry.get('字', '未知')}")

        system_prompt, user_prompt = self.construct_prompt(entry)

        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                response_format={"type": "json_object"}
            )
            ai_feedback_str = response.choices[0].message.content
            entry["AI校对结果"] = json.loads(ai_feedback_str)
        except Exception as e:
            print(f"处理条目 '{entry.get('字')}' 时发生错误: {e}")
            entry["AI校对结果"] = {"error": str(e)}

        return entry

    def process_entries(self):
        if not os.path.exists(INPUT_FILE):
            print(f"未找到输入文件 {INPUT_FILE}")
            return

        with open(INPUT_FILE, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # 加载已有的进度，实现断点续传
        processed_data = []
        start_index = 0
        if os.path.exists(OUTPUT_FILE):
            with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
                processed_data = json.load(f)
                start_index = len(processed_data)
                print(f"发现已有进度，从第 {start_index + 1} 条开始继续处理...")

        total = len(data)

        # 准备待处理的任务队列 (携带索引和总数以便打印)
        tasks = [(i, data[i], total) for i in range(start_index, total)]
        max_workers = self.config.get("max_workers", 20)

        print(f"\n🚀 启动多线程处理，当前并发数: {max_workers}")

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            # executor.map 会并发执行，但按任务提交的顺序 yielding 结果
            results = executor.map(self.process_single_entry, tasks)

            # 主线程负责按顺序接收结果并安全写入文件
            for result_entry in tqdm(results, total=len(tasks), desc="校对进度"):
                processed_data.append(result_entry)

                # 实时保存，防止意外中断 (由于在主线程执行，不需要加锁)
                with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
                    json.dump(processed_data, f, ensure_ascii=False, indent=4)

        print("\n🎉 校对完成！结果已保存至", OUTPUT_FILE)


if __name__ == "__main__":
    try:
        proofreader = DeepSeekProofreader()
        proofreader.process_entries()
    except Exception as e:
        print(f"程序运行中断: {e}")
