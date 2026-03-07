import pandas as pd
import json


def revert_shiyi(shiyi_dict):
    if not isinstance(shiyi_dict, dict):
        return str(shiyi_dict)

    result = []

    shuoming = shiyi_dict.get("說明", "").strip()
    if shuoming:
        result.append(shuoming)

    yixiang_list = shiyi_dict.get("義項", [])
    for i, yixiang in enumerate(yixiang_list, 1):
        result.append(f"{i} {yixiang}")

    return "\n".join(result)


def extract_ai_results(row):
    ai_res = row.get("AI校对结果", {})
    if not isinstance(ai_res, dict):
        return pd.Series(["", "", ""])

    status = ai_res.get("status", "")

    issues = ai_res.get("issues_found", [])
    if isinstance(issues, list) and issues:
        issues_str = "\n".join([f"• {iss}" for iss in issues])
    else:
        issues_str = str(issues) if issues else ""

    suggested_fix = ai_res.get("suggested_fix", {})
    if suggested_fix:
        fix_str = json.dumps(suggested_fix, ensure_ascii=False, indent=2)
    else:
        fix_str = ""

    return pd.Series([status, issues_str, fix_str])


with open('proofread_result.json', 'r', encoding='utf-8') as f:
    df = pd.DataFrame(json.load(f))

if '釋義' in df.columns:
    df['釋義'] = df['釋義'].apply(revert_shiyi)

if 'AI校对结果' in df.columns:
    df[['status', 'issues_found', 'suggested_fix']] = df.apply(extract_ai_results, axis=1)
    df = df.drop(columns=['AI校对结果'])

df.to_excel('AI_check.xlsx', index=False, engine='openpyxl')

