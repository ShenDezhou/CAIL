import json
import os
import jieba

output_path = "../result/result.json"

if __name__ == "__main__":
    os.chdir('ydljzs')
    os.system(
        "sudo python3 main.py")
    os.chdir('..')
    os.chdir('ydljzt')
    os.system(
        "sudo python3 main.py")
    os.chdir('..')

    result = {}
    for item in json.load(open(r"ydljzs\output\submissions\pred.json", "r", encoding="utf8")):
        result["answer"] = item["answer"]
    for item in json.load(open(r"ydljzt\output\submissions\pred.json", "r", encoding="utf8")):
        result["sp"] = item["sp"]

    json.dump(result, open(output_path, "w", encoding="utf8"), indent=2, ensure_ascii=False, sort_keys=True)
