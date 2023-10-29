# パッケージのインストール
from transformers import AutoModelForCausalLM, AutoTokenizer
import torch
from auto_gptq import exllama_set_max_input_length
import pandas as pd
# トークナイザーとモデルの準備
tokenizer = AutoTokenizer.from_pretrained(
    "TheBloke/Xwin-LM-13B-V0.1-GPTQ",
    use_fast=True
)

model = AutoModelForCausalLM.from_pretrained(
    "TheBloke/Xwin-LM-13B-V0.1-GPTQ",
    device_map="cuda",
    trust_remote_code=False,
    revision="main"
)
prompt = """### Instruction:
あなたは、メールを読むAIです。
何の予定なのか教えてください
また、予定の日時を教えてください

皆様、
11月1日に新商品開発会議を開催いたします。
会議の内容は、新商品の企画・開発についてです。
会議の日時は、11月1日（火）午前9時～10時と決定いたしました。
ご都合の良い方は、ご参加ください。
また、会議に出席するメンバーもご確認いただき、ご連絡ください。
ご返信をお待ちしております。
よろしくお願いいたします。

### Response:
日時 : 11月1日
イベント : 新商品開発会議

### Instruction:
あなたは、メールを読むAIです。
日時とイベントだけを抜き出してください
Instructionは省いてください
"""
df = pd.read_csv("/content/gmails.csv")
new_instruction = df["body"][0]
response="### Response:"

# 前の指示文と新しい指示文を結合
combined_instruction = prompt + new_instruction + response

# 指示文を生成
with torch.no_grad():
    token_ids = tokenizer.encode(combined_instruction, add_special_tokens=False, return_tensors="pt")
    output_ids = model.generate(
        token_ids.to(model.device),
        temperature=0.7,
        do_sample=True,
        top_p=0.95,
        top_k=40,
        max_new_tokens=256,
    )

# 結果を取得
output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1) :], skip_special_tokens=True)
output = output.split("\n\n")[0]
# 日時情報の抽出
date = output.split("日時 : ")[1]
event = output.split("イベント : ")[1]
print(output)
print("日時:", date)
print("イベント:", event)