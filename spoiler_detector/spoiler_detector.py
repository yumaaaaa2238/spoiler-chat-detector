#===============================================================
# ネタバレ検出システムのプロトタイプ
# youtubeのapiからチャットデータを取得し，BERTで判定した後，表示する

# 機能
# - youtubeからチャットデータを取得する
# - チャットを機械学習モデルで推論する
# - ネタバレチャットは保存する（投稿時間も）
# - 非ネタバレチャットは表示する
#===============================================================

import time
import sys
import pandas as pd
import torch
from transformers import BertJapaneseTokenizer, BertForSequenceClassification
import statistics

from datetime import datetime, timedelta, timezone
from googleapiclient.discovery import build

# =========================
# コマンドライン引数
# =========================

if len(sys.argv) < 3:
    print("実行例：python spoiler_detector.py VIDEO_ID CSV_FILE")
    sys.exit()

# youtubeのリンクの"v="以降の文字列
VIDEO_ID = sys.argv[1]

# 出力ファイル名
CSV_FILE = sys.argv[2]
CSV_FILE_NAME = f"{CSV_FILE}.csv"

# =========================
# 設定
# =========================

API_KEY = "YOUR_API_KEY"

# JST
JST = timezone(timedelta(hours=9))

# GPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# モデル
tokenizer = BertJapaneseTokenizer.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking")
model = BertForSequenceClassification.from_pretrained("cl-tohoku/bert-base-japanese-whole-word-masking", num_labels = 2)
# model.load_state_dict(torch.load("bloodborne_fold1.pt", map_location=device))
model.load_state_dict(torch.load("8ban_fold1.pt", map_location=device))
model.to(device)
model.eval()

# ネタバレチャットのリスト
spoilerList = []

# =========================
# 推論関数
# =========================

def infer(text):
    # チャットをトークナイズ
    inputs = tokenizer(
        text,
        return_tensors = "pt",
        truncation = True,
        max_length = 96
    )

    # GPUへ移動
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # 推論
    with torch.no_grad():
        pred = torch.argmax(model(**inputs).logits, dim=1).item()

    print(pred)
    # 推論結果を返す 
    return pred

# =========================
# YouTube API 初期化
# =========================

youtube = build(
    "youtube",
    "v3",
    developerKey=API_KEY
)

# =========================
# 初回情報取得
# =========================

# youtubeAPIのvideosリソースをlistメソッドで取得
# partで取得するプロパティを指定
request = youtube.videos().list(
    part = "liveStreamingDetails,snippet",
    id = VIDEO_ID
)

# リクエストを送信
response = request.execute()

# response(JSON形式)からitemsキーの値を取得
# itemsがなければ空リストを返す
items = response.get("items", [])

if not items:
    raise Exception("動画が見つかりません")

video = items[0]

# itemsの先頭要素から各プロパティの辞書を取り出す
# なければ空の辞書を返す
details = video.get("liveStreamingDetails", {})
snippet = video.get("snippet", {})

# snippetから配信タイトルとチャンネル名を取り出す
title = snippet.get("title", "Unknown Title")
channel_name = snippet.get("channelTitle", "Unknown Channel")

# 配信開始時刻
actual_start_time = details.get("actualStartTime")
 # チャットID
live_chat_id = details.get("activeLiveChatId")

if live_chat_id is None:
    raise Exception("ライブチャットが見つかりません")

if actual_start_time is None:
    raise Exception("ライブ開始時刻を取得できません")

# Stringからdatetimeに変換
utc_start = datetime.strptime(
    actual_start_time,
    "%Y-%m-%dT%H:%M:%SZ"
)

# JSTに変換
start_datetime = utc_start.replace(
    tzinfo=timezone.utc
).astimezone(JST)

# =========================
# 情報表示
# =========================

print("=================================")
print(f"配信タイトル : {title}")
print(f"配信者       : {channel_name}")
print("チャット取得開始")
print("=================================")

# =========================
# メインループ
# =========================

next_page_token = None

try:
    while True:
        # liveChatIdで指定したライブチャット(いくつかのチャットのかたまり)を取得
        chat_request = youtube.liveChatMessages().list(
            liveChatId = live_chat_id,
            part = "snippet",
            pageToken = next_page_token
        )

        chat_response = chat_request.execute()
        chat_messages = chat_response.get("items", [])

        # リストからチャットを一つずつ取得
        for chat_message in chat_messages:
            chat_snippet = chat_message.get("snippet", {})

            # 通常チャットでなければスキップ
            if chat_snippet.get("type") != "textMessageEvent":
                continue    

            text = chat_message["snippet"]["textMessageDetails"]["messageText"]
            published_at = chat_message["snippet"]["publishedAt"]

            if text is None:
                continue
            
            # 推論
            if infer(text) == 0:
                # 非ネタバレなら表示
                print(text)
            else:
                print("ooooooooooooooooooooooo")
                # 投稿時間を計算
                chat_published_datetime = datetime.fromisoformat(
                    published_at
                )

                # ネタバレなら保存
                spoilerList.append({
                    "spoiler_chat": text,
                    "published_at": chat_published_datetime - start_datetime
                })

                df = pd.DataFrame(spoilerList)
                df.to_csv(CSV_FILE_NAME, index=False)

        # 次のチャットを取得するまでの時間
        interval = chat_response.get("pollingIntervalMillis", -1)
        # 新しいチャットを示すトークン
        next_page_token = chat_response.get("nextPageToken", None)

        if interval == -1:
            raise Exception("インターバル取得エラー")
        
        if next_page_token is None:
            raise Exception("nextPageTokenの取得に失敗しました")
        
        # 次のチャット取得まで待機
        time.sleep(interval / 1000)

except KeyboardInterrupt:
    print("=================================")
    print("手動停止")
    print("=================================")