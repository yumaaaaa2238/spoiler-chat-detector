# Spoiler Chat Detector

ゲーム配信におけるネタバレチャット検出システムのプロトタイプです。

## 概要
YouTube Liveのチャットをリアルタイムで取得し、機械学習モデルを用いてネタバレ判定を行います。
大学院での研究の一環として開発しました。

## 使用技術
- Python
- YouTube Data API
- PyTorch
- Hugging Face Transformers
- BERT

## 主な機能
- YouTubeライブチャット取得
- 機械学習モデルによる分類
- リアルタイム判定

## 実行方法
※注意：このプログラムを実行するためには学習済みのBERTモデル(.ptファイル)が必要ですが，容量の関係で公開していません

### 必要ライブラリのインストール
```bash
pip install -r requirements.txt
```

### APIキーの設定
`spoiler_detector.py` の41行目にある `API_KEY` を、自身のYouTube Data APIキーに置き換えてください。

```python
API_KEY = "YOUR_API_KEY"
```

### 実行
```bash
python spoiler_detector.py VIDEO_ID CSV_FILE
```
