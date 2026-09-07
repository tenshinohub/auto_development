# auto_develop_split

`auto_develop(1).py` (Automatic RAW Developer v29) を、処理の責務ごとに分割した版です。

## 実行

```bash
python3 auto_develop.py photos -o developed --device cuda --debug
```

元スクリプトと同じCLIを維持しています。

## 構成

```text
auto_develop_split/
├── auto_develop.py          # CLI / エントリーポイント
└── raw_develop/
    ├── __init__.py
    ├── constants.py         # 定数・dataclass
    ├── utils.py             # 汎用関数・マスク処理
    ├── color.py             # 色空間変換・輝度
    ├── raw_io.py            # rawpy / LibRaw
    ├── metadata.py          # ExifTool / EXIF
    ├── stats.py             # 画像統計
    ├── shooting.py          # 撮影条件解析
    ├── segmentation.py      # DeepLabV3
    ├── saliency.py          # saliency
    ├── subjects.py          # 被写体ランキング・領域マスク
    ├── scene.py             # シーン判定・シーンプロファイル
    ├── tone.py              # 露出・コントラスト・彩度・トーン
    ├── regions.py           # 被写体/背景の局所処理
    ├── filters.py           # ノイズ除去・シャープ
    ├── search.py            # 自動パラメータ探索
    ├── debug.py             # デバッグ出力
    ├── developer.py         # 全工程のオーケストレーション
    └── collection.py        # RAWファイル収集
```

## 方針

今回は「アルゴリズムを変更する」のではなく、元のv29の処理内容をできるだけそのままにして、
**何を担当しているコードなのかがファイル単位で分かること**を優先しています。

次の段階では、各モジュールを個別にテストできる構造へ整理し、
「自動現像そのものをどう設計し直すか」をこの分割版の上で検討できます。
