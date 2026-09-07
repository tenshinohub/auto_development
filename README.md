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

## Evaluation foundation

The split now includes a lightweight human-evaluation system under `raw_develop/evaluation/`.
It uses only the Python standard library for the database and web UI.

### Candidate directory format

Prepare outputs as:

```text
candidates/
  IMG_0001/
    v29.jpg
    new_v1.jpg
    new_v2.jpg
    manual.jpg
  IMG_0002/
    v29.jpg
    new_v1.jpg
    new_v2.jpg
    manual.jpg
```

Optional metadata files:

```text
candidates/IMG_0001/image.json
candidates/IMG_0001/v29.json
```

`image.json` can contain `raw_path`, `metadata`, and `features`.
An algorithm JSON can contain `algorithm_version`, `params`, and `features`.

### Stage existing output directories

If each algorithm already writes to a separate directory:

```bash
python3 evaluate.py stage --output candidates \
  --algorithm v29=developed_v29 \
  --algorithm new_v1=developed_new_v1 \
  --algorithm new_v2=developed_new_v2
```

### Import and run the evaluator

```bash
python3 evaluate.py init-db --db evaluation.db
python3 evaluate.py import --db evaluation.db --root candidates
python3 evaluate.py serve --db evaluation.db --root candidates
```

Then open `http://127.0.0.1:8765/` in a browser.

The UI randomizes candidate order, hides algorithm names, requires a complete ranking,
and stores optional reason tags and notes in SQLite.

Export collected rankings:

```bash
python3 evaluate.py export-csv --db evaluation.db --output evaluation.csv
```

### Current database design

- `images`: one RAW/image identity and its metadata/features
- `candidates`: one development result plus parameters/features
- `evaluations`: one human evaluation event
- `rankings`: complete ranking for that event
- `reason_tags`: optional reasons for why a candidate was good/bad

This intentionally stores both the rendered JPEG and the development parameters/features,
so the same dataset can later be used for preference/ranking models.
