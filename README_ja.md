# bcal: 有機半導体結晶のバンド構造・有効質量計算プログラム
[![Python](https://img.shields.io/badge/python-3.11%20or%20newer-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue.svg)](https://opensource.org/licenses/MIT)
[![PyPI](https://img.shields.io/pypi/v/yu-bcal)](https://pypi.org/project/yu-bcal/)
[![docs](https://img.shields.io/badge/docs-here-11419572)](https://matsui-lab-yamagata-univ.github.io/bcal/)

[English](README.md) / 日本語

# 概要
`bcal` は有機半導体のバンド構造と有効質量を計算するツールです。結晶構造（CIF）から量子化学計算の入力を生成し、DFT 計算を実行し、分子軌道行列を抽出して強束縛（tight-binding）ハミルトニアンを構築します。得られたバンド分散から HOMO・LUMO のバンド端を求め、それぞれの主有効質量と軸ベクトルを報告し、指定した高対称 k 経路に沿ってバンド図を描画します。

# 必要環境
* Python 3.11 以降
* NumPy
* Pandas
* SciPy
* Matplotlib
* yu-mcal>=0.7.1
* yu-tcal>=5.0.2

## 量子化学計算ツール
以下のいずれか 1 つ以上が必要です:
* Gaussian 09 または 16
* PySCF（macOS / Linux / WSL2（Windows Subsystem for Linux））
* GPU4PySCF（macOS / Linux / WSL2（Windows Subsystem for Linux））
* ORCA 6.1.0 以降

# 注意事項
* Gaussian を使用する場合、Gaussian 実行ファイルのパスを設定する必要があります。
* PySCF は macOS / Linux に対応しています。Windows ユーザーは WSL2 を使用してください。

# インストール
PyPI から、使用するバックエンドに応じた extra を選んでインストールしてください。

## Gaussian 09 または 16 を使用（PySCF なし）
```bash
pip install yu-bcal
```

## PySCF を使用（CPU のみ、macOS / Linux / WSL2）
```bash
pip install "yu-bcal[pyscf]"
```

## PySCF で GPU アクセラレーションを使用（macOS / Linux / WSL2）
### 1. インストール済みの CUDA Toolkit バージョンを確認
```bash
nvcc --version
```

### 2. CUDA Toolkit バージョンに対応する GPU extra をインストール
CUDA Toolkit が 13.x の場合:
```bash
pip install "yu-bcal[gpu4pyscf-cuda13]"
```
CUDA Toolkit が 12.x の場合:
```bash
pip install "yu-bcal[gpu4pyscf-cuda12]"
```
CUDA Toolkit が 11.x の場合:
```bash
pip install "yu-bcal[gpu4pyscf-cuda11]"
```

## ORCA 6.1.0 以降を使用
```bash
pip install "yu-bcal[orca]"
```

## インストールの確認

インストール後、以下を実行して確認できます:

```bash
bcal --help
```

# bcal 使用マニュアル

## 基本的な使い方

```bash
bcal <cif_filename> [options]
```

### 必須引数

- `file`: CIF ファイルのパス。

移動度テンソル計算ツールとは異なり、`bcal` は `p`/`n` の半導体タイプを引数に取り**ません**。1 回の実行で、HOMO バンド端（p 型輸送に対応）と LUMO バンド端（n 型輸送に対応）の**両方**を報告します。

### 基本的な例

```bash
# 結晶のバンド構造と有効質量を計算
bcal xxx.cif

# 同じ計算を PySCF バックエンドで実行
bcal xxx.cif --engine pyscf
```

## オプション

|ショート|ロング|説明|
|----|----|----|
|-h|--help|オプションの説明を表示します。|
|-M|--method METHOD/BASIS|"METHOD/BASIS" 形式で計算手法と基底関数系を指定します。（デフォルト: PBEPBE/6-31G(d,p)）|
|-c|--cpu N|使用する CPU 数を指定します。（デフォルト: 4）|
|-m|--mem N|メモリ量を GB 単位で指定します。（デフォルト: 10）|
|-o|--output DIR|結果の出力ディレクトリを指定します。（デフォルト: 入力 CIF ファイルのあるディレクトリ）|
|-r|--read|新たに計算を実行せず、既存のログファイルから読み込みます。|
||--engine ENGINE|量子化学計算のバックエンドを指定します: g16, g09, pyscf, gpu4pyscf, orca。（デフォルト: g16）|
||--resume|中断した計算を、最後の未完了ステップから再開します。|
||--num-mo N|フロンティア軌道の各側について 1 分子あたり保持する MO 数。MO の総数は 2 * N。（デフォルト: 15）|
||--band-path PATH|バンド図の高対称 k 経路を 1 文字ラベルの並びで指定します（G = Γ 点）。（デフォルト: XGYGZ）|
||--bse|Basis Set Exchange から基底関数系を取得します。（PySCF/gpu4pyscf のみ）|

### 計算設定

#### `-M, --method <method>`
量子化学計算で使用する DFT 手法と基底関数系を指定します。
- **デフォルト**: `PBEPBE/6-31G(d,p)`
- **例**: `bcal xxx.cif -M "B3LYP/6-31G(d,p)"`

> **注意:** `PBEPBE` は PBE 汎関数の Gaussian 表記です。`pyscf` / `gpu4pyscf` / `orca` エンジンはこの名前を受け付けないため、`bcal` は先頭の `PBEPBE` を自動的に `PBE` へ書き換え（例: `PBEPBE/6-31G(d,p)` → `PBE/6-31G(d,p)`）、警告を stderr に出力します。

#### `-c, --cpu <number>`
使用する CPU 数を指定します。
- **デフォルト**: `4`
- **例**: `bcal xxx.cif -c 8`

#### `-m, --mem <memory>`
メモリ量を GB 単位で指定します。
- **デフォルト**: `10`
- **例**: `bcal xxx.cif -m 16`

#### `--engine <engine>`
量子化学計算のバックエンドを指定します。
- **選択肢**: `g16`, `g09`, `pyscf`, `gpu4pyscf`, `orca`
- **デフォルト**: `g16`（Gaussian 16）
- **例**:
  - `bcal xxx.cif --engine g09`（Gaussian 09）
  - `bcal xxx.cif --engine pyscf`（PySCF、CPU。`pyscf` extra が必要）
  - `bcal xxx.cif --engine gpu4pyscf`（GPU アクセラレーション版 PySCF。`gpu4pyscf-cudaXX` extra が必要）
  - `bcal xxx.cif --engine orca`（ORCA。`orca` extra が必要）

##### ORCA の並列実行
ORCA は ORCA Python Interface（OPI）経由で実行されます。複数の CPU コア（`--cpu N`）を使用するには、OpenMPI がインストールされ、ORCA から参照可能になっている必要があります。
まず `mpirun` が利用可能かを確認します:
```bash
which mpirun
```
OpenMPI が既に `$PATH` および `$LD_LIBRARY_PATH` に含まれている場合（`apt install` 後の Linux/WSL では一般的）、通常は追加設定は不要です。そうでない場合は、`OPI_MPI` 環境変数で OpenMPI のベースディレクトリ（`bin/` と `lib/` を含むディレクトリ）を ORCA に指定します:
```bash
# ソースからビルド、またはモジュールシステム経由でインストールした場合
which mpirun
# 例: /opt/openmpi/bin/mpirun  ->  ベース: /opt/openmpi
export OPI_MPI=$(dirname $(dirname $(which mpirun)))

# apt でシステム全体にインストールした場合（Ubuntu/Debian）
export OPI_MPI=/usr/lib/x86_64-linux-gnu/openmpi
```
> **注意:** ORCA は特定バージョンの OpenMPI を必要とします。`apt` で入手できるバージョンは一致しない場合があります。並列実行に失敗する場合は、[ORCA のドキュメント](https://www.faccts.de/docs/orca/6.0/manual/)で指定されているバージョンの OpenMPI をソースからビルドしてください。

#### `--bse`
`-M, --method` で指定した基底関数系を、エンジン内蔵の定義ではなく [Basis Set Exchange](https://www.basissetexchange.org/) から解決します。値を取らないフラグで、**`pyscf` および `gpu4pyscf` エンジンでのみ有効**です（`pyscf` extra に含まれる `basis-set-exchange` パッケージを使用します）。`g16` / `g09` / `orca` では黙って無視されます。
- **デフォルト**: 無効
- **例**: `bcal xxx.cif --engine pyscf --bse`

### バンド構造の設定

#### `--num-mo <number>`
フロンティア軌道の各側（HOMO 側・LUMO 側）について 1 分子あたり保持する分子軌道（MO）の数です。1 分子あたり使用される MO の総数は `2 * num_mo` になります。値を大きくすると、強束縛モデルに含まれる軌道の窓が広がります。
- **デフォルト**: `15`
- **例**: `bcal xxx.cif --num-mo 20`

#### `--band-path <path>`
バンド図の高対称 k 経路を、1 文字ラベルの並びで指定します（`G` = Γ 点）。
- **デフォルト**: `XGYGZ`（X → Γ → Y → Γ → Z をたどる）
- **例**: `bcal xxx.cif --band-path GXSY`

利用可能な高対称点:

| ラベル | 逆格子の分率座標 |
|--------|------------------|
| `G`    | (0.0, 0.0, 0.0) — Γ |
| `X`    | (0.5, 0.0, 0.0) |
| `Y`    | (0.0, 0.5, 0.0) |
| `Z`    | (0.0, 0.0, 0.5) |
| `S`    | (0.5, 0.5, 0.0) |
| `T`    | (0.0, 0.5, 0.5) |
| `U`    | (0.5, 0.0, 0.5) |
| `R`    | (0.5, 0.5, 0.5) |

指定した経路はバンド図（`band.png`）にのみ影響します。バンド端と有効質量はブリルアンゾーン全体から探索されます。HOMO/LUMO のバンド端が指定経路上に無い場合、`bcal` は警告を stderr に出力するので、該当する k 点を `--band-path` に追加できます。

### 出力設定

#### `-o, --output <directory>`
結果の出力ディレクトリです。その中に CIF 名を冠した結晶ごとのサブディレクトリが作成されます。
- **デフォルト**: 入力 CIF ファイルが置かれているディレクトリ。
- **例**: `bcal xxx.cif -o ./results`

### 計算の制御

#### `-r, --read`
新たに DFT 計算を実行せず、既存のログファイルから結果を読み込みます。入力生成をスキップし、結晶の `logs/` ディレクトリに既にあるログを再利用します。
- **例**: `bcal xxx.cif -r`

#### `--resume`
中断した計算を再開します。完了済みのステップを再利用し、最後の未完了ステップから続行します（例: 正常終了済みの DFT 計算は再実行されません）。
- **例**: `bcal xxx.cif --resume`

## 実用的な使用例

### 基本的な計算
```bash
# デフォルト実行（Gaussian 16、PBEPBE/6-31G(d,p)）
bcal xxx.cif

# 8 CPU・16 GB メモリを使用
bcal xxx.cif -c 8 -m 16
```

### バックエンドの選択
```bash
# PySCF（CPU）
bcal xxx.cif --engine pyscf

# GPU アクセラレーション版 PySCF
bcal xxx.cif --engine gpu4pyscf

# ORCA を 8 CPU で使用
bcal xxx.cif --engine orca -c 8

# Basis Set Exchange の基底関数系を使う PySCF
bcal xxx.cif --engine pyscf --bse
```

### バンド計算の調整
```bash
# 異なる手法 / 基底関数系
bcal xxx.cif -M "B3LYP/6-311G(d,p)"

# 軌道の窓を広げる
bcal xxx.cif --num-mo 25

# バンド図のカスタム k 経路
bcal xxx.cif --band-path GXSYG
```

### 結果の再利用
```bash
# 既存のログファイルから読み込み（DFT を再実行しない）
bcal xxx.cif -r

# 中断した計算を再開
bcal xxx.cif --resume
```

## 出力

### 標準出力
計算結果は **stdout** に出力され、警告や診断メッセージはすべて **stderr** に出力されます（`WARNING:` を前置し、stderr が対話的な端末の場合は色付き）。そのため両者は独立してリダイレクトできます。

実行の最後に、`bcal` は各フロンティアバンドについて以下を stdout に表示します:
- **バンド端**: 高対称点ラベル（該当する場合）と逆格子の分率座標。
- 3 つの**主有効質量** `m1`, `m2`, `m3`（電子質量 `m_e` 単位）。`|m|` の昇順に並べ替えられ、それぞれデカルト逆空間における単位主軸ベクトルと対にして表示されます。

```
LUMO band edge: G  k=(+0.000, +0.000, +0.000)
  m1 = +0.834 m_e  v=(+0.998, +0.043, +0.000)
  m2 = +1.207 m_e  v=(-0.043, +0.998, +0.000)
  m3 = +3.115 m_e  v=(+0.000, +0.000, +1.000)

HOMO band edge: X  k=(+0.500, +0.000, +0.000)
  m1 = -0.756 m_e  v=(+0.991, +0.000, +0.135)
  m2 = -1.042 m_e  v=(+0.000, +1.000, +0.000)
  m3 = -2.880 m_e  v=(-0.135, +0.000, +0.991)
```

バンド端が指定した `--band-path` の外にある場合は stderr に警告が出力され、保存先のパスと経過時間は stdout に続けて表示されます。

### 生成されるファイル
`bcal` は結晶ごとに自己完結したディレクトリを書き出し、パイプラインの各段階をサブディレクトリに分けて保持します:

```
<output>/<NAME>/
├── structure.json     # 結晶トポロジのメタデータ（lattice, method, engine,
│                      #   sites, dimer_types, pairs）— 人間可読
├── inputs/            # 生成された量子化学入力ファイル
│   ├── <NAME>_monomer_000.gjf   # g16/g09 は .gjf、pyscf/gpu4pyscf/orca は .xyz
│   ├── <NAME>_dimer_000.gjf
│   └── ...
├── logs/              # DFT 出力ログ（.log / .out, .chk）
├── matrices/          # 抽出された数値（NumPy .npz）
│   ├── monomers.npz   # モノマーの MO 係数とエネルギー準位
│   ├── dimers.npz     # ダイマーの重なり / Fock 行列
│   └── transfer.npz   # トランスファー積分とオンサイト準位
└── results/           # 強束縛計算の最終結果
    ├── band.png       # --band-path に沿ったバンド構造図
    ├── band.npz       # バンドエネルギー、k 距離、目盛位置とラベル
    └── effective_mass.csv  # HOMO/LUMO バンド端の有効質量と軸ベクトル
```

#### `structure.json`
生成された構造の人間可読な記録です: `name`, `method`, `engine`, 生成パラメータ、直接格子ベクトル `lattice`、ユニークな分子サイト `sites`、対称的にユニークな `dimer_types`、および全ペアの表 `pairs`（分子インデックス、格子オフセット、各ペアが対応するダイマー種）。計算ごとの数値は `matrices/` 配下に別途保存されます。

#### 入力ファイルの命名
モノマー・ダイマーの入力は `<NAME>_monomer_{index:03d}` および `<NAME>_dimer_{type:03d}` という名前です（拡張子はエンジンに応じて選択）。ファイル名にはサイト / ダイマー種の識別子のみを持たせ、強束縛モデルを再構成するために必要な写像情報（中心 / 近接分子、格子オフセット、軌道の並び順）は `structure.json` に格納されます。

## 補足

1. **計算時間**: 計算時間は、セルあたりの分子数、手法 / 基底関数系、選択したバックエンドに強く依存します。
2. **メモリ使用量**: 大きな系には十分なメモリを確保してください（`-m`）。
3. **Gaussian のインストール**: `g09` / `g16` エンジンには Gaussian 09 または Gaussian 16 が必要です。
4. **依存関係**: 選択したバックエンド用のオプション依存パッケージがインストールされていることを確認してください（インストールの項を参照）。

## トラブルシューティング

### 計算が途中で停止した場合
```bash
# --resume オプションで再開
bcal xxx.cif --resume
```

### メモリ不足エラー
```bash
# メモリ量を増やす
bcal xxx.cif -m 32
```

### バンド端がバンド図上に無い場合
`bcal` が HOMO/LUMO のバンド端が `--band-path` 上に無いと警告した場合でも、バンド端と有効質量自体は正しい値です（ブリルアンゾーン全体から求められています）。ただし `band.png` には真の極値が表示されません。報告された k 点の高対称点ラベルを `--band-path` に追加し、`-r` を付けて再実行すれば、DFT を再計算せずに再描画できます。

### CIF ファイルが読み込めない場合
CIF ファイルには様々な形式があり、bcal で読み込めないものもあります。以下をお試しください:

1. **別のソフトウェアで CIF 形式を変換する**: [Mercury](https://www.ccdc.cam.ac.uk/solutions/software/mercury/) などのソフトウェアで CIF ファイルを開き、再エクスポートすると解決する場合があります。
2. **お問い合わせ**: 読み込めない CIF ファイルを下記のメールアドレスにお送りいただければ、対応を検討します。

# 著者
[松井研究室, 有機エレクトロニクス研究センター（ROEL）, 山形大学](https://matsui-lab.yz.yamagata-u.ac.jp/)  
岡田 智悠、尾沢 昂輝、本間 友、松井 弘之  
Email: h-matsui[at]yz.yamagata-u.ac.jp  
[at] を @ に置き換えてください

# 参考文献
[1] Qiming Sun et al., Recent developments in the PySCF program package, *J. Chem. Phys.* **2020**, *153*, 024109.  
[2] Benjamin P. Pritchard et al., New Basis Set Exchange: An Open, Up-to-Date Resource for the Molecular Sciences Community, *J. Chem. Inf. Model.* **2019**, *59*, 4814-4820.  
[3] Frank Neese, The ORCA program system, *Wiley Interdiscip. Rev. Comput. Mol. Sci.*, **2012**, *2*, 73-78.
