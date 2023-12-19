# BayesModel

## ディレクトリ説明

- `model/`
    - `model/pmm.py`
        - 混合ポアソン分布のギブスサンプリングを行います。混合比率の事前分布としてディリクレ分布、強度の事前分布としてガンマ分布を用意しています。
- `tutorial/`
    - `tutorial/Comparison_withPyMC.ipynb`
        - PyMCの実装と本実装を簡単に比較した結果を載せたものです。

## 環境

- numpy
- scipy
- tqdm