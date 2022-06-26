# BayesModel

* model/phmm.py

    ポアソン隠れマルコフモデルを利用できます.
    
    完全分解および構造化の2つの変分推論アルゴリズムが実装されています.
    
    ギブスサンプリングについて将来的に実装する予定です.
    
* model/pmm.py

    ポアソン混合モデルを利用できます.
    
    ギブスサンプリング,変分推論,崩壊型ギブスサンプリングの3つの推論アルゴリズムを実装.
    
    ギブスサンプリングに関しては低次元データに対して高速化できる工夫が為されており,
    
    崩壊型よりも速く実行できます.

* tutorial_notebook_phmm.ipynb

    Poisson Hidden Markovの使い方について粗方書かれています.

* tutorial_notebook_pmm.ipynb

    Poisson Mixtureの使い方について粗方書かれています.