# This repository contain the C code to create neural network which could classify mnist.
You could create your own neural network by using "NN4_adam.c"

水色桜（みずいろさくら）です。
このリポジトリではC言語を用いてMNISTの分類が可能なニューラルネットワークを自作するためのコードを公開します。、
今回作成したモデルの精度を以下に示します。
20回の学習で100.0％の精度を達成できています。
```
1回目計算終了
error : 2.277329
accuracy : 75.000000
2回目計算終了
error : 2.092417
accuracy : 93.500000
3回目計算終了
error : 1.553882
accuracy : 50.000000
4回目計算終了
error : 5.791287
accuracy : 9.500000
5回目計算終了
error : 0.892912
accuracy : 90.000000
6回目計算終了
error : 7.462903
accuracy : 43.000000
7回目計算終了
error : 0.519350
accuracy : 100.000000
8回目計算終了
error : 7.747426
accuracy : 38.000000
9回目計算終了
error : 0.632810
accuracy : 93.000000
10回目計算終了
error : 8.048656
accuracy : 0.500000
11回目計算終了
error : 0.755729
accuracy : 80.500000
12回目計算終了
error : 9.222788
accuracy : 0.500000
13回目計算終了
error : 1.548664
accuracy : 72.500000
14回目計算終了
error : 13.746381
accuracy : 0.500000
15回目計算終了
error : 6.904788
accuracy : 92.000000
16回目計算終了
error : 13.746413
accuracy : 84.500000
17回目計算終了
error : 6.386152
accuracy : 100.000000
18回目計算終了
error : 13.746422
accuracy : 97.500000
19回目計算終了
error : 5.550708
accuracy : 100.000000
20回目計算終了
error : 13.746424
accuracy : 100.000000
```
何か不明な点、間違いなどありましたら、コメントいただけると嬉しいです。

# C言語とは
1972年にAT&Tベル研究所のデニス・リッチーらが主体となって開発した汎用プログラミング言語。コンパイラ言語であるため、ほかのプログラミング言語と比べて高速に動作します。（Pythonのように機械学習のライブラリが充実していないため、機械学習を行うためにはフルスクラッチする必要があります。そのため、C言語は機械学習にはほとんど使われません。）

# [MNIST](https://github.com/pjreddie/mnist-csv-png)とは
[MNIST(Mixed National Institute of Standards and Technology database)](https://github.com/pjreddie/mnist-csv-png)とは、手書き数字画像60,000枚と、テスト画像10,000枚を集めた、画像データセットです。さらに、手書きの数字「0〜9」に正解ラベルが与えられるデータセットでもあり、画像分類問題で人気の高いデータセットです。

# ニューラルネットワークとは

人間の脳のしくみ（ニューロン間のあらゆる相互接続）から着想を得たもので、脳機能の特性のいくつかをコンピュータ上で表現するために作られた数学モデルです。（[Udemyメディアから引用](https://udemy.benesse.co.jp/data-science/ai/neural-network.html)）
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2847349/f1212ea0-7f28-68e2-557a-ae1aae032711.png)
（[Udemyメディアから引用](https://udemy.benesse.co.jp/data-science/ai/neural-network.html)）

ニューラルネットワークは入力層、中間層、出力層の３つから構成されます。本記事で作成するニューラルネットワークのモデルは次のような感じです。
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2847349/cd4328ec-fb94-a8a0-c9f7-5007c67dcecc.png)

ニューラルネットワークにおいては、前の層からの出力に重みとバイアスを加え、それをアクティベーション関数と呼ばれる非線形関数にかけます。これにより、複雑な分布も表現することが可能になります。ニューラルネットワークは学習を行うことで出力層で人間が望む結果（正しい答え、正解）が出るようにします。学習によって中間層の重みとバイアスを最適化していきます。

# 数学的な背景

ニューラルネットワークを構成するうえで、中間層と出力層のアクティベーション関数を選ぶ必要があります。本記事では中間層のアクティベーション関数はLeaky ReLU、出力層のアクティベーション関数はsoftmax関数を用います。
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2847349/8011635f-3f11-6fb3-7a43-5b270897e6e2.png)
Leaky ReLU関数は先行研究において優れた成果を出している関数であり、勾配消失（勾配が小さくなり、ニューラルネットワークの学習が進まなくなってしまう問題）が起きにくいという性質を有しています。
softmax関数は出力をすべて足し合わせると１になるように設計された関数であり、分類問題に用いられている関数です。入力に対して０～１の任意の値を返すため、擬似的に確率として扱うことができます。
交差エントロピー誤差はsoftmax関数と一緒に用いると逆伝播が簡単な形に書けるように設計された関数です。一見複雑そうに見えますが、微分すると簡単な形になります。
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2847349/d1798e18-17e2-cb5d-530e-7c74a1b9f27d.png)

誤差逆伝播法はニューラルネットワークにおいて勾配を効率的に求めるための手法です。微分の連鎖率を用いて図のように逆側から勾配を求めていきます。
誤差逆伝播法の導出方法は下の画像の通りです。交差エントロピー誤差を出力で微分した値と、出力を入力で微分した値を掛け合わせると、結果としてとっても簡単な形に書き表すことができます。出力から教師データを引いたものになるので、実装もしやすくなっています（そうなるようにsoftmax関数と交差エントロピー誤差は設計されています。）
![image.png](https://qiita-image-store.s3.ap-northeast-1.amazonaws.com/0/2847349/b0948ed4-c25a-661a-d57e-91126394d224.png)



