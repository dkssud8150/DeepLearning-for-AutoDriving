# RegNet review

## Designing Network Design Spaces

RegNet은 EfficientNet보다 빠르고 성능이 좋은 모델이다. Tesla의 HydraNet에서도 RegNet을 사용했다.

RegNet은 EfficientNet처럼 하나의 best model을 만드는 것이 아니라 뛰어난 모델을 어떤 세팅에서도 사용할 수 있도록 design space를 구성하는 것이 중심이다. design space 라는 것은 NAS에서의 search space와 유사한 의미지만, search space에서는 단 하나의 instance를 뽑아내는 것이 목표였다면, design space에서는 instance들을 뽑아내기 위한 space를 직접 찾아내는 것이 목표가 된다.

&nbsp;

최근에 architecture을 만드는 방법으로는 기본적인 방법인 manual 방식과 NAS(Neural Architecture Search) 방식이 많이 사용된다고 한다. Manual 방식은 모델을 직접 만드는 것을 의미하고, NAS는 특정 search space 안에서 좋은 모델을 자동으로 찾아서 사용하는 것을 의미한다. 자동으로 찾아준다는 것은 특정 세팅에만 적합한 single network를 만드는 것이므로 general하지 않다.

그래서 `Designing Network Design Spaces` 논문에서는 manual design과 NAS 방식의 장점을 조합한 새로운 디자인 패러다임을 제시한다.

1. Manual : 네트워크 구조가 심플하고 general한 모델을 생성
2. NAS : semi-automated procedure을 통해 자동적 모델 생성

&nbsp;

&nbsp;

## Neural Architecture Search(NAS)

NAS 나 design space에 대해 짧게 리뷰하고 넘어가려고 한다.

&nbsp;

&nbsp;

| [NAS 논문 : Neural Architecture Search with Reinforcement Learning](https://arxiv.org/abs/1611.01578)
| [NAS 참고 블로그1](https://ahn1340.github.io/neural%20architecture%20search/2021/04/26/NAS.html)
| [NAS 참고 블로그2 - 고려대학교 DMQM 연구실 세미나](http://dmqm.korea.ac.kr/activity/seminar/226)
| [NAS 참고 블로그3 - NAS with RL 논문 리뷰](https://pigranya1218.tistory.com/entry/%EB%85%BC%EB%AC%B8-%EB%A6%AC%EB%B7%B0-Neural-Architecture-Search-with-Reinforcement-Learning)
| [NAS 참고 논문1 : Neural Architecture Search: A Survey](https://arxiv.org/abs/1808.05377)

&nbsp;

딥러닝의 시대로 접어들면서 수많은 모델들이 나왔다. 이러한 모델들은 전문가들이 손수 hidden layer를 얼마나 쓸지, stride, padding 등의 정보들을 디자인한 결과이다. 구조를 보며 공부하며 자신의 task에 적용할 수는 있겠지만, task별로, 또는 어떤 데이터셋으로 구성되어 있냐에 따라 성능이 달라진다. 즉 전문가가 직접 디자인한 모델을 그대로 가져와 사용한다 해도 최적의 학습과 테스트가 이루어질 수는 없다는 뜻이다.

그렇다면, 모든 구조와 파라미터를 직접 공부하여 최적화해야 하는데, 이러한 과정이 올바른 과정일까? 과거에는 이런 방법들 밖에 없었지만, NAS 방법론을 통해 이러한 작업을 자동화하여 주어진 task에 가장 최적인 네트워크 구조를 빠르게 탐색할 수 있어졌다.

&nbsp;

NAS에서 가장 유명한 논문으로는 `Neural Architecture Search with Reinforcement Learning`(https://arxiv.org/abs/1611.01578)이다. NAS 방법론을 처음 제시한 논문으로 신경망의 구조를 가변길이의 문자열로 나타낼 수 있다는 점을 활용하여, 문자열로 신경망 구조를 출력하는 RNN controller(RL-agent)를 만들어, 실제 신경망을 생성한 후 각 네트워크에서 얻어진 accuracy를 RL의 reward로 취급하여 reward가 최대가 되는 문자열을 찾는다. 이 논문의 구조는 아래 사진과 같다.

<img src="../../assets/NAS_with_RL.png">

&nbsp;

&nbsp;

### component of NAS

<img src="../../assets/component_of_nas.png">

NAS는 크게 `Search Space`, `Search Strategy`, `Performance Estimation Strategy` 3가지 요소로 구성된다.

1. Search Space(탐색 공간)

    탐색 공간은 알고리즘이 탐색을 수행하는 영역으로, 여기서 `convolution`,`fully-connected`,`pooling` 등과 같은 연산들이 어떻게 연결되는지, 총 몇개의 연산이 네트워크를 구성하는지 정의한다.

    &nbsp;

2. Search Strategy(탐색 전략)

    탐색 전략은 탐색 공간상의 많은 operation(configuration)들 중 어떤 것이 가장 좋은 연산인지를 빠르고 효율적으로 찾아낸다. 탐색 전략에는 탐색 공간을 어떻게 정의하느냐, 목표가 무엇이냐에 따라 `random search`, `reinforcement learning`, `evolutionary strategy`, `gradient descent`, `bayesian optimization` 등의 방법을 사용할 수 있다.

    탐색 전략을 디자인할 때는 탐색 공간을 모두 탐색(exploration)하면서 좋은 부분을 찾아낼 수 있도록 해야 한다(exploitation). 보통은 이 둘은 trade-off 관계에 있어 디자인할 때 두 가지 모두를 잘 수행할 수 있도록 설정해야 한다.

    &nbsp;

    NAS에 evolutionary strategy를 사용하여 만든 모델인 [AmoebaNet](https://arxiv.org/abs/1802.01548)이 있다. search space로부터 랜덤하게 샘플링된 네트워크들의 variable를 학습시키고, 그중 loss가 가장 낮은 네트워크들의 구조에 random weight를 주어 학습시킨다.

    &nbsp;

    NAS에서 search space를 미분가능한 형태로 변환시켜 gradient descent를 수행한 논문도 있다. 이 논문의 이름이 `Differentiable Architecture Search`[DARTS](https://arxiv.org/abs/1806.09055)이다.

    &nbsp;

    &nbsp;

3. Performance Estimation Strategy(모델 성능 추정 전략)

    탐색 전략에서 여러 개의 후보 configuration을 추출하면 여기서 후보들의 성능을 예측하고, 그 예측치를 바탕으로 최적의 configuration으로 수렴할 때까지 반복 수행한다. 여기서 evaluate(평가)가 아닌 estimation(추정)인 이유는 모든 configuration을 일일히 학습시켜 평가하면 너무 많은 시간이 소요되므로 성능을 예측하여 최적의 전략을 찾아낸다.

&nbsp;

&nbsp;

## Tools for Design Space Design

다시 RegNet으로 돌아와서, design space를 어떻게 평가하는지에 대한 설명을 살펴본다. 단순히 manual과 같은 방식을 사용하거나 두 개의 design space로부터 각각 찾아낸 best 모델들의 성능을 비교하는 것보다 distribution 자체를 비교하는게 더 robust하다. 따라서 모델의 distribution을 얻기 위해 design space에서 n개의 모델을 샘플링하여 학습한다. 평가에는 EDF(Error empirical Distribution Function), 즉 각 모델의 error가 $ e_i $보다 작은 모델의 수에서 전체 모델의 수를 나눈다.

$ F(e) = \frac{1}{n} \sum_{i=1}^n 1[e_i < e] $

&nbsp;

<img src="../../assets/edf_graph.png">

위의 그림들은 500개의 샘플링된 모델에 대한 그래프이다. 중간과 오른쪽 그래프는 다양한 네트워크의 속성들을 살펴볼 수 있다.

&nbsp;

추정하는 과정은 다음과 같다.

1. 디자인 공간으로부터 n개의 모델을 샘플링하고 학습하여 모델의 분포를 얻는다.
2. EDF를 사용해 디자인 공간을 평가하고 시각화한다.
3. 디자인 공간의 속성을 통해 insight를 얻는다.
4. insight를 통해 다시 디자인 공간을 재정의하여 더 나은 디자인 공간을 만든다.


&nbsp;

&nbsp;

## The AnyNet Design Space

이번에는 표준으로 고정된 네트워크로 가정한 네트워크의 구조를 탐구해보고자 한다. 즉, 블럭의 수, 블럭의 너비, 블럭의 다른 매개변수들에 대한 탐구이다. 

AnyNet의 design space는 아래 그림과 같은 구조로 되어 있다.

<img src="../../assets/anynet.png">

네트워크의 기본 뼈대는 `stem`, `body`, `head` 로 이루어져 있고, body는 다시 세부적으로 여러 개의 stage로 구성되어 있으며 각 stage는 또 다시 여러 개의 block으로 이루어져 있다. 각 stage의 자유도는 block의 개수, width, block parameter 들에 의해 결정되고, block들은 stride에 따라 다르게 생겨나기도 하며, 아예 다른 종류의 block도 사용 가능하다.

&nbsp;

표준이 되는 AnyNetX는 16 degrees of freedom 를 가지는데, 이는 4개의 stage, 각 stage마다의 4개의 parameter를 가지기 때문이다. 4개의 파라미터란 block의 개수 $ d_i $ , block width $ w_i $, bottleneck ratio $ b_i $, group width $ g_i $ 를 의미한다. 이 값들을 변경해가며 다양한 조합을 생성한다. 각 파라미터의 제약은 다음과 같다.

- $ d_i \leq 16 $
- $ w_i \leq 1024 \; (only, \, w_i\,\%\,8 \,= \,0 ) $
- $ b_i \in {1,2,4} $
- $ g_i \in {1,2,4,8,16,32} $

&nbsp;

&nbsp;

이런 파라미터들에 대한 제약(constriaint)를 다양하게 부여하며 RegNet에 도달하는 과정을 살펴본다. 그 과정이 아래의 그림이다.

<img src="../../assets/design_space_design.png">

&nbsp;

1. $ AnyNetX_A $
    아무런 제약이 존재하지 않는 AnyNetX 가 $ AnyNetX_A $ 이다. 많은 경우의 수가 존재하여 자유도가 굉장히 높다.

    &nbsp;

2. $ AnyNetX_B $
    A 버전에서 bottleneck ratio $ b_i $ 에 대한 제약(constraint)만 추가한 것이 $ AnyNetX_B $ 이다. bottleneck ratio를 1,2,4 중에 하나로 고정시킨다. ($b_i$ = b)

    <img src="../../assets/anynetx_b.png">

    &nbsp;

    위의 그래프는 A,B버전에 대한 EDF이다. 여기서 알 수 있는 것은 bottleneck ratio 제약조건이 있는 것과 없는 것이 차이가 없다는 것이다. 그래서 이후에는 bottleneck ratio를 고정한 채로 테스트한다.

    &nbsp;

3. $ AnyNetX_C $
    B버전에서 shared group width를 사용해본다. 즉 모든 스테이지에 대해 group width도 하나의 같은 수로 고정한다. ($g_i$ = g)

    <img src="../../assets/anynetx_c.png">

    여기서도 알 수 있듯이 차이가 없다. 따라서 다음 테스트에서도 고정한 채로 진행한다.

    &nbsp;

4. $ AnyNetX_D $
    이번에는 C 모델에서 네트워크의 뒤로 갈수록 block width가 커지도록 하는 제약을 추가한다. ($w_{i+1} \leq w_i$)

    <img src="../../assets/anynetx_d.png">

    그림에서 볼 수 있듯이 해당 제약으로 인해 성능이 좋아졌다. block width를 키웠을 때 가장 성능이 좋았고, 그 다음은 C버전, 그 다음은 width를 고정한 것이고, width가 뒤로갈수록 작아질 때 성능이 가장 안좋았다.

    &nbsp;

5. $ AnyNetX_E $
    D 모델에서 width 처럼 depth도 뒤로갈수록 키워보았다.

    <img src="../../assets/anynetx_e.png">

    이 또한, 성능이 개선되었다.

    &nbsp;

&nbsp;

## The RegNet Design Space

<img src="../../assets/anynetx_e_test.png">

위의 이미지는 $ AnyNetX_E $ 에 대한 더 심층 연구를 위해 20개의 최고의 성능을 가진 모델을 가져와 그들의 block width를 그린 것이다. 실제로 우리는 width를 매 index마다 높일 수 있지만, 좀 더 간결한 design space를 위해 경량화(quantizing)를 해야 한다. 이 때 index가 바로 depth, 즉 block의 개수가 된다.

&nbsp;

경량화를 위해 block width에 대해 `linear parameterization` 를 사용한다.

$ u_j = w_0 + w_a \cdot j \; for \; 0 \leq j < d $ 

- $ w_0 $ : 초기 너비 (initial width)
- $ w_a $ : 기울기 (slope)
- d : 깊이 (depth)

초기 너비는 당연히 0보다 커야 하고, width가 점차 증가해야 하므로 기울기도 양수가 되야 한다. ($w_o$ > 0, $w_a$ > 0)

&nbsp;

여기서 이전 width보다 얼마나 커졌는지에 대한 파라미터인 $ w_m $ 를 추가한다. 

$ u_j = w_0 \cdot w_m^{s_j} $

- $ w_m $ : 이전 width와 현재 width의 비율
- $ s_j $ : stride

&nbsp;

block의 수는 다음과 같은 식으로 구한다.

$ d_i = \sum_j 1[round(s_j) = i] $

&nbsp;

&nbsp;

이렇게 총 6개의 파라미터(d,w_0, w_a, w_m, b, g) 로 결정이 되는 degign space를 regular network, 또는 $ RegNet $ 이라 한다. 여기서 RegNet은 RegNetX와 같은 말이며, RegNetX에 SE(Squeeze-and-Excitation) 연산을 추가하여 RegNetY 로 표현할 수 있다.

&nbsp;

저자들은 RegNetX의 design space를 다음과 같은 조건을 제시했다.

- b = 1, d <= 40, w_m >= 2
- parameter와 activation에 제한을 둔다. -> square-root와 linear하게 증가.

&nbsp;

## Analyzing the RegNetX Design Space

### RegNet trend

<img src="../../assets/analyze_regnet.png">

테스트 결과 최적의 depth는 ~20 block정도로 다소 작은 값을 가진다. 최적의 bottleneck ratio는 1.0으로 즉 bottleneck이 없는 것이 가장 좋은 성능을 보였다. width multipler Wm 은 2.5정도가 가장 좋았다고 한다. 

&nbsp;

&nbsp;

## reference
- regNet
    - Designing Network Design Spaces, Ilija Radosavovic(2020) (arXiv:2003.13678)
    - https://towardsdatascience.com/regnet-the-most-flexible-network-architecture-for-computer-vision-2fd757f9c5cd
    - https://2-chae.github.io/category/2.papers/31
    - https://cocopambag.tistory.com/47



