# * 도미 데이터

bream_length = [25.4, 26.3, 26.5, 29.0, 29.0, 29.7, 29.7, 30.0, 30.0, 30.7, 31.0, 31.0, 31.5, 32.0, 32.0, 32.0, 33.0, 33.0, 33.5, 33.5, 34.0, 34.0, 34.5, 35.0, 35.0, 35.0, 35.0, 36.0, 36.0, 37.0, 38.5, 38.5, 39.5, 41.0, 41.0]
bream_weight = [242.0, 290.0, 340.0, 363.0, 430.0, 450.0, 500.0, 390.0, 450.0, 500.0, 475.0, 500.0, 500.0, 340.0, 600.0, 600.0, 700.0, 700.0, 610.0, 650.0, 575.0, 685.0, 620.0, 680.0, 700.0, 725.0, 720.0, 714.0, 850.0, 1000.0, 920.0, 955.0, 925.0, 975.0, 950.0]

import matplotlib.pyplot as plt

# * 도미 데이터 산점도

plt.scatter(bream_length, bream_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# * 빙어 데이터

smelt_length = [9.8, 10.5, 10.6, 11.0, 11.2, 11.3, 11.8, 11.8, 12.0, 12.2, 12.4, 13.0, 14.3, 15.0]
smelt_weight = [6.7, 7.5, 7.0, 9.7, 9.8, 8.7, 10.0, 9.9, 9.8, 12.2, 13.4, 12.2, 19.7, 19.9]

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

# ## 머신러닝 프로그램

# _도미의 길이(무게)와 빙어의 길이(무게)를 하나의 리스트로 합치기_

length = bream_length + smelt_length
weight = bream_weight + smelt_weight

fish_data = [[l, w] for l, w in zip(length, weight)]

print(fish_data)

fish_target = [1]*35 + [0]*14

print(fish_target)

from sklearn.neighbors import KNeighborsClassifier

kn = KNeighborsClassifier() # 클래스를 객체로 만들어서 사용

kn.fit(fish_data, fish_target)

# _fit() 메서드는 처음 두 매개변수로 훈련에 사용할 특성과 정답 데이터 전달 (사이킷런 모델을 훈련할 때 사용하는 메서드_

kn.score(fish_data, fish_target)

# _score() 메서드는 훈련된 사이킷런 모델의 성능 측정,
#     처음 두 매개변수로 특성과 정답 데이터 전달
#     predict() 메서드로 예측을 수행한 후 다음 분류 모델일 경우 정답과 비교하여 올바르게 예측한 개수의 비율 반환_

# ### k-최근접 이웃 알고리즘

plt.scatter(bream_length, bream_weight)
plt.scatter(smelt_length, smelt_weight)
plt.scatter(30, 600, marker = '^')
plt.xlabel('length')
plt.ylabel('weight')
plt.show()

kn.predict([[30, 600]])

# _predict() 메서드는 사이킷런 모델을 훈련하고 예측할 때 사용하는 메서드, 특성 데이터 하나만 매개변수로 받음_

print(kn._fit_X)