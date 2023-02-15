# Project 'Webcam pose classification with using mediapipe'

- 본 프로젝트는 김의석,권준혁,고준성,정병길,추태욱이 참여한 인공지능을 이용한 자세 기울어짐 판별 프로젝트입니다. 

- 본 프로젝트는 Linux Ubuntu WSL2 환경에서 Python 을 사용하여 제작되었습니다. 

- 총 소요기간은 2023-02-01 ~ 2023-02-15 까지로 총 2주입니다.



## Dataset 특징
 
 - Mediapipe 는 사전에 학습된 모델들을 바탕으로 영상 또는 카메라를 통해 얼굴, 동작 등을 인식하는 오픈소스 프레임워크입니다. 

 [미디어파이프 공식 홈페이지](https://google.github.io/mediapipe/)
 
 - 우리는 구글이 개발한 이런 인체 대상 비전인식 ML solution, Mediapipe 를 통해 33 개의 landmark 를 수집하고 각 landmark의 x,y 좌표와 depth(z), visualization(현재 화면에 나타나고 있는지) 여부 에 대한 데이터를 수집했습니다. 
 
 - 각 데이터는 이미지 크기를 바탕으로 0과 1 사이의 데이터로 정규화되어 있어 모델 성능에 긍정적인 영향을 미칩니다.
 
 - 우리는 이 중 *holistic model* 을 사용하여 왼쪽으로 기울여진 자세 2000개, 가운데 자세 2000 개, 오른쪽 자세 2000개의 총 6000 개의 자세를 통해 33개의 landmark들에 대하여 총 198,000 개의 자료를 수집 및 가공했습니다.



## Model 구현 방법
 
 - 최초에는 CNN 모델을 사용하여 pose 를 인식한 후 Detection을 할 수 있도록 구현했습니다.
 
 - 그러나 opencv, mediapipe 를 이용하여 좌표 자료를 입수할 수 있다는 사실을 깨닫고 모델 경량화와 성능향상을 위하여 FNN을 통해 단순 Classification 을 할 수 있도록 개량했습니다. 

 - RandomForestClassifer, GBM 같은 다른 ML 모델을 추가하여 앙상블 모델을 만드는 것도 괜찮은 선택이었을지 모르겠습니다만, FNN 단순 Classification 이 이미 성능이 잘 나오고 있고(accuracy: 0.993) FNN 모델 하나만을 사용하는 것에 비해 연산을 많이 잡아먹기 때문에 FNN 을 사용했습니다.

 - CNN, RNN 을 사용하지 않은 이유에 대해서는, 단순히 사용분야가 달랐기 때문입니다. CNN 은 이미지 형태의 데이터를 처리하기 위한 모델이고 RNN은 우리가 원하는 결과에 시퀀스가 큰 의미가 없었기 때문에 고려사항이 아니었습니다.

 - 추가적으로, 혹시 모를 과적합을 방지하기 위해 레이어 사이에 정규화 레이어를 추가하거나 Dropout layer 을 추가하는 것은 괜찮은 선택입니다. (본 코드에서는 dropout 을 사용했습니다.)



```python
# 구현하실 분들을 위해, 예시 모델을 같이 업로드합니다.
# 2= right, 1=middle , 0 =left

# model 
model = tf.keras.Sequential()
model.add(tf.keras.layers.Dense(128, activation='relu', input_shape=(132,)))
model.add(tf.keras.layers.Dropout(0.5)) # add dropout layer, for avoid overfitting.
model.add(tf.keras.layers.Dense(64, activation='relu'))
model.add(tf.keras.layers.Dense(3, activation='softmax')) # 1=forward, 2=right, 0=left
# for multiclassification, use softmax func.
# There is no CNN layer because there is no image input. 
# image input -> csv data -> modeling 

# Compile 

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```



### 구현시 문제점

 - 자료를 라벨링하고 분류 One-hot encoding 을 위해 loss function 으로 sparse_categorical_crossentrophy 를 사용하였으나 성능 문제가 발생하였습니다. (loss, accuracy 모두  안 좋은 결과를 얻었음. , epoch = 10)

 - 문제 탐색 결과 데이터셋의 자료구조 라벨링과 loss func에 문제가 있었습니다. 기존에는 중앙을 0으로 놓고 좌측으로 기울어졌을 때를 -1, 우측으로 기울어졌을 때를 1로 놓았습니다. 그러나 현존하는 One-hot encoding tool 은 음수를 처리하도록 되어있지 않기 때문에 문제가 발생했습니다.

 - 좋은 결과가 나온 모델도 프로토타입 서비스를 작성하여 평가해보니 자세를 제대로 예측하지 못하는 문제가 있었습니다. 데이터를 수집한 방법(opencv, video)과 실제 데이터를 입수한 방법(web_rtc)이 달라 문제가 발생했습니다.



 ### 해결방법
 
 - sparse_categorical_crossentrophy 자체의 문제도 존재했기 때문에, 이를 해결하기 위해 tf.keras.utils 의 to_categorical 메서드를 사용하여 원핫인코딩을 진행하고 loss func 를 categorical_crossentrophy 로 교체했습니다. 

 - 데이터 입수 방법 자체를 교정하여 재학습시켰습니다. web_rtc 를 통해 얻은 사진 결과를 학습시킴으로서 수집방법과 모델이 학습한 환경이 달라 발생했던 문제를 해결했습니다. 이에 충분히 좋은 결과를 얻을 수 있었고, test 결과도 좋게 나왔습니다. 



```python 

Epoch 10/10
81/81 [==============================] - 1s 7ms/step - loss: 0.0243 - accuracy: 0.9934

results = model.evaluate(X_test, Y_test)
print('accuracy: ', results[1])

9/9 [==============================] - 0s 9ms/step - loss: 0.0213 - accuracy: 0.9930
accuracy:  0.99303138256073
```
 


 ## 서비스 구현



