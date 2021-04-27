# Malaria-Detection
ML models for detecting malaria parasite in Thin-blood smear.

### VGG19 Architecture
<p align="center">
  <img width="700" align="center" src="https://github.com/RCYADAV/Malaria-Detection/blob/main/images/vgg19Archi.png" alt="demo"/>
</p>

**Fine Tuning** a model means training one or more layer of a pre-trained model so that it fits to our data better than using the pre-trained model as it is.

## Dataset
[Malaria Data](https://drive.google.com/drive/u/1/folders/1_5c5k3rkh3jPbEgEnOVC5lZlGw_pKBM2)

## 1. Basic CNN
 **Basic CNN Model**, Trainable params: **25,100,046**
```python
model=Sequential()
model.add(Conv2D(filters=16,kernel_size=2,padding="same",activation="relu",input_shape=(224,224,3)))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=32,kernel_size=2,padding="same",activation ="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add(Conv2D(filters=64,kernel_size=2,padding="same",activation="relu"))
model.add(MaxPooling2D(pool_size=2))
model.add( Dropout(0.2))
model.add(Flatten())
model.add(Dense(500,activation="relu"))
model.add( Dropout(0.4))
model.add(Dense(2,activation="softmax"))
model.summary()
```
<p align="center">
  <img width="700" align="center" src="https://github.com/RCYADAV/Malaria-Detection/blob/main/images/basiccnnModel.png" alt="demo"/>
</p>

Classification Report for the **Basic CNN Model**
<p align="center">
  <img width="700" align="center" src="https://github.com/RCYADAV/Malaria-Detection/blob/main/images/basiccnnAccuracy.png" alt="demo"/>
</p>

## 2. Frozen VGG19
Made all the layer of VGG19 as non-trainable 
```python
for layer in vgg19.layers:
    layer.trainable = False
```
<p align="center">
  <img width="700" align="center" src="https://github.com/RCYADAV/Malaria-Detection/blob/main/images/frozenTrainable.png" alt="demo"/>
</p>

**Frozen-VGG19 Model** added our own dense layer at the end with softmax as activation function, Trainable params: **50,178**
```python
predictions = Dense(2, activation = 'softmax')(x)
model = Model( inputs = vgg19.input, outputs = predictions)
model.summary()
```
<p align="center">
  <img width="700" align="center" src="https://github.com/RCYADAV/Malaria-Detection/blob/main/images/frozenModel.png" alt="demo"/>
</p>

Classification Report for the **Frozen-VGG19 Model**
<p align="center">
  <img width="700" align="center" src="https://github.com/RCYADAV/Malaria-Detection/blob/main/images/frozenAccuracy.png" alt="demo"/>
</p>

## 3. Fine Tuned VGG19
Made all the layer of VGG19 as non-trainable except **Block5_conv4** and **Block5_pool** 
```python
image_size = [224, 224]
# include_top = false ==> By setting this it will not include fully connected layer which is used to predict output
vgg19 = VGG19(input_shape = image_size + [3], weights = 'imagenet', include_top = False)
```
```python
trainable = False
for layer in vgg19.layers:
    if layer.name == 'block5_conv4':
        trainable = True
    layer.trainable = trainable
```
<p align="center">
  <img width="700" align="center" src="https://github.com/RCYADAV/Malaria-Detection/blob/main/images/fineTuningTrainable.png" alt="demo"/>
</p>

**Frozen-VGG19 Model** added our own dense layer at the end with softmax as activation function, Trainable params: **2,409,986**

```python
predictions = Dense(2, activation = 'softmax')(x)
model = Sequential([
        vgg19,
        Flatten(),
        Dropout(0.20),         
        Dense(2, activation='softmax')    
    ])
model.summary()
```
<p align="center">
  <img width="700" align="center" src="https://github.com/RCYADAV/Malaria-Detection/blob/main/images/fineTuningModel.png" alt="demo"/>
</p>

Classification Report for the **Frozen-VGG19 Model**
<p align="center">
  <img width="700" align="center" src="https://github.com/RCYADAV/Malaria-Detection/blob/main/images/fineTuningAccuracy.png" alt="demo"/>
</p>


## Contributors
[Vijayant Yadav](https://github.com/vijayantyadav11) , [Ramchandra Yadav](https://github.com/RCYADAV) , [Swapnil Patel](https://github.com/patelswapnil01)

## LICENSE

[MIT](LICENSE)
