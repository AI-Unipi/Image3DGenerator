![alt text](https://github.com/AI-Unipi/Image3DGenerator/blob/master/logo4.png)

# Table of Contents :mag_right: :book:
* [Introduction](#introduction)
* [Team Members](#team-members)
* [Inspiration](#inspiration)
* [Usage](#usage)
* [Data/File Formats](#data-format)
* [Examples](#examples)


# <a name="introduction"></a>Introduction :postal_horn:
This repository contains a class to permorm data augmentation on 3D objects (e.g. 3D medical images). It is a 3D (..well 4D with the number of channels included :sweat_smile:) version of the 2D "tf.keras.preprocessing.image.ImageDataGenerator". We also provide two examples as a simple guideline.

Data augmentation is a regularization technique that has been found extremely usefull when training CNNs. It is a techinique that prevents the model of seeing the original training and validation data during training, and instead applies some transofrmations on the original training data (or batches) and lets the model see those instead. Data augmentation is a mean to reduce overfitting and make a more robust model.



# <a name="team-members"></a>Team Members :busts_in_silhouette:
* "Stefanos Karageorgiou" <stefanoskarageorgiou94@gmail.com>
* "Ani Ajdini" <ajbajram@gmail.com>

# <a name="inspiration"></a>Inspiration :bulb:

While doing my thesis this summer (Stefanos), I realized that the tensorflow resources on 3D image model training are limited (almost none pretrained models, limited regularization techniques etc). 3D data are sometimes hard to find, especially medical and they are often not many in number. Hence, we really believe that data augmentation can have a huge impact on the overfitting prevention.

This repository is a quarantine project (yes, we were bored :relieved:) created to help the few other crazies working on similar projects. 

# <a name="usage"></a>Usage :clipboard:

The Image3DGenerator class despite its name is actually a "tf.keras.utils.Sequence" object, or in other words a base object for fitting to a sequence of data, such as a dataset. Sequence are a safer way to do multiprocessing as this structure guarantees that the network will only train once on each sample per epoch which is not the case with generators.

This class applies random transformations to the original training and validation data which change during each epoch. 

The options we provide (yet) are the following:
- Generation of batches without any transformation
- Rotation: Randomly rotates the whole object to a range of angles drawn from a normal distribution with 0 mean and variance specified by the user.
- Gaussian noise: Adds random noise to the 3D objects drawn from a normal distribution with 0 mean and variance specified by the user.
- Normalization: Applies a min max scaler transofrmation to the objects which bounds the voxel values between 0 and 1.

# <a name="data-format"></a>Data/File Formats :file_folder:

In order to use this class your data and folders should be structured as follows:

----data-folder/data.npy <br />
--Image3DGenerator.py <br />
--your_python_script

**Notes:** <br />
:zap: The data folder should contain each 3D object **seperately**, each in a numpy array form (.npy) <br />
:zap: Each 3D object should have the following dimension order: (object_length, object_height, object_width, number_of_channels (if grayscaled can be skipped)). <br >
To use the class you need to do the following steps:
- Create a dictionary containing the ID of the training (and validation examples if applicable).
- Create a dictionary containing all the training (and validation) IDs along with their classes. **The classes should be integers starting with 0**.

After having all the prerequirements ready you simply type the following:

```python
from Image3DGenerator import DataGenerator

params = { 
          'dim': your object's dimensions,
          'batch_size': opted batch size,
          'n_classes': number of your classes,
          'n_channels': 1 if grayscale, 3 if RGB,
          'rotation': True in case you want to apply random roation during training,
          'normalisation': True,
          'min_bound': in case normalisation is True, specify the minimum voxel value of your objects,
          'max_bound': in case normalisation is True, specify the maximum voxel value of your objects,
          'gaussian_noise': True,
          'noise_mean': 0,
          'noise_std': 0.01,
          'shuffle': True,
          'rotate_std':45,
          'path':'./data' #path of the folder containing the data,
          'display_ID':False}

# Generators
training_generator = DataGenerator(dictionary['train'], labels, **params)
validation_generator = DataGenerator(dictionary['validation'], labels, **params)

#After creating and compliling your tf model

model.fit(x = training_generator,
          epochs= no_epochs, 
          validation_data= validation_generator)
```

# <a name="examples"></a>Examples :eyes:

Two examples with codes and outputs are available at the examples folder. Below you will find visual examples with the intention to help the user understand how the class treats the data during training.

## Visual Examples

**1)** Grayscale CT scan <br >
Transformations applied: Rotation, Noise, Normalisation

Original            |  Transformed
:-------------------------:|:-------------------------:
![](https://github.com/AI-Unipi/Image3DGenerator/blob/master/gifs_images/ct_o.gif)  |  ![](https://github.com/AI-Unipi/Image3DGenerator/blob/master/gifs_images/ct_t.gif)

**2)** RGB gif <br >
Transformations applied: Rotation, Noise

Original            |  Transformed
:-------------------------:|:-------------------------:
![](https://github.com/AI-Unipi/Image3DGenerator/blob/master/gifs_images/dog_o.gif)  |  ![](https://github.com/AI-Unipi/Image3DGenerator/blob/master/gifs_images/dog_t.gif)


**3)** RGB gif <br >
Transformations applied: Rotation

Original            |  Transformed
:-------------------------:|:-------------------------:
![](https://github.com/AI-Unipi/Image3DGenerator/blob/master/gifs_images/cat_o.gif)  |  ![](https://github.com/AI-Unipi/Image3DGenerator/blob/master/gifs_images/cat_t.gif)

**4)** RGB gif <br >
Transformations applied: Noise

Original            |  Transformed
:-------------------------:|:-------------------------:
![](https://github.com/AI-Unipi/Image3DGenerator/blob/master/gifs_images/dog1_o.gif)  |  ![](https://github.com/AI-Unipi/Image3DGenerator/blob/master/gifs_images/dog1_t.gif)
