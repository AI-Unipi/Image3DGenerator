# -*- coding: utf-8 -*-

import numpy as np
import cv2 
import tensorflow as tf
import sys

class DataGenerator(tf.keras.utils.Sequence):
    
    'Generates data for Tensorflow'
    def __init__(self, IDlist : list, 
                 labels : np.array,
                 batch_size : np.int = 32,
                 dim : tuple = (96,176,176), 
                 n_channels : np.int  = 1,
                 n_classes : np.int =3, 
                 rotation : bool = False, 
                 rotate_std : np.float = 15 ,
                 normalisation : bool = False, 
                 min_bound : np.float = 0, 
                 max_bound : np.float = 255,
                 gaussian_noise : bool = False, 
                 noise_mean : np.float = 0, 
                 noise_std : np.float = 0.01,
                 path : str = './', 
                 shuffle : bool = True,
                 display_ID: bool = False):
        
        """
        Parameters
        ----------
        IDlist : list
            list containing the unique IDs of 3D ojects.
        labels : np.array
            numpy array containing the label of each 3D oject.
        batch_size : np.int, optional
            size of batch specified by the user. The default is 32.
        dim : tuple, optional
            tuple containing the dimensions of the 3D ojects. 
            The default is (96,176,176).
        n_channels : np.int, optional
            Number specifying the type of image. 1 for grayscale 3 for RGB. 
            The default is 1.
        n_classes : np.int, optional
            integer corresponding to the number of different labels of 3D objects.
            The default is 3.
        rotation : bool, optional
            Boolean specifying if the rotation of 3D objects is needed. 
            The default is False.
        rotate_std : np.int, optional
            Number specifying the variance of the normal distrubution 
            which defines the random angle taken for the rotation of each 3D object. 
            The default is 15.
        normalisation : bool, optional
            Boolean specifying if the normalisation of 3D objects is needed. 
            The normalisation method in use is the min max scaler, normalising 
            the 3D object values between 0 and 1. 
            The default is False.
        min_bound : np.int, optional
            Integer specifying the value of the minimum voxel. The default is 0.
        max_bound : np.int, optional
            Integer specifying the value of the maximum voxel. The default is 255.
        gaussian_noise : bool, optional
            Boolean specifying if added gaussian noise on 3D objects is needed. 
            The default is False.
        noise_mean : np.float, optional
            Fload specifying the mean of the normal distrubution 
            which defines the random noise added on each 3D object. 
            The default is 0.
        noise_std : np.float, optional
            Fload specifying the variance of the normal distrubution 
            which defines the random noise added on each 3D object. 
            The default is 0.01.
        path : str, optional
            String specifying the path where the folder of .npy 3D objects is at.
            The default is './'.
        shuffle : bool, optional
            Boolean specifying if tha data is being shuffled on each epoch.
            The default is True.

        Returns
        -------
        None.

        """
        
        #Initialization of parameters 
        self.dim = dim;
        self.batch_size = batch_size;
        self.labels = labels;
        self.IDlist = IDlist;
        self.n_channels = n_channels;
        self.n_classes = n_classes;
        self.shuffle = shuffle;
        self.rotation = rotation;
        self.rotate_std = rotate_std;
        self.normalisation = normalisation;
        self.min_bound = min_bound;
        self.max_bound = max_bound;
        self.gaussian_noise = gaussian_noise;
        self.noise_mean = noise_mean;
        self.noise_std = noise_std;
        self.path = path;
        self.on_epoch_end();
        self.display_ID = display_ID;
        
        if self.n_channels in [1,3]:
            pass
        else:
            print('n_channels should be 1 for grayscale, 3 for RGB')
            sys.exit(1)           

    def __len__(self) -> np.int :
        
        """
        Returns
        -------
        The integer number of the batches.
        """

        return (int(np.floor(len(self.IDlist) / self.batch_size)))


    def __getitem__(self, index : np.int) -> tuple([np.ndarray, np.array]):
        
        """
        A function which generates a single batch of data either or their 
        original form or after being transformed. It also returns the corresponding
        labels in a one hot encoded form. 
        
        Parameters
        ----------
        index : np.int
        Parameter specified by Tensorflow during model.fit().

        Returns
        -------
        X : np.ndarray with shape (batch_size, 3D object dim, n_channels).
        If specified by the user X contains the 3D objects transformed
        y : np.ndarray with shape (batch_size, n_classes)
        Labels one hot encoded.
        """
        
        # Generates indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size];

        # Creates a list with a batch of ID names
        IDlist_temp = [self.IDlist[k] for k in indexes];

        # Generates a batch of the original data
        X, y = self.__data_collection(IDlist_temp);
        
        if self.display_ID:
            print('IDs used: ',IDlist_temp)
            
        # If any of the transformations are specified by the user, returns the
        #batch of data transformed.
        if not self.rotation and not self.gaussian_noise and not self.normalisation:
            pass
        
        else:
            #Transorm data
            X = self.__data_transormation__(X)
            
        return(X, y)


    def __make_3D__(self, list_of_images : list) -> np.ndarray :
        
        """
        A function which vertically stack a list of 2D **grayscale** images into a single
        3D object.
        
        Parameters
        ----------
        list_of_images : list
            A list of np.ndarrays contaings 2D images.

        Returns
        -------
        A 3D object i.e and np.ndarray of stacked 2D images
        """

        
        r, c = list_of_images[0].shape;
        original3D = list_of_images[0].reshape(1, r, c);
    
        for l in range(1,len(list_of_images)):
            original3D = np.vstack((
                original3D,list_of_images[l].reshape(1, r, c)));
            
        return(original3D)
        
        
    def __make_4D__(self, list_of_images : list) -> np.ndarray :
        
        """
        A function which vertically stack a list of 2D **RBG** images into a single
        3D object.
        
        Parameters
        ----------
        list_of_images : list
            A list of np.ndarrays contaings 2D RBG images.

        Returns
        -------
        A 3D object i.e and np.ndarray of stacked 2D RBG images
        """

        
        r, c = list_of_images[0].shape[:2];
        original4D = list_of_images[0].reshape(1, r, c, self.n_channels);
    
        for l in range(1,len(list_of_images)):
            original4D = np.vstack((
                original4D,list_of_images[l].reshape(1, r, c, self.n_channels)));
            
        return(original4D)
        
    
    def __image_normalisation__(self, image : np.ndarray,
                                      minimum_b : np.float, 
                                      maximum_b : np.float) -> np.ndarray :
        
        """
        A function which normalised the voxels of a 3D object to be between 
        0 and 1. The method used in this function is the min max scaler which 
        is calculated as (x - minimum(x))/(maximum(x) - minimum(x)). 
        
        Parameters
        ----------
        image : np.ndarray
            A 3D object in the form of a numpy array
        min_b : Float specifying the value of the minimum voxel.
        The default is 0.
        max_b : Integer specifying the value of the maximum voxel.
        The default is 255.

        Returns
        -------
        A numpy array containing the normalised values of the 3D object (image). 
        """
        
        return((image - minimum_b)/(maximum_b - minimum_b))
    
    
    def __rotate__(self,image : np.ndarray, 
                       angle : np.float, 
                       center : np.float =None, 
                       scale :np.float =1.0) -> np.ndarray:
        
        """
        A function which rotates a 2D image by a specified angle. 
        
        Parameters
        ----------
        image : np.ndarray
            A 2D numpy array.
        angle : np.float
        center : np.float, optional
        scale : np.float, optional

        Returns
        -------
        Returns the 2D images rotated by the specified angle.
        """
        
        # grab the dimensions of the image
        (h, w) = image.shape[:2];
    
        # if the center is None, initialize it as the center of
        # the image
        if center is None:
            center = (w // 2, h // 2);
    
        # perform the rotation
        M = cv2.getRotationMatrix2D(center, angle, scale);
        rotated = cv2.warpAffine(image, M, (w, h));
    
        # return the rotated image
        return (rotated)
    
    
    def __add_noise__(self, image : np.ndarray, 
                      noise: np.ndarray) -> np.ndarray:
        
        """
        A function which adds the given noise into a 2D image.
        
        Parameters
        ----------
        image : np.ndarray
            A 2D image.
        noise : np.ndarray
            A numpy array with the same shape as the 2D image containing noise.

        Returns
        -------
        The 2D image with the added noise.
        """
        
        noisy_image = np.zeros(image.shape, np.float32);
        try:
            noisy_image = image + noise;
        except ValueError:
            print("image dim = {}\n gaussian_noise = {}.".format(
                image.shape,noise.shape));
        return(noisy_image)
    
    
    def on_epoch_end(self) -> None :
        
        """
        Updates indexes after each epoch.
 
        """

        self.indexes = np.arange(len(self.IDlist));
        if self.shuffle == True:
            np.random.shuffle(self.indexes);
    
    
    def __data_collection(self,
                          IDlist_temp : list)  -> tuple([np.ndarray,np.ndarray]):
        
        """
        A function which returns a matrix containg the data in a single batch 
        along with their labels. The function accepts a list of strings representing
        the unique IDs of the 3D objects along with the path of the folder the objects
        belong to. No transformation occurs in this function. The labels returned
        are one hot encoded.
    
        Parameters
        ----------
        IDlist_temp : list
           A list including the unique names of each 3D object which belong to 
           the specified path for a single batch. 
        path: str
          
        Returns
        -------
        Generates a numpy ndarray containing the 3D objects for a single batch
        along with their one hot encoded labels.

        """
        
        # Initialization
        X = np.empty((self.batch_size, *self.dim, self.n_channels));
        y = np.empty((self.batch_size), dtype=int);
        
        # Generate data
        for i, ID in enumerate(IDlist_temp):
        
            
            X[i,] = np.load(self.path + '/' + ID + '.npy').reshape(
                    (self.dim[0], self.dim[1], self.dim[2], self.n_channels));
            y[i] = self.labels[ID];
            
            

        return (X, tf.keras.utils.to_categorical(y, num_classes=self.n_classes))
    
    
    def __data_transormation__(self, X):
        
        """
        A function which accepts the batch of the original 3D objects and applies
        a series of transformation during each epoch. The available transorfmations
        to this day are: Normalisation, Add of gaussian noise and rotation. 
    
        Parameters
        ----------
        X : np.ndarray
           The array containg the 3D objects of the current batch drawn by the
           __data_collection function. 
           
          
        Returns
        -------
        Generates a numpy ndarray containing the tranformed 3D objects for a 
        single batch.
        """
        
        #Initialization
        X_transformed = np.empty((X.shape[0], *self.dim, self.n_channels));
        
       
        
        for i in range(int(X.shape[0])):
            
            if self.n_channels == 1:
                
                X_temp = np.empty((self.dim[0], self.dim[1], self.dim[2]))
                X_temp = X[i].reshape((self.dim[0], self.dim[1], self.dim[2]))
                
                
                #generates noise from gaussian with mean and variance specified 
                #by user. The same random noise is applied to all the slices of the
                #3D object. 
                noise = np.random.normal(self.noise_mean, 
                                     self.noise_std, 
                    (int(self.dim[1]),int(self.dim[2])));
                
            else:
                X_temp = np.empty((self.dim[0], 
                                   self.dim[1],
                                   self.dim[2],
                                   self.n_channels))
                X_temp = X[i]
                
                noise = np.random.normal(self.noise_mean, 
                                         self.noise_std, 
                   (int(self.dim[1]),int(self.dim[2]), self.n_channels));
                
            
            #generates angle from gaussian with mean 0 and variance specified 
            #by user. The same random angle is applied to all the slices of the
            #3D object.
            angle = np.random.normal(0,self.rotate_std);
            
            
                                     
            #When noise is applied and normalisation is required then the min and 
            #max bounds of the normalisation are modified accordingly so the 
            #normalised values are still between 0 and 1.
            min_noise = noise.min() +  self.min_bound;
            max_noise = noise.max() + self.max_bound;
            result = []
            for height in range(int(self.dim[0])):
            
                                
                #Case_1 : Rotation -> Noise -> Normalisation        
                if self.rotation and self.gaussian_noise and self.normalisation:

                         X_temp[height] = self.__rotate__(X_temp[height], 
                               angle = angle)
                         X_temp[height] = self.__add_noise__(X_temp[height],
                               noise = noise)
                         result.append(self.__image_normalisation__(
                                 X_temp[height], minimum_b =  min_noise,
                                 maximum_b = max_noise))
                         
                #Case_2 : Rotation -> Normalisation
                if self.rotation and not self.gaussian_noise and self.normalisation:
                    
                         X_temp[height] = self.__rotate__(X_temp[height], 
                               angle = angle)
                         result.append(self.__image_normalisation__(
                                 X_temp[height], minimum_b =  self.min_bound,
                                 maximum_b = self.max_bound))
        
                #Case_3 : Rotation -> Noise
                if self.rotation and self.gaussian_noise and not self.normalisation:
          
                         X_temp[height] = self.__rotate__(X_temp[height], 
                               angle = angle)
                         result.append(self.__add_noise__(X_temp[height],
                                                          noise = noise))
        
                #Case_4 : Noise -> Normalisation
                if not self.rotation and self.gaussian_noise and self.normalisation:
                    
                         X_temp[height] = self.__add_noise__(X_temp[height], 
                               noise = noise)
                         result.append(self.__image_normalisation__(
                                 X_temp[height], minimum_b =  min_noise,
                                 maximum_b = max_noise))
                #Case_5 : Noise
                if not self.rotation and self.gaussian_noise and not self.normalisation:
                         
                         result.append(self.__add_noise__(X_temp[height], noise))
                         
                #Case_6 : Rotation
                if  self.rotation and not self.gaussian_noise and not self.normalisation:
                         
                         result.append(self.__rotate__(X_temp[height], angle))
                
                #Case_7 : Normalisation
                if  not self.rotation and not self.gaussian_noise and self.normalisation:
        
                         result.append(self.__image_normalisation__(
                                 X_temp[height],self.min_bound,self.max_bound))
            
            
            if self.n_channels == 1:
            
                X_transformed[i,] = self.__make_3D__(result).reshape(int(self.dim[0]) 
                    ,int(self.dim[1]),int(self.dim[2]) , self.n_channels);
            else:
                
                X_transformed[i,] = self.__make_4D__(result);
                
            
        return (X_transformed)
