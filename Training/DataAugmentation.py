import tensorflow as tf
import numpy as np
import imgaug.augmenters as iaa
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

def preprocessing(sample):
    image = sample["image"]
    #print(image.shape)
    image = tf.cast(tf.image.resize(image,size=[224,224]),tf.uint8)#/255
    sample["image"]=image
    
    return (image,sample["label"])

def preprocessing_val(sample):
    image = sample["image"]
    #print(image.shape)
    image = tf.cast(tf.image.resize(image,size=[224,224]),tf.float32)/255
    sample["image"]=image
    
    return (image,sample["label"])

def da_policy(image,label):

    img_size = 224

    #image = sample[0]
    policy = np.random.randint(4)
    
    #policy = 2
    if policy == 0:
        
        p = np.random.random()
        if p<=0.6:
            aug = iaa.TranslateX(px=(-60, 60),cval=128)
            image = aug(image=image)

    
        p = np.random.random()
        if p<=0.8:
            aug = iaa.HistogramEqualization()
            image = aug(image=image)

    
    elif policy==1:
        
        p=np.random.random()
        if p<=0.2:
            aug = iaa.TranslateY(px=(int(-0.18*img_size), int(0.18*img_size)),cval=128)
            image = aug(image=image)
        
        p=np.random.random()
        if p<=0.8:
            square_size = np.random.randint(48)
            aug = iaa.Cutout(nb_iterations=1, size=square_size/img_size, squared=True)
            image = aug(image=image)

            
    elif policy==2:
        p=np.random.random()
        if p<=1:
            aug = iaa.ShearY(shear=(int(-0.06*img_size), int(0.06*img_size)), order=1, cval=128)
            image = aug(image=image)
            
        p=np.random.random()
        if p<=0.6:
            aug = iaa.TranslateX(px=(-60, 60),cval=128)
            image = aug(image=image)
            
    elif policy==3:
        p=np.random.random()
        if p<=0.6:    
            aug = iaa.Rotate(rotate=(-30, 30), order=1, cval=128)
            image = aug(image=image)

        
        p=np.random.random()
        if p<=1:
            aug = iaa.MultiplySaturation((0.54, 1.54))
            image = aug(image=image)
    
    #Para EFFICIENTNET NO es necesario NORMALIZAR        
    return (tf.cast(image,tf.float32),tf.cast(label,tf.int64))

@tf.function(input_signature=[tf.TensorSpec((224,224,3), tf.uint8),tf.TensorSpec((), tf.int64)]) 
def augmentations(image,label):
    '''
    Despues de pasar el tf.numpy_function es necesario definir manualmente el tamaño de "image" y "label"
    '''
    
    
    #boxes_shape = bboxes.shape
    im_shape = image.shape    
    label_shape = label.shape
    
    image,label = tf.numpy_function(da_policy,[image,label],Tout =[tf.float32,tf.int64])

    image.set_shape(im_shape)
    label.set_shape(label_shape)
    #bboxes.set_shape(boxes_shape)
    
    return image,label