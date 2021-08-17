from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.python.keras.layers.core import Dropout
from tensorflow.python.keras.layers.pooling import GlobalAveragePooling2D
from tensorflow.python.training.tracking import base
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Input, Add, Flatten, Activation, Conv2D, MaxPooling2D, BatchNormalization, ZeroPadding2D, GlobalAveragePooling2D, Convolution2D
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping
from tensorflow.keras.applications.vgg16 import preprocess_input
from tensorflow.keras.preprocessing.image import ImageDataGenerator 


class vgg16_model():

    base_model = None
    vgg = None

    def __init__(self) -> None:
        self.base_model = VGG16(weights='imagenet', include_top=False,input_tensor=Input(shape=(224,224,3)))

    def show(self):
        self.base_model.summary()
    

    def fine_tune(self):
        for layer in self.base_model.layers:
            layer.trainable = False
        x = self.base_model.get_layer('block5_pool').output

        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        x=BatchNormalization()(x)
        x = Dense(1024, activation='relu')(x)
        x=BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x=BatchNormalization()(x)
        x = Dense(14, activation='softmax')(x)

        self.vgg = Model(inputs=self.base_model.input, outputs=x)
        opt = Adam(learning_rate=0.001)
        
        self.vgg.compile(loss='categorical_crossentropy',
                    optimizer=opt,
                    metrics=['accuracy'])
        self.vgg.summary()
        #self.vgg.load_weights('Save_model/model.h5')
    
    def train(self,x_train,y_train,x_test,y_test):

        #data generator
        # augmentation cho training data
        aug_train = ImageDataGenerator(rescale=1./255, rotation_range=30, width_shift_range=0.1, height_shift_range=0.1, shear_range=0.2, 
                                zoom_range=0.2, horizontal_flip=True, fill_mode='nearest')
        # augementation cho test
        aug_test= ImageDataGenerator(rescale=1./255)


        x_train=preprocess_input(x_train)
        x_test=preprocess_input(x_test)

        #make folder save model
        import os
        save_dir = 'Save_model/'
        model_name = 'model.h5'
        if not os.path.isdir(save_dir):
            os.makedirs(save_dir)
        filepath = os.path.join(save_dir, model_name)

        EarlyStoppingearlystop = EarlyStopping(monitor = 'val_loss',
                          min_delta = 0,
                          patience = 3,
                          verbose = 1,
                          restore_best_weights = True)

        Checkpoint = ModelCheckpoint(filepath=filepath,
                             monitor='val_loss',
                             verbose=1,
                             save_best_only=True)
        callback=[EarlyStoppingearlystop,Checkpoint]

        H = self.vgg.fit_generator(aug_train.flow(x_train, y_train, batch_size=32), 
                        steps_per_epoch=len(x_train)//32,
                        validation_data=(aug_test.flow(x_test, y_test, batch_size=32)),
                        validation_steps=len(x_test)//32,
                        callbacks=callback,
                        shuffle=True,
                        epochs=10)
        #self.vgg.save('Save_model/model.h5')
        return H


