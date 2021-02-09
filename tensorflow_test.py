import pandas as pd, numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
import random
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, accuracy_score

import cv2
import os
import math
from math import ceil, floor, log

# to be used to get better performance
# from sklearn.model_selection import KFold
import efficientnet.tfkeras as efn

import scikitplot as skplt
print("TF version:", tf.__version__)
print(tf.config.list_physical_devices('GPU'))

N_epochs = 30
SEED = 1970
N_TTA = 5
# in the version 9 of the botebook I made images size = 300 for all models
MODELS = {
          'ResNet50':[tf.keras.applications.ResNet50,32,300]
         }
path = './'
path_org = './' #path to the main dataset
df = pd.read_csv(path + path_org + "labels.csv")
df.head(5)

df['path'] = df.path.apply(lambda x: x.replace('\\', '/') )
df['path'] = df.path.apply(lambda x: path_org + x )

labels_cols = ['is_bee', 'is_wasp', 'is_otherinsect', 'is_other']
df_train = df.loc[(df.is_validation == 0) & (df.is_final_validation == 0)]
df_valid = df.loc[(df.is_validation == 1)]
df_test = df.loc[(df.is_final_validation == 1)]

df_test_ext = pd.concat([df_test])

y_train = df_train.loc[:,['id']+labels_cols]
y_train.set_index('id', inplace = True)
y_valid = df_valid.loc[:,['id']+labels_cols]
y_valid.set_index('id', inplace = True)
y_test = df_test.loc[:,['id']+labels_cols]
y_test.set_index('id', inplace = True)
y_test_ext = df_test_ext.loc[:,['id']+labels_cols]
y_test_ext.set_index('id', inplace = True)

print(df_train.head(5))

# check that we've got extened test set labels 
print(y_test_ext.head(-5))

def img_plot(df_list, n_samples):
    df = df_list.sample(n = n_samples, random_state = SEED)
    images = []
    f, ax = plt.subplots(n_samples//5, 5, figsize=(12,8))
    i = 0
    for img_path in df['path']:
        img = cv2.imread(path+img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        images.append(img)
        ax[i//5, i%5].imshow(img)
        ax[i//5, i%5].axis('off')
        ax[i//5, i%5].set_title('label: %s' % df.iloc[i]['label'])
        i +=1
    plt.show()

img_plot(df,25)

def get_lr_callback(batch_size = 16, plot=False):
    start_lr = 0.001
    def step_decay(epoch):
        drop = 0.5
        epochs_drop = 5.0
        lr = start_lr * math.pow(drop, math.floor((1+epoch)/epochs_drop))
        return lr
    
    lr_callback = tf.keras.callbacks.LearningRateScheduler(step_decay)
    if plot == True:
        rng = [i for i in range(N_epochs)]
        y = [step_decay(x) for x in rng]
        plt.plot(rng, y)
        plt.xlabel('epoch', size=14)
        plt.ylabel('learning_rate', size=14)
        plt.title('Training Schedule', size=16)
        plt.show()
        
    return lr_callback


es_callback = tf.keras.callbacks.EarlyStopping(patience=10, 
                                               monitor='val_loss',
                                               verbose=1, 
                                               restore_best_weights=True)
lr = get_lr_callback(plot=True)


def gen_init(BS, IMG_Size):
    train_gen = tf.keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=30,
        width_shift_range=0.1,
        height_shift_range=0.1,
        vertical_flip = True,
        horizontal_flip=True)

    valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)    
    train_generator = train_gen.flow_from_dataframe(dataframe=df_train, directory=path, 
                                                  x_col="path", y_col=labels_cols, 
                                                  class_mode="raw", 
                                                  target_size=(IMG_Size,IMG_Size), batch_size = BS)

    valid_generator = valid_gen.flow_from_dataframe(dataframe=df_valid, directory=path, 
                                                  x_col="path", y_col=labels_cols, 
                                                  class_mode="raw", 
                                                  # class_mode="categorical", 
                                                  target_size=(IMG_Size,IMG_Size), batch_size = BS)
    test_generator = valid_gen.flow_from_dataframe(dataframe=df_test, directory=path, 
                                                  x_col="path", y_col=labels_cols, 
                                                  class_mode="raw", 
                                                  shuffle = False,
                                                  target_size=(IMG_Size,IMG_Size), batch_size = BS)
    return   train_generator,   valid_generator, test_generator

train_generator, valid_generator, test_generator = gen_init(32,200)

ti, tl = train_generator.next()
imgs = []
for i in range(ti.shape[0]):
    img = np.array(ti[i]*255, dtype = 'int32')
    imgs.append(img)

f, ax = plt.subplots(4, 8, figsize=(15,10))
for i, img in enumerate(imgs):
    ax[i//8, i%8].imshow(img)
    ax[i//8, i%8].axis('off')
    ax[i//8, i%8].set_title('label: %s' % tl[i])
plt.show()


def build_model(model_engine, IMG_Size):
    inp = tf.keras.layers.Input(shape=(IMG_Size,IMG_Size,3))
    base = model_engine(input_shape=(IMG_Size,IMG_Size,3),weights='imagenet',include_top=False)
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    x = tf.keras.layers.Dropout(0.2)(x)
    x = tf.keras.layers.Dense(4,activation='softmax')(x)
    model = tf.keras.Model(inputs=inp,outputs=x)
    opt = tf.keras.optimizers.Adam(learning_rate = 0.001)   
    model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['accuracy'])
    return model

all_model = []
all_history = []
all_preds = []
all_accuracies = []
all_confusion_matrices = []

for model_name in MODELS:
    engine = MODELS[model_name][0]
    BS = MODELS[model_name][1]
    IMG_Size = MODELS[model_name][2]
    train_generator, valid_generator, test_generator = gen_init(BS, IMG_Size)

    model = tf.keras.models.load_model('models/model-ResNet50.h5')

    '''
    model = build_model(engine, IMG_Size)
    print('------------------------------------------------------------------')
    print('Training model ', model_name)
    history = model.fit(train_generator,
              steps_per_epoch=len(df_train) / BS, epochs = N_epochs, verbose = 1,
              callbacks=[es_callback, get_lr_callback(BS)],
              validation_data = valid_generator)

    model.save('model-%s.h5'%model_name)  
    all_history.append(history)
    
    pd.DataFrame(history.history)[['accuracy', 'val_accuracy']].plot()
    pd.DataFrame(history.history)[['loss', 'val_loss']].plot()
    plt.show()
    '''

    '''
    preds = model.predict(test_generator, verbose = 1)
    all_preds.append(preds)
    cm = confusion_matrix(np.argmax(np.array(y_test), axis=1), np.argmax(preds, axis = 1))
    all_confusion_matrices.append(cm)
    acc = accuracy_score(np.argmax(np.array(y_test), axis=1), np.argmax(preds, axis = 1))    
    all_accuracies.append(acc)
    print('------------------------------------------------------------------')
    print(cm)
    print(acc)
    '''
    valid_gen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)
    print(df_test)
    path = "./test"
    test_generator = valid_gen.flow_from_directory(path, 
                                                    shuffle = False,
                                                    target_size=(IMG_Size,IMG_Size), batch_size = BS)
    '''
    test_generator = valid_gen.flow_from_dataframe(dataframe=df_test, directory=path, 
                                                    x_col="path", y_col=labels_cols, 
                                                    class_mode="raw", 
                                                    shuffle = False,
                                                    target_size=(IMG_Size,IMG_Size), batch_size = BS)
    '''
    preds = model.predict(test_generator, verbose = 1)
    files = []
    res = []
    for f in test_generator.filenames:
        files.append(f[5:])
    
    for output in preds:
        if output[0] == max(output):
            res.append('negative')
        elif output[1] == max(output):
            #j = random.randrange(0,5)
            #if j==1:
            if output[1] > 0.8:
                res.append('positive')
            else:
                res.append('negative')
        else:
            res.append('negative')
    #print(files)
    #print(preds)
    #print(res)
    output = open('res+.csv','a+')
    output.write('filename,res\n')
    print(len(res))
    for i in range(0,len(res)):
        output.write(f'{files[i]},{res[i]}\n')
    output.close()
    
