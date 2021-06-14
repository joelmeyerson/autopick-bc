import os
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt
import random

# TF imports
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, BatchNormalization
from tensorflow.keras.layers import Activation, Flatten, Dense, Dropout

# local imports
import extract_star_meta
import extract_particles

# disable GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

def gen_model():
    
    # reset Keras global state
    tf.keras.backend.clear_session()
    
    # set random seed
    seed = 123
    #os.environ['PYTHONHASHSEED']=str(seed)
    #random.seed(seed)
    #np.random.seed(seed)
    #tf.random.set_seed(seed)
    
    # parse arguments
    parser = argparse.ArgumentParser(description='Classify good and bad particles.')
    parser.add_argument('-p', '--projpath', type=str, help='path for project', required=True)
    parser.add_argument('-g', '--stargood', type=str, help='input star file with good particles', required=True)
    parser.add_argument('-b', '--starbad', type=str, help='input star file with bad particles', required=True)
    parser.add_argument('-c', '--cleardata', action='store_true', help='clear training/test data')
    parser.add_argument('-s', '--skiptraining', action='store_true', help='skip training and just run inference')
    args = parser.parse_args()
    
    # create working directory
    work_dir = args.projpath
    data_dir = args.projpath + '/ClassBin'
    train_dir = args.projpath + '/ClassBin/train'
    test_dir = args.projpath + '/ClassBin/test'
    if os.path.exists(data_dir) == False:
        os.mkdir(data_dir)
    
    # extract box size, box apix, and original image apix
    meta_good = extract_star_meta.extract(args.stargood)
    meta_bad = extract_star_meta.extract(args.starbad)
    if meta_good != meta_bad:
        print("Headers for good and bad particle star files do not match. Exiting.")
        exit()
    else:
        print("Box size: " + meta_good[0] + ", Box apix: " + meta_good[1] + ", Image apix: " + meta_good[2])
        box = int(meta_good[0])
        b_apix = float(meta_good[1])
        i_apix = float(meta_good[2])
    
    # create directories to store training/test data
    if (len(os.listdir(data_dir)) == 0 or args.cleardata == True):
        
        # FOR DEBUGGING
        if args.cleardata == True:
            print('Clearing particle cache...', end="")
            shutil.rmtree(data_dir)
            os.mkdir(data_dir)
            print('Done.')
        
        os.mkdir(train_dir)
        os.mkdir(test_dir)
        good_train_dir = train_dir + '/good'
        os.mkdir(good_train_dir)
        
        bad_train_dir = train_dir + '/bad'
        os.mkdir(bad_train_dir)
        
        good_test_dir = test_dir + '/good'
        os.mkdir(good_test_dir)
        
        bad_test_dir = test_dir + '/bad'
        os.mkdir(bad_test_dir)
        
        # extract good and bad particle data
        extract_particles.extract(args.projpath, args.stargood, good_train_dir, good_test_dir, 'good')
        extract_particles.extract(args.projpath, args.starbad, bad_train_dir, bad_test_dir, 'bad')
        
    else:  # FOR DEBUGGING
        "Particle cache not cleared."
        
    # loader parameters
    batch_size = 32
    image_size = (box, box)
    class_names=['bad', 'good']
    train_val_split = 0.2

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=train_val_split,
        class_names=class_names,
        label_mode='binary',
        subset='training',
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        color_mode='grayscale'
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        validation_split=train_val_split,
        class_names=class_names,
        label_mode='binary',
        subset='validation',
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        color_mode='grayscale'
    )
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        class_names=class_names,
        label_mode='binary',
        seed=seed,
        image_size=image_size,
        batch_size=batch_size,
        color_mode='grayscale'
    )

    train_ds = train_ds.prefetch(buffer_size=32)
    val_ds = val_ds.prefetch(buffer_size=32)
    test_ds = val_ds.prefetch(buffer_size=32)

    # build model
    model = Sequential()
    
    model.add(tf.keras.layers.experimental.preprocessing.RandomRotation(0.2, seed=seed)) # augmentation
    #model.add(tf.keras.layers.experimental.preprocessing.CenterCrop(height=round(0.8*box), width=round(0.8*box))) # augmentation
    #model.add(tf.keras.layers.experimental.preprocessing.RandomTranslation(height_factor=(0.1), width_factor=(0.1), seed=seed)) # augmentation
    
    model.add(Conv2D(
        filters=32,
        kernel_size=(2,2),
        strides=(1,1),
        padding='same',
        input_shape=(box,box,1)
    ))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=2))
    model.add(Conv2D(filters=64,kernel_size=(2,2),strides=(1,1),padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=2))
    model.add(Conv2D(filters=128,kernel_size=(2,2),strides=(1,1),padding='valid'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    
    # model.add(MaxPooling2D(pool_size=(2,2),strides=2))
#     model.add(Conv2D(filters=128,kernel_size=(2,2),strides=(1,1),padding='valid'))
#     model.add(BatchNormalization())
#     model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2,2),strides=2))
    model.add(Flatten())
    model.add(Dense(32))

    model.add(Activation('relu'))
    model.add(Dropout(0.25))
    model.add(Dense(1))
    model.add(Activation('sigmoid'))

    model.compile(
        loss='binary_crossentropy',
        optimizer=keras.optimizers.Adam(0.00005),
        metrics=['accuracy']
    )
    
    # callbacks
    checkpoint_filepath = '/tmp/checkpoint'
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
        #keras.callbacks.ModelCheckpoint(checkpoint_filepath, save_freq='epoch'),
        #keras.callbacks.TensorBoard(log_dir='./logs')
    ]

    if args.skiptraining != True:
        model.fit(
            train_ds,
            epochs=50,
            callbacks=callbacks,
            validation_data=val_ds
        )
        model.save(data_dir + '/autopick-bc-model.h5')
    else:
        model = tf.keras.models.load_model(data_dir + '/autopick-bc-model.h5')
    
    target_dir = test_dir + '/bad'
    files = os.listdir(target_dir)
    count = 0
    for f in files:
        img = keras.preprocessing.image.load_img(
            target_dir + '/' + f, target_size=image_size, color_mode="grayscale"
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)
        score = predictions[0]
        score_rnd = int(score.round())
        if score_rnd == 0:
            count = count + 1
         
    file_bad = len(files)   
    bad_correct = count
    
    target_dir = test_dir + '/good'
    files = os.listdir(target_dir)
    count = 0
    for f in files:
        img = keras.preprocessing.image.load_img(
            target_dir + '/' + f, target_size=image_size, color_mode="grayscale"
        )
        img_array = keras.preprocessing.image.img_to_array(img)
        img_array = tf.expand_dims(img_array, 0)  # Create batch axis

        predictions = model.predict(img_array)
        score = predictions[0]
        score_rnd = int(score.round())
        if score_rnd == 1:
            count = count + 1

    files_total = file_bad + len(files)   
    total_correct = bad_correct + count
    
    print(total_correct)
    print(files_total)
    print((total_correct/files_total)*100)
    
        #print(f + " = " + str(score.round()))
        # print(
#             "This image is %.2f percent cat and %.2f percent dog."
#             % (100 * (1 - score), 100 * score)
#         )
        
    # # run inference
#     predictions = model.predict_classes(
#         test_ds,
#         batch_size=batch_size,
#         verbose=1
#     )
#     #predictions = predictions.round()
#     labels = np.concatenate([y for x, y in test_ds], axis=0)
#     labels = np.reshape(labels,(-1,1)) # add an index to the array
#
#     results = np.hstack((predictions, labels))
#     print(sum(abs(results[:,0]-results[:,1])))
  
if __name__ == "__main__":
   gen_model()