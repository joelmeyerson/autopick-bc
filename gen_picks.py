import os
import shutil
import argparse
import numpy as np
import matplotlib.pyplot as plt

# TF imports
import tensorflow as tf
from tensorflow import keras

# local imports
# import extract_star_meta
# import extract_particles
# import save_results

# disable GPU
# os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
# os.environ["CUDA_VISIBLE_DEVICES"] = ""

def gen_picks():
    
    # parse arguments
    parser = argparse.ArgumentParser(description='Use model to classify extracted images as good (true particles) or bad (empty area, ice contamination, gold/carbon film).')
    parser.add_argument('-p', '--projpath', type=str, help='path for project', required=True)
    parser.add_argument('-i', '--imagestar', type=str, help='particle star file containing inputs for classification using model', required=True)
    args = parser.parse_args()
    
    # format project path 
    work_dir = args.projpath
    if work_dir.endswith('/'):
        work_dir = work_dir.rstrip('/')
    
    # check that sub-directory exists
    data_dir = work_dir + '/ClassBin'
    if os.path.exists(data_dir) == False:
        "No ClassBin directory found."
        exit()
    
    # check that model exists
    if os.path.exists(data_dir + '/model.h5') == False:
        "No model file found (h5)."
        exit()
        
    # extract box size, box apix, and original image apix
    meta = extract_star_meta.extract(args.inputstar)
    print("Box size: " + meta[0] + ", Box apix: " + meta[1] + ", Image apix: " + meta[2])
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
    test_ds = test_ds.prefetch(buffer_size=32)

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
        metrics = [
            keras.metrics.BinaryAccuracy(name='accuracy'),
            keras.metrics.TruePositives(name='tp'),
            keras.metrics.FalsePositives(name='fp'),
            keras.metrics.TrueNegatives(name='tn'),
            keras.metrics.FalseNegatives(name='fn'),
            keras.metrics.Precision(name='precision'),
            keras.metrics.Recall(name='recall'),
            keras.metrics.AUC(name='auc')
        ]
    )
    
    # callbacks
    checkpoint_filepath = '/tmp/checkpoint'
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=2),
        #keras.callbacks.ModelCheckpoint(checkpoint_filepath, save_freq='epoch'),
        #keras.callbacks.TensorBoard(log_dir='./logs')
    ]

    if args.skiptraining != True:
        # fit model to data and save training and validation history
        history = model.fit( 
            train_ds,
            epochs=50,
            callbacks=callbacks,
            validation_data=val_ds
        )
        model.save(data_dir + '/model.h5')
        
        # save log of error and accuracy per epoch (dict)
        with open(data_dir + "/training_log.txt", "w") as text_file:
            text_file.write(json.dumps(history.history))
    else:
        try:
            model = tf.keras.models.load_model(data_dir + '/model.h5')
        except:
            print("No model found. Must run training.")
    
    # run prediction with test data
    # each batch in the test dataset is a tuple with two elements
    # element 0 is tuple with (batch_size, box, box, 1)
    # element 1 is tuple with (batch_size, 1)
    batch_labels = []
    batch_data = []
    labels = []
    predictions = []
    for batch in test_ds:
        #print(len(batch))
        batch_data = batch[0]
        batch_labels = batch[1]
        batch_labels = np.array(batch_labels[:,0]) # convert tuple to array
        #print(np.shape(batch_data))
        #print(np.shape(batch_labels))

        batch_pred = model.predict(batch_data)
        batch_pred = abs(batch_pred.round())
        batch_pred = np.array(batch_pred[:,0]) # convert tuple to array
        
        # store batch labels and batch predictions
        labels = np.concatenate([labels, batch_labels])
        labels = labels.astype(int)
        predictions = np.concatenate([predictions, batch_pred])
        predictions = predictions.astype(int)
        
        # save log of labels and predictions
        with open(data_dir + "/testing_results.txt", "w") as text_file:
            text_file.write(json.dumps({"labels":labels.tolist(), "predictions":predictions.tolist()}))
        
    # make summary of training and test results (png)
    save_results.make_summary(data_dir, history.history, labels, predictions)

if __name__ == "__main__":
   gen_picks()