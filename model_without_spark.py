
import os 
import zipfile 
import tensorflow as tf 
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras import layers 
from tensorflow.keras import Model 
import matplotlib.pyplot as plt

base_dir = '/home/dsp_kass/data/faces'
train_dir = os.path.join(base_dir, 'train')
validation_dir = os.path.join(base_dir, 'val')
test_a_dir = os.path.join(base_dir, 'testa')
test_b_dir = os.path.join(base_dir, 'testb')
test_c_dir = os.path.join(base_dir, 'testc')

# Directory with our training zero pictures
train_zeros_dir = os.path.join(train_dir, '0')

# Directory with our training one pictures
train_ones_dir = os.path.join(train_dir, '1')

# Directory with our validation zero pictures
validation_zeros_dir = os.path.join(validation_dir, '0')

# Directory with our validation one pictures
validation_ones_dir = os.path.join(validation_dir, '1')


# Set up matplotlib fig, and size it to fit 4x4 pics
import matplotlib.image as mpimg
nrows = 4
ncols = 4

fig = plt.gcf()
fig.set_size_inches(ncols*4, nrows*4)
pic_index = 100
train_zero_fnames = os.listdir( train_zeros_dir )
train_one_fnames = os.listdir( train_ones_dir )


next_zero_pix = [os.path.join(train_zeros_dir, fname) 
                for fname in train_zero_fnames[ pic_index-8:pic_index] 
               ]

next_one_pix = [os.path.join(train_ones_dir, fname) 
                for fname in train_one_fnames[ pic_index-8:pic_index]
               ]

for i, img_path in enumerate(next_zero_pix+next_one_pix):
  # Set up subplot; subplot indices start at 1
  sp = plt.subplot(nrows, ncols, i + 1)
  sp.axis('Off') # Don't show axes (or gridlines)

  img = mpimg.imread(img_path)
  plt.imshow(img)

plt.show()


# Add our data-augmentation parameters to ImageDataGenerator
train_datagen = ImageDataGenerator(rescale = 1./255., rotation_range = 10, horizontal_flip = True)

# Note that the validation data should not be augmented!
test_datagen = ImageDataGenerator( rescale = 1.0/255. )


train_generator = train_datagen.flow_from_directory(train_dir, batch_size = 1000, class_mode = 'binary', target_size = (150, 150))

# Flow validation images in batches of 20 using test_datagen generator
validation_generator = test_datagen.flow_from_directory( validation_dir,  batch_size = 1000, class_mode = 'binary', target_size = (150, 150))

test_a_generator = test_datagen.flow_from_directory(test_a_dir, shuffle=False, batch_size = 1, class_mode=None, target_size = (150, 150))
test_b_generator = test_datagen.flow_from_directory(test_b_dir, shuffle=False, batch_size = 1, class_mode=None, target_size = (150, 150))
test_c_generator = test_datagen.flow_from_directory(test_c_dir, shuffle=False, batch_size = 1, class_mode=None, target_size = (150, 150))




from tensorflow.keras.applications.inception_v3 import InceptionV3
base_model = InceptionV3(input_shape = (150, 150, 3), include_top = False, weights = 'imagenet')


for layer in base_model.layers:
    layer.trainable = False


from tensorflow.keras.optimizers import RMSprop

x = layers.Flatten()(base_model.output)
x = layers.Dense(1024, activation='relu')(x)
x = layers.Dropout(0.5)(x)
x = layers.Dense(1, activation='sigmoid')(x)

model = tf.keras.models.Model(base_model.input, x)

model.compile(optimizer = RMSprop(lr=0.0001), loss = 'binary_crossentropy', metrics = ['acc'])


inc_history = model.fit(train_generator, validation_data = validation_generator, steps_per_epoch = 534, epochs = 3)


def predict_and_save(model, generator, file_name):
    STEP_SIZE_TEST=generator.n//generator.batch_size
    generator.reset()
    pred=model.predict(generator,
    steps=STEP_SIZE_TEST,
    verbose=1)

    pred_array = pred.squeeze()

    pred_to_int = [1 if x > 0.5 else 0 for x in pred_array]

    import numpy as np
    np_array = np.array(pred_to_int).astype(int)
    filenames = np.array(generator.filenames)
    np.savetxt(f'/home/dsp_kass/data/faces/outputs/{file_name}', np_array)
    np.savetxt(f'/home/dsp_kass/data/faces/outputs/files_{file_name}', filenames, fmt="%s")

predict_and_save(model, test_a_generator, "out_a.txt")
predict_and_save(model, test_b_generator, "out_b.txt")
predict_and_save(model, test_c_generator, "out_c.txt")
