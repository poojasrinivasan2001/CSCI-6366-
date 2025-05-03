"""# Loading all the dependencies"""

import matplotlib.pyplot as plt
import time
import tensorflow as tf
from tqdm import tqdm
from glob import glob
from PIL import Image
import nltk
import string
import warnings
import os
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.applications.inception_v3 import preprocess_input

warnings.filterwarnings("ignore")
config = ConfigProto()
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

tf.config.experimental.list_physical_devices()

"""# Defining the constants"""

IMG_H = 299
IMG_W = 299

BATCH_SIZE = 100
BUFFER_SIZE = 1000

"""# Defining img file paths"""

IMG_FILE_PATH = "asset1/Flickr8k_dataset"

"""# Defining txt file paths"""

# txt file contaning the unprocessed captions
CAP_TEXT_PATH = r"asset1/Flickr8k_text/Flickr8k.token.txt"
# txt file containing the names of the train imgs
TRAIN_TXT_PATH = r"asset1/Flickr8k_text/Flickr_8k.trainImages.txt"
# txt file containing the names of the test imgs
TEST_TXT_PATH = r"asset1/Flickr8k_text/Flickr_8k.testImages.txt"
# txt file containing the names of the validation imgs
VAL_TXT_PATH = r"asset1/Flickr8k_text/Flickr_8k.devImages.txt"

# Text file path
TXT_PATH = r"asset1/Flickr8k_text/Flickr_8k.trainImages.txt"

"""# Utility Functions to load and clean images"""

def all_img_name_vector(images_path: str, ext: str = r".jpg") -> list:
    """
    :param ext: extension of the image
    :returns the path of all the images in that dir
    :param images_path: the path of the dir in which the images are present
    """
    images_path_list = glob(images_path + '*.jpg')
    print(f"{len(images_path_list)} images found from {images_path}.")
    return images_path_list

def cnn_model() -> tf.keras.Model:
    """
    returns the cnn model needed for feature extraction
    :return:InceptionV3 without the last layer.
    """
    model = tf.keras.applications.InceptionV3(include_top=False, weights='imagenet')
    model = tf.keras.Model(inputs=model.inputs, outputs=model.layers[-1].output)
    print(model.summary())
    return model

def load_img(image_path: str, img_h: str = IMG_H, img_w: str = IMG_W) -> (object, str):
    """
    Returns the numpy image and the image path
    :param image_path: the path of the image
    EG: 'C:\\Users\\inba2\\Documents\\DataSet\\8k\\Images\\Images\\1000268201_693b08cb0e.jpg'
    :param img_w: the width of image taken into the cnn
    :param img_h: the height of image taken into the cnn
    :return: image and image_path
    """
    img = tf.io.read_file(image_path)
    img = tf.image.decode_jpeg(img, channels=3)
    img = tf.image.resize(img, (img_h, img_w))
    img = preprocess_input(img)
    return img, image_path

def load_doc(filename: str) -> str:
    """
    to open the file as read only
    :param filename: name of the file
    :return: the entire text as a string
    """
    with open(filename, 'r') as file:
        text = file.read()
    return text

def load_set(text_file_path: str) -> set:
    """
    to load a pre-defined list of photo names
    returns the names of the images form the set_text_file
    :param text_file_path:
    :return:
    """
    doc = load_doc(text_file_path)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
            continue
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

def img_name_2_path(image_name: str, img_file_path: str = IMG_FILE_PATH, ext: str = r".jpg") -> str:
    """
    Converts the name of the image to image path
    :param image_name: The name of the image
    :param img_file_path: The path where the image is stored
    :param ext:The extension of the image default is .jpg
    :return: The image path
    """
    image_path = img_file_path + str(image_name) + ext
    return os.path.join(img_file_path, image_name + ext)

def load_img_dataset(txt_path: str, batch_size=BATCH_SIZE):
    img_name_vector = load_set(txt_path)
    img_path_list = list(map(img_name_2_path, img_name_vector))

    img_path_list = [path for path in img_path_list if os.path.exists(path)]

    if not img_path_list:
        raise ValueError("No valid image files found. Please check your dataset path.")

    print(f"{len(img_path_list)} valid image files found.")

    encode_train = sorted(img_path_list)
    image_dataset = tf.data.Dataset.from_tensor_slices(encode_train)
    image_dataset = image_dataset.map(load_img, num_parallel_calls=tf.data.AUTOTUNE).batch(batch_size)

    return image_dataset

"""# Features extraction

## Loading the train images
"""

TRAIN_TXT_PATH

image_train_dataset = load_img_dataset(TRAIN_TXT_PATH)

image_train_dataset

"""## Creating our image features extraction model"""

image_features_extract_model = cnn_model()

"""## Extracting features and saving it in the same dir"""

def check_missing_images(image_paths: list):
    missing = [p for p in image_paths if not os.path.exists(p)]
    if missing:
        print(f"Missing {len(missing)} files! Example: {missing[:1]}")
    else:
        print("All image files found.")
image_paths = list(map(img_name_2_path, load_set(TRAIN_TXT_PATH)))
check_missing_images(image_paths)

if str(input("Do you want to extract the features of the images[y/Y]: ")).casefold() == 'y':
    for img_batch, path_batch in tqdm(image_train_dataset):
        batch_features = image_features_extract_model(img_batch)
        batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
        for bf, p in zip(batch_features, path_batch):
            try:
                path_of_feature = p.numpy().decode("utf-8")
                np.save(path_of_feature, bf.numpy())
            except Exception as e:
                print(f"Skipping file due to error: {e}")

"""'## Loading the test images"""

TEST_TXT_PATH

image_test_dataset = load_img_dataset(TEST_TXT_PATH)

"""## Extracting features and saving it in the same dir"""

if str(input("Do you want to extract the features of the images[y/Y]: ")).casefold() == 'y':
    for img, path in tqdm(image_test_dataset):
        batch_features = image_features_extract_model(img)
        batch_features = tf.reshape(batch_features, (batch_features.shape[0], -1, batch_features.shape[3]))
        for bf, p in zip(batch_features, path):
            path_of_feature = p.numpy().decode("utf-8")
            np.save(path_of_feature, bf.numpy())

"""# Word/Sentence preprocessing

## Helper functions
"""

def clean_cap(caption: str) -> str:
    """
    preprocessing the caption
    :param caption: unprocessed caption
    :return: processed caption
    """
    # Removes Punctuations
    cap = ''.join([ch for ch in caption if ch not in string.punctuation])
    # tokenize
    cap = cap.split()
    # convert to lower case
    cap = [word.casefold() for word in cap]
    # remove hanging 's' and 'a'
    cap = [word for word in cap if len(word) > 1]
    # remove tokens with numbers in them
    cap = [word for word in cap if word.isalpha()]
    # Lemmatizing
    lemmatizer = nltk.WordNetLemmatizer()
    cap = [lemmatizer.lemmatize(word) for word in cap]
    # store as string
    return ' '.join(cap)

def load_cap(caption_txt_path: str) -> dict:
    """
    To read the text file containing captions and store it in a dict.
    mapping(dict) contains image_name as key and the corresponding captions
    as a list of words.
    :param caption_txt_path:
    :return: mapping(dict) contains image_name as key and the
    corresponding captions as a list of captions
    """
   
    with open(caption_txt_path, 'r', encoding='utf-8') as caption_txt:
        captions_list = caption_txt.readlines()
    mapping = dict()
    for line in captions_list:
        caption = line.split('\t')
        image_name = caption[0][:-2].split('.')[0]
        image_caption = clean_cap(caption[-1][:-1])
        # adding start and end of seq
        image_caption = ('startofseq ' + image_caption + ' endofseq')
        if image_name in mapping:
            mapping[image_name].append(image_caption)
        else:
            mapping[image_name] = [image_caption]
    return mapping

def save_captions(mapping: dict, filename: str) -> bool:
    """
    To write the mappings in the disk, one per line.
    written as "key<space>caption"
    :param mapping: dict with image_name as key and the
    corresponding captions as a list of captions
    :param filename: file name to save
    """
    lines = list()
    for key, cap_list in mapping.items():
        for cap in cap_list:
            lines.append(key + ' ' + cap)
    data = '\n'.join(lines)
    with open(filename, 'w') as file:
        file.write(data)
    return True

def load_doc(filename: str) -> str:
    """
    to open the file as read only
    :param filename: name of the file
    :return: the entire text as a string
    """
    with open(filename, 'r', encoding='utf-8') as file:
        text = file.read()
    return text

def load_set(text_file_path: str) -> set:
    """
    to load a pre-defined list of photo names
    returns the names of the images form the set_text_file
    :param text_file_path:
    :return:
    """
   
    doc = load_doc(text_file_path)
    dataset = list()
    for line in doc.split('\n'):
        if len(line) < 1:
        identifier = line.split('.')[0]
        dataset.append(identifier)
    return set(dataset)

def load_clean_cap(caption_txt_path: str, dataset: set) -> dict:
    """
    load clean descriptions into memory
    :param caption_txt_path: The path where the clean captions were saved
    :param dataset: the img names of the tining or test data set
    :return: dict of captions mapped with its curr name
    """
    doc = load_doc(caption_txt_path)
    clean_captions = dict()
    for line in doc.split('\n'):
        tokens = line.split()
        image_name, image_cap = tokens[0], " ".join(tokens[1:])
        if image_name in dataset:
            if image_name not in clean_captions:
                clean_captions[image_name] = list()
            clean_captions[image_name].append(image_cap)
    return clean_captions

def max_len(clean_captions: dict) -> int:
    """
    Returns the length of the caption with most words
    :param clean_captions: a dictionary of captions
    :return: length of the longest caption
    """
    # Converts a dictionary of clean captions and returns a list of captions.
    clean_captions_list = [caption.split() for captions in clean_captions.values()
                           for caption in captions]
    return max(len(caption) for caption in clean_captions_list)

def create_tokenizer(captions_dict: dict, top_k: int = 2000) -> Tokenizer:
    """
    Fit a tokenizer given caption descriptions
    :param captions_dict: (dict) of clean captions
    :param top_k: number of words in vocabulary
    :return: tokenizer object
    """
    clean_captions_list = [caption for captions in captions_dict.values()
                           for caption in captions]
    tokenizer = Tokenizer(num_words=top_k, oov_token="<unk>")
    # Map '<pad>' to '0'
    tokenizer.word_index['<pad>'] = 0
    tokenizer.index_word[0] = '<pad>'
    tokenizer.fit_on_texts(clean_captions_list)
    return tokenizer

def path_cap_list(img_names_set: set, tokenizer: Tokenizer, captions_dict) -> (list, list):
    """
    a list of image paths and a list of captions for images with corresponding values
    Note: the captions will be tokenized and padded in this function
    :param img_names_set: The set on which the processing is done
    :param tokenizer: tokenizer
    :param captions_dict: clean captions for that set without any tokenization
    """
    tokenized_caps_dict = tokenize_cap(tokenizer, captions_dict)
    image_name_list = sorted(img_names_set)
    capt_list = [cap for name in image_name_list for cap in tokenized_caps_dict[name]]
    img_path_list = [img_name_2_path(name) for name in image_name_list for i in range(len(tokenized_caps_dict[name]))]
    return img_path_list, capt_list

def load_npy(image_path: str, cap: str) -> (str, str):
    """
    :returns image tensor vector with the image path
    :param image_path:
    :param cap:
    """
    img_tensor = np.load(image_path.decode('utf-8') + '.npy')
    return img_tensor, cap

def create_dataset(img_path_list: str, cap_list: str) -> object:
    """
    :param img_path_list: The ordered list of img paths with duplication acc to number of captions
    :param cap_list: the padded caption list with the curr order
    :return: dataset
    """
    dataset = tf.data.Dataset.from_tensor_slices((img_path_list, cap_list))
    dataset = dataset.map(lambda item1, item2: tf.numpy_function(load_npy, [item1, item2], [tf.float32, tf.int32]),
                          num_parallel_calls=tf.data.experimental.AUTOTUNE)
    dataset = dataset.shuffle(BUFFER_SIZE).batch(BATCH_SIZE).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)
    return dataset

CAP_TEXT_PATH

"""Pre processing is also done while loading the captions"""

import nltk
nltk.download('wordnet')

cap_dict = load_cap(CAP_TEXT_PATH)

"""Finding the maximum len of the captption"""

MAX_CAP_LEN = max_len(cap_dict)
print(MAX_CAP_LEN)

def tokenize_cap(tokenizer: Tokenizer, captions_dict: dict, pad_len: int = MAX_CAP_LEN) -> dict:
    """
    Tokenizes the captions and
    :param pad_len: The maximum caption length in the whole dataset
    (should include both train and test dataset)
    :param tokenizer: Tokenizer object
    :param captions_dict: The dict of train/test cap which have to be tokenized
    :return: A dict of tokenized captions
    """
    pad_caps_dict = {img_name: pad_sequences(tokenizer.texts_to_sequences(captions), maxlen=pad_len, padding='post')
                     for img_name, captions in captions_dict.items()}
    return pad_caps_dict

CLEAN_CAP_TEXT_PATH = r'asset1/Flickr8k_text/Flickr8k_clean_ca.txt'

"""## Saving the clean captions"""

save_captions(cap_dict,CLEAN_CAP_TEXT_PATH)

"""# Tokenizing words

## Loading the train img names
"""

train_img_names = sorted(load_set(TRAIN_TXT_PATH))

"""Loading the train img captions as a dict"""

train_img_cap = load_clean_cap(CLEAN_CAP_TEXT_PATH, train_img_names)

len(train_img_cap)

for x in train_img_cap.values():
    print(x)
    break

"""## Creating a tokenizer for training set"""

tokenizer = create_tokenizer(train_img_cap)

VOCAB_SIZE = len(tokenizer.word_index) + 1
print(VOCAB_SIZE)

tokenizer.word_index

"""# Train dataset

## Tokenizing the train captions and storing it in lists
"""

img_name_train, caption_train = path_cap_list(train_img_names, tokenizer, train_img_cap)

img_name_train[:10]

caption_train[:10]

"""## Creating train dataset"""

train_dataset = create_dataset(img_name_train, caption_train)

"""# Test Dataset

## Loading the test img names
"""

test_img_names = sorted(load_set(TEST_TXT_PATH))

"""## Loading the test img captions as a dict"""

test_img_cap = load_clean_cap(CLEAN_CAP_TEXT_PATH, test_img_names)

len(test_img_cap)

"""## Creating the test dataset"""

img_name_test, caption_test = path_cap_list(test_img_names, tokenizer, test_img_cap)

test_dataset = create_dataset(img_name_test, caption_test)

"""# Defining some constants"""

embedding_dim = 256
units = 512
vocab_size = len(tokenizer.word_index) + 1 # 6537
num_steps = len(train_img_names) // BATCH_SIZE  # 187
EPOCHS = 20
features_shape = 512
attention_features_shape = 49

"""# Addititve Attention """

class BahdanauAttention(tf.keras.Model):
  def __init__(self, units):
    super(BahdanauAttention, self).__init__()
    self.W1 = tf.keras.layers.Dense(units)
    self.W2 = tf.keras.layers.Dense(units)
    self.V = tf.keras.layers.Dense(1)

  def call(self, features, hidden):
    hidden_with_time_axis = tf.expand_dims(hidden, 1)

    attention_hidden_layer = (tf.nn.tanh(self.W1(features) +
                                         self.W2(hidden_with_time_axis)))

    score = self.V(attention_hidden_layer)

    attention_weights = tf.nn.softmax(score, axis=1)

    context_vector = attention_weights * features
    context_vector = tf.reduce_sum(context_vector, axis=1)

    return context_vector, attention_weights

class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        x = self.fc(x)
        x = tf.nn.relu(x)
        return x

def rnn_type(units):
    if tf.test.is_gpu_available():
        return tf.compat.v1.keras.layers.CuDNNLSTM(units,
                                        return_sequences=True,
                                        return_state=True,
                                        recurrent_initializer='glorot_uniform')
    else:
        return tf.keras.layers.GRU(units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_activation='sigmoid',
                                   recurrent_initializer='glorot_uniform')

class RNN_Decoder(tf.keras.Model):
  def __init__(self, embedding_dim, units, vocab_size):
    super(RNN_Decoder, self).__init__()
    self.units = units

    self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
    self.gru = tf.keras.layers.GRU(self.units,
                                   return_sequences=True,
                                   return_state=True,
                                   recurrent_initializer='glorot_uniform')
    self.fc1 = tf.keras.layers.Dense(self.units)
    self.fc2 = tf.keras.layers.Dense(vocab_size)

    self.attention = BahdanauAttention(self.units)

  def call(self, x, features, hidden):
   
    context_vector, attention_weights = self.attention(features, hidden)
    x = self.embedding(x)
    x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)
    output, state = self.gru(x)
    x = self.fc1(output)
    x = tf.reshape(x, (-1, x.shape[2]))
    x = self.fc2(x)

    return x, state, attention_weights

  def reset_state(self, batch_size):
    return tf.zeros((batch_size, self.units))

encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)

optimizer = tf.keras.optimizers.Adam()
loss_object = tf.keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')


def loss_function(real, pred):
  mask = tf.math.logical_not(tf.math.equal(real, 0))
  loss_ = loss_object(real, pred)

  mask = tf.cast(mask, dtype=loss_.dtype)
  loss_ *= mask

  return tf.reduce_mean(loss_)

"""# Checkpoint"""

checkpoint_path = "./checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder,
                           decoder=decoder,
                           optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
  start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
  ckpt.restore(ckpt_manager.latest_checkpoint)

"""## Training

* We extract the features stored in the respective `.npy` files and then pass those features through the encoder.
* The encoder output, hidden state(initialized to 0) and the decoder input (which is the start token) is passed to the decoder.
* The decoder returns the predictions and the decoder hidden state.
* The decoder hidden state is then passed back into the model and the predictions are used to calculate the loss.
* We use teacher forcing to decide the next input to the decoder.
* Teacher forcing is the technique where the target word is passed as the next input to the decoder.
* The final step is to calculate the gradients and apply it to the optimizer and backpropagate.

"""

loss_plot = []

@tf.function
def train_step(img_tensor, target):
  loss = 0

  hidden = decoder.reset_state(batch_size=target.shape[0])

  dec_input = tf.expand_dims([tokenizer.word_index['startofseq']] * target.shape[0], 1)

  with tf.GradientTape() as tape:
      features = encoder(img_tensor)

      for i in range(1, target.shape[1]):
          predictions, hidden, _ = decoder(dec_input, features, hidden)

          loss += loss_function(target[:, i], predictions)

          dec_input = tf.expand_dims(target[:, i], 1)

  total_loss = (loss / int(target.shape[1]))

  trainable_variables = encoder.trainable_variables + decoder.trainable_variables

  gradients = tape.gradient(loss, trainable_variables)

  optimizer.apply_gradients(zip(gradients, trainable_variables))

  return loss, total_loss


"""# InceptionV3"""

import tensorflow as tf
import numpy as np
from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import preprocess_input

inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')


def extract_image_features(img_path):
    img = image.load_img(img_path, target_size=(299, 299))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = preprocess_input(img_array)  

    features = inception_model.predict(img_array) 
    return features


image_path = 'asset1/Flickr8k_dataset/1022454332_6af2c1449a.jpg'
features = extract_image_features(image_path)
print(features.shape)  

img_path = "asset1/Flickr8k_dataset/1022454332_6af2c1449a.jpg"

import numpy as np
import os
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.inception_v3 import InceptionV3, preprocess_input
from tensorflow.keras.models import Model


base_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
model = Model(inputs=base_model.input, outputs=base_model.output)


image_dir = 'asset1/Flickr8k_dataset/'


npy_dir = 'asset1/Flickr8k_features/'
os.makedirs(npy_dir, exist_ok=True)

# Function to extract features and save them
def extract_and_save_features(image_path):
    img = image.load_img(image_path, target_size=(299, 299))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array = preprocess_input(img_array)

    # Extracting features using InceptionV3
    features = model.predict(img_array)

    filename = os.path.basename(image_path)
    name_without_extension = os.path.splitext(filename)[0]

    # Saving features as a .npy file
    np.save(os.path.join(npy_dir, name_without_extension + '.npy'), features)


for img_name in os.listdir(image_dir):
    if img_name.endswith('.jpg'):  
        img_path = os.path.join(image_dir, img_name)
        extract_and_save_features(img_path)

print("Feature extraction completed!")

# Function to load pre-saved features from the .npy file
def load_features_from_npy(image_filename):
    npy_file_path = os.path.join('asset1/Flickr1k_features', image_filename + '.npy')

    if os.path.exists(npy_file_path):
        features = np.load(npy_file_path)
        return features
    else:
        print(f"File {npy_file_path} not found!")
        return None

import os
import numpy as np

def load_image_and_extract_features(img_path):
    
    image_filename = os.path.basename(img_path.numpy().decode())

    filename_without_ext = os.path.splitext(image_filename)[0]

    feature_path = os.path.join("asset1/Flickr8k_features", filename_without_ext + ".npy")

    print(f"Loading features from: {feature_path}")
    return np.load(feature_path)

def map_func(img_path, caption):
    img_tensor = load_image_and_extract_features(img_path)
    return img_tensor, caption

train_dataset = train_dataset.map(
    lambda img, cap: tf.py_function(
        func=map_func,
        inp=[img, cap],
        Tout=(tf.float32, tf.int32)  
    ),
    num_parallel_calls=tf.data.AUTOTUNE
)

def data_generator(img_paths, caps):
    for img_path, caption in zip(img_paths, caps):
        filename = os.path.splitext(os.path.basename(img_path))[0] + ".npy"
        npy_path = os.path.join("asset1/Flickr1k_features", filename)

        if os.path.exists(npy_path):
            features = np.load(npy_path).astype(np.float32)
            yield features, caption
        else:
            print(f"[WARNING] Feature not found: {npy_path}")
            yield np.zeros((2048,), dtype=np.float32), caption

train_dataset = tf.data.Dataset.from_generator(
    lambda: data_generator(img_name_train, caption_train),
    output_signature=(
        tf.TensorSpec(shape=(2048,), dtype=tf.float32),     
        tf.TensorSpec(shape=(None,), dtype=tf.int32)        
    )
)

def load_image_and_extract_features(img_path):
    def load_npy(path):
        try:
            path_str = path.decode("utf-8")  
            image_filename = os.path.splitext(os.path.basename(path_str))[0]  
            npy_path = os.path.join('asset1/Flickr8k_features', image_filename + '.npy')

            if os.path.exists(npy_path):
                return np.load(npy_path).astype(np.float32)  
            else:
                print(f"[SKIPPED] File not found: {npy_path}")
                return np.zeros((2048,), dtype=np.float32)  
        except Exception as e:
            print(f"[ERROR] {e}")
            return np.zeros((2048,), dtype=np.float32) 

  
    img_tensor = tf.py_function(func=load_npy, inp=[img_path], Tout=tf.float32)
    img_tensor.set_shape([2048]) 
    return img_tensor

import os

feature_dir = 'asset1/Flickr8k_features'
train_image_paths = [
    os.path.join(feature_dir, os.path.splitext(f)[0])
    for f in os.listdir(feature_dir)
    if f.endswith('.npy')
]
print(f"Found {len(train_image_paths)} .npy files")

import os

feature_dir = 'asset1/Flickr8k_features'

def check_npy_files(image_paths):
    missing_files = []
    for path in image_paths:
        npy_path = path + '.npy' if not path.endswith('.npy') else path
        if not os.path.exists(npy_path):
            missing_files.append(npy_path)
    return missing_files

missing = check_npy_files(train_image_paths)
if missing:
    print(f"Missing {len(missing)} .npy files:", missing)
else:
    print("All .npy files found.")

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input
import numpy as np

def extract_features(image_path, model):
    try:
        img = load_img(image_path, target_size=(299, 299))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = model.predict(img)
        return np.squeeze(features)  
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# Initializing InceptionV3
inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')

# Regenerating missing files
dataset_dir = 'asset1/Flickr8k_dataset'
for npy_path in missing:
    image_id = os.path.basename(npy_path).replace('.npy', '')
    image_path = os.path.join(dataset_dir, image_id + '.jpg')  
    if os.path.exists(image_path):
        features = extract_features(image_path, inception_model)
        if features is not None:
            np.save(npy_path, features)
            print(f"Saved features to {npy_path}")
    else:
        print(f"Image not found: {image_path}")

# Re-check after regeneration
missing = check_npy_files(train_image_paths)
if missing:
    raise ValueError(f"Still missing {len(missing)} .npy files after regeneration")

import os
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

feature_dir = 'asset1/Flickr8k_features'
caption_file = 'asset1/Flickr8k_text/Flickr8k_clean_cap.txt'

# Loading captions and aligning them with .npy files
caption_dict = {}
with open(caption_file, 'r') as f:
    for line in f:
        parts = line.strip().split(' ', 1)  
        if len(parts) != 2:
            continue
        image_id, caption = parts
        if image_id not in caption_dict:
            caption_dict[image_id] = []
        caption_dict[image_id].append(caption)

# Defining train_image_paths and raw_captions
train_image_paths = []
raw_captions = []
for image_id in caption_dict:
    npy_path = os.path.join(feature_dir, image_id)
    if os.path.exists(npy_path + '.npy'):
        for caption in caption_dict[image_id]:
            train_image_paths.append(npy_path)  
            raw_captions.append(caption)
    else:
        print(f"No .npy file found for {image_id}")

print(f"Found {len(train_image_paths)} image-caption pairs")
print(f"Found {len(raw_captions)} captions")

"""Tokenize caption"""

vocab_size = 5000
MAX_CAP_LEN = 50

tokenizer = Tokenizer(num_words=vocab_size, oov_token='<unk>')
tokenizer.fit_on_texts(raw_captions)
if 'startofseq' not in tokenizer.word_index:
    tokenizer.word_index['startofseq'] = len(tokenizer.word_index) + 1
    tokenizer.index_word[tokenizer.word_index['startofseq']] = 'startofseq'
if 'endofseq' not in tokenizer.word_index:
    tokenizer.word_index['endofseq'] = len(tokenizer.word_index) + 1
    tokenizer.index_word[tokenizer.word_index['endofseq']] = 'endofseq'

train_captions = tokenizer.texts_to_sequences(raw_captions)
train_captions = pad_sequences(train_captions, maxlen=MAX_CAP_LEN, padding='post')
print(f"train_captions shape: {np.array(train_captions).shape}")

"""checking for .npy files"""

def check_npy_files(image_paths):
    missing_files = []
    for path in image_paths:
        npy_path = path + '.npy'
        if not os.path.exists(npy_path):
            missing_files.append(npy_path)
    return missing_files

missing = check_npy_files(train_image_paths)
if missing:
    print(f"Missing {len(missing)} .npy files:", missing)
else:
    print("All .npy files found.")

"""regenrate missing .npy files"""

from tensorflow.keras.applications import InceptionV3
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.inception_v3 import preprocess_input

def extract_features(image_path, model):
    try:
        img = load_img(image_path, target_size=(299, 299))
        img = img_to_array(img)
        img = np.expand_dims(img, axis=0)
        img = preprocess_input(img)
        features = model.predict(img)
        return np.squeeze(features)  
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

inception_model = InceptionV3(weights='imagenet', include_top=False, pooling='avg')
dataset_dir = 'asset1/Flickr8k_dataset'

for npy_path in missing:
    image_id = os.path.basename(npy_path).replace('.npy', '')
    image_path = os.path.join(dataset_dir, image_id + '.jpg') 
    if os.path.exists(image_path):
        features = extract_features(image_path, inception_model)
        if features is not None:
            np.save(npy_path, features)
            print(f"Saved features to {npy_path}")
    else:
        print(f"Image not found: {image_path}")

# Re-check
missing = check_npy_files(train_image_paths)
if missing:
    raise ValueError(f"Still missing {len(missing)} .npy files")

#load npy
def load_npy(image_path: str, cap: str) -> (tf.Tensor, tf.Tensor):
    npy_path = image_path.decode('utf-8') + '.npy'
    if not os.path.exists(npy_path):
        raise FileNotFoundError(f"Feature file not found: {npy_path}")
    img_tensor = np.load(npy_path)
    img_tensor = np.squeeze(img_tensor)
    if img_tensor.shape != (2048,):
        raise ValueError(f"Unexpected .npy shape: {img_tensor.shape}, expected (2048,)")
    return tf.convert_to_tensor(img_tensor, dtype=tf.float32), cap

"""train dataset"""

import tensorflow as tf

BATCH_SIZE = 64

def create_dataset(image_paths, captions, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((image_paths, captions))
    dataset = dataset.map(
        lambda img_path, cap: tf.numpy_function(
            load_npy, [img_path, cap], [tf.float32, tf.int32]
        ),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.map(
        lambda img, cap: (img, tf.ensure_shape(cap, [MAX_CAP_LEN])),
        num_parallel_calls=tf.data.AUTOTUNE
    )
    dataset = dataset.batch(batch_size, drop_remainder=True)
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset

train_dataset = create_dataset(train_image_paths, train_captions, BATCH_SIZE)

# Verify shapes
for img_tensor, target in train_dataset.take(1):
    print("img_tensor shape:", img_tensor.shape) 
    print("target shape:", target.shape)  
    break

"""CNN Encoder"""

class CNN_Encoder(tf.keras.Model):
    def __init__(self, embedding_dim):
        super(CNN_Encoder, self).__init__()
        self.fc = tf.keras.layers.Dense(embedding_dim)

    def call(self, x):
        if len(x.shape) == 1:
            x = tf.expand_dims(x, axis=0)  
        x = self.fc(x)  
        x = tf.nn.relu(x)
        x = tf.expand_dims(x, axis=1)  
        return x

class RNN_Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, units, vocab_size):
        super(RNN_Decoder, self).__init__()
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = tf.keras.layers.GRU(
            units, return_sequences=True, return_state=True
        )
        self.fc = tf.keras.layers.Dense(vocab_size)

    def call(self, x, features, hidden):
        x = self.embedding(x)
        output, state = self.gru(x, initial_state=hidden)
        output = self.fc(output)
        return output, state, None

    def reset_state(self, batch_size):
        return tf.zeros((batch_size, self.units))

"""initalize model and constant"""

embedding_dim = 256
units = 512
vocab_size = 5000
EPOCHS = 20
encoder = CNN_Encoder(embedding_dim)
decoder = RNN_Decoder(embedding_dim, units, vocab_size)
optimizer = tf.keras.optimizers.Adam()

"""loss function"""

def loss_function(real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.keras.losses.sparse_categorical_crossentropy(
        real, pred, from_logits=True
    )
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_mean(loss_)

"""train_step and looping"""

@tf.function
def train_step(img_tensor, target):
    print("img_tensor shape:", img_tensor.shape)
    print("target shape:", target.shape)
    loss = 0
    hidden = decoder.reset_state(batch_size=target.shape[0])
    dec_input = tf.expand_dims([tokenizer.word_index['startofseq']] * target.shape[0], 1)

    with tf.GradientTape() as tape:
        features = encoder(img_tensor)
        print("features shape:", features.shape)
        for i in range(1, target.shape[1]):
            predictions, hidden, _ = decoder(dec_input, features, hidden)
            predictions = tf.squeeze(predictions, axis=1)
            loss += loss_function(target[:, i], predictions)
            dec_input = tf.expand_dims(target[:, i], 1)

    total_loss = (loss / int(target.shape[1]))
    trainable_variables = encoder.trainable_variables + decoder.trainable_variables
    gradients = tape.gradient(loss, trainable_variables)
    optimizer.apply_gradients(zip(gradients, trainable_variables))

    return loss, total_loss

# Training loop
loss_plot = []
checkpoint_path = "asset1/checkpoints/train"
ckpt = tf.train.Checkpoint(encoder=encoder, decoder=decoder, optimizer=optimizer)
ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

start_epoch = 0
if ckpt_manager.latest_checkpoint:
    start_epoch = int(ckpt_manager.latest_checkpoint.split('-')[-1])
    ckpt.restore(ckpt_manager.latest_checkpoint)
    print(f"Restored from checkpoint: {ckpt_manager.latest_checkpoint}")

for epoch in range(start_epoch, EPOCHS):
    start = time.time()
    total_loss = 0
    for (batch, (img_tensor, target)) in enumerate(tqdm(train_dataset, desc=f"Epoch {epoch+1}/{EPOCHS}")):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss
    avg_loss = total_loss / (batch + 1)
    loss_plot.append(avg_loss)
    if (epoch + 1) % 5 == 0:
        ckpt_manager.save()
        print(f"Checkpoint saved at epoch {epoch+1}")
    print(f'Epoch {epoch+1} Loss {avg_loss:.4f}')
    print(f'Time taken for epoch: {time.time() - start:.2f} secs\n')

# Saving weights
encoder.save_weights('asset1/Flickr1k_text/encoder_model.h5')
decoder.save_weights('asset1/Flickr1k_text/decoder_model.h5')
print("Model weights saved.")

def evaluate(image_path):
    image_filename = os.path.splitext(os.path.basename(image_path))[0]
    npy_file_path = os.path.join(feature_dir, image_filename + '.npy')
    if not os.path.exists(npy_file_path):
        return f"Features not found for {image_path}"

    features = np.load(npy_file_path)
    features = np.squeeze(features)
    if features.shape != (2048,):
        return f"Unexpected feature shape: {features.shape} for {image_path}"

    features = tf.convert_to_tensor(features, dtype=tf.float32)
    hidden = decoder.reset_state(batch_size=1)
    dec_input = tf.expand_dims([tokenizer.word_index['startofseq']], 1)
    result = []

    for _ in range(MAX_CAP_LEN):
        features_reshaped = encoder(features)
        predictions, hidden, _ = decoder(dec_input, features_reshaped, hidden)
        predictions = tf.squeeze(predictions, axis=1)
        predicted_id = tf.argmax(predictions[0]).numpy()
        word = tokenizer.index_word.get(predicted_id, '<unk>')
        if word == 'endofseq':
            break
        result.append(word)
        dec_input = tf.expand_dims([predicted_id], 1)

    return ' '.join(result)

for img_tensor, target in train_dataset.take(1):
    print("img_tensor shape:", img_tensor.shape)
    print("target shape:", target.shape)
    break

"""RUN SINGLE BATCH"""

for img_tensor, target in train_dataset.take(1):
    batch_loss, t_loss = train_step(img_tensor, target)
    print(f"Batch loss: {t_loss:.4f}")
    break

"""RUN TRAINING"""

for epoch in range(1):
    total_loss = 0
    for batch, (img_tensor, target) in enumerate(tqdm(train_dataset, desc="Epoch 1/1")):
        batch_loss, t_loss = train_step(img_tensor, target)
        total_loss += t_loss
    print(f'Epoch 1 Loss {total_loss / (batch + 1):.4f}')

# Plot loss
plt.plot(loss_plot)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss Over Epochs')
plt.savefig('asset1/Flickr8k_text/loss_plot.png')
plt.close()
print("Loss plot saved.")

"""GENERATE CAPTION"""

test_image_path = 'asset1/Flickr8k_dataset/1022454332_6af2c1449a.jpg'
caption = evaluate(test_image_path)
print(f"Generated caption: {caption}")
