import tensorflow as tf
import tensorflow_datasets as tfds

# !  use of IMDB subwords dataset :


(train_data, test_data), info = tfds.load(
    "imdb_reviews/subwords8k",
    split=(tfds.Split.TRAIN, tfds.Split.TEST),
    as_supervised=True,
    with_info=True,
)


encoder = info.features["text"].encoder

print("Vocabulary size: {}".format(encoder.vocab_size))

# print(encoder.subwords)

sample_words = " sam do something bro "

encoded_words = encoder.encode(sample_words)

print("the encoded string is {}".format(encoded_words))

decoded_words = encoder.decode(encoded_words)
print("Decoded string: {}".format(decoded_words))

print(encoder.subwords[291])