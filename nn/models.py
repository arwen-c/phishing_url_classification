from keras.utils import to_categorical

from string_embedding.cnn import model_builder_cnn_character_level
from feature_vector.cnn import model_builder_cnn
from feature_vector.dnn import model_builder_dnn
from feature_vector.lstm import model_builder_lstm

from data.load.load import load_dill as load_dill, load_feature_vector as load_feature_vector


def cnn_load(**kwargs):
    return load_dill(**kwargs)[:-1]


string_embedding_models = {
    "cnn": (
        model_builder_cnn_character_level,
        cnn_load,
        lambda x: x,
        lambda x: to_categorical(x, num_classes=2)
    ),
}
feature_vector_models = {
    "cnn": (
        model_builder_cnn,
        load_feature_vector,
        lambda x: x,
        lambda x: to_categorical(x, num_classes=2)
    ),
    "dnn": (
        model_builder_dnn,
        load_feature_vector,
        lambda x: x,
        lambda x: to_categorical(x, num_classes=2)
    ),
    "lstm": (
        model_builder_lstm,
        load_feature_vector,
        lambda x: x,
        lambda x: to_categorical(x, num_classes=2)
    ),
}
