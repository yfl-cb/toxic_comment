# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load in

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)

# Input data files are available in the "../input/" directory.
# For example, running this (by clicking run or pressing Shift+Enter) will list the files in the input directory

from subprocess import check_output
print(check_output(["ls", "data"]).decode("utf8"))

# Any results you write to the current directory are saved as output.

from keras.models import Model
from keras.layers import Dense, Embedding, Input
from keras.layers import LSTM, Bidirectional, GlobalMaxPool1D, Dropout
from keras.preprocessing import text, sequence
from keras.callbacks import EarlyStopping, ModelCheckpoint

max_features = 20000
maxlen = 100


train = pd.read_csv("data/train.csv")
test = pd.read_csv("data/test.csv")
# 返回随机抽样，这里frac = 1是返回全部的训练集
train = train.sample(frac=1)
# 取出comment_text，并给空值的部分添值
list_sentences_train = train["comment_text"].fillna("CVxTz").values
list_classes = ["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]
y = train[list_classes].values
list_sentences_test = test["comment_text"].fillna("CVxTz").values

# 对文本中的词进行统计计数，生成文档词典，以支持基于词典位序生成文本的向量表示，超过max_features的单词被丢掉
# 使用一系列文档来生成token词典，texts为list类，每个元素为一个文档
tokenizer = text.Tokenizer(num_words=max_features)
tokenizer.fit_on_texts(list(list_sentences_train))
# 将多个文档转换为word下标的向量形式，即把每个文档变成元素为词索引的向量
list_tokenized_train = tokenizer.texts_to_sequences(list_sentences_train)
list_tokenized_test = tokenizer.texts_to_sequences(list_sentences_test)
# 将序列填充到maxlen长度
X_t = sequence.pad_sequences(list_tokenized_train, maxlen=maxlen)
X_te = sequence.pad_sequences(list_tokenized_test, maxlen=maxlen)

# 函数式模型构建
def get_model():
    embed_size = 128
    # 输入是一个一阶张量，即一篇文档的sequence
    inp = Input(shape=(maxlen, ))
    # 是把输入的张量每个元素转化为embed_size维度的张量，输入到LSTM中去，这里边maxlen像是time_steps
    # 这里可以考虑用已经训练好的词向量，可能是写一个embeddings_regularizer方法
    x = Embedding(max_features, embed_size)(inp)
    # 把每个词的张量挨个输入，输出的每个词的张量维度是50，最终是maxlen * 50
    x = Bidirectional(LSTM(50, return_sequences=True))(x)
    # 对时间信号的全局最大池化，看起来是挑一个最大的
    x = GlobalMaxPool1D()(x)
    x = Dropout(0.1)(x)
    x = Dense(50, activation="relu")(x)
    x = Dropout(0.1)(x)
    x = Dense(6, activation="sigmoid")(x)
    model = Model(inputs=inp, outputs=x)
    model.compile(loss='binary_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    return model


model = get_model()
batch_size = 32
epochs = 2

# ModelCheckpoint回调函数将在每个epoch后保存模型到filepath，当save_best_only=True保存验证集误差最小的参数
file_path="weights_base.best.hdf5"
checkpoint = ModelCheckpoint(file_path, monitor='val_loss', verbose=1, save_best_only=True, mode='min')
# 当监测值不再改善时，该回调函数将中止训练
early = EarlyStopping(monitor="val_loss", mode="min", patience=20)

# 开始训练
callbacks_list = [checkpoint, early] #early
model.fit(X_t, y, batch_size=batch_size, epochs=epochs, validation_split=0.1, callbacks=callbacks_list)

model.load_weights(file_path)

y_test = model.predict(X_te)
sample_submission = pd.read_csv("data/sample_submission.csv")
sample_submission[list_classes] = y_test
sample_submission.to_csv("baseline.csv", index=False)