import numpy as np

import keras
from keras.optimizers import Adam, RMSprop
from keras import  backend as K

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score

import matplotlib.pyplot as plt
import seaborn as sns

class Experiment:
    def __init__(self, model, batch_size=32, epochs=100, optimizer="Adam", loss="categorical_crossentropy", test_size=0.2, val_size=0.2):
        self.model = model
        self.optimizer = optimizer
        self.batch_size = batch_size
        self.loss = loss
        self.epochs = epochs
        self.test_size = test_size
        self.val_size = val_size

    def train(self, X_train, X_val, y_train, y_val):
        self.model.compile(
            optimizer=self.optimizer,
            loss=self.loss,
            metrics=['accuracy'])

        return self.model.fit(x=X_train, y=y_train, epochs=self.epochs, batch_size=self.batch_size, validation_data=(X_val, y_val))

    def test(self, X_test, y_test):
        y_pred = self.model.predict(X_test)
        y_pred = np.argmax(y_pred, axis=1)
        y_test_label = np.argmax(y_test, axis=1)
        result = confusion_matrix(y_test_label, y_pred, normalize='pred')


        print(f'f1 score macro: {f1_score(y_test_label, y_pred, average='macro')}')
        print(f'f1 score micro: {f1_score(y_test_label, y_pred, average='micro')}')
        print(f'f1 score weighted: {f1_score(y_test_label, y_pred, average='weighted')}')
        print()
        print(classification_report(y_test_label, y_pred))


        sns_heatmap = sns.heatmap(result, annot=True)
        fig = sns_heatmap.get_figure()
        plt.show()