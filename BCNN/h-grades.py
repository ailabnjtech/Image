import numpy as np
from keras.models import Model, load_model
from keras.layers import Dropout, Flatten, Dense, Input, GlobalAveragePooling2D, Activation,\
    BatchNormalization, concatenate
from keras.layers.convolutional import Conv2D, MaxPooling2D
from keras.utils import plot_model, to_categorical
from keras import optimizers
import os
import cv2
from keras.applications import VGG16
from keras.applications.vgg16 import preprocess_input
from keras.callbacks import TensorBoard
import non_local
import channel_attention

class vgg():
    def __init__(self, shape, num_classes, data_path, label_path, model_path):
        self.shape = shape
        self.num_classes = num_classes
        self.data_path = data_path
        self.label_path = label_path
        self.model_path = model_path
        self.log_path = "./logs"
        self.classes = self.classname()

    def classname(self, prepath="./two_mi8_iphone7/"):
        classes = os.listdir(prepath)
        class_dict = {int(Class.split(".")[0]): Class.split(".")[1] for Class in classes[0:5]}
        return class_dict

    def generate_data(self, prepath="two_mi8_iphone7/"):
        classes = os.listdir(prepath)

        data_path = self.data_path
        label_path = self.label_path
        datas = []
        labels = []
        for i, abspath in enumerate(classes):
            img_names = os.listdir(prepath + abspath)
            for img_name in img_names:
                img = cv2.imread(os.path.join(prepath + abspath, img_name))
                if not isinstance(img, np.ndarray):
                    print("read img error")
                    continue
                img = cv2.resize(img, (224, 224))
                # img = img.astype(np.float32)
                img = preprocess_input(img)
                label = to_categorical(i, self.num_classes)
                labels.append(label)
                datas.append(img)
        datas = np.array(datas)
        labels = np.array(labels)
        np.save(data_path, datas)
        np.save(label_path, labels)
        return True

    def vgg_model(self):
        input_1 = Input(shape=self.shape)
        # x = input_1
        # base_model = load_model('./model_my/tea_base_category.h5')
        #
        # for layer in base_model.layers[1:11]:
        #     layer.trainable = False
        #     x = layer(x)
        #
        #
        # x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu",
        #            name='block4_conv1')(x)
        # x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu",
        #            name='block4_conv2')(x)
        # x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu",
        #            name='block4_conv3')(x)  # 512 28*28
        # x = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME", name='block4_pool')(x)  # 512 14*14
        #

        # x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu",
        #            name='block5_conv1')(x)
        # x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu",
        #            name='block5_conv2')(x)
        # x = Conv2D(filters=512, kernel_size=(3, 3), strides=(1, 1), padding="SAME", activation="relu",
        #            name='block5_conv3')(x)  # 512 14*14
        # cnn_output = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding="SAME", name='block5_pool')(x)  # 512 7*7
        # # self-attention
        # x = non_local.non_local_block(cnn_output, intermediate_dim=None, compression=2, mode='embedded',
        #                               add_residual=False)
        # x = BatchNormalization()(x)
        #
        # # channel-attention
        # y = channel_attention.squeeze_excitation_layer(cnn_output, 512, ratio=4, concate=False)
        # y = BatchNormalization()(y)
        #
        # # concat
        # x = concatenate([cnn_output, x], axis=3)
        # x = concatenate([x, y], axis=3)

        # # spp
        # gap = GlobalAveragePooling2D()(x)
        # x = Dense(512, activation='relu', name="9")(gap)
        # x = BatchNormalization()(x)
        # predict = Dense(self.num_classes, activation='softmax', name="10")(x)

        # spp ORI
        # gap = GlobalAveragePooling2D()(x)
        # x = Flatten()(x)
        # x = concatenate([gap, x])
        # x = Dense(512, activation='relu')(x)
        # x = BatchNormalization()(x)
        # x = Dense(512, activation='relu')(x)
        # x = BatchNormalization()(x)
        # predict = Dense(self.num_classes, activation='softmax')(x)
        #
        # model = Model(inputs=input_1, outputs=predict)


        # backbone
        base_model = VGG16(input_tensor=input_1, weights='imagenet', include_top=False)
        base_output = base_model.output

        # self-attention
        x = non_local.non_local_block(base_output, intermediate_dim=None, compression=2, mode='embedded',
                                      add_residual=False)
        x = BatchNormalization()(x)

        # channel-attention
        y = channel_attention.squeeze_excitation_layer(base_output, 512, ratio=4, concate=False)
        y = BatchNormalization()(y)

        # concat
        x = concatenate([base_output, x], axis=3)
        x = concatenate([x, y], axis=3)

        # spp
        gap = GlobalAveragePooling2D()(x)
        x = Flatten()(x)
        x = concatenate([gap, x])
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        x = Dense(512, activation='relu')(x)
        x = BatchNormalization()(x)
        predict = Dense(self.num_classes, activation='softmax')(x)
        model = Model(inputs=input_1, outputs=predict)

        for layer in (base_model.layers):
            layer.trainable = False

        for l in model.layers:
            print(l.name)
        model.summary()
        print(model.summary())
        sgd = optimizers.sgd(lr=0.001, momentum=0.9, nesterov=True)
        model.compile(sgd, loss="categorical_crossentropy", metrics=["accuracy"])
        plot_model(model,"model.png")
        return model

    def pretrain_vgg(self):
        model_vgg = VGG16(include_top=False, weights='imagenet', input_shape=(224, 224, 3))
        model = Flatten(name='Flatten')(model_vgg.output)
        model = Dense(self.num_classes, activation='softmax')(model)

        model_vgg = Model(inputs=model_vgg.input, outputs=model, name='vgg16')
        sgd = optimizers.SGD(lr=0.001, momentum=0.9, nesterov=True)
        # adam = optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
        model_vgg.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])
        # model_vgg.summary()
        return model_vgg

    def train(self, load_pretrain=False, batch_size=32, epoch=50):
        # if load_pretrain:
        #     model = self.pretrain_vgg()
        # else:
        model = self.vgg_model()
        # TensorBoard查看日志
        logs = TensorBoard(log_dir=self.log_path, write_graph=True, write_images=True)

        data_path = self.data_path
        label_path = self.label_path
        save_path = self.model_path
        x = np.load(data_path)
        y = np.load(label_path)

        np.random.seed(200)
        np.random.shuffle(x)
        np.random.seed(200)
        np.random.shuffle(y)
        model.fit(x, y, batch_size=batch_size, epochs=epoch, verbose=1, validation_split=0.3, callbacks=[logs])
        model.save(save_path)

    def predict(self, img_path="test.jpg"):
        # model = vgg_model((224,224,3),5)
        model_path = self.model_path
        model = load_model(model_path)
        test_img = cv2.imread(img_path)
        test_img = cv2.resize(test_img, (224, 224))
        test_img = preprocess_input(test_img)
        ans = model.predict(test_img.reshape(1, 224, 224, 3))
        max_index = np.argmax(ans, axis=1)
        print("predict accuracy:%s" % (self.classes[max_index[0] + 1]))


data = "datas_grade/train_data.npy"
label = "label_grade/labels.npy"
mode_path = "two_mi8_iphone7/tea_grade.h5"

vgg16 = vgg((224, 224, 3), 5, data, label, mode_path)
# vgg16.generate_data()
vgg16.train(batch_size=32,epoch=100)
# vgg16.predict(imgpath)