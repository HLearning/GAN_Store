from keras.datasets import mnist
from keras.layers import *
from keras.models import Sequential, Model
from keras.optimizers import Adam
import matplotlib.pyplot as plt
import os
import numpy as np

class GAN():
    def __init__(self):
        self.img_rows = 28
        self.img_cols = 28
        self.channels = 1
        self.img_shape = (self.img_rows, self.img_cols, self.channels)
        self.latent_dim = 100

        optimizer = Adam(0.0002, 0.5)

        # 设置判别器
        self.D = self.D()
        self.D.compile(loss='binary_crossentropy',optimizer=optimizer, metrics=['accuracy'])

        # 设置生成器
        self.G = self.G()

        # 噪声输入G， 生成图片
        z = Input(shape=(self.latent_dim,))
        img = self.G(z)

        # For the combined model we will only train the G
        self.D.trainable = False

        # 将G生成的数据， 传入D去判定
        validity = self.D(img)

        # 组合模型
        # 训练 G 去迷惑 D
        self.combined = Model(z, validity)
        self.combined.compile(loss='binary_crossentropy', optimizer=optimizer)
        #self.combined.summary()

    def G(self):
        model = Sequential()
        model.add(Dense(256, input_dim=self.latent_dim))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(1024))
        model.add(LeakyReLU(alpha=0.2))
        model.add(BatchNormalization(momentum=0.8))
        model.add(Dense(np.prod(self.img_shape), activation='tanh'))
        model.add(Reshape(self.img_shape))
        #model.summary()

        noise = Input(shape=(self.latent_dim,))
        img = model(noise)

        return Model(noise, img)

    def D(self):
        model = Sequential()
        model.add(Flatten(input_shape=self.img_shape))
        model.add(Dense(512))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(256))
        model.add(LeakyReLU(alpha=0.2))
        model.add(Dense(1, activation='sigmoid'))
        #model.summary()

        img = Input(shape=self.img_shape)
        validity = model(img)

        return Model(img, validity)

    def train(self, epochs, batch_size=128, sample_interval=50):

        # 加载数据
        (X_train, _), (_, _) = mnist.load_data()

        # 数据缩放到[-1, 1]
        X_train = X_train / 127.5 - 1.
        X_train = np.expand_dims(X_train, axis=3)

        # 两类数据的标签值
        valid = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))

        for epoch in range(epochs):

            # ---------------------
            #  训练 D
            # ---------------------
            # 随机挑选一个batch的数据
            idx = np.random.randint(0, X_train.shape[0], batch_size)
            imgs = X_train[idx]

            # 生成一个batch的噪声， 数据范围[0, 1]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # 生成一批新图
            gen_imgs = self.G.predict(noise)

            # 训练判别器
            d_loss_real = self.D.train_on_batch(imgs, valid)
            d_loss_fake = self.D.train_on_batch(gen_imgs, fake)
            d_loss = 0.5 * np.add(d_loss_real, d_loss_fake)

            # ---------------------
            #  训练 G
            # ---------------------
            # 生成一个batch的噪声， 数据范围[0, 1]
            noise = np.random.normal(0, 1, (batch_size, self.latent_dim))

            # 训练G， 让噪声接近真实图
            g_loss = self.combined.train_on_batch(noise, valid)

            # 打印训练结果
            print("epoch：%5d, [D loss: %f, acc.: %.2f%%] [G loss: %f]" % (epoch, d_loss[0], 100*d_loss[1], g_loss))

            # 保存生成的样例结果
            if epoch % sample_interval == 0:
                self.sample_images(epoch)

    def sample_images(self, epoch):
        r, c = 5, 5
        noise = np.random.normal(0, 1, (r * c, self.latent_dim))
        gen_imgs = self.G.predict(noise)
        # 预测结果在[-1,1]之间， 缩放到[0, 1]
        gen_imgs = 0.5 * gen_imgs + 0.5

        fig, axs = plt.subplots(r, c)
        cnt = 0
        for i in range(r):
            for j in range(c):
                axs[i,j].imshow(gen_imgs[cnt, :,:,0], cmap='gray')
                axs[i,j].axis('off')
                cnt += 1

        save_path = "./images/"
        if not os.path.isdir(save_path):
            os.makedirs(save_path)
        fig.savefig(save_path + "%d.png" % epoch)
        plt.close()

if __name__ == '__main__':
    gan = GAN()
    gan.train(epochs=50000, batch_size=32, sample_interval=200)