import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import tensorflow as tf
from keras import layers, optimizers
from sklearn.metrics import mean_absolute_percentage_error as MAPE
from sklearn.metrics import mean_squared_error as MSE
from sklearn.preprocessing import StandardScaler


index = 'data'
path = './data.csv'
sequence_length = 144            # 序列长度(输入层)
horizon = 1                     # 移动的步长
train_set = 0.6                 # 训练集
test_set = 0.25                 # 测试集
batch_size = 32                 # 一次训练所抓取的数据样本数量
epochs = 200                    # 训练次数
dense_1 = 36                    # 隐藏层1
dense_2 = 8                     # 隐藏层2
loss = 'mean_squared_error'     # 损失函数
#metrics = ['mae', 'acc']        # 评价函数
optimizer = optimizers.Adam(learning_rate=1e-3)  # 优化器
# 参考 https://keras.io/zh/


# 读取数据
def read_data(datapath):
    data1 = pd.read_csv(datapath, index_col=0)
    return data1


# 分割数据
def splitdata(data, sequence_length, horizon):
    all_data = []
    for dta in range(len(data) - sequence_length - horizon + 1):
        all_data.append(data[dta: dta + sequence_length])

    all_data = np.array(all_data)
    ydata = data[(horizon + sequence_length - 1):]
    ydata = list(ydata)

    # 加入y变量
    all_data = pd.DataFrame(all_data)
    all_data['y'] = ydata
    all_data = np.array(all_data)

    row_1 = round(train_set * int(all_data.shape[0]))
    row_2 = round(test_set * int(all_data.shape[0]))

    x_train_initial = all_data[:-int(row_1), :-1]
    y_train_initial = all_data[:-int(row_1), -1]
    x_val_initial = all_data[-int(row_1):-int(row_2), :-1]
    y_val_initial = all_data[-int(row_1):-int(row_2), -1]
    x_test_initial = all_data[-int(row_2):, :-1]
    y_test_initial = all_data[-int(row_2):, -1]

    return x_train_initial, y_train_initial, x_val_initial, y_val_initial, x_test_initial, y_test_initial

# 标准化处理
def standard(x_train_initial, y_train_initial, x_val_initial, y_val_initial, x_test_initial, y_test_initial):
    x_scaler = StandardScaler()
    y_scale = StandardScaler()

    # 分别对x与y进行标准化
    x_train = x_scaler.fit_transform(x_train_initial)
    x_val = x_scaler.transform(x_val_initial)
    x_test = x_scaler.transform(x_test_initial)

    y_train = y_scale.fit_transform(y_train_initial.reshape(-1, 1))
    y_val = y_scale.transform(y_val_initial.reshape(-1, 1))
    y_test = y_scale.transform(y_test_initial.reshape(-1, 1))

    # 生成正式的数据格式
    amount_of_features = 1
    x_train = np.reshape(
        x_train, (x_train.shape[0], x_train.shape[1], amount_of_features))
    x_val = np.reshape(
        x_val, (x_val.shape[0], x_val.shape[1], amount_of_features))
    x_test = np.reshape(
        x_test, (x_test.shape[0], x_test.shape[1], amount_of_features))

    return x_train, y_train, x_val, y_val, x_test, y_test

# keras 版本模型
def get_uncompiled_model():
    inputs = tf.keras.Input(shape=(sequence_length,), name='digits')

    x = layers.Dense(dense_1, activation='relu',
                     name='dense_1')(inputs)  # 隐藏层1
    x = layers.Dense(dense_2, activation='relu', name='dense_2')(x)  # 隐藏层2
    x = layers.Dense(1, name='dense_3')(x)  # 输出层

    model = tf.keras.Model(inputs=inputs, outputs=x)
    return model

# 转换到标准化前的格式
def inverse_data(y_train_initial, y_pre, y_test):
    y_scale = StandardScaler()
    y_scale = y_scale.fit(y_train_initial.reshape(-1, 1))
    y_pre = y_scale.inverse_transform(y_pre)
    y_test = y_scale.inverse_transform(y_test)

    return y_pre, y_test

# 画图
def plt_image(y_pre_rel, y_test_rel):
    plt.figure()
    plt.plot(y_pre_rel, 'y-', label='predictions')
    plt.plot(y_test_rel, 'r--', label='test')
    plt.legend(loc='best')
    plt.savefig('result.png')
    plt.show()


def main():
    # 读取数据
    data = read_data(path)[index]
    # 分割数据
    x_train_initial, y_train_initial, x_val_initial, y_val_initial, x_test_initial, y_test_initial = splitdata(
        data, sequence_length, horizon=horizon)
    # 标准化
    x_train, y_train, x_val, y_val, x_test, y_test = standard(
        x_train_initial, y_train_initial, x_val_initial, y_val_initial, x_test_initial, y_test_initial)

    # 训练模型
    model = get_uncompiled_model()
    # model.compile()模型配置损失和度量、优化器
    model.compile(optimizer=optimizer,
                  loss=loss, 
                  )
    # model.fit()对模型进行训练
    model.fit(x_train, y_train, batch_size=batch_size,
              epochs=epochs, validation_data=(x_val, y_val))
    # model.predict()对模型进行预测
    y_pre = model.predict(x_test)
    model.summary()

    # 转换到原来的数据格式
    y_pre_rel, y_test_rel = inverse_data(y_train_initial, y_pre, y_test)
    y_pre_rel = np.array(y_pre_rel).reshape(-1)
    y_test_rel = np.array(y_test_rel).reshape(-1)

    # error
    rmse = np.sqrt(MSE(y_pre_rel, y_test_rel))
    mape = MAPE(y_pre_rel, y_test_rel)
    print('RMSE : %.4f' % (rmse))
    print('MAPE : %.4f' % (mape))

    # 输出
    np.savetxt("result.csv", np.array(
        [y_test_rel, y_pre_rel]).transpose(), delimiter=',')

    # 画图
    plt_image(y_pre_rel, y_test_rel)


if __name__ == '__main__':
    main()
