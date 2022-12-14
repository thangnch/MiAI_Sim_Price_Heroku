import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
from flask import Flask, render_template, request
from flask import Markup

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
from tensorflow.keras.optimizers import Adam

import random
from numpy import random
import numpy as np


def build_model_and_load(weigth_path):
    model = Sequential()
    # 3 lớp LSTM chồng nhau có trả về sequence
    model.add(LSTM(units=512, return_sequences=True, input_shape=(9, 1)))
    model.add(Dropout(0.2))
    model.add(LSTM(units=512, return_sequences=True))
    model.add(Dropout(0.2))
    model.add(LSTM(units=512, return_sequences=True))
    model.add(Dropout(0.2))
    # 1 lớp LSTM không trả về sequence
    model.add(LSTM(units=128, return_sequences=False))
    model.add(Dropout(0.2))
    # Đưa qua 1 lớp Dense
    model.add(Dense(units=512))
    # Lớp output ra kết quả
    model.add(Dense(units=6, activation="softmax"))

    optimizer = Adam()
    model.compile(loss="categorical_crossentropy", metrics=["accuracy"], optimizer=optimizer)

    model.load_weights(weigth_path)
    return model


# Khởi tạo Flask
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = "static"

# Load model
weights = 'models/ckpt_best.hdf5'
model = build_model_and_load(weights)


# Hàm xử lý request
@app.route("/", methods=['GET', 'POST'])
def home_page():
    # Nếu là POST (gửi file)
    if request.method == "POST":
         try:
            # Lấy file gửi lên
            simno = request.form['simno']
            simno = ''.join(i for i in simno if i.isdigit())
            if len(simno)==10:

                simarr = [int(c) for c in str(simno[1:])]
                result = model.predict(np.expand_dims(simarr, axis=0))
                result = np.argmax(result)

                return_value = ["dưới 500K","500K","1M","3M","5M","trên 5M"]

                # Trả về kết quả
                return render_template("index.html", simno = simno,  msg="Thày phán: SIM của bạn đáng giá " + return_value[result])

            else:
                # Nếu không có file thì yêu cầu tải file
                return render_template('index.html', msg='Nhập vào số SIM gồm 10 chữ số!')

         except Exception as ex:
            # Nếu lỗi thì thông báo
            print(ex)
            return render_template('index.html', msg='Nhập vào số SIM gồm 10 chữ số')

    else:
        # Nếu là GET thì hiển thị giao diện upload
        return render_template('index.html')


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=False, port=80, use_reloader=False)
