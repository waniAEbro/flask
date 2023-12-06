from flask import Flask, jsonify, request, render_template
from flask_cors import CORS, cross_origin
import pandas as pd
from sklearn.preprocessing import (StandardScaler,LabelEncoder)
from sklearn.decomposition import PCA
import numpy as np
import tensorflow as tf

app = Flask(__name__)

cors = CORS(app)
app.config["CORS_HEADERS"] = "Content-Type"

tambahan = pd.read_csv("dataset_unsw.csv")

# preprocessing
def preprocessing(data):
    data = pd.read_csv(data)
    data = pd.concat([data, tambahan], ignore_index=True)
    data = data.drop(columns=["attack_cat", "id"])
    numeric_columns = ['dur', 'spkts', 'dpkts', 'sbytes', 'dbytes', 'rate', 'sttl', 'dttl',
       'sload', 'dload', 'sloss', 'dloss', 'sinpkt', 'dinpkt', 'sjit', 'djit',
       'swin', 'stcpb', 'dtcpb', 'dwin', 'tcprtt', 'synack', 'ackdat', 'smean',
       'dmean', 'trans_depth', 'response_body_len', 'ct_srv_src',
       'ct_state_ttl', 'ct_dst_ltm', 'ct_src_dport_ltm', 'ct_dst_sport_ltm',
       'ct_dst_src_ltm', 'is_ftp_login', 'ct_ftp_cmd', 'ct_flw_http_mthd',
       'ct_src_ltm', 'ct_srv_dst', 'is_sm_ips_ports']
    x = data.loc[:, numeric_columns].values
    data_normalisasi=StandardScaler().fit_transform(x)
    pca = PCA(n_components=7)
    principal_components = pca.fit_transform(data_normalisasi)
    principal_df = pd.DataFrame(data=principal_components, columns=['Component 1', 'Component 2', 'Component 3', 'Component 4', 'Component 5', 'Component 6', 'Component 7'])
    df_final = pd.concat([principal_df, data[['proto', 'service', 'state','label']]], axis=1)
    df_final = pd.get_dummies(df_final, columns=['proto', 'service', 'state'], prefix="", prefix_sep="")
    df_final = df_final.drop(columns=["label"])
    df_final = df_final.iloc[:-165]
    df_final = np.array(df_final)
    df_final = np.reshape(df_final, (df_final.shape[0], df_final.shape[1], 1))
    return df_final

@app.route("/", methods=["GET"])
@cross_origin()
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
@cross_origin()
def api():
    data = request.files['file']
    df = preprocessing(data)
    y_tensor = tf.convert_to_tensor(df, dtype=tf.int64)
    model = tf.keras.models.load_model('model_unsw.h5')
    y_pred = model.predict(y_tensor)
    return jsonify(y_pred.tolist())

if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0')