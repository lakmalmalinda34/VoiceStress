import os
from flask import Flask, request, jsonify
import numpy as np
import librosa
import librosa.display
import numpy as np
import pandas as pd
import keras

input_duration=3

app = Flask(__name__)


@app.route('/predict', methods=['POST'])
def predict_stress():
    try:
        # Check if a file was uploaded in the request
        if 'file' not in request.files:
            return jsonify({'error': 'No file part'})
        save_path = "temp.wave"
        request.files['file'].save(save_path)

        # Check if the saved file exists
        if not os.path.exists(save_path):
            return jsonify({'error': 'File not saved properly'})

        data = preprocess(save_path)
        model = keras.models.load_model('Model\model_last.h5')
        prediction = model.predict(data)
        prediction = prediction.argmax(axis=1)
        prediction = prediction.astype(int).flatten()
        labels = ['negative', 'neutral', 'positive']
        prediction = labels[int(prediction)]
        os.remove(save_path)
        return jsonify({'prediction': prediction})

    except FileNotFoundError as e:
        return jsonify({'error': 'File not found'})

    except Exception as e:
        return jsonify({'error': str(e)})


def preprocess(audio):
    samples, sample_rate = librosa.load(audio, res_type='kaiser_fast',duration=input_duration, sr=22050*2,offset=0.5)
    trimmed , index = librosa.effects.trim(samples, top_db=30)
    sample_rate = np.array(sample_rate)
    mfccs = np.mean(librosa.feature.mfcc(y=trimmed, sr=sample_rate, n_mfcc=13), axis=0)
    feature = mfccs
    data = pd.DataFrame(columns=['feature'])
    data.loc[0] = [feature]
    data.loc[1] = [np.zeros((259,))]
    trimmed_df = pd.DataFrame(data['feature'].values.tolist())
    trimmed_df = trimmed_df.fillna(0)
    X_test = np.array(trimmed_df)
    x_testcnn = np.expand_dims(X_test, axis=2)
    new_array = np.delete(x_testcnn, 1, axis=0)

    return new_array

if __name__ == '__main__':
    app.run(host='0.0.0.0')
