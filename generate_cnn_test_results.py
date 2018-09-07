from keras.models import model_from_json
import pandas as pd

# load json and create model
json_file = open('cnn_model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("cnn_model.h5")
print("Loaded model from disk")



x_test = pd.read_csv('/Users/karan/Documents/karan/karan_Project/test.csv')
x_test = x_test.iloc[:, :]
x_test = x_test.values
x_test = x_test.reshape(x_test.shape[0], 1, 28, 28).astype('float32')
x_test /= 255

predictions = loaded_model.predict_classes(x_test, batch_size=128, verbose=1)

df = pd.DataFrame(predictions)
df.to_csv("predictions_cnn.csv")
print('Saved the predictions!')

