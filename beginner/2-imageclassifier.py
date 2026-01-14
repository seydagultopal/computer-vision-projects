import os # for file path operations
import pickle # for saving and loading models
from skimage.io import imread # for reading images
from skimage.transform import resize # for resizing images
import numpy as np # for numerical operations
from sklearn.model_selection import GridSearchCV, train_test_split # for model selection and data splitting
from sklearn.svm import SVC # Support Vector Classifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score # HATA BURADAYDI: accuracy_score eklendi

# prepare data
input_dir = "c:/Users/MONSTER/Masaüstü/computer vision/beginner/clf-data/" # directory containing images
categories = ["empty", "not_empty"] # two categories: empty and not_empty

data = []
labels = []
for category_idx, category in enumerate(categories):
    # Klasör yolunu kontrol ederek ilerlemek daha güvenlidir
    category_path = os.path.join(input_dir, category)
    for file in os.listdir(category_path):
        img_path = os.path.join(category_path, file) # create full path to image
        img = imread(img_path) # read image as array
        img = resize(img, (15,15)) # resize image to 15x15
        data.append(img.flatten()) # flatten 15x15x3 image to 675 element array
        labels.append(category_idx) # append label (0 or 1)

data = np.asarray(data) # convert list to numpy array
labels = np.asarray(labels) # convert list to numpy array

# train \ test split
x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels) # 20% data for testing

# train classifier
classifier = SVC() # create SVM classifier object

parameters = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}] # parameter combinations
grid_search = GridSearchCV(classifier, parameters) # create grid search object
grid_search.fit(x_train, y_train) # x_train: features for training, y_train: labels for training

# test performance
best_estimator = grid_search.best_estimator_ # get the best model from grid search

y_prediction = best_estimator.predict(x_test) # store predicted labels

# Skoru hesapla
score = accuracy_score(y_test, y_prediction) # calculate accuracy
print("{}% of samples were classified correctly".format(str(score * 100))) # print accuracy

# Detaylı rapor (isteğe bağlı, performansı görmek için iyidir)
print("\nClassification Report:\n", classification_report(y_test, y_prediction))

# Modeli kaydet
with open("./model.p", 'wb') as f:
    pickle.dump(best_estimator, f) # save the trained model using 'with' for safer file handling