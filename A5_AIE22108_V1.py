import os
import cv2
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
# Define the path to the directory containing the training data
train_data_path = r"train"

# Initialize lists to store file paths and labels
filepaths = []
labels = []

# Get the list of subdirectories (class labels)
folds = os.listdir(train_data_path)

# Iterate over each subdirectory
for fold in folds:
    # Get the full path to the subdirectory
    f_path = os.path.join(train_data_path, fold)
    # Get the list of file names in the subdirectory
    filelists = os.listdir(f_path)
    
    # Iterate over each file in the subdirectory
    for file in filelists:
        # Get the full path to the file
        filepaths.append(os.path.join(f_path, file))
        # Store the label (subdirectory name) for the file
        labels.append(fold)

# Initialize a list to store image vectors
images = []

# Iterate over each file path
for filepath in filepaths:
    # Read the image from the file
    img = cv2.imread(filepath)
    # Resize the image to a fixed size
    img = cv2.resize(img, (100, 100))  # Adjust the size as needed
    # Flatten the image into a 1D array
    img_vector = img.flatten()
    # Append the flattened image vector to the list
    images.append(img_vector)

# Convert the list of image vectors to a numpy array
images_array = np.array(images)

# Create a DataFrame to store the image vectors and labels
df = pd.DataFrame(images_array, columns=[f"pixel_{i}" for i in range(images_array.shape[1])])
df['label'] = labels

print("Shape of DataFrame:", df.shape)
print("Head of DataFrame:", df.head())

# Separate data into two classes: "normal" and "OSSC"
normal_class = df[df['label'] == 'Normal']
oscc_class = df[df['label'] == 'OSCC']



# Assuming X contains your feature vectors and y contains the corresponding class labels
# Assuming 'df' is your DataFrame containing the dataset
X = df.drop('label', axis=1)  # Features (pixel_0 to pixel_29999)
y = df['label']  # Class labels
# Splitting the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Check the shape of the training and testing sets
print("Shape of X_train:", X_train.shape)
print("Shape of X_test:", X_test.shape)
print("Shape of y_train:", y_train.shape)
print("Shape of y_test:", y_test.shape)


# Assuming X_train and y_train are your training feature vectors and corresponding class labels
# Assuming X_test and y_test are your testing feature vectors and corresponding class labels

# Initialize the kNN classifier with k=3
neigh = KNeighborsClassifier(n_neighbors=3)

# Train the classifier on the training data
neigh.fit(X_train, y_train)

# Predict the class labels for the test data
y_pred = neigh.predict(X_test)

# Evaluate the classifier
accuracy = np.mean(y_pred == y_test)
print("Accuracy:", accuracy)


# Assuming 'neigh' is your trained kNN classifier
accuracy = neigh.score(X_test, y_test)
print("Accuracy:", accuracy)

# Assuming 'neigh' is your trained kNN classifier
accuracy = neigh.score(X_test, y_test)
print("Accuracy:", accuracy)
# Assuming 'neigh' is your trained kNN classifier
predictions = neigh.predict(X_test)
print("Predictions:", predictions)


# Define the range of k values
k_values = np.arange(1, 12)

# Initialize lists to store accuracies for NN and kNN classifiers
accuracy_nn = []
accuracy_knn = []

# Train and test the classifiers for each value of k
for k in k_values:
    # Train NN classifier
    nn_classifier = KNeighborsClassifier(n_neighbors=1)
    nn_classifier.fit(X_train, y_train)
    nn_pred = nn_classifier.predict(X_test)
    accuracy_nn.append(accuracy_score(y_test, nn_pred))
    print(f"Accuracy for Nearest Neighbor (k=1) with k={k}: {accuracy_nn[-1]}")
    
    # Train kNN classifier
    knn_classifier = KNeighborsClassifier(n_neighbors=k)
    knn_classifier.fit(X_train, y_train)
    knn_pred = knn_classifier.predict(X_test)
    accuracy_knn.append(accuracy_score(y_test, knn_pred))
    print(f"Accuracy for kNN (k={k}) with k={k}: {accuracy_knn[-1]}")

# Plotting the accuracies
plt.plot(k_values, accuracy_nn, label='Nearest Neighbor (k=1)')
plt.plot(k_values, accuracy_knn, label='kNN (k=3)')
plt.xlabel('k')
plt.ylabel('Accuracy')
plt.title('Accuracy vs. k for Nearest Neighbor and kNN classifiers')
plt.legend()
plt.show()


# Predictions on training data
y_train_pred = knn_classifier.predict(X_train)
# Predictions on test data
y_test_pred = knn_classifier.predict(X_test)

# Confusion matrix for training data
conf_matrix_train = confusion_matrix(y_train, y_train_pred)
print("Confusion Matrix for Training Data:")
print(conf_matrix_train)

# Confusion matrix for test data
conf_matrix_test = confusion_matrix(y_test, y_test_pred)
print("\nConfusion Matrix for Test Data:")
print(conf_matrix_test)

# Classification report for training data
print("\nClassification Report for Training Data:")
print(classification_report(y_train, y_train_pred))

# Classification report for test data
print("\nClassification Report for Test Data:")
print(classification_report(y_test, y_test_pred))

#A2

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Load the data
df = pd.read_excel('Lab Session1 Data.xlsx', sheet_name='Purchase data')

# Create a binary classification target variable
df['Category'] = df['Payment (Rs)'].apply(lambda x: 'RICH' if x > 200 else 'POOR')

# Encode the categorical target variable
label_encoder = LabelEncoder()
df['Category'] = label_encoder.fit_transform(df['Category'])

# Extracting features and target variable
X = df.iloc[:, 1:4]  # Features: Candies, Mangoes, Milk Packets
y = df['Category']  # Numerical Target: 1 for RICH, 0 for POOR

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train_std = scaler.fit_transform(X_train)
X_test_std = scaler.transform(X_test)

# Creating and training the K-NN classifier model
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train_std, y_train)

# Predictions on training set
y_train_pred = knn_classifier.predict(X_train_std)

# Predictions on test set
y_test_pred = knn_classifier.predict(X_test_std)

# Classification metrics
mse_train = mean_squared_error(y_train, y_train_pred)
rmse_train = np.sqrt(mse_train)
mae_train = mean_absolute_error(y_train, y_train_pred)
r2_train = r2_score(y_train, y_train_pred)

mse_test = mean_squared_error(y_test, y_test_pred)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_test_pred)
r2_test = r2_score(y_test, y_test_pred)

print("Classification Metrics (Training):")
print("MSE:", mse_train)
print("RMSE:", rmse_train)
print("MAE:", mae_train)
print("R2 Score:", r2_train)

print("\nClassification Metrics (Test):")
print("MSE:", mse_test)
print("RMSE:", rmse_test)
print("MAE:", mae_test)
print("R2 Score:", r2_test)

#A3
import matplotlib.pyplot as plt
import numpy as np

# Generate random data
np.random.seed(42)
X_train_a3 = np.random.uniform(1, 10, (20, 2))
y_train_a3 = np.random.choice([0, 1], 20)

# Separate data points for class 0 and class 1
class_0_points = X_train_a3[y_train_a3 == 0]
class_1_points = X_train_a3[y_train_a3 == 1]

# Scatter plot
plt.scatter(class_0_points[:, 0], class_0_points[:, 1], color='blue', label='Class 0')
plt.scatter(class_1_points[:, 0], class_1_points[:, 1], color='red', label='Class 1')

plt.title("Scatter Plot of Training Data")
plt.xlabel("X")
plt.ylabel("Y")
plt.legend()
plt.show()

#A4
from sklearn.neighbors import KNeighborsClassifier

# Generate test set data
x_test_values = np.arange(0, 10.1, 0.1)
y_test_values = np.arange(0, 10.1, 0.1)

X_test_a4 = np.array(np.meshgrid(x_test_values, y_test_values)).T.reshape(-1, 2)

# Classify using kNN classifier
knn_classifier = KNeighborsClassifier(n_neighbors=3)
knn_classifier.fit(X_train_a3, y_train_a3)
y_test_pred_a4 = knn_classifier.predict(X_test_a4)

# Scatter plot of test data output
plt.scatter(X_test_a4[:, 0], X_test_a4[:, 1], c=y_test_pred_a4, cmap=plt.cm.Paired)

plt.title("Scatter Plot of Test Data Output (kNN, k=3)")
plt.xlabel("X")
plt.ylabel("Y")
plt.show()

#A5
k_values = [1, 5, 10]

for k in k_values:
    knn_classifier_a5 = KNeighborsClassifier(n_neighbors=k)
    knn_classifier_a5.fit(X_train_a3, y_train_a3)
    y_test_pred_a5 = knn_classifier_a5.predict(X_test_a4)

    plt.scatter(X_test_a4[:, 0], X_test_a4[:, 1], c=y_test_pred_a5, cmap=plt.cm.Paired)
    plt.title(f"Scatter Plot of Test Data Output (kNN, k={k})")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.show()
    #A6
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import cv2
import numpy as np
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial import distance
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix, classification_report
# Define the path to the directory containing the training data
train_data_path = r"train"

# Initialize lists to store file paths and labels
filepaths = []
labels = []

# Get the list of subdirectories (class labels)
folds = os.listdir(train_data_path)

# Iterate over each subdirectory
for fold in folds:
    # Get the full path to the subdirectory
    f_path = os.path.join(train_data_path, fold)
    # Get the list of file names in the subdirectory
    filelists = os.listdir(f_path)
    
    # Iterate over each file in the subdirectory
    for file in filelists:
        # Get the full path to the file
        filepaths.append(os.path.join(f_path, file))
        # Store the label (subdirectory name) for the file
        labels.append(fold)

# Initialize a list to store image vectors
images = []

# Iterate over each file path
for filepath in filepaths:
    # Read the image from the file
    img = cv2.imread(filepath)
    # Resize the image to a fixed size
    img = cv2.resize(img, (100, 100))  # Adjust the size as needed
    # Flatten the image into a 1D array
    img_vector = img.flatten()
    # Append the flattened image vector to the list
    images.append(img_vector)

# Convert the list of image vectors to a numpy array
images_array = np.array(images)

# Create a DataFrame to store the image vectors and labels
df = pd.DataFrame(images_array, columns=[f"pixel_{i}" for i in range(images_array.shape[1])])
df['label'] = labels
# Assuming df is your DataFrame containing the dataset
# Extract pixel_0 and pixel_1 columns
pixel_0 = df['pixel_0']
pixel_1 = df['pixel_1']

# Generate random classes for the data points
np.random.seed(42)
classes = np.random.randint(0, 2, size=len(df))

# Create scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(pixel_0[classes == 0], pixel_1[classes == 0], color='blue', label='Class 0 (Blue)')
plt.scatter(pixel_0[classes == 1], pixel_1[classes == 1], color='red', label='Class 1 (Red)')
plt.title('Scatter Plot of Training Data')
plt.xlabel('Pixel 0')
plt.ylabel('Pixel 1')
plt.legend()
plt.grid(True)
plt.show()





