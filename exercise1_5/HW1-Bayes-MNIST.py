import numpy as np
from scipy.stats import norm, multivariate_normal
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image, ImageDraw


class MyBayesClassifier:
    def __init__(self):
        self.class_priors = {}
        self.class_stats = {}

    def train(self, X, y):
        """
    Train the classifier under the assumption of Gaussian distributions:
      calculate priors and Gaussian distribution parameters for each class.

    Args:
    X (pd.DataFrame): DataFrame with features.
    y (pd.Series): Series with target class labels.
    """

        self.classes_ = np.unique(y)
        for class_label in self.classes_:
            # Filter data by class
            X_class = X[y == class_label]

            # Calculate prior probability for the class
            self.class_priors[class_label] = len(X_class.values) / len(X)

            # Calculate mean and covariance for the class
            # Adding a small value to the covariance for numerical stability
            self.class_stats[class_label] = {
                'mean': np.mean(X_class.values, axis=0),
                'cov': np.cov(X_class, rowvar=False, bias=True)
            }

    def predict(self, X):
        """
    Predict class labels for each test sample in X.

    Args:
    X (pd.DataFrame): DataFrame with features to predict.

    Returns:
    np.array: Predicted class labels.
    """
        predictions = X.apply(self._predict_instance, axis=1)
        return np.array(predictions)

    def _predict_instance(self, x):
        """
    Private helper to predict the class for a single instance.

    Args:
    x (pd.Series): A single data point's features.

    Returns:
    The predicted class label.
    """
        posteriors = []

        # Calculate the posterior probability for each class
        for class_label in self.classes_:
            sample_mean, sample_std = self.class_stats[class_label].values()
            p = self._calculate_likelihood(x, sample_mean, sample_std)
            posteriors.append((class_label, p * self.class_priors[class_label]))

        # Choose the class with the highest posterior probability
        prediction = max(posteriors, key=lambda z: z[1])
        return prediction[0]

    def _calculate_likelihood(self, x, mean, cov):
        """
    Calculate the Gaussian likelihood of the data x given class statistics.

    Args:
    x (pd.Series): Features of the data point.
    mean (pd.Series): Mean feature for the class.
    cov (pd.DataFrame): Covariance matrix for the class.

    Returns:
    float: The likelihood value.
    """
        if mean.shape[0] > 1:
            likelihood = multivariate_normal.pdf(x, mean=mean, cov=cov)
        else:
            likelihood = norm.pdf(x, loc=mean, scale=cov)
        return likelihood


# Calculate the bounding box
def calculate_bounding_box(image):
    # Find non-zero foreground pixels
    nonzero_pixels = np.nonzero(image)
    # Check if there are any foreground pixels
    if nonzero_pixels[0].size == 0:
        return np.nan  # Return NaN if no foreground pixels found

    # Get minimum and maximum coordinates of foreground pixels
    min_row = np.min(nonzero_pixels[0]) - 1
    max_row = np.max(nonzero_pixels[0]) + 1
    min_col = np.min(nonzero_pixels[1]) - 1
    max_col = np.max(nonzero_pixels[1]) + 1

    return min_col, min_row, max_col, max_row


# Function to calculate aspect ratio
def aspect_ratio(image):
    """Calculates the aspect ratio of the bounding box around the foreground pixels."""
    try:
        # Extract image data and reshape it (assuming data is in a column named 'image')
        img = image.values.reshape(28, 28)

        # Find non-zero foreground pixels
        nonzero_pixels = np.nonzero(img)

        # Check if there are any foreground pixels
        if nonzero_pixels[0].size == 0:
            return np.nan  # Return NaN if no foreground pixels found

        # Get minimum and maximum coordinates of foreground pixels
        min_row = np.min(nonzero_pixels[0]) - 1
        max_row = np.max(nonzero_pixels[0]) + 1
        min_col = np.min(nonzero_pixels[1]) - 1
        max_col = np.max(nonzero_pixels[1]) + 1

        # Calculate bounding box dimensions
        width = max_col - min_col
        height = max_row - min_row

        # Calculate aspect ratio
        aspect_ratio = width / height

        return aspect_ratio

    except (KeyError, ValueError) as e:
        print(f"Error processing image in row {image.name}: {e}")
        return np.nan  # Return NaN for rows with errors


def foreground_pixels(image):
    """
  Calculate the pixel density of the image, defined as the
  count of non-zero pixels

  Args:
  image (np.array): A 1D numpy array representing the image.

  Returns:
  int: The pixel density of the image.
  """
    try:
        # Extract image data and reshape it (assuming data is in a column named 'image')
        img = image.values.reshape(28, 28)

        # Find non-zero foreground pixels
        nonzero_pixels = np.nonzero(img)[0].size

        if nonzero_pixels == 0:
            print(f"Warning: Couldn't find nonzero pixels on  {image.name}")
            return np.nan  # Return NaN if no foreground pixels found
    except (KeyError, ValueError) as e:
        print(f"Error processing image in row  {image.name}: {e}")
        return np.nan  # Return NaN for rows with errors

    return nonzero_pixels


def calculate_centroid(image):
    """
    Calculate the normalized centroid (center of mass) of the image.

    Returns:
    tuple: The (x, y) coordinates of the centroid normalized by image dimensions.
    """
    # Extract image data and reshape it (assuming data is in a column named 'image')
    img = image.values.reshape(28, 28)

    rows, cols = img.shape

    x_center = 0
    y_center = 0
    total_mass = 0
    for y in range(cols):
        for x in range(rows):
            intensity = img[y, x]
            if intensity > 0:
                x_center += x * intensity
                y_center += y * intensity
                total_mass += intensity

    x_center = x_center / total_mass
    y_center = y_center / total_mass

    # Create a single scalar as a centroid feature using x+(y * w) where w is the width of the image
    centroid = x_center + (y_center * cols)

    return centroid


def min_max_scaling(X, min_val=-1, max_val=1):
    """Scales features to a range between min_val and max_val."""
    X_scaled = min_val + (max_val - min_val) * ((X - min(X)) / (max(X) - min(X)))
    return X_scaled


def visualize_bounding_box(image, color='red'):
    """Visualizes the bounding box around the digit in an image."""
    bbox = calculate_bounding_box(image)

    # Create a drawing object
    sample_image_img = Image.fromarray(image.astype(np.uint8)).convert('RGB')
    scaling = 10
    sample_image_XL = sample_image_img.resize((28 * scaling, 28 * scaling), resample=Image.NEAREST)

    draw = ImageDraw.Draw(sample_image_img)
    # Draw the rectangle with desired fill color and outline (optional)
    draw.rectangle(bbox, outline=color, width=1)

    sample_image_XL.show()
    sample_image_XL_bbox = sample_image_img.resize((28 * scaling, 28 * scaling), resample=Image.NEAREST)
    sample_image_XL_bbox.show()


##############################################################################
######    MAIN - CREATE FEATURES - TRAIN (NAIVE) BAYES CLASSIFIER
##############################################################################
def main():
    for i in range(10):
        for j in range(i, 10):
            my_labels = [i, j]
            if i == j: continue
            print(my_labels)
            # Read the training samples from the corresponding file
            nTrainSamples = None  # specify 'None' if you want to read the whole file
            df_train = pd.read_csv('data/mnist_train.csv', delimiter=',', nrows=nTrainSamples)
            df_train = df_train[df_train['label'].isin(my_labels)]  # Get samples from the selected digits only
            target_train = df_train.label
            data_train = df_train.iloc[:, 1:]

            # Read the test samples from the corresponding file
            nTestSamples = None  # specify 'None' if you want to read the whole file
            df_test = pd.read_csv('data/mnist_test.csv', delimiter=',', nrows=nTestSamples)
            df_test = df_test[df_test['label'].isin(my_labels)]  # Get samples from the selected digits only
            target_test = df_test.label
            data_test = df_test.iloc[:, 1:]

            #################### Create the features #############################
            # a)
            # Calculate aspect ratio as the first feature
            df_train['aspect_ratio'] = data_train.apply(aspect_ratio, axis=1)

            # ## Find and print the max and min aspect ratio for labels 1 and 2
            # max_1 = df_train[df_train['label'] == 1]['aspect_ratio'].max()
            # max_2 = df_train[df_train['label'] == 2]['aspect_ratio'].max()
            # min_1 = df_train[df_train['label'] == 1]['aspect_ratio'].min()
            # min_2 = df_train[df_train['label'] == 2]['aspect_ratio'].min()
            # print(max_1, max_2, min_1, min_2)

            # b)
            ## Draw 10 sample images from the training data to make sure aspect ratio is correct
            # for sample in range(10):
            #   # print(aspect_ratio(data_train.iloc[sample]))
            #   sample_image = data_train.iloc[sample].values.reshape(28, 28)
            #   visualize_bounding_box(sample_image)

            # c)
            df_train['aspect_ratio'] = min_max_scaling(df_train['aspect_ratio'])

            # d, e)
            # Create the Classifier object and train the Gaussian parameters (prior, mean, cov)
            classifier = MyBayesClassifier()

            # Train the classifier
            features = ['aspect_ratio']
            trainData = df_train[features]
            classifier.train(trainData, target_train)

            assert (sum(classifier.class_priors.values()) == 1)

            # f)
            # Predict on the test samples (for the given feature set)
            df_test['aspect_ratio'] = data_test.apply(aspect_ratio, axis=1)
            df_test['aspect_ratio'] = min_max_scaling(df_test['aspect_ratio'])
            test_data = df_test[features]
            predictions = classifier.predict(test_data)

            # g)
            # Calculate accuracy as an example of validation
            accuracy = accuracy_score(target_test, predictions)
            print(f"Classification accuracy: {round(accuracy * 100, 4)}%")

            # h)
            # Calculate the number of non-zero pixels as the second feature
            df_train['fg_pixels'] = data_train.apply(foreground_pixels, axis=1)
            df_train['fg_pixels'] = min_max_scaling(df_train['fg_pixels'])
            classifier = MyBayesClassifier()
            features = ['aspect_ratio', 'fg_pixels']
            trainData = df_train[features]
            classifier.train(trainData, target_train)
            assert (sum(classifier.class_priors.values()) == 1)
            df_test['fg_pixels'] = data_test.apply(foreground_pixels, axis=1)
            df_test['fg_pixels'] = min_max_scaling(df_test['fg_pixels'])
            test_data = df_test[features]
            predictions = classifier.predict(test_data)
            accuracy = accuracy_score(target_test, predictions)
            print(f"Classification accuracy: {round(accuracy * 100, 4) }%")


            # i)
            # Calculate the centroid feature as the third feature
            df_train['centroid'] = data_train.apply(calculate_centroid, axis=1)
            df_train['centroid'] = min_max_scaling(df_train['centroid'])
            classifier = MyBayesClassifier()
            features = ["aspect_ratio", "fg_pixels", "centroid"]
            trainData = df_train[features]
            classifier.train(trainData, target_train)
            assert (sum(classifier.class_priors.values()) == 1)
            df_test['centroid'] = data_test.apply(calculate_centroid, axis=1)
            df_test['centroid'] = min_max_scaling(df_test['centroid'])
            test_data = df_test[features]
            predictions = classifier.predict(test_data)
            accuracy = accuracy_score(target_test, predictions)
            print(f"Classification accuracy: {round(accuracy * 100, 4)}%")

###########################################################
###########################################################
if __name__ == "__main__":
    main()
