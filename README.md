# Deep Fake Face Detection using Deep Learning

## Project Overview

Deep Fake Face Detection using Deep Learning is a web-based application developed to identify whether an uploaded facial image is **Real** or **Fake**. The project uses multiple deep learning models to analyze facial features and detect manipulated images with high accuracy. This system helps identify digitally altered media and supports the prevention of misinformation.

## Features

* Detects Real and Fake facial images.
* Image upload and prediction.
* User-friendly web interface.
* Deep learning-based classification.
* Fast and accurate prediction results.

## Deep Learning Models Used

* Custom CNN (Convolutional Neural Network)
* VGG16 (Transfer Learning)
* ResNet50 (Transfer Learning)

These models were trained and evaluated to compare their performance in detecting deepfake facial images.
## Technologies Used

* Programming Language: Python
* Framework: Flask
* Deep Learning: TensorFlow, Keras
* Deep Learning Models: Custom CNN, VGG16, ResNet50
* Image Processing: OpenCV, NumPy
* Frontend: HTML
* Database: SQLite
* Development Environment: Jupyter Notebook

## Project Structure

* app.py
* db_setup.py
* requirements.txt
* db.sqlite
* DeepLearningCode.ipynb
* templates/

  * index.html
  * login.html
  * register.html
  * predict.html
* static/
* README.md

## Installation

1. Clone this repository.
2. Install the required libraries:

   ```
   pip install -r requirements.txt
   ```
3. Run the application:

   ```
   python app.py
   ```
4. Open the application in your web browser.

---

## How It Works

1. Upload a facial image.
2. The image is preprocessed using OpenCV.
3. The trained deep learning models analyze the image.
4. The system predicts whether the uploaded image is **Real** or **Fake**.
5. The prediction result is displayed on the screen.

## Dataset

The project uses a Deep Fake Face Dataset collected from Kaggle for training and testing the deep learning models.

## Future Enhancements

* Real-time webcam detection.
* Video deepfake detection.
* Higher model accuracy using advanced architectures.
* Cloud deployment.
* Mobile application support.

## Author

**Pallavi Kolusu**

## License

This project is developed for academic and educational purposes.
