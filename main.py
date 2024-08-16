import cv2
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from tensorflow.keras.preprocessing.image import ImageDataGenerator

def preprocessing_image(image_path): 
    #first read the image
    img = cv2.imread(image_path)
    #second convert the image to gryscale
    img_gray = cv2.cvtColor(img,cv2.COLOR_RGB2GRAY)
    #third apply binary threshold
    _,binary = cv2.threshold(img_gray,128,255,cv2.THRESH_BINARY_INV)
    #fourth remove noise
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    cleaned = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel)
    
    return cleaned

def detect_text_regions(image):
    # Find contours
    contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    text_regions = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        text_regions.append((x, y, w, h))
        
    return text_regions

model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(128, 128, 1)),
    MaxPooling2D((2, 2)),
    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D((2, 2)),
    Flatten(),
    Dense(64, activation='relu'),
    Dense(10, activation='softmax')  # Assuming 10 classes for characters
])

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# Example data generator setup (customize based on your dataset)
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)
train_generator = datagen.flow_from_directory('path/to/data', target_size=(128, 128), color_mode='grayscale', class_mode='sparse', subset='training')
validation_generator = datagen.flow_from_directory('path/to/data', target_size=(128, 128), color_mode='grayscale', class_mode='sparse', subset='validation')

model.fit(train_generator, validation_data=validation_generator, epochs=10)

def recognize_text(region_image, model):
    # Preprocess region image
    region_image = cv2.resize(region_image, (128, 128))
    region_image = region_image.astype('float32') / 255
    region_image = region_image.reshape((1, 128, 128, 1))

    # Predict
    prediction = model.predict(region_image)
    predicted_class = prediction.argmax()

    return predicted_class

def ocr_pipeline(image_path, model):
    preprocessed_image = preprocessing_image(image_path)
    text_regions = detect_text_regions(preprocessed_image)

    extracted_text = []
    for (x, y, w, h) in text_regions:
        region_image = preprocessed_image[y:y+h, x:x+w]
        text = recognize_text(region_image, model)
        extracted_text.append(text)

    return extracted_text
