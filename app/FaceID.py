from kivy.app import App
from kivy.uix.boxlayout import BoxLayout

# Kivy UX imports
from kivy.uix.label import Label
from kivy.uix.button import Button
from kivy.uix.image import Image

# Kivy core imports
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger

# ML imports
import cv2
import os
from scipy.datasets import face
import tensorflow as tf
import numpy as np
from layers import L1Dist

#build the FaceID App
class CamApp(App):

    def build(self):
        # Main Layout
        self.web_cam = Image(size_hint=(1, .8))
        self.button = Button(text="Capture Face",on_press = self.verify ,size_hint=(1, .1))
        self.verification_Label = Label(text="Verification Uninitiated ", size_hint=(1, .1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.verification_Label)
        layout.add_widget(self.button)

        # load the trained model
        self.model = tf.keras.models.load_model('siamese_modelv2.keras',custom_objects={'L1Dist': L1Dist})
        
        # Setup camera
        self.capture = cv2.VideoCapture(0)
        Clock.schedule_interval(self.update, 1.0 / 33.0)

        return layout

    def update(self, *args):
        
        # Read frame from camera
        ret, frame = self.capture.read()
        frame = frame[120:370, 200:450, :]

        # Flip the frame horizontally
        buf = cv2.flip(frame, 0).tostring()
        img_texture = Texture.create(
            size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # PreProcess Function
    def preprocess(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # load the image
        img = tf.io.decode_jpeg(byte_img)
        # Resize the image to be 105x105x3
        img = tf.image.resize(img, (105,105))
        # Scale image to be between 0 and 1
        img = img / 255.0

        return img

    # Verification Function
    def verify(self, *args):
        detection_threshold=0.7 
        verification_threshold=0.5

        # Collect input image
        Save_path = os.path.join("Test", "InputImages", "InputImages.jpg")
        rat, frame = self.capture.read()
        frame = frame[120:370, 200:450, :]
        cv2.imwrite(Save_path, frame)

        results = []
        for image in os.listdir(os.path.join("Test", "VerificationImages")):
            input_img = self.preprocess(os.path.join("Test", "InputImages", "InputImages.jpg"))
            validation_img = self.preprocess(os.path.join("Test", "VerificationImages", image))

            # Add batch dimension
            input_img = np.expand_dims(input_img, axis=0)
            validation_img = np.expand_dims(validation_img, axis=0)
            
            result = self.model.predict([input_img, validation_img], verbose=0)
            results.append(result)
            
        # Detection: count matches above threshold
        detection = np.sum(np.array(results) > detection_threshold)

        # Verification: require high percentage of matches
        verification = detection / len(os.listdir(os.path.join("Test", "VerificationImages")))
        verified = verification > verification_threshold
        

        # Set verification text
        self.verification_Label.text = "Verified" if verified else "Unverified"

        Logger.info(f"Results: {results}")
        Logger.info(f"Matches: {detection} / {len(os.listdir(os.path.join('Test', 'VerificationImages')))}")
        Logger.info(f"Verification: {verification:.2f}")
        Logger.info(f"Verified: {verified}")

        return results, verified

if __name__ == "__main__":
    CamApp().run()