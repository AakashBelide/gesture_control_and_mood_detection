# gesture_control_and_mood_detection
Store the different moods using [OpenCV](https://opencv.org/), [dlib](http://dlib.net/) and train [KNN](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) model so that it can predict your mood. Also, used dlib [facial landmarking](https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv-python/) to control the keyboard scroll moving with mouth (Gesture control)

## Python libraries used:
> * [OpenCV](https://opencv.org/)
> * [dlib](http://dlib.net/)
> * [Numpy](https://numpy.org/)
> * [os](https://docs.python.org/3/library/os.html)
> * [keyboard](https://pypi.org/project/keyboard/)
> * [sklearn](https://scikit-learn.org/stable/)

## Directions of use:
### For mood detection:
> * First run the face_landmark.py file, provide the input mood and then click "c" on the kerboard to capture the mood/expression to save it as npy file. You can do this for multiple moods by running the file multiple times,
> * After you have done this, headover to the mood_detect.py file and when you run it, you can see the mood of your facial expression being printed based on it's training.

### For gesture control:
> * The facial landmarking of upper lip and lower lip is respectively 62 and 66 respectively. We have defined a function which triggers the keyboard "down" button whenever the difference between the lower and upper lip is less than 5, and when it is less than 5 (i.e. the mouth is closed), it releases the button.
> * You can use this to scroll through web pages or you can customize the code to play various games like the dinosaur game.
