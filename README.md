 # Sport Detection from images  
 This is a project to classify an uploaded image by the sport type from 3 kinds of sports:
   1. Cricket
   2. Football
   3. Tennis

**Check out 'app.py' file for the code.**

 The detection process incorporates a customized version of the Tensorflow object detection API model. Faster R-CNN is chosen for the object detection process as to bring in a higher accuracy on object detections with less number of training images. 60 images were used per class to train this Faster R-CNN model with 20 classes of objects that could be detected from an image uploaded.

Objects that could be detected on an uploaded image:

  1. Cricket Player
  2. Cricket Boundary
  3. Cricket Batsman
  4. Cricket Umpire
  5. Cricket Stumps
  6. Cricket Bat
  7. Cricket Pitch
  8. Cricket Ball
  9. FootBall
  10. FootBall player
  11. FootBall GoalPost
  12. FootBall DBox
  13. FootBall Box
  14. FootBall CornerFlag
  15. FootBall Circle
  16. Tennis Player
  17. Tennis Racket
  18. Tennis Net
  19. Tennis Court
  20. Tennis Ball

Then the number of detections for each object on the image is added as an observation to a pandas dataframe. The above 20 objects are the features for that dataframe and the number of detections are entered in each row adding one row per image. A row will contain the number of times each object appeared on an image. 

Then this row is fed to a Random Forest Classifier to predict the sport depending on the objects present in the image.
