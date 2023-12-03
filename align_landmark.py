# http://dlib.net/face_landmark_detection.py.html

import os
import glob
import dlib
import matplotlib.pyplot as plt

# 5 face landmarks model is used to crop and align faces
# 68 face landamarks model is used to detect precise landmark points on faces
predictor_fl5_path = "face_detection_model/shape_predictor_5_face_landmarks.dat"
predictor_fl68_path = "face_detection_model/shape_predictor_68_face_landmarks.dat"

faces_folder_path = "Strokefaces/stroke"

# Load all the models we need: a detector to find the faces, a shape predictor
# to find face landmarks so we can precisely localize the face
fl5_detector = dlib.get_frontal_face_detector()
fl5_sp = dlib.shape_predictor(predictor_fl5_path)

fl68_detector = dlib.get_frontal_face_detector()
fl68_sp = dlib.shape_predictor(predictor_fl68_path)

for f in glob.glob(os.path.join(faces_folder_path, "*.jpg")):
    # Load the image using Dlib
    img = dlib.load_rgb_image(f)

    # Make plot
    fig, ax = plt.subplots()

    # Ask the detector to find the bounding boxes of each face. The 1 in the
    # second argument indicates that we should upsample the image 1 time. This
    # will make everything bigger and allow us to detect more faces.
    fl5_dets = fl5_detector(img, 1)

    num_faces = len(fl5_dets)
    if num_faces == 0:
        print("Sorry, there were no faces found in '{}'".format(f))
        plt.imshow(img)
        plt.axis('off')
        plt.show()
        continue

    # Find the 5 face landmarks we need to do the alignment.
    faces = dlib.full_object_detections()
    for detection in fl5_dets:
        faces.append(fl5_sp(img, detection))

    # Get aligned and cropped face
    aligned_img = dlib.get_face_chip(img, faces[0])

    fl68_dets = fl68_detector(aligned_img, 5)

    for k, d in enumerate(fl68_dets):
        # Get the landmarks/parts for the face in box d.
        shape = fl68_sp(aligned_img, d)

        # Draw 68 landmark circles on plot
        for point in shape.parts():
            circle = plt.Circle((point.x, point.y), 1, color="#00FF00", fill=True)
            ax.add_patch(circle)
        
    # Display the face with 68 landmarks
    plt.imshow(aligned_img)
    plt.axis('off')
    plt.show()