# Face Recognition using Siamese Network

This project implements face recognition using a Siamese Network. Follow the instructions below to train a new model or use the provided pre-trained model.

---

## **Training a New Model**

1. **Prepare Negative Images**  
   - Download a collection of negative images (images of faces that do not belong to the target individuals).  
   - Place all these images in a folder named `negative`.

2. **Run the `train.py` Script**  
   - To capture positive and anchor images using a webcam, uncomment the following line in the script:
     ```python
     enroll_face(img_number, pos_path, anchor_path, pos_req=True)
     ```
   - This will automatically create and store positive and anchor images in the respective folders.

   - At the end of the training process, the model will be saved as `siamesemodel.keras` in the project directory.

---

## **Using a Pre-trained Model**

If you don’t want to train a model from scratch, you can use the pre-trained model:

1. **Download the Pre-trained Model**  
   - Download the pre-trained model from [this link](https://drive.google.com/file/d/1BAzqnMf5PuAsAcHk2Thy7WNp4ZpBreyD/view?usp=drive_link).

2. **Add the Model to Your Project**  
   - Save the `siamesemodel.keras` file in the root directory of the project.

---

## **Face Recognition Workflow**

1. **Run the `main.py` Script**  
   Execute the script by running:
   ```bash
   python main.py

## **Demonstration**
    soon.....


