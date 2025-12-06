### SmartVision Machine Learning Model(YOLOv11)

This repository contains the source code for the computer vision component of the **Smart Vision** project. It utilizes the YOLO (You Only Look Once) architecture to perform real-time object detection and classification of retail products on store shelves.

This module is responsible for:
1.  **Preprocessing:** Enhancing shelf images (contrast/sharpness).
2.  **Detection:** Identifying product bounding boxes and counting stock.
3.  **Classification:** Recognizing specific brands to verify compliance with merchandising agreements.

---

## 🛠 Installation & Setup Instructions

Follow these steps to set up your development environment. We use a **Virtual Environment** to manage dependencies and ensure the AI model functions consistently across different machines (Windows, macOS, Linux).

### 1. Clone the Repository

Open your terminal or command prompt and run the following command to download the project files:

```bash
git clone https://github.com/SmartVision-RetailAuditing/ML-model-backend.git smart-vision-yolo
cd smart-vision-yolo
```

### 2. Create a Virtual Environment

It is critical to use a virtual environment to isolate the specific library versions required for our AI model.

**For Windows:**

```markdown
# Verify you are in the project root directory
python -m venv venv
```

**For macOS / Linux:**

```markdown
# Verify you are in the project root directory
python3 -m venv venv
```

*Note: This will create a folder named `venv` in your directory. You do not commit this folder to git due to .gitignore.*

### 3. Activate the Virtual Environment

You must activate the environment every time you work on the project. When active, your terminal prompt will usually show `(venv)`.

**For Windows:**

- **Command Prompt (cmd.exe):**
	DOS
	```terminal
	venv\Scripts\activate.bat
	```
- **PowerShell:**
	PowerShell
	```terminal
	venv\Scripts\Activate.ps1
	```
	*(Note: If you get a permission error, run `Set-ExecutionPolicy Unrestricted -Scope Process` first).*

**For macOS / Linux:**

Bash

```bash
source venv/bin/activate
```

### 4\. Install Dependencies

Once the environment is active (look for `(venv)` in your terminal), install the required packages (PyTorch, OpenCV, etc.) from the `requirements.txt` file.

```terminal
pip install --upgrade pip
pip install -r requirements.txt
```

### 5\. Deactivating the Environment

When you are finished working, you can exit the virtual environment to return to your global Python settings. This command works on all platforms:

```terminal
deactivate
```

---

## 📂 Project Structure


```markdown
smart-vision-yolo/
├── assets/             # Raw and processed datasets (Git ignored)
├── runs/               # Pre-trained YOLO weights (Git ignored)
├── utils/              # Utility functions for image processing
├── main.py             # Functions for training new model
├── requirements.txt    # Project dependencies
└── README.md           # Project documentation
    ```

## 🚀 Usage


### Train Custom Dataset

#### Dataset Implementation
1. Find a dataset that contains raw images without annotation.
2. (in venv) 
```terminal
pip install label-studio
label-studio
```
3. Create a new local account
4. Create a project and import images(max. 100 image at a time)
5. Set labeling tags (Ex. Apple, Banana, carrot etc.)
6. Label each image
7. Export -> YOLO with Images(image segmentation, object detection, keypoints)
8. Unzip the downloaded zip folder into project's "/assets" directory.

On Linux/Mac OS
(from project's root directory)
```bash
unzip ~/Downloads/<zip_file.zip> -d ./assets/<new_dataset>/
```

#### Train New Model
1. Rename 'data.yaml.example' file as 'data.yaml'
2. Fill the blanks (<...>) of the file
3. Enter virtual environment(venv) in terminal 
4. Tune training model's parameters at main.py
5. run 'python main.py'
6. new training created at runs/detect
7. move new models into root directory


### Testing with a camera

```terminal
python utils/yolo_detect.py --model <my_model.pt> --source usb0 --resolution 1280x720

```

## 🤝 Contributing

1. Create a branch for your new feature (`git switch -c feature/<feature_name>`).
2. Commit your changes.
3. Push to the branch and open a Pull Request to development branch.
