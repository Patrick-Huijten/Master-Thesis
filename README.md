# Master Thesis - An NLP Pipeline for Classification & SpanCat Information Extraction from Dutch Legal News Articles

Welcome to the codebase for my Master's Thesis. This repository contains all scripts, notebooks, and configurations used throughout the project. Below, you'll find an overview of the repository structure and setup instructions for running and compiling the Dash application.

---
## 🧪 Python Environment

- All Jupyter Notebooks were executed using **Python 3.10.9**.
- The Dash application (`app.py`) was executed using **Python 3.10.11**.

## 📁 Repository Structure

Here's a breakdown of each folder and file in the repository:

- **`Jupyter Notebooks/`**  
  Contains all Jupyter Notebooks used for developing, tuning and testing the Sector Classification and Span Detection models:
  - `Classification Pre-Processing.ipynb`: Prepares the classification dataset (consisting of pdf files) for tuning, and stores it as a csv file.
  - `Classification.ipynb`: Tunes, tests and analyzes sector classification models.
  - `Span Detection Pre-Processing.ipynb`: Prepares the manually labeled dataset obtained from Doccano for span detection.
  - `Span Detection.ipynb`: Tuning and testing of class-specific span detection through the use of Spacy SpanCat models.
  - `data/`: Contains all data used by the notebooks in this folder. This includes the outputs of the pre-processing notebooks.
  - `Old Classification Code/`: Contains previous versions of `Classification.ipynb`
  - `Old Span Detection Code/`: Contains previous versions of `Span Detection.ipynb`

 - **`Classifier data/`**  
  Contains per-class data filled and used by `app.py` to train sector classifiers.

 - **`Classifier models/`**  
  Contains sector classifier models, trained and used by `app.py`.

 - **`SpanCat data/`**  
  Contains per-class data filled and used by `app.py` to train SpanCat models.

 - **`SpanCat models/`**  
  Contains SpanCat models, trained and used by `app.py`.

- **`app.py`**  
  The Python script that loads the Dash application.

- **`Sector_Classification.py`**  
  Contains all helper functions used in `app.py`, relating to sector classification.

- **`SpanCat_code.py`**  
  Contains all helper functions used in `app.py`, relating to span detection.

- **`requirements.txt`**  
  A list of all Python dependencies required to run the `app.py` Dash app. Install these before running `app.py`.

- **`app.spec`**  
  PyInstaller configuration script that defines how to package `app.py` into an executable.

- **`ManneartsAppels_icon.ico`**  
  Contains the logo of MannaertsAppels, which is used as the icon for the resulting executable of `app.py`.

- **`.gitignore`**  
  Specifies files and folders to be ignored by Git.

---

## ▶️ Running the Dash Application

To run the Dash app locally, follow these steps:

1. **Clone the repository:**

    ```
    git clone https://github.com/Patrick-Huijten/Master-Thesis.git
    cd Master-Thesis
    ```

2. **Set up a virtual environment (if not already created):**

    ```
    python -m venv venv_thesis
    ```

3. **Activate the virtual environment:**

    - On Windows:

        ```
        .\venv_thesis\Scripts\activate
        ```

    - On macOS/Linux:

        ```
        source venv_thesis/bin/activate
        ```

4. **Install Dependencies:**
Run the following commands in the terminal:

```
pip install -r requirements.txt
```
```
python -m spacy download nl_core_news_md
```

5. **Run the application:**

    ```
    python app.py
    ```

---

## 📦 Building the Executable (PyInstaller)

The executable of the Dash app is **not included** in this repository, as its size exceeds GitHub's upload limits. However, you can create the executable yourself using the following steps (after installing all dependencies and activating the virtual environment):

1. **Install PyInstaller**:
```
pip install pyinstaller
```

2. **Find the path to the `nl_core_news_md` spaCy model**:

Run the following Python snippet to get the path to the model directory (excluding `__init__.py`):

```
import os
import nl_core_news_md
print(os.path.dirname(nl_core_news_md.__file__))
```

This will return a path like:

```
C:\Users\<YourUser>\AppData\Local\Programs\Python\Python310\Lib\site-packages\nl_core_news_md
```

3. **Run PyInstaller with the required options**:

```
pyinstaller app.py --onedir --icon=ManneartsAppels_icon.ico ^
--hidden-import=spacy_alignments ^
--hidden-import=nl_core_news_md ^
--add-data "<path-to-nl_core_news_md>;nl_core_news_md"
```
Replace `<path-to-nl_core_news_md>` with the directory path from step 2.

> ⚠️ The used PyInstaller command was designed with **Windows** in mind, On **macOS/Linux**, replace ";" with ":" and "^" with "\\"

4. **Locate the compiled executable**:<br>
   The newly created `dist/` folder contains a folder called `app/`, which contains the compiled executable.

6. **Copy the `Classifier data/`, `Classifier models/`, `SpanCat data/` and `Spancat models/` folders into `app/`**:
   This allows the resulting executable to detect, read and save data and models.
