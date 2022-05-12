# Segmentation 
For given set of images (grayscale and color). We can apply various types of thresholding and do unsupervised segmentation.

## Setup
1. From the command line create a virtual environment and activate.
```sh
# Windows
> python -m venv .venv
> .venv\Scripts\activate

# Linux
> python3 -m venv .venv
> source .venv/bin/activate
```

2. Install the dependencies.
```sh
> pip install -r requirements.txt
```

3. To run a demo of each method run the files in the *src/* folder and supply a valid image path.
```sh
> python kmeans.py ./images/beach.jpg
```