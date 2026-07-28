# picture_s
picture_s is an ai software in python that allows prompts as input or a photo to modify and outputs a new photo.

to modify a photo, place it in a folder and give the input the file destination when it asks.

To install try to use the following:

pip install diffusers transformers torch

To run, use the following:

python picture_s.py

The program will create a /pix folder where it is being run when it runs where you can view your output.

DISCLAIMER: this program is in development and installation dependencies take a lot of memory. please report issues with the understanding your system may not be equipped with proper hardware and gpus and shit.



Running it on Windows

First install PyTorch using the official selector for your specific NVIDIA CUDA version—or select CPU when no NVIDIA GPU is available. PyTorch’s installation command differs according to the hardware and CUDA build.

Then run:

pip install -r requirements_picture_s.txt
python picture_s.py

Text-to-image from the command line:

python picture_s.py --prompt "A futuristic Philadelphia skyline at sunset"

Image-to-image:

python picture_s.py --image "C:\Pictures\source.jpg" --prompt "Transform this into an impressionist oil painting"

For reproducible results:

python picture_s.py --prompt "A castle above the clouds" --seed 42

The corrected file should replace the repository’s present picture_s.py.
