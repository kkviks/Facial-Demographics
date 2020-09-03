How to use:

1. Activate the conda environment provided or pip install requirements.txt
2. Run using python app.py
3. Copy and paste the URL generated in browser to run.
4. You can now upload your photo and generate results.

Word of warning: 
Works only for humans. If you're an alien or a smart chimp, please don't feel left out. 

Find trained keras model in .h5 format in models directory for your custom application.
Model expects (200,200,3) input shape, first forward pass though "base" then specific model. Cropping from half forehead to chin.

Possible future works:
Integrating model with real-time prediction in videos.
