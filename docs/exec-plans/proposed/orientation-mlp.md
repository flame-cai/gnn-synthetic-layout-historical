But some ambiguity will still remain. To solve this, we will train a simple orientation detector MLP, which will predict the right orientation.

This MLP will take a latent hidden state of the OCR model as input, and will predict the orientation of the text-line. This prediction will be used to select the right orientation, fo

So the vertical, circular, and curved lines (not horizontal) in e

We will handle this using a simple trainable MLP



    - During training, use the CER of the orientations line to detect the right orientation. Then use these as labels (with OCR model final layer hidden states as inputs) to train a small MLP classifier to decide orientation during inference.