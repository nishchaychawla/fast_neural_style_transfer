import cv2
import transformer
import torch
import utils
from imutils import paths
import argparse
import itertools

STYLE_TRANSFORM_PATH = "transforms/mosaic_light.pth"
PRESERVE_COLOR = False
WIDTH = 1280
HEIGHT = 720

ap = argparse.ArgumentParser()
ap.add_argument("-m","--models", required=True,
    help="path to directory containing neural style transfer models")
args = vars(ap.parse_args())

modelPaths = paths.list_files(args['models'], validExts=('.pth',))
# modelPaths = sorted(list(modelPaths))
modelIter = itertools.cycle(modelPaths)

def webcam(style_transform_path, width=1280, height=720):
    """
    Captures and saves an image, perform style transfer, and again saves the styled image.
    Reads the styled image and show in window. 

    Saving and loading SHOULD BE eliminated, however this produces too much whitening in
    the "generated styled image". This may be caused by the async nature of VideoCapture,
    and I don't know how to fix it. 
    """
    # Device
    device = ("cuda" if torch.cuda.is_available() else "cpu")

    # Load Transformer Network
    print("Loading Transformer Network")
    net = transformer.TransformerNetwork()
    modelPath = next(modelIter)
    net.load_state_dict(torch.load(modelPath))
    net = net.to(device)
    print("Done Loading Transformer Network")

    # Set webcam settings
    cam = cv2.VideoCapture(0)
    cam.set(3, width)
    cam.set(4, height)

    # Main loop
    with torch.no_grad():
        count = 1
        while True:
            # Get webcam input
            ret_val, img = cam.read()

            # Mirror 
            img = cv2.flip(img, 1)

            # Free-up unneeded cuda memory
            torch.cuda.empty_cache()
            
            # Generate image
            content_tensor = utils.itot(img).to(device)
            generated_tensor = net(content_tensor)
            generated_image = utils.ttoi(generated_tensor.detach())
            if (PRESERVE_COLOR):
                generated_image = utils.transfer_color(content_image, generated_image)
            img2 = cv2.imdecode(cv2.imencode(".png", generated_image)[1], cv2.IMREAD_UNCHANGED)

            count += 2
            # Show webcam
            cv2.imshow('Demo webcam', img2)
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('n'):
                modelPath = next(modelIter)
                net.load_state_dict(torch.load(modelPath))
                net = net.to(device)

            elif key == ord('q'): 
                break  # q to quit
        
    # Free-up memories
    cam.release()
    cv2.destroyAllWindows()

webcam(modelIter, WIDTH, HEIGHT)
