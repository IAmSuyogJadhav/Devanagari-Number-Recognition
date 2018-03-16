from keras.models import model_from_json
from matplotlib.pyplot import imshow, show
import skimage.transform
from skimage import io
import numpy as np

print('Loading model.....', end='')
with open('./model 91.49%.json', 'r') as model_file:
    model = model_from_json(model_file.read())
model.load_weights('./weights 91.49%.h5')

print('Done!')

model.compile('adam', 'categorical_crossentropy', metrics=['accuracy'])


def imgreader():
    print('Input the full path of the image file.\n\t>> ', end='')
    path = input()

    try:
        img_raw = io.imread(path, as_grey=True)
        x = skimage.transform.resize(img_raw, (36, 36)).flatten().reshape(1, 1296)
        print('Received Image:')
        imshow(img_raw)
        show()
    except:
        print('The file was not found. Please specify the correct path.')
        imgreader()
    prediction = model.predict(x)

    print('**********************************')
    print('Prediction: ', prediction)
    print('**********************************')
    test_another()


def test_another():
    x = input('Want to test another one?[y/n]')
    if x == 'y':
        imgreader()
    elif x=='n':
        pass
    else:
        print('Sorry, didn\'t get that. Try again.')
        test_another()


imgreader()

print("Exiting...")
