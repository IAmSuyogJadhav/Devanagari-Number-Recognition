import numpy as np
import matplotlib.pyplot as plt
from skimage.transform import resize
from skimage import io
from functions import  predict

w1 = np.load('./cost= 155.76 acc= 98.16/w1.npy')
w2 = np.load('./cost= 155.76 acc= 98.16/w2.npy')
w3 = np.load('./cost= 155.76 acc= 98.16/w3.npy')


def imgreader():
    print('Input the full path of the image file.\n\t>> ', end='')
    path = input()
    
    try:
        img_raw = io.imread(path, as_grey=True)
        x = resize(img_raw, (36, 36)).flatten().reshape(1, 1296)
        print('Received Image:')
        plt.imshow(img_raw)
        plt.show()
        
        prediction = predict(x, w1, w2, w3)
        
        print('**********************************')
        print('Prediction: ', int(prediction))
        print('**********************************')
        test_another()
    except:
        print('The file was not found. Please specify the correct path.')
        imgreader()

    
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
