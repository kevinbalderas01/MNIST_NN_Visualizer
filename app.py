
import streamlit as st
import json
import requests
import numpy as np
import matplotlib.pyplot as plt

URI='http://127.0.0.1:5000'

st.set_option('deprecation.showPyplotGlobalUse', False)
st.title('Neural Network Visualizer')
st.sidebar.markdown('## Input image')

if st.button('Get random prediction'):
    response = requests.post(URI,data={})
    response = json.loads(response.text)
    pred = response.get('prediction')
    img = response.get('img')
    img = np.reshape(img, (28,28))
    st.sidebar.image(img, width=150)

    for layer, p in enumerate(pred):
        numbers = np.squeeze(np.array(p))
        plt.figure(figsize=(32,4))
        if layer == 2:
            row=1
            col=10
        else:
            row=2
            col=16
        for i, number in enumerate(numbers):
            plt.subplot(row, col, i+1)
            plt.imshow(number * np.ones((8,8,3)).astype('float32'))
            plt.xticks([])
            plt.yticks([])
            if layer == 2:
                plt.xlabel(str(i), fontsize=40)
        plt.subplots_adjust(wspace=0.05, hspace=0.05)
        plt.tight_layout()
        st.text('Layer{}'.format(layer))
        st.pyplot()
else:
    response = requests.get(URI)
    st.text(response.text)
