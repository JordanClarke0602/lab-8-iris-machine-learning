# Copyright (c) Streamlit Inc. (2018-2022) Snowflake Inc. (2022)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import streamlit as st
from streamlit.logger import get_logger
import numpy as np
import joblib
import sklearn

LOGGER = get_logger(__name__)


def run():
    st.set_page_config(
        page_title="Hello",
        page_icon="👋",
    )

    #load Scaler
    scaler = joblib.load("iris-scaler.pkl")

    #load Model
    model = joblib.load("svc_model.pkl")

    st.write("# Welcome to the Iris Classifier 👋")

    st.sidebar.success("Select a demo above.")

    st.markdown(
        """
       This application classifies Iris flowers into four classes: The classes here

       Input the values for each feature to classify your flower 
       """
    )

    sepal_length = st.text_input(label = 'sepal_length')
    sepal_width = st.text_input(label = 'sepal_width')
    petal_length = st.text_input(label = 'petal_length')
    petal_width = st.text_input(label = 'petal_width')
    attributes_list = eval('[' + st.text_input(label = 'parameters list') + ']')

    if st.button('Submit'):
        if len(attributes_list) > 0: # when there is a values
            st.write(f'The value you submitted are: ', attributes_list )
            user_iris = np.array([attributes_list])
        else : #No list of values. Use the first four fields instead
            st.write(f' The values you submitted are : ' , sepal_length, sepal_width, petal_length, petal_width)
            user_iris = np.array([[sepal_length, sepal_width , petal_length, petal_width]])
        

        #Scale the inputs
        user_iris_scaled = scaler.transform(user_iris)
        st.write(f'Scaled Data: {user_iris_scaled}')

        # Use the model to predict
        results = model.predict(user_iris_scaled) #

        st.write(f' THeir results are : {results}')
        iris_classes = [' Iris-Setosa' , 'Iris-Versicolor', 'Iris-Viginica']
        for i in results:
            st.write(f' Your Iris is of type : {iris_classes[i]}')




if __name__ == "__main__":
    run()
