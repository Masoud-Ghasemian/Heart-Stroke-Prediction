# Heart-Stroke-Prediction


![Demo](https://user-images.githubusercontent.com/53497388/116324515-49c87280-a78e-11eb-959b-68074e420303.gif)


This web application uses A **Random Forest Classifier** with 150 trees and entropy criterion to predict how likely is
a patient will have a heart stroke. A **GridSearch CV** was used to fine tune the hyperparameters. I used [this kaggle dataset](https://www.kaggle.com/fedesoriano/stroke-prediction-dataset) to train the model. **Synthetic Minority Oversampling Technique (SMOTE)**
approach was used to address the imbalanced datasets. This model has been deployed to the production using [Streamlit](https://streamlit.io/) data app. The web app can be found [here](https://share.streamlit.io/masoud-ghasemian/heart-stroke-prediction/main/app.py)
