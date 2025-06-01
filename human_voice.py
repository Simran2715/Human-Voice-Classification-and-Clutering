import streamlit as st
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import MinMaxScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import seaborn as sns

choice= st.sidebar.selectbox("Navigator :",["Introduction","EDA","Human Voice Prediction","About me"])

# -----------------------------------------------------------------------INTRODUCTION---------------------------------------------------------------------------------------
if choice=="Introduction":
    st.title('🗣️Human Voice Classification')
    st.image('Human_voice.png')
    st.subheader('This project is related to machine learning-based model to classify and cluster human voice samples based on extracted audio features. The system will preprocess the dataset, apply clustering and classification models, and evaluate their performance. The final application will provide an interface for uploading audio samples and receiving predictions.')

# -----------------------------------------------------------------------EDA ANALYSIS--------------------------------------------------------------------------------------------
elif choice=="EDA":
    st.title('📌EDA Analysis')
    human_voice=pd.read_csv('vocal_gender_features_new.csv')
    X= human_voice.drop(columns='label')
    y= human_voice['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    eda=st.selectbox('EDA Analysis :',['Heatmap','Pairplot','Pie Chart','Confusion Matrix of Random Forest','Confusion Matrix for SVC','Confusion Matrix for KNneighbors','Confusion Matrix for MLP Classifier','KMEANS Clustering','DBSCAN Clustering'])
    if eda== 'Heatmap':
        plt.figure(figsize=(20,15))
        sns.heatmap(human_voice.corr(), annot=True, cmap='coolwarm',fmt='.2f')
        plt.title('Correlation Heatmap')
        st.pyplot(plt)

    elif eda== 'Pairplot':
        feature_columns = human_voice.columns[:-1]
        human_voice[feature_columns].hist(bins=30, figsize=(20, 15))
        plt.suptitle('Feature Distributions')
        st.pyplot(plt)

    elif eda=='Pie Chart':
        gender_counts = human_voice['label'].value_counts()
        plt.figure(figsize=(5,5))
        plt.pie(gender_counts, labels= gender_counts.index,autopct='%1.1f%%', startangle=90)
        plt.title('Gender Distribution')
        plt.axis('equal')
        st.pyplot(plt)

    elif eda=='Confusion Matrix of Random Forest':
        rf=RandomForestClassifier(random_state=42)
        rf.fit(X_train, y_train)
        pred_y_rf= rf.predict(X_test)
        cm=confusion_matrix(y_test, pred_y_rf)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Accent', xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.title('Confusion Matrix for RandomForest Classifier')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        st.pyplot(plt)

    elif eda=='Confusion Matrix for SVC':
        svm= SVC(random_state=42)
        svm.fit(X_train, y_train)
        pred_y_svm= svm.predict(X_test)
        cm=confusion_matrix(y_test, pred_y_svm)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Accent', xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.title('Confusion Matrix for SVC')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        st.pyplot(plt)

    elif eda=='Confusion Matrix for KNneighbors':
        knn = KNeighborsClassifier(n_neighbors=4)
        knn.fit(X_train, y_train)
        pred_y_knn= knn.predict(X_test)
        cm=confusion_matrix(y_test, pred_y_knn)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.title('Confusion Matrix for KNeighbors Classifier')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        st.pyplot(plt)

    elif eda=='Confusion Matrix for MLP Classifier':
        nnc= MLPClassifier(random_state=42, max_iter= 300)
        nnc.fit(X_train, y_train)
        pred_y_nnc= nnc.predict(X_test)
        cm=confusion_matrix(y_test, pred_y_nnc)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=np.unique(y), yticklabels=np.unique(y))
        plt.title('Confusion Matrix for MLPClassifier')
        plt.xlabel('Predicted Label')
        plt.ylabel('True Label')
        st.pyplot(plt)

    elif eda=='KMEANS Clustering':
        kmeans = KMeans(n_clusters=2, random_state=42, n_init=10)
        kmeans_train= kmeans.fit_predict(X_train)
        plt.scatter(X_train.iloc[:,0],X_train.iloc[:,1],c=kmeans_train, cmap='viridis')
        plt.title('K-means Clustering')
        st.pyplot(plt)

    elif eda=='DBSCAN Clustering':
        dbscan= DBSCAN(eps=0.5, min_samples=5)
        dbscan_train= dbscan.fit_predict(X_train)
        plt.figure(figsize=(10, 6))
        scatter=plt.scatter(X_train.iloc[:,0],X_train.iloc[:,1],c=dbscan_train, cmap='viridis')
        plt.colorbar(scatter)
        plt.title('DBSCAN Clustering')
        st.pyplot(plt)

# ---------------------------------------------------------------------------------HUMAN VOICE PREDICTION---------------------------------------------------------------------------

elif choice=="Human Voice Prediction":
    st.title('🔊Human Voice Prediction')

    model = pickle.load(open('model_selected.pkl', 'rb'))
    scaler = pickle.load(open('scaler.pkl', 'rb'))

    mfcc_5_mean= st.number_input('mcff_5_mean')
    mean_spectral_contrast= st.number_input('mean_spectral_contrast')
    mfcc_3_std= st.number_input('mfcc_3_std')
    mfcc_2_mean= st.number_input('mfcc_2_mean')
    mfcc_1_mean= st.number_input('mfcc_1_mean')
    std_spectral_bandwidth= st.number_input('std_spectral_bandwidt')
    mfcc_12_mean= st.number_input('mfcc_12_mean')
    mfcc_10_mean= st.number_input('mfcc_10_mean')
    rms_energy= st.number_input('rms_energy')
    mfcc_10_std= st.number_input('mfcc_10_std')



    human_voice=pd.read_csv('vocal_gender_features_new.csv')

    if st.button('Predict Gender Voice'):
        input_data= np.array([[mfcc_5_mean,mean_spectral_contrast,mfcc_3_std,mfcc_2_mean,mfcc_1_mean,std_spectral_bandwidth,mfcc_12_mean,mfcc_10_mean,rms_energy,mfcc_10_std]])

        prediction = model.predict(input_data)
        st.success(f"Predicted Gender: {prediction}")

elif choice=="About me":
    st.title('👩‍💻About Me')
    st.image("AboutMe.webp")
    st.write("""
    Developed by: Simran Paul

    email: simranpaul1010@gmail.com

    *Skills : Machine Learning , Data preparation, Feature Engeneering 

    I am very passionate to grasp new skills and very quick adaptive to learning environment!!
    """)

