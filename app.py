import streamlit as st
import pandas as pd
from pyngrok import ngrok
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
from streamlit_option_menu import option_menu
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


# Menu Data
def menu_data():
    st.subheader('Data Emotional Speech')
    st.write('Berikut ini data hasil ekstraksi dari data audio emotional speech')
    df = pd.read_csv('data.csv')
    df
# Menu Preprocessing Data
def menu_preprocessing_data():
    st.subheader('Preprocessing Data')
    preprocessing_option = st.radio('Pilihan Preprocessing', ['Min-Max Scaler'])
    if preprocessing_option == 'Min-Max Scaler':
        st.write('Min-Max Scaler')

        # Memanggil data hepatitis C
        data = pd.read_csv('data.csv')
        X = data[["Mean", "Std", "Median", "Max", "Min", "Modus", "Skew", "Kurtosis", "Q1", "Q3", "IQR", "ZCRMean", "ZCRMedian", "ZCRStd", "ZCRSkew", "ZCRKurt", "EneMean", "EneMedian", "ENeStd", "EneSkew", "EneKurt"]].values
        y = data["Label"].values
        scaler = MinMaxScaler()
        data=[]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        st.subheader('Data TRAIN Setelah Scaling')
        st.dataframe(X_train_scaled)
        st.subheader('Data TEST Setelah Scaling')
        st.dataframe(X_test_scaled)
# Menu Modelling
def menu_modelling():
    st.subheader('Modelling KNN & PCA')
    st.subheader('Akurasi')
    data = pd.read_csv('data.csv')
    X = data[["Mean", "Std", "Median", "Max", "Min", "Modus", "Skew", "Kurtosis", "Q1", "Q3", "IQR", "ZCRMean", "ZCRMedian", "ZCRStd", "ZCRSkew", "ZCRKurt", "EneMean", "EneMedian", "ENeStd", "EneSkew", "EneKurt"]].values
    y = data["Label"].values
    scaler = MinMaxScaler()
    data=[]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pca=PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    knn = KNeighborsClassifier(n_neighbors=12)
    knn.fit(X_train_pca, y_train)
    y_pred = knn.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    st.write("Kombinasi terbaik: PCA = 10, K = 12")
    accuracy
    
# Menu Implementasi
def menu_implementasi():
    st.subheader('Implementasi Modelling KNN & PCA')
    st.subheader('Akurasi')
    data = pd.read_csv('data.csv')
    X = data[["Mean", "Std", "Median", "Max", "Min", "Modus", "Skew", "Kurtosis", "Q1", "Q3", "IQR", "ZCRMean", "ZCRMedian", "ZCRStd", "ZCRSkew", "ZCRKurt", "EneMean", "EneMedian", "ENeStd", "EneSkew", "EneKurt"]].values
    y = data["Label"].values
    scaler = MinMaxScaler()
    data=[]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=100)
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    pca=PCA(n_components=10)
    X_train_pca = pca.fit_transform(X_train_scaled)
    X_test_pca = pca.transform(X_test_scaled)
    knn = KNeighborsClassifier(n_neighbors=12)
    knn.fit(X_train_pca, y_train)
    y_pred = knn.predict(X_test_pca)
    accuracy = accuracy_score(y_test, y_pred)
    accuracy
    label=['YAF_angry',
    'YAF_pleasant_surprised',
    'OAF_neutral',
    'YAF_sad',
    'YAF_neutral',
    'YAF_fear',
    'YAF_disgust',
    'OAF_Sad',
    'YAF_happy',
    'OAF_Pleasant_surprise',
    'OAF_happy',
    'OAF_angry',
    'OAF_disgust',
    'OAF_Fear']
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm.T, square=True, annot=True, fmt='d', cbar=False, cmap='viridis',
                xticklabels=label, yticklabels=label)

    plt.title('Confusion matrix')
    plt.xlabel('true label')
    plt.ylabel('predicted label');
    plt.show()
    df = pd.DataFrame({'Real Values':y_test, 'Predicted Values':y_pred})
    df
# Judul halaman
st.title('Aplikasi Data Science "Ekstraksi Data Audio"')

with st.sidebar:
    selected_menu = option_menu("Main Menu", ['Data','Preprocessing Data','Modelling','Implementasi'], 
         default_index=0)


# Tampilkan konten sesuai menu yang dipilih
if selected_menu == 'Data':
    menu_data()
elif selected_menu == 'Preprocessing Data':
    menu_preprocessing_data()
elif selected_menu == 'Modelling':
    menu_modelling()
elif selected_menu == 'Implementasi':
    menu_implementasi()
