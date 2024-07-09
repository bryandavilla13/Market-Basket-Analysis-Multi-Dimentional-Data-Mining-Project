import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from mlxtend.frequent_patterns import apriori, association_rules
from mlxtend.preprocessing import TransactionEncoder
from sklearn.impute import SimpleImputer
from streamlit_carousel import carousel
import random
from streamlit_navigation_bar import st_navbar
from PIL import Image
import base64
import os

url = 'heart_disease_uci.csv'
df = pd.read_csv(url)

df.replace('Unknown', np.nan, inplace=True)
df_temp = df

print("Unique values in 'sex' column before mapping:", df['sex'].unique())
print(df['sex'].value_counts())

sek = df['sex']
numeric_cols = ['age', 'trestbps', 'chol', 'thalach', 'oldpeak']
for col in numeric_cols:
    df[col] = df[col].astype(float)

imputer = SimpleImputer(strategy='mean')
df[numeric_cols] = imputer.fit_transform(df[numeric_cols])

df['age_bins'] = pd.cut(df['age'], bins=5, labels=False) + 1
df['trestbps_bins'] = pd.cut(df['trestbps'], bins=5, labels=False) + 1
df['chol_bins'] = pd.cut(df['chol'], bins=5, labels=False) + 1
df['thalach_bins'] = pd.cut(df['thalach'], bins=5, labels=False) + 1
df['oldpeak_bins'] = pd.cut(df['oldpeak'], bins=5, labels=False) + 1

def get_data():
    return df

def get_data_cleveland():
    df_cleveland = df[df['dataset'] == 'Cleveland'] 
    return df_cleveland

def user_input_features():
    umur = st.slider('Umur', int(df['age'].min()), int(df['age'].max()), int(df['age'].mean()))
    jenis_kelamin = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
    cp = st.select_slider('Tipe Nyeri Dada (cp)', options=list(df['cp'].unique()))
    tekanan_darah_istirahat = st.slider('Tekanan Darah Istirahat (trestbps)', int(df['trestbps'].min()), int(df['trestbps'].max()), int(df['trestbps'].mean()))
    kolesterol = st.slider('Kolesterol (chol)', int(df['chol'].min()), int(df['chol'].max()), int(df['chol'].mean()))
    gula_darah_puasa = st.select_slider('Gula Darah Puasa (fbs)', options=list(df['fbs'].unique()))
    ecg_istirahat = st.select_slider('Hasil EKG Istirahat (restecg)', options=list(df['restecg'].unique()))
    detak_jantung_maksimum = st.slider('Detak Jantung Maksimum (thalach)', int(df['thalach'].min()), int(df['thalach'].max()), int(df['thalach'].mean()))
    angina_induksi_olahraga = st.select_slider('Angina Induksi Olahraga (exang)', options=list(df['exang'].unique()))
    st_depresi_olahraga = st.slider('Depresi ST Induksi Olahraga Relatif terhadap Istirahat (oldpeak)', float(df['oldpeak'].min()), float(df['oldpeak'].max()), float(df['oldpeak'].mean()))
    slope = st.select_slider('Kemiringan Segmen ST Olahraga Puncak (slope)', options=list(df['slope'].unique()))
    jumlah_pembuluh_utama = st.select_slider('Jumlah Pembuluh Utama (0-3) yang Berwarna oleh Flourosopi (ca)', options=list(df['ca'].unique()))
    thal = st.select_slider('Thalassemia (thal)', options=list(df['thal'].unique()))
    data = {'umur': umur, 'jenis_kelamin': jenis_kelamin, 'cp': cp, 'tekanan_darah_istirahat': tekanan_darah_istirahat, 'kolesterol': kolesterol, 'gula_darah_puasa': gula_darah_puasa,
            'ecg_istirahat': ecg_istirahat, 'detak_jantung_maksimum': detak_jantung_maksimum, 'angina_induksi_olahraga': angina_induksi_olahraga, 'st_depresi_olahraga': st_depresi_olahraga, 'slope': slope,
            'jumlah_pembuluh_utama': jumlah_pembuluh_utama, 'thal': thal}
    fitur = pd.DataFrame(data, index=[0])
    return fitur

def user_input_features_teknis():
    umur = st.slider('Umur', int(df['age'].min()), int(df['age'].max()), int(df['age'].mean()))
    jenis_kelamin = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
    cp = st.select_slider('Tipe Nyeri Dada (cp)', options=list(df['cp'].unique()))
    tekanan_darah_istirahat = st.slider('Tekanan Darah Istirahat (trestbps)', int(df['trestbps'].min()), int(df['trestbps'].max()), int(df['trestbps'].mean()))
    kolesterol = st.slider('Kolesterol (chol)', int(df['chol'].min()), int(df['chol'].max()), int(df['chol'].mean()))
    gula_darah_puasa = st.select_slider('Gula Darah Puasa (fbs)', options=list(df['fbs'].unique()))
    data = {'umur': umur, 'jenis_kelamin': jenis_kelamin, 'cp': cp, 'tekanan_darah_istirahat': tekanan_darah_istirahat, 'kolesterol': kolesterol, 'gula_darah_puasa': gula_darah_puasa,
            }
    fitur = pd.DataFrame(data, index=[0])
    return fitur

def user_input_features_kritis():
    umur = st.slider('Umur', int(df['age'].min()), int(df['age'].max()), int(df['age'].mean()))
    jenis_kelamin = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
    ecg_istirahat = st.select_slider('Hasil EKG Istirahat (restecg)', options=list(df['restecg'].unique()))
    detak_jantung_maksimum = st.slider('Detak Jantung Maksimum (thalach)', int(df['thalach'].min()), int(df['thalach'].max()), int(df['thalach'].mean()))
    angina_induksi_olahraga = st.select_slider('Angina Induksi Olahraga (exang)', options=list(df['exang'].unique()))
    st_depresi_olahraga = st.slider('Depresi ST Induksi Olahraga Relatif terhadap Istirahat (oldpeak)', float(df['oldpeak'].min()), float(df['oldpeak'].max()), float(df['oldpeak'].mean()))
    slope = st.select_slider('Kemiringan Segmen ST Olahraga Puncak (slope)', options=list(df['slope'].unique()))
    jumlah_pembuluh_utama = st.select_slider('Jumlah Pembuluh Utama (0-3) yang Berwarna oleh Flourosopi (ca)', options=list(df['ca'].unique()))
    thal = st.select_slider('Thalassemia (thal)', options=list(df['thal'].unique()))
    data = {'umur': umur, 'jenis_kelamin': jenis_kelamin,
            'ecg_istirahat': ecg_istirahat, 'detak_jantung_maksimum': detak_jantung_maksimum, 'angina_induksi_olahraga': angina_induksi_olahraga, 'st_depresi_olahraga': st_depresi_olahraga, 'slope': slope,
            'jumlah_pembuluh_utama': jumlah_pembuluh_utama, 'thal': thal}
    fitur = pd.DataFrame(data, index=[0])
    return fitur

def user_input_features_cleveland():
    df_cleveland = df[df['dataset'] == 'Cleveland']
    umur = st.slider('Umur', int(df_cleveland['age'].min()), int(df_cleveland['age'].max()), int(df_cleveland['age'].mean()))
    jenis_kelamin = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
    cp = st.select_slider('Tipe Nyeri Dada (cp)', options=list(df_cleveland['cp'].unique()))
    tekanan_darah_istirahat = st.slider('Tekanan Darah Istirahat (trestbps)', int(df_cleveland['trestbps'].min()), int(df_cleveland['trestbps'].max()), int(df_cleveland['trestbps'].mean()))
    kolesterol = st.slider('Kolesterol (chol)', int(df_cleveland['chol'].min()), int(df_cleveland['chol'].max()), int(df_cleveland['chol'].mean()))
    gula_darah_puasa = st.select_slider('Gula Darah Puasa (fbs)', options=list(df_cleveland['fbs'].unique()))
    ecg_istirahat = st.select_slider('Hasil EKG Istirahat (restecg)', options=list(df_cleveland['restecg'].unique()))
    detak_jantung_maksimum = st.slider('Detak Jantung Maksimum (thalach)', int(df_cleveland['thalach'].min()), int(df_cleveland['thalach'].max()), int(df_cleveland['thalach'].mean()))
    angina_induksi_olahraga = st.select_slider('Angina Induksi Olahraga (exang)', options=list(df_cleveland['exang'].unique()))
    st_depresi_olahraga = st.slider('Depresi ST Induksi Olahraga Relatif terhadap Istirahat (oldpeak)', float(df_cleveland['oldpeak'].min()), float(df_cleveland['oldpeak'].max()), float(df_cleveland['oldpeak'].mean()))
    slope = st.select_slider('Kemiringan Segmen ST Olahraga Puncak (slope)', options=list(df_cleveland['slope'].unique()))
    jumlah_pembuluh_utama = st.select_slider('Jumlah Pembuluh Utama (0-3) yang Berwarna oleh Flourosopi (ca)', options=list(df_cleveland['ca'].unique()))
    thal = st.select_slider('Thalassemia (thal)', options=list(df_cleveland['thal'].unique()))
    data = {'umur': umur, 'jenis_kelamin': jenis_kelamin, 'cp': cp, 'tekanan_darah_istirahat': tekanan_darah_istirahat, 'kolesterol': kolesterol, 'gula_darah_puasa': gula_darah_puasa,
            'ecg_istirahat': ecg_istirahat, 'detak_jantung_maksimum': detak_jantung_maksimum, 'angina_induksi_olahraga': angina_induksi_olahraga, 'st_depresi_olahraga': st_depresi_olahraga, 'slope': slope,
            'jumlah_pembuluh_utama': jumlah_pembuluh_utama, 'thal': thal}
    fitur = pd.DataFrame(data, index=[0])
    return fitur

def user_input_features_cleveland2():
    df_cleveland = df[df['dataset'] == 'Cleveland']
    umur = st.slider('Umur', int(df_cleveland['age'].min()), int(df_cleveland['age'].max()), int(df_cleveland['age'].mean()))
    jenis_kelamin = st.selectbox('Jenis Kelamin', ['Male', 'Female'])
    cp = st.select_slider('Tipe Nyeri Dada (cp)', options=list(df_cleveland['cp'].unique()))
    thal = st.select_slider('Thalassemia (thal)', options=list(df_cleveland['thal'].unique()))
    data = {'umur': umur, 'jenis_kelamin': jenis_kelamin, 'cp': cp, 'thal': thal}
    fitur = pd.DataFrame(data, index=[0])
    return fitur

def encode(df):
    df_trans = df[['age_bins', 'sex', 'trestbps_bins', 'chol_bins', 'thalach_bins', 'oldpeak_bins', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'num']]
    df_trans = df_trans.astype(str)
    te = TransactionEncoder()
    te_ary = te.fit_transform(df_trans.values)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    
    recommendations = []
    for idx, row in rules.iterrows():
        antecedent = ', '.join(list(row['antecedents']))
        consequent = ', '.join(list(row['consequents']))
        recommendations.append((antecedent, consequent))
    return rules, recommendations

def encode_cleveland2(df):
    df_cleveland = df[df['dataset'] == 'Cleveland']
    df_trans = df_cleveland[['age_bins', 'sex', 'cp', 'thal', 'restecg', 'num']]
    df_trans = df_trans.astype(str)
    te = TransactionEncoder()
    te_ary = te.fit_transform(df_trans.values)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df_encoded, min_support=0.001, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.7)
    
    recommendations = []
    for idx, row in rules.iterrows():
        antecedent = ', '.join(list(row['antecedents']))
        consequent = ', '.join(list(row['consequents']))
        recommendations.append((antecedent, consequent))
    return rules, recommendations

def encode_cleveland(df):
    df_cleveland = df[df['dataset'] == 'Cleveland']
    df_trans = df_cleveland[['age_bins', 'sex', 'trestbps_bins', 'chol_bins', 'thalach_bins', 'oldpeak_bins', 'cp', 'fbs', 'restecg', 'exang', 'slope', 'ca', 'thal', 'num']]
    df_trans = df_trans.astype(str)
    te = TransactionEncoder()
    te_ary = te.fit_transform(df_trans.values)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df_encoded, min_support=0.01, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    
    recommendations = []
    for idx, row in rules.iterrows():
        antecedent = ', '.join(list(row['antecedents']))
        consequent = ', '.join(list(row['consequents']))
        recommendations.append((antecedent, consequent))
    return rules, recommendations

def encode_teknis(df):
    df_trans = df[['age_bins', 'sex', 'trestbps_bins', 'chol_bins', 'cp', 'fbs', 'ca', 'thal', 'num']]
    df_trans = df_trans.astype(str)
    te = TransactionEncoder()
    te_ary = te.fit_transform(df_trans.values)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df_encoded, min_support=0.001, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    #st.markdown(rules)
    recommendations = []
    for idx, row in rules.iterrows():
        antecedent = ', '.join(list(row['antecedents']))
        consequent = ', '.join(list(row['consequents']))
        recommendations.append((antecedent, consequent))
        #print("Consequents:", list(row['consequents']))
        #if all(item.startswith('ca') or item.startswith('thal') or item.startswith('num') or item.startswith('restecg') for item in row['consequents']):
            #recommendations.append((antecedent, consequent))
            #st.markdown(consequent)

    return rules, recommendations
        
    return rules, recommendations

def encode_kritis(df):
    df_trans = df[['age_bins', 'sex', 'chol_bins', 'fbs', 'oldpeak_bins', 'exang', 'cp', 'slope', 'ca', 'thal', 'num']]
    df_trans = df_trans.astype(str)
    te = TransactionEncoder()
    te_ary = te.fit_transform(df_trans.values)
    df_encoded = pd.DataFrame(te_ary, columns=te.columns_)
    frequent_itemsets = apriori(df_encoded, min_support=0.001, use_colnames=True)
    rules = association_rules(frequent_itemsets, metric="confidence", min_threshold=0.7)
    
    recommendations = []
    for idx, row in rules.iterrows():
        antecedent = ', '.join(list(row['antecedents']))
        consequent = ', '.join(list(row['consequents']))
        recommendations.append((antecedent, consequent))
        
    return rules, recommendations

def is_real_string(a):
  if a.isdigit():
    return False
  try:
    float(a)
    return False
  except :
    if a == "True" or a =="False":
      return False
    return True

def get_image_base64(image_path):
    with open(image_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()

def main():
    st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;700&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Lobster&display=swap');
    
    body {
        background-color: #FFFFFF; /* white */
    }
    .stButton>button {
        color: #000000; /* button text color */
        background-color: #A9A9A9; /* button background color */
    }
    .stHeader, .stSubheader, .stMarkdown, .stCaption {
        color: #000080; /* text color */
    }
    .custom-text {
        color: #FFFFFF; /* white text */
        font-family: 'Roboto', sans-serif; /* Roboto font */
        font-size: 18px; /* Adjust font size as needed */
        text-align: justify; /* Justified text alignment */
    }
    .custom-subheader {
        color:#FFFFE0; /* yellow text */
        font-family: 'Lobster', sans-serif; /* Lobster font */
        font-size: 60px; /* Increase font size */
    }
    .custom-subheader2 {
        color:#FFFFFF; /* yellow text */
        font-family: 'Roboto', sans-serif; /* Lobster font */
        font-size: 40px; /* Increase font size */
    }
    </style>
    
    """, unsafe_allow_html=True)
    menu = ['Homepage', 'Analisis Keseluruhan', 'Analisis Gejala Teknis', 'Analisis Gejala Kritis', 'Analisis Keseluruhan Cleveland', 'Analisis Cleveland2', 'Team Pembuat']
    choice = st_navbar(menu)
    st.title('Analisis Penyakit Jantung Sesuai dengan Tingkatan Penyakit yang berbeda-beda')

    if choice == 'Homepage':
        st.markdown('<h2 class="custom-subheader2">Pengertian Penyakit Jantung</h2>', unsafe_allow_html=True)
        test_items = [
            dict(
                title="",
                text="",
                img="https://tse3.mm.bing.net/th?id=OIP.8inb5bhKMGBYhTO7p3cyCgHaEs&pid=Api&P=0&h=180",
                link=""
            ),
            dict(
                title="",
                text="",
                img="https://tse2.mm.bing.net/th?id=OIP.Z4pFW_X79Q3ViBeJurXo8QHaE8&pid=Api&P=0&h=180",
                link=""
            ),
            dict(
                title="",
                text="",
                img="https://tse1.mm.bing.net/th?id=OIP.rWGbG-W2cqWPU4KkGOGuFwHaE8&pid=Api&P=0&h=180",
                link=""
            ),
        ]

        carousel(items=test_items)
        st.markdown("""
        <p class="custom-text">
        Penyakit jantung adalah kondisi ketika bagian jantung yang meliputi pembuluh darah jantung, 
        selaput jantung, katup jantung, dan otot jantung mengalami gangguan. Penyakit jantung bisa disebabkan oleh berbagai hal, seperti sumbatan pada pembuluh darah jantung, 
        peradangan, infeksi, atau kelainan bawaan.
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown('<h2 class="custom-subheader">Tujuan Pembuatan Website</h2>', unsafe_allow_html=True)
        st.write("""
        <p class="custom-text">
        Website ini dibuat dengan tujuan dapat membantu user untuk dapat melakukan pengecekan apakah mereka terkena penyakit jantung sesuai
        dengan tingkatan tertentu yang sudah ditetapkan. Namun, yang perlu ditekankan bahwa rekomendasi penyakit tersebut hanya berdasarkan 
        kumpulan dataset yang sudah disediakan. Website ini juga dibuat untuk memenuhi tugas Project Mata Kuliah Data Mining di Universitas 
        Kristen Petra Tahun Ajaran 2023/2024
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown('<h2 class="custom-subheader">Penjelasan Dataset</h2>', unsafe_allow_html=True)
        st.write("""
        <p class="custom-text">
        Ini adalah jenis kumpulan data multivariat yang berarti menyediakan atau melibatkan berbagai variabel matematika atau statistik terpisah, 
        analisis data numerik multivariat. Ini terdiri dari 14 atribut yaitu usia, jenis kelamin, jenis nyeri dada, tekanan darah istirahat, 
        kolesterol serum, gula darah puasa, hasil elektrokardiografi istirahat, detak jantung maksimum yang dicapai, angina akibat olahraga, 
        oldpeak — depresi ST yang disebabkan oleh olahraga relatif terhadap istirahat, kemiringan puncak latihan segmen ST, jumlah pembuluh darah 
        besar dan Thalassemia. Basis data ini mencakup 76 atribut, namun semua penelitian yang dipublikasikan berhubungan dengan penggunaan 14 
        atribut di antaranya. Basis data Cleveland adalah satu-satunya yang digunakan oleh peneliti ML hingga saat ini. Salah satu tugas utama 
        pada kumpulan data ini adalah untuk memprediksi berdasarkan atribut yang diberikan pada pasien apakah orang tersebut menderita penyakit 
        jantung atau tidak dan yang lainnya adalah tugas eksperimental untuk mendiagnosis dan menemukan berbagai wawasan dari kumpulan data ini 
        yang dapat membantu dalam pemahaman. masalahnya lebih lanjut.
        </p>
        <p class="custom-text">
        Berikut adalah tingkatan penyakit jantung:
        <ul class ="custom-text">
            <li>Tingkatan 0: Tidak ada penyakit jantung.</li>
            <li>Tingkatan 1: Penyakit jantung ringan.</li>
            <li>Tingkatan 2: Penyakit jantung sedang.</li>
            <li>Tingkatan 3: Penyakit jantung berat.</li>
            <li>Tingkatan 4: Penyakit jantung sangat berat atau kronis.</li>
        </ul>
        </p>
        <p class="custom-text">
        Berikut adalah Range Umur:
        <ul class ="custom-text">
            <li>Umur termudah berada di umur 28 tahun</li>
            <li>Umur tertua berada di umur 77 tahun</li>
        </ul>
        </p>
        <p class="custom-text">
        Berikut adalah Tipe Jenis Kelamin:
        <ul class ="custom-text">
            <li>Tipe 1 : Jenis Kelamin Pria</li>
            <li>Tipe 2 : Jenis Kelamin Wanita</li>
        </ul>
        </p>
        <p class="custom-text">
        Berikut adalah Lokasi Dataset:
        <ul class ="custom-text">
            <li>Dataset 1 : Cleveland</li>
            <li>Dataset 2 : Hungary</li>
            <li>Dataset 3 : VA Long Beach</li>
            <li>Dataset 4 : Switzerland</li>
        </ul>
        </p>
        <p class="custom-text">
        Adapun juga beberapa komponen lainnya yang bisa dimasukkan untuk menemukan rekomendasinya, seperti jenis nyeri dada, tekanan darah, kolestrol, puasa gula darah, elektrokardiografi saat istirahat, detak jantung maksimum, angina, depresi, slope saat latian ST, jumlah pembuluh darah besar, dan thalasemia
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown('<h2 class="custom-subheader">Penjelasan Pengelolaan Data</h2>', unsafe_allow_html=True)
        st.markdown("""
        <p class="custom-text">
        Dataset tersebut akan diolah menggunakan algoritma Market Basket Analysis Multi Dimentional menggunakan Apriori. Dengan demikian, data tersebut 
        di analisis sesuai dengan keranjang belanja (market basket analysis) untuk menemukan pola asosiasi 
        antar item dalam kumpulan data transaksi. Algoritma ini bekerja dengan mencari itemset yang sering muncul bersama dalam transaksi, yang 
        disebut dengan "itemset sering". Dengan menggunakan itemset sering, kita dapat membuat aturan asosiasi yang dapat digunakan untuk 
        mengidentifikasi hubungan antara item dalam keranjang belanja.
        </p>
        """, unsafe_allow_html=True)
        
        st.markdown('<h2 class="custom-subheader">Tabel Dataset</h2>', unsafe_allow_html=True)
        st.markdown("""
        <p class="custom-text">
        Dataset tersebut berisi 14 kategori sesuai dengan penjelasan di atas
        </p>
        """, unsafe_allow_html=True)
        st.write(get_data())

        st.markdown('<h2 class="custom-subheader">Visualisasi Dataset</h2>', unsafe_allow_html=True)
        if st.button('Tampilkan Grafik'):
            df['sex'] = sek        
            st.markdown("""
            <p class="custom-text">
            Grafik Distribusi Umur
            </p>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.histplot(df['age'], kde=True, ax=ax)
            st.pyplot(fig)

            st.markdown("""
            <p class="custom-text">
            Grafik Distribusi Tekanan Darah
            </p>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.histplot(df['trestbps'], kde=True, ax=ax)
            st.pyplot(fig)
            
            st.markdown("""
            <p class="custom-text">
            Grafik Distribusi Usia Berdasarkan Jenis Kelamin
            </p>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.boxplot(x='sex', y='age', data=df_temp, ax=ax)
            ax.set_title('Age ~ Sex')
            st.pyplot(fig)
            
            st.markdown("""
            <p class="custom-text">
            Grafik Distribusi Usia Berdasarkan Dataset
            </p>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.boxplot(x='dataset', y='age', data=df_temp, ax=ax)
            ax.set_title('Age ~ Dataset')
            st.pyplot(fig)
            
            st.markdown("""
            <p class="custom-text">
            Grafik Distribusi Jenis Kelamin Berdasarkan Dataset
            </p>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.countplot(x='dataset', hue='sex', data=df_temp, ax=ax)
            ax.set_title('Sex ~ Dataset')
            st.pyplot(fig)

            st.markdown("""
            <p class="custom-text">
            Grafik Distribusi Usia Berdasarkan Rest ECG
            </p>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.boxplot(x='restecg', y='age', data=df_temp, ax=ax)
            ax.set_title('Age ~ Rest ECG')
            st.pyplot(fig)

            st.markdown("""
            <p class="custom-text">
            Grafik Distribusi Usia Berdasarkan Tipe Nyeri Data
            </p>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.boxplot(x='cp', y='age', data=df_temp, ax=ax)
            ax.set_title('Age ~ Chest Pain Type')
            st.pyplot(fig)

            st.markdown("""
            <p class="custom-text">
            Grafik Distribusi Usia Berdasarkan Kemiringan Segmen ST
            </p>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.boxplot(x='slope', y='age', data=df_temp, ax=ax)
            ax.set_title('Age ~ Slope')
            st.pyplot(fig)

            st.markdown("""
            <p class="custom-text">
            Grafik Distribusi Usia Berdasarkan Diagnosis Penyakit Jantung
            </p>
            """, unsafe_allow_html=True)
            fig, ax = plt.subplots()
            sns.boxplot(x='num', y='age', data=df_temp, ax=ax)
            ax.set_title('Age ~ Diagnosis of Heart Disease')
            st.pyplot(fig)
        else:
            st.markdown("""
        <p class="custom-text">
        Klik tombol untuk menampilkan grafik distribusi umur dan tekanan darah.
        </p>
        """, unsafe_allow_html=True)

    elif choice == 'Analisis Keseluruhan':
        st.markdown("""
        <p class="custom-text">
        Tampilan Dataset :
        </p>
        """, unsafe_allow_html=True)
        st.write(get_data())

        fitur_input = user_input_features()
        st.markdown("""
        <p class="custom-text">
        Data yang telah di input oleh User : 
        </p>
        """, unsafe_allow_html=True)

        st.write(fitur_input)
        
        st.markdown("""
        <p class="custom-text">
        Hasil Rekomendasi... (proses terkadang memerlukan waktu yang sedikit lama)
        </p>
        """, unsafe_allow_html=True)

        rules, recommendations = encode(df)
        
        if recommendations:   
                for antecedent, consequent in recommendations:
                    if is_real_string(antecedent) and is_real_string(consequent):
                        antecedent_set = set(antecedent.split(', '))
                        match_found = all(any(fitur_input[column][0] == value for column in fitur_input.columns) for value in antecedent_set)
                        if match_found:
                            st.success(f"Rekomendasi: Jika pasien memiliki '{antecedent}', maka kemungkinan memiliki '{consequent}'")
                            break 
        else:
            st.warning('Tidak ada rekomendasi yang ditemukan.')

    elif choice == 'Analisis Gejala Teknis':
        st.markdown("""
        <p class="custom-text">
        Tampilan Dataset :
        </p>
        """, unsafe_allow_html=True)
        st.write(get_data())

        fitur_input = user_input_features_teknis()
        st.markdown("""
        <p class="custom-text">
        Data yang telah di input oleh User : 
        </p>
        """, unsafe_allow_html=True)
        st.write(fitur_input)
        rules, recommendations = encode_teknis(df)
        
        st.markdown("""
        <p class="custom-text">
        Hasil Rekomendasi... (proses terkadang memerlukan waktu yang sedikit lama)
        </p>
        """, unsafe_allow_html=True)
        
        if recommendations:   
                for antecedent, consequent in recommendations:
                    if is_real_string(antecedent) and is_real_string(consequent):
                        antecedent_set = set(antecedent.split(', '))
                        match_found = all(any(fitur_input[column][0] == value for column in fitur_input.columns) for value in antecedent_set)
                        if match_found:
                            st.success(f"Rekomendasi: Jika pasien memiliki '{antecedent}', maka kemungkinan memiliki '{consequent}'")
                            break 
        else:
            st.warning('Tidak ada rekomendasi yang ditemukan.')
    
    elif choice == 'Analisis Gejala Kritis':
        st.markdown("""
        <p class="custom-text">
        Tampilan Dataset :
        </p>
        """, unsafe_allow_html=True)
        st.write(get_data())

        fitur_input = user_input_features_kritis()
        st.markdown("""
        <p class="custom-text">
        Data yang telah di input oleh User : 
        </p>
        """, unsafe_allow_html=True)
        st.write(fitur_input)

        rules, recommendations = encode_kritis(df)
        
        st.markdown("""
        <p class="custom-text">
        Hasil Rekomendasi... (proses terkadang memerlukan waktu yang sedikit lama)
        </p>
        """, unsafe_allow_html=True)
        
        if recommendations:   
                for antecedent, consequent in recommendations:
                    if is_real_string(antecedent) and is_real_string(consequent):
                        antecedent_set = set(antecedent.split(', '))
                        match_found = all(any(fitur_input[column][0] == value for column in fitur_input.columns) for value in antecedent_set)
                        if match_found:
                            st.success(f"Rekomendasi: Jika pasien memiliki '{antecedent}', maka kemungkinan memiliki '{consequent}'")
                            break 
        else:
            st.warning('Tidak ada rekomendasi yang ditemukan.')
    
    elif choice == 'Analisis Keseluruhan Cleveland':
        st.markdown("""
        <p class="custom-text">
        Tampilan Dataset Cleveland:
        </p>
        """, unsafe_allow_html=True)
        st.write(get_data_cleveland())

        fitur_input = user_input_features_cleveland()
        st.markdown("""
        <p class="custom-text">
        Data yang telah diinput oleh User:
        </p>
        """, unsafe_allow_html=True)
        st.write(fitur_input)
        
        st.markdown("""
        <p class="custom-text">
        Hasil Rekomendasi... (proses terkadang memerlukan waktu yang sedikit lama)
        </p>
        """, unsafe_allow_html=True)

        rules_cleveland, recommendations_cleveland = encode_cleveland(df)
        
        if recommendations_cleveland:
            for antecedent, consequent in recommendations_cleveland:
                if is_real_string(antecedent) and is_real_string(consequent):
                    ada = 0
                    for x in fitur_input.columns:
                        if fitur_input[x][0] == antecedent:
                            ada = 1
                            ran =random.randrange(1, 10) % 2
                            print(ran)
                            if ran  == 1:
                                 ada=0
                                 continue
                    if ada == 0:
                        continue
                    st.success(f"Rekomendasi: Jika pasien memiliki '{antecedent}' , maka kemungkinan memiliki '{consequent}'")
                    break
        else:
            st.warning('Tidak ada rekomendasi yang ditemukan.')
            
    elif choice == 'Analisis Cleveland2':
        st.markdown("""
        <p class="custom-text">
        Tampilan Dataset Cleveland:
        </p>
        """, unsafe_allow_html=True)
        st.write(get_data_cleveland())

        st.markdown(""" <p class = "custom-text">
                    Pada page ini kita ingin memberikan rekomendasi kepada user perihal hasil elektrokardiografi istirahat (restcg) dan tingkat penyakit jantung yang diderita oleh user
                    </p>
                    """, unsafe_allow_html = True)
        
        fitur_input = user_input_features_cleveland2()
        st.markdown("""
        <p class="custom-text">
        Data yang telah diinput oleh User:
        </p>
        """, unsafe_allow_html=True)
        st.write(fitur_input)
        
        st.markdown("""
        <p class="custom-text">
        Hasil Rekomendasi... (proses terkadang memerlukan waktu yang sedikit lama)
        </p>
        """, unsafe_allow_html=True)

        rules_cleveland, recommendations_cleveland = encode_cleveland2(df)
        
        if recommendations_cleveland:   
                for antecedent, consequent in recommendations_cleveland:
                    if is_real_string(antecedent) and is_real_string(consequent):
                        antecedent_set = set(antecedent.split(', '))
                        match_found = all(any(fitur_input[column][0] == value for column in fitur_input.columns) for value in antecedent_set)
                        if match_found:
                            st.success(f"Rekomendasi: Jika pasien dari Cleveland memiliki '{antecedent}', maka kemungkinan memiliki '{consequent}'")
                            break 
        else:
            st.warning('Tidak ada rekomendasi yang ditemukan untuk pasien dari Cleveland.')

    elif choice == 'Team Pembuat':
        st.subheader('Kelompok 2 - Data Mining 23/24')
        image_dir = "C://Users//Bryan//KB//basket_acc//basket_acc//images//"
        
        team_members = [
            {"name": "Bryan Davila Effendi", "role": "19 tahun - Data Science 22", "img": os.path.join(image_dir, "Bryan.jpg")},
            {"name": "Edwin Christopher Henry", "role": "20 tahun - Data Science 22", "img": os.path.join(image_dir, "Edwin.jpg")},
            {"name": "Berlynn Callista", "role": "21 tahun - Data Science 22", "img": os.path.join(image_dir, "Berlin.jpg")},
            {"name": "Welsey Vijaya Pranata", "role": "21 tahun - Data Science 22", "img": os.path.join(image_dir, "Wesley.jpg")}
        ]
        
        for member in team_members:
            img_base64 = get_image_base64(member["img"])
            st.markdown(f"""
                <div style="display: flex; align-items: center; margin-bottom: 20px;">
                    <img src="data:image/jpg;base64,{img_base64}" width="100" height="100" style="border-radius: 50%; margin-right: 20px;">
                    <div>
                        <h3 style="margin: 0; color: white;">{member['name']}</h3>
                        <p style="margin: 0; color: white;">{member['role']}</p>
                    </div>
                </div>
                """, unsafe_allow_html=True)

if __name__ == '__main__':
    main()