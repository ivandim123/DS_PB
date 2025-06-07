import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Dashboard Analytics",
    layout="wide"
)

# Set style
sns.set(style="whitegrid")

# Dashboard title
st.title("Dashboard Analytics")
st.markdown("Dashboard ini menampilkan visualisasi data berdasarkan berbagai fitur.")

@st.cache_data
def load_csv_from_path():
    """Try to load CSV from various paths"""
    possible_paths = [
        "data.csv",
        "./data.csv", 
        "dataset/data.csv",
        "data/data.csv"
    ]
    
    for path in possible_paths:
        try:
            df = pd.read_csv(path)
            return df, path
        except FileNotFoundError:
            continue
    
    return None, None

@st.cache_data
def create_sample_data():
    """Create sample data for demonstration"""
    np.random.seed(42)
    n = 500
    
    data = {
        'Student_ID': range(1, n+1),
        'Age': np.random.randint(17, 25, n),
        'Gender': np.random.choice(['Male', 'Female'], n),
        'Application_mode': np.random.choice([1, 2, 17, 18, 39, 42, 43], n),
        'Fathers_qualification': np.random.choice([1, 2, 3, 4, 5, 19, 34, 35], n),
        'Mothers_qualification': np.random.choice([1, 2, 3, 4, 5, 19, 34, 35], n),
        'Tuition_fees_up_to_date': np.random.choice([0, 1], n, p=[0.15, 0.85]),
        'Scholarship_holder': np.random.choice([0, 1], n, p=[0.7, 0.3]),
        'Grade_1st_semester': np.random.normal(13, 3, n).clip(0, 20),
        'Grade_2nd_semester': np.random.normal(13, 3, n).clip(0, 20),
        'Curricular_units_enrolled': np.random.randint(4, 8, n),
        'Curricular_units_approved': np.random.randint(2, 8, n),
        'Target': np.random.choice(['Dropout', 'Enrolled', 'Graduate'], n, p=[0.3, 0.4, 0.3])
    }
    
    return pd.DataFrame(data)

# Try to load data from file first
df, file_path = load_csv_from_path()

if df is not None:
    st.success(f"Data berhasil dimuat dari {file_path}")
else:
    # Show file uploader if no file found
    st.warning("File 'data.csv' tidak ditemukan di direktori. Silakan upload file CSV Anda.")
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("Data berhasil dimuat dari file yang diupload")
        except Exception as e:
            st.error(f"Error saat membaca file yang diupload: {e}")
            df = None
    else:
        # Use sample data if no file uploaded
        st.info("Menggunakan data sampel untuk demonstrasi. Upload file CSV Anda untuk analisis data sebenarnya.")
        df = create_sample_data()

if df is not None:
    # Display basic info about the dataset
    st.sidebar.header("Informasi Dataset")
    st.sidebar.write(f"Jumlah baris: {df.shape[0]}")
    st.sidebar.write(f"Jumlah kolom: {df.shape[1]}")
    
    # Show column names and types
    st.sidebar.subheader("Kolom dalam Dataset")
    for col in df.columns:
        col_type = "Numerik" if pd.api.types.is_numeric_dtype(df[col]) else "Kategorikal"
        st.sidebar.write(f"• {col} ({col_type})")
    
    # Feature selection
    st.sidebar.header("Pilih Fitur untuk Divisualisasikan")
    
    # Get all columns for visualization
    all_columns = list(df.columns)
    selected_feature = st.sidebar.selectbox("Pilih Fitur Utama:", all_columns, index=0)
    
    # Option to select a grouping variable (for comparison)
    grouping_options = ["Tidak ada"] + [col for col in df.columns if not pd.api.types.is_numeric_dtype(df[col]) and df[col].nunique() <= 10]
    selected_grouping = st.sidebar.selectbox("Pilih Variabel Pembanding (opsional):", grouping_options)
    
    # Detect feature type
    is_numeric = pd.api.types.is_numeric_dtype(df[selected_feature])
    st.sidebar.write(f"Tipe fitur: {'Numerik' if is_numeric else 'Kategorikal'}")
    
    # Main visualization
    st.subheader(f"Distribusi Fitur '{selected_feature}'")
    
    plt.rcParams["figure.figsize"] = (12, 6)
    
    if selected_grouping != "Tidak ada":
        # Create comparison visualization
        st.subheader(f"Perbandingan berdasarkan '{selected_grouping}'")
        
        # Get unique groups
        unique_groups = df[selected_grouping].unique()
        unique_groups = [group for group in unique_groups if pd.notna(group)]
        
        if len(unique_groups) <= 5:  # Only show if not too many groups
            cols = st.columns(min(len(unique_groups), 3))
            
            for i, group in enumerate(unique_groups):
                group_data = df[df[selected_grouping] == group]
                
                with cols[i % 3]:
                    st.markdown(f"**{selected_grouping}: {group}**")
                    
                    if not group_data.empty and not group_data[selected_feature].isna().all():
                        fig, ax = plt.subplots(figsize=(8, 5))
                        
                        if is_numeric:
                            # Histogram for numeric data
                            sns.histplot(group_data[selected_feature].dropna(), 
                                       bins=min(20, len(group_data)), 
                                       kde=True, ax=ax)
                            ax.set_title(f"{group}")
                            ax.set_xlabel(selected_feature)
                            ax.set_ylabel("Frekuensi")
                        else:
                            # Count plot for categorical data
                            value_counts = group_data[selected_feature].value_counts()
                            if not value_counts.empty:
                                sns.countplot(y=selected_feature, data=group_data,
                                            order=value_counts.index[:10], ax=ax)  # Show top 10
                                ax.set_title(f"{group}")
                                ax.set_xlabel("Jumlah")
                        
                        plt.xticks(rotation=45)
                        plt.tight_layout()
                        st.pyplot(fig)
                    else:
                        st.info(f"Tidak ada data untuk {group}")
        else:
            st.warning(f"Terlalu banyak kategori dalam '{selected_grouping}' ({len(unique_groups)}). Menampilkan visualisasi gabungan.")
    
    # Overall distribution
    st.subheader(f"Distribusi Keseluruhan '{selected_feature}'")
    
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if is_numeric:
            # Histogram for numeric data
            sns.histplot(df[selected_feature].dropna(), bins=30, kde=True, ax=ax)
            ax.set_title(f"Distribusi {selected_feature}")
            ax.set_xlabel(selected_feature)
            ax.set_ylabel("Frekuensi")
        else:
            # Count plot for categorical data
            value_counts = df[selected_feature].value_counts()
            if len(value_counts) <= 20:  # Show all if not too many
                if len(value_counts) > 10:
                    sns.countplot(y=selected_feature, data=df, 
                                order=value_counts.index, ax=ax)
                else:
                    sns.countplot(x=selected_feature, data=df, 
                                order=value_counts.index, ax=ax)
                    plt.xticks(rotation=45)
            else:
                # Show only top 20
                top_values = value_counts.head(20)
                sns.countplot(y=selected_feature, data=df[df[selected_feature].isin(top_values.index)], 
                            order=top_values.index, ax=ax)
                ax.set_title(f"Top 20 {selected_feature}")
            
            ax.set_xlabel("Jumlah")
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error saat membuat visualisasi: {e}")
    

    # Statistics section
    st.subheader("Statistik Deskriptif")
    
    if is_numeric:
        # Numeric statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Statistik Keseluruhan**")
            st.write(df[selected_feature].describe())
        
        with col2:
            st.markdown("**Informasi Tambahan**")
            st.write(f"Jumlah nilai hilang: {df[selected_feature].isna().sum()}")
            st.write(f"Jumlah nilai unik: {df[selected_feature].nunique()}")
            
            if selected_grouping != "Tidak ada":
                st.markdown(f"**Statistik berdasarkan {selected_grouping}**")
                group_stats = df.groupby(selected_grouping)[selected_feature].describe()
                st.write(group_stats)
    else:
        # Categorical statistics
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Distribusi Kategori**")
            value_counts = df[selected_feature].value_counts().reset_index()
            value_counts.columns = [selected_feature, 'Jumlah']
            st.write(value_counts.head(10))
        
        with col2:
            st.markdown("**Informasi Tambahan**")
            st.write(f"Jumlah kategori: {df[selected_feature].nunique()}")
            st.write(f"Jumlah nilai hilang: {df[selected_feature].isna().sum()}")
            
            if selected_grouping != "Tidak ada":
                st.markdown(f"**Crosstab dengan {selected_grouping}**")
                crosstab = pd.crosstab(df[selected_feature], df[selected_grouping])
                st.write(crosstab)
    
    # Data preview
    if st.checkbox("Tampilkan Preview Data"):
        st.subheader("Preview Data")
        
        # Show selected columns
        if selected_grouping != "Tidak ada":
            preview_cols = [selected_feature, selected_grouping]
        else:
            preview_cols = [selected_feature]
        
        # Add option to show all columns
        if st.checkbox("Tampilkan semua kolom"):
            st.dataframe(df)
        else:
            st.dataframe(df[preview_cols])
    
    # Data summary
    st.subheader("Ringkasan Dataset")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Baris", df.shape[0])
    
    with col2:
        st.metric("Total Kolom", df.shape[1])
    
    with col3:
        missing_values = df.isnull().sum().sum()
        st.metric("Total Nilai Hilang", missing_values)

else:
    st.error("Tidak dapat memuat data. Pastikan file 'data.csv' tersedia dan dapat dibaca.")
