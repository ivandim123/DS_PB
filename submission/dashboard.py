import os
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set page configuration
st.set_page_config(
    page_title="Dashboard Analisis Status Siswa",
    layout="wide"
)

# Set style
sns.set(style="whitegrid")

# Dashboard title
st.title("Dashboard Analisis Status Siswa")
st.markdown("Dashboard ini menampilkan visualisasi data dengan fokus pada status siswa (Dropout, Enrolled, Graduate).")

@st.cache_data
def load_csv_from_path():
    """Try to load CSV from various paths"""
    # Get the directory where the script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    possible_paths = [
        "data.csv",  # Current working directory
        os.path.join(script_dir, "data.csv"),  # Same directory as script
        "./data.csv", 
        "dataset/data.csv",
        "data/data.csv",
        os.path.join(script_dir, "dataset", "data.csv"),
        os.path.join(script_dir, "data", "data.csv")
    ]
    
    # Also check for any CSV file in the current directory
    current_dir_files = [f for f in os.listdir('.') if f.endswith('.csv')]
    if current_dir_files:
        for csv_file in current_dir_files:
            possible_paths.insert(0, csv_file)  # Add to beginning of list
    
    # Check script directory for CSV files too
    try:
        script_dir_files = [f for f in os.listdir(script_dir) if f.endswith('.csv')]
        if script_dir_files:
            for csv_file in script_dir_files:
                possible_paths.insert(0, os.path.join(script_dir, csv_file))
    except:
        pass
    
    st.sidebar.write("**Mencari file CSV di lokasi berikut:**")
    for path in possible_paths[:5]:  # Show first 5 paths being checked
        st.sidebar.write(f"• {path}")
        try:
            if os.path.exists(path):
                df = pd.read_csv(path)
                st.sidebar.success(f"✅ Found: {path}")
                return df, path
            else:
                st.sidebar.write(f"❌ Not found: {path}")
        except Exception as e:
            st.sidebar.write(f"❌ Error reading {path}: {str(e)[:50]}")
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
        # Menggunakan 'Status' sebagai nama kolom di data sampel
        'Status': np.random.choice(['Dropout', 'Enrolled', 'Graduate'], n, p=[0.3, 0.4, 0.3])
    }
    
    return pd.DataFrame(data)

# Display current working directory info
st.sidebar.subheader("Info Direktori")
st.sidebar.write(f"Working Directory: {os.getcwd()}")
st.sidebar.write(f"Script Directory: {os.path.dirname(os.path.abspath(__file__)) if '__file__' in globals() else 'Unknown'}")

# List all CSV files in current directory
csv_files = [f for f in os.listdir('.') if f.endswith('.csv')]
if csv_files:
    st.sidebar.write("**File CSV yang ditemukan:**")
    for csv_file in csv_files:
        st.sidebar.write(f"• {csv_file}")

# Try to load data from file first
df, file_path = load_csv_from_path()

if df is not None:
    st.success(f"✅ Data berhasil dimuat dari: {file_path}")
    st.info(f"Dataset memiliki {df.shape[0]} baris dan {df.shape[1]} kolom")
else:
    # Show file uploader if no file found
    st.warning("⚠️ File CSV tidak ditemukan secara otomatis. Silakan upload file CSV Anda.")
    
    # Show what files are available in current directory
    all_files = os.listdir('.')
    st.write("**File yang tersedia di direktori saat ini:**")
    st.write(all_files[:10])  # Show first 10 files
    
    uploaded_file = st.file_uploader("Pilih file CSV", type="csv")
    
    if uploaded_file is not None:
        try:
            df = pd.read_csv(uploaded_file)
            st.success("✅ Data berhasil dimuat dari file yang diupload")
        except Exception as e:
            st.error(f"❌ Error saat membaca file yang diupload: {e}")
            df = None
    else:
        # Use sample data if no file uploaded
        st.info("📊 Menggunakan data sampel untuk demonstrasi. Upload file CSV Anda untuk analisis data sebenarnya.")
        df = create_sample_data()

# --- START PERUBAHAN DI SINI ---
if df is not None:
    # Cek apakah kolom 'Status' ada dan ganti namanya menjadi 'Target'
    if 'Status' in df.columns:
        df = df.rename(columns={'Status': 'Target'})
        st.sidebar.success("Kolom 'Status' ditemukan dan diganti namanya menjadi 'Target'.")
    elif 'Target' not in df.columns: # Jika tidak ada 'Status' dan tidak ada 'Target'
        st.error("Kolom 'Status' atau 'Target' tidak ditemukan dalam dataset Anda. Pastikan dataset memiliki salah satu kolom tersebut untuk analisis status siswa.")
        st.stop() # Hentikan eksekusi jika tidak ada kolom status
# --- AKHIR PERUBAHAN DI SINI ---

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
    st.sidebar.header("Pilih Fitur untuk Analisis Multivariat")
    
    # Exclude 'Student_ID' and 'Target' from features to analyze against target
    features_for_analysis = [col for col in df.columns if col not in ['Student_ID', 'Target']]
    selected_feature = st.sidebar.selectbox("Pilih Fitur untuk Dianalisis:", features_for_analysis)
    
    # Detect feature type
    is_numeric = pd.api.types.is_numeric_dtype(df[selected_feature])
    st.sidebar.write(f"Tipe fitur yang dipilih: {'Numerik' if is_numeric else 'Kategorikal'}")
    
    # --- Multivariate Analysis Section ---
    st.markdown("---")
    st.header("Analisis Multivariat terhadap Status Siswa (Target)")
    st.markdown("Bagian ini menampilkan hubungan antara fitur yang dipilih dengan status siswa (Dropout, Enrolled, Graduate).")

    plt.rcParams["figure.figsize"] = (12, 6)

    # Visualization based on feature type
    if is_numeric:
        st.subheader(f"Distribusi '{selected_feature}' berdasarkan Status Siswa")
        st.info(f"Visualisasi ini menunjukkan bagaimana nilai '{selected_feature}' (misalnya, nilai ujian, usia) berbeda di antara siswa yang Dropout, Enrolled, dan Graduate.")
        
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.boxplot(x='Target', y=selected_feature, data=df, ax=ax, palette='viridis')
        ax.set_title(f'Box Plot {selected_feature} vs. Target')
        ax.set_xlabel('Status Siswa (Target)')
        ax.set_ylabel(selected_feature)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

        fig, ax = plt.subplots(figsize=(10, 6))
        sns.violinplot(x='Target', y=selected_feature, data=df, ax=ax, palette='plasma')
        ax.set_title(f'Violin Plot {selected_feature} vs. Target')
        ax.set_xlabel('Status Siswa (Target)')
        ax.set_ylabel(selected_feature)
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        st.pyplot(fig)

    else: # Categorical feature
        st.subheader(f"Distribusi '{selected_feature}' berdasarkan Status Siswa")
        st.info(f"Visualisasi ini menunjukkan proporsi kategori '{selected_feature}' (misalnya, jenis kelamin, beasiswa) untuk setiap status siswa (Dropout, Enrolled, Graduate).")

        fig, ax = plt.subplots(figsize=(12, 7))
        # Use value_counts to get order for better visualization if many categories
        order_values = df[selected_feature].value_counts().index
        if len(order_values) > 10: # Limit for readability
            order_values = order_values[:10]
            st.warning(f"Menampilkan top 10 kategori untuk '{selected_feature}' karena terlalu banyak kategori.")

        sns.countplot(x=selected_feature, hue='Target', data=df, ax=ax, order=order_values, palette='coolwarm')
        ax.set_title(f'Count Plot {selected_feature} vs. Target')
        ax.set_xlabel(selected_feature)
        ax.set_ylabel('Jumlah Siswa')
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Status Siswa')
        plt.tight_layout()
        st.pyplot(fig)

        # Show crosstab for categorical features
        st.subheader(f"Tabel Kontingensi '{selected_feature}' dan 'Target'")
        crosstab = pd.crosstab(df[selected_feature], df['Target'], normalize='index').style.format("{:.2%}")
        st.dataframe(crosstab)
        st.markdown("Tabel di atas menunjukkan persentase setiap kategori dari fitur yang dipilih untuk setiap status siswa.")

    # --- Correlation Matrix (for numeric features vs. encoded target) ---
    st.markdown("---")
    st.header("Matriks Korelasi (Fitur Numerik vs. Target)")
    st.info("Matriks ini menunjukkan seberapa kuat hubungan linear antara fitur numerik dan status 'Dropout'. Untuk keperluan korelasi, 'Dropout' diwakili sebagai 1 dan status lain sebagai 0.")

    # Create a numerical representation of 'Target' for correlation
    df_corr = df.copy()
    df_corr['Is_Dropout'] = df_corr['Target'].apply(lambda x: 1 if x == 'Dropout' else 0)
    
    numeric_cols = df_corr.select_dtypes(include=np.number).columns.tolist()
    # Ensure 'Is_Dropout' is included and 'Student_ID' is excluded if it exists
    if 'Student_ID' in numeric_cols:
        numeric_cols.remove('Student_ID')
    
    if 'Is_Dropout' not in numeric_cols:
        numeric_cols.append('Is_Dropout')

    if len(numeric_cols) > 1:
        correlation_matrix = df_corr[numeric_cols].corr()

        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, ax=ax)
        ax.set_title('Matriks Korelasi Fitur Numerik dan Status Dropout')
        plt.tight_layout()
        st.pyplot(fig)
        st.markdown("Nilai korelasi mendekati 1 atau -1 menunjukkan hubungan yang kuat, sementara mendekati 0 menunjukkan hubungan yang lemah.")
    else:
        st.warning("Tidak cukup fitur numerik dalam dataset untuk membuat matriks korelasi.")

    # --- Overall Distribution (Existing section, kept for completeness) ---
    st.markdown("---")
    st.subheader(f"Distribusi Keseluruhan '{selected_feature}'")
    st.info(f"Visualisasi ini menampilkan distribusi fitur '{selected_feature}' secara keseluruhan tanpa membedakan status siswa.")
    
    try:
        fig, ax = plt.subplots(figsize=(12, 6))
        
        if is_numeric:
            sns.histplot(df[selected_feature].dropna(), bins=30, kde=True, ax=ax)
            ax.set_title(f"Distribusi {selected_feature}")
            ax.set_xlabel(selected_feature)
            ax.set_ylabel("Frekuensi")
        else:
            value_counts = df[selected_feature].value_counts()
            if len(value_counts) <= 20:
                if len(value_counts) > 10:
                    sns.countplot(y=selected_feature, data=df, 
                                  order=value_counts.index, ax=ax)
                else:
                    sns.countplot(x=selected_feature, data=df, 
                                  order=value_counts.index, ax=ax)
                    plt.xticks(rotation=45)
            else:
                top_values = value_counts.head(20)
                sns.countplot(y=selected_feature, data=df[df[selected_feature].isin(top_values.index)], 
                              order=top_values.index, ax=ax)
                ax.set_title(f"Top 20 {selected_feature}")
            
            ax.set_xlabel("Jumlah")
        
        plt.tight_layout()
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error saat membuat visualisasi: {e}")
    
    # --- Statistics section (Existing section, modified for target analysis) ---
    st.markdown("---")
    st.subheader("Statistik Deskriptif")
    
    if is_numeric:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Statistik Keseluruhan**")
            st.write(df[selected_feature].describe())
        
        with col2:
            st.markdown("**Informasi Tambahan**")
            st.write(f"Jumlah nilai hilang: {df[selected_feature].isna().sum()}")
            st.write(f"Jumlah nilai unik: {df[selected_feature].nunique()}")
            
        st.markdown(f"**Statistik '{selected_feature}' berdasarkan Status Siswa (Target)**")
        group_stats = df.groupby('Target')[selected_feature].describe()
        st.write(group_stats)
        st.info("Tabel di atas menampilkan statistik ringkasan (rata-rata, standar deviasi, kuartil) dari fitur numerik untuk setiap kategori status siswa.")
    else:
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("**Distribusi Kategori Keseluruhan**")
            value_counts = df[selected_feature].value_counts().reset_index()
            value_counts.columns = [selected_feature, 'Jumlah']
            st.write(value_counts.head(10))
        
        with col2:
            st.markdown("**Informasi Tambahan**")
            st.write(f"Jumlah kategori: {df[selected_feature].nunique()}")
            st.write(f"Jumlah nilai hilang: {df[selected_feature].isna().sum()}")
            
        st.markdown(f"**Crosstab '{selected_feature}' dengan 'Target'**")
        crosstab = pd.crosstab(df[selected_feature], df['Target'])
        st.write(crosstab)
        st.info("Tabel ini menunjukkan jumlah siswa untuk setiap kombinasi kategori fitur dan status siswa.")
    
    # Data preview
    st.markdown("---")
    if st.checkbox("Tampilkan Preview Data"):
        st.subheader("Preview Data")
        
        # Show selected columns
        preview_cols = [selected_feature, 'Target'] if 'Target' in df.columns else [selected_feature]
        
        # Add option to show all columns
        if st.checkbox("Tampilkan semua kolom dataset"):
            st.dataframe(df)
        else:
            st.dataframe(df[preview_cols])
    
    # Data summary
    st.markdown("---")
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
    st.error("❌ Tidak dapat memuat data. Pastikan file CSV tersedia dan dapat dibaca.")
    st.write("**Solusi yang bisa dicoba:**")
    st.write("1. Pastikan file CSV berada di direktori yang sama dengan script Python")
    st.write("2. Periksa nama file (case-sensitive)")
    st.write("3. Pastikan file tidak sedang dibuka di aplikasi lain")
    st.write("4. Upload file melalui file uploader di atas")
