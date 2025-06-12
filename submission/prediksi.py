import streamlit as st

# PENTING: st.set_page_config() HARUS DIPANGGIL PERTAMA KALI, SEBELUM IMPORT LAINNYA
st.set_page_config(
    page_title="Student Status Prediction",
    page_icon="🎓",
    layout="wide"
)

import pandas as pd
import numpy as np
import random
import joblib
import io
import os
from sklearn.preprocessing import StandardScaler
import plotly.express as px
import plotly.graph_objects as go

# Cache untuk loading model dan preprocessing
@st.cache_resource
def load_models():
    """Load pre-trained models from submission folder"""
    try:
        # Model ada di folder submission/
        rf_model = joblib.load('submission/random_forest_model.pkl')
        dt_model = joblib.load('submission/decision_tree_model.pkl')
        return rf_model, dt_model
    except FileNotFoundError as e:
        st.error(f"❌ Model files not found! Error: {str(e)}")
        st.error("Please ensure 'random_forest_model.pkl' and 'decision_tree_model.pkl' are in the 'submission/' folder.")
        
        # Debug info
        st.write("**Debug Info:**")
        st.write(f"Current working directory: {os.getcwd()}")
        st.write(f"Files in current directory: {os.listdir('.')}")
        if os.path.exists('submission'):
            st.write(f"Files in submission folder: {os.listdir('submission/')}")
        else:
            st.write("❌ 'submission' folder not found!")
        
        return None, None
    except Exception as e:
        st.error(f"❌ Error loading models: {str(e)}")
        return None, None

@st.cache_data
def load_default_data():
    """Load default dataset from submission folder"""
    try:
        # Data juga ada di folder submission/
        df = pd.read_csv('submission/data.csv', low_memory=False)
        return df
    except FileNotFoundError:
        st.error("❌ Default dataset 'data.csv' not found in submission folder!")
        return None

def preprocess_data(df_raw):
    """Preprocess the data similar to training preprocessing"""
    # Drop baris dengan NaN dan reset index
    df_clean = df_raw.dropna().reset_index(drop=True).copy()
    
    # Salin kolom Status asli untuk referensi
    if 'Status' in df_clean.columns:
        df_clean['Status_Original'] = df_clean['Status']
    
    # Daftar fitur yang akan di-capping
    capping_features = [
        'Age_at_enrollment', 
        'Admission_grade',
        'Curricular_units_1st_sem_grade', 
        'Previous_qualification_grade', 
        'Course', 
        'Curricular_units_2nd_sem_grade'
    ]
    
    # Capping outlier hanya pada fitur tertentu
    for col in capping_features:
        if col in df_clean.columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            df_clean[col] = np.clip(df_clean[col], lower_bound, upper_bound)
    
    # Standardisasi HANYA fitur yang di-capping
    scaler = StandardScaler()
    existing_capped_features = [col for col in capping_features if col in df_clean.columns]
    
    if existing_capped_features:
        df_clean[existing_capped_features] = scaler.fit_transform(df_clean[existing_capped_features])
    
    return df_clean, scaler, existing_capped_features

def make_predictions(df_processed, rf_model, dt_model, num_samples=10):
    """Make predictions on processed data"""
    try:
        # Siapkan data prediksi
        X_all = df_processed[rf_model.feature_names_in_]
        
        # Ambil sampel acak
        sample_size = min(num_samples, len(df_processed))
        sample_indices = random.sample(range(len(df_processed)), sample_size)
        X_sample = X_all.iloc[sample_indices]
        
        # Ambil data asli
        status_original = df_processed.get('Status_Original', ['Unknown'] * len(sample_indices)).iloc[sample_indices].values
        
        # Prediksi
        rf_pred = rf_model.predict(X_sample)
        dt_pred = dt_model.predict(X_sample)
        
        # Buat hasil
        results = pd.DataFrame({
            'Sample_ID': range(1, len(sample_indices) + 1),
            'Status (Real)': status_original,
            'Random_Forest': rf_pred,
            'Decision_Tree': dt_pred
        })
        
        # Mapping untuk status (sesuaikan dengan encoding yang digunakan saat training)
        # Biasanya: 0: Dropout, 1: Enrolled, 2: Graduate atau sesuai alfabetical order
        try:
            # Coba deteksi dari data asli
            if 'Status_Original' in df_processed.columns:
                unique_statuses = sorted(df_processed['Status_Original'].unique())
                if len(unique_statuses) == 3:
                    status_mapping = {i: status for i, status in enumerate(unique_statuses)}
                else:
                    status_mapping = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
            else:
                status_mapping = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
        except:
            status_mapping = {0: 'Dropout', 1: 'Enrolled', 2: 'Graduate'}
        
        # Konversi prediksi numerik ke label jika diperlukan
        if results['Random_Forest'].dtype in ['int64', 'float64']:
            results['Random_Forest'] = results['Random_Forest'].map(status_mapping)
        if results['Decision_Tree'].dtype in ['int64', 'float64']:
            results['Decision_Tree'] = results['Decision_Tree'].map(status_mapping)
        
        return results, X_sample
        
    except Exception as e:
        st.error(f"Error during prediction: {str(e)}")
        return None, None

def create_visualizations(results):
    """Create visualizations for predictions"""
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("📊 Prediction Distribution")
        
        # Count predictions untuk kedua model
        rf_counts = results['Random_Forest'].value_counts()
        dt_counts = results['Decision_Tree'].value_counts()
        
        # Ensure all categories are present
        all_categories = ['Graduate', 'Dropout', 'Enrolled']
        rf_counts = rf_counts.reindex(all_categories, fill_value=0)
        dt_counts = dt_counts.reindex(all_categories, fill_value=0)
        
        fig = go.Figure(data=[
            go.Bar(name='Random Forest', x=rf_counts.index, y=rf_counts.values, marker_color='#2E8B57'),
            go.Bar(name='Decision Tree', x=dt_counts.index, y=dt_counts.values, marker_color='#4682B4')
        ])
        fig.update_layout(
            barmode='group', 
            title="Student Status Predictions by Model",
            xaxis_title="Status",
            yaxis_title="Count"
        )
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        st.subheader("🎯 Model Agreement")
        
        # Check agreement between models
        agreement = (results['Random_Forest'] == results['Decision_Tree']).sum()
        total = len(results)
        
        fig = go.Figure(data=[go.Pie(
            labels=['Agree', 'Disagree'],
            values=[agreement, total - agreement],
            hole=0.3,
            marker_colors=['#2E8B57', '#DC143C']
        )])
        fig.update_layout(title=f"Model Agreement: {agreement}/{total} ({agreement/total*100:.1f}%)")
        st.plotly_chart(fig, use_container_width=True)

def show_status_distribution(df):
    """Show distribution of Status in the dataset"""
    if 'Status' in df.columns:
        st.subheader("📈 Dataset Status Distribution")
        
        status_counts = df['Status'].value_counts()
        
        # Create pie chart
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
        fig = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=0.3,
            marker_colors=colors
        )])
        fig.update_layout(title="Distribution of Student Status")
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.plotly_chart(fig, use_container_width=True)
        with col2:
            st.write("**Status Counts:**")
            for status, count in status_counts.items():
                percentage = (count / len(df)) * 100
                st.write(f"• {status}: {count} ({percentage:.1f}%)")

# Header aplikasi
st.title("🎓 Student Status Prediction Dashboard")
st.markdown("*Predict whether students will Graduate, Dropout, or remain Enrolled*")
st.markdown("---")

# Load models
rf_model, dt_model = load_models()

if rf_model is None or dt_model is None:
    st.stop()

# Sidebar untuk pengaturan
st.sidebar.header("⚙️ Settings")

# Pilihan sumber data
data_source = st.sidebar.selectbox(
    "Select Data Source:",
    ["Default Dataset (data.csv)", "Upload New CSV"]
)

# Load atau upload data
df_raw = None

if data_source == "Default Dataset (data.csv)":
    df_raw = load_default_data()
    if df_raw is not None:
        st.sidebar.success(f"✅ Default data loaded: {len(df_raw)} rows")
else:
    uploaded_file = st.sidebar.file_uploader(
        "Choose CSV file:",
        type=['csv'],
        help="Upload a CSV file with the same features as the training data"
    )
    
    if uploaded_file is not None:
        try:
            df_raw = pd.read_csv(uploaded_file, low_memory=False)
            st.sidebar.success(f"✅ File uploaded: {len(df_raw)} rows")
        except Exception as e:
            st.sidebar.error(f"❌ Error reading file: {str(e)}")

# Pengaturan prediksi
if df_raw is not None:
    max_samples = min(50, len(df_raw))
    num_samples = st.sidebar.slider(
        "Number of samples to predict:",
        min_value=1,
        max_value=max_samples,
        value=min(10, max_samples)
    )
    
    show_data_info = st.sidebar.checkbox("Show Data Information", value=True)
    show_feature_importance = st.sidebar.checkbox("Show Feature Analysis", value=False)

# Main content
if df_raw is not None:
    # Informasi dataset
    if show_data_info:
        st.header("📋 Dataset Information")
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Rows", len(df_raw))
        with col2:
            st.metric("Total Columns", len(df_raw.columns))
        with col3:
            missing_values = df_raw.isnull().sum().sum()
            st.metric("Missing Values", missing_values)
        with col4:
            if 'Status' in df_raw.columns:
                unique_status = df_raw['Status'].nunique()
                st.metric("Status Categories", unique_status)
            else:
                st.metric("Status Categories", "N/A")
        
        # Show status distribution
        show_status_distribution(df_raw)
        
        with st.expander("View Raw Data Sample"):
            st.dataframe(df_raw.head())
    
    # Preprocessing
    with st.spinner("Processing data..."):
        df_processed, scaler, capped_features = preprocess_data(df_raw)
    
    st.success("✅ Data preprocessing completed!")
    
    # Show preprocessing info
    with st.expander("Preprocessing Details"):
        st.write("**Features processed with outlier capping + standardization:**")
        for feature in capped_features:
            st.write(f"• {feature}")
        
        st.write("\n**Processing steps:**")
        st.write("1. Outlier capping using IQR method (Q1 - 1.5×IQR, Q3 + 1.5×IQR)")
        st.write("2. Standardization (StandardScaler) applied only to capped features")
        st.write("3. All other numerical features remain unchanged")
        st.write("4. No categorical encoding (all features are numerical except Status)")
    
    # Prediksi
    st.header("🔮 Student Status Predictions")
    
    if st.button("🚀 Run Predictions", type="primary"):
        with st.spinner("Making predictions..."):
            results, X_sample = make_predictions(df_processed, rf_model, dt_model, num_samples)
        
        if results is not None:
            st.success(f"✅ Predictions completed for {len(results)} students!")
            
            # Tampilkan hasil
            st.subheader("📋 Prediction Results")
            
            # Color coding untuk hasil
            def highlight_predictions(row):
                colors = []
                for col in row.index:
                    if col in ['Random_Forest', 'Decision_Tree', 'Status (Real)']:
                        if row[col] == 'Graduate':
                            colors.append('background-color: #d4edda; color: #155724')
                        elif row[col] == 'Dropout':
                            colors.append('background-color: #f8d7da; color: #721c24')
                        elif row[col] == 'Enrolled':
                            colors.append('background-color: #fff3cd; color: #856404')
                        else:
                            colors.append('')
                    else:
                        colors.append('')
                return colors
            
            styled_results = results.style.apply(highlight_predictions, axis=1)
            st.dataframe(styled_results, use_container_width=True)
            
            # Visualisasi
            create_visualizations(results)
            
            # Summary statistics
            st.subheader("📊 Prediction Summary")
            
            # Random Forest Summary
            st.write("**Random Forest Predictions:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                rf_graduate = (results['Random_Forest'] == 'Graduate').sum()
                st.metric("Graduate", rf_graduate)
            with col2:
                rf_dropout = (results['Random_Forest'] == 'Dropout').sum()
                st.metric("Dropout", rf_dropout)
            with col3:
                rf_enrolled = (results['Random_Forest'] == 'Enrolled').sum()
                st.metric("Enrolled", rf_enrolled)
            
            # Decision Tree Summary
            st.write("**Decision Tree Predictions:**")
            col1, col2, col3 = st.columns(3)
            with col1:
                dt_graduate = (results['Decision_Tree'] == 'Graduate').sum()
                st.metric("Graduate", dt_graduate)
            with col2:
                dt_dropout = (results['Decision_Tree'] == 'Dropout').sum()
                st.metric("Dropout", dt_dropout)
            with col3:
                dt_enrolled = (results['Decision_Tree'] == 'Enrolled').sum()
                st.metric("Enrolled", dt_enrolled)
            
            # Download hasil
            csv_buffer = io.StringIO()
            results.to_csv(csv_buffer, index=False)
            csv_data = csv_buffer.getvalue()
            
            st.download_button(
                label="📥 Download Results as CSV",
                data=csv_data,
                file_name=f"student_status_predictions_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv"
            )
    
    # Feature analysis
    if show_feature_importance:
        st.header("🔍 Feature Analysis")
        
        col1, col2 = st.columns(2)
        
        # Random Forest Feature Importance
        if hasattr(rf_model, 'feature_importances_'):
            with col1:
                st.subheader("Random Forest Feature Importance")
                rf_importance = pd.DataFrame({
                    'Feature': rf_model.feature_names_in_,
                    'Importance': rf_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    rf_importance.head(15),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 15 Features - Random Forest",
                    color='Importance',
                    color_continuous_scale='greens'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)
        
        # Decision Tree Feature Importance
        if hasattr(dt_model, 'feature_importances_'):
            with col2:
                st.subheader("Decision Tree Feature Importance")
                dt_importance = pd.DataFrame({
                    'Feature': dt_model.feature_names_in_,
                    'Importance': dt_model.feature_importances_
                }).sort_values('Importance', ascending=False)
                
                fig = px.bar(
                    dt_importance.head(15),
                    x='Importance',
                    y='Feature',
                    orientation='h',
                    title="Top 15 Features - Decision Tree",
                    color='Importance',
                    color_continuous_scale='blues'
                )
                fig.update_layout(yaxis={'categoryorder': 'total ascending'})
                st.plotly_chart(fig, use_container_width=True)

else:
    st.info("👆 Please select a data source from the sidebar to begin.")
    
    # Tampilkan informasi tentang format data yang diharapkan
    st.header("📝 Expected Data Format")
    st.markdown("""
    Your CSV file should contain student academic data with numerical features:
    
    **Features that will be processed (capped + standardized):**
    - `Age_at_enrollment`: Student's age at enrollment
    - `Admission_grade`: Grade obtained in admission
    - `Curricular_units_1st_sem_grade`: First semester grade
    - `Previous_qualification_grade`: Previous qualification grade
    - `Course`: Course identifier
    - `Curricular_units_2nd_sem_grade`: Second semester grade
    
    **Other numerical features:**
    - All other numerical columns will remain unchanged (no capping, no scaling)
    
    **Target Variable (categorical):**
    - `Status`: Student status with values:
      - **Graduate**: Student successfully completed the program
      - **Dropout**: Student dropped out
      - **Enrolled**: Student is currently enrolled
    
    **Important Notes:**
    - All features should be numerical except `Status`
    - No categorical encoding is performed
    - Only the 6 specified features undergo outlier capping and standardization
    """)
    
    # Show example data structure
    st.subheader("📊 Expected Status Distribution")
    example_data = {
        'Status': ['Graduate', 'Dropout', 'Enrolled'],
        'Expected Count': [2209, 1421, 794],
        'Percentage': ['49.9%', '32.1%', '18.0%']
    }
    st.table(pd.DataFrame(example_data))

# Footer
st.markdown("---")
st.markdown("*Student Status Prediction Dashboard - Random Forest & Decision Tree Models*")
