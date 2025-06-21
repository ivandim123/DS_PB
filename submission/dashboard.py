import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os

# PENTING: st.set_page_config() HARUS DIPANGGIL PERTAMA KALI
st.set_page_config(
    page_title="Student Dropout Analysis Dashboard",
    page_icon="📊",
    layout="wide"
)

@st.cache_data
def load_data():
    """Load dataset dari folder submission"""
    try:
        df = pd.read_csv('submission/data.csv', low_memory=False)
        return df
    except FileNotFoundError:
        st.error("❌ Dataset 'data.csv' tidak ditemukan di folder submission!")
        return None

def create_status_overview(df):
    """Membuat overview status siswa"""
    st.header("📈 Overview Status Siswa")
    
    # Hitung distribusi status
    status_counts = df['Status'].value_counts()
    total_students = len(df)
    
    # Metrics cards
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            label="Total Siswa",
            value=f"{total_students:,}",
            delta=None
        )
    
    with col2:
        graduate_pct = (status_counts.get('Graduate', 0) / total_students) * 100
        st.metric(
            label="Graduate",
            value=f"{status_counts.get('Graduate', 0):,}",
            delta=f"{graduate_pct:.1f}%"
        )
    
    with col3:
        enrolled_pct = (status_counts.get('Enrolled', 0) / total_students) * 100
        st.metric(
            label="Enrolled",
            value=f"{status_counts.get('Enrolled', 0):,}",
            delta=f"{enrolled_pct:.1f}%"
        )
    
    with col4:
        dropout_pct = (status_counts.get('Dropout', 0) / total_students) * 100
        st.metric(
            label="Dropout",
            value=f"{status_counts.get('Dropout', 0):,}",
            delta=f"{dropout_pct:.1f}%"
        )

def create_status_distribution_charts(df):
    """Membuat chart distribusi status siswa"""
    st.subheader("🎯 Distribusi Status Siswa")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Pie Chart
        status_counts = df['Status'].value_counts()
        colors = ['#2E8B57', '#4682B4', '#DC143C']  # Green, Blue, Red
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=status_counts.index,
            values=status_counts.values,
            hole=0.4,
            marker_colors=colors,
            textinfo='label+percent',
            textfont_size=12
        )])
        
        fig_pie.update_layout(
            title="Distribusi Status Siswa",
            title_x=0.5,
            height=400
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)
    
    with col2:
        # Bar Chart
        fig_bar = px.bar(
            x=status_counts.index,
            y=status_counts.values,
            color=status_counts.index,
            color_discrete_map={
                'Graduate': '#2E8B57',
                'Enrolled': '#4682B4',
                'Dropout': '#DC143C'
            },
            title="Jumlah Siswa per Status"
        )
        
        fig_bar.update_layout(
            xaxis_title="Status",
            yaxis_title="Jumlah Siswa",
            showlegend=False,
            height=400
        )
        
        # Tambahkan angka di atas bar
        fig_bar.update_traces(texttemplate='%{y}', textposition='outside')
        
        st.plotly_chart(fig_bar, use_container_width=True)

def create_demographic_analysis(df):
    """Analisis faktor demografis"""
    st.header("👥 Analisis Faktor Demografis")
    
    # Age Analysis
    st.subheader("📊 Analisis Usia")
    col1, col2 = st.columns(2)
    
    with col1:
        # Age distribution by status
        fig_age = px.histogram(
            df, 
            x='Age_at_enrollment', 
            color='Status',
            color_discrete_map={
                'Graduate': '#2E8B57',
                'Enrolled': '#4682B4',
                'Dropout': '#DC143C'
            },
            title="Distribusi Usia Berdasarkan Status",
            nbins=20
        )
        fig_age.update_layout(xaxis_title="Usia saat Pendaftaran", yaxis_title="Jumlah Siswa")
        st.plotly_chart(fig_age, use_container_width=True)
    
    with col2:
        # Average age by status
        avg_age = df.groupby('Status')['Age_at_enrollment'].mean().reset_index()
        fig_avg_age = px.bar(
            avg_age,
            x='Status',
            y='Age_at_enrollment',
            color='Status',
            color_discrete_map={
                'Graduate': '#2E8B57',
                'Enrolled': '#4682B4',
                'Dropout': '#DC143C'
            },
            title="Rata-rata Usia per Status"
        )
        fig_avg_age.update_layout(
            xaxis_title="Status",
            yaxis_title="Rata-rata Usia",
            showlegend=False
        )
        fig_avg_age.update_traces(texttemplate='%{y:.1f}', textposition='outside')
        st.plotly_chart(fig_avg_age, use_container_width=True)
    
    # Gender Analysis
    st.subheader("🚻 Analisis Gender")
    col1, col2 = st.columns(2)
    
    # Map gender values
    df['Gender_Label'] = df['Gender'].map({1: 'Male', 0: 'Female'})
    
    with col1:
        # Gender distribution by status
        gender_status = df.groupby(['Gender_Label', 'Status']).size().reset_index(name='Count')
        fig_gender = px.bar(
            gender_status,
            x='Gender_Label',
            y='Count',
            color='Status',
            color_discrete_map={
                'Graduate': '#2E8B57',
                'Enrolled': '#4682B4',
                'Dropout': '#DC143C'
            },
            title="Distribusi Gender Berdasarkan Status"
        )
        fig_gender.update_layout(xaxis_title="Gender", yaxis_title="Jumlah Siswa")
        st.plotly_chart(fig_gender, use_container_width=True)
    
    with col2:
        # Dropout rate by gender
        dropout_by_gender = df.groupby('Gender_Label').apply(
            lambda x: (x['Status'] == 'Dropout').sum() / len(x) * 100
        ).reset_index(name='Dropout_Rate')
        
        fig_dropout_gender = px.bar(
            dropout_by_gender,
            x='Gender_Label',
            y='Dropout_Rate',
            color='Gender_Label',
            color_discrete_map={'Male': '#4169E1', 'Female': '#FF69B4'},
            title="Tingkat Dropout per Gender (%)"
        )
        fig_dropout_gender.update_layout(
            xaxis_title="Gender",
            yaxis_title="Tingkat Dropout (%)",
            showlegend=False
        )
        fig_dropout_gender.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
        st.plotly_chart(fig_dropout_gender, use_container_width=True)

def create_academic_performance_analysis(df):
    """Analisis performa akademik"""
    st.header("🎓 Analisis Performa Akademik")
    
    # Admission Grade Analysis
    st.subheader("📝 Analisis Nilai Penerimaan")
    col1, col2 = st.columns(2)
    
    with col1:
        # Average admission grade by status
        avg_admission = df.groupby('Status')['Admission_grade'].mean().reset_index()
        fig_admission = px.bar(
            avg_admission,
            x='Status',
            y='Admission_grade',
            color='Status',
            color_discrete_map={
                'Graduate': '#2E8B57',
                'Enrolled': '#4682B4',
                'Dropout': '#DC143C'
            },
            title="Rata-rata Nilai Penerimaan per Status"
        )
        fig_admission.update_layout(
            xaxis_title="Status",
            yaxis_title="Rata-rata Nilai Penerimaan",
            showlegend=False
        )
        fig_admission.update_traces(texttemplate='%{y:.1f}', textposition='outside')
        st.plotly_chart(fig_admission, use_container_width=True)
    
    with col2:
        # Admission grade distribution
        fig_admission_dist = px.histogram(
            df,
            x='Admission_grade',
            color='Status',
            color_discrete_map={
                'Graduate': '#2E8B57',
                'Enrolled': '#4682B4',
                'Dropout': '#DC143C'
            },
            title="Distribusi Nilai Penerimaan",
            nbins=30
        )
        fig_admission_dist.update_layout(xaxis_title="Nilai Penerimaan", yaxis_title="Jumlah Siswa")
        st.plotly_chart(fig_admission_dist, use_container_width=True)
    
    # Semester Performance
    st.subheader("📚 Performa Semester")
    col1, col2 = st.columns(2)
    
    with col1:
        # 1st semester grades
        avg_sem1 = df.groupby('Status')['Curricular_units_1st_sem_grade'].mean().reset_index()
        fig_sem1 = px.bar(
            avg_sem1,
            x='Status',
            y='Curricular_units_1st_sem_grade',
            color='Status',
            color_discrete_map={
                'Graduate': '#2E8B57',
                'Enrolled': '#4682B4',
                'Dropout': '#DC143C'
            },
            title="Rata-rata Nilai Semester 1"
        )
        fig_sem1.update_layout(
            xaxis_title="Status",
            yaxis_title="Rata-rata Nilai Semester 1",
            showlegend=False
        )
        fig_sem1.update_traces(texttemplate='%{y:.1f}', textposition='outside')
        st.plotly_chart(fig_sem1, use_container_width=True)
    
    with col2:
        # 2nd semester grades
        avg_sem2 = df.groupby('Status')['Curricular_units_2nd_sem_grade'].mean().reset_index()
        fig_sem2 = px.bar(
            avg_sem2,
            x='Status',
            y='Curricular_units_2nd_sem_grade',
            color='Status',
            color_discrete_map={
                'Graduate': '#2E8B57',
                'Enrolled': '#4682B4',
                'Dropout': '#DC143C'
            },
            title="Rata-rata Nilai Semester 2"
        )
        fig_sem2.update_layout(
            xaxis_title="Status",
            yaxis_title="Rata-rata Nilai Semester 2",
            showlegend=False
        )
        fig_sem2.update_traces(texttemplate='%{y:.1f}', textposition='outside')
        st.plotly_chart(fig_sem2, use_container_width=True)

def create_financial_analysis(df):
    """Analisis faktor finansial"""
    st.header("💰 Analisis Faktor Finansial")
    
    col1, col2 = st.columns(2)
    
    # Map financial indicators
    df['Scholarship_Label'] = df['Scholarship_holder'].map({1: 'Yes', 0: 'No'})
    df['Debtor_Label'] = df['Debtor'].map({1: 'Yes', 0: 'No'})
    df['Tuition_Label'] = df['Tuition_fees_up_to_date'].map({1: 'Yes', 0: 'No'})
    
    with col1:
        # Scholarship analysis
        scholarship_status = df.groupby(['Scholarship_Label', 'Status']).size().reset_index(name='Count')
        fig_scholarship = px.bar(
            scholarship_status,
            x='Scholarship_Label',
            y='Count',
            color='Status',
            color_discrete_map={
                'Graduate': '#2E8B57',
                'Enrolled': '#4682B4',
                'Dropout': '#DC143C'
            },
            title="Status Siswa Berdasarkan Beasiswa"
        )
        fig_scholarship.update_layout(xaxis_title="Penerima Beasiswa", yaxis_title="Jumlah Siswa")
        st.plotly_chart(fig_scholarship, use_container_width=True)
        
        # Dropout rate by scholarship
        dropout_by_scholarship = df.groupby('Scholarship_Label').apply(
            lambda x: (x['Status'] == 'Dropout').sum() / len(x) * 100
        ).reset_index(name='Dropout_Rate')
        
        fig_dropout_scholarship = px.bar(
            dropout_by_scholarship,
            x='Scholarship_Label',
            y='Dropout_Rate',
            color='Scholarship_Label',
            color_discrete_map={'Yes': '#32CD32', 'No': '#FF6347'},
            title="Tingkat Dropout: Penerima vs Non-Penerima Beasiswa"
        )
        fig_dropout_scholarship.update_layout(
            xaxis_title="Penerima Beasiswa",
            yaxis_title="Tingkat Dropout (%)",
            showlegend=False
        )
        fig_dropout_scholarship.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
        st.plotly_chart(fig_dropout_scholarship, use_container_width=True)
    
    with col2:
        # Tuition fees analysis
        tuition_status = df.groupby(['Tuition_Label', 'Status']).size().reset_index(name='Count')
        fig_tuition = px.bar(
            tuition_status,
            x='Tuition_Label',
            y='Count',
            color='Status',
            color_discrete_map={
                'Graduate': '#2E8B57',
                'Enrolled': '#4682B4',
                'Dropout': '#DC143C'
            },
            title="Status Siswa Berdasarkan Pembayaran SPP"
        )
        fig_tuition.update_layout(xaxis_title="SPP Up to Date", yaxis_title="Jumlah Siswa")
        st.plotly_chart(fig_tuition, use_container_width=True)
        
        # Dropout rate by tuition payment
        dropout_by_tuition = df.groupby('Tuition_Label').apply(
            lambda x: (x['Status'] == 'Dropout').sum() / len(x) * 100
        ).reset_index(name='Dropout_Rate')
        
        fig_dropout_tuition = px.bar(
            dropout_by_tuition,
            x='Tuition_Label',
            y='Dropout_Rate',
            color='Tuition_Label',
            color_discrete_map={'Yes': '#32CD32', 'No': '#FF6347'},
            title="Tingkat Dropout Berdasarkan Status Pembayaran SPP"
        )
        fig_dropout_tuition.update_layout(
            xaxis_title="SPP Up to Date",
            yaxis_title="Tingkat Dropout (%)",
            showlegend=False
        )
        fig_dropout_tuition.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
        st.plotly_chart(fig_dropout_tuition, use_container_width=True)

def create_attendance_analysis(df):
    """Analisis pola kehadiran"""
    st.header("🕐 Analisis Pola Kehadiran")
    
    # --- Add this line to debug ---
    st.write("Columns in DataFrame:", df.columns.tolist())
    # --- End debug line ---

    # Map attendance pattern
    df['Attendance_Label'] = df['Daytime/evening_attendance'].map({1: 'Daytime', 0: 'Evening'})
    # ... rest of your code

def create_parents_background_analysis(df):
    """Analisis latar belakang orang tua"""
    st.header("👨‍👩‍👧‍👦 Analisis Latar Belakang Orang Tua")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Mother's qualification vs dropout rate
        st.subheader("📚 Kualifikasi Ibu")
        
        # Group mother's qualification into ranges for better visualization
        df['Mother_Qual_Group'] = pd.cut(df["Mother's_qualification"], 
                                       bins=[0, 10, 20, 30, 44], 
                                       labels=['1-10', '11-20', '21-30', '31-44'])
        
        dropout_by_mother = df.groupby('Mother_Qual_Group').apply(
            lambda x: (x['Status'] == 'Dropout').sum() / len(x) * 100
        ).reset_index(name='Dropout_Rate')
        
        fig_mother = px.bar(
            dropout_by_mother,
            x='Mother_Qual_Group',
            y='Dropout_Rate',
            color='Mother_Qual_Group',
            title="Tingkat Dropout Berdasarkan Kualifikasi Ibu"
        )
        fig_mother.update_layout(
            xaxis_title="Tingkat Kualifikasi Ibu",
            yaxis_title="Tingkat Dropout (%)",
            showlegend=False
        )
        fig_mother.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
        st.plotly_chart(fig_mother, use_container_width=True)
    
    with col2:
        # Father's qualification vs dropout rate
        st.subheader("📚 Kualifikasi Ayah")
        
        # Group father's qualification into ranges
        df['Father_Qual_Group'] = pd.cut(df["Father's_qualification"], 
                                       bins=[0, 10, 20, 30, 44], 
                                       labels=['1-10', '11-20', '21-30', '31-44'])
        
        dropout_by_father = df.groupby('Father_Qual_Group').apply(
            lambda x: (x['Status'] == 'Dropout').sum() / len(x) * 100
        ).reset_index(name='Dropout_Rate')
        
        fig_father = px.bar(
            dropout_by_father,
            x='Father_Qual_Group',
            y='Dropout_Rate',
            color='Father_Qual_Group',
            title="Tingkat Dropout Berdasarkan Kualifikasi Ayah"
        )
        fig_father.update_layout(
            xaxis_title="Tingkat Kualifikasi Ayah",
            yaxis_title="Tingkat Dropout (%)",
            showlegend=False
        )
        fig_father.update_traces(texttemplate='%{y:.1f}%', textposition='outside')
        st.plotly_chart(fig_father, use_container_width=True)

def create_key_insights(df):
    """Membuat rangkuman insight kunci"""
    st.header("🔍 Insight Kunci")
    
    # Calculate key statistics
    total_students = len(df)
    dropout_rate = (df['Status'] == 'Dropout').sum() / total_students * 100
    
    # Age insights
    avg_age_dropout = df[df['Status'] == 'Dropout']['Age_at_enrollment'].mean()
    avg_age_graduate = df[df['Status'] == 'Graduate']['Age_at_enrollment'].mean()
    
    # Academic insights
    avg_admission_dropout = df[df['Status'] == 'Dropout']['Admission_grade'].mean()
    avg_admission_graduate = df[df['Status'] == 'Graduate']['Admission_grade'].mean()
    
    # Financial insights
    scholarship_dropout_rate = (df[df['Scholarship_holder'] == 1]['Status'] == 'Dropout').mean() * 100
    no_scholarship_dropout_rate = (df[df['Scholarship_holder'] == 0]['Status'] == 'Dropout').mean() * 100
    
    # Create insights cards
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.info(f"""
        **📊 Statistik Umum**
        - Tingkat dropout keseluruhan: **{dropout_rate:.1f}%**
        - Rata-rata usia dropout: **{avg_age_dropout:.1f} tahun**
        - Rata-rata usia graduate: **{avg_age_graduate:.1f} tahun**
        """)
    
    with col2:
        st.success(f"""
        **🎓 Performa Akademik**
        - Nilai penerimaan rata-rata dropout: **{avg_admission_dropout:.1f}**
        - Nilai penerimaan rata-rata graduate: **{avg_admission_graduate:.1f}**
        - Selisih: **{avg_admission_graduate - avg_admission_dropout:.1f} poin**
        """)
    
    with col3:
        st.warning(f"""
        **💰 Faktor Finansial**
        - Dropout rate dengan beasiswa: **{scholarship_dropout_rate:.1f}%**
        - Dropout rate tanpa beasiswa: **{no_scholarship_dropout_rate:.1f}%**
        - Selisih: **{no_scholarship_dropout_rate - scholarship_dropout_rate:.1f}%**
        """)
    
    # Key recommendations
    st.subheader("💡 Rekomendasi")
    st.markdown("""
    Berdasarkan analisis data, berikut adalah rekomendasi untuk mengurangi tingkat dropout:
    
    1. **🎯 Fokus pada Siswa Berisiko Tinggi**: Siswa dengan nilai penerimaan rendah dan usia yang lebih tua memerlukan perhatian khusus
    
    2. **💸 Program Bantuan Finansial**: Tingkatkan program beasiswa dan bantuan keuangan, karena terbukti efektif mengurangi dropout
    
    3. **📚 Dukungan Akademik**: Implementasikan program mentoring dan tutoring untuk siswa dengan performa semester rendah
    
    4. **🕐 Fleksibilitas Jadwal**: Pertimbangkan program yang lebih fleksibel untuk siswa malam hari yang mungkin bekerja
    
    5. **👨‍👩‍👧‍👦 Keterlibatan Keluarga**: Program yang melibatkan orang tua, terutama untuk siswa dari keluarga dengan latar belakang pendidikan rendah
    """)

# Main Dashboard
def main():
    st.title("📊 Dashboard Analisis Student Dropout")
    st.markdown("*Analisis komprehensif faktor-faktor yang mempengaruhi tingkat dropout siswa*")
    st.markdown("---")
    
    # Load data
    df = load_data()
    if df is None:
        st.stop()
    
    # Sidebar for navigation
    st.sidebar.header("🧭 Navigasi Dashboard")
    analysis_option = st.sidebar.selectbox(
        "Pilih Analisis:",
        [
            "Overview Status Siswa",
            "Analisis Demografis",
            "Performa Akademik", 
            "Faktor Finansial",
            "Pola Kehadiran",
            "Latar Belakang Orang Tua",
            "Insight Kunci"
        ]
    )
    
    # Display selected analysis
    if analysis_option == "Overview Status Siswa":
        create_status_overview(df)
        create_status_distribution_charts(df)
        
    elif analysis_option == "Analisis Demografis":
        create_demographic_analysis(df)
        
    elif analysis_option == "Performa Akademik":
        create_academic_performance_analysis(df)
        
    elif analysis_option == "Faktor Finansial":
        create_financial_analysis(df)
        
    elif analysis_option == "Pola Kehadiran":
        create_attendance_analysis(df)
        
    elif analysis_option == "Latar Belakang Orang Tua":
        create_parents_background_analysis(df)
        
    elif analysis_option == "Insight Kunci":
        create_key_insights(df)
    
    # Footer
    st.markdown("---")
    st.markdown("*Dashboard Student Dropout Analysis - Memberikan insight mendalam untuk pengambilan keputusan*")

if __name__ == "__main__":
    main()
