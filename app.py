import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer

# Page configuration
st.set_page_config(page_title="Student Performance PCA", layout="wide")

st.title("📊 Kathmandu University Student Performance PCA Analysis")

# File uploader
uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data
    df = pd.read_csv(uploaded_file)
    
    st.subheader("📋 Data Preview")
    st.write(f"Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    st.dataframe(df.head())
    
    # ============================================
    # PREPROCESS DATA FOR PCA (FIXES THE ERROR)
    # ============================================
    def preprocess_for_pca(data):
        """Convert all data to numeric for PCA"""
        data_processed = data.copy()
        
        # Remove ID columns
        if 'student_id' in data_processed.columns:
            data_processed = data_processed.drop('student_id', axis=1)
        
        # Get numeric columns
        numeric_cols = data_processed.select_dtypes(include=[np.number]).columns.tolist()
        
        # Encode categorical columns
        categorical_cols = data_processed.select_dtypes(include=['object']).columns.tolist()
        
        if categorical_cols:
            for col in categorical_cols:
                le = LabelEncoder()
                data_processed[col] = le.fit_transform(data_processed[col].astype(str))
                numeric_cols.append(col)
        
        # Use only numeric columns
        X = data_processed[numeric_cols]
        
        # Handle missing values
        if X.isnull().any().any():
            imputer = SimpleImputer(strategy='mean')
            X = pd.DataFrame(imputer.fit_transform(X), columns=X.columns)
        
        return X, numeric_cols
    
    # Process data
    with st.spinner("🔄 Processing data for PCA..."):
        X, feature_names = preprocess_for_pca(df)
    
    st.subheader("🔢 Processed Data (All Numeric)")
    st.write(f"Features used: {len(feature_names)}")
    st.dataframe(X.head())
    
    # ============================================
    # RUN PCA
    # ============================================
    with st.spinner("📊 Running PCA analysis..."):
        # Standardize
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA
        pca = PCA()
        X_pca = pca.fit_transform(X_scaled)
    
    # ============================================
    # RESULTS
    # ============================================
    explained_variance = pca.explained_variance_ratio_
    cumulative_variance = np.cumsum(explained_variance)
    
    # Components for 80% variance
    n_components_80 = np.argmax(cumulative_variance >= 0.80) + 1
    
    # Metrics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Features", X.shape[1])
    with col2:
        st.metric("PCs for 80% Variance", n_components_80)
    with col3:
        st.metric("Total Variance Explained", f"{cumulative_variance[-1]:.1%}")
    
    # ============================================
    # SCREE PLOT
    # ============================================
    st.subheader("📈 Explained Variance Analysis")
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Scree plot
    axes[0].bar(range(1, min(len(explained_variance), 20)+1), 
                explained_variance[:20], alpha=0.7, color='steelblue')
    axes[0].set_xlabel('Principal Component')
    axes[0].set_ylabel('Explained Variance Ratio')
    axes[0].set_title('Scree Plot (First 20 Components)')
    axes[0].grid(True, alpha=0.3)
    
    # Cumulative variance
    axes[1].plot(range(1, len(cumulative_variance)+1), cumulative_variance, 'bo-', 
                 linewidth=2, markersize=4)
    axes[1].axhline(y=0.80, color='r', linestyle='--', label='80% threshold', linewidth=2)
    axes[1].axvline(x=n_components_80, color='g', linestyle='--', label=f'PC{n_components_80}', linewidth=2)
    axes[1].set_xlabel('Number of Components')
    axes[1].set_ylabel('Cumulative Explained Variance')
    axes[1].set_title('Cumulative Explained Variance')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig)
    
    # ============================================
    # PCA BI-PLOT
    # ============================================
    st.subheader("🎯 PCA Visualization - First 2 Components")
    
    fig, ax = plt.subplots(figsize=(10, 7))
    scatter = ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6, c='blue', s=30)
    ax.set_xlabel(f'PC1 ({explained_variance[0]*100:.1f}%)')
    ax.set_ylabel(f'PC2 ({explained_variance[1]*100:.1f}%)')
    ax.set_title('PCA Biplot - All Students')
    ax.grid(True, alpha=0.3)
    
    st.pyplot(fig)
    
    # ============================================
    # FEATURE LOADINGS
    # ============================================
    st.subheader("📊 Top Features Contributing to PC1")
    
    loadings = pd.DataFrame(
        pca.components_.T,
        columns=[f'PC{i+1}' for i in range(len(pca.components_))],
        index=feature_names
    )
    
    # Top 10 features for PC1
    top_features = loadings['PC1'].abs().sort_values(ascending=False).head(10)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    top_features.sort_values().plot(kind='barh', ax=ax, color='coral')
    ax.set_title('Top 10 Features - PC1 (Absolute Loading)')
    ax.set_xlabel('Absolute Loading Value')
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    # ============================================
    # FEATURE LOADINGS TABLE
    # ============================================
    st.subheader("📋 Feature Loadings Table (First 3 PCs)")
    
    # Show top 20 features
    st.dataframe(
        loadings[['PC1', 'PC2', 'PC3']].head(20).style.background_gradient(cmap='RdBu_r', axis=0)
    )
    
    # ============================================
    # DOWNLOAD RESULTS
    # ============================================
    st.subheader("💾 Download Results")
    
    # Create PCA results dataframe
    pca_df = pd.DataFrame(
        X_pca[:, :min(10, X_pca.shape[1])],
        columns=[f'PC{i+1}' for i in range(min(10, X_pca.shape[1]))]
    )
    
    # Add student_id if available
    if 'student_id' in df.columns:
        pca_df['student_id'] = df['student_id'].values
    
    csv = pca_df.to_csv(index=False)
    st.download_button(
        label="📥 Download PCA Results (CSV)",
        data=csv,
        file_name="pca_results.csv",
        mime="text/csv"
    )
    
    st.success("✅ PCA Analysis Completed Successfully!")
