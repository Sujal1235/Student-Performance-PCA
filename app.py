import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns

st.set_page_config(page_title="PCA Analyzer", layout="wide")

st.title("📊 Student Performance PCA Analyzer")
st.markdown("---")

uploaded_file = st.file_uploader("Upload your CSV file", type=['csv'])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)
    
    st.success("✅ Dataset loaded!")
    
    col1, col2, col3 = st.columns(3)
    col1.metric("Rows", df.shape[0])
    col2.metric("Columns", df.shape[1])
    col3.metric("Missing Values", df.isnull().sum().sum())
    
    st.subheader("📄 Data Preview")
    st.dataframe(df.head(10))
    
    if st.button("Run PCA Analysis"):
        with st.spinner("Processing..."):
            df_encoded = df.copy()
            for col in df_encoded.select_dtypes(include=['object']).columns:
                df_encoded[col] = LabelEncoder().fit_transform(df_encoded[col].astype(str))
            
            scaler = StandardScaler()
            X = scaler.fit_transform(df_encoded)
            
            n_components = min(10, X.shape[1])
            pca = PCA(n_components=n_components)
            X_pca = pca.fit_transform(X)
            
            st.success("✅ PCA Complete!")
            
            col1, col2, col3 = st.columns(3)
            col1.metric("Original Features", df_encoded.shape[1])
            col2.metric("PCA Components", n_components)
            col3.metric("Variance Retained", f"{pca.explained_variance_ratio_.sum():.1%}")
            
            st.subheader("📊 Explained Variance")
            var_df = pd.DataFrame({
                'Component': [f'PC{i+1}' for i in range(n_components)],
                'Variance': pca.explained_variance_ratio_,
                'Cumulative': np.cumsum(pca.explained_variance_ratio_)
            })
            var_df['Variance'] = var_df['Variance'].apply(lambda x: f"{x:.2%}")
            var_df['Cumulative'] = var_df['Cumulative'].apply(lambda x: f"{x:.2%}")
            st.dataframe(var_df)
            
            fig, ax = plt.subplots(figsize=(10, 5))
            ax.bar(range(1, n_components+1), pca.explained_variance_ratio_, alpha=0.7)
            ax.plot(range(1, n_components+1), np.cumsum(pca.explained_variance_ratio_), 'ro-')
            ax.axhline(y=0.9, color='g', linestyle='--', label='90% threshold')
            ax.set_xlabel('Principal Component')
            ax.set_ylabel('Explained Variance Ratio')
            ax.set_title('Scree Plot')
            ax.legend()
            ax.grid(True, alpha=0.3)
            st.pyplot(fig)
            
            if n_components >= 2:
                fig, ax = plt.subplots(figsize=(10, 8))
                ax.scatter(X_pca[:, 0], X_pca[:, 1], alpha=0.6)
                ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%})')
                ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%})')
                ax.set_title('PCA 2D Projection')
                ax.grid(True, alpha=0.3)
                st.pyplot(fig)
            
            pca_df = pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(n_components)])
            csv = pca_df.to_csv(index=False)
            st.download_button("Download PCA Results (CSV)", data=csv, file_name="pca_results.csv", mime="text/csv")
else:
    st.info("👆 Upload a CSV file to start")