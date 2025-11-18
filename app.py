# --- Imports: All libraries needed for Phases 1-7 ---
import streamlit as st
import pandas as pd
import numpy as np
from datetime import date
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import warnings
import time

# Suppress warnings for a cleaner output
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)

# --- App Configuration ---
st.set_page_config(
    page_title="Acme Insight Engine",
    page_icon="🚀",  # New Icon
    layout="wide"
)

# --- Caching: Speed up the app ---
# (Phases 1-5 are unchanged, as the core logic is the same)

# --- Phase 1: Project Definition & Data Understanding ---
@st.cache_data
def load_data(file):
    """Loads and cleans the column names of an uploaded file."""
    try:
        df_raw = pd.read_csv(file, sep='\t')
    except Exception as e:
        st.error(f"Error loading data: Could not parse file. {e}")
        return None
    df_raw.columns = df_raw.columns.str.lower().str.strip()
    return df_raw

# --- Phase 2: Automated Preprocessing & Feature Engineering ---
@st.cache_data
def preprocess_data(df):
    """
    Cleans, engineers features, and scales the customer data.
    This version is robust to missing columns.
    """
    df_processed = df.copy()
    
    # 1. Drop useless columns (if they exist)
    df_processed = df_processed.drop(columns=['id', 'z_costcontact', 'z_revenue'], errors='ignore')
    
    # --- Feature Engineering & Imputation (All Optional) ---
    current_year = date.today().year
    
    if 'income' in df_processed.columns:
        median_income = df_processed['income'].median()
        df_processed['income'] = df_processed['income'].fillna(median_income)
    
    if 'year_birth' in df_processed.columns:
        df_processed['age'] = current_year - df_processed['year_birth']

    if 'dt_customer' in df_processed.columns:
        try:
            df_processed['dt_customer'] = pd.to_datetime(df_processed['dt_customer'], format='%d-%m-%Y')
        except ValueError:
            try:
                df_processed['dt_customer'] = pd.to_datetime(df_processed['dt_customer'])
            except Exception:
                df_processed['dt_customer'] = pd.NaT # Failed to parse
        
        if pd.api.types.is_datetime64_any_dtype(df_processed['dt_customer']):
             df_processed['years_customer'] = current_year - df_processed['dt_customer'].dt.year

    if all(col in df_processed.columns for col in ['marital_status', 'kidhome', 'teenhome']):
        df_processed['adults'] = df_processed['marital_status'].apply(lambda x: 2 if x in ['married', 'together'] else 1)
        df_processed['family_size'] = df_processed['adults'] + df_processed['kidhome'] + df_processed['teenhome']
    
    mnt_cols = [col for col in df_processed.columns if 'mnt' in col]
    if mnt_cols: # Only if at least one 'mnt' column exists
        df_processed['total_spent'] = df_processed[mnt_cols].sum(axis=1)
    
    # --- Feature Selection (for clustering) ---
    cluster_features = []
    possible_features = ['income', 'age', 'total_spent', 'family_size', 'years_customer']
    for feat in possible_features:
        if feat in df_processed.columns:
            if pd.api.types.is_numeric_dtype(df_processed[feat]):
                cluster_features.append(feat)
            else:
                st.warning(f"Column '{feat}' was skipped for clustering as it is not numeric.")
            
    if not cluster_features:
        st.error("Error: The uploaded file does not contain any of the required numeric columns (like 'income', 'age', 'total_spent', etc.) to perform an analysis.")
        return None, None, None # Stop the pipeline

    # --- Scaling ---
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(df_processed[cluster_features])
    df_scaled = pd.DataFrame(scaled_features, columns=cluster_features)
    
    return df_processed, df_scaled, cluster_features 

# --- Phase 3: Unsupervised Segmentation (The Discovery) ---
@st.cache_data
def find_clusters(df_scaled, optimal_k=4):
    """Finds optimal K (Elbow) and runs K-Means."""
    kmeans = KMeans(n_clusters=optimal_k, init='k-means++', random_state=42, n_init=10)
    kmeans.fit(df_scaled) 
    return kmeans.labels_

# --- Phase 4: Dynamic Cluster Profiling (The Naming) ---
@st.cache_data
def profile_clusters(df_clean, cluster_labels, cluster_features):
    """Analyzes the clusters and returns a profile DataFrame."""
    df_clean['cluster'] = cluster_labels
    
    profile_features = cluster_features.copy()
    
    other_features = ['mntwines', 'mntmeatproducts', 'numdealspurchases']
    for feat in other_features:
        if feat in df_clean.columns and feat not in profile_features:
            if pd.api.types.is_numeric_dtype(df_clean[feat]):
                profile_features.append(feat)
            
    cluster_profile = df_clean.groupby('cluster')[profile_features].mean().reset_index()
    return df_clean, cluster_profile

# --- Phase 5: Predictive Model Development (The Engine) ---
@st.cache_data
def train_model(df_clean, cluster_features):
    """Trains a classifier to predict the cluster labels."""
    features = cluster_features.copy()
    
    other_features = ['mntwines', 'mntmeatproducts', 'numdealspurchases', 'numwebpurchases',
                      'numcatalogpurchases', 'numstorepurchases']
    for feat in other_features:
        if feat in df_clean.columns and feat not in features:
            if pd.api.types.is_numeric_dtype(df_clean[feat]):
                features.append(feat)

    if not features:
        st.error("Error: No predictive features found in the data.")
        return None, None, None, None

    X = df_clean[features]
    y = df_clean['cluster']
    X_train, X_test, y_train, y_test = train_test_split(X, y, 
                                                        test_size=0.2, 
                                                        random_state=42, 
                                                        stratify=y)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, output_dict=True)
    
    return rf_model, accuracy, report, features


# =========================================================================
# --- Phase 7: Streamlit Deployment (The NEW GUI) ---
# =========================================================================

# --- Sidebar: Control Panel ---
# Using a placeholder image - replace 'logo.png' with a real file path or URL for extra effect
# st.sidebar.image("logo.png", width=100) 
st.sidebar.title("🚀 Acme Insight Engine")
st.sidebar.markdown("Powering data-driven decisions.")
st.sidebar.divider()

st.sidebar.header("Data Input")
uploaded_file = st.sidebar.file_uploader("Upload your proprietary customer data", type=["csv", "tsv"])

if "pipeline_complete" not in st.session_state:
    st.session_state.pipeline_complete = False

if uploaded_file:
    # 2. Analyze (Button to run the pipeline)
    if st.sidebar.button("💡 Generate Insights", type="primary", use_container_width=True):
        with st.spinner("Processing data... The Engine is learning..."):
            
            st.session_state.df_raw = load_data(uploaded_file)
            if st.session_state.df_raw is None:
                st.stop() # Stop if loading failed
            
            df_clean, df_scaled, cluster_features = preprocess_data(st.session_state.df_raw)
            
            if df_clean is None:
                st.session_state.pipeline_complete = False
                st.stop() # Stop execution
            
            st.session_state.df_clean = df_clean
            st.session_state.cluster_features = cluster_features
            
            cluster_labels = find_clusters(df_scaled, optimal_k=4)
            
            df_with_labels, profile = profile_clusters(st.session_state.df_clean, cluster_labels, st.session_state.cluster_features)
            st.session_state.df_with_labels = df_with_labels
            st.session_state.profile = profile
            
            model, acc, report, model_features = train_model(st.session_state.df_with_labels, st.session_state.cluster_features)
            st.session_state.model = model
            st.session_state.accuracy = acc
            st.session_state.report = report
            st.session_state.model_features = model_features
            
            st.session_state.cluster_names = {
                0: "Segment A",
                1: "Segment B",
                2: "Segment C",
                3: "Segment D"
            }
            
            st.session_state.pipeline_complete = True
            time.sleep(1) 
        
        st.sidebar.success("Insight Generation Complete!")
        st.balloons()
else:
    st.sidebar.info("Upload a data file to activate the engine.")


# --- Main Page: Title & Tabs ---
st.title("💡 Strategic Insight Dashboard")
st.markdown("Unlock your market's potential. Identify and target key customer segments.")

# Define the tabs with new names
tab_home, tab_segments, tab_forecast, tab_performance = st.tabs([
    "🏠 Dashboard Home", 
    "🎯 Strategic Segments", 
    "🔍 New Customer Forecast", 
    "🧠 Engine Performance"
])


# --- Tab 1: Dashboard Home ---
with tab_home:
    st.header("Welcome to the Acme Insight Engine")
    
    if st.session_state.pipeline_complete:
        st.success("Your data has been processed and your custom insight model is ready.")
        st.subheader("Data Overview (Processed Sample with Segment Tags)")
        st.markdown("Here is a sample of your processed data, now tagged with the new strategic segments.")
        st.dataframe(st.session_state.df_with_labels.head(100))
    else:
        st.info("Upload your customer data in the sidebar and click *'Generate Insights'* to begin.")
        
        st.subheader("Our 4-Step Process")
        st.markdown("""
        This proprietary tool provides a complete, end-to-end solution for market segmentation.
        
        1.  *Ingest & Clean:* We process and refine your raw customer data, handling missing values and engineering new features for maximum signal.
        2.  *Segment:* Our unsupervised AI discovers the 4 most statistically significant, hidden groupings within your customer base.
        3.  *Learn:* A high-performance predictive model is custom-built and trained on your data to learn the unique patterns of these new segments.
        4.  *Forecast:* Use the live 'New Customer Forecast' tool to instantly classify new leads and drive your marketing strategy.
        """)
        st.markdown("Don't have a file? You can download the test data [from Kaggle](https://www.kaggle.com/datasets/imakash3011/customer-personality-analysis).")

# --- Tab 2: Strategic Segments ---
with tab_segments:
    if not st.session_state.pipeline_complete:
        st.warning("Please run the 'Generate Insights' process from the sidebar to view your segments.")
    else:
        st.header("🎯 Strategic Segment Profiles")
        st.markdown("These are the core personas identified from your data, based on their average behaviors and attributes.")
        st.dataframe(st.session_state.profile.style.format("{:.2f}"))
        st.markdown(f"""
        * *Core Segmentation Drivers:* {(', '.join(st.session_state.cluster_features))}
        """)
        
        st.divider()
        
        st.header("✏ Create Segment Personas")
        st.markdown("Define a persona name for each segment. This will be used in the Forecast tool for easy reference.")
        
        col1, col2 = st.columns(2)
        with col1:
            st.session_state.cluster_names[0] = st.text_input("Name for Segment 0", value=st.session_state.cluster_names[0])
            st.session_state.cluster_names[1] = st.text_input("Name for Segment 1", value=st.session_state.cluster_names[1])
        with col2:
            st.session_state.cluster_names[2] = st.text_input("Name for Segment 2", value=st.session_state.cluster_names[2])
            st.session_state.cluster_names[3] = st.text_input("Name for Segment 3", value=st.session_state.cluster_names[3])
            
        st.success(f"Persona names updated: {st.session_state.cluster_names[0]}, {st.session_state.cluster_names[1]}, {st.session_state.cluster_names[2]}, {st.session_state.cluster_names[3]}")


# --- Tab 3: New Customer Forecast ---
with tab_forecast:
    if not st.session_state.pipeline_complete:
        st.warning("Please run the 'Generate Insights' process from the sidebar to activate the Forecast tool.")
    else:
        st.header("🔍 New Customer Forecast")
        st.markdown("Run a new lead or customer profile against the predictive model to instantly determine their strategic segment.")
        
        expected_features = ['age', 'income', 'total_spent', 'family_size', 'years_customer',
                             'mntwines', 'mntmeatproducts', 'numdealspurchases', 'numwebpurchases',
                             'numcatalogpurchases', 'numstorepurchases']
        
        model_feature_set = set(st.session_state.model_features)
        expected_feature_set = set(expected_features)

        if expected_feature_set.issubset(model_feature_set):
            with st.form("prediction_form"):
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.markdown("##### 👤 Customer Vitals")
                    age = st.number_input("Age", min_value=18, max_value=120, value=40)
                    income = st.number_input("Annual Income", min_value=0, max_value=700000, value=50000)
                    family_size = st.slider("Family Size", min_value=1, max_value=10, value=2)
                    years_customer = st.slider("Loyalty (Years)", min_value=0, max_value=20, value=3)
                
                with col2:
                    st.markdown("##### 💳 Transaction Profile (Spent)")
                    mnt_wines = st.number_input("Wine", min_value=0, value=300)
                    mnt_meat = st.number_input("Meat", min_value=0, value=150)
                    mnt_fruits = st.number_input("Fruits", min_value=0, value=20)
                    mnt_fish = st.number_input("Fish", min_value=0, value=20)
                    mnt_sweets = st.number_input("Sweets", min_value=0, value=20)
                    mnt_gold = st.number_input("Gold", min_value=0, value=20)
                
                with col3:
                    st.markdown("##### 🛍 Channel Activity (Purchases)")
                    num_deals = st.number_input("Deals", min_value=0, value=2)
                    num_web = st.number_input("Web", min_value=0, value=4)
                    num_catalog = st.number_input("Catalog", min_value=0, value=2)
                    num_store = st.number_input("In-Store", min_value=0, value=5)
                    
                submitted = st.form_submit_button("Run Forecast", type="primary")

                if submitted:
                    cluster_names_map = st.session_state.cluster_names
                    
                    total_spent = mnt_wines + mnt_meat + mnt_fruits + mnt_fish + mnt_sweets + mnt_gold
                    
                    input_data_dict = {
                        'age': age, 'income': income, 'total_spent': total_spent, 
                        'family_size': family_size, 'years_customer': years_customer,
                        'mntwines': mnt_wines, 
                        'mntmeatproducts': mnt_meat,
                        'numdealspurchases': num_deals, 
                        'numwebpurchases': num_web,
                        'numcatalogpurchases': num_catalog, 
                        'numstorepurchases': num_store,
                        'mntfruits': mnt_fruits,
                        'mntfishproducts': mnt_fish,
                        'mntsweetproducts': mnt_sweets,
                        'mntgoldprods': mnt_gold 
                    }
                    
                    input_data = []
                    for feat in st.session_state.model_features:
                        if feat in input_data_dict:
                            input_data.append(input_data_dict[feat])
                        else:
                            input_data.append(0) 

                    input_df = pd.DataFrame([input_data], columns=st.session_state.model_features)
                    
                    prediction = st.session_state.model.predict(input_df)
                    prediction_proba = st.session_state.model.predict_proba(input_df)
                    
                    cluster_num = prediction[0]
                    cluster_name = cluster_names_map.get(cluster_num, f"Segment {cluster_num}")

                    st.subheader(f"Forecast: This lead matches the *{cluster_name}* persona.")
                    
                    # Display as a gauge
                    prob_percent = prediction_proba.max()
                    st.progress(prob_percent, text=f"Confidence: {prob_percent*100:.2f}%")
                    
                    st.divider()
                    st.subheader("Profile Comparison: New Lead vs. Persona Average")
                    col1_res, col2_res = st.columns(2)
                    with col1_res:
                        st.write("*New Lead's Profile:*")
                        st.dataframe(input_df.style.format("{:.2f}"))
                    with col2_res:
                        st.write(f"'{cluster_name}' Persona Average:")
                        st.dataframe(st.session_state.profile[st.session_state.profile.cluster == cluster_num].style.format("{:.2f}"))
        
        else:
            st.error("Forecast Tool Disabled: Incompatible Data")
            st.markdown(f"""
            The live forecast tool requires a standard set of data columns to function. The file you uploaded is missing some of the required inputs.
            
            *The model was trained on:*
            {st.session_state.model_features}
            
            *The tool requires:*
            {expected_features}
            
            Please upload a file with the standard columns to activate this feature.
            """)


# --- Tab 4: Engine Performance ---
with tab_performance:
    if not st.session_state.pipeline_complete:
        st.warning("Please run the 'Generate Insights' process from the sidebar to view performance metrics.")
    else:
        st.header("🧠 Engine Performance & Telemetry")
        st.markdown("This dashboard provides transparency into the performance of the custom-built predictive model.")
        
        st.metric("Overall Predictive Accuracy", f"{st.session_state.accuracy * 100:.2f}%")
        st.caption("This score represents how accurately the model can re-identify the segments it was trained on.")
        
        st.divider()
        st.subheader("Per-Segment Accuracy (Classification Report)")
        st.markdown("This shows the model's 'precision' and 'recall' for each individual segment. A high score (near 1.0) is ideal.")
        report_df = pd.DataFrame(st.session_state.report).transpose()
        st.dataframe(report_df.style.format("{:.3f}"))
        
        st.divider()
        st.subheader("Key Predictive Features (Model Inputs)")
        st.markdown("The model was trained to make its predictions using the following data points:")
        st.json(st.session_state.model_features)