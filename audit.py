import streamlit as st
import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split
import seaborn as sns
import matplotlib.pyplot as plt
import re

st.set_page_config(page_title="STR Performance Dashboard", layout="wide")
st.title("📊 STR Performance Analyzer")

# --- Upload File ---
uploaded_file = st.file_uploader("Upload your STR dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Read the file based on its type
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == 'csv':
        df = pd.read_csv(uploaded_file)
    else:  # xlsx or xls
        df = pd.read_excel(uploaded_file)

    # --- Clean & Convert Numeric Columns ---
    def clean_numeric(val):
        if isinstance(val, str):
            val = re.sub(r'[^0-9.-]', '', val)
            return float(val) if val != '' else np.nan
        return val

    numeric_cols = [
        'Total Revenue', 'Total Revenue STLY', 'Total Revenue LY',
        'Rental RevPAR', 'Rental RevPAR STLY', 'Rental RevPAR LY',
        'Occupancy %', 'Occupancy % STLY', 'Occupancy % LY',
        'Rental Revenue', 'Rental Revenue LY', 'Market Penetration Index %',
        'Market Occupancy %', 'Average Booking Window', 'Average Market Booking Window',
        'Average LOS', 'Average Market LOS', 'Total ADR', 'Market ADR', 'ADR Index',
        'Total RevPAR', 'Total RevPAR STLY', 'Total RevPAR LY',
        'Market RevPAR', 'Market RevPAR STLY', 'Market RevPAR LY', 'RevPAR Index'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)

    # --- Feature Engineering ---
    # Extract year and month, handle missing values properly
    year_extract = df['Year & Month'].str.extract(r'(\d{4})')[0]
    month_extract = df['Year & Month'].str.extract(r'-(\d{2})')[0]
    
    # Convert to integers safely
    df['Year'] = pd.to_numeric(year_extract, errors='coerce')
    df['Month'] = pd.to_numeric(month_extract, errors='coerce')
    
    # Remove rows with invalid dates
    df = df.dropna(subset=['Year', 'Month'])
    
    # Convert to integers after removing NaN
    df['Year'] = df['Year'].astype(int)
    df['Month'] = df['Month'].astype(int)

    # Clean booking window values with outlier detection
    def clean_booking_window_values(series, max_reasonable_days=90):
        """Clean booking window values by removing extreme outliers"""
        # Remove negative values
        series = series[series > 0]
        
        # Remove extreme outliers (more than max_reasonable_days)
        series = series[series <= max_reasonable_days]
        
        # If series is empty after cleaning, return NaN
        if len(series) == 0:
            return np.nan
            
        return series.mean()
    
    # Clean booking window data
    df['Average Booking Window'] = df['Average Booking Window'].apply(
        lambda x: x if 0 < x <= 90 else np.nan
    )
    df['Average Market Booking Window'] = df['Average Market Booking Window'].apply(
        lambda x: x if 0 < x <= 90 else np.nan
    )
    
    # Calculate monthly averages using only reasonable values
    monthly_positive_bw = df[df['Average Booking Window'].notna()].groupby(['Year', 'Month'])['Average Booking Window'].mean()
    monthly_positive_market_bw = df[df['Average Market Booking Window'].notna()].groupby(['Year', 'Month'])['Average Market Booking Window'].mean()
    
    # Replace NaN values with monthly averages
    for idx in df.index:
        year = df.loc[idx, 'Year']
        month = df.loc[idx, 'Month']
        
        if pd.isna(df.loc[idx, 'Average Booking Window']):
            try:
                df.loc[idx, 'Average Booking Window'] = monthly_positive_bw.loc[year, month]
            except KeyError:
                # If no positive values for that month-year, use overall positive mean
                overall_mean = df[df['Average Booking Window'].notna()]['Average Booking Window'].mean()
                df.loc[idx, 'Average Booking Window'] = overall_mean if not pd.isna(overall_mean) else 14  # Default to 14 days
        
        if pd.isna(df.loc[idx, 'Average Market Booking Window']):
            try:
                df.loc[idx, 'Average Market Booking Window'] = monthly_positive_market_bw.loc[year, month]
            except KeyError:
                # If no positive values for that month-year, use overall positive mean
                overall_mean = df[df['Average Market Booking Window'].notna()]['Average Market Booking Window'].mean()
                df.loc[idx, 'Average Market Booking Window'] = overall_mean if not pd.isna(overall_mean) else 21  # Default to 21 days

    df['ADR Index'] = df['Total ADR'] / df['Market ADR']
    df['Occupancy Index'] = df['Occupancy %'] / df['Market Occupancy %']
    df['LOS Index'] = df['Average LOS'] / df['Average Market LOS']
    df['Booking Window Index'] = df['Average Booking Window'] / df['Average Market Booking Window']

    # --- Fill NA with median or safe default ---
    df.fillna(df.median(numeric_only=True), inplace=True)

    # --- Train Model and Calculate Performance Status ---
    features = ['ADR Index', 'Occupancy Index', 'LOS Index', 'Booking Window Index',
                'Market ADR', 'Market Occupancy %', 'Market RevPAR', 'Month', 'Year']
    target = 'Total RevPAR'

    X = df[features]
    y = df[target]

    # --- Data Cleaning for XGBoost ---
    try:
        # Remove infinite values and replace with NaN
        X = X.replace([np.inf, -np.inf], np.nan)
        y = y.replace([np.inf, -np.inf], np.nan)
        
        # Fill NaN values with median for features and mean for target
        X_filled = X.fillna(X.median())
        y_filled = y.fillna(y.median())
        
        # If we have enough data, proceed with model training
        if len(X_filled) > 10 and len(y_filled) > 10:
            X_train, X_test, y_train, y_test = train_test_split(X_filled, y_filled, test_size=0.2, random_state=42)

            model = xgb.XGBRegressor(n_estimators=100, max_depth=4, learning_rate=0.1)
            model.fit(X_train, y_train)

            # Predict for all data (using filled values)
            df['Predicted RevPAR'] = model.predict(X_filled)
            df['RevPAR Residual'] = df['Total RevPAR'] - df['Predicted RevPAR']
        else:
            # If not enough data, use simple predictions
            df['Predicted RevPAR'] = df['Total RevPAR']
            df['RevPAR Residual'] = 0
    except Exception as e:
        st.warning(f"Model training failed: {str(e)}. Using simple predictions.")
        df['Predicted RevPAR'] = df['Total RevPAR']
        df['RevPAR Residual'] = 0
    
    # --- Enhanced Performance Classification ---
    # Calculate RevPAR Index and other metrics
    df['RevPAR Index'] = df['Total RevPAR'] / df['Market RevPAR']
    df['Market Performance Gap'] = (df['Market RevPAR'] - df['Total RevPAR']) / df['Market RevPAR'] * 100
    df['Prediction Gap'] = (df['Predicted RevPAR'] - df['Total RevPAR']) / df['Predicted RevPAR'] * 100
    
    # Calculate percentiles for different metrics
    revpar_25th = df['RevPAR Index'].quantile(0.25)
    revpar_75th = df['RevPAR Index'].quantile(0.75)
    residual_25th = df['RevPAR Residual'].quantile(0.25)
    residual_75th = df['RevPAR Residual'].quantile(0.75)
    
    def enhanced_classify(row):
        # Check market performance first
        market_performance = row['RevPAR Index'] < 0.95  # Underperforming if below 95% of market
        strong_market = row['RevPAR Index'] > 1.05  # Strong if above 105% of market
        
        # Check prediction performance
        prediction_performance = row['RevPAR Residual'] < residual_25th  # Bottom 25% of residuals
        strong_prediction = row['RevPAR Residual'] > residual_75th  # Top 25% of residuals
        
        # Classification logic:
        # 1. If significantly above market (>105%), cannot be underperforming
        # 2. If below market, check prediction to determine severity
        # 3. If within market range (0.95-1.05), only consider prediction for potential underperformance
        
        if strong_market:
            if strong_prediction:
                return 'Overperforming'
            else:
                return 'Normal'
        elif market_performance:
            if prediction_performance:
                return 'Significantly Underperforming'
            else:
                return 'Underperforming'
        else:  # Within market range (0.95-1.05)
            if prediction_performance and row['RevPAR Index'] < 1.0:
                return 'Underperforming'
            elif strong_prediction:
                return 'Overperforming'
            else:
                return 'Normal'

    # First, apply basic classification
    df['Performance Status'] = df.apply(enhanced_classify, axis=1)
    
    # Calculate revenue potential with safety checks
    def calculate_revenue_potential(row):
        if (row['Performance Status'] not in ['Significantly Underperforming', 'Underperforming'] or
            row['Total RevPAR'] == 0 or row['Total Revenue'] == 0):
            return 0
            
        try:
            target_revpar = row['Market RevPAR'] * 0.95
            if target_revpar <= row['Total RevPAR']:
                return 0
                
            # Calculate potential based on the gap to 95% of market RevPAR
            potential = (target_revpar - row['Total RevPAR']) * row['Total Revenue'] / row['Total RevPAR']
            return max(0, potential)  # Ensure we don't return negative potential
        except:
            return 0  # Return 0 if any calculation errors occur
    
    df['Monthly Revenue Potential'] = df.apply(calculate_revenue_potential, axis=1)
    
    # Now refine the classification based on revenue potential
    def refine_classification(row):
        if row['Performance Status'] in ['Significantly Underperforming', 'Underperforming']:
            if row['Total Revenue'] == 0:
                return 'Normal'  # No revenue = can't meaningfully classify as underperforming
            revenue_opportunity_pct = (row['Monthly Revenue Potential'] / row['Total Revenue']) * 100
            if revenue_opportunity_pct <= 5:  # Less than 5% improvement potential
                return 'Normal'
        return row['Performance Status']
    
    df['Performance Status'] = df.apply(refine_classification, axis=1)
    
    # Calculate total revenue potential for the period
    property_metrics = df.groupby(['Listing Name', 'City Name', 'Bedroom count category']).agg({
        'Total Revenue': 'sum',
        'Monthly Revenue Potential': 'sum',
        'Total RevPAR': 'mean',
        'Market RevPAR': 'mean',
        'RevPAR Residual': 'mean',
        'Occupancy %': 'mean',
        'Market Occupancy %': 'mean',
        'Total ADR': 'mean',
        'Market ADR': 'mean'
    }).reset_index()
    
    # Rename Monthly Revenue Potential to Incremental Revenue Potential for consistency
    property_metrics = property_metrics.rename(columns={'Monthly Revenue Potential': 'Incremental Revenue Potential'})
    
    # Calculate additional metrics
    property_metrics['RevPAR Gap %'] = ((property_metrics['Market RevPAR'] - property_metrics['Total RevPAR']) / property_metrics['Market RevPAR'] * 100).round(1)
    property_metrics['Occupancy Gap %'] = (property_metrics['Market Occupancy %'] - property_metrics['Occupancy %']).round(1)
    property_metrics['ADR Gap %'] = ((property_metrics['Market ADR'] - property_metrics['Total ADR']) / property_metrics['Market ADR'] * 100).round(1)
    
    # Add revenue opportunity percentage
    property_metrics['Revenue Opportunity %'] = (property_metrics['Incremental Revenue Potential'] / property_metrics['Total Revenue'] * 100).round(2)
    
    # Determine final performance status based on revenue opportunity
    def determine_final_status(row):
        if row['Revenue Opportunity %'] <= 5:  # If opportunity is less than 5%, always Normal
            return 'Normal'
        
        # Get the most severe status from monthly data
        monthly_status = df[df['Listing Name'] == row['Listing Name']]['Performance Status'].unique()
        if 'Significantly Underperforming' in monthly_status:
            return 'Significantly Underperforming'
        elif 'Underperforming' in monthly_status:
            return 'Underperforming'
        else:
            return 'Normal'
    
    property_metrics['Performance Status'] = property_metrics.apply(determine_final_status, axis=1)
    
    # Sort by revenue potential
    property_metrics = property_metrics.sort_values('Incremental Revenue Potential', ascending=False)
    
    # Debug output
    # debug_property = "3B-AddressOpera2-1102 -- Manzil - 3BR | Downtown | Connected to Dubai Mall"
    # st.write("### Sample Property Verification")
    
    # # Display property metrics
    # debug_metrics = property_metrics[property_metrics['Listing Name'] == debug_property]
    # st.write("Property Metrics:")
    # st.write(f"Revenue: ${debug_metrics['Total Revenue'].iloc[0]:,.2f}")
    # st.write(f"Revenue Opportunity: ${debug_metrics['Incremental Revenue Potential'].iloc[0]:,.2f}")
    # st.write(f"Opportunity %: {debug_metrics['Revenue Opportunity %'].iloc[0]:.2f}%")
    # st.write(f"Final Status: {debug_metrics['Performance Status'].iloc[0]}")
    
    # # Display monthly performance
    # debug_monthly = df[df['Listing Name'] == debug_property][
    #     ['Year & Month', 'RevPAR Index', 'Total RevPAR', 'Market RevPAR', 
    #      'Total Revenue', 'Monthly Revenue Potential', 'Performance Status']
    # ].sort_values('Year & Month')
    # st.write("Monthly performance:")
    # st.dataframe(debug_monthly)
    
    # # Display summary metrics
    # debug_summary = property_metrics[property_metrics['Listing Name'] == debug_property]
    # st.write("### Summary Metrics")
    # col1, col2, col3 = st.columns(3)
    # with col1:
    #     st.metric("Total Revenue", f"${debug_summary['Total Revenue'].iloc[0]:,.0f}")
    # with col2:
    #     st.metric("Total Revenue Opportunity", f"${debug_summary['Incremental Revenue Potential'].iloc[0]:,.0f}")
    # with col3:
    #     st.metric("Opportunity as % of Revenue", f"{debug_summary['Revenue Opportunity %'].iloc[0]:.2f}%")
    
    # Add performance distribution analysis
    st.write("### Overall Performance Distribution")
    
    # Monthly performance distribution
    monthly_perf = df.groupby(['Year & Month', 'Performance Status']).size().unstack(fill_value=0)
    monthly_perf_pct = monthly_perf.div(monthly_perf.sum(axis=1), axis=0) * 100
    
    # Create visualization
    plt.figure(figsize=(12, 6))
    
    # Prepare data for stacked bar chart
    monthly_perf_pct_reset = monthly_perf_pct.reset_index()
    
    # Create stacked bar chart using matplotlib
    x_pos = range(len(monthly_perf_pct_reset))
    width = 0.8
    
    # Define colors
    colors = {'Significantly Underperforming': 'red', 
              'Underperforming': 'orange', 
              'Normal': 'gray', 
              'Overperforming': 'green'}
    
    # Plot each status as a stacked bar
    bottom = np.zeros(len(monthly_perf_pct_reset))
    
    for status in ['Significantly Underperforming', 'Underperforming', 'Normal', 'Overperforming']:
        if status in monthly_perf_pct_reset.columns:
            plt.bar(x_pos, monthly_perf_pct_reset[status], width, 
                   bottom=bottom, label=status, color=colors[status], alpha=0.8)
            bottom += monthly_perf_pct_reset[status]
    
    plt.title("Monthly Distribution of Property Performance")
    plt.xlabel("Period")
    plt.ylabel("Percentage of Properties")
    plt.xticks(x_pos, monthly_perf_pct_reset['Year & Month'], rotation=45)
    plt.legend(title='Performance Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.ylim(0, 100)
    plt.tight_layout()
    
    st.pyplot(plt.gcf())
    plt.clf()
    
    # Add summary statistics
    st.write("### Performance Classification Summary")
    
    # Overall statistics
    total_properties = len(df['Listing Name'].unique())
    total_records = len(df)
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Properties", total_properties)
        
        # Performance breakdown
        status_counts = df.groupby('Performance Status').size()
        status_pct = (status_counts / len(df) * 100).round(1)
        
        st.write("Performance Distribution (all months):")
        for status in ['Significantly Underperforming', 'Underperforming', 'Normal', 'Overperforming']:
            if status in status_counts:
                st.write(f"{status}: {status_counts[status]} records ({status_pct[status]}%)")
    
    with col2:
        st.metric("Total Monthly Records", total_records)
        
        # Property consistency analysis
        property_status_counts = df.groupby('Listing Name')['Performance Status'].nunique()
        consistent_properties = (property_status_counts == 1).sum()
        mixed_performance = (property_status_counts > 1).sum()
        
        st.write("Property Performance Consistency:")
        st.write(f"Consistent Performance: {consistent_properties} properties")
        st.write(f"Mixed Performance: {mixed_performance} properties")
    
    # Create Date column for filtering
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2) + '-01')
    
    # --- Monthly Performance Analysis ---
    st.subheader("📊 Performance Distribution")
    
    # Add year selector
    years = sorted(df['Year'].unique())
    st.write(f"Available years in data: {years}")
    st.write(f"Total records: {len(df)}")
    selected_year = st.selectbox("Select Year for Analysis", years)
    
    # Filter data for selected year
    year_df = df[df['Year'] == selected_year]
    st.write(f"Records for {selected_year}: {len(year_df)}")
    
    # Prepare monthly performance data for selected year
    monthly_perf = year_df.groupby(['Month', 'Performance Status']).size().reset_index(name='count')
    monthly_total = monthly_perf.groupby(['Month'])['count'].sum().reset_index(name='total')
    monthly_perf = monthly_perf.merge(monthly_total, on=['Month'])
    monthly_perf['percentage'] = (monthly_perf['count'] / monthly_perf['total'] * 100).round(1)
    

    # Year-over-Year Analysis
    st.subheader("📈 Year-over-Year Analysis")
    
    # Yearly Performance Summary
    yearly_perf = df.groupby(['Year', 'Performance Status']).size().reset_index(name='count')
    yearly_total = yearly_perf.groupby(['Year'])['count'].sum().reset_index(name='total')
    yearly_perf = yearly_perf.merge(yearly_total, on=['Year'])
    yearly_perf['percentage'] = (yearly_perf['count'] / yearly_perf['total'] * 100).round(1)
    
    # Create year-over-year visualization
    plt.figure(figsize=(10, 6))
    
    sns.barplot(data=yearly_perf, 
                x='Year', 
                y='percentage', 
                hue='Performance Status',
                palette={'Significantly Underperforming': 'red', 
                        'Underperforming': 'orange', 
                        'Normal': 'gray', 
                        'Overperforming': 'green'})
    
    plt.title("Performance Distribution by Year")
    plt.xlabel("Year")
    plt.ylabel("Percentage of Properties")
    plt.legend(title='Performance Status', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout()
    
    st.pyplot(plt.gcf())
    plt.clf()

    # Display summary statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write(f"### Performance Summary ({selected_year})")
        year_summary_stats = year_df['Performance Status'].value_counts().to_frame()
        year_summary_stats.columns = ['Count']
        year_summary_stats['Percentage'] = (year_summary_stats['Count'] / len(year_df) * 100).round(1)
        st.dataframe(year_summary_stats)
        
    with col2:
        st.write(f"### Average Revenue Gap ({selected_year})")
        year_gap_stats = year_df.groupby('Performance Status')[['Market Performance Gap', 'Prediction Gap']].mean().round(1)
        st.dataframe(year_gap_stats)
        
    with col3:
        st.write("### Year-over-Year Metrics")
        yearly_metrics = df.groupby('Year').agg({
            'Total RevPAR': 'mean',
            'Market RevPAR': 'mean',
            'Occupancy %': 'mean',
            'Total ADR': 'mean'
        }).round(2)
        st.dataframe(yearly_metrics)

    # --- Incremental Revenue Estimation ---
    df['Occupied Nights'] = df['Total Revenue'] / df['Total RevPAR']
    
    # Calculate potential based on both market and predicted RevPAR
    df['Market Revenue Potential'] = np.where(
        df['Performance Status'].isin(['Significantly Underperforming', 'Underperforming']),
        (df['Market RevPAR'] * 0.95 - df['Total RevPAR']) * df['Occupied Nights'],  # Target 95% of market RevPAR
        0
    )
    
    df['Model Revenue Potential'] = np.where(
        df['Performance Status'].isin(['Significantly Underperforming', 'Underperforming']),
        (df['Predicted RevPAR'] - df['Total RevPAR']) * df['Occupied Nights'],
        0
    )
    
    df['Incremental Revenue Potential'] = df[['Market Revenue Potential', 'Model Revenue Potential']].max(axis=1)

    # --- Market Performance Overview ---
    st.subheader("🎯 Market Performance Overview")
    
    # Add filters for City and Year
    col1, col2 = st.columns(2)
    with col1:
        # Handle potential NaN values and convert to string
        cities = df['City Name'].fillna('Unknown').astype(str).unique()
        cities = sorted([city for city in cities if city != 'nan'])
        selected_city = st.selectbox("Select City", cities)
    
    with col2:
        # Add year filter
        years_available = sorted(df['Year'].unique())
        selected_year_overview = st.selectbox("Select Year", years_available, key="overview_year")
    
    # Clean the dataframe columns before filtering
    df['City Name'] = df['City Name'].fillna('Unknown').astype(str)
    #df['Bedroom count category'] = df['Bedroom count category'].fillna('Unknown').astype(str)
    
    # Filter data based on city and year selection
    filtered_df = df[(df['City Name'] == selected_city) & (df['Year'] == selected_year_overview)]
    
    # Create Date column for both dataframes
    df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2) + '-01')
    filtered_df['Date'] = pd.to_datetime(filtered_df['Year'].astype(str) + '-' + filtered_df['Month'].astype(str).str.zfill(2) + '-01')
    
    # KPI Metrics with filtered data
    col1, col2, col3, col4, col5, col6 = st.columns(6)
    with col1:
        avg_revpar_index = (filtered_df['Total RevPAR'] / filtered_df['Market RevPAR']).mean()
        st.metric(
            "RevPAR Index", 
            f"{avg_revpar_index:.2f}x Market",
            delta=f"{((avg_revpar_index - 1) * 100):.1f}% vs Market"
        )
    with col2:
        avg_adr_index = (filtered_df['Total ADR'] / filtered_df['Market ADR']).mean()
        st.metric(
            "ADR Index", 
            f"{avg_adr_index:.2f}x Market",
            delta=f"{((avg_adr_index - 1) * 100):.1f}% vs Market"
        )
    with col3:
        avg_occ_index = (filtered_df['Occupancy %'] / filtered_df['Market Occupancy %']).mean()
        st.metric(
            "Occupancy Index", 
            f"{avg_occ_index:.2f}x Market",
            delta=f"{((avg_occ_index - 1) * 100):.1f}% vs Market"
        )
    with col4:
        avg_los_index = (filtered_df['Average LOS'] / filtered_df['Average Market LOS']).mean()
        st.metric(
            "LOS Index", 
            f"{avg_los_index:.2f}x Market",
            delta=f"{((avg_los_index - 1) * 100):.1f}% vs Market"
        )
    with col5:
        avg_booking_window = filtered_df['Average Booking Window'].mean()
        st.metric(
            "Your Avg Booking Window", 
            f"{avg_booking_window:.1f} days",
            delta=f"{avg_booking_window - filtered_df['Average Market Booking Window'].mean():.1f} vs Market"
        )
    with col6:
        avg_market_booking_window = filtered_df['Average Market Booking Window'].mean()
        st.metric(
            "Market Avg Booking Window", 
            f"{avg_market_booking_window:.1f} days",
            delta=f"{(avg_booking_window / avg_market_booking_window):.2f}x ratio"
        )

    # Market Position Analysis
    with st.expander("📊 Market Position Analysis", expanded=True):
        # Create a bar chart for market indices (seaborn equivalent of radar chart)
        avg_booking_window_index = (filtered_df['Average Booking Window'] / filtered_df['Average Market Booking Window']).mean()
        categories = ['RevPAR Index', 'ADR Index', 'Occupancy Index', 'LOS Index', 'Booking Window Index']
        values = [avg_revpar_index, avg_adr_index, avg_occ_index, avg_los_index, avg_booking_window_index]
        market_values = [1, 1, 1, 1, 1]
        
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(categories))
        width = 0.35
        
        plt.bar(x - width/2, values, width, label=f'Your Properties in {selected_city}', alpha=0.8)
        plt.bar(x + width/2, market_values, width, label='Market Average', alpha=0.8)
        
        plt.xlabel('Market Indices')
        plt.ylabel('Index Value')
        plt.title(f"Market Position Comparison - Properties in {selected_city} ({selected_year_overview})")
        plt.xticks(x, categories, rotation=45)
        plt.legend()
        plt.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Market Benchmark')
        plt.tight_layout()
        
        st.pyplot(plt.gcf())
        plt.clf()
        
        # Add monthly trend analysis
        monthly_metrics = filtered_df.groupby('Month').agg({
            'Total RevPAR': 'mean',
            'Market RevPAR': 'mean',
            'Total ADR': 'mean',
            'Market ADR': 'mean',
            'Occupancy %': 'mean',
            'Market Occupancy %': 'mean',
            'Average Booking Window': 'mean',
            'Average Market Booking Window': 'mean'
        }).round(2)
        
        # Debug: Show absolute numbers for August and surrounding months
       
        
        # Create monthly performance trend
        plt.figure(figsize=(12, 6))
        
        months = monthly_metrics.index
        revpar_index = monthly_metrics['Total RevPAR'] / monthly_metrics['Market RevPAR']
        adr_index = monthly_metrics['Total ADR'] / monthly_metrics['Market ADR']
        occ_index = monthly_metrics['Occupancy %'] / monthly_metrics['Market Occupancy %']
        booking_window_index = monthly_metrics['Average Booking Window'] / monthly_metrics['Average Market Booking Window']
        
        plt.plot(months, revpar_index, marker='o', label='RevPAR Index', linewidth=2)
        plt.plot(months, adr_index, marker='s', label='ADR Index', linewidth=2)
        plt.plot(months, occ_index, marker='^', label='Occupancy Index', linewidth=2)
        plt.plot(months, booking_window_index, marker='d', label='Booking Window Index', linewidth=2)
        
        plt.axhline(y=1, color='gray', linestyle='--', alpha=0.7, label='Market Benchmark')
        
        plt.title(f"Monthly Performance Trends - Properties in {selected_city} ({selected_year_overview})")
        plt.xlabel("Month")
        plt.ylabel("Index (Market = 1.0)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        st.pyplot(plt.gcf())
        plt.clf()

        # Revenue Optimization Model
        st.subheader("💰 Revenue Optimization Model")
        
        # Filter data for selected year
        future_df = df[df['Year'] == selected_year_overview].copy()
        future_filtered_df = filtered_df[filtered_df['Year'] == selected_year_overview].copy()
        
        # Prepare features for the optimization model
        optimization_features = [
            'Total ADR',  # Main input variable
            'Average Booking Window',  # Additional input variable
            'Average LOS',
            'Market ADR', 'Market RevPAR',
            'Month'  # Keep month for seasonality
        ]
        
        # Create dummy variables for City categories
        city_dummies = pd.get_dummies(future_df['City Name'], prefix='City')
        
        # First, train a model to predict occupancy based on ADR and booking window
        X_occ = pd.concat([
            future_df[['Total ADR', 'Average Booking Window', 'Market ADR', 'Month']],
            city_dummies
        ], axis=1)
        
        y_occ = future_df['Occupancy %']
        
        occupancy_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=4,
            learning_rate=0.1
        )
        occupancy_model.fit(X_occ, y_occ)
        
        # Then, train the revenue model
        X_opt = pd.concat([
            future_df[optimization_features],
            city_dummies
        ], axis=1)
        
        y_opt = future_df['Total Revenue']
        
        # Train an XGBoost model for revenue prediction
        revenue_model = xgb.XGBRegressor(
            n_estimators=200,
            max_depth=6,
            learning_rate=0.1,
            objective='reg:squarederror'
        )
        revenue_model.fit(X_opt, y_opt)
        
        # Feature importance analysis
        feature_importance = pd.DataFrame({
            'Feature': X_opt.columns,
            'Importance': revenue_model.feature_importances_
        }).sort_values('Importance', ascending=False)
        
        # Display feature importance
        st.write("#### Key Revenue Drivers")
        plt.figure(figsize=(10, 8))
        
        top_features = feature_importance.head(10)
        sns.barplot(data=top_features, x='Importance', y='Feature', palette='viridis')
        
        plt.title('Top 10 Revenue Drivers')
        plt.xlabel('Importance')
        plt.ylabel('Feature')
        plt.tight_layout()
        
        st.pyplot(plt.gcf())
        plt.clf()
        
        # Optimization Analysis
        st.write("#### Revenue Optimization Analysis")
        
        # Get current metrics for the selected segment
        current_metrics = future_filtered_df.agg({
            'Total ADR': 'mean',
            'Average Booking Window': 'mean',
            'Occupancy %': 'mean',
            'Total Revenue': 'mean'
        })
        
        # Create ranges for ADR and Booking Window
        adr_range = np.linspace(
            max(future_filtered_df['Total ADR'].min(), future_filtered_df['Market ADR'].min() * 0.5),
            min(future_filtered_df['Total ADR'].max(), future_filtered_df['Market ADR'].max() * 1.5),
            20
        )
        
        bw_range = np.linspace(
            max(future_filtered_df['Average Booking Window'].min(), 0),
            future_filtered_df['Average Booking Window'].max(),
            10
        )
        
        # Create prediction grid
        predictions = []
        
        # Check if we have data to work with
        if future_filtered_df.empty and filtered_df.empty:
            st.warning("No data available for the selected segment in 2025 or any other year.")
            st.stop()
            
        # Get base sample, with better error handling
        try:
            if not future_filtered_df.empty:
                base_sample = future_filtered_df.iloc[0].copy()
            elif not filtered_df.empty:
                base_sample = filtered_df.iloc[0].copy()
            else:
                st.warning("No data available for optimization analysis.")
                st.stop()
                
            for adr in adr_range:
                for bw in bw_range:
                    # Predict occupancy for this ADR and booking window
                    occ_features = pd.DataFrame([{
                        'Total ADR': adr,
                        'Average Booking Window': bw,
                        'Market ADR': base_sample['Market ADR'],
                        'Month': base_sample['Month']
                    }])
                    
                    # Add dummy variables
                    for col in city_dummies.columns:
                        occ_features[col] = 1 if col == f'City_{selected_city}' else 0
                    
                    predicted_occupancy = occupancy_model.predict(occ_features)[0]
                    predicted_occupancy = min(max(predicted_occupancy, 0), 100)  # Clip between 0 and 100
                    
                    # Prepare features for revenue prediction
                    rev_features = pd.DataFrame([base_sample[optimization_features]])
                    rev_features['Total ADR'] = adr
                    rev_features['Average Booking Window'] = bw
                    
                    # Add dummy variables
                    for col in city_dummies.columns:
                        rev_features[col] = 1 if col == f'City_{selected_city}' else 0
                    
                    # Predict revenue
                    pred_revenue = revenue_model.predict(rev_features)[0]
                    
                    predictions.append({
                        'ADR': adr,
                        'Booking Window': bw,
                        'Predicted Occupancy': predicted_occupancy,
                        'Predicted Revenue': pred_revenue
                    })
            
            # Convert to DataFrame
            pred_df = pd.DataFrame(predictions)
            
            if pred_df.empty:
                st.warning("No predictions could be generated for the selected segment.")
                st.stop()
                
            # Find optimal point
            optimal_point = pred_df.loc[pred_df['Predicted Revenue'].idxmax()]
            
            # Create ADR optimization visualization
            st.write("#### ADR Optimization Analysis")
            
            # Aggregate predictions by ADR
            adr_analysis = pred_df.groupby('ADR').agg({
                'Predicted Revenue': 'mean',
                'Predicted Occupancy': 'mean'
            }).reset_index()
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot revenue curve
            ax1.plot(adr_analysis['ADR'], adr_analysis['Predicted Revenue'], 
                    'b-', linewidth=2, label='Predicted Revenue')
            ax1.set_xlabel('ADR')
            ax1.set_ylabel('Predicted Revenue', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # Add current and optimal position markers
            ax1.scatter([current_metrics['Total ADR']], [current_metrics['Total Revenue']], 
                       color='red', s=100, marker='*', label='Current Position', zorder=5)
            
            optimal_adr = adr_analysis.loc[adr_analysis['Predicted Revenue'].idxmax()]
            ax1.scatter([optimal_adr['ADR']], [optimal_adr['Predicted Revenue']], 
                       color='lime', s=100, marker='D', label='Optimal Position', zorder=5)
            
            # Create second y-axis for occupancy
            ax2 = ax1.twinx()
            ax2.plot(adr_analysis['ADR'], adr_analysis['Predicted Occupancy'], 
                    'r--', linewidth=2, label='Predicted Occupancy')
            ax2.set_ylabel('Predicted Occupancy %', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_ylim(0, 100)
            
            plt.title(f'ADR Optimization - Properties in {selected_city} ({selected_year_overview})')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()
            
            # Create Booking Window optimization visualization
            st.write("#### Booking Window Optimization Analysis")
            
            # Aggregate predictions by Booking Window
            bw_analysis = pred_df.groupby('Booking Window').agg({
                'Predicted Revenue': 'mean',
                'Predicted Occupancy': 'mean'
            }).reset_index()
            
            fig, ax1 = plt.subplots(figsize=(12, 6))
            
            # Plot revenue curve
            ax1.plot(bw_analysis['Booking Window'], bw_analysis['Predicted Revenue'], 
                    'b-', linewidth=2, label='Predicted Revenue')
            ax1.set_xlabel('Booking Window (Days)')
            ax1.set_ylabel('Predicted Revenue', color='b')
            ax1.tick_params(axis='y', labelcolor='b')
            
            # Add current and optimal position markers
            ax1.scatter([current_metrics['Average Booking Window']], [current_metrics['Total Revenue']], 
                       color='red', s=100, marker='*', label='Current Position', zorder=5)
            
            optimal_bw = bw_analysis.loc[bw_analysis['Predicted Revenue'].idxmax()]
            ax1.scatter([optimal_bw['Booking Window']], [optimal_bw['Predicted Revenue']], 
                       color='lime', s=100, marker='D', label='Optimal Position', zorder=5)
            
            # Create second y-axis for occupancy
            ax2 = ax1.twinx()
            ax2.plot(bw_analysis['Booking Window'], bw_analysis['Predicted Occupancy'], 
                    'r--', linewidth=2, label='Predicted Occupancy')
            ax2.set_ylabel('Predicted Occupancy %', color='r')
            ax2.tick_params(axis='y', labelcolor='r')
            ax2.set_ylim(0, 100)
            
            plt.title(f'Booking Window Optimization - Properties in {selected_city} ({selected_year_overview})')
            
            # Combine legends
            lines1, labels1 = ax1.get_legend_handles_labels()
            lines2, labels2 = ax2.get_legend_handles_labels()
            ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()
            
            # Display optimization insights
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Current ADR",
                    f"${current_metrics['Total ADR']:.2f}",
                    f"Optimal: ${optimal_adr['ADR']:.2f}"
                )
            with col2:
                st.metric(
                    "Current Booking Window",
                    f"{current_metrics['Average Booking Window']:.1f} days",
                    f"Optimal: {optimal_bw['Booking Window']:.1f} days"
                )
            with col3:
                st.metric(
                    "Current Occupancy",
                    f"{current_metrics['Occupancy %']:.1f}%",
                    f"Predicted: {optimal_point['Predicted Occupancy']:.1f}%"
                )
            with col4:
                revenue_potential = max(optimal_adr['Predicted Revenue'], optimal_bw['Predicted Revenue']) - current_metrics['Total Revenue']
                st.metric(
                    "Revenue Opportunity",
                    f"${revenue_potential:,.2f}",
                    f"{(revenue_potential/current_metrics['Total Revenue']*100):.1f}% potential increase"
                )

        except Exception as e:
            st.error(f"Error during optimization analysis: {str(e)}")
            st.warning("Unable to complete optimization analysis. Please check your data and filters.")
            st.stop()

    # --- Seasonality Analysis ---
    st.subheader("🗓️ Seasonality Analysis")
    
    with st.expander("📈 Seasonal Patterns", expanded=True):
        # Year selector for seasonality analysis
        selected_year_seasonality = st.selectbox("Select Year for Seasonality Analysis", years, key="seasonality_year")
        year_df_seasonality = df[df['Year'] == selected_year_seasonality]
        st.write(f"Seasonality data for {selected_year_seasonality}: {len(year_df_seasonality)} records")
        
        # Monthly performance heatmap for selected year
        monthly_metrics = year_df_seasonality.groupby('Month').agg({
            'Total RevPAR': 'mean',
            'Occupancy %': 'mean',
            'Total ADR': 'mean',
            'Average LOS': 'mean'
        }).round(2)
        
        # Debug: Show the data
        st.write("Monthly metrics data:")
        st.write(monthly_metrics)
        
        # Normalize for heatmap with safety check
        monthly_metrics_clean = monthly_metrics.fillna(0)  # Fill NaN with 0
        monthly_metrics_norm = monthly_metrics_clean.copy()
        
        # Only normalize if there's variation in the data
        for col in monthly_metrics_clean.columns:
            col_min = monthly_metrics_clean[col].min()
            col_max = monthly_metrics_clean[col].max()
            if col_max > col_min:  # Only normalize if there's variation
                monthly_metrics_norm[col] = (monthly_metrics_clean[col] - col_min) / (col_max - col_min)
            else:
                monthly_metrics_norm[col] = 0.5  # Set to middle value if no variation
        
        # Debug: Show normalized data
        st.write("Normalized data:")
        st.write(monthly_metrics_norm)
        
        # Create heatmap with safety checks
        plt.figure(figsize=(10, 6))
        
        if len(monthly_metrics_norm) > 0 and len(monthly_metrics_norm.columns) > 0:
            # Use normalized data for better visualization
            sns.heatmap(monthly_metrics_norm.T, 
                       annot=monthly_metrics.T.round(2), 
                       fmt='.2f',
                       cmap='viridis',
                       cbar_kws={'label': 'Normalized Value'})
        else:
            # Fallback: create a simple heatmap with original data
            sns.heatmap(monthly_metrics.T, 
                       annot=True, 
                       fmt='.2f',
                       cmap='viridis',
                       cbar_kws={'label': 'Value'})
        
        plt.title(f'Monthly Performance Heatmap ({selected_year_seasonality})')
        plt.xlabel('Month')
        plt.ylabel('Metric')
        plt.tight_layout()
        
        st.pyplot(plt.gcf())
        plt.clf()

        # Revenue Breakdown
        col1, col2 = st.columns(2)
        with col1:
            # Monthly RevPAR Comparison with year-over-year view
            plt.figure(figsize=(12, 6))
            
            # Add traces for each year
            for year in years:
                year_data = df[df['Year'] == year]
                if len(year_data) > 0:  # Only add if there's data
                    monthly_revpar = year_data.groupby('Month').agg({
                        'Total RevPAR': 'mean',
                        'Market RevPAR': 'mean'
                    }).reset_index()
                    
                    if len(monthly_revpar) > 0:
                        alpha = 1.0 if year == selected_year_seasonality else 0.3
                        plt.bar(monthly_revpar['Month'] - 0.2, monthly_revpar['Total RevPAR'], 
                               width=0.4, label=f'Your RevPAR {year}', 
                               color='lightblue', alpha=alpha)
                        plt.bar(monthly_revpar['Month'] + 0.2, monthly_revpar['Market RevPAR'], 
                               width=0.4, label=f'Market RevPAR {year}', 
                               color='lightgray', alpha=alpha)
            
            plt.title('Monthly RevPAR Comparison (Year-over-Year)')
            plt.xlabel('Month')
            plt.ylabel('RevPAR')
            plt.legend()
            plt.tight_layout()
            
            st.pyplot(plt.gcf())
            plt.clf()
            
        with col2:
            # Occupancy vs ADR Scatter for selected year
            if len(year_df_seasonality) > 0:
                plt.figure(figsize=(10, 6))
                
                scatter = plt.scatter(year_df_seasonality['Occupancy %'], 
                                    year_df_seasonality['Total ADR'],
                                    c=year_df_seasonality['Month'],
                                    s=year_df_seasonality['Total RevPAR'] * 2,
                                    alpha=0.6,
                                    cmap='viridis')
                
                plt.colorbar(scatter, label='Month')
                plt.xlabel('Occupancy Rate (%)')
                plt.ylabel('ADR')
                plt.title(f'Occupancy vs ADR by Month ({selected_year_seasonality})')
                plt.tight_layout()
                
                st.pyplot(plt.gcf())
                plt.clf()
            else:
                st.write("No data available for this year")

        # Year-over-Year Seasonal Comparison
        st.subheader("Year-over-Year Seasonal Comparison")
        
        # Calculate year-over-year changes
        yearly_seasonal_metrics = df.groupby(['Year', 'Month']).agg({
            'Total RevPAR': 'mean',
            'Market RevPAR': 'mean',
            'Occupancy %': 'mean',
            'Total ADR': 'mean'
        }).round(2)
        
        # Create YoY comparison heatmap
        yoy_comparison = yearly_seasonal_metrics.unstack(level=0)
        
        # Convert the data to a format suitable for heatmap
        yoy_data = yoy_comparison['Total RevPAR'].values
        
        plt.figure(figsize=(10, 8))
        
        sns.heatmap(yoy_data,
                   xticklabels=sorted(df['Year'].unique()),
                   yticklabels=list(range(1, 13)),
                   cmap='viridis',
                   cbar_kws={'label': 'RevPAR'})
        
        plt.title("RevPAR Heatmap by Year and Month")
        plt.xlabel("Year")
        plt.ylabel("Month")
        plt.tight_layout()
        
        st.pyplot(plt.gcf())
        plt.clf()

    # --- Booking Window Analysis ---
    st.subheader("📅 Booking Pattern Analysis")
    
    # Calculate booking window metrics
    df['Booking Window Difference'] = df['Average Booking Window'] - df['Average Market Booking Window']
    df['Booking Window Ratio'] = df['Average Booking Window'] / df['Average Market Booking Window']
    
    # Year selector for booking analysis
    selected_year_booking = st.selectbox("Select Year for Booking Analysis", years, key="booking_year")
    year_df_booking = df[df['Year'] == selected_year_booking]
    
    # Using expanders instead of tabs
    with st.expander("📈 Booking Window Trends", expanded=True):
        # Trend analysis by month and year
        monthly_avg = df.groupby(['Year', 'Month']).agg({
            'Average Booking Window': 'mean',
            'Average Market Booking Window': 'mean',
            'Total RevPAR': 'mean'
        }).reset_index()
        
        # Debug: Show booking window statistics
        st.write("### Booking Window Data Quality Check")
        st.write(f"**Your Booking Window Stats:**")
        st.write(f"- Min: {monthly_avg['Average Booking Window'].min():.1f} days")
        st.write(f"- Max: {monthly_avg['Average Booking Window'].max():.1f} days")
        st.write(f"- Mean: {monthly_avg['Average Booking Window'].mean():.1f} days")
        
        st.write(f"**Market Booking Window Stats:**")
        st.write(f"- Min: {monthly_avg['Average Market Booking Window'].min():.1f} days")
        st.write(f"- Max: {monthly_avg['Average Market Booking Window'].max():.1f} days")
        st.write(f"- Mean: {monthly_avg['Average Market Booking Window'].mean():.1f} days")
        
        st.write("### Monthly Booking Window Data")
        st.dataframe(monthly_avg[['Year', 'Month', 'Average Booking Window', 'Average Market Booking Window']].round(1))
        
        # Try a simple approach first
        if len(monthly_avg) > 0:
            plt.figure(figsize=(12, 6))
            
            # Plot booking window trends
            plt.plot(monthly_avg['Month'], monthly_avg['Average Booking Window'], 
                    marker='o', linewidth=2, label='Your Booking Window', color='blue')
            plt.plot(monthly_avg['Month'], monthly_avg['Average Market Booking Window'], 
                    marker='s', linewidth=2, label='Market Booking Window', color='red')
            
            plt.title("Booking Window Trends by Month and Year")
            plt.xlabel('Month')
            plt.ylabel('Days')
            plt.legend()
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            st.write("Chart created, attempting to display...")
            st.pyplot(plt.gcf())
            plt.clf()
        else:
            st.write("No data available for chart")

    with st.expander("💹 Performance Impact", expanded=True):
        # Create scatter plot for selected year
        plt.figure(figsize=(10, 6))
        
        # Create color mapping
        color_map = {'Overperforming': 'green', 'Normal': 'blue', 'Underperforming': 'red'}
        colors = year_df_booking['Performance Status'].map(color_map).fillna('gray').tolist()
        
        # Add scatter points
        scatter = plt.scatter(year_df_booking['Booking Window Ratio'],
                            year_df_booking['Total RevPAR'],
                            c=colors,
                            s=50,
                            alpha=0.6)
        
        # Add reference lines
        plt.axhline(y=year_df_booking['Total RevPAR'].mean(), 
                   linestyle='--', color='gray', alpha=0.7,
                   label=f"Avg RevPAR ({selected_year_booking})")
        plt.axvline(x=1, 
                   linestyle='--', color='gray', alpha=0.7,
                   label="Market Booking Window")
        
        # Update layout
        plt.title(f"Booking Window Ratio vs RevPAR ({selected_year_booking})")
        plt.xlabel("Property vs Market Booking Window Ratio")
        plt.ylabel("RevPAR")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        st.pyplot(plt.gcf())
        plt.clf()
        
        # Calculate and display correlation for selected year
        correlation = year_df_booking['Booking Window Ratio'].corr(year_df_booking['Total RevPAR'])
        st.metric(f"Correlation between Booking Window Ratio and RevPAR ({selected_year_booking})", 
                 f"{correlation:.2f}")

    with st.expander("📊 Distribution Analysis", expanded=True):
        col1, col2 = st.columns(2)
        
        with col1:
            plt.figure(figsize=(8, 6))
            
            plt.hist(year_df_booking['Booking Window Difference'], 
                    bins=20, alpha=0.7, color='skyblue', edgecolor='black')
            
            plt.title(f"Distribution of Booking Window Difference vs Market ({selected_year_booking})")
            plt.xlabel('Days Difference from Market')
            plt.ylabel('Frequency')
            plt.grid(True, alpha=0.3)
            plt.tight_layout()
            
            st.pyplot(plt.gcf())
            plt.clf()
            
        with col2:
            # Performance breakdown by booking pattern for selected year
            # Handle binning with potential duplicate values
            def create_booking_pattern(ratio):
                if ratio <= 0.8:  # 20% below market
                    return 'Short Window'
                elif ratio <= 1.2:  # Within 20% of market
                    return 'Average Window'
                else:  # More than 20% above market
                    return 'Long Window'
            
            year_df_booking['Booking Pattern'] = year_df_booking['Booking Window Ratio'].apply(create_booking_pattern)
            
            pattern_performance = year_df_booking.groupby('Booking Pattern').agg({
                'Total RevPAR': 'mean',
                'Listing Name': 'count'
            }).reset_index()
            
            # Ensure consistent order
            pattern_order = ['Short Window', 'Average Window', 'Long Window']
            pattern_performance['Booking Pattern'] = pd.Categorical(
                pattern_performance['Booking Pattern'],
                categories=pattern_order,
                ordered=True
            )
            pattern_performance = pattern_performance.sort_values('Booking Pattern')
            
            plt.figure(figsize=(8, 6))
            
            bars = plt.bar(pattern_performance['Booking Pattern'], 
                          pattern_performance['Total RevPAR'],
                          color=['lightcoral', 'lightblue', 'lightgreen'])
            
            # Add value labels on bars
            for bar, value in zip(bars, pattern_performance['Total RevPAR']):
                plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.1,
                        f'{value:.2f}', ha='center', va='bottom')
            
            plt.title(f"Average RevPAR by Booking Pattern ({selected_year_booking})")
            plt.xlabel('Booking Pattern')
            plt.ylabel('Average RevPAR')
            plt.xticks(rotation=45)
            plt.tight_layout()
            
            st.pyplot(plt.gcf())
            plt.clf()
            
            # Add distribution information
            st.write("### Booking Pattern Distribution")
            pattern_dist = pattern_performance.set_index('Booking Pattern')
            pattern_dist['Percentage'] = (pattern_dist['Listing Name'] / pattern_dist['Listing Name'].sum() * 100).round(1)
            pattern_dist = pattern_dist.rename(columns={'Listing Name': 'Count'})
            st.dataframe(pattern_dist[['Count', 'Percentage']])

        # Year-over-Year Booking Pattern Analysis
        st.subheader("Year-over-Year Booking Pattern Analysis")
        
        yearly_booking_metrics = df.groupby('Year').agg({
            'Average Booking Window': 'mean',
            'Average Market Booking Window': 'mean',
            'Booking Window Ratio': 'mean',
            'Total RevPAR': 'mean'
        }).round(2)
        
        plt.figure(figsize=(10, 6))
        
        x = np.arange(len(yearly_booking_metrics.index))
        width = 0.35
        
        plt.bar(x - width/2, yearly_booking_metrics['Average Booking Window'], 
               width, label='Your Avg Booking Window', color='lightblue', alpha=0.8)
        plt.bar(x + width/2, yearly_booking_metrics['Average Market Booking Window'], 
               width, label='Market Avg Booking Window', color='lightgray', alpha=0.8)
        
        plt.title("Year-over-Year Booking Window Comparison")
        plt.xlabel('Year')
        plt.ylabel('Days')
        plt.xticks(x, yearly_booking_metrics.index)
        plt.legend()
        plt.tight_layout()
        
        st.pyplot(plt.gcf())
        plt.clf()

        # Summary statistics
        st.write("### Key Insights")
        col3, col4, col5 = st.columns(3)
        
        with col3:
            optimal_window = year_df_booking.loc[year_df_booking['Total RevPAR'].idxmax(), 'Average Booking Window']
            st.metric(f"Optimal Booking Window ({selected_year_booking})", f"{optimal_window:.1f} days")
            
        with col4:
            market_median = year_df_booking['Average Market Booking Window'].median()
            st.metric(f"Market Median Window ({selected_year_booking})", f"{market_median:.1f} days")
            
        with col5:
            best_pattern = pattern_performance.loc[pattern_performance['Total RevPAR'].idxmax(), 'Booking Pattern']
            st.metric(f"Best Performing Pattern ({selected_year_booking})", best_pattern)

    # --- Summary Table ---
    st.subheader("📋 Performance Summary")
    summary = df.groupby('Performance Status').agg({
        'Listing Name': 'count',
        'Incremental Revenue Potential': 'sum'
    }).rename(columns={'Listing Name': 'Count'})
    st.dataframe(summary)

    # --- Revenue Opportunity Table ---
    st.subheader("💸 Top Underperformers")
    under_df = df[df['Performance Status'] == 'Underperforming']
    top_under = under_df.sort_values(by='Incremental Revenue Potential', ascending=False)
    
    # Create a more informative display for month
    top_under['Period'] = top_under['Year'].astype(str) + '-' + top_under['Month'].astype(str).str.zfill(2)
    
    st.dataframe(top_under[['Listing Name', 'Period', 'Total Revenue', 'Predicted RevPAR', 'Total RevPAR',
                           'RevPAR Residual', 'Incremental Revenue Potential']].head(35))

    # --- Visualization ---
    st.subheader("📈 Actual vs Predicted RevPAR")
    
    plt.figure(figsize=(10, 8))
    
    # Create color mapping
    color_map = {'Overperforming': 'green', 'Normal': 'blue', 'Underperforming': 'orange', 'Significantly Underperforming': 'red'}
    colors = df['Performance Status'].map(color_map).fillna('gray').tolist()
    
    # Create scatter plot
    scatter = plt.scatter(df['Predicted RevPAR'], df['Total RevPAR'], 
                         c=colors, alpha=0.6, s=30)
    
    # Add diagonal line for perfect prediction
    min_val = min(df['Predicted RevPAR'].min(), df['Total RevPAR'].min())
    max_val = max(df['Predicted RevPAR'].max(), df['Total RevPAR'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--', alpha=0.5, label='Perfect Prediction')
    
    plt.title("Actual vs Predicted RevPAR")
    plt.xlabel("Predicted RevPAR")
    plt.ylabel("Actual RevPAR")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    st.pyplot(plt.gcf())
    plt.clf()

    # --- Detailed Underperformance Analysis ---
    st.subheader("🔍 Detailed Underperformance Analysis (June 2024 - June 2025)")
    
    # Create Date column if not exists
    if 'Date' not in df.columns:
        df['Date'] = pd.to_datetime(df['Year'].astype(str) + '-' + df['Month'].astype(str).str.zfill(2) + '-01')
    
    # Filter for June 2024 to June 2025
    mask = (df['Date'] >= '2024-06-01') & (df['Date'] <= '2025-06-01')
    analysis_df = df[mask].copy()
    
    # First, identify properties that were underperforming in any month during the period
    underperforming_properties = analysis_df[
        analysis_df['Performance Status'].isin(['Significantly Underperforming', 'Underperforming'])
    ]['Listing Name'].unique()
    
    # Then get all months for these properties
    underperform_df = analysis_df[analysis_df['Listing Name'].isin(underperforming_properties)].copy()
    
    # Debug data for specific property
    debug_property = "3B-AddressOpera2-1102 -- Manzil - 3BR | Downtown | Connected to Dubai Mall"
    st.write("### Property Performance Check")
    property_debug = underperform_df[underperform_df['Listing Name'] == debug_property][
        ['Date', 'Total Revenue', 'Performance Status']
    ].sort_values('Date')
    st.write("Monthly data for the property (all months):")
    st.dataframe(property_debug)
    st.write(f"Total Revenue across all months: {property_debug['Total Revenue'].sum():,.2f}")
    st.write(f"Number of months as Underperforming: {len(property_debug[property_debug['Performance Status'] == 'Underperforming'])}")
    st.write(f"Number of months as Significantly Underperforming: {len(property_debug[property_debug['Performance Status'] == 'Significantly Underperforming'])}")
    
    # Calculate metrics for the entire period
    property_metrics = underperform_df.groupby(['Listing Name', 'City Name', 'Bedroom count category']).agg({
        'Total Revenue': 'sum',
        'Total RevPAR': 'mean',
        'Market RevPAR': 'mean',
        'Predicted RevPAR': 'mean',
        'RevPAR Residual': 'mean',
        'Incremental Revenue Potential': 'sum',
        'Occupancy %': 'mean',
        'Market Occupancy %': 'mean',
        'Total ADR': 'mean',
        'Market ADR': 'mean'
    }).reset_index()
    
    # Calculate additional metrics
    property_metrics['RevPAR Gap %'] = ((property_metrics['Market RevPAR'] - property_metrics['Total RevPAR']) / property_metrics['Market RevPAR'] * 100).round(1)
    property_metrics['Occupancy Gap %'] = (property_metrics['Market Occupancy %'] - property_metrics['Occupancy %']).round(1)
    property_metrics['ADR Gap %'] = ((property_metrics['Market ADR'] - property_metrics['Total ADR']) / property_metrics['Market ADR'] * 100).round(1)
    property_metrics['Revenue Opportunity %'] = (property_metrics['Incremental Revenue Potential'] / property_metrics['Total Revenue'] * 100).round(2)
    
    # Determine performance status with revenue opportunity threshold
    def determine_final_status(row):
        if row['Revenue Opportunity %'] <= 5:  # If opportunity is less than 5%, always Normal
            return 'Normal'
        
        # Get the most severe status from monthly data
        monthly_status = underperform_df[underperform_df['Listing Name'] == row['Listing Name']]['Performance Status'].unique()
        if 'Significantly Underperforming' in monthly_status:
            return 'Significantly Underperforming'
        elif 'Underperforming' in monthly_status:
            return 'Underperforming'
        else:
            return 'Normal'
    
    property_metrics['Performance Status'] = property_metrics.apply(determine_final_status, axis=1)
    
    # Sort by revenue potential
    property_metrics = property_metrics.sort_values('Incremental Revenue Potential', ascending=False)
    
    # Create summary by performance status
    status_summary = property_metrics[property_metrics['Performance Status'] != 'Normal'].groupby('Performance Status').agg({
        'Incremental Revenue Potential': 'sum',
        'RevPAR Gap %': 'mean',
        'Occupancy Gap %': 'mean',
        'ADR Gap %': 'mean'
    }).round(1)
    
    status_summary.columns = ['Revenue Opportunity (Jun 24-Jun 25)', 'Avg RevPAR Gap %', 'Avg Occupancy Gap %', 'Avg ADR Gap %']
    
    # Display summary
    st.write("### Summary by Performance Status")
    st.dataframe(status_summary)
    
    # Display property lists by status
    st.write("### Properties by Performance Status")
    for status in ['Significantly Underperforming', 'Underperforming']:
        status_properties = property_metrics[property_metrics['Performance Status'] == status].copy()
        
        if len(status_properties) > 0:  # Only show if there are properties with this status
            st.write(f"#### {status} Properties")
            st.write(f"Total Revenue Opportunity (Jun 24-Jun 25): ${status_properties['Incremental Revenue Potential'].sum():,.0f}")
            
            # Format the revenue columns
            status_properties['Total Revenue'] = status_properties['Total Revenue'].round(0).astype(int)
            status_properties['Incremental Revenue Potential'] = status_properties['Incremental Revenue Potential'].round(0).astype(int)
            
            # Select and reorder columns for display
            display_cols = [
                'Listing Name', 'City Name', 'Bedroom count category',
                'Total Revenue', 'Incremental Revenue Potential', 'Revenue Opportunity %',
                'RevPAR Gap %', 'Occupancy Gap %', 'ADR Gap %'
            ]
            
            # Display the properties
            st.dataframe(status_properties[display_cols])
    
    # Calculate and display total opportunity (only for actual underperforming properties)
    underperforming_metrics = property_metrics[property_metrics['Performance Status'] != 'Normal']
    total_opportunity = underperforming_metrics['Incremental Revenue Potential'].sum()
    avg_per_property = total_opportunity / len(underperforming_metrics) if len(underperforming_metrics) > 0 else 0
    
    col1, col2 = st.columns(2)
    with col1:
        st.metric(
            "Total Revenue Opportunity (Jun 24-Jun 25)",
            f"${total_opportunity:,.0f}",
            f"Across {len(underperforming_metrics)} properties"
        )
    with col2:
        st.metric(
            "Average Opportunity per Property",
            f"${avg_per_property:,.0f}",
            f"For 13-month period"
        )

else:
    st.info("Please upload a CSV or Excel file to begin analysis.")