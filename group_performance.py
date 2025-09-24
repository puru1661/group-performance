import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import re

st.set_page_config(page_title="Group Performance Dashboard", layout="wide")
st.title("📊 Group Performance Analyzer (June-December 2025)")

# --- Upload File ---
uploaded_file = st.file_uploader("Upload your Group Performance dataset (CSV or Excel)", type=["csv", "xlsx", "xls"])

if uploaded_file is not None:
    # Read the file based on its type
    file_type = uploaded_file.name.split('.')[-1].lower()
    
    if file_type == 'csv':
        df = pd.read_csv(uploaded_file)
    else:  # xlsx or xls
        df = pd.read_excel(uploaded_file)

    # --- Data Cleaning & Preparation ---
    def clean_numeric(val):
        """Clean numeric values by removing currency symbols and converting to float"""
        if isinstance(val, str):
            # Remove currency symbols and other non-numeric characters except decimal point and minus
            val = re.sub(r'[^\d.-]', '', val)
            return float(val) if val != '' else np.nan
        return val

    # Clean numeric columns
    numeric_cols = [
        'Total Revenue', 'Total Revenue STLY', 'Total Revenue LY',
        'Rental RevPAR', 'Rental RevPAR STLY', 'Rental RevPAR LY',
        'Occupancy %', 'Occupancy % STLY', 'Occupancy % LY',
        'Market Occupancy %', 'Market Occupancy % STLY', 'Market Occupancy % LY',
        'Total ADR', 'Total ADR STLY', 'Total ADR LY',
        'Rental ADR', 'Rental ADR STLY', 'Rental ADR LY',
        'ADR Index', 'RevPAR Index', 'Market Penetration Index %'
    ]

    for col in numeric_cols:
        if col in df.columns:
            df[col] = df[col].apply(clean_numeric)

    # Clean percentage columns
    percentage_cols = [
        'Total Revenue STLY YoY %', 'Rental RevPAR STLY YoY %', 'Occupancy STLY YoY Difference',
        'Total ADR STLY YoY %', 'Rental ADR STLY YoY %'
    ]

    for col in percentage_cols:
        if col in df.columns:
            df[col] = df[col].apply(lambda x: clean_numeric(str(x).replace('%', '')) if pd.notna(x) else np.nan)

    # Extract year and month
    df['Year'] = df['Year & Month'].str.extract(r'(\d{4})')[0].astype(int)
    df['Month'] = df['Year & Month'].str.extract(r'-(\d{2})')[0].astype(int)
    df['Month_Name'] = df['Year & Month'].str.extract(r'\((\w+)\)')[0]

    # Filter for June-December 2025
    df_filtered = df[(df['Year'] == 2025) & (df['Month'].isin([6, 7, 8, 9, 10, 11, 12]))].copy()

    if len(df_filtered) == 0:
        st.error("No data found for June-December 2025. Please check your data.")
        st.stop()

    # --- Sidebar Controls ---
    st.sidebar.header("Dashboard Controls")
    
    # Month selector
    available_months = sorted(df_filtered['Month'].unique())
    month_names = {6: 'June', 7: 'July', 8: 'August', 9: 'September', 10: 'October', 11: 'November', 12: 'December'}
    selected_months = st.sidebar.multiselect(
        "Select Months to Analyze",
        available_months,
        default=available_months,
        format_func=lambda x: f"{month_names.get(x, x)} 2025",
        help="Choose which months to include in the analysis"
    )

    # Group selector
    available_groups = sorted(df_filtered['Group Name'].unique())
    selected_groups = st.sidebar.multiselect(
        "Select Groups to Analyze",
        available_groups,
        default=available_groups,
        help="Choose which groups to include in the analysis"
    )

    # Aggregation method selector
    aggregation_method = st.sidebar.selectbox(
        "Select Aggregation Method",
        ["Average", "Median", "Weighted Average", "Best Performer", "Worst Performer"],
        index=1,  # Set Median as default (index 1)
        help="Choose how to aggregate performance metrics within each group"
    )
    
    # KPI weights
    st.sidebar.subheader("KPI Weights")
    revpar_weight = 0.7 #st.sidebar.slider("RevPAR Index Weight", 0.0, 1.0, 0.6, 0.1)
    revenue_weight = 0.1 #st.sidebar.slider("YoY Revenue Growth Weight", 0.0, 1.0, 0.1, 0.1)
    mpi_weight = 0.1 #st.sidebar.slider("MPI Weight", 0.0, 1.0, 0.1, 0.1)
    adr_weight = 0.2 #st.sidebar.slider("ADR Index Weight", 0.0, 1.0, 0.2, 0.1)
    
    # Normalize weights
    total_weight = revpar_weight + revenue_weight + mpi_weight + adr_weight
    if total_weight > 0:
        revpar_weight /= total_weight
        revenue_weight /= total_weight
        mpi_weight /= total_weight
        adr_weight /= total_weight

    # Filter data for selected groups and months
    df_analysis = df_filtered[
        (df_filtered['Group Name'].isin(selected_groups)) & 
        (df_filtered['Month'].isin(selected_months))
    ].copy()

    # --- Calculate Group Metrics Function ---
    def calculate_group_metrics(group_data, method):
        """Calculate aggregated metrics for a group based on the selected method"""
        metrics = {}
        
        if method == "Average":
            metrics['RevPAR_Index'] = group_data['RevPAR Index'].mean()
            metrics['YoY_Revenue_Growth'] = group_data['Total Revenue STLY YoY %'].mean()
            metrics['MPI'] = group_data['Market Penetration Index %'].mean()
            metrics['ADR_Index'] = group_data['ADR Index'].mean()
            metrics['Occupancy_vs_Market'] = (group_data['Occupancy %'] / group_data['Market Occupancy %']).mean()
            metrics['Total_Revenue'] = group_data['Total Revenue'].sum()
            metrics['Property_Count'] = len(group_data)
            metrics['Revenue_per_Property'] = metrics['Total_Revenue'] / metrics['Property_Count']
            
        elif method == "Median":
            metrics['RevPAR_Index'] = group_data['RevPAR Index'].median()
            metrics['YoY_Revenue_Growth'] = group_data['Total Revenue STLY YoY %'].median()
            metrics['MPI'] = group_data['Market Penetration Index %'].median()
            metrics['ADR_Index'] = group_data['ADR Index'].median()
            metrics['Occupancy_vs_Market'] = (group_data['Occupancy %'] / group_data['Market Occupancy %']).median()
            metrics['Total_Revenue'] = group_data['Total Revenue'].sum()
            metrics['Property_Count'] = len(group_data)
            metrics['Revenue_per_Property'] = metrics['Total_Revenue'] / metrics['Property_Count']
            
        elif method == "Weighted Average":
            # Weight by revenue
            weights = group_data['Total Revenue']
            metrics['RevPAR_Index'] = np.average(group_data['RevPAR Index'], weights=weights)
            metrics['YoY_Revenue_Growth'] = np.average(group_data['Total Revenue STLY YoY %'], weights=weights)
            metrics['MPI'] = np.average(group_data['Market Penetration Index %'], weights=weights)
            metrics['ADR_Index'] = np.average(group_data['ADR Index'], weights=weights)
            metrics['Occupancy_vs_Market'] = np.average(
                group_data['Occupancy %'] / group_data['Market Occupancy %'], weights=weights
            )
            metrics['Total_Revenue'] = group_data['Total Revenue'].sum()
            metrics['Property_Count'] = len(group_data)
            metrics['Revenue_per_Property'] = metrics['Total_Revenue'] / metrics['Property_Count']
            
        elif method == "Best Performer":
            # Get the property with highest RevPAR Index
            best_property = group_data.loc[group_data['RevPAR Index'].idxmax()]
            metrics['RevPAR_Index'] = best_property['RevPAR Index']
            metrics['YoY_Revenue_Growth'] = best_property['Total Revenue STLY YoY %']
            metrics['MPI'] = best_property['Market Penetration Index %']
            metrics['ADR_Index'] = best_property['ADR Index']
            metrics['Occupancy_vs_Market'] = best_property['Occupancy %'] / best_property['Market Occupancy %']
            metrics['Total_Revenue'] = group_data['Total Revenue'].sum()
            metrics['Property_Count'] = len(group_data)
            metrics['Revenue_per_Property'] = metrics['Total_Revenue'] / metrics['Property_Count']
            
        elif method == "Worst Performer":
            # Get the property with lowest RevPAR Index
            worst_property = group_data.loc[group_data['RevPAR Index'].idxmin()]
            metrics['RevPAR_Index'] = worst_property['RevPAR Index']
            metrics['YoY_Revenue_Growth'] = worst_property['Total Revenue STLY YoY %']
            metrics['MPI'] = worst_property['Market Penetration Index %']
            metrics['ADR_Index'] = worst_property['ADR Index']
            metrics['Occupancy_vs_Market'] = worst_property['Occupancy %'] / worst_property['Market Occupancy %']
            metrics['Total_Revenue'] = group_data['Total Revenue'].sum()
            metrics['Property_Count'] = len(group_data)
            metrics['Revenue_per_Property'] = metrics['Total_Revenue'] / metrics['Property_Count']
        
        # Calculate consistency (lower standard deviation = more consistent)
        metrics['Consistency_Score'] = 1 / (1 + group_data['RevPAR Index'].std())
        
        # Calculate composite score
        metrics['Composite_Score'] = (
            metrics['RevPAR_Index'] * revpar_weight +
            metrics['YoY_Revenue_Growth'] * revenue_weight +
            metrics['MPI'] * mpi_weight +
            metrics['ADR_Index'] * adr_weight
        )
        
        return metrics

    # --- Monthly Performance Analysis ---
    st.subheader("📅 Monthly Performance Analysis")
    
    # Month selector for detailed analysis
    selected_month_detail = st.selectbox(
        "Select Month for Detailed Analysis",
        selected_months,
        format_func=lambda x: f"{month_names.get(x, x)} 2025",
        help="Choose a specific month to analyze in detail"
    )
    
    # Filter data for selected month
    df_monthly = df_analysis[df_analysis['Month'] == selected_month_detail].copy()
    
    if len(df_monthly) > 0:
        st.write(f"### {month_names.get(selected_month_detail, selected_month_detail)} 2025 Performance")
        
        # Calculate monthly metrics for each group
        monthly_group_metrics = {}
        for group in selected_groups:
            group_data = df_monthly[df_monthly['Group Name'] == group]
            if len(group_data) > 0:
                monthly_group_metrics[group] = calculate_group_metrics(group_data, aggregation_method)
        
        if monthly_group_metrics:
            monthly_metrics_df = pd.DataFrame(monthly_group_metrics).T
            monthly_metrics_df = monthly_metrics_df.sort_values('Composite_Score', ascending=False)
            
            # Monthly summary cards
            cols = st.columns(len(monthly_group_metrics))
            for i, (group, metrics) in enumerate(monthly_metrics_df.iterrows()):
                with cols[i]:
                    st.metric(
                        label=f"Group {group}",
                        value=f"{metrics['Composite_Score']:.2f}",
                        delta=f"#{i+1} Rank",
                        help=f"RevPAR Index: {metrics['RevPAR_Index']:.2f}\nYoY Growth: {metrics['YoY_Revenue_Growth']:.1f}%\nMPI: {metrics['MPI']:.1f}%"
                    )
            
                # Monthly performance comparison
                fig, axes = plt.subplots(2, 3, figsize=(18, 10))
                
                # RevPAR Index
                axes[0, 0].bar(monthly_metrics_df.index, monthly_metrics_df['RevPAR_Index'], color='skyblue', alpha=0.7)
                axes[0, 0].set_title(f'RevPAR Index by Group - {month_names.get(selected_month_detail, selected_month_detail)}')
                axes[0, 0].set_ylabel('RevPAR Index')
                axes[0, 0].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Market Average')
                axes[0, 0].legend()
                
                # YoY Revenue Growth
                axes[0, 1].bar(monthly_metrics_df.index, monthly_metrics_df['YoY_Revenue_Growth'], color='lightgreen', alpha=0.7)
                axes[0, 1].set_title(f'YoY Revenue Growth by Group - {month_names.get(selected_month_detail, selected_month_detail)}')
                axes[0, 1].set_ylabel('Growth %')
                axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
                
                # MPI
                axes[0, 2].bar(monthly_metrics_df.index, monthly_metrics_df['MPI'], color='orange', alpha=0.7)
                axes[0, 2].set_title(f'Market Penetration Index by Group - {month_names.get(selected_month_detail, selected_month_detail)}')
                axes[0, 2].set_ylabel('MPI %')
                axes[0, 2].axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Market Average')
                axes[0, 2].legend()
                
                # ADR Index
                axes[1, 0].bar(monthly_metrics_df.index, monthly_metrics_df['ADR_Index'], color='lightcoral', alpha=0.7)
                axes[1, 0].set_title(f'ADR Index by Group - {month_names.get(selected_month_detail, selected_month_detail)}')
                axes[1, 0].set_ylabel('ADR Index')
                axes[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Market Average')
                axes[1, 0].legend()
                
                # Occupancy vs Market
                axes[1, 1].bar(monthly_metrics_df.index, monthly_metrics_df['Occupancy_vs_Market'], color='gold', alpha=0.7)
                axes[1, 1].set_title(f'Occupancy vs Market by Group - {month_names.get(selected_month_detail, selected_month_detail)}')
                axes[1, 1].set_ylabel('Occupancy Ratio')
                axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Market Average')
                axes[1, 1].legend()
                
                # Composite Score
                axes[1, 2].bar(monthly_metrics_df.index, monthly_metrics_df['Composite_Score'], color='purple', alpha=0.7)
                axes[1, 2].set_title(f'Composite Score by Group - {month_names.get(selected_month_detail, selected_month_detail)}')
                axes[1, 2].set_ylabel('Composite Score')
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()
            
            # Monthly detailed table
            st.write(f"**Detailed Performance Table - {month_names.get(selected_month_detail, selected_month_detail)} 2025:**")
            display_monthly_metrics = monthly_metrics_df[['RevPAR_Index', 'YoY_Revenue_Growth', 'MPI', 'ADR_Index',
                                                        'Occupancy_vs_Market', 'Composite_Score', 'Consistency_Score',
                                                        'Property_Count', 'Revenue_per_Property']].round(2)
            
            display_monthly_metrics.columns = ['RevPAR Index', 'YoY Revenue Growth (%)', 'MPI (%)', 'ADR Index',
                                             'Occupancy vs Market', 'Composite Score', 'Consistency Score',
                                             'Property Count', 'Revenue per Property']
            
            st.dataframe(display_monthly_metrics)
        else:
            st.warning(f"No data available for the selected groups in {month_names.get(selected_month_detail, selected_month_detail)} 2025.")
    else:
        st.warning(f"No data available for {month_names.get(selected_month_detail, selected_month_detail)} 2025.")

    # --- Monthly Trends Analysis ---
    st.subheader("📈 Monthly Trends Analysis")
    
    if len(selected_months) > 1:
        # Calculate metrics for each month
        monthly_trends_data = []
        for month in selected_months:
            month_data = df_analysis[df_analysis['Month'] == month]
            for group in selected_groups:
                group_month_data = month_data[month_data['Group Name'] == group]
                if len(group_month_data) > 0:
                    metrics = calculate_group_metrics(group_month_data, aggregation_method)
                    metrics['Month'] = month
                    metrics['Month_Name'] = month_names.get(month, month)
                    metrics['Group'] = group
                    monthly_trends_data.append(metrics)
        
        if monthly_trends_data:
            trends_df = pd.DataFrame(monthly_trends_data)
            
            # Create monthly trends visualization
            fig, axes = plt.subplots(2, 3, figsize=(20, 10))
            
            # RevPAR Index trends
            for group in selected_groups:
                group_trends = trends_df[trends_df['Group'] == group].sort_values('Month')
                if len(group_trends) > 0:
                    axes[0, 0].plot(group_trends['Month'], group_trends['RevPAR_Index'], 
                                   marker='o', label=f'Group {group}', linewidth=2)
            axes[0, 0].set_title('RevPAR Index Trends by Group')
            axes[0, 0].set_xlabel('Month')
            axes[0, 0].set_ylabel('RevPAR Index')
            axes[0, 0].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Market Average')
            axes[0, 0].legend()
            axes[0, 0].grid(True, alpha=0.3)
            
            # YoY Revenue Growth trends
            for group in selected_groups:
                group_trends = trends_df[trends_df['Group'] == group].sort_values('Month')
                if len(group_trends) > 0:
                    axes[0, 1].plot(group_trends['Month'], group_trends['YoY_Revenue_Growth'], 
                                   marker='s', label=f'Group {group}', linewidth=2)
            axes[0, 1].set_title('YoY Revenue Growth Trends by Group')
            axes[0, 1].set_xlabel('Month')
            axes[0, 1].set_ylabel('Growth %')
            axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
            axes[0, 1].legend()
            axes[0, 1].grid(True, alpha=0.3)
            
            # MPI trends
            for group in selected_groups:
                group_trends = trends_df[trends_df['Group'] == group].sort_values('Month')
                if len(group_trends) > 0:
                    axes[0, 2].plot(group_trends['Month'], group_trends['MPI'], 
                                   marker='^', label=f'Group {group}', linewidth=2)
            axes[0, 2].set_title('Market Penetration Index Trends by Group')
            axes[0, 2].set_xlabel('Month')
            axes[0, 2].set_ylabel('MPI %')
            axes[0, 2].axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Market Average')
            axes[0, 2].legend()
            axes[0, 2].grid(True, alpha=0.3)
            
            # ADR Index trends
            for group in selected_groups:
                group_trends = trends_df[trends_df['Group'] == group].sort_values('Month')
                if len(group_trends) > 0:
                    axes[1, 0].plot(group_trends['Month'], group_trends['ADR_Index'], 
                                   marker='v', label=f'Group {group}', linewidth=2)
            axes[1, 0].set_title('ADR Index Trends by Group')
            axes[1, 0].set_xlabel('Month')
            axes[1, 0].set_ylabel('ADR Index')
            axes[1, 0].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Market Average')
            axes[1, 0].legend()
            axes[1, 0].grid(True, alpha=0.3)
            
            # Occupancy vs Market trends
            for group in selected_groups:
                group_trends = trends_df[trends_df['Group'] == group].sort_values('Month')
                if len(group_trends) > 0:
                    axes[1, 1].plot(group_trends['Month'], group_trends['Occupancy_vs_Market'], 
                                   marker='<', label=f'Group {group}', linewidth=2)
            axes[1, 1].set_title('Occupancy vs Market Trends by Group')
            axes[1, 1].set_xlabel('Month')
            axes[1, 1].set_ylabel('Occupancy Ratio')
            axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Market Average')
            axes[1, 1].legend()
            axes[1, 1].grid(True, alpha=0.3)
            
            # Composite Score trends
            for group in selected_groups:
                group_trends = trends_df[trends_df['Group'] == group].sort_values('Month')
                if len(group_trends) > 0:
                    axes[1, 2].plot(group_trends['Month'], group_trends['Composite_Score'], 
                                   marker='d', label=f'Group {group}', linewidth=2)
            axes[1, 2].set_title('Composite Score Trends by Group')
            axes[1, 2].set_xlabel('Month')
            axes[1, 2].set_ylabel('Composite Score')
            axes[1, 2].legend()
            axes[1, 2].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()
            
            # Monthly trends table
            st.write("**Monthly Trends Summary:**")
            trends_display = trends_df.pivot(index='Group', columns='Month_Name', values=['RevPAR_Index', 'YoY_Revenue_Growth', 'MPI', 'ADR_Index', 'Composite_Score']).round(2)
            st.dataframe(trends_display)
        else:
            st.warning("No trend data available for the selected months and groups.")
    else:
        st.info("Select multiple months to see trends analysis.")

    # --- Overall Group Performance Analysis ---
    st.subheader("📈 Overall Group Performance (Selected Period)")

    # Calculate metrics for each group
    group_metrics = {}
    for group in selected_groups:
        group_data = df_analysis[df_analysis['Group Name'] == group]
        group_metrics[group] = calculate_group_metrics(group_data, aggregation_method)

    # Convert to DataFrame for easier analysis
    metrics_df = pd.DataFrame(group_metrics).T
    metrics_df = metrics_df.sort_values('Composite_Score', ascending=False)

    # --- Summary Cards ---
    st.subheader("🏆 Group Performance Summary")
    
    cols = st.columns(len(selected_groups))
    for i, (group, metrics) in enumerate(metrics_df.iterrows()):
        with cols[i]:
            st.metric(
                label=f"Group {group}",
                value=f"{metrics['Composite_Score']:.2f}",
                delta=f"#{i+1} Rank",
                help=f"RevPAR Index: {metrics['RevPAR_Index']:.2f}\nYoY Growth: {metrics['YoY_Revenue_Growth']:.1f}%\nMPI: {metrics['MPI']:.1f}%"
            )

    # --- Group Comparison Charts ---
    st.subheader("📊 Group Performance Comparison")

    # Create tabs for different visualizations
    tab1, tab2, tab3, tab4 = st.tabs(["📈 Performance Metrics", "🎯 Market Position", "📅 Monthly Trends", "📋 Detailed Table"])

    with tab1:
        # Performance metrics bar chart
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # RevPAR Index
        axes[0, 0].bar(metrics_df.index, metrics_df['RevPAR_Index'], color='skyblue', alpha=0.7)
        axes[0, 0].set_title('RevPAR Index by Group')
        axes[0, 0].set_ylabel('RevPAR Index')
        axes[0, 0].axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Market Average')
        axes[0, 0].legend()
        
        # YoY Revenue Growth
        axes[0, 1].bar(metrics_df.index, metrics_df['YoY_Revenue_Growth'], color='lightgreen', alpha=0.7)
        axes[0, 1].set_title('YoY Revenue Growth by Group')
        axes[0, 1].set_ylabel('Growth %')
        axes[0, 1].axhline(y=0, color='red', linestyle='--', alpha=0.5)
        
        # MPI
        axes[1, 0].bar(metrics_df.index, metrics_df['MPI'], color='orange', alpha=0.7)
        axes[1, 0].set_title('Market Penetration Index by Group')
        axes[1, 0].set_ylabel('MPI %')
        axes[1, 0].axhline(y=100, color='red', linestyle='--', alpha=0.5, label='Market Average')
        axes[1, 0].legend()
        
        # Composite Score
        axes[1, 1].bar(metrics_df.index, metrics_df['Composite_Score'], color='purple', alpha=0.7)
        axes[1, 1].set_title('Composite Score by Group')
        axes[1, 1].set_ylabel('Composite Score')
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()

    with tab2:
        # Market position scatter plot
        fig, ax = plt.subplots(figsize=(10, 6))
        
        scatter = ax.scatter(metrics_df['MPI'], metrics_df['RevPAR_Index'], 
                           s=metrics_df['Total_Revenue']/1000, 
                           c=metrics_df['Composite_Score'], 
                           cmap='viridis', alpha=0.7)
        
        for group in metrics_df.index:
            ax.annotate(f'Group {group}', 
                       (metrics_df.loc[group, 'MPI'], metrics_df.loc[group, 'RevPAR_Index']),
                       xytext=(5, 5), textcoords='offset points')
        
        ax.set_xlabel('Market Penetration Index (%)')
        ax.set_ylabel('RevPAR Index')
        ax.set_title('Group Market Position (Bubble size = Total Revenue)')
        ax.axhline(y=1, color='red', linestyle='--', alpha=0.5, label='Market RevPAR')
        ax.axvline(x=100, color='red', linestyle='--', alpha=0.5, label='Market MPI')
        ax.legend()
        
        plt.colorbar(scatter, label='Composite Score')
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()

    with tab3:
        # Monthly trends (if we have monthly data)
        if len(df_analysis['Month'].unique()) > 1:
            monthly_trends = df_analysis.groupby(['Group Name', 'Month']).agg({
                'RevPAR Index': 'mean',
                'Total Revenue STLY YoY %': 'mean',
                'Market Penetration Index %': 'mean'
            }).reset_index()
            
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            
            for i, metric in enumerate(['RevPAR Index', 'Total Revenue STLY YoY %', 'Market Penetration Index %']):
                for group in selected_groups:
                    group_data = monthly_trends[monthly_trends['Group Name'] == group]
                    axes[i].plot(group_data['Month'], group_data[metric], 
                               marker='o', label=f'Group {group}', linewidth=2)
                
                axes[i].set_title(f'{metric} Trends')
                axes[i].set_xlabel('Month')
                axes[i].set_ylabel(metric)
                axes[i].legend()
                axes[i].grid(True, alpha=0.3)
            
            plt.tight_layout()
            st.pyplot(fig)
            plt.clf()
        else:
            st.info("Monthly trend analysis requires data from multiple months.")

    with tab4:
        # Detailed performance table
        display_metrics = metrics_df[['RevPAR_Index', 'YoY_Revenue_Growth', 'MPI', 'ADR_Index',
                                     'Occupancy_vs_Market', 'Composite_Score', 'Consistency_Score',
                                     'Property_Count', 'Revenue_per_Property']].round(2)
        
        display_metrics.columns = ['RevPAR Index', 'YoY Revenue Growth (%)', 'MPI (%)', 'ADR Index',
                                 'Occupancy vs Market', 'Composite Score', 'Consistency Score',
                                 'Property Count', 'Revenue per Property']
        
        st.dataframe(display_metrics)

    # --- Property Drill-down ---
    st.subheader("🔍 Property-Level Analysis")
    
    selected_group_detail = st.selectbox("Select Group for Detailed Analysis", selected_groups)
    
    if selected_group_detail:
        group_properties = df_analysis[df_analysis['Group Name'] == selected_group_detail].copy()
        
        # Property performance table
        property_metrics = group_properties[['Listing Name', 'RevPAR Index', 'Total Revenue STLY YoY %', 
                                           'Market Penetration Index %', 'Occupancy %', 'Total ADR']].copy()
        property_metrics = property_metrics.sort_values('RevPAR Index', ascending=False)
        
        st.write(f"**Properties in Group {selected_group_detail}:**")
        st.dataframe(property_metrics)
        
        # Property performance distribution
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))
        
        # RevPAR Index distribution
        axes[0, 0].hist(group_properties['RevPAR Index'], bins=10, alpha=0.7, color='skyblue')
        axes[0, 0].set_title('RevPAR Index Distribution')
        axes[0, 0].set_xlabel('RevPAR Index')
        axes[0, 0].axvline(x=1, color='red', linestyle='--', alpha=0.5, label='Market Average')
        axes[0, 0].legend()
        
        # YoY Growth distribution
        axes[0, 1].hist(group_properties['Total Revenue STLY YoY %'], bins=10, alpha=0.7, color='lightgreen')
        axes[0, 1].set_title('YoY Revenue Growth Distribution')
        axes[0, 1].set_xlabel('Growth %')
        axes[0, 1].axvline(x=0, color='red', linestyle='--', alpha=0.5)
        
        # MPI distribution
        axes[1, 0].hist(group_properties['Market Penetration Index %'], bins=10, alpha=0.7, color='orange')
        axes[1, 0].set_title('MPI Distribution')
        axes[1, 0].set_xlabel('MPI %')
        axes[1, 0].axvline(x=100, color='red', linestyle='--', alpha=0.5, label='Market Average')
        axes[1, 0].legend()
        
        # Revenue vs RevPAR scatter
        axes[1, 1].scatter(group_properties['Total Revenue'], group_properties['RevPAR Index'], 
                          alpha=0.7, s=50)
        axes[1, 1].set_title('Revenue vs RevPAR Index')
        axes[1, 1].set_xlabel('Total Revenue')
        axes[1, 1].set_ylabel('RevPAR Index')
        axes[1, 1].axhline(y=1, color='red', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        st.pyplot(fig)
        plt.clf()

    # --- Data Quality Check ---
    with st.expander("🔍 Data Quality & Summary", expanded=False):
        st.write("**Data Summary:**")
        st.write(f"- Total Properties: {len(df_analysis)}")
        st.write(f"- Groups Analyzed: {len(selected_groups)}")
        st.write(f"- Date Range: June-December 2025")
        st.write(f"- Selected Months: {[month_names.get(m, m) for m in selected_months]}")
        st.write(f"- Aggregation Method: {aggregation_method}")
        
        st.write("**Group Statistics:**")
        group_stats = df_analysis.groupby('Group Name').agg({
            'Listing Name': 'count',
            'Total Revenue': ['sum', 'mean'],
            'RevPAR Index': ['mean', 'std'],
            'Total Revenue STLY YoY %': ['mean', 'std']
        }).round(2)
        
        st.dataframe(group_stats)

else:
    st.info("Please upload a CSV or Excel file to begin the group performance analysis.")
