# -*- coding: utf-8 -*-
"""
Created on Tue Jan 16 11:18:21 2024

@author: anoru
"""

#%%
##############################################################################
### PACKAGES ###
##############################################################################
import seaborn as sns
import streamlit as st
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import statsmodels.api as sm # For estimating and interpreting statistical models
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
#%%
##############################################################################
### STREAMLIT SETUP ###
##############################################################################
# Streamlit UI setup
st.set_page_config(layout='wide', initial_sidebar_state='expanded')
st.title('Analysis of Fault Detection and Diagnosis')
#plt.style.use('seaborn-dark')
sns.set_style('darkgrid')

# Loading fault labels data
@st.cache_data
def load_fault_labels_data(file_object):
    #data_df = pd.read_pickle(r"")
    data_df['contact_date'] = pd.to_datetime(data_df['contact_date'])
    data_df['year'] = data_df['contact_date'].dt.year
    groups = data_df.groupby('meter_id')
    mask = groups['contact_date'].transform(lambda x: x.nunique() > 1) | ~data_df['meter_id'].duplicated(keep=False)
    data_df_no_duplicates = data_df[mask]
    filtered_data = data_df_no_duplicates[
        #(data_df_no_duplicates['contact_date'].dt.year == 2023) &
        (data_df_no_duplicates['assessment'] == "Proven") &
        data_df_no_duplicates['action'].isin(["Error is resolved", "Customer contacts VVS", "No action"]) &
        ~data_df_no_duplicates['Fault Labels'].isin(["Other (DHW)", "Other (SH)", "User behaviour"])
    ]
    return data_df, filtered_data

# Loading the SHM data
@st.cache_data
def load_2023(file_path):
    #data = pd.read_pickle(r"")
    return data

# Loading the BBR data
@st.cache_data
def load_bbr_data(file_path):
    #data = pd.read_pickle(r"")
    return data

# Loading the Weather data
@st.cache_data
def load_weather_data(file_path):
    #data = pd.read_pickle(r")
    data['average_time'] = data[['from', 'to']].mean(axis=1)
    data['from'] = pd.to_datetime(data['from']).dt.tz_convert('UTC')
    data.set_index('from', inplace=True)
    return data

#%%
##############################################################################
### UPLOADING DATA FILES ###
##############################################################################
# Upload the data files
uploaded_file_fault_labels = st.sidebar.file_uploader("Upload Fault Labels Data", type=["pkl"])
if uploaded_file_fault_labels:
    data_df, filtered_data = load_fault_labels_data(uploaded_file_fault_labels)
else:
    st.sidebar.warning("Please upload the Fault Labels data file to proceed.")
    st.stop()  # Stop further execution until file is uploaded

# Load SHM data
uploaded_file_shm_data = st.sidebar.file_uploader("Upload SHM Data File", type=["pkl"])
if uploaded_file_shm_data:
    shm_data = load_2023(uploaded_file_shm_data)  # Only one variable to catch the return value
    df = shm_data[shm_data['heat_meter_id'].isin(filtered_data['meter_id'])]
else:
    st.sidebar.warning("Please upload the SHM data file to proceed.")
    st.stop()  # Stop further execution until file is uploaded

# Load BBR data
uploaded_file_bbr_data = st.sidebar.file_uploader("Upload BBR Data File", type=["pkl"])
if uploaded_file_bbr_data:
    bbr_data = load_bbr_data(uploaded_file_bbr_data)
else:
    st.sidebar.warning("Please upload the BBR data file to proceed.")
    st.stop() # Stop further execution until file is uploaded

# Upload the Weather data file
uploaded_file_weather_data = st.sidebar.file_uploader("Upload Weather Data File", type=["pkl"])
if uploaded_file_weather_data:
    weather_data = load_weather_data(uploaded_file_weather_data)
else:

    st.sidebar.warning("Please upload the Weather data file to proceed.")
    st.stop()  # Stop further execution until file is uploaded
#%%
##############################################################################
### MATCHING FAULT LABELS TO CATEGORIES ###
##############################################################################
# Map fault labels to broader categories (example, adjust as necessary)
fault_label_categories = {
    'Radiator thermostat': 'Radiator',
    'Radiator valve': 'Radiator',
    'Towel dryer': 'Radiator',
    'Too small heating surfaces': 'Radiator',
    'Return thermostat set too high': 'UFH',
    'Telestat in UFH': 'UFH',
    'UFH shunt': 'UFH',
    'Error in UFH master controller': 'UFH',
    'UFH valve': 'UFH',
    'DHW regulator - heat exchanger': 'DHW',
    'Temperature regulator - storage tank': 'DHW',
    'Incorrectly set DHW temperature - heat exchanger': 'DHW',
    'Incorrectly set DHW temperature - storage tank': 'DHW'
 }
# Apply the mapping to create a new column for categories
filtered_data['Category'] = filtered_data['Fault Labels'].map(fault_label_categories)

# Drop rows with NaN values in 'Categorized Faults'
filtered_data = filtered_data.dropna(subset=['Category'])

# Filter data for each category
radiators_data = filtered_data[filtered_data['Category'] == 'Radiator']
ufh_data = filtered_data[filtered_data['Category'] == 'UFH']
dhw_data = filtered_data[filtered_data['Category'] == 'DHW']
#%%
##############################################################################
### CREATING THE INTERFACE ###
##############################################################################
# Sidebar configurations    
st.sidebar.header('SHM Proven')
#%%
# Sidebar for Category selection
st.sidebar.header("Category Selection")
category_options = ["Radiator", "UFH", "DHW"]
selected_category = st.sidebar.radio("Select Category", category_options)

# Function to get meter IDs for the selected fault label
def get_meter_ids_for_fault_label(fault_label, data):
    return data[data['Fault Labels'] == fault_label]['meter_id'].unique()

# Filter fault labels based on selected category
if selected_category == "Radiator":
    relevant_fault_labels = radiators_data['Fault Labels'].unique()
elif selected_category == "UFH":
    relevant_fault_labels = ufh_data['Fault Labels'].unique()
elif selected_category == "DHW":
    relevant_fault_labels = dhw_data['Fault Labels'].unique()

# Let the user select a fault label from the relevant_fault_labels
selected_fault_label = st.sidebar.selectbox(f"Select Fault Label for {selected_category}", relevant_fault_labels)

# For the selected fault label, get the associated meter IDs that are in both Excel and Parquet data
associated_meter_ids = get_meter_ids_for_fault_label(selected_fault_label, filtered_data)

# Now in the Streamlit sidebar, you can select a meter ID from the associated_meter_ids for the selected fault label
st.sidebar.subheader("Associated Meter IDs")
selected_meter_id = st.sidebar.selectbox(f"Select Meter ID for {selected_category}", associated_meter_ids)

#%%
# Date picker
st.sidebar.write("## Date Selection")
start_date = st.sidebar.date_input("Select start date:", pd.to_datetime("2018-01-31"))
end_date = st.sidebar.date_input("Select end date:", pd.to_datetime("2023-12-31"))
#%%
# Filter SHM data by selected meter ID
filtered_shm_data = df[df['heat_meter_id'] == selected_meter_id]

# Extract failure dates for the selected meter ID
failure_dates = filtered_data[filtered_data['meter_id'] == selected_meter_id][['failure_start_date', 'failure_end_date']]

# Convert naive start_date and end_date to timezone-aware datetimes matching the 'time_rounded' data
start_date = pd.to_datetime(start_date).tz_localize('UTC').tz_convert('Europe/Copenhagen')
end_date = pd.to_datetime(end_date).tz_localize('UTC').tz_convert('Europe/Copenhagen')

# Assuming 'Date' is the column that should be used as datetime index in filtered_shm_data
filtered_shm_data['Date'] = pd.to_datetime(filtered_shm_data['Date'])
filtered_shm_data.set_index('Date', inplace=True)

# Sort the DataFrame by datetime index
filtered_shm_data_sorted = filtered_shm_data.sort_index()

# Convert start_date and end_date to datetime, adjust if necessary
start_date = pd.to_datetime(start_date)
end_date = pd.to_datetime(end_date)

if filtered_shm_data_sorted.index.tzinfo is not None:
    start_date = start_date.tz_localize(None).tz_convert(filtered_shm_data_sorted.index.tzinfo)
    end_date = end_date.tz_localize(None).tz_convert(filtered_shm_data_sorted.index.tzinfo)
else:
    start_date = start_date.tz_localize(None)
    end_date = end_date.tz_localize(None)

# Slice the DataFrame by the specified date range
filtered_shm_data_by_date_range = filtered_shm_data_sorted.loc[start_date:end_date]

# Separate numeric and non-numeric columns for the sliced data
numeric_shm_data = filtered_shm_data_by_date_range.select_dtypes(include=np.number)
non_numeric_shm_data = filtered_shm_data_by_date_range.select_dtypes(exclude=np.number)

# Calculate daily averages for numeric data
filtered_daily_avg_numeric_data = numeric_shm_data.resample('D').mean()

# Handle non-numeric data
daily_non_numeric_data = non_numeric_shm_data.resample('D').first()

# Combining numeric and non-numeric data
filtered_daily_avg_shm_data = pd.concat([filtered_daily_avg_numeric_data, daily_non_numeric_data], axis=1)

# Ensure both indices are datetime and have the same time zone
filtered_daily_avg_shm_data.index = pd.to_datetime(filtered_daily_avg_shm_data.index, utc=True).tz_convert(None)
weather_data.index = pd.to_datetime(weather_data.index, utc=True).tz_convert(None)

# Ensure both indices have the same precision
filtered_daily_avg_shm_data.index = filtered_daily_avg_shm_data.index.astype('datetime64[ns]')
weather_data.index = weather_data.index.astype('datetime64[ns]')

# Merge the data
merged_data = pd.merge_asof(
    filtered_daily_avg_shm_data.sort_index(),
    weather_data.sort_index(),
    left_index=True, 
    right_index=True,
    direction='nearest'
)

# Filter merged data by date (if necessary)
filtered_merged_data = merged_data
#%%
##############################################################################
### CONVERTING TO DATETIME ###
##############################################################################
# Assuming 'Date' is the datetime column in df
df['Date'] = pd.to_datetime(df['Date'])
df.set_index('Date', inplace=True)

# Now you can resample
numeric_cols = df.select_dtypes(include=np.number)
daily_avg_numeric_data = numeric_cols.resample('D').mean()

# Handling Non-Numeric Data
non_numeric_cols = df.select_dtypes(exclude=np.number)
daily_avg_non_numeric_data = non_numeric_cols.resample('D').first()

# Combining Numeric and Non-Numeric Data
daily_avg_data = pd.concat([daily_avg_numeric_data, daily_avg_non_numeric_data], axis=1)
#%%
##############################################################################
### LOGARITHMIC TREND LINE  ###
##############################################################################
def calculate_log_trendline(x, y):
    # Replace zero or negative values with NaN
    x = np.where(x <= 0, np.nan, x)
    y = np.where(y <= 0, np.nan, y)
    
    # Remove any NaN values
    mask = ~np.isnan(x) & ~np.isnan(y)
    x = x[mask]
    y = y[mask]

    # Check if there are enough data points for regression
    if len(x) > 1 and len(y) > 1:
        # Fit the logarithmic trend line
        coeffs = np.polyfit(np.log(x), y, 1)
        return coeffs
    else:
        return [np.nan, np.nan]

# Calculate coefficients for each trend line
coeffs1 = calculate_log_trendline(filtered_merged_data['mean_temp'], filtered_merged_data['heat_energy_kwh_demand'])
coeffs2 = calculate_log_trendline(filtered_merged_data['volume_m3_demand'], filtered_merged_data['delta_T'])
coeffs3 = calculate_log_trendline(filtered_merged_data['volume_m3_demand'], filtered_merged_data['heat_energy_kwh_demand'])
coeffs4 = calculate_log_trendline(filtered_merged_data['heat_energy_kwh_demand'], filtered_merged_data['return_temperature'])

# Function to generate trendline data remains the same
def generate_trendline_data(x, coeffs):
    x_vals = np.linspace(np.min(x), np.max(x), 100)
    y_vals = coeffs[0] * np.log(x_vals) + coeffs[1]
    return x_vals, y_vals

# Generate trendline data for each subplot
x_trend1, y_trend1 = generate_trendline_data(filtered_merged_data['mean_temp'], coeffs1)
x_trend2, y_trend2 = generate_trendline_data(filtered_merged_data['volume_m3_demand'], coeffs2)
x_trend3, y_trend3 = generate_trendline_data(filtered_merged_data['volume_m3_demand'], coeffs3)
x_trend4, y_trend4 = generate_trendline_data(filtered_merged_data['heat_energy_kwh_demand'], coeffs4)
#%%
##############################################################################
### MIN MAX RANGE###
##############################################################################    
#return_temperature = filtered_merged_data['return_temperature']
#consumption_kwh = filtered_merged_data['heat_energy_kwh_demand']

# Define min and max range
#min_temp = 25
#max_temp = 70  
   
#%%
##############################################################################
### FILTERING PERIOD OF INTERVENTION ###
##############################################################################
# Check the type of the index
print("Index type:", type(filtered_merged_data.index))

# If the index is not a DateTimeIndex, set it correctly
if not isinstance(filtered_merged_data.index, pd.DatetimeIndex):
    # Convert the 'Date' column to datetime and set it as the index
    filtered_merged_data['Date'] = pd.to_datetime(filtered_merged_data['Date'])
    filtered_merged_data.set_index('Date', inplace=True)

# Now, filtered_merged_data should have a DateTimeIndex
# Extract the date component from the index
filtered_merged_data['Index_Date'] = filtered_merged_data.index.date

# Debugging
print("Newly Added Index_Date column:", filtered_merged_data['Index_Date'].head())

# Prepare the failure_dates_set
failure_dates_set = set()
for _, row in failure_dates.iterrows():
    start_date = row['failure_start_date'].date() if not pd.isnull(row['failure_start_date']) else None
    end_date = row['failure_end_date'].date() if not pd.isnull(row['failure_end_date']) else None
    if start_date and end_date:
        failure_period = pd.date_range(start=start_date, end=end_date)
        failure_dates_set.update(failure_period.date)

# Add a column to indicate whether the data point is in the failure period
filtered_merged_data['Is_Failure'] = filtered_merged_data['Index_Date'].isin(failure_dates_set)
      
#%%
##############################################################################
### CREATING THE SCATTER PLOT ###
##############################################################################
st.subheader("SHM DATA ANALYSIS")

# Create a subplot figure with 1 row and 4 columns
fig = make_subplots(rows=1, cols=4, subplot_titles=('Energy kWh vs. Outdoor Temp ', 
                                                    'Cooling °C vs. Volume m³', 
                                                    'Energy kWh vs. Volume m³', 
                                                    'Return Temp vs. Energy kWh'))

# Define colors based on 'Is_Failure' values
colors = ['red' if is_failure else 'royalblue' for is_failure in filtered_merged_data['Is_Failure']]
#%%
##############################################################################
### SCATTER PLOT 0 ###
##############################################################################
# Plotting for the first subplot (as an example)
x_data = filtered_merged_data['mean_temp']
y_data = filtered_merged_data['heat_energy_kwh_demand']
frames = len(x_data)

# First plot: Outdoor Temp vs. Consumption kWh
fig.add_trace(
    go.Scatter(
        x=filtered_merged_data['mean_temp'], 
        y=filtered_merged_data['heat_energy_kwh_demand'], 
        mode='markers',
        marker=dict(color=colors),
        name='Outdoor Temp vs. Energy'
    ),
    row=1, col=1
)
#%%
##############################################################################
### SCATTER PLOT 1 ###
##############################################################################
# Plotting for the Second subplot: Cooling °C vs. Volume m³
x_data = filtered_merged_data['volume_m3_demand']
y_data = filtered_merged_data['delta_T']

# Second plot: Cooling °C vs. Volume m³
fig.add_trace(
    go.Scatter(
        x=filtered_merged_data['volume_m3_demand'], 
        y=filtered_merged_data['delta_T'], 
        mode='markers',
        marker=dict(color=colors),
        name='Cooling vs. Volume'
    ),
    row=1, col=2
)

#%%
##############################################################################
### SCATTER PLOT 2 ###
##############################################################################
# Plotting for the Third subplot: Consumption kWh vs. Volume m³
x_data = filtered_merged_data['volume_m3_demand']
y_data = filtered_merged_data['heat_energy_kwh_demand']
dates = filtered_merged_data.index  # Assuming this is a datetime index

# Third plot: Consumption kWh vs. Volume m³
fig.add_trace(
    go.Scatter(
        x=filtered_merged_data['volume_m3_demand'], 
        y=filtered_merged_data['heat_energy_kwh_demand'], 
        mode='markers',
        marker=dict(color=colors),
        name='Energy vs. Volume'
    ),
    row=1, col=3
)

#%%
##############################################################################
### SCATTER PLOT 3 ###
##############################################################################
# Plotting for the Fourth subplot: Consumption kWh vs. Return
x_data = filtered_merged_data['heat_energy_kwh_demand']
y_data = filtered_merged_data['return_temperature']

# Fourth plot: Consumption kWh vs. Return Temperature
fig.add_trace(
    go.Scatter(
        x=filtered_merged_data['heat_energy_kwh_demand'], 
        y=filtered_merged_data['return_temperature'], 
        mode='markers',
        marker=dict(color=colors),
        name='Energy vs. Return Temp'
    ),
    row=1, col=4
)

# Adjust the layout
plt.tight_layout()

# Display the plots in Streamlit
#st.pyplot(fig)

# Adding the trend line to each subplot
fig.add_trace(go.Scatter(x=x_trend1, y=y_trend1, mode='lines', name='Log Trend', line=dict(color='black', width=2)), row=1, col=1)
fig.add_trace(go.Scatter(x=x_trend2, y=y_trend2, mode='lines', name='Log Trend', line=dict(color='black', width=2)), row=1, col=2)
fig.add_trace(go.Scatter(x=x_trend3, y=y_trend3, mode='lines', name='Log Trend', line=dict(color='black', width=2)), row=1, col=3)
fig.add_trace(go.Scatter(x=x_trend4, y=y_trend4, mode='lines', name='Log Trend', line=dict(color='black', width=2)), row=1, col=4)

# Update layout if needed
fig.update_layout(height=600, width=1200, title_text="Measurements and Outdoor Temperature Correlations")

# Display the figure in Streamlit
st.plotly_chart(fig, use_container_width=True)
#%%
##############################################################################
### SUBPLOTS FOR SUPPLY, RETURN, DELTA T, VOLUME, AND ENERGY kWh ###
##############################################################################
st.subheader("System Performance Before and After Intervention")

# Two Column Layout
col1, col2 = st.columns([2, 1])

with col1:
    # Stacked Plot: Supply, Return, Volume, and Energy
    fig, axs = plt.subplots(4, 1, figsize=(8, 8), dpi=100)
    
    # Step 1: Create a Custom Colormap
    cmap = plt.cm.inferno
    colors = cmap(np.linspace(0, 1, cmap.N))
    colors[0] = colors[-1] = [0, 0, 0, 1]  # Set the extremes to black
    custom_cmap = mcolors.LinearSegmentedColormap.from_list('inferno', colors)

    # Use the original normalization
    norm = mcolors.Normalize(vmin=filtered_merged_data['mean_temp'].min(),
                             vmax=filtered_merged_data['mean_temp'].max())
    
   # Iterate over each subplot to add failure start and end date lines
    for ax in axs:
        # Get current y-axis limits for the subplot
        ymin, ymax = ax.get_ylim()
        
        # Add vertical lines for each failure date
        for _, row in failure_dates.iterrows():
            failure_start_date = row['failure_start_date']
            failure_end_date = row['failure_end_date']

            if not pd.isnull(failure_start_date):
                ax.axvline(x=failure_start_date, ymin=0, ymax=1, color='black', linestyle='--', linewidth=1, label='Failure Start Date')
            
            if not pd.isnull(failure_end_date):
                ax.axvline(x=failure_end_date, ymin=0, ymax=1, color='black', linestyle='--', linewidth=1, label='Failure End Date')
    
    # Plot Supply and Return in the same subplot
    axs[0].plot(filtered_merged_data.index, filtered_merged_data['supply_temperature'], label='Supply (°C)', color='tab:red', linewidth=0.8)
    axs[0].plot(filtered_merged_data.index, filtered_merged_data['return_temperature'], label='Return (°C)', color='tab:blue', linewidth=0.8)
    axs[0].set_ylabel('Temperature (°C)', fontsize=8)
    axs[0].tick_params(axis='y', labelsize=8)
    axs[0].legend(loc='upper left', fontsize=8)
    axs[0].set_xticklabels([])  # Remove x-axis labels
    
    # Place legend outside the subplot
    axs[0].legend(loc='upper left', bbox_to_anchor=(1, 1), fontsize=8)
   
   # Lowess line and Scatter plot for Delta T with outdoor temperature colormap
    y = filtered_merged_data['delta_T']
    x = range(len(y))
    smoothed = sm.nonparametric.lowess(y, x, frac=0.1)
    axs[1].plot(filtered_merged_data.index[:len(smoothed)], smoothed[:, 1], color='k', linewidth=0.5)
    
    sc = axs[1].scatter(filtered_merged_data.index, filtered_merged_data['delta_T'],
                        c=filtered_merged_data['mean_temp'],
                        cmap=cmap, norm=norm, edgecolor='none', alpha=0.6, s=15)
    axs[1].set_ylabel('delta T (Δt)', fontsize=8)
    axs[1].tick_params(axis='y', labelsize=8)
    axs[1].set_xticklabels([])  # Remove x-axis labels

    cb = plt.colorbar(sc, ax=axs[1], orientation='vertical', fraction=0.02)
    cb.set_label('Outdoor Temp (°C)', fontsize=8)
    cb.ax.tick_params(labelsize=8)

    # Lowess line and Scatter plot for Volume with outdoor temperature colormap
    y = filtered_merged_data['volume_m3_demand']
    x = range(len(y))
    smoothed = sm.nonparametric.lowess(y, x, frac=0.1)
    axs[2].plot(filtered_merged_data.index[:len(smoothed)], smoothed[:, 1], color='k', linewidth=0.5)
    
    sc = axs[2].scatter(filtered_merged_data.index, filtered_merged_data['volume_m3_demand'],
                        c=filtered_merged_data['mean_temp'],
                        cmap=cmap, norm=norm, edgecolor='none', alpha=0.6, s=15)
    axs[2].set_ylabel('Volume (m³)', fontsize=8)
    axs[2].tick_params(axis='y', labelsize=8)
    axs[2].set_xticklabels([])  # Remove x-axis labels

    cb = plt.colorbar(sc, ax=axs[2], orientation='vertical', fraction=0.02)
    cb.set_label('Outdoor Temp (°C)', fontsize=8)
    cb.ax.tick_params(labelsize=8)

    # Scatter plot for Energy with outdoor temperature colormap
    y = filtered_merged_data['heat_energy_kwh_demand']
    x = range(len(y))
    smoothed = sm.nonparametric.lowess(y, x, frac=0.1)
    axs[3].plot(filtered_merged_data.index[:len(smoothed)], smoothed[:, 1], color='k', linewidth=0.5)
    
    sc = axs[3].scatter(filtered_merged_data.index, filtered_merged_data['heat_energy_kwh_demand'],
                        c=filtered_merged_data['mean_temp'],
                        cmap=cmap, norm=norm, edgecolor='none', alpha=0.6, s=15)
    axs[3].set_ylabel('Energy (kWh)', fontsize=8)
    axs[3].tick_params(axis='y', labelsize=8)

    cb = plt.colorbar(sc, ax=axs[3], orientation='vertical', fraction=0.02)
    cb.set_label('Outdoor Temp (°C)', fontsize=8)
    cb.ax.tick_params(labelsize=8)

    # Set common X-axis label for the bottom subplot
    axs[3].set_xlabel('Date', fontsize=8)
    axs[3].tick_params(axis='x', labelsize=6, rotation=45)

    # Adjust layout
    fig.tight_layout()
    st.pyplot(fig, use_container_width=True)
    pass

with col2:
    st.markdown('### Meter Information')
        
    # Extracting the specific information about the heat meter ID
    meter_info = filtered_data[
        (filtered_data['meter_id'] == selected_meter_id) &
        (filtered_data['Fault Labels'] == selected_fault_label)
    ]
        
    # Displaying the information
    for index, row in meter_info.iterrows():
        st.write("Contact Date:", row['contact_date'].strftime('%Y-%m-%d'))
        st.write("Fault Labels:", row['Fault Labels'])
        st.write("Contact Type:", row['contact_type'])
        st.write("Assessment:", row['assessment'])
        st.write("Failure Start Date:", row['failure_start_date'].strftime('%Y-%m-%d') if pd.notnull(row['failure_start_date']) else 'N/A')
        st.write("Failure End Date:", row['failure_end_date'].strftime('%Y-%m-%d') if pd.notnull(row['failure_end_date']) else 'N/A')
        st.write("Action:", row['action'])
        st.write("Heating System:", row['heating system'])
        st.write("Employee:", row['employee'])
        st.write("Description:", row['description'])
        st.write("---")
                
    # Attempting to get BBR information for the selected heat meter ID
        bbr_info = bbr_data[bbr_data['heat_meter_id'] == selected_meter_id]
        st.markdown('### BBR Information')

    # Check if the resulting DataFrame is empty
        if not bbr_info.empty:
            # If not empty, extract the first row's information
            bbr_info = bbr_info.iloc[0]
            st.write(f"Total Area: {bbr_info['unit_total_area']} sqm")  
            st.write(f"Building Year: {bbr_info['bldg_constrcution_year']}") 
            st.write(f"Renovation Year: {bbr_info['bldg_conversion_year']}")
        else:
            # If empty, handle the case (e.g., display a message or use a default value)
            st.error(f"No BBR data found for meter ID {selected_meter_id}")