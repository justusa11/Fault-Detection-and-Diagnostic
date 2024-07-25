# -*- coding: utf-8 -*-
"""
Created on Thu Jul 25 11:59:49 2024

@author: anoru
"""

##############################################################################
### PACKAGES ###
##############################################################################

import numpy as pd
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go

#Style sheet#
plt.style.use('ggplot')

#number of columns#
pd.set_option('display.max_columns', 25)

#%%

##############################################################################
### LOAD DATA ###
##############################################################################

# Specify the path to the Excel file
excel_file_path = 

df = pd.read_excel(excel_file_path)
print(df.head())
#%%
##############################################################################
### UNIQUE METERS ###
##############################################################################
# Count the number of unique meters in the dataset
unique_meters_count = df['meter_id'].nunique()

unique_meters_count
#%%

##############################################################################
### FAILURE VISUALIZATION ###
##############################################################################
# Set the style
sns.set_style("whitegrid")

# Plot the most common types of failures
plt.figure(figsize=(15, 8))
failure_counts = df['failure_type'].value_counts().head(10)
sns.barplot(y=failure_counts.index, x=failure_counts.values, palette="viridis")
plt.xlabel('Count')
plt.ylabel('Failure Type')
plt.title('Top 10 Most Common Failure Types')
plt.show()
#%%
##############################################################################
### FAILURES RESPONSE VISUALIZATION ###
##############################################################################
# Plot the most common actions taken in response to failures
plt.figure(figsize=(15, 8))
action_counts = df['action'].value_counts().head(10)
sns.barplot(y=action_counts.index, x=action_counts.values, palette="cividis")
plt.xlabel('Count')
plt.ylabel('Action Taken')
plt.title('Top 10 Most Common Actions Taken in Response to Failures')
plt.show()
#%%
# Plot the most common actions taken in response to failures
plt.figure(figsize=(15, 8))
action_counts = df['assessment'].value_counts().head(10)
sns.barplot(y=action_counts.index, x=action_counts.values, palette="cividis")
plt.xlabel('Count')
plt.ylabel('Action Taken')
plt.title('Most Common to Failures')
plt.show()

#%%
##############################################################################
### CONTACT TYPES ###
##############################################################################
# counts of each contact type
overall_contact_type_counts = df['contact_type'].value_counts().rename(index={1: 'Tlf/email', 2: 'Visit'})

# Plotting the results
overall_contact_type_counts.plot(kind='bar', color='seagreen', figsize=(10, 6))
plt.title("Overall Contact Types Distribution")
plt.xlabel("Contact Type")
plt.ylabel("Number of Occurrences")
plt.xticks(rotation=0)
plt.tight_layout()
plt.show()

#%%
##############################################################################
### REPORTED FAILURES ###
##############################################################################
# Resample the data by month and count the number of reported failures
monthly_failures = df.resample('M', on='contact_date').size()

# Plot the time series of reported failures
plt.figure(figsize=(18, 8))
monthly_failures.plot(marker='o', linestyle='-', color='b', linewidth=2)

# Set title and labels with increased font sizes
plt.title('Trends in the Number of Reported Failures Over Time', fontsize=18)
plt.xlabel('Date', fontsize=16)
plt.ylabel('Number of Reported Failures', fontsize=16)

# Increase tick label font sizes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.grid(True, which='both')
plt.tight_layout()
plt.show()

#%%
##############################################################################
### DESCRIPTIVE ANALYSIS ###
##############################################################################
# Plot the distribution of failure types count
plt.figure(figsize=(12, 6))
failure_types_counts_df = df['failure_type'].value_counts()
sns.barplot(y=failure_types_counts_df.index, x=failure_types_counts_df.values, palette="viridis")
plt.title('Distribution of Failure Types')
plt.xlabel('Count')
plt.ylabel('Failure Type')
plt.tight_layout()
plt.show()
#%%
# Correcting the data processing and visualization steps
# 1. Define colors and explode parameters
failure_types_counts_df = df['failure_type'].value_counts()
colors = sns.color_palette("pastel", n_colors=len(failure_types_counts_df))
explode = [0.1 if value > 10 else 0.02 for value in failure_types_counts_df]

# 2. Plotting
plt.figure(figsize=(14, 10))
wedges, texts, autotexts = plt.pie(failure_types_counts_df.values, labels=failure_types_counts_df.index, autopct='%1.1f%%', startangle=140, colors=colors, wedgeprops=dict(width=0.3), pctdistance=0.85, explode=explode)

# 3. Enhance chart appearance
plt.setp(autotexts, size=10, weight='bold', color="white")
plt.setp(texts, size=10)
plt.title('Percentage Distribution of Usable Failure Types', size=16)
plt.legend(failure_types_counts_df.index, loc="lower left", fontsize=10, bbox_to_anchor=(1, 0, 0.2, 1))

# Draw a circle in the center to make it a donut chart
centre_circle = plt.Circle((0,0),0.70,fc='white')
fig = plt.gcf()
fig.gca().add_artist(centre_circle)

plt.tight_layout()
plt.show()
#%%
##############################################################################
### PROCESSING STAGE ###
##############################################################################
# Define source, target, and value for the Sankey diagram
source = [0, 0, 1, 1, 2, 2, 3, 3]
target = [1, 2, 3, 4, 3, 4, 4, 5]
value = [382, 117, 265, 148, 227, 110, 117, 46]

# Define labels for each node
label = [
    "Original Data (382)", 
    "Excluding Other (SH/DHW) (265)",
    "Contact Type: Visit (227)",
    "Assessment: Proven (117)",
    "Action: Error Resolved (71)",
    "Data Loss (46)"
]

# Create the Sankey diagram
fig = go.Figure(go.Sankey(
    arrangement="snap",
    node=dict(
        pad=15,
        thickness=20,
        line=dict(color="black", width=0.5),
        label=label
    ),
    link=dict(
        source=source,
        target=target,
        value=value
    )
))

fig.update_layout(title_text="Data Processing", font_size=12)
fig.show()
#%%
##############################################################################
### DATA FILTERATION ###
##############################################################################
# Filter the data based on the given criteria
filtered_data = df[
    ~df['failure_type'].isin(["Other (DHW)", "Other (SH)"]) &
    (df['assessment'] == "Proven") &
    (df['action'] == "Error is resolved") &
    (df['contact_type'] == 2)  # Assuming 2 corresponds to "Visit"
]
# Displaying the first few rows of the filtered dataset
filtered_data.head()
#%%
# Getting the shape of the filtered dataframe
filtered_data_shape = filtered_data.shape
filtered_data_shape
#%%
##############################################################################
### COUNTS AFTER FILTERATION ###
##############################################################################
# Calculate the count distribution of failure types
failure_type_count = filtered_data['failure_type'].value_counts()

# Sort the counts in descending order
failure_type_count_sorted = failure_type_count.sort_values(ascending=False)
#%%
# Set the Viridis color palette
sns.set_palette('viridis')

# Plotting the count distribution as a horizontal bar chart with Viridis colors
plt.figure(figsize=(10, 6))
ax = failure_type_count_sorted.plot(kind='barh')
plt.gca().invert_yaxis()  # Invert the y-axis to start with the highest count at the top

plt.title("Ideal Failure Types Scenarios")
plt.xlabel("Count")
plt.ylabel("Failure Type")
plt.tight_layout()
plt.show()
#%%
##############################################################################
### FAILURES OVER TIME AFTER FILTERATION ###
##############################################################################
# Group by contact_date and count the number of reported failures on the filtered data
failures_over_time_filtered = filtered_data.resample('M', on='contact_date').size()

# Plot the aggregated data for the filtered dataset
plt.figure(figsize=(18, 8))
failures_over_time_filtered.plot(marker='o', linestyle='-', color='b', linewidth=2)

# Set title and labels with increased font sizes
plt.title('Reported Failures Over Time (Filtered Data)')
plt.xlabel('Date')
plt.ylabel('Number of Reported Failures', fontsize=16)

# Increase tick label font sizes
plt.xticks(fontsize=14)
plt.yticks(fontsize=14)

plt.grid(True, which='both')
plt.tight_layout()
plt.show()
#%%
##############################################################################
### SUMMER ###
##############################################################################
# Filter the data for June, July, and August 2022
summer_date_filter = (filtered_data['contact_date'] >= '2022-06-01') & (filtered_data['contact_date'] <= '2022-08-31')
summer_data = filtered_data[summer_date_filter]

# Get the counts of each failure type during these months
summer_failure_types = summer_data['failure_type'].value_counts()

summer_failure_types
#%%
# Set the red_h color palette
sns.set_palette('Reds_r')

# Plotting the count distribution as a horizontal bar chart with red_h colors
plt.figure(figsize=(10, 6))
ax = summer_failure_types.plot(kind='barh', color=sns.color_palette('Reds_r', len(summer_failure_types)))
plt.gca().invert_yaxis()  # Invert the y-axis to start with the highest count at the top

plt.title("Failure Types in Summer 2022 (June-August)")
plt.xlabel("Count")
plt.ylabel("Failure Type")
plt.tight_layout()
plt.show()
#%%
##############################################################################
### WINTER ###
##############################################################################
# Filter for the months December 2022, January, February 2023
winter_filter = ((filtered_data['contact_date'] >= '2022-12-01') & (filtered_data['contact_date'] <= '2022-12-31')) | \
                ((filtered_data['contact_date'] >= '2023-01-01') & (filtered_data['contact_date'] <= '2023-01-31')) | \
                ((filtered_data['contact_date'] >= '2023-02-01') & (filtered_data['contact_date'] <= '2023-02-28'))
winter_failures = filtered_data[winter_filter]

# Extract the failure types for the summer and winter months
winter_failure_types = winter_failures['failure_type'].value_counts()

winter_failure_types
#%%
# Plotting the count distribution as a horizontal bar chart with red_h colors
plt.figure(figsize=(10, 6))
ax = winter_failure_types.plot(kind='barh',
                               color=sns.color_palette('Blues_r', len(winter_failure_types)))
plt.gca().invert_yaxis()  # Invert the y-axis to start with the highest count at the top

plt.title("Failure Types in Winter 2022-2023 (December-February)")
plt.xlabel("Count")
plt.ylabel("Failure Type")
plt.tight_layout()
plt.show()
#%%
##############################################################################
### CONNECTION TYPE ###
##############################################################################
# Counting the number of unique failure types for each connection type in the filtered data
failure_types_by_connection = filtered_data.groupby('connection_type')['failure_type'].nunique()

failure_types_by_connection
#%%
# Renaming the connection_type values
filtered_data['connection_type'] = filtered_data['connection_type'].replace({1: 'Direct', 2: 'Indirect'})

# Getting the unique faults for each connection type
faults_by_connection = filtered_data.groupby('connection_type')['failure_type'].unique()

faults_by_connection
#%%
# Plotting the unique faults for each connection type
plt.figure(figsize=(12, 8))

# Create a countplot for failure types, split by connection type
sns.countplot(y="failure_type",
              hue="connection_type",
              data=filtered_data,
              order=filtered_data['failure_type'].value_counts().index,
              palette="viridis")

plt.title('Distribution of Failure Types by Connection Type', fontsize=16)
plt.xlabel('Count', fontsize=14)
plt.ylabel('Failure Type', fontsize=14)
plt.legend(title='Connection Type', fontsize=12)
plt.xticks(fontsize=12)
plt.yticks(fontsize=12)
plt.tight_layout()
plt.show()
