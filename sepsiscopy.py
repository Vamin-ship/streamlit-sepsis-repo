import streamlit as st
import requests 
import os
import pandas as pd
import base64
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import pickle
import shap
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler,LabelEncoder
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.neighbors import KNeighborsClassifier
from imblearn.over_sampling import RandomOverSampler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import roc_curve, roc_auc_score,RocCurveDisplay
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, precision_score, recall_score, f1_score, make_scorer
from sklearn.metrics import roc_curve, auc, confusion_matrix
from xgboost import XGBClassifier
import xgboost as xgb




# Page configuaration
st.set_page_config(page_title="Sepsis Project",page_icon=":hospital:",layout='wide')
st.cache(suppress_st_warning=True)(lambda: None)
# Custom CSS to position the title at the top-center of the page
st.markdown("""
    <style>
        .title-wrapper {
            display: flex;
            flex-direction: column;
            align-items: center;
            justify-content: center;
            margin-top: -80px;
            margine-bottom: 40px;
        }
        .title{
            color:black;
            
        }
        .subtitle {
            color: #666;
            fornt-size:20px;
            margine-top:10px;
            text-align: center;
        }
        .right-align {
            text-align: right;
        }
        
    </style>
    """, unsafe_allow_html=True)

 #Read the image file as binary data
with open("Atrius_Health_Logo.jpg", "rb") as img_file:
# Encode the binary data as base64
    encoded_image = base64.b64encode(img_file.read()).decode('utf-8')
 # Define width and height for the image
image_width = 70 # Adjust the width as needed
image_height = 70  # Adjust the height as needed

# Embed the base64 encoded image in HTML with specified width and height
html_code = f"""
<div class='title-wrapper'>
    <h1 class='title'><img src='data:image/jpg;base64,{encoded_image}' alt='Atrius Health Logo' width='{image_width}' height='{image_height}'> Sepsis Machine Learning Predictive Dashboard</h1>
    <p class='subtitle' style='font-size: 24px; font-weight: bold;'>Predictive Modeling for Early Sepsis Identification</p>
</div>
"""

st.markdown(html_code, unsafe_allow_html=True) 
# Load the balanced and shuffled dataset directly
df = pd.read_csv("balanced_shuffled_sepsis_dataset.csv")

# Print columns to verify their names and structure
print("Columns of 'data':", df.columns)
# Convert values in 'Gender' column
df['Gender'] = df['Gender'].map({1: 'Male', 0: 'Female'})

# Convert values in 'SepsisLabel' column
df['SepsisLabel'] = df['SepsisLabel'].map({1: 'Sepsis', 0: 'No Sepsis'})

# Convert numerical values to categorical labels in 'Unit1' and 'Unit2' columns
df['Unit1'] = df['Unit1'].map({1: 'Medical ICU', 0: 'Surgical ICU'}).fillna('Other ICU')
df['Unit2'] = df['Unit2'].map({1: 'Surgical ICU', 0: 'Medical ICU'}).fillna('Other ICU')

# Print unique values to check
print("Unique values in 'Gender' column:", df['Gender'].unique())
print("Unique values in 'SepsisLabel' column:", df['SepsisLabel'].unique())
print("Unique values in 'Unit1' column:", df['Unit1'].unique())
print("Unique values in 'Unit2' column:", df['Unit2'].unique())
    # Display the information about the dataset
print("\nInformation about the dataset:")
print(df.info())
# Display descriptive statistics for numerical columns
print("\nDescriptive statistics for numerical columns:")
print(df.describe().T)

# Calculate percentage of missing values for df
missing_percentage = (df.isnull().sum() / len(df)) * 100
# Print the percentage of missing values
print("\nPercentage of missing values ")
print(missing_percentage)
# Calculate missing values percentage for each feature
missing_perc = (df.isnull().sum() / len(df)) * 100

# Create a DataFrame to hold missing values information
df_miss = pd.DataFrame({'feature': missing_perc.index, 'missing_percent': missing_perc.values})

# Sort DataFrame by missing percentage in descending order
df_miss = df_miss.sort_values(by='missing_percent', ascending=False)

# Set the font scale
sns.set(font_scale=1.5)  # Adjust the scale as needed

plt.figure(figsize=(12, 14))
ax = sns.barplot(x='missing_percent', y='feature', data=df_miss, color='#22577a')

# Bolden x-axis and y-axis labels
ax.set_xlabel('Missing Percentage', fontsize=16, fontweight='bold')
ax.set_ylabel('Feature', fontsize=16, fontweight='bold')

# Set title with bold font
plt.title('Missing Data Percentage by Feature', fontsize=16, fontweight='bold')

# Set y-axis tick labels to bold
for label in ax.get_yticklabels():
    label.set_fontweight('bold')

# Set background color to white
ax.set_facecolor('white')

# Remove gridlines
plt.grid(False)
# Calculate average age for females with sepsis
ave_age_sepsis_female = df.loc[(df['SepsisLabel'] == 'Sepsis') & (df['Gender'] == 'Female'), 'Age'].mean()

# Calculate average age for males with sepsis
ave_age_sepsis_male = df.loc[(df['SepsisLabel'] == 'Sepsis') & (df['Gender'] == 'Male'), 'Age'].mean()

# Calculate total number of patients with sepsis
total_sepsis_patients = df[df['SepsisLabel'] == 'Sepsis'].shape[0]

# Filter rows based on the updated 'SepsisLabel' column
df_no_sepsis = df[df['SepsisLabel'] == 'No Sepsis']
df_sepsis = df[df['SepsisLabel'] == 'Sepsis']

# Add an empty space to push the next content to the right
st.markdown('<div style="text-align: right; font-weight: bold; font-style: italic; color: #168aad; margin-bottom: 10px; margin-right: 50px;">Timeframe: 04-15-2023<div style="font-weight: bold; font-style: italic; color: #168aad;">to 06-01-2024</div> Data Source: Hospital DB</div>', unsafe_allow_html=True)
st.markdown('<div style="text-align: right; font-weight: bold; font-style: italic; color: #168aad; margin-bottom: 10px; margin-right: 50px;">Owner: Sunita Amin</div>', unsafe_allow_html=True)
# Set up a two-column layout
col1, col2 = st.columns([1, 2])

# Display the image in the first column
with col1:
    st.image("images.png", width=300)  # Adjust the width as needed

# Display the content in the second column
with col2:
    st.markdown('<div style="padding-left: 20px;">', unsafe_allow_html=True)
    st.markdown("""
  Sepsis is a severe medical condition characterized by the body's overwhelming response to an infection, often leading to tissue damage, organ failure, and even death if not promptly treated.
   If you suspect sepsis, seek medical attention immediately.
   """)
    st.markdown('</div>', unsafe_allow_html=True)
    
    st.markdown('<div style="padding-left: 20px;">', unsafe_allow_html=True)
    st.markdown("""
Dashboard summary:
Our dashboard provides a comprehensive analysis of vital signs and laboratory values, aiding in the early detection and management of sepsis. Through interactive visualizations and data insights, healthcare professionals can efficiently monitor and assess patient status, facilitating timely interventions and improved patient outcomes.
    """)
    st.markdown('</div>', unsafe_allow_html=True)
    # Arrange the info boxes horizontally
col1, col2, col3, col4 = st.columns(4)

# Define the style for the info boxes
info_box_style = "padding: 10px; color: white; background-color: #3498db; border-color: #2980b9;"

# Apply custom CSS styles to the select boxes for ICU unit and Sepsis filters
st.markdown("""
    <style>
        /* Style for select box options */
        .st-df option, .st-df-placeholder, .st-df-value, .stSelectbox option, .stSelectbox .Select__placeholder, .stSelectbox .Select__single-value {
            background-color: #3498db !important; /* Background color inside select boxes */
            color: white !important; /* Text color inside select boxes */
        }
    </style>
""", unsafe_allow_html=True)

# Create boxes for the average age and total number of patients
with col1:
    st.markdown(f"<div style='{info_box_style}'>Ave Age - Females: {ave_age_sepsis_female:.2f}</div>", unsafe_allow_html=True)
with col2:
    st.markdown(f"<div style='{info_box_style}'>Ave Age - Males: {ave_age_sepsis_male:.2f}</div>", unsafe_allow_html=True)
with col3:
    st.markdown(f"<div style='{info_box_style}'>Total Sepsis Patients: {total_sepsis_patients}</div>", unsafe_allow_html=True)

with col4:
    # Add custom-styled select box for ICU unit filter
    icu_unit_filter = st.selectbox(
        "**Filter by ICU Unit**",
        ["Surgical ICU", "Medical ICU"],
        key="icu_unit_filter",
        help="Select ICU Unit"
        )
    
    # Add custom-styled select box for Sepsis filter
    sepsis_filter = st.selectbox(
        "**Filter by Sepsis**",
        ["Sepsis", "No Sepsis"],
        key="sepsis_filter",
        help="Select Sepsis"
        )
# Define the data for the box plot
data = df[["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2"]]

# Create a list to hold box plot traces
box_traces = []

# Define colors for the box plots
colors = ['#588157', '#f6bd60', '#00b4d8', '#22577a', '#e76f51', '#0077b6', '#22333b', '#778da9']

# Iterate over each column and create a box plot trace
for i, column in enumerate(data.columns):
    box_trace = go.Box(
        y=data[column],
        name=column,
        marker=dict(color=colors[i])  # Set the color for this box plot
    )
    box_traces.append(box_trace)

# Create layout for the box plot
layout = go.Layout(
    title=dict(text="<b>Box Plot of Vital Signs</b>", font=dict(size=20, color='black')),  # Make the title bold and bigger
    title_x=0.5,  # Center the title horizontally
    title_y=0.9,  # Set the title position at the top
    yaxis=dict(title="<b>Values for Vital Signs</b>", showgrid=False, showline=True, linewidth=1, linecolor='black', titlefont=dict(size=18, color='black')),  # Make y-axis title bold and remove grid lines
    xaxis=dict(
        title="<b>Parameter</b>", showgrid=False, showline=True, linewidth=1, linecolor='black',
        titlefont=dict(size=18, color='black'),
        tickfont=dict(size=14, color='black', family='Arial Black, sans-serif')  # Use bold font family for x-axis feature names
    ),
    showlegend=False,  # Hide legend
    height=400,  # Adjust the height of the chart
    width=600,   # Adjust the width of the chart
)

# Create figure for box plot
box_fig = go.Figure(data=box_traces, layout=layout)

# Add description for box plot
box_description = """
<div style="font-size: 13px">
<strong>Box Plot of Vital Signs</strong><br><br>
This chart displays a box plot visualization of vital signs. Each box represents the distribution of values for a specific vital sign parameter, such as blood pressure or heart rate. The box extends from the lower to the upper quartile values, with a line inside representing the median value. The whiskers extend to show the range of the data, excluding outliers. This visualization provides insights into the distribution and variability of vital signs among the dataset.
</div>
"""

# Define correlation features
correlation_features = ["SBP", "MAP", "DBP", "HCO3", "pH", "PaCO2", "Bilirubin_direct",
                        "Bilirubin_total", "Hct", "Hgb"]

# Calculate correlation matrix
cor_mat = df[correlation_features].corr()

# Create heatmap trace
heatmap_trace = go.Heatmap(
    z=cor_mat.values,
    x=cor_mat.columns,
    y=cor_mat.columns,
    zmin=-1,
    zmax=1,
    colorscale='YlGnBu',  # Using YlGnBu colorscale
    showscale=False  # Disable the color scale legend
)

# Create annotations
annotations = []
for i in range(len(cor_mat)):
    for j in range(len(cor_mat)):
        annotations.append(
            dict(
                text="{:.2f}".format(cor_mat.values[i][j]),
                x=cor_mat.columns[j],
                y=cor_mat.columns[i],
                xref='x1',
                yref='y1',
                font=dict(color='white' if abs(cor_mat.values[i][j]) > 0.5 else 'black'),
                showarrow=False,
            )
        )

# Define layout for correlation heatmap
heatmap_layout = go.Layout(
    title="<b>Correlational Analysis</b>",
    title_x=0.5,  # Center the title horizontally
    title_y=0.9,  # Set the title position at the top
    titlefont=dict(size=20, color='black'),
    xaxis=dict(
        title="<b>Features</b>", showgrid=False, showline=True, linewidth=1, linecolor='black',
        titlefont=dict(size=18, color='black'),
        tickfont=dict(size=14, color='black', family='Arial Black, sans-serif')  # Use bold font family for x-axis feature names
    ),
    yaxis=dict(
        title="<b>Features</b>", showgrid=False, showline=True, linewidth=1, linecolor='black',
        titlefont=dict(size=18, color='black'),
        tickfont=dict(size=14, color='black', family='Arial Black, sans-serif')  # Use bold font family for y-axis feature names
    ),
    annotations=annotations,
    width=600,  # Adjust the width of the chart
    height=400,  # Adjust the height of the chart
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    showlegend=False
)

# Create figure for correlation heatmap
heatmap_fig = go.Figure(data=heatmap_trace, layout=heatmap_layout)

# Add description for heatmap
heatmap_description = """
<div style="font-size: 13px">
<strong>Correlational Analysis</strong><br><br>
This heatmap illustrates the correlation between different features related to sepsis. Each cell represents the correlation coefficient between two features, with warmer colors indicating stronger positive correlations and cooler colors indicating stronger negative correlations. This analysis helps identify potential relationships and dependencies among the features in the dataset.
</div>
"""

# Display the box plot and correlation heatmap side by side using Streamlit's columns
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(box_fig)
    st.markdown(box_description, unsafe_allow_html=True)

with col2:
    st.plotly_chart(heatmap_fig)
    st.markdown(heatmap_description, unsafe_allow_html=True)

# Custom color palette
custom_palette = {"No Sepsis": '#168aad', "Sepsis": '#ffbf69'}

# Plot gender distribution
def plot_gender_distribution(df, gender_col, sepsis_label_col):
    gender = gender_col.copy()
    gender[gender == "Female"] = "Female"
    gender[gender == "Male"] = "Male"

    # Count occurrences of each gender and sepsis label combination
    counts = df.groupby([gender, sepsis_label_col]).size().unstack().fillna(0)

    # Create traces for each sepsis label
    traces = []
    for label, color in custom_palette.items():
        if label in counts:
            trace = go.Bar(
                x=counts.index,
                y=counts[label],
                name=label,
                marker=dict(color=color, line=dict(width=0.5)),
                text=[f'{count / total * 100:.2f}%' for count, total in zip(counts[label], counts.sum(axis=1))],
                textposition='outside',
                hoverinfo='text',
                hovertext=[f'{label}: {count} ({count / total * 100:.2f}%)' for count, total in zip(counts[label], counts.sum(axis=1))],
                textfont=dict(size=10)
            )
            traces.append(trace)

    layout = go.Layout(
        title=dict(
            text="<b>Gender Distribution</b>",
            xanchor="center",
            yanchor="top",
            x=0.5,
            y=0.98,
            pad=dict(b=10),
            font=dict(size=20, color='black')
        ),
        xaxis=dict(title="<b>Gender</b>", tickfont=dict(size=12, color='black', family='Arial Black, sans-serif'),
                   showline=True, linewidth=1, linecolor='black', titlefont=dict(size=18, color='black')),
        yaxis=dict(title="<b>Count</b>", showline=True, linewidth=1, linecolor='black', titlefont=dict(size=18, color='black')),
        barmode='group',
        plot_bgcolor='rgba(0,0,0,0)',
        margin=dict(t=50),
        legend=dict(title="<b>Sepsis Label</b>", font=dict(size=14)),
        hovermode='x'
    )

    fig = go.Figure(data=traces, layout=layout)
    fig.update_layout(height=450, width=450)

    return fig

# Function to plot ICU distribution
def plot_icu_distribution(df, unit1_col, unit2_col):
    Unit1 = unit1_col[unit1_col == 'Medical ICU'].count()
    Unit2 = unit2_col[unit2_col == 'Surgical ICU'].count()
    total = len(unit1_col) - len(unit1_col[(unit1_col == 'Other ICU') & (unit2_col == 'Other ICU')])

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=["Medical ICU", "Surgical ICU"],
        y=[Unit1, Unit2],
        marker_color=['#a4c3b2', '#738290'],
        text=[f'{(v/total)*100:.2f}%' for v in [Unit1, Unit2]],
        textposition='outside',  # Change text position to outside (above the bars)
        textfont=dict(size=12),  # Set font size for text
        width=[0.4, 0.4]  # Adjust the width of the bars
    ))

    fig.update_layout(
        title=dict(
            text="<b>Medical ICU vs Surgical ICU</b>",
            xanchor="center",
            yanchor="top",
            x=0.5,
            y=0.98,  # Adjust the y position
            pad=dict(b=10),  # Add padding at the bottom
            font=dict(size=20, color='black')  # Set font size to 20 and color to black
        ),
        xaxis=dict(title="<b>ICU Unit</b>", tickfont=dict(size=12, color='black', family='Arial Black, sans-serif'), showline=True, linewidth=1, linecolor='black', titlefont=dict(size=18, color='black')),  # Make x-axis label bold and add line
        yaxis=dict(title="<b>Number of Patients</b>", tickfont=dict(size=12, color='black'), showline=True, linewidth=1, linecolor='black', titlefont=dict(size=18, color='black')),  # Make y-axis label bold and add line
        height=450,
        width=450 ,
        plot_bgcolor='white',
        xaxis_tickfont=dict(size=12),
        yaxis_tickfont=dict(size=12),
        font=dict(size=20, color='black')
    )

    return fig




# Side-by-side display of gender and ICU distribution charts
col1, col2 = st.columns(2)

# Plot gender distribution
with col1:
    fig_gender = plot_gender_distribution(df, df["Gender"], df["SepsisLabel"])
    fig_gender.update_layout(yaxis=dict(showgrid=False))
    st.plotly_chart(fig_gender, use_container_width=True)
    st.markdown("""<div style="font-size: 13px"><strong>Gender Distribution</strong><br><br>This chart illustrates the distribution of genders among patients, categorized by sepsis label. Each bar represents the count of patients belonging to a particular gender and sepsis label category.</div>""", unsafe_allow_html=True)

# Plot ICU distribution
with col2:
    fig_ICU = plot_icu_distribution(df, df["Unit1"], df["Unit2"])
    fig_ICU.update_layout(yaxis=dict(showgrid=False))
    st.plotly_chart(fig_ICU, use_container_width=True)
    st.markdown("""<div style="font-size: 13px"><strong>ICU Distribution (Medical ICU vs Surgical ICU)</strong><br><br>This chart compares the distribution of patients between Medical ICU and Surgical ICU. Each bar represents the count of patients admitted to a specific ICU unit, allowing for comparison between the two.</div>""", unsafe_allow_html=True)
# Filter the dataframe for septic individuals (both female and male)
septic_df = df[(df['SepsisLabel'] == 'Sepsis') & (df['Gender'].isin(['Female', 'Male']))]

# Define custom colors for female and male
custom_colors = {'Female': '#588157', 'Male': '#f6bd60'}

# Create the box plot for pH distribution by gender (Only Septic Individuals)
fig_ph = px.box(septic_df, x='Gender', y='pH', color='Gender', 
                title='<b>pH Distribution by Gender in Septic Patients</b>',
                labels={'Gender': 'Gender', 'pH': 'pH Level'},  # Update x-axis label
                color_discrete_map=custom_colors)

# Center the title
fig_ph.update_layout(title_x=0.5, title_y=0.95, titlefont=dict(size=20), title_xanchor='center')  # Adjust title_y to position the title at the top center
# Remove white background color
fig_ph.update_layout(plot_bgcolor='rgba(0,0,0,0)')
# Adjust margins and padding to display x-axis and y-axis
fig_ph.update_layout(margin=dict(l=50, r=50, t=50, b=50),
                     xaxis=dict(showline=True, linewidth=1, linecolor='black', title=dict(text="<b>Gender</b>", font=dict(size=18, color='black')), 
                                tickfont=dict(size=12, color='black', family='Arial Black, sans-serif')), 
                     yaxis=dict(showline=True, linewidth=1, linecolor='black', title=dict(text="<b>pH Level</b>", font=dict(size=18, color='black'))),
                     xaxis_showgrid=False,  # Remove x-axis gridlines
                     yaxis_showgrid=False)  # Remove y-axis gridlines

# Create a box plot for platelets distribution by sepsis label
fig_platelets = px.box(df, x='SepsisLabel', y='Platelets', color='SepsisLabel',
                       title='<b>Platelets Distribution by Sepsis Label</b>',
                       labels={'SepsisLabel': 'Sepsis Label', 'Platelets': 'Platelets Count'})

# Center the title
fig_platelets.update_layout(title_x=0.5, title_y=0.95, titlefont=dict(size=20), title_xanchor='center')  # Adjust title_y to position the title at the top center
# Remove white background color
fig_platelets.update_layout(plot_bgcolor='rgba(0,0,0,0)')
# Adjust margins and padding to display x-axis and y-axis
fig_platelets.update_layout(margin=dict(l=50, r=50, t=50, b=50),
                            xaxis=dict(showline=True, linewidth=1, linecolor='black', title=dict(text="<b>Sepsis Label</b>", font=dict(size=18, color='black')), 
                                       tickfont=dict(size=12, color='black', family='Arial Black, sans-serif')), 
                            yaxis=dict(showline=True, linewidth=1, linecolor='black', title=dict(text="<b>Platelets Count</b>", font=dict(size=18, color='black'))),
                            xaxis_showgrid=False,  # Remove x-axis gridlines
                            yaxis_showgrid=False)  # Remove y-axis gridlines

# Display the plots using Streamlit on the same line horizontally
col1, col2 = st.columns(2)
with col1:
    st.plotly_chart(fig_ph, use_container_width=True)
    st.markdown("""<div style="font-size: 13px; "><strong>pH Distribution by Gender in Septic Patients</strong><br><br>
    This box plot illustrates the distribution of pH levels among septic patients, categorized by gender. Each box represents the distribution of pH levels for a specific gender, providing insights into the variability of pH levels.</div>""", unsafe_allow_html=True)

with col2:
    st.plotly_chart(fig_platelets, use_container_width=True)
    st.markdown("""<div style="font-size: 13px; "><strong>Platelets Distribution by Sepsis Label</strong><br><br>
    This box plot displays the distribution of platelets among individuals with and without sepsis. It helps visualize the spread and central tendency of platelet values across different sepsis labels.</div>""", unsafe_allow_html=True)
  # Machine Learning Part with Custom Styled Header and Padding
st.markdown("""
    <h2 style='text-align: center; color: #22577a; font-family: Arial; font-size: 36px; padding-top: 10px; padding-bottom: 10px;'>
        Machine Learning Insights
    </h2>
""", unsafe_allow_html=True)
X = df.drop(['SepsisLabel'], axis=1)  # Features
y = df['SepsisLabel']  # Target variable
# Split the data into training and temporary sets (80% train, 20% temp)
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.2, random_state=42)

# Split the temporary set into validation and test sets (50% validation, 50% test)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

# Display the shapes of the resulting datasets
print("Training set shape:", X_train.shape, y_train.shape)
print("Validation set shape:", X_val.shape, y_val.shape)
print("Test set shape:", X_test.shape, y_test.shape)
# Calculate percentage of missing values for each column in X_train
missing_percentage_train = (X_train.isnull().sum() / len(X_train)) * 100

# Display the missing percentage for each column in X_train
print("Missing percentage for each column in X_train:")
print(missing_percentage_train)
# Define thresholds for missing value categorization
high_missing_threshold = 90
moderate_missing_threshold_low = 30
moderate_missing_threshold_high = 90

# List to store feature names based on missing value categories
high_missing_features = []
moderate_missing_features = []
low_missing_features = []

# Iterate through each column and categorize based on missing values
for column in X_train.columns:
    missing_percentage = (X_train[column].isnull().sum() / len(X_train)) * 100
    if missing_percentage > high_missing_threshold:
        high_missing_features.append(column)
    elif moderate_missing_threshold_low <= missing_percentage <= moderate_missing_threshold_high:
        moderate_missing_features.append(column)
    else:
        low_missing_features.append(column)

# Print the categorized features
print("Features with High Missing Values (Above 90%):")
print(high_missing_features)
print("\nFeatures with Moderate Missing Values (Between 30% to 90%):")
print(moderate_missing_features)
print("\nFeatures with Low or No Missing Values (Below 10%):")
print(low_missing_features)
# List of features with high missing values (above 90%)
features_with_high_missing = ['EtCO2', 'BaseExcess', 'HCO3', 'pH', 'PaCO2', 'SaO2', 
                              'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 
                              'Creatinine', 'Bilirubin_direct', 'Lactate', 'Magnesium', 
                              'Phosphate', 'Bilirubin_total', 'TroponinI', 'Hct', 
                              'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']

# Drop features with high missing values from X_train
X_train_cleaned = X_train.drop(columns=features_with_high_missing)

# Reset index after dropping columns
X_train_cleaned = X_train_cleaned.reset_index(drop=True)
# Check the shape of X_train after dropping
print("Shape of X_train after dropping:", X_train_cleaned.shape)
# Print the list of features after dropping
print("List of features after dropping:")
print(X_train_cleaned.columns.tolist())
# Identify numerical features
numerical_features = X_train_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()
print("Numerical features:")
print(numerical_features)
# Impute missing values with mean
for feature in numerical_features:
    mean_value = X_train_cleaned[feature].mean()  # Calculate the mean value
    X_train_cleaned[feature] = X_train_cleaned[feature].fillna(mean_value)  # Fill NaN values with mean_value
#Calculate the number of missing values after imputation
missing_after = X_train_cleaned[numerical_features].isnull().sum()
print("\nMissing Values After Imputation:")
print(missing_after)
# Identify categorical features in X_train
categorical_features_train = X_train.select_dtypes(include=['object']).columns.tolist()
print("Categorical features:", categorical_features_train)
# One-hot encode categorical features for X_train
onehot_encoder = OneHotEncoder(drop=None)  # Ensure no columns are dropped
X_train_encoded = onehot_encoder.fit_transform(X_train[categorical_features_train])
# Convert the encoded array to a DataFrame
X_train_encoded_df = pd.DataFrame(X_train_encoded.toarray(), columns=onehot_encoder.get_feature_names_out(categorical_features_train))
# Drop the gender scale column from X_train_cleaned_processed
X_train_cleaned_processed = X_train_cleaned.drop(columns=['Gender', 'Unit1', 'Unit2'])
# Concatenate preprocessed features (without gender) with encoded features
X_train_final = pd.concat([X_train_cleaned_processed.reset_index(drop=True), X_train_encoded_df.reset_index(drop=True)], axis=1)

# Set display format to show two decimal places for floating-point numbers
pd.options.display.float_format = '{:.2f}'.format
# Check data types of X_train_final_processed
print(X_train_final)
# Step 1: Check data types of columns in X_train_final
print(X_train_final.dtypes)
# Check for missing values in X_train_final
missing_values = X_train_final.isnull().sum()

# Print columns with missing values, if any
print("Columns with missing values:")
print(missing_values[missing_values > 0])
# Calculate missing percentage for X_test
missing_percentage_test = (X_test.isnull().sum() / len(X_test)) * 100

# Display the missing percentage for each feature in X_test
print("Missing percentage for each feature in X_test:")
print(missing_percentage_test)
# Define thresholds for missing value categorization
high_missing_threshold = 90
moderate_missing_threshold_low = 30
moderate_missing_threshold_high = 90

# List to store feature names based on missing value categories for X_test
high_missing_features_test = []
moderate_missing_features_test = []
low_missing_features_test = []

# Iterate through each column in X_test and categorize based on missing values
for column in X_test.columns:
    missing_percentage = (X_test[column].isnull().sum() / len(X_test)) * 100
    if missing_percentage > high_missing_threshold:
        high_missing_features_test.append(column)
    elif moderate_missing_threshold_low <= missing_percentage <= moderate_missing_threshold_high:
        moderate_missing_features_test.append(column)
    else:
        low_missing_features_test.append(column)

# Print the categorized features for X_test
print("Features with High Missing Values (Above 90%):")
print(high_missing_features_test)
print("\nFeatures with Moderate Missing Values (Between 30% to 90%):")
print(moderate_missing_features_test)
print("\nFeatures with Low or No Missing Values (Below 10%):")
print(low_missing_features_test)
# List of features with high missing values (above 90%) in X_test
features_with_high_missing_test = ['EtCO2', 'BaseExcess', 'HCO3', 'pH', 'PaCO2', 'SaO2', 
                                   'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 
                                   'Creatinine', 'Bilirubin_direct', 'Lactate', 'Magnesium', 
                                   'Phosphate', 'Bilirubin_total', 'TroponinI', 'Hct', 
                                   'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']

# Drop features with high missing values from X_test
X_test_cleaned = X_test.drop(columns=features_with_high_missing_test)

# Reset index after dropping columns
X_test_cleaned = X_test_cleaned.reset_index(drop=True)
# Check the shape of X_test after dropping
print("Shape of X_test after dropping:", X_test_cleaned.shape)
# Print the list of features after dropping
print("List of features after dropping:")
print(X_test_cleaned.columns.tolist())
# Identify numerical features
numerical_features = X_test_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()
print("Numerical features:")
print(numerical_features)
# Impute missing values with mean
for feature in numerical_features:
    # Calculate the mean once for each feature to avoid recalculating it in each iteration
    mean_value = X_test_cleaned[feature].mean()
    X_test_cleaned[feature] = X_test_cleaned[feature].fillna(mean_value)

# Calculate the number of missing values after imputation
missing_after = X_test_cleaned[numerical_features].isnull().sum()
print("\nMissing Values After Imputation:")
print(missing_after)
# Identify categorical features in X_test
categorical_features_test = X_test.select_dtypes(include=['object']).columns.tolist()
print("Categorical features:", categorical_features_test)
# One-hot encode categorical features for X_test
onehot_encoder = OneHotEncoder(drop=None)  # Ensure no columns are dropped
X_test_encoded = onehot_encoder.fit_transform(X_test[categorical_features_test])
#Perform KNN imputation for encoded categorical features
knn_imputer = KNNImputer()
X_test_encoded_imputed = knn_imputer.fit_transform(X_test_encoded.toarray())
# Convert the encoded array to a DataFrame
X_test_encoded_df = pd.DataFrame(X_test_encoded.toarray(), columns=onehot_encoder.get_feature_names_out(categorical_features_test))
print(X_train_encoded_df.head())
# Drop the gender scale column from X_train_cleaned_processed
X_test_cleaned_processed = X_test_cleaned.drop(columns=['Gender', 'Unit1', 'Unit2'])
# Concatenate preprocessed features (without gender) with encoded features
X_test_final = pd.concat([X_test_cleaned_processed.reset_index(drop=True), X_test_encoded_df.reset_index(drop=True)], axis=1)
# Set display format to show two decimal places for floating-point numbers
pd.options.display.float_format = '{:.2f}'.format
print(X_test_final)
# Check for missing values in X_train_final
missing_values = X_test_final.isnull().sum()

# Print columns with missing values, if any
print("Columns with missing values:")
print(missing_values[missing_values > 0])
# Calculate missing percentage for X_val
missing_percentage_val = (X_val.isnull().sum() / len(X_val)) * 100

# Display the missing percentage for each feature in X_val
print("Missing percentage for each feature in X_val:")
print(missing_percentage_val)
# Define thresholds for missing value categorization
high_missing_threshold = 90
moderate_missing_threshold_low = 30
moderate_missing_threshold_high = 90

# List to store feature names based on missing value categories for X_val
high_missing_features_val = []
moderate_missing_features_val = []
low_missing_features_val = []

# Iterate through each column in X_val and categorize based on missing values
for column in X_val.columns:
    missing_percentage = (X_val[column].isnull().sum() / len(X_val)) * 100
    if missing_percentage > high_missing_threshold:
        high_missing_features_val.append(column)
    elif moderate_missing_threshold_low <= missing_percentage <= moderate_missing_threshold_high:
        moderate_missing_features_val.append(column)
    else:
        low_missing_features_val.append(column)

# Print the categorized features for X_val
print("Features with High Missing Values (Above 90%):")
print(high_missing_features_val)
print("\nFeatures with Moderate Missing Values (Between 30% to 90%):")
print(moderate_missing_features_val)
print("\nFeatures with Low or No Missing Values (Below 10%):")
print(low_missing_features_val)
# List of features with high missing values (above 90%) in X_val
features_with_high_missing_val = ['EtCO2', 'BaseExcess', 'HCO3', 'pH', 'PaCO2', 'SaO2', 
                                   'AST', 'BUN', 'Alkalinephos', 'Calcium', 'Chloride', 
                                   'Creatinine', 'Bilirubin_direct', 'Lactate', 'Magnesium', 
                                   'Phosphate', 'Bilirubin_total', 'TroponinI', 'Hct', 
                                   'Hgb', 'PTT', 'WBC', 'Fibrinogen', 'Platelets']

# Drop features with high missing values from X_val
X_val_cleaned = X_val.drop(columns=features_with_high_missing_val)

# Reset index after dropping columns
X_val_cleaned = X_val_cleaned.reset_index(drop=True)

print("Shape of X_val after dropping:", X_val_cleaned.shape)
# Print the list of features after dropping
print("List of features after dropping:")
print(X_val_cleaned.columns.tolist())
# Identify numerical features
numerical_features = X_val_cleaned.select_dtypes(include=['float64', 'int64']).columns.tolist()
print("Numerical features:")
print(numerical_features)
# Impute missing values with mean
for feature in numerical_features:
    # Fill missing values with the mean of the column, without using inplace=True
    X_val_cleaned[feature] = X_val_cleaned[feature].fillna(X_val_cleaned[feature].mean())
# Calculate the number of missing values after imputation
missing_after = X_val_cleaned[numerical_features].isnull().sum()
print("\nMissing Values After Imputation:")
print(missing_after)
# Identify categorical features in X_val
categorical_features_val = X_val.select_dtypes(include=['object']).columns.tolist()
print("Categorical features:", categorical_features_val)
# One-hot encode categorical features for X_test
onehot_encoder = OneHotEncoder(drop=None)  # Ensure no columns are dropped
X_val_encoded = onehot_encoder.fit_transform(X_val[categorical_features_val])
#Perform KNN imputation for encoded categorical features
knn_imputer = KNNImputer()
X_val_encoded_imputed = knn_imputer.fit_transform(X_val_encoded.toarray())
# Convert the encoded array to a DataFrame
X_val_encoded_df = pd.DataFrame(X_val_encoded.toarray(), columns=onehot_encoder.get_feature_names_out(categorical_features_val))
print(X_val_encoded_df.head())
# Drop the gender scale column from X_train_cleaned_processed
X_val_cleaned_processed = X_val_cleaned.drop(columns=['Gender', 'Unit1', 'Unit2'])
# Concatenate preprocessed features (without gender) with encoded features
X_val_final = pd.concat([X_val_cleaned_processed.reset_index(drop=True), X_val_encoded_df.reset_index(drop=True)], axis=1)
# Set display format to show two decimal places for floating-point numbers
pd.options.display.float_format = '{:.2f}'.format
print(X_val_final)


# Check the Data
print("Unique labels in y_train:", y_train.unique())
# Before encoding
print("Unique labels in y_train before encoding:", y_train.unique())
# if your labels are in string format, we use label encoding to convert them to numeric format
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
# After encoding
print("Unique labels in y_train after encoding:", np.unique(y_train_encoded))
# Check the unique labels in y_test before encoding
print("Unique labels in y_test before encoding:", y_test.unique())
# Initialize >a LabelEncoder object
label_encoder = LabelEncoder()
# Encode the labels in y_test
y_test_encoded = label_encoder.fit_transform(y_test)
# Check the unique labels in y_test after encoding
print("Unique labels in y_test after encoding:", np.unique(y_test_encoded))
# Check the unique labels in y_test before encoding
print("Unique labels in y_val before encoding:", y_val.unique())
# Initialize a LabelEncoder object
label_encoder = LabelEncoder()
# Encode the labels in y_test
y_val_encoded = label_encoder.fit_transform(y_val)
# Check the unique labels in y_test after encoding
print("Unique labels in y_val after encoding:", np.unique(y_val_encoded))
# to resolve AttributeError: 'numpy.ndarray' object has no attribute 'value_counts' we have to convert in pandas series
# we have encoded your labels using something like LabelEncoder, the result will be a NumPy array, and you can't directly call value_counts() on it.
# Convert encoded labels to pandas Series
y_train_encoded_series = pd.Series(y_train_encoded)
y_test_encoded_series = pd.Series(y_test_encoded)
y_val_encoded_series = pd.Series(y_val_encoded)

# Check for Data Imbalance
print("Distribution of classes in y_train_encoded:")
print(y_train_encoded_series.value_counts())

print("Distribution of classes in y_test_encoded:")
print(y_test_encoded_series.value_counts())

print("Distribution of classes in y_val_encoded:")
print(y_val_encoded_series.value_counts())
# Initialize RandomOverSampler
ros = RandomOverSampler(random_state=42)

# Perform Random Oversampling on the training data
X_train_resampled, y_train_resampled = ros.fit_resample(X_train_final, y_train_encoded_series)
# Check the shapes of the resampled arrays
print("Shape of X_train_resampled:", X_train_resampled.shape)
print("Shape of y_train_resampled:", y_train_resampled.shape)
# Print distribution of classes in resampled training set
print("Distribution of classes in resampled training set:")
print(pd.Series(y_train_resampled).value_counts())
model = xgb.XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
model.fit(X_train_resampled, y_train_resampled)
# Create a SHAP explainer
explainer = shap.Explainer(model)
# Calculate SHAP values for the resampled training data
shap_values = explainer(X_train_resampled)
# Calculate mean absolute SHAP values for each feature
feature_importance = np.abs(shap_values.values).mean(axis=0)
feature_importance_df = pd.DataFrame(list(zip(X_train_final.columns, feature_importance)), columns=['Feature', 'Importance'])
# Sort features by importance in descending order and select top 10 features
top_10_features_df = feature_importance_df.sort_values(by='Importance', ascending=False).head(10)
top_10_features = top_10_features_df['Feature'].tolist()
print("Top 10 features based on SHAP values:", top_10_features)
# Extract the top 10 features from the training set
X_train_top_10 = X_train_final[top_10_features]

# Extract the top 10 features from the validation set
X_val_top_10 = X_val_final[top_10_features]

# Extract the top 10 features from the test set
X_test_top_10 = X_test_final[top_10_features]
# Retrain the model using only the top 10 features
model.fit(X_train_resampled[top_10_features], y_train_resampled)

# Validate the model on the training set
y_train_pred = model.predict(X_train_resampled[top_10_features])

# Calculate evaluation metrics for the training set
train_accuracy = accuracy_score(y_train_resampled, y_train_pred)
train_precision = precision_score(y_train_resampled, y_train_pred)
train_recall = recall_score(y_train_resampled, y_train_pred)
train_f1 = f1_score(y_train_resampled, y_train_pred)
# Print evaluation metrics for the training set
print("Training Set Metrics:")
print("Accuracy:", train_accuracy)
print("Precision:", train_precision)
print("Recall:", train_recall)
print("F1 Score:", train_f1)
# Validate the model on the validation set
y_val_pred = model.predict(X_val_top_10)

# Calculate evaluation metrics for the validation set
val_accuracy = accuracy_score(y_val_encoded_series, y_val_pred)
val_precision = precision_score(y_val_encoded_series, y_val_pred)
val_recall = recall_score(y_val_encoded_series, y_val_pred)
val_f1 = f1_score(y_val_encoded_series, y_val_pred)

# Print evaluation metrics for the validation set
print("Validation Set Metrics:")
print("Accuracy:", val_accuracy)
print("Precision:", val_precision)
print("Recall:", val_recall)
print("F1 Score:", val_f1)
# Predict probabilities on the training set
y_train_probabilities = model.predict_proba(X_train_resampled[top_10_features])[:, 1]

# Define a range of threshold values to try
threshold_values = [0.3, 0.4, 0.5, 0.6, 0.7]

# Evaluate performance for each threshold value
for threshold in threshold_values:
    y_train_pred_threshold = (y_train_probabilities >= threshold).astype(int)
    train_precision = precision_score(y_train_resampled, y_train_pred_threshold)
    train_recall = recall_score(y_train_resampled, y_train_pred_threshold)
    train_f1 = f1_score(y_train_resampled, y_train_pred_threshold)
    
    print(f"Threshold: {threshold}")
    print("Precision:", train_precision)
    print("Recall:", train_recall)
    print("F1 Score:", train_f1)
    print()
    # Confusion Matrix for Training Set
conf_matrix_train = confusion_matrix(y_train_resampled, y_train_pred)
print("\nConfusion Matrix - Training Set:")
print(conf_matrix_train)

# ROC Curve for Training Set
fpr_train, tpr_train, thresholds_train = roc_curve(y_train_resampled, y_train_pred)
roc_auc_train = auc(fpr_train, tpr_train)

# Confusion Matrix for Validation Set
conf_matrix_val = confusion_matrix(y_val_encoded_series, y_val_pred)
print("\nConfusion Matrix - Validation Set:")
print(conf_matrix_val)

# ROC Curve for Validation Set
fpr_val, tpr_val, thresholds_val = roc_curve(y_val_encoded_series, y_val_pred)
roc_auc_val = auc(fpr_val, tpr_val)

# Plot ROC Curves
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
plt.plot(fpr_train, tpr_train, color='darkorange', lw=2, label='Training ROC curve (area = %0.2f)' % roc_auc_train)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Training Set')
plt.legend(loc="lower right")

plt.subplot(1, 2, 2)
plt.plot(fpr_val, tpr_val, color='darkorange', lw=2, label='Validation ROC curve (area = %0.2f)' % roc_auc_val)
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic - Validation Set')
plt.legend(loc="lower right")

plt.tight_layout()
# Confusion Matrix - Training Set
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_train, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - Training Set')


# Confusion Matrix - Validation Set
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_val, annot=True, fmt='d', cmap='Blues', annot_kws={"size": 16})
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('Confusion Matrix - Validation Set')

    # Save the model to a file
with open('xgb_model_top_10.pkl', 'wb') as file:
    pickle.dump(model, file)
    # Load the model from the file
with open('xgb_model_top_10.pkl', 'rb') as file:
    loaded_model = pickle.load(file)
    # Predict probabilities on the test set using the loaded model
y_test_probabilities = loaded_model.predict_proba(X_test_top_10)[:, 1]

# Apply threshold to the predicted probabilities
selected_threshold = 0.4 # Replace with your selected threshold value
y_test_pred_threshold = (y_test_probabilities >= selected_threshold).astype(int)

# Evaluate the model's performance on the test set
test_accuracy = accuracy_score(y_test_encoded_series, y_test_pred_threshold)
test_precision = precision_score(y_test_encoded_series, y_test_pred_threshold)
test_recall = recall_score(y_test_encoded_series, y_test_pred_threshold)
test_f1 = f1_score(y_test_encoded_series, y_test_pred_threshold)

# Print evaluation metrics for the test set
print("Test Set Metrics:")
print("Accuracy:", test_accuracy)
print("Precision:", test_precision)
print("Recall:", test_recall)
print("F1 Score:", test_f1)




# Your existing confusion matrix computation
conf_matrix = confusion_matrix(y_test_encoded_series, y_test_pred_threshold)

# Interactive Confusion Matrix
conf_fig = go.Figure(data=go.Heatmap(
    z=conf_matrix,
    x=['Predicted Negative', 'Predicted Positive'],
    y=['Actual Negative', 'Actual Positive'],
    colorscale='Blues',
    showscale=False,
    hoverongaps=False,
    text=conf_matrix,
    hoverinfo='text',  # Show the text when hovering over the heatmap
    texttemplate='%{text}'
))

conf_fig.update_layout(
    title='<b>Sepsis Confusion Matrix</b>',
    title_x=0.5,  # Center the title
    title_y=0.9,  # Set the title position at the top
    title_font=dict(size=18, color='black'),
    xaxis=dict(title='<b>Predicted Label</b>', titlefont=dict(size=18, color='black'), tickfont=dict(size=12, color='black', family='Arial Black, sans-serif'), showline=True, linewidth=1, linecolor='black', showgrid=False),
    yaxis=dict(title='<b>True Label</b>', titlefont=dict(size=18, color='black'), tickfont=dict(size=12, color='black', family='Arial Black, sans-serif'), showline=True, linewidth=1, linecolor='black', showgrid=False),
    width=450,
    height=450,
    plot_bgcolor='rgba(0,0,0,0)'
)

# Create a DataFrame to display the metrics
metrics_df = pd.DataFrame({
    'Metric': ['Accuracy', 'Precision', 'Recall', 'F1 Score'],
    'Value': [test_accuracy, test_precision, test_recall, test_f1]
})

# Set color palette for dataframe
colors = ['#588157', '#f6bd60', '#00b4d8', '#22577a']

# Reset index before styling
metrics_df_reset = metrics_df.reset_index(drop=True)

# Apply the styles to the dataframe
styled_metrics_df = metrics_df_reset.style.apply(
    lambda x: ['background-color: %s' % colors[i % len(colors)] for i in range(len(metrics_df_reset))], axis=0
).set_properties(
    **{'text-align': 'center', 'font-size': '20px', 'color': 'white'}
).set_table_styles([
    {'selector': 'th', 'props': [('border', '1px solid black'), ('font-size', '22px'), ('color', 'black'), ('text-align', 'center'), ('background-color', '#f2f2f2')]},
    {'selector': 'td', 'props': [('padding', '20px'), ('height', '50px'), ('width', '100px'), ('font-size', '20px')]},
    {'selector': 'th', 'props': [('padding', '20px'), ('height', '50px'), ('width', '100px'), ('font-size', '20px')]}
]).format({'Value': "{:.2f}"})

# Side-by-side display of confusion matrix and metrics DataFrame
col1, col2 = st.columns(2)
# Convert the styled DataFrame to HTML with index=False
html_metrics_df = styled_metrics_df.hide(axis="index").to_html(escape=False)
# Get the height of the confusion matrix chart
conf_chart_height = 450  # Update this with the actual height of your confusion matrix chart

# Define the desired height for the metrics DataFrame
metrics_df_height = 450  # Adjust this value to match the height of the confusion matrix
# Define the desired width for the metrics DataFrame
metrics_df_width = 600  # Adjust this value as needed
# Display the confusion matrix chart
with col1:
    st.plotly_chart(conf_fig)

# Remove index column before converting to HTML
styled_metrics_html = styled_metrics_df.to_html(escape=False)
styled_metrics_html = styled_metrics_html.replace('<th></th>', '')  # Remove the empty header for the index column

# Adjust vertical alignment and size of the metrics DataFrame
with col2:
    st.markdown(f"""
        <div style="display: flex; flex-direction: column; align-items: flex-start; justify-content: flex-start; height: {conf_chart_height}px;">
            <!-- Title -->
            <h2 style="font-size: 18px; margin-top: 10px; text-align: left; color: black; font-weight: bold;">Performance Metrics</h2>
            <!-- Display the metrics DataFrame -->
            <div style="padding-top: 10px; width: {metrics_df_width}px;"> <!-- Use the defined width -->
                {html_metrics_df}
   
    """, unsafe_allow_html=True)








# Create the feature importance plot
fig = px.bar(top_10_features_df, x='Feature', y='Importance', title='Feature Importances')

fig.update_layout(
    title_text='Feature Importances',
    title_font=dict(size=18, color='black', family='Arial, sans-serif'),  # Update font size and family
    title_x=0.5,
    title_y=0.95,
    xaxis_title='<b>Feature</b>',
    yaxis_title='<b>Weights</b>',
    width=450,
    height=450,
    plot_bgcolor='rgba(0,0,0,0)',
    xaxis=dict(
        showline=True,
        linecolor='black',
        title=dict(font=dict(size=18, color='black', family='Arial, sans-serif')),
        tickfont=dict(size=12, color='black', family='Arial Black, sans-serif')  # Use Arial Black for bold labels
    ),
    yaxis=dict(
        showline=True,
        linecolor='black',
        title=dict(font=dict(size=18, color='black', family='Arial, sans-serif'))
    ),
    xaxis_showgrid=False,  # Remove grid lines
    yaxis_showgrid=False
)

# Calculate ROC curve
fpr, tpr, thresholds = roc_curve(y_test_encoded_series, y_test_probabilities)
roc_auc = auc(fpr, tpr)

# Interactive ROC Curve
roc_fig = go.Figure()

roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode='lines', name='ROC curve (area = %0.2f)' % roc_auc, line=dict(color='darkorange', width=2)))
roc_fig.add_trace(go.Scatter(x=[0, 1], y=[0, 1], mode='lines', line=dict(color='navy', width=2, dash='dash')))

# Update the layout for the ROC curve chart
roc_fig.update_layout(
    title='<b>Receiver Operating Characteristic Curve</b>',
    title_x=0,  # Align the title to the left
    title_y=0.9,  # Set the title position at the top
    title_font=dict(size=18, color='black', family='Arial, sans-serif'),  # Update font size and family
    xaxis=dict(title='<b>False Positive Rate</b>', titlefont=dict(size=18, color='black', family='Arial, sans-serif'), showline=True, linewidth=1, linecolor='black', showgrid=False),
    yaxis=dict(title='<b>True Positive Rate</b>', titlefont=dict(size=18, color='black', family='Arial, sans-serif'), showline=True, linewidth=1, linecolor='black', showgrid=False),
    width=450,
    height=450,
    plot_bgcolor='rgba(0,0,0,0)'
)
# Display both charts side by side using columns
col1, col2 = st.columns(2)

with col1:
    st.plotly_chart(fig)

with col2:
    st.plotly_chart(roc_fig)

# Set the layout width to accommodate longer titles
st.markdown("<style> .css-1aumxhk { max-width: 100%; }</style>", unsafe_allow_html=True)


# Predict probabilities on the test set using the loaded model
y_test_probabilities = loaded_model.predict_proba(X_test_top_10)[:, 1]

# Compute predicted sepsis flags for the test dataset
y_pred_sepsis = loaded_model.predict(X_test_top_10)

# Add predicted probabilities and sepsis flags to the test data

X_test_top_10 = X_test_top_10.copy() 
X_test_top_10['Predicted Sepsis Probabilities'] = y_test_probabilities
X_test_top_10['Predicted Sepsis Flag'] = y_pred_sepsis

# Filter the dataframe to include only rows where the predicted sepsis flag is 1
sepsis_predictions = X_test_top_10[X_test_top_10['Predicted Sepsis Flag'] == 1]
# Filter the dataframe to include only rows where the predicted sepsis flag is 1
sepsis_predictions = X_test_top_10[X_test_top_10['Predicted Sepsis Flag'] == 1].copy()
# Rename the column "Unit1_Surgical ICU" to "Unit1 Surgical ICU"
sepsis_predictions.rename(columns={"Unit1_Surgical ICU": "Unit1 Surgical ICU"}, inplace=True)

# Reset the index of the filtered DataFrame
sepsis_predictions = sepsis_predictions.reset_index(drop=True)

# Select the first 10 rows of the filtered DataFrame
sepsis_predictions_top_10 = sepsis_predictions.head(10)
# Reset index to add default index numbers
sepsis_predictions_top_10 = sepsis_predictions_top_10.reset_index()
# Display the subset of the test dataset with predictions using Streamlit
title_style = "color: #22577a; font-size: 20px; text-shadow: 1px 1px 2px rgba(0,0,0,0.4);"
st.markdown(f"<h2 style='{title_style}'>Predicted Sepsis Patients</h2>", unsafe_allow_html=True)

# Apply text shadow and color to all column names
styled_column_names = [f"<span style='color: #ffffff; text-shadow: 1px 1px 2px rgba(0,0,0,0.4); background-color: #22577a; padding: 8px;'>{col}</span>" for col in sepsis_predictions_top_10.columns]

# Convert the subset DataFrame to HTML without index column and with styled column names
sepsis_predictions_top_10_html = sepsis_predictions_top_10.rename(columns=dict(zip(sepsis_predictions_top_10.columns, styled_column_names))).to_html(index=False, escape=False)

# Display the HTML table using Streamlit
st.write(sepsis_predictions_top_10_html, unsafe_allow_html=True)

# Apply style to center align text in Predicted Sepsis Probabilities and Predicted Sepsis Flag columns
st.write("""
<style>
    .dataframe td:nth-child(11), .dataframe td:nth-child(12) {
        text-align: center;
    }
    
    .dataframe {
        border-collapse: collapse;
        font-family: Arial, sans-serif;
        font-size: 14px;
    }

    .dataframe th {
        background-color: #22577a;
        color: white;
        font-weight: bold;
        padding: 8px;
        text-align: center;
    }

    .dataframe td {
        background-color: #f2f2f2;
        color: black;
        padding: 8px;
        border: 1px solid black;
    }

    .dataframe tr:hover {
        background-color: #dddddd;
    }
</style>
""", unsafe_allow_html=True)   

