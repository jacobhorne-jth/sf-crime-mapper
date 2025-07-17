import pandas as pd

df = pd.read_csv("../data/sf_incident_data.csv")

# Keep only the useful columns
columns_to_keep = [
    "Incident Datetime",
    "Incident Date",
    "Incident Year",
    "Incident Day of Week",
    "Incident Category",
    "Police District",
    "Analysis Neighborhood",
    "Latitude",
    "Longitude"
]

df = df[columns_to_keep]

# Drop rows without coordinates, need for mapping
df = df.dropna(subset=["Latitude", "Longitude"])

neighborhood_counts = (
    df.groupby("Analysis Neighborhood")
    .size()
    .reset_index(name="Crime Count")
    .sort_values("Crime Count", ascending = False)
)

category_counts = (
    df.groupby("Incident Category")
    .size()
    .reset_index(name="Crime Count")
    .sort_values("Crime Count", ascending = False)
)


# Count by neighborhood and crime type
neighborhood_category_counts = (
    df.groupby(["Analysis Neighborhood", "Incident Category"])
    .size()
    .reset_index(name="Crime Count")
    .sort_values("Crime Count", ascending=False)  # Sort by count DESC
)

print(neighborhood_category_counts.head())



