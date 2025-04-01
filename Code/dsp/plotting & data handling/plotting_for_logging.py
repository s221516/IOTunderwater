import matplotlib.pyplot as plt
import pandas as pd

# Define the file path
file_path = "C:/Users/morte/OneDrive - Danmarks Tekniske Universitet/Bachelor/IOTunderwater/log.csv"

# Load the CSV file with the correct delimiter and preview the first few rows
df = pd.read_csv(file_path, delimiter=';')
print("Column names:", df.columns)  # Print column names to verify
print(df.head())  # Preview the first few rows of the DataFrame

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Filter the DataFrame for 'AAA' original messages
df_aaa = df[df["Original Message"] == "AAA"]

# Plot each unique decoded message with different markers for 'AAA' original messages
for message in df_aaa["Decoded Message"].unique():
    subset = df_aaa[df_aaa["Decoded Message"] == message]
    ax1.scatter(subset["Carrier Frequency"], subset["Decoded Message"], label=message)

# Labeling the axes and title for the first subplot
ax1.set_xlabel("Carrier Frequency (Hz)")
ax1.set_ylabel("Decoded Message")
ax1.set_title("Decoded Messages vs. Carrier Frequency (Original Message: AAA)")
ax1.legend(title="Decoded Messages", bbox_to_anchor=(1.05, 1), loc='upper left')
ax1.grid(True)

# Filter the DataFrame for 'Hello World!' original messages
df_hello_world = df[df["Original Message"] == "Hello World !"]

# Plot each unique decoded message with different markers for 'Hello World!' original messages
for message in df_hello_world["Decoded Message"].unique():
    subset = df_hello_world[df_hello_world["Decoded Message"] == message]
    ax2.scatter(subset["Carrier Frequency"], subset["Decoded Message"], label=message)

# Labeling the axes and title for the second subplot
ax2.set_xlabel("Carrier Frequency (Hz)")
ax2.set_ylabel("Decoded Message")
ax2.set_title("Decoded Messages vs. Carrier Frequency (Original Message: Hello World !)")
ax2.legend(title="Decoded Messages", bbox_to_anchor=(1.05, 1), loc='upper left')
ax2.grid(True)

# Adjust layout and show the plot
plt.tight_layout()
plt.show()