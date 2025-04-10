import pandas as pd
import matplotlib.pyplot as plt

file_path = "C:/Users/morte/OneDrive - Danmarks Tekniske Universitet/Bachelor/IOTunderwater/conv_encode_test.csv"
output_file_path = "C:/Users/morte/OneDrive - Danmarks Tekniske Universitet/Bachelor/IOTunderwater/conv_encode_test_updated.csv"

# Load the CSV file with correct delimiter
df = pd.read_csv(output_file_path, delimiter=";", usecols=[0, 1, 2, 3, 4, 5])
df.columns = ["ID", "Bitrate", "Carrier_Frequency", "Original_Message", "Decoded_Message", "Conv_Encode"]

# Function to compute Hamming distance
def hamming_distance(str1, str2):
    if len(str1) != len(str2):
        return None  # Handle cases where messages have different lengths
    return sum(c1 != c2 for c1, c2 in zip(str1, str2))

# Compute the Hamming distance
df["Hamming_Distance"] = df.apply(
    lambda row: hamming_distance(row["Original_Message"], row["Decoded_Message"])
    if "No preamble found" not in row["Decoded_Message"] else None,
    axis=1
)

boolean_column = 'Conv_Encode'  # Replace with your boolean column name
int_column = 'Hamming_Distance'  # Replace with your integer column name

# Create the figure and subplots
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

# Filter the DataFrame for True values
df_true = df[df[boolean_column] == True]

# Plot True values
ax1.scatter(df_true["ID"], df_true[int_column], color='blue', label='True')
ax1.set_xlabel("TestID")
ax1.set_ylabel(int_column)

ax1.set_title(f"Convolutional encoding (True) and Hamming Distance")

ax1.legend()
ax1.grid(True)

# Set integer ticks for x and y axes
ax1.set_xticks(range(int(df["ID"].min()), int(df["ID"].max()) + 1))
ax1.set_yticks(range(int(df[int_column].min()), int(df[int_column].max()) + 1))

# Filter the DataFrame for False values
df_false = df[df[boolean_column] == False]

# Plot False values
ax2.scatter(df_false["ID"], df_false[int_column], color='red', label='False')
ax2.set_xlabel("TestID")
ax2.set_ylabel(int_column)
ax2.set_title("Convolutional encoding (False) and Hamming Distance")
ax2.legend()
ax2.grid(True)

# Set integer ticks for x and y axes
ax2.set_xticks(range(int(df["ID"].min()), int(df["ID"].max()) + 1))
ax2.set_yticks(range(int(df[int_column].min()), int(df[int_column].max()) + 1))

# Filter the DataFrame for points where the preamble wasn't found
df_no_preamble = df[df["Decoded_Message"].str.contains("No preamble found", na=False)]


# Adjust layout and show the plot
plt.tight_layout()
plt.show()