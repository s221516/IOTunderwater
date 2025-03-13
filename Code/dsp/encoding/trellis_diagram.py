from numpy import array
from commpy.channelcoding import Trellis

# Define the number of memory elements 
# per input in the convolutional encoder
memory = array([2])

# Define the generator matrix of 
# the convolutional encoder 
# Entries are in octal format
g_matrix = array([[05, 07]])

# Create a trellis representation 
# from the encoder generator matrix 
trellis = Trellis(memory, g_matrix)

# Specify the number of time steps 
# in the trellis diagram
trellis_length = 3

# Specify the order in which states 
# should appear on the trellis diagram
state_order = [0, 2, 1, 3]

# Specify the colors for 0, 1 inputs
# '#FF0000' --> Red   (edge corresponding to input 0)
# '#00FF00' --> Green (edge corresponding to input 1)
bit_colors = ['#FF0000', '#00FF00'] 

# Plot the trellis diagram
trellis.visualize(trellis_length, state_order, 
                  edge_colors = bit_colors)