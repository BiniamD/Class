import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_layer_box(ax, xy, width, height, color, label, alpha=0.3):
    rect = patches.Rectangle(xy, width, height, facecolor=color, alpha=alpha, edgecolor='black')
    ax.add_patch(rect)
    ax.text(xy[0] + width/2, xy[1] + height/2, label, ha='center', va='center', fontsize=9, wrap=True)

def create_arrow(ax, start, end):
    ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
             head_width=0.1, head_length=0.1, fc='black', ec='black', length_includes_head=True)

# Create figure and axis
fig, ax = plt.subplots(figsize=(15, 10))

# Layer positions and dimensions
width = 2.5
height = 0.6
x_spacing = 3
y_start = 8

# Input Layer
create_layer_box(ax, (1, y_start), width, height, 'lightblue', 
                'Input Layer\n(seq_length=30, features=76)')

# CNN Layers
create_layer_box(ax, (1+x_spacing, y_start), width, height, 'lightgreen',
                'Conv1D\n(filters=64, kernel_size=3, ReLU)')
create_layer_box(ax, (1+x_spacing, y_start-1), width, height, 'lightgreen',
                'MaxPooling1D\n(pool_size=2)')
create_layer_box(ax, (1+x_spacing, y_start-2), width, height, 'lightgreen',
                'Dropout\n(rate=0.2)')

# First BiLSTM Block
create_layer_box(ax, (1+2*x_spacing, y_start), width, height, 'orange',
                'BiLSTM Layer 1\n(units=128, return_sequences=True)')
create_layer_box(ax, (1+2*x_spacing, y_start-1), width, height, 'orange',
                'Dropout\n(rate=0.2)')

# Second BiLSTM Block
create_layer_box(ax, (1+3*x_spacing, y_start), width, height, 'orange',
                'BiLSTM Layer 2\n(units=32, return_sequences=True)')
create_layer_box(ax, (1+3*x_spacing, y_start-1), width, height, 'orange',
                'Dropout\n(rate=0.2)')

# Attention Mechanism
create_layer_box(ax, (1+4*x_spacing, y_start), width, height, 'yellow',
                'Dense(1)\nAttention Scores')
create_layer_box(ax, (1+4*x_spacing, y_start-1), width, height, 'yellow',
                'Softmax\nAttention Weights')
create_layer_box(ax, (1+4*x_spacing, y_start-2), width, height, 'yellow',
                'Weighted Sum\nContext Vector')

# Final BiLSTM and Dense Layers
create_layer_box(ax, (1+5*x_spacing, y_start), width, height, 'orange',
                'BiLSTM Layer 3\n(units=32)')
create_layer_box(ax, (1+5*x_spacing, y_start-1), width, height, 'plum',
                'Dense Layer\n(units=32, ReLU)')
create_layer_box(ax, (1+5*x_spacing, y_start-2), width, height, 'plum',
                'Output Layer\n(units=1, Sigmoid)')

# Add arrows
for i in range(6):
    # Horizontal connections between major components
    create_arrow(ax, (1+width+(i*x_spacing), y_start+height/2), 
                (1+width+x_spacing*(i+1)-0.5, y_start+height/2))
    
    # Vertical connections within components
    if i > 0:
        for j in range(2):
            ax.plot([1+width+(i*x_spacing), 1+width+(i*x_spacing)],
                   [y_start-j, y_start-j-1], 'k--', alpha=0.3)

# Add labels for major components
component_labels = ['Input', 'CNN Block', 'BiLSTM\nBlock 1', 'BiLSTM\nBlock 2', 'Attention\nMechanism', 'Output\nBlock']
for i, label in enumerate(component_labels):
    ax.text(1+i*x_spacing+width/2, y_start+1, label, ha='center', fontsize=12, fontweight='bold')

# Set axis properties
ax.set_xlim(0, 20)
ax.set_ylim(5, 10)
ax.axis('off')

# Add title
plt.title('Detailed CNN-BiLSTM Model Architecture with Attention', pad=20, fontsize=14, fontweight='bold')

# Add model parameters as text
params_text = """Model Parameters:
• Input Shape: (30, 76) - 30 time steps, 76 features
• CNN: 64 filters, kernel size 3, ReLU activation
• BiLSTM: [128, 32, 32] units with return_sequences
• Attention: Dense(1) with Softmax normalization
• Dropout Rate: 0.2 throughout
• Final Dense: 32 units with ReLU
• Output: Single unit with Sigmoid
• Optimizer: Adam(learning_rate=0.001)
• Loss: Binary Cross-Entropy"""

plt.text(0.02, 0.02, params_text, transform=ax.transAxes, fontsize=10,
         verticalalignment='bottom', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))

# Save the figure
plt.savefig('cnn_bilstm_model_architecture.png', bbox_inches='tight', dpi=300, facecolor='white')
plt.close() 