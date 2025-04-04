import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path

def create_box(ax, xy, width, height, color, label, alpha=0.3):
    rect = patches.Rectangle(xy, width, height, facecolor=color, alpha=alpha, edgecolor='black')
    ax.add_patch(rect)
    # Split label into title and params if it contains \n
    parts = label.split('\n')
    if len(parts) > 1:
        ax.text(xy[0] + width/2, xy[1] + height * 0.7, parts[0], ha='center', va='center', fontsize=12)
        ax.text(xy[0] + width/2, xy[1] + height * 0.3, parts[1], ha='center', va='center', fontsize=10)
    else:
        ax.text(xy[0] + width/2, xy[1] + height/2, label, ha='center', va='center', fontsize=12)

def create_component_box(ax, xy, width, height, color, title, components, alpha=0.3):
    # Main container box
    rect = patches.Rectangle(xy, width, height, facecolor=color, alpha=alpha, edgecolor='black')
    ax.add_patch(rect)
    
    # Title with black background
    title_height = 0.3
    title_box = patches.Rectangle((xy[0], xy[1] + height - title_height), width, title_height, 
                                facecolor='black', alpha=0.1)
    ax.add_patch(title_box)
    ax.text(xy[0] + width/2, xy[1] + height - title_height/2, title, 
            ha='center', va='center', fontsize=12, fontweight='bold')
    
    # Components
    usable_height = height - title_height
    spacing = usable_height / (len(components) + 1)
    for i, comp in enumerate(components, 1):
        y = xy[1] + usable_height - i * spacing
        subrect = patches.Rectangle((xy[0] + 0.1, y - 0.15), width - 0.2, spacing * 0.8, 
                                  facecolor='white', edgecolor='black', alpha=0.5)
        ax.add_patch(subrect)
        ax.text(xy[0] + width/2, y, comp, ha='center', va='center', fontsize=10)

def create_arrow(ax, start, end, y_offset=0):
    # Create straight arrows with slight offset
    start_x, start_y = start
    end_x, end_y = end
    
    # Add small vertical offset to avoid overlapping
    actual_start_y = start_y + y_offset
    actual_end_y = end_y + y_offset
    
    ax.arrow(start_x, actual_start_y, 
            end_x - start_x, actual_end_y - actual_start_y,
            head_width=0.1, head_length=0.2, fc='black', ec='black',
            length_includes_head=True, linewidth=1)

# Create figure and axis
fig, ax = plt.subplots(figsize=(15, 6))  # Further reduced height

# Define spacing constants
box_width = 2.2
box_spacing = 3
start_x = 1
main_block_y = 2.2  # Moved blocks down
component_y = 0.2   # Moved components up

# Create main boxes with adjusted positions and spacing
create_box(ax, (start_x, main_block_y), box_width, 1, 'lightblue', 'Input Layer\n(seq_len=30,features=76)')
create_box(ax, (start_x + box_spacing, main_block_y), box_width, 1, 'lightgreen', 'CNN Block\n(filter=64,kernel_sz=3)')
create_box(ax, (start_x + box_spacing*2, main_block_y), box_width, 1, 'orange', 'BiLSTM Layers\n(units=[128,32,32])')
create_box(ax, (start_x + box_spacing*3, main_block_y), box_width, 1, 'yellow', 'Attention\nMechanism')
create_box(ax, (start_x + box_spacing*4, main_block_y), box_width, 1, 'plum', 'Output Block\n(Dense + Sigmoid)')

# Create component boxes
comp_box_width = 2.2
comp_box_height = 1.8  # Slightly reduced height

cnn_components = ['Conv1D Layer', 'MaxPooling1D', 'Dropout(0.2)']
create_component_box(ax, (start_x + box_spacing, component_y), comp_box_width, comp_box_height, 'lightgreen', 
                    'CNN', cnn_components)

lstm_components = ['Layer 1 (128)', 'Dropout(0.2)', 'Layer 2 (32)', 'Dropout(0.2)', 'Layer 3 (32)']
create_component_box(ax, (start_x + box_spacing*2, component_y), comp_box_width, comp_box_height, 'orange', 
                    'BiLSTM', lstm_components)

attention_components = ['Dense(1)', 'Softmax', 'Weighted Sum']
create_component_box(ax, (start_x + box_spacing*3, component_y), comp_box_width, comp_box_height, 'yellow', 
                    'Attention', attention_components)

output_components = ['Dense(32, ReLU)', 'Dense(1, Sigmoid)']
create_component_box(ax, (start_x + box_spacing*4, component_y), comp_box_width, comp_box_height, 'plum', 
                    'Output', output_components)

# Create arrows with minimal offsets
for i in range(4):
    create_arrow(ax, (start_x + box_spacing*i + box_width, main_block_y + 0.5), 
                (start_x + box_spacing*(i+1), main_block_y + 0.5), 
                y_offset=0.05 if i % 2 == 0 else -0.05)

# Create dotted connection lines with increased length
for i in range(4):
    x = start_x + box_spacing*(i+1) + box_width/2
    ax.plot([x, x], [main_block_y, component_y + comp_box_height], 'k--', alpha=0.5, zorder=1)

# Set axis properties
ax.set_xlim(0, 18)
ax.set_ylim(-0.1, 3.5)  # Further reduced y-limits
ax.axis('off')

# Add title with adjusted padding and larger font
plt.title('CNN-BiLSTM Architecture with Attention Mechanism', pad=5, fontsize=16, fontweight='bold')

# Add parameters text box with adjusted position and larger font
params_text = (
    'Model Parameters:\n'
    '• Input Shape: (30, 76) - 30 time steps, 76 features\n'
    '• CNN: 64 filters, kernel size 3, ReLU activation\n'
    '• BiLSTM: [128, 32, 32] units with return sequences\n'
    '• Attention: Dense(1) with Softmax normalization\n'
    '• Dropout Rate: 0.2 throughout\n'
    '• Final Dense: 32 units with ReLU\n'
    '• Output: Single unit with Sigmoid\n'
    '• Optimizer: Adam(learning_rate=0.001)\n'
    '• Loss: Binary Cross-Entropy'
)

# Position the parameters box closer to the diagram
plt.figtext(0.02, 0.12, params_text, fontsize=10, 
            bbox=dict(facecolor='white', alpha=0.8, edgecolor='black', boxstyle='round,pad=1'))

# Save the figure with minimal padding
plt.savefig('cnn_bilstm_architecture.png', bbox_inches='tight', dpi=300, pad_inches=0.1) 