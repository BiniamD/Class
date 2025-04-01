import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.path as path

def create_box(ax, xy, width, height, color, label, alpha=0.3):
    rect = patches.Rectangle(xy, width, height, facecolor=color, alpha=alpha, edgecolor='black')
    ax.add_patch(rect)
    ax.text(xy[0] + width/2, xy[1] + height/2, label, ha='center', va='center', fontsize=10)

def create_component_box(ax, xy, width, height, color, title, components, alpha=0.3):
    # Main box
    rect = patches.Rectangle(xy, width, height, facecolor=color, alpha=alpha, edgecolor='black')
    ax.add_patch(rect)
    
    # Title
    ax.text(xy[0] + width/2, xy[1] + height - 0.1, title, ha='center', va='top', fontsize=10, fontweight='bold')
    
    # Components
    spacing = height / (len(components) + 1)
    for i, comp in enumerate(components, 1):
        y = xy[1] + height - (i + 0.5) * spacing
        subrect = patches.Rectangle((xy[0] + 0.1, y - 0.1), width - 0.2, spacing * 0.8, 
                                  facecolor='white', edgecolor='black', alpha=0.5)
        ax.add_patch(subrect)
        ax.text(xy[0] + width/2, y, comp, ha='center', va='center', fontsize=8)

def create_arrow(ax, start, end):
    ax.arrow(start[0], start[1], end[0]-start[0], end[1]-start[1],
             head_width=0.1, head_length=0.1, fc='black', ec='black', length_includes_head=True)

# Create figure and axis
fig, ax = plt.subplots(figsize=(12, 6))

# Create boxes
create_box(ax, (1, 2), 2, 1, 'lightblue', 'Raw Data Input\n(Price, Volume, etc.)')
create_box(ax, (4, 2), 2, 1, 'lightgreen', 'CNN\nSpatial Feature\nExtraction')
create_box(ax, (7, 2), 2, 1, 'orange', 'LSTM\nTemporal Sequence\nAnalysis')
create_box(ax, (10, 2), 2, 1, 'plum', 'Trading Signals\n(Buy/Sell/Hold)')

# Create component boxes
cnn_components = ['Convolutional Layers', 'Pooling Layers', 'Feature Maps']
create_component_box(ax, (4, 0), 2, 1.5, 'lightgreen', 'CNN Components', cnn_components)

lstm_components = ['Memory Cells', 'Gates (Input/Forget/Output)', 'Hidden State']
create_component_box(ax, (7, 0), 2, 1.5, 'orange', 'LSTM Components', lstm_components)

# Create arrows
create_arrow(ax, (3, 2.5), (4, 2.5))
create_arrow(ax, (6, 2.5), (7, 2.5))
create_arrow(ax, (9, 2.5), (10, 2.5))

# Create dotted lines to components
ax.plot([5, 5], [2, 1.5], 'g--', alpha=0.5)
ax.plot([8, 8], [2, 1.5], 'orange', linestyle='--', alpha=0.5)

# Set axis properties
ax.set_xlim(0, 13)
ax.set_ylim(-0.5, 4)
ax.axis('off')

# Add title
plt.title('CNN-LSTM Architecture for Trading Signal Generation', pad=20, fontsize=14)

# Save the figure
plt.savefig('cnn_lstm_architecture.png', bbox_inches='tight', dpi=300)
plt.close() 