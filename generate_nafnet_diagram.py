import matplotlib.pyplot as plt
import matplotlib.patches as patches

def draw_block(ax, x, y, width, height, label, color, fontsize=10):
    rect = patches.FancyBboxPatch((x, y), width, height, boxstyle="round,pad=0.1", 
                                  linewidth=2, edgecolor='black', facecolor=color)
    ax.add_patch(rect)
    ax.text(x + width/2, y + height/2, label, ha='center', va='center', 
            fontsize=fontsize, fontweight='bold', color='white')

def create_nafnet_diagram():
    fig, ax = plt.subplots(figsize=(14, 8))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 10)
    ax.axis('off')
    
    # Colors
    c_enc = '#3498DB' # Blue
    c_mid = '#E67E22' # Orange
    c_dec = '#2ECC71' # Green
    c_blk = '#9B59B6' # Purple (NAFBlock detail)
    
    # ---------------------------------------------------------
    # 1. U-Net Macro Architecture
    # ---------------------------------------------------------
    
    # Encoder
    draw_block(ax, 1, 8, 2, 1, "Encoder 1\n(2 Blocks)", c_enc)
    draw_block(ax, 1, 6, 2, 1, "Encoder 2\n(2 Blocks)", c_enc)
    draw_block(ax, 1, 4, 2, 1, "Encoder 3\n(4 Blocks)", c_enc)
    draw_block(ax, 1, 2, 2, 1, "Encoder 4\n(8 Blocks)", c_enc)
    
    # Downsample Arrows
    ax.annotate("", xy=(2, 7.1), xytext=(2, 7.9), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(2, 5.1), xytext=(2, 5.9), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(2, 3.1), xytext=(2, 3.9), arrowprops=dict(arrowstyle="->", lw=2))
    
    # Middle
    draw_block(ax, 4, 1, 4, 1, "Middle (Bottleneck)\n(8 Blocks)", c_mid)
    
    # Connection Enc -> Middle
    ax.annotate("", xy=(4, 1.5), xytext=(3.1, 2.5), arrowprops=dict(arrowstyle="->", lw=2, connectionstyle="arc3,rad=0.2"))

    # Decoder
    draw_block(ax, 9, 2, 2, 1, "Decoder 4\n(2 Blocks)", c_dec)
    draw_block(ax, 9, 4, 2, 1, "Decoder 3\n(2 Blocks)", c_dec)
    draw_block(ax, 9, 6, 2, 1, "Decoder 2\n(2 Blocks)", c_dec)
    draw_block(ax, 9, 8, 2, 1, "Decoder 1\n(2 Blocks)", c_dec)

    # Upsample Arrows
    ax.annotate("", xy=(10, 3.9), xytext=(10, 3.1), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(10, 5.9), xytext=(10, 5.1), arrowprops=dict(arrowstyle="->", lw=2))
    ax.annotate("", xy=(10, 7.9), xytext=(10, 7.1), arrowprops=dict(arrowstyle="->", lw=2))
    
    # Connection Middle -> Dec
    ax.annotate("", xy=(9, 2.5), xytext=(8.1, 1.5), arrowprops=dict(arrowstyle="->", lw=2, connectionstyle="arc3,rad=0.2"))

    # Skip Connections
    ax.annotate("Skip", xy=(8.9, 2.5), xytext=(3.1, 2.5), arrowprops=dict(arrowstyle="->", lw=1.5, ls="--", color='gray'))
    ax.annotate("Skip", xy=(8.9, 4.5), xytext=(3.1, 4.5), arrowprops=dict(arrowstyle="->", lw=1.5, ls="--", color='gray'))
    ax.annotate("Skip", xy=(8.9, 6.5), xytext=(3.1, 6.5), arrowprops=dict(arrowstyle="->", lw=1.5, ls="--", color='gray'))
    ax.annotate("Skip", xy=(8.9, 8.5), xytext=(3.1, 8.5), arrowprops=dict(arrowstyle="->", lw=1.5, ls="--", color='gray'))

    # ---------------------------------------------------------
    # 2. Micro Architecture (NAFBlock Detail)
    # ---------------------------------------------------------
    
    # Container
    rect = patches.FancyBboxPatch((4.5, 4), 3, 5, boxstyle="round,pad=0.2", 
                                  linewidth=2, edgecolor='#555', facecolor='#F5F5F5', alpha=0.9)
    ax.add_patch(rect)
    ax.text(6, 9.2, "Inside a NAFBlock", ha='center', fontsize=12, fontweight='bold')

    # Components
    draw_block(ax, 5, 8.2, 2, 0.5, "LayerNorm", c_blk, 8)
    ax.annotate("", xy=(6, 8.2), xytext=(6, 8.8), arrowprops=dict(arrowstyle="->"))
    
    draw_block(ax, 5, 7.2, 2, 0.5, "Conv 1x1", c_blk, 8)
    ax.annotate("", xy=(6, 7.2), xytext=(6, 7.7), arrowprops=dict(arrowstyle="->"))

    draw_block(ax, 5, 6.2, 2, 0.5, "DwConv 3x3", c_blk, 8)
    ax.annotate("", xy=(6, 6.2), xytext=(6, 6.7), arrowprops=dict(arrowstyle="->"))
    
    draw_block(ax, 5, 5.2, 2, 0.5, "SimpleGate", '#E74C3C', 8)
    ax.annotate("", xy=(6, 5.2), xytext=(6, 5.7), arrowprops=dict(arrowstyle="->"))
    
    draw_block(ax, 5, 4.2, 2, 0.5, "SCA (Attention)", '#F1C40F', 8)
    ax.annotate("", xy=(6, 4.2), xytext=(6, 4.7), arrowprops=dict(arrowstyle="->"))

    # Title
    plt.text(7, 0.2, "NAFNet Architecture (trainingNafnet.py)", ha='center', fontsize=16, fontweight='bold', color='#333333')

    # Save
    output_path = 'nafnet_architecture.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    return output_path

if __name__ == "__main__":
    path = create_nafnet_diagram()
    print(f"Diagram created at: {path}")
