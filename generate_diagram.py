import matplotlib.pyplot as plt
import matplotlib.patches as patches

def create_conclusion_diagram():
    # Setup figure
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 6)
    ax.axis('off')

    # Styles
    # Removed boxstyle from here to avoid conflict
    base_props = dict(ec='black', lw=2)
    arrow_props = dict(arrowstyle='->', lw=2, color='#333333')
    
    # 1. Start: Input Image
    ax.text(1, 3, "Degraded\nImage", ha='center', va='center', fontsize=12,
            bbox=dict(boxstyle='square,pad=0.5', fc='#E0E0E0', ec='gray', lw=1))
    
    # Arrow 1
    ax.annotate("", xy=(2.5, 3), xytext=(1.5, 3), arrowprops=arrow_props)

    # 2. Restoration (NAFNet)
    ax.text(3.5, 3, "NAFNet\n(Restoration)", ha='center', va='center', fontsize=12, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.7', fc='#007ACC', **base_props))

    # Split Arrows
    ax.annotate("", xy=(5.5, 4.5), xytext=(4.5, 3.2), arrowprops=dict(arrowstyle='->', lw=2, color='#333333', connectionstyle="arc3,rad=0.2"))
    ax.annotate("", xy=(5.5, 1.5), xytext=(4.5, 2.8), arrowprops=dict(arrowstyle='->', lw=2, color='#333333', connectionstyle="arc3,rad=-0.2"))

    # 3. Diagnosis (ResNet18) - Top Branch
    ax.text(6.5, 4.5, "ResNet18\n(Diagnosis)", ha='center', va='center', fontsize=12, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.7', fc='#8E44AD', **base_props))
    
    # 3. Analysis (Cellpose) - Bottom Branch
    ax.text(6.5, 1.5, "Cellpose\n(Analysis)", ha='center', va='center', fontsize=12, fontweight='bold', color='white',
            bbox=dict(boxstyle='round,pad=0.7', fc='#27AE60', **base_props))

    # Merge Arrows
    ax.annotate("", xy=(8.5, 3.2), xytext=(7.5, 4.5), arrowprops=dict(arrowstyle='->', lw=2, color='#333333', connectionstyle="arc3,rad=0.2"))
    ax.annotate("", xy=(8.5, 2.8), xytext=(7.5, 1.5), arrowprops=dict(arrowstyle='->', lw=2, color='#333333', connectionstyle="arc3,rad=-0.2"))

    # 4. Final Result
    ax.text(9.5, 3, "Final\nDiagnosis", ha='center', va='center', fontsize=12, fontweight='bold', color='black',
            bbox=dict(boxstyle='round,pad=0.7', fc='#FFD700', **base_props))

    # Labels for context (Problem/Solution)
    plt.text(6, 0.2, "Our Solution: A Complete AI Pipeline", ha='center', fontsize=14, fontweight='bold', color='#333333')

    # Save
    output_path = 'conclusion_diagram.png'
    plt.savefig(output_path, dpi=300, bbox_inches='tight', transparent=True)
    return output_path

if __name__ == "__main__":
    path = create_conclusion_diagram()
    print(f"Diagram created at: {path}")
