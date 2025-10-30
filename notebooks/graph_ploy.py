from graphviz import Digraph

# Create a new directed graph
dot = Digraph(comment='MMoE Architecture')
dot.attr(rankdir='TB', newrank='true', splines='ortho', fontname='Arial', fontsize='12')
dot.attr('node', fontname='Arial', fontsize='11', style='filled', margin='0.2,0.1')
dot.attr('edge', fontname='Arial', fontsize='9', penwidth='2')

# --- 1. Input & Backbone (Frozen) ---
with dot.subgraph(name='cluster_input') as c:
    c.attr(label='Input & Frozen Backbone', style='filled,rounded', color='#E8EAF6', 
           fontsize='13', fontcolor='#1A237E', labeljust='l', penwidth='2')
    c.node('input', 'Input Image', shape='box', fillcolor='#C5CAE9', color='#3F51B5', 
           fontcolor='#1A237E', style='filled,rounded')
    c.node('backbone', 'Frozen DINOv3 (ViT-L)', shape='box', fillcolor='#9FA8DA', 
           color='#3F51B5', fontcolor='#1A237E', style='filled,rounded', penwidth='2')
    c.node('features', 'Features (x) [B, D]', shape='box', fillcolor='#7986CB', 
           color='#3F51B5', fontcolor='white', style='filled,rounded,dashed', penwidth='2')
    c.edge('input', 'backbone', color='#5C6BC0', penwidth='2')
    c.edge('backbone', 'features', color='#5C6BC0', penwidth='2')

# --- 2. MMoE Layer (Trainable) ---
with dot.subgraph(name='cluster_mmoe') as c:
    c.attr(label='Trainable MMoE Layer', style='filled,rounded', color='#FFF9C4', 
           fontsize='13', fontcolor='#F57F17', labeljust='l', penwidth='2')
    
    # Experts
    with dot.subgraph(name='cluster_experts') as e:
        e.attr(label='Experts', rank='same', style='filled,rounded', color='#FFF59D',
               fontsize='11', fontcolor='#F57F17')
        e.node('e1', 'Expert 1\n(MLP)', shape='box', fillcolor='#FFE082', 
               color='#FFA726', fontcolor='#E65100', style='filled,rounded', penwidth='2')
        e.node('e_dots', '...', shape='plaintext', fillcolor='transparent', fontsize='16')
        e.node('eN', 'Expert E\n(MLP)', shape='box', fillcolor='#FFE082', 
               color='#FFA726', fontcolor='#E65100', style='filled,rounded', penwidth='2')
    
    # Gates
    with dot.subgraph(name='cluster_gates') as g:
        g.attr(label='Gates (1 per task)', rank='same', style='filled,rounded', 
               color='#FFECB3', fontsize='11', fontcolor='#F57F17')
        g.node('g_family', 'Gate_Family\n(Softmax)', shape='diamond', fillcolor='#FFCA28', 
               color='#FF8F00', fontcolor='#E65100', style='filled', penwidth='2')
        g.node('g_order', 'Gate_Order\n(Softmax)', shape='diamond', fillcolor='#FFCA28', 
               color='#FF8F00', fontcolor='#E65100', style='filled', penwidth='2')
        g.node('g_habitat', 'Gate_Habitat\n(Softmax)', shape='diamond', fillcolor='#FFCA28', 
               color='#FF8F00', fontcolor='#E65100', style='filled', penwidth='2')
        g.node('g_troph', 'Gate_Troph\n(Softmax)', shape='diamond', fillcolor='#FFCA28', 
               color='#FF8F00', fontcolor='#E65100', style='filled', penwidth='2')

    # Expert Outputs (invisible nodes for layout)
    dot.node('h1', '', shape='point', width='0')
    dot.node('hN', '', shape='point', width='0')
    
    # Connections from features to experts and gates
    dot.edge('features', 'e1', lhead='cluster_experts', color='#7986CB', penwidth='2')
    dot.edge('features', 'eN', lhead='cluster_experts', color='#7986CB', penwidth='2')
    dot.edge('features', 'g_family', lhead='cluster_gates', color='#7986CB', penwidth='2')
    dot.edge('features', 'g_troph', lhead='cluster_gates', color='#7986CB', penwidth='2')
    
    # Show expert outputs
    dot.edge('e1', 'h1', color='#FFA726', penwidth='2')
    dot.edge('eN', 'hN', color='#FFA726', penwidth='2')


# --- 3. Towers & Outputs (Trainable) ---
with dot.subgraph(name='cluster_output') as c:
    c.attr(label='Trainable Task Towers & Outputs', style='filled,rounded', 
           color='#E1F5FE', fontsize='13', fontcolor='#01579B', labeljust='l', penwidth='2')
    
    # Invisible nodes for weighted sum - with colors for each task
    dot.node('mix_family', 'Weighted\nSum', shape='circle', fillcolor='#FF8A80', 
             color='#D32F2F', fontcolor='white', style='filled', penwidth='2', fixedsize='true', width='1.2')
    dot.node('mix_order', 'Weighted\nSum', shape='circle', fillcolor='#82B1FF', 
             color='#1976D2', fontcolor='white', style='filled', penwidth='2', fixedsize='true', width='1.2')
    dot.node('mix_habitat', 'Weighted\nSum', shape='circle', fillcolor='#B9F6CA', 
             color='#388E3C', fontcolor='#1B5E20', style='filled', penwidth='2', fixedsize='true', width='1.2')
    dot.node('mix_troph', 'Weighted\nSum', shape='circle', fillcolor='#EA80FC', 
             color='#7B1FA2', fontcolor='white', style='filled', penwidth='2', fixedsize='true', width='1.2')

    # Final Tower Heads
    dot.node('t_family', 'Tower_Family\n(MLP)', shape='box', fillcolor='#FFCDD2', 
             color='#D32F2F', fontcolor='#B71C1C', style='filled,rounded', penwidth='2')
    dot.node('t_order', 'Tower_Order\n(MLP)', shape='box', fillcolor='#BBDEFB', 
             color='#1976D2', fontcolor='#0D47A1', style='filled,rounded', penwidth='2')
    dot.node('t_habitat', 'Tower_Habitat\n(MLP)', shape='box', fillcolor='#C8E6C9', 
             color='#388E3C', fontcolor='#1B5E20', style='filled,rounded', penwidth='2')
    dot.node('t_troph', 'Tower_Troph\n(MLP)', shape='box', fillcolor='#E1BEE7', 
             color='#7B1FA2', fontcolor='#4A148C', style='filled,rounded', penwidth='2')

    # Final Outputs
    dot.node('out_family', 'Family\nLogits', shape='box', fillcolor='#EF5350', 
             color='#C62828', fontcolor='white', style='filled,rounded,bold', penwidth='3')
    dot.node('out_order', 'Order\nLogits', shape='box', fillcolor='#42A5F5', 
             color='#1565C0', fontcolor='white', style='filled,rounded,bold', penwidth='3')
    dot.node('out_habitat', 'Habitat\nLogits', shape='box', fillcolor='#66BB6A', 
             color='#2E7D32', fontcolor='white', style='filled,rounded,bold', penwidth='3')
    dot.node('out_troph', 'Troph\nPrediction', shape='box', fillcolor='#AB47BC', 
             color='#6A1B9A', fontcolor='white', style='filled,rounded,bold', penwidth='3')
    
    # Connections for Family (Red theme)
    dot.edge('h1', 'mix_family', label='w_fam_1', style='dotted', arrowhead='none', 
             color='#FFA726', fontcolor='#E65100')
    dot.edge('hN', 'mix_family', label='w_fam_E', style='dotted', arrowhead='none', 
             color='#FFA726', fontcolor='#E65100')
    dot.edge('g_family', 'mix_family', style='dashed', color='#FF6F00', penwidth='2')
    dot.edge('mix_family', 't_family', color='#D32F2F', penwidth='3')
    dot.edge('t_family', 'out_family', color='#C62828', penwidth='3')

    # Connections for Order (Blue theme)
    dot.edge('h1', 'mix_order', style='dotted', arrowhead='none', color='#FFA726')
    dot.edge('hN', 'mix_order', style='dotted', arrowhead='none', color='#FFA726')
    dot.edge('g_order', 'mix_order', style='dashed', color='#FF6F00', penwidth='2')
    dot.edge('mix_order', 't_order', color='#1976D2', penwidth='3')
    dot.edge('t_order', 'out_order', color='#1565C0', penwidth='3')

    # Connections for Habitat (Green theme)
    dot.edge('h1', 'mix_habitat', style='dotted', arrowhead='none', color='#FFA726')
    dot.edge('hN', 'mix_habitat', style='dotted', arrowhead='none', color='#FFA726')
    dot.edge('g_habitat', 'mix_habitat', style='dashed', color='#FF6F00', penwidth='2')
    dot.edge('mix_habitat', 't_habitat', color='#388E3C', penwidth='3')
    dot.edge('t_habitat', 'out_habitat', color='#2E7D32', penwidth='3')
    
    # Connections for Troph (Purple theme)
    dot.edge('h1', 'mix_troph', style='dotted', arrowhead='none', color='#FFA726')
    dot.edge('hN', 'mix_troph', style='dotted', arrowhead='none', color='#FFA726')
    dot.edge('g_troph', 'mix_troph', style='dashed', color='#FF6F00', penwidth='2')
    dot.edge('mix_troph', 't_troph', color='#7B1FA2', penwidth='3')
    dot.edge('t_troph', 'out_troph', color='#6A1B9A', penwidth='3')


# Render the graph
# This will save a PDF and a source file, and open the PDF viewer
dot.render('mmoe_architecture', view=False, format='png')

print("Generated mmoe_architecture.png")