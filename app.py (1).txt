import streamlit as st
import numpy as np
import pandas as pd
import altair as alt
import graphviz

# --- Configuration ---
st.set_page_config(
    page_title="Architect of Attention",
    layout="wide",
    page_icon="🧠"
)

# --- Constants ---
STATES = ['Discovery', 'Rabbit Hole', 'Echo Chamber']
COLORS = {'Discovery': '#3b82f6', 'Rabbit Hole': '#8b5cf6', 'Echo Chamber': '#ef4444'}

# --- Math Functions ---
def multiply_vector_matrix(v, m):
    """v * M"""
    return np.dot(v, m)

def matrix_power(m, n):
    """M^n"""
    return np.linalg.matrix_power(m, n)

def check_irreducible(m):
    """
    Simplified check for 3x3: Check if (I+P)^2 has no zeros.
    If all entries are > 0, the graph is strongly connected (for this small size).
    """
    m_np = np.array(m)
    p2 = np.linalg.matrix_power(m_np, 2)
    p3 = np.linalg.matrix_power(m_np, 3)
    
    # Check if sum of paths of length 1, 2, 3 covers all transitions
    total_connectivity = m_np + p2 + p3
    return not np.any(total_connectivity == 0)

def check_aperiodic(m):
    """
    Simplified heuristic: If irreducible and has a self-loop (diagonal > 0), it is aperiodic.
    """
    m_np = np.array(m)
    return np.any(np.diagonal(m_np) > 0)

# --- Main App ---

def main():
    st.title("The Architect of Attention")
    st.markdown("**Probabilistic Modeling of Social Media Echo Chambers**")
    
    # --- Sidebar / Top Control Panel ---
    
    col_inputs, col_viz = st.columns([1, 2])
    
    with col_inputs:
        st.subheader("Simulation Parameters")
        
        # 1. Scalar Inputs
        c1, c2 = st.columns(2)
        with c1:
            time_steps = st.number_input("Time Steps (Days)", min_value=1, max_value=365, value=30)
        with c2:
            num_sims = st.number_input("Simulations (N)", min_value=1, value=100, help="Used for stochastic reference, though calculation is analytic.")

        # 2. Starting Vector Input
        st.markdown("### Starting Distribution ($\pi_0$)")
        # We use a helper dataframe for input
        default_start = pd.DataFrame([1.0, 0.0, 0.0], index=STATES, columns=["Probability"])
        
        # Using data_editor for vector
        start_df = st.data_editor(
            default_start.T, 
            column_config={
                "Discovery": st.column_config.NumberColumn(min_value=0, max_value=1, step=0.01),
                "Rabbit Hole": st.column_config.NumberColumn(min_value=0, max_value=1, step=0.01),
                "Echo Chamber": st.column_config.NumberColumn(min_value=0, max_value=1, step=0.01),
            },
            hide_index=True,
            key="start_vector"
        )
        start_vec = start_df.values.flatten()
        
        # Validation for Vector
        if not np.isclose(np.sum(start_vec), 1.0, atol=0.001):
            st.error(f"Starting vector sums to {np.sum(start_vec):.3f}. Must sum to 1.0")

        # 3. Transition Matrix Input
        st.markdown("### Transition Matrix ($P$)")
        default_matrix = pd.DataFrame(
            [
                [0.70, 0.25, 0.05],
                [0.15, 0.70, 0.15],
                [0.05, 0.05, 0.90]
            ],
            index=STATES,
            columns=STATES
        )
        
        matrix_df = st.data_editor(
            default_matrix,
            column_config={
                "Discovery": st.column_config.NumberColumn(min_value=0, max_value=1, step=0.01),
                "Rabbit Hole": st.column_config.NumberColumn(min_value=0, max_value=1, step=0.01),
                "Echo Chamber": st.column_config.NumberColumn(min_value=0, max_value=1, step=0.01),
            },
            key="matrix_input"
        )
        matrix = matrix_df.values
        
        # Validation for Matrix
        row_sums = np.sum(matrix, axis=1)
        valid_matrix = True
        for i, r_sum in enumerate(row_sums):
            if not np.isclose(r_sum, 1.0, atol=0.001):
                st.error(f"Row '{STATES[i]}' sums to {r_sum:.3f}. Must sum to 1.0")
                valid_matrix = False

        # 4. State Diagram
        st.markdown("### State Diagram")
        if valid_matrix:
            graph = graphviz.Digraph()
            graph.attr(rankdir='LR', size='8,5')
            
            for i, source in enumerate(STATES):
                # Node style
                color = COLORS[source]
                graph.node(source, style='filled', fillcolor='white', color=color, fontcolor='black')
                
                for j, target in enumerate(STATES):
                    weight = matrix[i][j]
                    if weight > 0.01:
                        # Edge style based on weight
                        penwidth = str(1 + weight * 3)
                        label = f"{weight:.2f}"
                        graph.edge(source, target, label=label, penwidth=penwidth, color='gray')
            
            st.graphviz_chart(graph)

    # --- Calculations & Visualization ---
    
    with col_viz:
        if valid_matrix and np.isclose(np.sum(start_vec), 1.0, atol=0.001):
            
            # 1. Properties Header
            is_irreducible = check_irreducible(matrix)
            is_aperiodic = check_aperiodic(matrix)
            
            status_cols = st.columns(4)
            status_cols[0].metric("Irreducible", "Yes" if is_irreducible else "No", delta_color="normal")
            status_cols[1].metric("Aperiodic", "Yes" if is_aperiodic else "No", delta_color="normal")
            
            # 2. Run Simulation (Analytic)
            history = []
            current_vec = start_vec
            
            # Day 0
            history.append({'Day': 0, **dict(zip(STATES, current_vec))})
            
            for d in range(1, time_steps + 1):
                current_vec = multiply_vector_matrix(current_vec, matrix)
                history.append({'Day': d, **dict(zip(STATES, current_vec))})
            
            df_history = pd.DataFrame(history)
            
            # 3. Line Chart
            st.subheader("State Probabilities Over Time")
            
            # Melt for Altair
            df_melted = df_history.melt('Day', var_name='State', value_name='Probability')
            
            # Custom color scale
            domain = STATES
            range_colors = [COLORS[s] for s in STATES]
            
            # Base chart
            base = alt.Chart(df_melted).encode(
                x='Day',
                y=alt.Y('Probability', scale=alt.Scale(domain=[0, 1])),
                color=alt.Color('State', scale=alt.Scale(domain=domain, range=range_colors))
            )

            # Lines - Thick red line logic handled by strokeWidth condition
            lines = base.mark_line().encode(
                strokeWidth=alt.condition(
                    alt.datum.State == 'Echo Chamber',
                    alt.value(4),  # Thick line for Echo Chamber
                    alt.value(2)   # Thin for others
                ),
                tooltip=['Day', 'State', alt.Tooltip('Probability', format='.4f')]
            )
            
            st.altair_chart(lines.interactive(), use_container_width=True)
            
            # 4. Detailed Day View
            st.subheader("Deep Dive: Single Day Analysis")
            selected_day = st.slider("Select Day", 0, time_steps, 0)
            
            # Get data for selected day
            day_data = df_history[df_history['Day'] == selected_day].iloc[0]
            
            c_bar, c_matrix = st.columns(2)
            
            with c_bar:
                st.markdown(f"**Market Distribution (Day {selected_day})**")
                # Prepare data for bar chart
                bar_data = pd.DataFrame({
                    'State': STATES,
                    'Probability': [day_data[s] for s in STATES]
                })
                
                bar_chart = alt.Chart(bar_data).mark_bar().encode(
                    x=alt.X('State', sort=None),
                    y=alt.Y('Probability', scale=alt.Scale(domain=[0, 1])),
                    color=alt.Color('State', scale=alt.Scale(domain=domain, range=range_colors)),
                    tooltip=[alt.Tooltip('Probability', format='.4f')]
                )
                st.altair_chart(bar_chart, use_container_width=True)
                
            with c_matrix:
                st.markdown(f"**Transition Matrix ($P^{{{selected_day}}}$)**")
                matrix_n = matrix_power(matrix, selected_day)
                st.dataframe(pd.DataFrame(matrix_n, index=STATES, columns=STATES).style.format("{:.4f}"))
                
                # Architect's Notes Logic
                note = ""
                if selected_day == 0:
                    note = "Day 0 represents the initial state. The matrix shown is the Identity matrix (P^0)."
                elif selected_day > 0 and selected_day < 5:
                    note = "Early simulation phase. Users are beginning to migrate based on transition probabilities."
                elif selected_day >= 5 and not is_irreducible:
                    note = "⚠️ Warning: The model contains absorbing states or trap sectors."
                elif selected_day > 20 and is_irreducible:
                    note = "The system is approaching Equilibrium. Notice how the rows of P^n become identical."
                
                if note:
                    st.info(note)

if __name__ == "__main__":
    main()
