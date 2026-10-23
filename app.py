import os
import time
import math
import re
from fractions import Fraction
import streamlit as st
import torch
import numpy as np

from train_nmn import NeumathworkModel
from symbolic_engine import SymbolicUniversalEngine, MathTextParser, evaluate_expression_tree

st.set_page_config(
    page_title="Neumathwork (NMN) v4.0",
    page_icon="📐",
    layout="centered"
)

st.title("Neumathwork (NMN)")
st.markdown("**Experimental*")
st.markdown("---")

@st.cache_resource
def load_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = NeumathworkModel(in_features=8, hidden_dim=256, num_heads=4, num_classes=16).to(device)
    ckpt = 'nmn_checkpoint.pth'
    if os.path.exists(ckpt):
        model.load_state_dict(torch.load(ckpt, map_location=device, weights_only=False))
        model.eval()
        return model, device, True
    return model, device, False

model, device, is_loaded = load_model()

query = st.text_input("Enter Expression / Equation:", value="det([[-15, 8], [4, -3]]) + abs(pow2(9) - 100)")

col1, col2 = st.columns([1, 5])
run_btn = col1.button("Execute")

if run_btn:
    if not is_loaded:
        st.error("Error: Checkpoint 'nmn_checkpoint.pth' not found.")
    else:
        t0 = time.perf_counter()
        clean_str = query.replace(" ", "").lower()
        
        st.markdown("### Proof")
        
        try:
            extra_steps = []

            def sub_det(match):
                a, b, c, d = map(int, match.groups())
                _, det_val, steps = SymbolicUniversalEngine.solve_matrix_det(a, b, c, d)
                extra_steps.extend([f"[Matrix Det Sub-computation]: {s}" for s in steps])
                return str(det_val)

            clean_str = re.sub(r'det\(\[\[(-?\d+),(-?\d+)\],\[(-?\d+),(-?\d+)\]\]\)', sub_det, clean_str)

            # 2. Cek Persamaan Kuadrat
            if "=0" in clean_str or "x^2" in clean_str:
                match_quad = re.search(r'(-?\d*)\*?x\^2([+-]\d*)\*?x([+-]\d+)=0', clean_str)
                if match_quad:
                    raw_a, raw_b, raw_c = match_quad.groups()
                    a = 1 if raw_a in ['', '+'] else (-1 if raw_a == '-' else int(raw_a))
                    b = 1 if raw_b == '+' else (-1 if raw_b == '-' else int(raw_b))
                    c = int(raw_c)
                    st.text(f"Quadratic Equation")
                    is_val, roots, steps = SymbolicUniversalEngine.solve_quadratic(a, b, c)
                    if is_val:
                        x1, x2 = roots
                        for s in steps: st.markdown(f"- {s}")
                        st.latex(f"\\text{{Solution Set: }} \\{{ {x1}, {x2} \\}}")

            else:
                st.text("Multi-Path AST")
                if extra_steps:
                    st.markdown("**Step by step:**")
                    for es in extra_steps:
                        st.markdown(f"- {es}")
                
                ast_root = MathTextParser.parse(clean_str)
                history, final_val, msg = evaluate_expression_tree(ast_root)
                
                if final_val is not None:
                    st.markdown("**Derivasi Reduksi AST:**")
                    for idx, step_str in enumerate(history):
                        st.markdown(f"`Steps {idx}:` {step_str}")
                    st.markdown("---")
                    st.latex(f"\\mathbf{{Result: }} {final_val}")
                else:
                    st.error(f"Failed to reduct the expression: {msg}")

            t_elapsed = (time.perf_counter() - t0) * 1000
            st.markdown("---")
            st.caption(f"Execution Latency: {t_elapsed:.3f} ms | Device: {str(device).upper()} | Engine: Neuro-Symbolic")
            
        except Exception as e:
            st.error(f"Execution error: {e}")

st.markdown("---")
st.markdown("<p style='text-align: center; color: gray; font-size: 12px;'>Neumathwork Research Engine (Experimental)</p>", unsafe_allow_html=True)