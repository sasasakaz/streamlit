import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import itertools

# --- ページ設定 ---
st.set_page_config(page_title="Bayesian 5-Group Comparison", layout="wide")
st.title("📊 ベイズ統計：5群比較分析ツール")

# --- サイドバー ---
with st.sidebar:
    st.header("1. データ入力")
    input_mode = st.radio("入力方法", ["サンプルデータ", "テキスト入力"])
    
    x_list = []
    if input_mode == "サンプルデータ":
        x_A = [14.63, 20.18, 19.85, 20.86, 20.01, 16.96, 17.66, 18.46, 16.19, 15.76, 14.84, 20.56,
 15.09, 17.18]
        x_B = [14.34, 13.09, 14.62, 14.31, 19.25, 13.03, 13.56, 13.07, 17.93, 15.09, 14.56, 15.51,
 15.06, 17.36]
        x_C = [20.43, 24.43, 24.2,  17.74, 20.75, 18.87, 18.13, 17.01, 19.4,  17.37, 17.67, 23.76,
 18.99, 23.72]
        x_D = [6.75, 17.33, 18.26, 20.43, 17.38, 17.34, 20.08, 16.26, 16.19, 18.88, 16.96, 20.56,
 15.53, 16.75]
        x_E = [14.54, 15.4,  12.52, 18.74, 12.64, 16.48, 13.61, 16.12, 13.93, 17.89, 12.44, 12.51,
 17.24, 14.29]
        x_list = [x_A, x_B, x_C, x_D, x_E]
    else:
        for i in range(5):
            raw_input = st.text_area(f"群 {i+1}", "1.0, 1.2, 1.1", key=f"input_{i}")
            x_list.append([float(x.strip()) for x in raw_input.split(",") if x.strip()])

    st.header("2. ハイパーパラメータ")
    tune = st.number_input("Tune", value=1000)
    draws = st.number_input("Draws", value=1000)

# --- モデル ---
if st.button("ベイズ推定を開始"):
    all_data = np.concatenate(x_list)
    pooled_mean = all_data.mean()
    pooled_std = all_data.std() * 2

    with st.spinner("サンプリング中..."):
        with pm.Model() as model_t:
            nu_minus_1 = pm.Exponential('nu_minus_1', 1/29) + 1
            
            mus, stds = [], []
            for i in range(5):
                mu = pm.Normal(f'mu_{i}', mu=pooled_mean, sigma=pooled_std)
                std = pm.Uniform(f'std_{i}', lower=0.1, upper=10)

                pm.StudentT(f'obs_{i}', mu=mu, sigma=std, nu=nu_minus_1, observed=x_list[i])

                mus.append(mu)
                stds.append(std)

            for i, j in itertools.combinations(range(5), 2):
                pm.Deterministic(f'diff_{i}_{j}', mus[i] - mus[j])
                pm.Deterministic(f'es_{i}_{j}', (mus[i] - mus[j]) / np.sqrt((stds[i]**2 + stds[j]**2) / 2))

            trace = pm.sample(tune=tune, draws=draws, chains=4, return_inferencedata=True)

        st.session_state['trace'] = trace
        st.success("完了✨")

# --- 結果表示 ---
if 'trace' in st.session_state:
    trace = st.session_state['trace']

    mu_vars = [f"mu_{i}" for i in range(5)]
    diff_vars = [f"diff_{i}_{j}" for i, j in itertools.combinations(range(5), 2)]

    tab1, tab2, tab3 = st.tabs(["📈 全体俯瞰", "🔍 2群比較", "🛠️ 診断"])

    # =====================
    # 📈 全体俯瞰
    # =====================
    with tab1:
        st.subheader("平均の比較")

        # 各群の平均値の分布を比較
        fig, ax = plt.subplots(figsize=(10,4))
        colors = plt.cm.tab10.colors
        for i, v in enumerate(mu_vars):
            az.plot_kde(trace.posterior[v].values.flatten(), ax=ax, label=v, 
                        plot_kwargs={"color": colors[i]})
        ax.legend()
        ax.set_title("Posterior Distribution of Means")
        ax.set_xlabel("Value")
        ax.set_ylabel("Density")
        st.pyplot(fig)

        # 各群の平均値の分布を個別に表示
        fig, axes = plt.subplots(2, 3, figsize=(12, 6))
        axes = axes.flatten()
        colors = plt.cm.tab10.colors
        for i, v in enumerate(mu_vars):
            az.plot_posterior(
                trace,
                var_names=[v],
                ax=axes[i],
                color=colors[i]
            )
            axes[i].set_title(v)

        # 余りを消す
        for j in range(len(mu_vars), 6):
            axes[j].axis("off")

        plt.tight_layout()
        st.pyplot(fig)

    # =====================
    # 🔍 2群比較
    # =====================
    with tab2:
        st.subheader("2群比較")

        i = st.selectbox("群A", range(5))
        j = st.selectbox("群B", range(5), index=1)

        if i != j:
            name = f"diff_{min(i,j)}_{max(i,j)}"

            fig, ax = plt.subplots(figsize=(6,4))
            az.plot_posterior(trace, var_names=[name], ref_val=0, ax=ax)
            st.pyplot(fig)

            samples = trace.posterior[name].values.flatten()

            st.write(f"平均差: {samples.mean():.3f}")
            st.write(f"P(>0): {(samples > 0).mean()*100:.1f}%")

    # =====================
    # 🛠️ 診断
    # =====================
    with tab3:
        st.subheader("収束診断")

        with st.expander("Summary"):
            st.dataframe(az.summary(trace, var_names=mu_vars))

        with st.expander("Trace Plot"):
            axes = az.plot_trace(trace, var_names=mu_vars)
            fig = axes.ravel()[0].figure
            st.pyplot(fig)