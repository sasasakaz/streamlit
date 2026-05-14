import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pymc as pm
import arviz as az
import re
import itertools

# --- ページ設定 ---
st.set_page_config(page_title="Bayesian Estimation of the Difference in Means", layout="wide", initial_sidebar_state="auto")
st.title("📊 平均の差のベイズ推定")

# --- サイドバー：データ入力 ---
with st.sidebar:
    st.header("データ入力")

    group_names = ["A", "B", "C", "D", "E"]

    # デフォルトのサンプルデータ
    default_data = [
        "14.63, 20.18, 19.85, 20.86, 20.01, 16.96, 17.66, 18.46, 16.19, 15.76, 14.84, 20.56, 15.09, 17.18",
        "14.34, 13.09, 14.62, 14.31, 19.25, 13.03, 13.56, 13.07, 17.93, 15.09, 14.56, 15.51, 15.06, 17.36",
        "20.43, 24.43, 24.2, 17.74, 20.75, 18.87, 18.13, 17.01, 19.4, 17.37, 17.67, 23.76, 18.99, 23.72",
        "6.75, 17.33, 18.26, 20.43, 17.38, 17.34, 20.08, 16.26, 16.19, 18.88, 16.96, 20.56, 15.53, 16.75",
        "14.54, 15.4, 12.52, 18.74, 12.64, 16.48, 13.61, 16.12, 13.93, 17.89, 12.44, 12.51, 17.24, 14.29"
    ]

    # 1. コールバック関数を定義
    def clear_all():
        for name in group_names:
            st.session_state[f"input_{name}"] = ""

    def set_sample():
        for name, default in zip(group_names, default_data):
            st.session_state[f"input_{name}"] = default

    # 2. ボタンを横並びに配置
    # col1, col2 = st.columns(2)
    col1, col2 = st.columns([2, 1]) # 比率を2:1にする
    with col1:
        st.button("⚡️ サンプルデータを入力する", on_click=set_sample, use_container_width=True)
    with col2:
        st.button("🗑️ clear all", on_click=clear_all, use_container_width=True)

    x_list = []
    for name in group_names:
        key_name = f"input_{name}"
        
        # 初回起動時はテキストエリアは空にする
        if key_name not in st.session_state:
            # st.session_state[key_name] = default  # デフォルトデータをセットする場合
            st.session_state[key_name] = ""

        # テキストエリアを表示
        raw_input = st.text_area(f"Group {name}", key=key_name, height=100)
        
        # 1. 桁区切りカンマの処理: 数字に挟まれたカンマを除去
        # 桁区切りの書式設定セルからコピペ入力された場合の対応策
        cleaned_input = re.sub(r'(?<=\d),(?=\d)', '', raw_input)
        # 2. 数値の抽出: 正規表現 [^0-9.-]+ を使って、数字・ドット・マイナス記号「以外」の文字で分割
        # 入力データの区切り形式が、カンマ、タブ、スペース、改行のどれでも対応するため
        parsed_data = [float(x) for x in re.split(r'[^0-9.-]+', cleaned_input) if x.strip()]
        x_list.append(parsed_data)

    st.write("---")

    with st.expander("推定がうまくいかないとき用"):
        st.info("増やしてみるとうまくいく場合もあります")
        tune = st.number_input("Tune", value=1000, step=1000, min_value=1000, max_value=4000)
        draws = st.number_input("Draws", value=1000, step=1000, min_value=1000, max_value=4000)



# --- ベイズ推定実行 ---
st.write("A/Bテスト結果を入力して、クリックボタンを押してください。")

# 1. 入力がある群だけを特定する
valid_indices = [i for i, data in enumerate(x_list) if len(data) > 0]
valid_group_names = [group_names[i] for i in valid_indices]
valid_x_list = [x_list[i] for i in valid_indices]

# --- チェックと警告（サンプリング自体は走る） ---
if len(valid_x_list) > 0:
    all_values = np.concatenate(valid_x_list)
    if np.any(np.abs(all_values) > 1000):
        st.warning("⚠️ 入力値に大きな数字が含まれています。推定がうまくいかない場合あるため、入力データがいずれも1000未満になるように一律で桁下げすることを推奨します")

if st.button("🚀 Click to Estimate"):
    # 2群以上ないと比較できないのでチェック
    if len(valid_x_list) < 2:
        st.error("⚠️ 推定には少なくとも2つの群にデータ入力が必要です。手元にデータがなければサンプルデータ入力ボタンを押してみてください。")
    else:

        # --- 観測データの集計結果を表示 ---
        st.write("入力データの集計値")

        # リスト内包表記で各群の統計量を計算して辞書のリストを作る
        summary_data = []
        for name, data in zip(valid_group_names, valid_x_list):
            summary_data.append({
                "Group": name,
                "データ件数": len(data),
                "平均値 (Mean)": np.mean(data),
                "標準偏差 (Std)": np.std(data),
                "最小値": np.min(data),
                "最大値": np.max(data)
            })

        # DataFrameに変換して表示
        obs_df = pd.DataFrame(summary_data).set_index("Group")
        st.table(obs_df.style.format({
            "平均値 (Mean)": "{:.2f}",
            "標準偏差 (Std)": "{:.2f}",
            "最小値": "{:.2f}",
            "最大値": "{:.2f}"
        }))


        # ベイズ推定の実行
        all_data = np.concatenate(valid_x_list)
        pooled_mean = all_data.mean()
        pooled_std = all_data.std() * 2

        with st.spinner(f"{len(valid_group_names)}群でベイズ推定の実行中... MCMCサンプリングが終わるまで待ってにゃん"):
            with pm.Model() as model_t:
                nu = pm.Exponential('nu', 1/29) + 1
                
                mus = []
                stds = []
                
                # 入力があった群の数だけループを回す
                for i, name in enumerate(valid_group_names):
                    mu = pm.Normal(f'mu_{name}', mu=pooled_mean, sigma=pooled_std)
                    std = pm.Uniform(f'std_{name}', lower=0.1, upper=10)
                    
                    pm.StudentT(f'obs_{name}', mu=mu, sigma=std, nu=nu, observed=valid_x_list[i])
                    
                    mus.append(mu)
                    stds.append(std)

                # 入力があった群同士の全組み合わせで差分を計算
                for (idx1, name1), (idx2, name2) in itertools.combinations(enumerate(valid_group_names), 2):
                    pm.Deterministic(f'diff_{name1}_{name2}', mus[idx1] - mus[idx2])

                trace = pm.sample(tune=tune, draws=draws, chains=4, return_inferencedata=True, random_seed=42, # target_accept=0.95
                                  )

            # どの群で計算したかを後で使うためにsession_stateに保存
            st.session_state['valid_group_names'] = valid_group_names
            st.session_state['trace'] = trace
            st.success("推定完了！ 解説やPointを参考にしながら推定結果を読み解いてください")

# --- 結果表示 ---
if 'trace' in st.session_state:
    trace = st.session_state['trace']
    current_groups = st.session_state['valid_group_names']  # 計算に使った群名を取得
    mu_vars = [f"mu_{name}" for name in current_groups]
    diff_vars = [v for v in trace.posterior.data_vars if v.startswith("diff_")]

    tab1, tab2, tab3 = st.tabs(["📈 全体俯瞰", "🔍 全組み合わせ比較", "🛠️ 診断"])

    # =====================
    # 📈 全体俯瞰
    # =====================
    with tab1:
        st.subheader("平均値の事後分布") 
        st.markdown("""
        - 各群の平均値がとりそうな値の範囲を表します（meanもmuも平均のことです）。
        """)
       
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = plt.cm.tab10.colors
        for i, name in enumerate(current_groups):
            az.plot_kde(trace.posterior[f"mu_{name}"].values.flatten(), ax=ax, label=f"Group {name}", plot_kwargs={"color": colors[i]})
        ax.legend()
        ax.set_title("Posterior Distribution of Means")
        # ax.set_xlabel("Mean Value")
        ax.set_ylabel("Density")
        st.pyplot(fig)

        # 個別の要約統計量
        mu_summary = az.summary(trace, var_names=mu_vars)[["mean", "hdi_3%", "hdi_97%"]]
        st.dataframe(mu_summary)

        with st.expander("解説"):
            st.markdown("""
            - 表のmean列は、各群の平均値がとりそうな値を**点**で推定した結果です。
            - hdi_3%列とhdi_97%列は、各群の平均値がとりそうな値を**区間**で推定した結果です。上図各群の分布においてデータ全体100%から中央の94%を取ってきた両端の値です（Highest Density Interval 事後分布の中で最も密度が高い範囲）。
            """)

            st.warning("""
            ✅ Point: 
            - 入力された観測データ件数が十分に多い、あるいはバラツキ（標準偏差）が小さい場合は、mean列の値と入力データの平均値はほぼ一致します。グラフは尖った形になりやすく、推定結果の確からしさを表します。
            - 入力された観測データ件数が少ない、あるいはバラツキが大きい場合はやや違いが生じるときもあります。グラフは裾が広い形になりやすく、推定結果の不確かさを表します。
            """)

            st.info("""
            🏷️ 推定結果の不確かさが大きいときの対応策: 
            1. 入力データが変な値で読み取られていないか集計値を確認する
            1. A/Bテスト実施中であればテスト期間を延長するなどして観測データ件数を増やす
            1. 左下カラムで推定方法を調整して再度実行してみる
            """)



    # =====================
    # 🔍 全組み合わせ比較
    # =====================
    with tab2:
        st.subheader("2群間の平均値の差分")
        st.markdown("""
        - 2群間の平均値の差がとりそうな値の範囲を表します。
        - 横棒がオレンジ線をまたいでいない（94%HDIが0を含まない）場合は、2群間の平均値に十分な差があると解釈しやすいです。
        - diff_A_B: A群の平均値からB群の平均値を差し引いた値（以降同様です）
        """)
        
        # 出力するグラフの数のもととなる値を取得しておく（グラフのサイズ調整のため）
        num_diffs = len(diff_vars)
 
        # フォレストプロットで2群の差の94%HDIを全て表示
        # 最小で3インチ、1項目あたり0.5インチ + 上下の余白で1.5インチ、といった計算で高さを動的に決める
        forest_height = max(3, num_diffs * 0.5 + 1.5)
        
        fig, ax = plt.subplots(figsize=(8, forest_height))
        az.plot_forest(trace, var_names=diff_vars, combined=True, markersize=4, ax=ax)
        ax.axvline(0, color='tab:orange', linestyle='dashed', alpha=0.5)
        ax.set_title("Summary of Group Differences (94% HDI)")        
        # plt.tight_layout()
        st.pyplot(fig)

        # 詳細数値
        # diff_summary = az.summary(trace, var_names=diff_vars)[["mean", "hdi_3%", "hdi_97%"]]
        # st.dataframe(diff_summary)

        # 2群の差の分布を個別に表示
        if num_diffs > 0:
            # 1行に3つ表示するとした場合の必要行数を計算
            cols = 3
            rows = (num_diffs + cols - 1) // cols  # 切り上げ計算
            
            # 1行あたりの高さを2.5インチとして動的に高さを決める
            dynamic_height = rows * 2.5 
            
            fig, axes = plt.subplots(rows, cols, figsize=(12, dynamic_height))
            
            # 1つの場合はaxesが配列にならないため平坦化
            if num_diffs == 1:
                axes = np.array([axes])
            axes = axes.flatten()

            for i, v in enumerate(diff_vars):
                az.plot_posterior(
                    trace,
                    var_names=[v],
                    ax=axes[i],
                    ref_val=0,
                )
                axes[i].set_title(v)

            # 使わないサブプロット（余白）を非表示にする
            for j in range(num_diffs, len(axes)):
                axes[j].axis("off")

            plt.suptitle("Detail: Absolute Difference", fontsize=20)
            plt.tight_layout()
            st.pyplot(fig)

        with st.expander("解説"):
            st.warning("""
            ✅ Point: 
            - 下段グラフのオレンジ色の数値は、片方の群の平均値がもう片方の群の平均値を上回っている（下回っている）確率を表します。
            - 2群間の平均値の差が0を上回る（下回る）確率が示唆される、これがベイズ推定の利点の一つです（94%HDIに0が含まれていてもこの値で2群間の平均値の差を判断しやすいです）。
            - もしもこの値が50%に近い場合は、平均値に差があるかは不確かです。100%（0%）に近い場合は、平均値の差の確からしさが高まります。
            - この値の判断基準にはp値のような決まりはありません。
              - 例えばこの値が80%以上のとき（20%未満のとき）、80%を確からしいとみなして平均値に差があると判断してもよいです。
              - 80%は不確かとみなして平均値に差はないと判断してもよいです。
              - ここでの推定結果に固執せず、関連する周辺情報も考慮しつつ総合的観点から判断することを強く推奨します。
            """)


    # =====================
    # 🛠️ 診断
    # =====================
    with tab3:
        st.subheader("収束診断")
        st.warning("""
        ✅ Point: 
        - ベイズ推定においては、サンプリング（平均の差がどのくらいの値を取りそうかなどをたくさんシミュレーションしているイメージ）が正常にできているかを確認する必要があります。
        - それぞれの確認項目において問題なさそうであれば、適切な推定結果と解釈しやすいです。
        """)

        with st.expander("Summary"):
            st.markdown("""
            - 目安として、r_hatが1.1以下、mcse_meanが0.01以下、ess_bulkが400以上であれば概ね問題ないと考えてください。
            """)
            # var_names を指定して基本パラメータのみプロットする（Deterministicなdiff_varsを除外）
            main_params = mu_vars + [f"std_{name}" for name in current_groups] + ["nu"]
            # st.dataframe(az.summary(trace))        
            st.dataframe(az.summary(trace, var_names=main_params))

        with st.expander("Trace Plot"):
            st.markdown("""
            - どのグラフも4本の線がプロットされています。
            - 左側のグラフでは4本の線がだいたい同じ形に、右側のグラフでは毛虫のような見た目（縦軸の値が一定範囲を何度も行き来して変動幅がだいたい同じくらい）になっていれば概ね問題ないと考えてください。
            - もしも左右グラフの下部に縦のバーが表示されている場合は少々注意です。左下カラムで推定方法を調整して再実行しても改善しない場合は作成者に相談してみましょう。
            """)        
            # var_names を指定して基本パラメータのみプロットする（Deterministicなdiff_varsを除外）
            # 全部入りだと21個になりarviz基本設定max20を超える（表示はされるがwarningが出る）
            main_params = mu_vars + [f"std_{name}" for name in current_groups] + ["nu"]
            # axes = az.plot_trace(trace)
            axes = az.plot_trace(trace, var_names=main_params)
            plt.tight_layout()
            st.pyplot(axes[0][0].figure)
