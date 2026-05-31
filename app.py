import streamlit as st
import pandas as pd
import joblib
import plotly.express as px

# -------------------------------------------------------
# CONFIGURAÇÃO GERAL DA APLICAÇÃO
# -------------------------------------------------------

st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    page_icon="🏥",
    layout="wide"
)

# -------------------------------------------------------
# CARREGAMENTO DOS ARQUIVOS DO MODELO
# -------------------------------------------------------

modelo = joblib.load("modelo_obesidade.pkl")
encoder = joblib.load("encoder_obesidade.pkl")
features = joblib.load("features_modelo.pkl")

# -------------------------------------------------------
# FUNÇÃO DE TRATAMENTO DA BASE PARA O DASHBOARD
# -------------------------------------------------------

@st.cache_data
def carregar_base_dashboard():

    df_obesity = pd.read_csv("Obesity.csv")

    df_obesity = df_obesity.rename(columns={
        'Gender': 'GENERO',
        'Age': 'IDADE',
        'Height': 'ALTURA',
        'Weight': 'PESO',
        'family_history': 'HISTORICO_FAMILIAR',
        'Obesity': 'NIVEL_OBESIDADE'
    })

    df_obesity["GENERO"] = df_obesity["GENERO"].replace({
        "Male": "Masculino",
        "Female": "Feminino"
    })

    df_obesity["HISTORICO_FAMILIAR"] = df_obesity["HISTORICO_FAMILIAR"].replace({
        "yes": "Sim",
        "no": "Não"
    })

    df_obesity["NIVEL_OBESIDADE"] = df_obesity["NIVEL_OBESIDADE"].replace({
        "Insufficient_Weight": "Abaixo do Peso",
        "Normal_Weight": "Peso Adequado",
        "Overweight_Level_I": "Sobrepeso Grau I",
        "Overweight_Level_II": "Sobrepeso Grau II",
        "Obesity_Type_I": "Obesidade Grau I",
        "Obesity_Type_II": "Obesidade Grau II",
        "Obesity_Type_III": "Obesidade Grau III"
    })

    df_obesity["IDADE"] = df_obesity["IDADE"].astype(int)
    df_obesity["ALTURA"] = df_obesity["ALTURA"].round(2)
    df_obesity["PESO"] = df_obesity["PESO"].round(1)

    df_obesity["FAVC"] = df_obesity["FAVC"].replace({
        "yes": "Sim",
        "no": "Não"
    })

    df_obesity["SMOKE"] = df_obesity["SMOKE"].replace({
        "yes": "Sim",
        "no": "Não"
    })

    df_obesity["SCC"] = df_obesity["SCC"].replace({
        "yes": "Sim",
        "no": "Não"
    })

    df_obesity["FCVC"] = df_obesity["FCVC"].round().astype(int)

    df_obesity["FCVC_DESC"] = df_obesity["FCVC"].replace({
        1: "Raramente",
        2: "Às vezes",
        3: "Sempre"
    })

    df_obesity["NCP"] = df_obesity["NCP"].round().astype(int)

    df_obesity["NCP_DESC"] = df_obesity["NCP"].replace({
        1: "Uma refeição",
        2: "Duas refeições",
        3: "Três refeições",
        4: "Quatro ou mais refeições"
    })

    df_obesity["CH2O"] = df_obesity["CH2O"].round().astype(int)

    df_obesity["CH2O_DESC"] = df_obesity["CH2O"].replace({
        1: "Menos de 1 litro",
        2: "Entre 1 e 2 litros",
        3: "Mais de 2 litros"
    })

    df_obesity["FAF"] = df_obesity["FAF"].round().astype(int)

    df_obesity["FAF_DESC"] = df_obesity["FAF"].replace({
        0: "Nenhuma",
        1: "1 a 2 vezes por semana",
        2: "3 a 4 vezes por semana",
        3: "5 vezes ou mais por semana"
    })

    df_obesity["TUE"] = df_obesity["TUE"].round().astype(int)

    df_obesity["TUE_DESC"] = df_obesity["TUE"].replace({
        0: "0 a 2 horas",
        1: "3 a 5 horas",
        2: "Mais de 5 horas"
    })

    df_obesity["CAEC"] = df_obesity["CAEC"].replace({
        "no": "Não",
        "Sometimes": "Às vezes",
        "Frequently": "Frequentemente",
        "Always": "Sempre"
    })

    df_obesity["CALC"] = df_obesity["CALC"].replace({
        "no": "Não",
        "Sometimes": "Às vezes",
        "Frequently": "Frequentemente",
        "Always": "Sempre"
    })

    df_obesity["MTRANS"] = df_obesity["MTRANS"].replace({
        "Public_Transportation": "Transporte Público",
        "Walking": "Caminhada",
        "Automobile": "Automóvel",
        "Motorbike": "Motocicleta",
        "Bike": "Bicicleta"
    })

    df_obesity = df_obesity.drop_duplicates()

    df_obesity["IMC"] = df_obesity["PESO"] / (df_obesity["ALTURA"] ** 2)

    df_obesity["GRUPO_OBESIDADE"] = df_obesity["NIVEL_OBESIDADE"].apply(
        lambda x: "Com obesidade" if "Obesidade" in x else "Sem obesidade"
    )

    return df_obesity


# -------------------------------------------------------
# FUNÇÃO AUXILIAR PARA CRIAR GRÁFICOS DE BARRAS
# -------------------------------------------------------
# Criamos uma função para padronizar os gráficos do dashboard.
# Assim todos os gráficos ficam com rótulos de dados e aparência uniforme.

def grafico_barras(df, x, y, titulo, cor=None):
    fig = px.bar(
        df,
        x=x,
        y=y,
        text=y,
        title=titulo,
        color=cor
    )

    fig.update_traces(
        textposition="outside"
    )

    fig.update_layout(
        xaxis_title="",
        yaxis_title="Quantidade",
        uniformtext_minsize=8,
        uniformtext_mode="hide"
    )

    return fig


# -------------------------------------------------------
# MENU LATERAL
# -------------------------------------------------------

st.sidebar.title("Menu")

pagina = st.sidebar.radio(
    "Selecione uma página:",
    ["Sistema Preditivo", "Dashboard Analítico"]
)

# -------------------------------------------------------
# PÁGINA 1 - SISTEMA PREDITIVO
# -------------------------------------------------------

if pagina == "Sistema Preditivo":

    st.title("🏥 Sistema Preditivo de Obesidade")
    st.write("Aplicação para auxiliar equipes médicas na previsão do nível de obesidade.")

    st.header("Dados do Paciente")

    col1, col2, col3 = st.columns(3)

    with col1:
        genero = st.selectbox("Gênero", ["Feminino", "Masculino"])
        idade = st.number_input("Idade", min_value=1, max_value=100, value=30)
        altura = st.number_input("Altura (m)", min_value=1.20, max_value=2.30, value=1.70, step=0.01)

    with col2:
        peso = st.number_input("Peso (kg)", min_value=30.0, max_value=250.0, value=80.0, step=0.1)
        historico_familiar = st.selectbox("Histórico familiar de excesso de peso?", ["Não", "Sim"])
        favc = st.selectbox("Consome alimentos calóricos com frequência?", ["Não", "Sim"])

    with col3:
        smoke = st.selectbox("Fuma?", ["Não", "Sim"])
        scc = st.selectbox("Monitora calorias diariamente?", ["Não", "Sim"])
        mtrans = st.selectbox(
            "Meio de transporte habitual",
            ["Transporte Público", "Automóvel", "Caminhada", "Motocicleta", "Bicicleta"]
        )

    st.header("Hábitos de Saúde")

    col4, col5, col6 = st.columns(3)

    with col4:
        fcvc = st.selectbox("Consumo de vegetais", ["Raramente", "Às vezes", "Sempre"])
        ncp = st.selectbox(
            "Refeições principais por dia",
            ["Uma refeição", "Duas refeições", "Três refeições", "Quatro ou mais refeições"]
        )

    with col5:
        ch2o = st.selectbox(
            "Consumo de água diário",
            ["Menos de 1 litro", "Entre 1 e 2 litros", "Mais de 2 litros"]
        )
        faf = st.selectbox(
            "Frequência de atividade física",
            ["Nenhuma", "1 a 2 vezes por semana", "3 a 4 vezes por semana", "5 vezes ou mais por semana"]
        )

    with col6:
        tue = st.selectbox(
            "Tempo em dispositivos eletrônicos",
            ["0 a 2 horas", "3 a 5 horas", "Mais de 5 horas"]
        )
        caec = st.selectbox(
            "Come entre as refeições?",
            ["Não", "Às vezes", "Frequentemente", "Sempre"]
        )
        calc = st.selectbox(
            "Consumo de álcool",
            ["Não", "Às vezes", "Frequentemente", "Sempre"]
        )

    # Conversão das entradas para o padrão usado no modelo
    genero_bin = 0 if genero == "Feminino" else 1
    historico_bin = 1 if historico_familiar == "Sim" else 0
    favc_bin = 1 if favc == "Sim" else 0
    smoke_bin = 1 if smoke == "Sim" else 0
    scc_bin = 1 if scc == "Sim" else 0

    fcvc_map = {"Raramente": 1, "Às vezes": 2, "Sempre": 3}

    ncp_map = {
        "Uma refeição": 1,
        "Duas refeições": 2,
        "Três refeições": 3,
        "Quatro ou mais refeições": 4
    }

    ch2o_map = {
        "Menos de 1 litro": 1,
        "Entre 1 e 2 litros": 2,
        "Mais de 2 litros": 3
    }

    faf_map = {
        "Nenhuma": 0,
        "1 a 2 vezes por semana": 1,
        "3 a 4 vezes por semana": 2,
        "5 vezes ou mais por semana": 3
    }

    tue_map = {
        "0 a 2 horas": 0,
        "3 a 5 horas": 1,
        "Mais de 5 horas": 2
    }

    score_map = {
        "Não": 0,
        "Às vezes": 1,
        "Frequentemente": 2,
        "Sempre": 3
    }

    imc = peso / (altura ** 2)

    dados = {
        "IDADE": idade,
        "ALTURA": altura,
        "PESO": peso,
        "FCVC": fcvc_map[fcvc],
        "NCP": ncp_map[ncp],
        "CH2O": ch2o_map[ch2o],
        "FAF": faf_map[faf],
        "TUE": tue_map[tue],
        "IMC": imc,
        "GENERO_BIN": genero_bin,
        "HISTORICO_FAMILIAR_BIN": historico_bin,
        "FAVC_BIN": favc_bin,
        "SMOKE_BIN": smoke_bin,
        "SCC_BIN": scc_bin,
        "CAEC_SCORE": score_map[caec],
        "CALC_SCORE": score_map[calc]
    }

    entrada = pd.DataFrame([dados])

    for col in features:
        if col.startswith("MTRANS_"):
            entrada[col] = 0

    col_mtrans = f"MTRANS_{mtrans}"

    if col_mtrans in entrada.columns:
        entrada[col_mtrans] = 1

    entrada = entrada[features]

    st.subheader("Resumo do Paciente")
    st.write(f"IMC calculado: **{imc:.2f}**")

    if st.button("Realizar Predição"):
        predicao = modelo.predict(entrada)
        resultado = encoder.inverse_transform(predicao)[0]

        st.success(f"Nível previsto: {resultado}")

        st.info(
            "Este sistema é um apoio à decisão e não substitui a avaliação clínica realizada por profissionais de saúde."
        )


# -------------------------------------------------------
# PÁGINA 2 - DASHBOARD ANALÍTICO
# -------------------------------------------------------

if pagina == "Dashboard Analítico":

    st.title("📊 Dashboard Analítico - Obesidade")
    st.write("Visão analítica para apoiar a equipe médica na compreensão dos principais fatores associados à obesidade.")

    df_dash = carregar_base_dashboard()

    # KPIs principais
    total_pacientes = len(df_dash)
    imc_medio = df_dash["IMC"].mean()
    peso_medio = df_dash["PESO"].mean()

    perc_obesidade = (
        df_dash[df_dash["GRUPO_OBESIDADE"] == "Com obesidade"].shape[0] / total_pacientes
    ) * 100

    col1, col2, col3, col4 = st.columns(4)

    col1.metric("Total de Pacientes", f"{total_pacientes:,}".replace(",", "."))
    col2.metric("IMC Médio", f"{imc_medio:.2f}")
    col3.metric("Peso Médio", f"{peso_medio:.1f} kg")
    col4.metric("% com Obesidade", f"{perc_obesidade:.1f}%")

    st.divider()

    # -------------------------------------------------------
    # GRÁFICO 1 - DISTRIBUIÇÃO DOS NÍVEIS DE OBESIDADE
    # -------------------------------------------------------

    st.subheader("Distribuição dos Níveis de Obesidade")

    dist_obesidade = df_dash["NIVEL_OBESIDADE"].value_counts().reset_index()
    dist_obesidade.columns = ["Nível de Obesidade", "Quantidade"]

    fig_dist = grafico_barras(
        dist_obesidade,
        x="Nível de Obesidade",
        y="Quantidade",
        titulo="Distribuição dos Níveis de Obesidade"
    )

    st.plotly_chart(fig_dist, use_container_width=True)

    st.info(
        "Insight: A distribuição dos níveis de obesidade permite identificar quais grupos concentram maior volume de pacientes e devem receber maior atenção da equipe médica."
    )

    # -------------------------------------------------------
    # GRÁFICO 2 - HISTÓRICO FAMILIAR
    # -------------------------------------------------------

    st.subheader("Histórico Familiar x Nível de Obesidade")

    hist_familiar = (
        df_dash
        .groupby(["HISTORICO_FAMILIAR", "NIVEL_OBESIDADE"])
        .size()
        .reset_index(name="Quantidade")
    )

    fig_hist = px.bar(
        hist_familiar,
        x="HISTORICO_FAMILIAR",
        y="Quantidade",
        color="NIVEL_OBESIDADE",
        text="Quantidade",
        barmode="group",
        title="Histórico Familiar x Nível de Obesidade"
    )

    fig_hist.update_traces(textposition="outside")
    fig_hist.update_layout(xaxis_title="", yaxis_title="Quantidade")

    st.plotly_chart(fig_hist, use_container_width=True)

    st.info(
        "Insight: Pacientes com histórico familiar de excesso de peso apresentam maior concentração nos níveis de sobrepeso e obesidade."
    )

    # -------------------------------------------------------
    # GRÁFICO 3 - ATIVIDADE FÍSICA
    # -------------------------------------------------------

    st.subheader("Atividade Física x Nível de Obesidade")

    atividade = (
        df_dash
        .groupby(["FAF_DESC", "NIVEL_OBESIDADE"])
        .size()
        .reset_index(name="Quantidade")
    )

    fig_atividade = px.bar(
        atividade,
        x="FAF_DESC",
        y="Quantidade",
        color="NIVEL_OBESIDADE",
        text="Quantidade",
        barmode="group",
        title="Atividade Física x Nível de Obesidade"
    )

    fig_atividade.update_traces(textposition="outside")
    fig_atividade.update_layout(xaxis_title="", yaxis_title="Quantidade")

    st.plotly_chart(fig_atividade, use_container_width=True)

    st.info(
        "Insight: A frequência de atividade física é um fator comportamental relevante para análise preventiva da obesidade."
    )

    # -------------------------------------------------------
    # GRÁFICO 4 - CONSUMO DE ÁGUA
    # -------------------------------------------------------

    st.subheader("Consumo de Água x Nível de Obesidade")

    agua = (
        df_dash
        .groupby(["CH2O_DESC", "NIVEL_OBESIDADE"])
        .size()
        .reset_index(name="Quantidade")
    )

    fig_agua = px.bar(
        agua,
        x="CH2O_DESC",
        y="Quantidade",
        color="NIVEL_OBESIDADE",
        text="Quantidade",
        barmode="group",
        title="Consumo de Água x Nível de Obesidade"
    )

    fig_agua.update_traces(textposition="outside")
    fig_agua.update_layout(xaxis_title="", yaxis_title="Quantidade")

    st.plotly_chart(fig_agua, use_container_width=True)

    st.info(
        "Insight: O consumo diário de água pode ser analisado em conjunto com outros hábitos de saúde para entender o perfil dos pacientes."
    )

    # -------------------------------------------------------
    # GRÁFICO 5 - GÊNERO
    # -------------------------------------------------------

    st.subheader("Gênero x Nível de Obesidade")

    genero_obesidade = (
        df_dash
        .groupby(["GENERO", "NIVEL_OBESIDADE"])
        .size()
        .reset_index(name="Quantidade")
    )

    fig_genero = px.bar(
        genero_obesidade,
        x="GENERO",
        y="Quantidade",
        color="NIVEL_OBESIDADE",
        text="Quantidade",
        barmode="group",
        title="Gênero x Nível de Obesidade"
    )

    fig_genero.update_traces(textposition="outside")
    fig_genero.update_layout(xaxis_title="", yaxis_title="Quantidade")

    st.plotly_chart(fig_genero, use_container_width=True)

    st.info(
        "Insight: A análise por gênero auxilia a equipe médica a identificar diferenças de perfil entre homens e mulheres."
    )

    # -------------------------------------------------------
    # GRÁFICO 6 - MEIO DE TRANSPORTE
    # -------------------------------------------------------

    st.subheader("Meio de Transporte x Nível de Obesidade")

    transporte = (
        df_dash
        .groupby(["MTRANS", "NIVEL_OBESIDADE"])
        .size()
        .reset_index(name="Quantidade")
    )

    fig_transporte = px.bar(
        transporte,
        x="MTRANS",
        y="Quantidade",
        color="NIVEL_OBESIDADE",
        text="Quantidade",
        barmode="group",
        title="Meio de Transporte x Nível de Obesidade"
    )

    fig_transporte.update_traces(textposition="outside")
    fig_transporte.update_layout(xaxis_title="", yaxis_title="Quantidade")

    st.plotly_chart(fig_transporte, use_container_width=True)

    st.info(
        "Insight: O meio de transporte pode indicar padrões de mobilidade e sedentarismo associados ao nível de obesidade."
    )

    # Base tratada
    st.subheader("Base Tratada para Análise")
    st.dataframe(df_dash.head(50), use_container_width=True)
