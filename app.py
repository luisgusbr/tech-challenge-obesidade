import streamlit as st
import pandas as pd
import joblib

# -------------------------------------------------------
# CONFIGURAÇÃO GERAL DA APLICAÇÃO
# -------------------------------------------------------
# Nesta etapa configuramos o título, ícone e layout da aplicação.
# O objetivo foi criar uma interface simples e objetiva para uso
# por profissionais da saúde e avaliadores do projeto.

st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    page_icon="🏥",
    layout="wide"
)

# -------------------------------------------------------
# CARREGAMENTO DOS ARQUIVOS DO MODELO
# -------------------------------------------------------
# Nesta etapa carregamos os arquivos gerados no Google Colab:
#
# 1. modelo_obesidade.pkl  -> modelo Random Forest treinado
# 2. encoder_obesidade.pkl -> tradutor da classe prevista
# 3. features_modelo.pkl   -> lista de variáveis usadas no treinamento
#
# Isso garante que a aplicação utilize exatamente o mesmo modelo
# desenvolvido e avaliado na etapa de Machine Learning.

modelo = joblib.load("modelo_obesidade.pkl")
encoder = joblib.load("encoder_obesidade.pkl")
features = joblib.load("features_modelo.pkl")

# -------------------------------------------------------
# FUNÇÃO DE TRATAMENTO DA BASE PARA O DASHBOARD
# -------------------------------------------------------
# Esta função carrega a base original Obesity.csv e aplica os mesmos
# tratamentos utilizados na etapa de preparação dos dados.
#
# O objetivo é transformar a base técnica em uma base amigável para
# análise de negócio, com campos em português e categorias mais claras
# para a equipe médica.

@st.cache_data
def carregar_base_dashboard():

    # Leitura da base original
    df_obesity = pd.read_csv("Obesity.csv")

    # Renomeação das principais colunas para português
    df_obesity = df_obesity.rename(columns={
        'Gender': 'GENERO',
        'Age': 'IDADE',
        'Height': 'ALTURA',
        'Weight': 'PESO',
        'family_history': 'HISTORICO_FAMILIAR',
        'Obesity': 'NIVEL_OBESIDADE'
    })

    # Tradução do campo gênero
    df_obesity["GENERO"] = df_obesity["GENERO"].replace({
        "Male": "Masculino",
        "Female": "Feminino"
    })

    # Tradução do histórico familiar
    df_obesity["HISTORICO_FAMILIAR"] = df_obesity["HISTORICO_FAMILIAR"].replace({
        "yes": "Sim",
        "no": "Não"
    })

    # Tradução da variável alvo para facilitar a leitura da equipe médica
    df_obesity["NIVEL_OBESIDADE"] = df_obesity["NIVEL_OBESIDADE"].replace({
        "Insufficient_Weight": "Abaixo do Peso",
        "Normal_Weight": "Peso Adequado",
        "Overweight_Level_I": "Sobrepeso Grau I",
        "Overweight_Level_II": "Sobrepeso Grau II",
        "Obesity_Type_I": "Obesidade Grau I",
        "Obesity_Type_II": "Obesidade Grau II",
        "Obesity_Type_III": "Obesidade Grau III"
    })

    # Tratamento dos campos numéricos para facilitar leitura no dashboard
    df_obesity["IDADE"] = df_obesity["IDADE"].astype(int)
    df_obesity["ALTURA"] = df_obesity["ALTURA"].round(2)
    df_obesity["PESO"] = df_obesity["PESO"].round(1)

    # Tradução de campos binários
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

    # Tratamento da frequência de consumo de vegetais
    # Como o dicionário informa que os valores possuem ruído decimal,
    # arredondamos para a categoria inteira mais próxima.
    df_obesity["FCVC"] = df_obesity["FCVC"].round().astype(int)

    df_obesity["FCVC_DESC"] = df_obesity["FCVC"].replace({
        1: "Raramente",
        2: "Às vezes",
        3: "Sempre"
    })

    # Tratamento do número de refeições principais por dia
    df_obesity["NCP"] = df_obesity["NCP"].round().astype(int)

    df_obesity["NCP_DESC"] = df_obesity["NCP"].replace({
        1: "Uma refeição",
        2: "Duas refeições",
        3: "Três refeições",
        4: "Quatro ou mais refeições"
    })

    # Tratamento do consumo diário de água
    df_obesity["CH2O"] = df_obesity["CH2O"].round().astype(int)

    df_obesity["CH2O_DESC"] = df_obesity["CH2O"].replace({
        1: "Menos de 1 litro",
        2: "Entre 1 e 2 litros",
        3: "Mais de 2 litros"
    })

    # Tratamento da frequência de atividade física
    df_obesity["FAF"] = df_obesity["FAF"].round().astype(int)

    df_obesity["FAF_DESC"] = df_obesity["FAF"].replace({
        0: "Nenhuma",
        1: "1 a 2 vezes por semana",
        2: "3 a 4 vezes por semana",
        3: "5 vezes ou mais por semana"
    })

    # Tratamento do tempo de uso de dispositivos eletrônicos
    df_obesity["TUE"] = df_obesity["TUE"].round().astype(int)

    df_obesity["TUE_DESC"] = df_obesity["TUE"].replace({
        0: "0 a 2 horas",
        1: "3 a 5 horas",
        2: "Mais de 5 horas"
    })

    # Tradução do consumo de alimentos entre refeições
    df_obesity["CAEC"] = df_obesity["CAEC"].replace({
        "no": "Não",
        "Sometimes": "Às vezes",
        "Frequently": "Frequentemente",
        "Always": "Sempre"
    })

    # Tradução do consumo de álcool
    df_obesity["CALC"] = df_obesity["CALC"].replace({
        "no": "Não",
        "Sometimes": "Às vezes",
        "Frequently": "Frequentemente",
        "Always": "Sempre"
    })

    # Tradução do meio de transporte habitual
    df_obesity["MTRANS"] = df_obesity["MTRANS"].replace({
        "Public_Transportation": "Transporte Público",
        "Walking": "Caminhada",
        "Automobile": "Automóvel",
        "Motorbike": "Motocicleta",
        "Bike": "Bicicleta"
    })

    # Remoção de registros duplicados para evitar distorções nos indicadores
    df_obesity = df_obesity.drop_duplicates()

    # Criação do IMC como variável analítica
    # Fórmula: IMC = Peso / Altura²
    df_obesity["IMC"] = df_obesity["PESO"] / (df_obesity["ALTURA"] ** 2)

    # Criação de uma visão simplificada para separar pacientes com e sem obesidade
    df_obesity["GRUPO_OBESIDADE"] = df_obesity["NIVEL_OBESIDADE"].apply(
        lambda x: "Com obesidade" if "Obesidade" in x else "Sem obesidade"
    )

    return df_obesity


# -------------------------------------------------------
# MENU LATERAL
# -------------------------------------------------------
# Criamos um menu lateral para separar a aplicação em duas visões:
#
# 1. Sistema Preditivo: onde o usuário informa os dados do paciente
#    e recebe a previsão do nível de obesidade.
#
# 2. Dashboard Analítico: onde são apresentados indicadores e gráficos
#    para apoiar a tomada de decisão da equipe médica.

st.sidebar.title("Menu")

pagina = st.sidebar.radio(
    "Selecione uma página:",
    ["Sistema Preditivo", "Dashboard Analítico"]
)

# -------------------------------------------------------
# PÁGINA 1 - SISTEMA PREDITIVO
# -------------------------------------------------------
# Nesta página disponibilizamos uma interface para entrada dos dados
# do paciente. As respostas são convertidas para o mesmo padrão usado
# no treinamento do modelo.

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

    # -------------------------------------------------------
    # CONVERSÃO DAS ENTRADAS PARA O PADRÃO DO MODELO
    # -------------------------------------------------------
    # O modelo foi treinado com variáveis numéricas.
    # Por isso, as respostas preenchidas em texto na interface
    # são convertidas para códigos numéricos antes da predição.

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

    # Cálculo do IMC do paciente informado
    imc = peso / (altura ** 2)

    # Montagem do registro que será enviado ao modelo
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

    # Tratamento do campo MTRANS
    # No treinamento, o meio de transporte foi convertido em variáveis binárias.
    # Aqui recriamos essas colunas para manter o mesmo padrão do modelo.
    for col in features:
        if col.startswith("MTRANS_"):
            entrada[col] = 0

    col_mtrans = f"MTRANS_{mtrans}"

    if col_mtrans in entrada.columns:
        entrada[col_mtrans] = 1

    # Garantia da mesma ordem das variáveis utilizadas no treinamento
    entrada = entrada[features]

    st.subheader("Resumo do Paciente")
    st.write(f"IMC calculado: **{imc:.2f}**")

    # -------------------------------------------------------
    # EXECUÇÃO DA PREDIÇÃO
    # -------------------------------------------------------
    # Quando o usuário clica no botão, o modelo Random Forest realiza
    # a previsão e o encoder traduz o resultado para a classe textual.

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
# Nesta página apresentamos indicadores e gráficos para apoiar
# a análise exploratória dos dados.
#
# A proposta é que a equipe médica consiga compreender padrões
# importantes relacionados à obesidade, como histórico familiar,
# atividade física, consumo de água, gênero e meio de transporte.

if pagina == "Dashboard Analítico":

    st.title("📊 Dashboard Analítico - Obesidade")
    st.write("Visão analítica para apoiar a equipe médica na compreensão dos principais fatores associados à obesidade.")

    df_dash = carregar_base_dashboard()

    # -------------------------------------------------------
    # KPIs EXECUTIVOS
    # -------------------------------------------------------
    # Criamos indicadores principais para resumir rapidamente
    # o perfil da população analisada.

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
    # Mostra a quantidade de pacientes em cada categoria.
    # Esse gráfico ajuda a equipe médica a identificar os grupos
    # com maior concentração de pacientes.

    st.subheader("Distribuição dos Níveis de Obesidade")

    dist_obesidade = df_dash["NIVEL_OBESIDADE"].value_counts().reset_index()
    dist_obesidade.columns = ["Nível de Obesidade", "Quantidade"]

    st.bar_chart(dist_obesidade.set_index("Nível de Obesidade"))

    st.info(
        "Insight: A distribuição dos níveis de obesidade permite identificar quais grupos concentram maior volume de pacientes e devem receber maior atenção da equipe médica."
    )

    # -------------------------------------------------------
    # GRÁFICO 2 - HISTÓRICO FAMILIAR
    # -------------------------------------------------------
    # Avalia a relação entre histórico familiar de excesso de peso
    # e os níveis de obesidade observados na base.

    st.subheader("Histórico Familiar x Nível de Obesidade")

    hist_familiar = pd.crosstab(
        df_dash["HISTORICO_FAMILIAR"],
        df_dash["NIVEL_OBESIDADE"]
    )

    st.bar_chart(hist_familiar)

    st.info(
        "Insight: Pacientes com histórico familiar de excesso de peso apresentam maior concentração nos níveis de sobrepeso e obesidade."
    )

    # -------------------------------------------------------
    # GRÁFICO 3 - ATIVIDADE FÍSICA
    # -------------------------------------------------------
    # Analisa como a frequência de atividade física se distribui
    # entre os diferentes níveis de obesidade.

    st.subheader("Atividade Física x Nível de Obesidade")

    atividade = pd.crosstab(
        df_dash["FAF_DESC"],
        df_dash["NIVEL_OBESIDADE"]
    )

    st.bar_chart(atividade)

    st.info(
        "Insight: A frequência de atividade física é um fator comportamental relevante para análise preventiva da obesidade."
    )

    # -------------------------------------------------------
    # GRÁFICO 4 - CONSUMO DE ÁGUA
    # -------------------------------------------------------
    # Permite observar padrões de consumo de água entre os grupos
    # de peso corporal.

    st.subheader("Consumo de Água x Nível de Obesidade")

    agua = pd.crosstab(
        df_dash["CH2O_DESC"],
        df_dash["NIVEL_OBESIDADE"]
    )

    st.bar_chart(agua)

    st.info(
        "Insight: O consumo diário de água pode ser analisado em conjunto com outros hábitos de saúde para entender o perfil dos pacientes."
    )

    # -------------------------------------------------------
    # GRÁFICO 5 - GÊNERO
    # -------------------------------------------------------
    # Compara a distribuição dos níveis de obesidade entre homens
    # e mulheres.

    st.subheader("Gênero x Nível de Obesidade")

    genero_obesidade = pd.crosstab(
        df_dash["GENERO"],
        df_dash["NIVEL_OBESIDADE"]
    )

    st.bar_chart(genero_obesidade)

    st.info(
        "Insight: A análise por gênero auxilia a equipe médica a identificar diferenças de perfil entre homens e mulheres."
    )

    # -------------------------------------------------------
    # GRÁFICO 6 - MEIO DE TRANSPORTE
    # -------------------------------------------------------
    # Avalia a relação entre mobilidade cotidiana e níveis de obesidade.
    # Essa análise pode indicar padrões relacionados ao sedentarismo.

    st.subheader("Meio de Transporte x Nível de Obesidade")

    transporte = pd.crosstab(
        df_dash["MTRANS"],
        df_dash["NIVEL_OBESIDADE"]
    )

    st.bar_chart(transporte)

    st.info(
        "Insight: O meio de transporte pode indicar padrões de mobilidade e sedentarismo associados ao nível de obesidade."
    )

    # -------------------------------------------------------
    # BASE TRATADA
    # -------------------------------------------------------
    # Exibimos uma amostra da base tratada para dar transparência
    # ao processo de preparação dos dados.

    st.subheader("Base Tratada para Análise")
    st.dataframe(df_dash.head(50), use_container_width=True)
