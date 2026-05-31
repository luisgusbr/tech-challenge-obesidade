import streamlit as st
import pandas as pd
import joblib

# Carregar arquivos do modelo
modelo = joblib.load("modelo_obesidade.pkl")
encoder = joblib.load("encoder_obesidade.pkl")
features = joblib.load("features_modelo.pkl")

st.set_page_config(
    page_title="Sistema Preditivo de Obesidade",
    page_icon="🏥",
    layout="wide"
)

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
    fcvc = st.selectbox(
        "Consumo de vegetais",
        ["Raramente", "Às vezes", "Sempre"]
    )

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

# Conversões para o padrão do modelo
genero_bin = 0 if genero == "Feminino" else 1
historico_bin = 1 if historico_familiar == "Sim" else 0
favc_bin = 1 if favc == "Sim" else 0
smoke_bin = 1 if smoke == "Sim" else 0
scc_bin = 1 if scc == "Sim" else 0

fcvc_map = {
    "Raramente": 1,
    "Às vezes": 2,
    "Sempre": 3
}

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

# Criar dataframe com uma linha
entrada = pd.DataFrame([dados])

# Criar colunas MTRANS com valor 0
for col in features:
    if col.startswith("MTRANS_"):
        entrada[col] = 0

# Marcar o transporte selecionado
col_mtrans = f"MTRANS_{mtrans}"

if col_mtrans in entrada.columns:
    entrada[col_mtrans] = 1

# Garantir mesma ordem de colunas usada no treinamento
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