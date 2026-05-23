import streamlit as st
import joblib
from groq import Groq
import pandas as pd
import numpy as np

modelo = joblib.load('modelo_lightgbm.pkl')
explainer = joblib.load('shap_explainer.pkl')
features = joblib.load('features.pkl')
df = joblib.load('df_fifa_limpio.pkl')

client = Groq(
    api_key="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"
)

st.title("⚽ FIFA AI Scout")

nombre = st.text_input(
    "Ingrese nombre del jugador"
)

if st.button("Analizar"):

    jugador = df[
        df['short_name'].str.lower() ==
        nombre.lower()
    ]

    if len(jugador) == 0:

        st.error("Jugador no encontrado")

    else:

        jugador = jugador.iloc[0]

        X = jugador[features].values.reshape(1, -1)

        pred = modelo.predict(X)[0]

        pred_label = modelo.predict(X)[0]

        pred_texto = pred_label

        pred_idx = np.where(
            modelo.classes_ == pred_label
        )[0][0]

        shap_values = explainer.shap_values(X)

        if isinstance(shap_values, list):
            shap_clase = shap_values[pred_idx]
        else:
            shap_clase = shap_values[0, :, pred_idx]

        if shap_clase.ndim > 1:
            shap_clase = shap_clase.flatten()

        importancia = pd.DataFrame({
            'feature': features,
            'impacto': shap_clase
        })

        importancia['abs'] = importancia['impacto'].abs()

        top = importancia.sort_values(
            'abs',
            ascending=False
        ).head(5)

        factores = ""

        for _, row in top.iterrows():
            factores += f"- {row['feature']}: impacto {row['impacto']:.2f}\n"

        prompt = f"""
Eres un scout profesional de fútbol.

Nombre: {jugador['short_name']}
Edad: {jugador['age']}
Posición: {jugador['player_positions']}
Predicción: {pred_texto}

Factores importantes:
{factores}

Redacta un scouting report profesional.
"""

        respuesta = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            temperature=0.7,
            max_tokens=250
        )

        reporte = respuesta.choices[0].message.content

        st.subheader("Predicción")
        st.success(pred_texto)

        st.subheader("Factores importantes")
        st.dataframe(top[['feature', 'impacto']])

        st.subheader("Scouting Report")
        st.write(reporte)