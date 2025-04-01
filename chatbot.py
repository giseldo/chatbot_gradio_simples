import gradio as gr
import joblib
import os

# Verificar se o modelo existe, se não existir, treinar
if not os.path.exists('intent_model.pkl'):
    import subprocess
    subprocess.run(['python', 'train_intent_model.py'])

# Carregar o modelo
try:
    modelo = joblib.load('intent_model.pkl')
    print("Modelo de classificação de intenções carregado com sucesso!")
except Exception as e:
    print(f"Erro ao carregar o modelo: {e}")
    print("Usando modo de fallback...")
    modelo = None

# Respostas para cada intenção
respostas = {
    'saudar': [
        "Olá! Como posso ajudar você hoje?",
        "Oi! Em que posso ser útil?",
        "Olá, que bom ver você por aqui!",
    ],
    'despedir': [
        "Até logo! Tenha um ótimo dia!",
        "Foi um prazer te ajudar. Até a próxima!",
        "Tchau! Volte sempre que precisar.",
    ],
    'abrir_conta': [
        "Para abrir uma conta você precisará dos seguintes documentos: RG, CPF, comprovante de residência e renda.",
        "Ótimo! Para abrir sua conta, posso te ajudar com o processo online ou você pode visitar uma agência.",
        "Ficaremos felizes em recebê-lo como cliente! Para abrir uma conta, você pode começar o processo online em nossa página de cadastro.",
    ]
}

import random

def response(message, history):
    if modelo is None:
        return f"Modo fallback: você disse: {message}"
    
    # Classificar a intenção do usuário
    try:
        intencao = modelo.predict([message])[0]
        
        # Escolher uma resposta aleatória para a intenção detectada
        if intencao in respostas:
            return random.choice(respostas[intencao])
        else:
            return f"Não entendi sua intenção. Você disse: {message}"
    except Exception as e:
        return f"Erro ao processar: {str(e)}"

demo = gr.ChatInterface(
    fn=response,
    title="Chatbot com Classificação de Intenções",
    description="Um chatbot que identifica suas intenções: saudação, despedida ou abertura de conta."
)

if __name__ == "__main__":
    demo.launch()