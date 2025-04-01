import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
import joblib

# Dados de exemplo para treinamento
intencoes = {
    'saudar': [
        'olá', 'oi', 'e aí', 'tudo bem?', 'bom dia', 'boa tarde', 'boa noite',
        'como vai?', 'prazer em conhecê-lo', 'olá tudo bem', 'oi, como vai você'
    ],
    'despedir': [
        'tchau', 'até logo', 'até mais', 'adeus', 'foi um prazer', 'até breve',
        'até amanhã', 'tenha um bom dia', 'boa noite', 'preciso ir agora'
    ],
    'abrir_conta': [
        'quero abrir uma conta', 'como faço para abrir uma conta', 'procedimento para abertura de conta',
        'gostaria de me tornar cliente', 'quais documentos preciso para abrir uma conta',
        'quero me cadastrar', 'posso abrir uma conta online?', 'processo de abertura de conta',
        'requisitos para abrir conta', 'preciso de uma nova conta'
    ]
}

# Preparar dados de treinamento
X = []  # Exemplos de texto
y = []  # Rótulos (intenções)

for intencao, exemplos in intencoes.items():
    for exemplo in exemplos:
        X.append(exemplo)
        y.append(intencao)

# Criar e treinar o modelo
modelo = Pipeline([
    ('vectorizer', CountVectorizer()),
    ('classifier', MultinomialNB())
])

modelo.fit(X, y)

# Salvar o modelo em disco
joblib.dump(modelo, 'intent_model.pkl')

print("Modelo treinado e salvo com sucesso!")

# Teste
exemplos_teste = [
    "olá, como vai?",
    "até logo, obrigado",
    "gostaria de abrir uma conta"
]

for exemplo in exemplos_teste:
    predicao = modelo.predict([exemplo])[0]
    print(f"Texto: '{exemplo}' -> Intenção detectada: '{predicao}'")