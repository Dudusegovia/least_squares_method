# MMQRegressor

Solução robusta para o Método dos Mínimos Quadrados em Python, projetada para ajustar polinômios de grau elevado (10+) sem divergência numérica.
Utiliza aritmética de precisão arbitrária, correção automática para falta de dados e estabilização de matrizes mal-condicionadas.

# 🎯 O Problema

Bibliotecas padrão como NumPy utilizam aritmética de ponto flutuante (float64).
Ao ajustar polinômios de grau alto (ex.: grau 10) ou trabalhar com valores muito grandes (ex.: 2000^10), ocorrem:

Overflow / Underflow

Perda catastrófica de precisão

Coeficientes sem sentido

RankWarning (matriz quase singular)

# 🚀 A Solução: MMQRegressor

O MMQRegressor resolve esses problemas substituindo floats por objetos de precisão arbitrária usando mpmath.
Isso permite cálculos com 50, 100, 200+ casas decimais, garantindo estabilidade mesmo em matrizes de Vandermonde extremamente mal-condicionadas.

# 🔥 Principais Diferenciais
⚡ Precisão Infinita

Não depende de float64.

Você escolhe a precisão (ex.: 200 casas decimais).

# 🛡️ Blindagem Numérica (Data Augmentation)

Detecta automaticamente falta de dados (sistema indeterminado).

Gera micro-variações sintéticas (jittering) para permitir o cálculo sem distorcer a curva.

# 🔧 Regularização Ridge Automática

Aplica Tikhonov somente quando necessário (matriz singular).

# 📊 Normalização Interna

Normaliza dados via Z-score automaticamente:

𝑧
=
𝑥
−
𝜇
𝜎
z=
σ
x−μ
	​


Melhora a estabilidade sem intervenção do usuário.

# 📦 Instalação
git clone https://github.com/dudusegovia/mmq-regressor.git
cd mmq-regressor
pip install -r requirements.txt

# 🛠️ Como Usar

A API segue o padrão Scikit-Learn (fit / predict).

Exemplo 1 — Teste de Estresse (Grau Alto)
from mmq_regressor import MMQRegressor

# Dados que normalmente quebrariam o NumPy devido à magnitude (2015^10)
x = [2010, 2011, 2012, 2013, 2014, 2015]
y = [10, 12, 15, 18, 22, 28]

# 1. Inicializa com alta precisão (100 casas decimais)
# Grau 7 com apenas 6 pontos ativa automaticamente o Data Augmentation
modelo = MMQRegressor(grau=7, precision=100)

# 2. Treinamento
coeficientes = modelo.fit(x, y)

print("Ajuste concluído com sucesso!")
print(f"Coeficientes: {coeficientes}")

# 3. Previsão na escala original
previsao = modelo.predict(2016)

print(f"Previsão para 2016: {previsao:.4f}")

📋 Dependências

Python 3.8+

numpy — operações vetoriais

mpmath — núcleo de alta precisão

📄 Licença

Este projeto está licenciado sob a MIT License.
Consulte o arquivo LICENSE para mais detalhes.
