import numpy as np
from mpmath import mp

class MMQRegressor:
    """
    Classe para Regressão Polinomial utilizando o Método dos Mínimos Quadrados (MMQ).
    
    Características:
    - Suporte a aritmética de precisão arbitrária (via mpmath) para evitar erros de arredondamento em graus elevados.
    - Data Augmentation (Jittering) automático para evitar sistemas subdeterminados.
    - Regularização Ridge (Tikhonov) automática para matrizes mal-condicionadas.
    - Normalização interna dos dados (Z-score) para estabilidade numérica.
    """

    def __init__(self, grau: int, precision=50):
        """
        Inicializa o regressor.

        :param grau: O grau do polinômio a ser ajustado.
        :param precision: Casas decimais de precisão para os cálculos internos (padrão: 50).
        """
        self.grau = grau
        self.precision = precision
        self.coeficientes = None
        self.modelo_final = None 
        
        # Configuração global de precisão do mpmath
        mp.dps = self.precision
    def _somar_polinomios(self, p1, p2):
        """Soma dois polinômios representados por listas (grau maior primeiro)."""
        # Iguala o tamanho das listas preenchendo com zeros à esquerda
        diff = len(p1) - len(p2)
        if diff > 0:
            p2 = [mp.mpf(0)] * diff + p2
        elif diff < 0:
            p1 = [mp.mpf(0)] * (-diff) + p1
        
        # Soma termo a termo mantendo precisão mpf
        return [c1 + c2 for c1, c2 in zip(p1, p2)]

    def _multiplicar_polinomios(self, p1, p2):
        """Multiplica dois polinômios (convolução) mantendo precisão mpf."""
        deg1 = len(p1) - 1
        deg2 = len(p2) - 1
        res = [mp.mpf(0)] * (deg1 + deg2 + 1)
        
        for i in range(len(p1)):
            for j in range(len(p2)):
                res[i+j] += p1[i] * p2[j]
        return res
    def _tipos_de_soma(self, x, exp, maior_exp=None):
        """Gera as somas das potências de X para a Matriz de Vandermonde."""
        if maior_exp is None:
            maior_exp = exp
            
        lista_tipos = [[] for _ in range(len(x))]
        for i in range(len(x)):
            for j in range(exp + 1):
                lista_tipos[i].append((x[i] ** maior_exp) * (x[i] ** (exp - j)))
                
        lista_final = []
        if not lista_tipos:
            return lista_final

        for j in range(len(lista_tipos[0])):
            soma = 0
            for i in range(len(lista_tipos)):
                soma += lista_tipos[i][j]
            lista_final.append(soma)
            
        return lista_final

    def _faz_y(self, x, y, exp):
        """Gera o vetor independente do sistema linear."""
        lista_y = []
        for j in range(exp + 1):
            soma = 0
            for i, yv in enumerate(y):
                soma += yv * (x[i] ** (exp - j))
            lista_y.append(soma)
        return lista_y

    def _forma_matriz(self, x, y, exp):
        """Monta a matriz do sistema normal (Xt * X) e o vetor (Xt * y)."""
        matriz_finalizada = []
        for i in range(exp + 1):
            somas_da_vez = self._tipos_de_soma(x, exp, exp - i)
            matriz_finalizada.append(somas_da_vez)
        return [matriz_finalizada, self._faz_y(x, y, exp)]

    def _completar_dados(self, x, y):
        """
        Aplica Data Augmentation (Jittering) se o número de amostras for insuficiente 
        para o grau do polinômio, evitando matrizes singulares.
        """
        pontos_necessarios = self.grau + 1
        qtd_atual = len(x)
        
        if qtd_atual >= pontos_necessarios:
            return x, y
        
        faltam = pontos_necessarios - qtd_atual
        # Log opcional para depuração ou aviso ao usuário
        print(f"[MMQRegressor] Aviso: Dados insuficientes. Gerando {faltam} amostras sintéticas via jittering.")
        
        x_novo = list(x)
        y_novo = list(y)
        epsilon = 1e-10 
        i = 0
        
        while faltam > 0:
            idx = i % qtd_atual
            x_base, y_base = x[idx], y[idx]
            x_fantasma = x_base + epsilon
            
            # Interpolação linear simples para determinar o Y do ponto sintético
            if idx < qtd_atual - 1:
                inclinacao = (y[idx+1] - y_base) / (x[idx+1] - x_base)
                y_fantasma = y_base + inclinacao * epsilon
            else:
                y_fantasma = y_base
                
            x_novo.append(x_fantasma)
            y_novo.append(y_fantasma)
            faltam -= 1
            i += 1
            
        return x_novo, y_novo

    def fit(self, x, y):
        """
        Ajusta o modelo aos dados fornecidos.

        :param x: Array-like com os dados da variável independente.
        :param y: Array-like com os dados da variável dependente.
        :return: Lista com os coeficientes do polinômio ajustado.
        """
        # Garante consistência da precisão
        mp.dps = self.precision
        
        # 1. Pré-processamento (Data Augmentation)
        x_proc, y_proc = self._completar_dados(x, y)
        
        # 2. Normalização Z-score (Estabilidade Numérica)
        self.media_x = np.mean(x_proc)
        self.std_x = np.std(x_proc)
        if self.std_x == 0:
            self.std_x = 1
        
        x_proc = np.array(x_proc) 
        x_norm = (x_proc - self.media_x) / self.std_x
        
        # 3. Montagem do Sistema Linear
        dados = self._forma_matriz(x_norm, y_proc, self.grau)
        matriz_A = mp.matrix(dados[0])
        vetor_B = mp.matrix(dados[1])
        
        solucao_mp = None
        
        # 4. Resolução do Sistema com Fallback para Regularização
        try:
            solucao_mp = mp.lu_solve(matriz_A, vetor_B)
        except Exception:
            # Aplica Regularização de Tikhonov (Ridge) na diagonal
            dim = self.grau + 1
            eps = mp.mpf('1e-100')
            for i in range(dim):
                matriz_A[i, i] += eps
            try:
                solucao_mp = mp.lu_solve(matriz_A, vetor_B)
            except Exception as e:
                print(f"[MMQRegressor] Erro crítico: Falha na resolução do sistema linear. {e}")
                return None

        # 5. Desnormalização e Construção do Polinômio Final
        lista_coefs = [val for val in solucao_mp]
        
        A_mp = mp.mpf(1) / mp.mpf(self.std_x)
        B_mp = mp.mpf(-self.media_x) / mp.mpf(self.std_x)
        
        poly_transformacao = [A_mp, B_mp] 
        
        poly_final = [mp.mpf(0)]
        
        for coef in lista_coefs:
            termo_mult = self._multiplicar_polinomios(poly_final, poly_transformacao)
            
            poly_final = self._somar_polinomios(termo_mult, [coef])
            
        self.coeficientes = poly_final
        self.modelo_final = poly_final 
        
        return self.coeficientes

    def predict(self, x):
        """
        Realiza previsões utilizando o modelo ajustado.

        :param x: Valor único ou array-like de valores para previsão.
        :return: Valores previstos pelo modelo.
        :raises Exception: Se o modelo ainda não tiver sido treinado (.fit).
        """
        if self.modelo_final is None:
            raise RuntimeError("Modelo não treinado. Execute o método .fit(x, y) antes de realizar previsões.")
        
        return np.polyval(self.modelo_final, x)
