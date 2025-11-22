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
        
        mp.dps = self.precision

        def _horner_method(val):
            val_mp = mp.mpf(val)
            result = mp.mpf(0)
            
            for coef in self.modelo_final:
                result = result * val_mp + coef
            return result

        if hasattr(x, '__len__') and not isinstance(x, (str, bytes)):
            return [_horner_method(val) for val in x]
        else:
            return _horner_method(x)  
    def function(self):
        """
        Retorna uma função Python independente (callable) que representa o polinômio.
        """
        if self.modelo_final is None:
            raise RuntimeError("Modelo não treinado. Execute .fit() primeiro.")

        coefs_locais = list(self.modelo_final)
        precisao_local = self.precision

        def funcao_polinomial(x):
            mp.dps = precisao_local

            def _calcular_valor(val):
                if hasattr(val, 'item'):
                    val = val.item()
                

                val_mp = mp.mpf(val)
                resultado = mp.mpf(0)
                
                for c in coefs_locais:
                    resultado = resultado * val_mp + c
                return resultado

            if hasattr(x, '__len__') and not isinstance(x, (str, bytes)):
                return [_calcular_valor(v) for v in x]
            else:
                return _calcular_valor(x)

        return funcao_polinomial
    
    def plotar(self, x=None, a=None, b=None, delta=0.01):
        """
        Plota o gráfico do polinômio ajustado.
        
        Pode ser usado de duas formas:
        1. Passando um intervalo [a, b] e um delta.
        2. Passando uma lista/array de pontos x específicos.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[Erro] A biblioteca matplotlib não está instalada.")
            return

        try:
            funcao_poly = self.function()
        except RuntimeError as e:
            print(f"[Erro] {e}")
            return

        x_plot = None

        # Lógica de Definição do Eixo X
        if a is not None and b is not None:
            x_plot = np.arange(a, b, delta)
        elif x is not None:
            x_plot = np.array(sorted(x))
        else:
            print("[Erro] Você deve fornecer 'x' (lista) OU os limites 'a' e 'b'.")
            return

        # Cálculo do Eixo Y
        y_mpmath = funcao_poly(x_plot)
        y_plot = [float(val) for val in y_mpmath]

        # Plotagem
        plt.figure(figsize=(10, 6))
        plt.plot(x_plot, y_plot, color='blue', linewidth=2, label=f'Polinômio (Grau {self.grau})')
        
        plt.title(f"Regressão Polinomial (MMQ) - Grau {self.grau}")
        plt.xlabel("X")
        plt.ylabel("Y")
        plt.grid(True, linestyle='--', alpha=0.6)
        plt.legend()
        plt.show()
    def derivadas(self):
        """
        Retorna:
         - coef_deriv: lista de coeficientes (highest-first) da 1ª derivada
         - coef_2deriv: lista de coeficientes (highest-first) da 2ª derivada
         - f_deriv: callable da 1ª derivada (aceita escalar ou iterável)
         - f_2deriv: callable da 2ª derivada
        """
        if self.modelo_final is None:
            raise RuntimeError("Modelo não treinado. Execute .fit() primeiro.")

        # coeficientes em formato highest-first
        coefs = [mp.mpf(c) for c in self.modelo_final]
        n = len(coefs) - 1  

        # 1ª derivada: c_i * (n - i)
        coef_deriv = []
        for i, c in enumerate(coefs):
            power = n - i
            if power > 0:
                coef_deriv.append(c * mp.mpf(power))

        # 2ª derivada: derivar coef_deriv de novo
        m = len(coef_deriv) - 1
        coef_2deriv = []
        for i, c in enumerate(coef_deriv):
            power = m - i
            if power > 0:
                coef_2deriv.append(c * mp.mpf(power))

        # Horner-based callables (preservam mp.dps localmente)
        def _horner_from_coefs(coefs_list, x):
            mp.dps = self.precision
            def _eval(val):
                if hasattr(val, 'item'):
                    val = val.item()
                val_mp = mp.mpf(val)
                res = mp.mpf(0)
                for cc in coefs_list:
                    res = res * val_mp + cc
                return res
            if hasattr(x, '__len__') and not isinstance(x, (str, bytes)):
                return [_eval(v) for v in x]
            else:
                return _eval(x)

        f_deriv = lambda x: _horner_from_coefs(coef_deriv, x)
        f_2deriv = lambda x: _horner_from_coefs(coef_2deriv, x)

        return coef_deriv, coef_2deriv, f_deriv, f_2deriv

    def max_min(self, func_or_regressor=None, a=None, b=None, delta=0.01, tol=1e-12):
        """
        Versão como método de instância.
        Se `func_or_regressor` for None, usa `self` (a instância).
        Caso contrário aceita um callable ou outra instância/regressor.
        Retorna dicionário com 'global_max', 'global_min', 'maximos', 'minimos'.
        """
        if func_or_regressor is None:
            func_or_regressor = self

        if a is None or b is None:
            raise ValueError("Você deve fornecer a e b (limites do intervalo).")

        mp.dps = getattr(func_or_regressor, 'precision', mp.dps)

        if hasattr(func_or_regressor, 'function') and callable(func_or_regressor.function):
            f = func_or_regressor.function()
            coef_deriv, coef_2deriv, f_deriv, f_2deriv = func_or_regressor.derivadas()
        elif callable(func_or_regressor):
            f = func_or_regressor
            f_deriv = lambda x: mp.diff(lambda t: f(t), mp.mpf(x))
            f_2deriv = lambda x: mp.diff(lambda t: f(t), mp.mpf(x), 2)
        else:
            raise TypeError("func_or_regressor deve ser um callable ou instância de MMQRegressor.")

        a_mp = mp.mpf(a)
        b_mp = mp.mpf(b)
        if b_mp <= a_mp:
            raise ValueError("b deve ser maior que a.")

        def _bisseccao(func, l, r, tol_local=tol, maxiter=200):
            fl = func(l)
            fr = func(r)
            if fl == 0:
                return mp.mpf(l)
            if fr == 0:
                return mp.mpf(r)
            if mp.sign(fl) == mp.sign(fr):
                return None
            left = mp.mpf(l)
            right = mp.mpf(r)
            for _ in range(maxiter):
                mid = (left + right) / 2
                fm = func(mid)
                if abs(fm) <= tol_local or (right - left) / 2 < tol_local:
                    return mp.mpf(mid)
                if mp.sign(fm) == mp.sign(fl):
                    left = mid
                    fl = fm
                else:
                    right = mid
                    fr = fm
            return (left + right) / 2

        xs = []
        x = a_mp
        step = mp.mpf(delta)
        if step <= 0:
            step = mp.mpf('1e-6')
        while x < b_mp:
            xs.append(x)
            x = x + step
        xs.append(b_mp)

        ys = [mp.mpf(f(xi)) for xi in xs]

        max_idx = max(range(len(ys)), key=lambda i: ys[i])
        min_idx = min(range(len(ys)), key=lambda i: ys[i])
        global_max = (mp.mpf(xs[max_idx]), mp.mpf(ys[max_idx]))
        global_min = (mp.mpf(xs[min_idx]), mp.mpf(ys[min_idx]))

        candidatos = []
        for i in range(len(xs)-1):
            l = xs[i]
            r = xs[i+1]
            fl = f_deriv(l)
            fr = f_deriv(r)
            if fl == 0:
                candidatos.append(mp.mpf(l))
            if mp.sign(fl) != mp.sign(fr):
                raiz = _bisseccao(f_deriv, l, r)
                if raiz is not None:
                    candidatos.append(mp.mpf(raiz))
            else:
                if abs(fl) < mp.mpf('1e-10') or abs(fr) < mp.mpf('1e-10'):
                    raiz = _bisseccao(f_deriv, l, r)
                    if raiz is not None:
                        candidatos.append(mp.mpf(raiz))

        candidatos_unique = []
        for c in candidatos:
            if c < a_mp - tol or c > b_mp + tol:
                continue
            found = False
            for ex in candidatos_unique:
                if abs(c - ex) < mp.mpf('1e-10'):
                    found = True
                    break
            if not found:
                candidatos_unique.append(mp.mpf(c))

        maximos = []
        minimos = []

        for c in candidatos_unique:
            y_c = mp.mpf(f(c))
            sec_deriv = mp.mpf(f_2deriv(c))
            if sec_deriv < 0:
                maximos.append((mp.mpf(c), mp.mpf(y_c)))
            elif sec_deriv > 0:
                minimos.append((mp.mpf(c), mp.mpf(y_c)))
            else:
                h = step * mp.mpf('1e-3') if step != 0 else mp.mpf('1e-6')
                left = f_deriv(c - h)
                right = f_deriv(c + h)
                if left > 0 and right < 0:
                    maximos.append((mp.mpf(c), mp.mpf(y_c)))
                elif left < 0 and right > 0:
                    minimos.append((mp.mpf(c), mp.mpf(y_c)))

        for lst in (maximos, minimos):
            for i, (xc, yc) in enumerate(lst):
                y_ref = mp.mpf(f(xc))
                lst[i] = (mp.mpf(xc), mp.mpf(y_ref))
                if y_ref > global_max[1]:
                    global_max = (mp.mpf(xc), mp.mpf(y_ref))
                if y_ref < global_min[1]:
                    global_min = (mp.mpf(xc), mp.mpf(y_ref))

        maximos.sort(key=lambda t: t[0])
        minimos.sort(key=lambda t: t[0])

        return {
            'global_max': global_max,
            'global_min': global_min,
            'maximos': maximos,
            'minimos': minimos
        }

    def maximizar(self, a, b, delta=0.01, tol=1e-12):
        """
        Retorna o máximo global no intervalo [a, b] usando a função max_min.
        Resultado: tupla (x_max, y_max) com valores mp.mpf.
        """
        if self.modelo_final is None:
            raise RuntimeError("Modelo não treinado. Execute .fit(x, y) antes.")
        mp.dps = self.precision
        resultado = self.max_min(a=a, b=b, delta=delta, tol=tol)
        return resultado['global_max']

    def minimizar(self, a, b, delta=0.01, tol=1e-12):
        """
        Retorna o mínimo global no intervalo [a, b] usando a função max_min.
        Resultado: tupla (x_min, y_min) com valores mp.mpf.
        """
        if self.modelo_final is None:
            raise RuntimeError("Modelo não treinado. Execute .fit(x, y) antes.")
        mp.dps = self.precision
        resultado = self.max_min(a=a, b=b, delta=delta, tol=tol)
        return resultado['global_min']
    def integral(self, a, b, delta=0.01):
        """
        Calcula a integral definida da função ajustada no intervalo [a, b]
        usando a Regra do Trapézio com passo 'delta'.
        Retorna um mp.mpf (alta precisão).
        """
        if self.modelo_final is None:
            raise RuntimeError("Modelo não treinado. Execute .fit(x, y) antes.")

        mp.dps = self.precision

        # Garante que a < b
        if b < a:
            a, b = b, a

        # pega o callable que avalia o polinômio
        func = self.function()  

        n = int(mp.ceil((mp.mpf(b) - mp.mpf(a)) / mp.mpf(delta)))
        if n <= 0:
            return mp.mpf('0')

        x = mp.mpf(a)
        area = mp.mpf('0')

        # Trapézios
        for i in range(n):
            x1 = x
            x2 = x + mp.mpf(delta)
            if x2 > mp.mpf(b):
                x2 = mp.mpf(b)
            y1 = mp.mpf(func(x1))
            y2 = mp.mpf(func(x2))
            area += (y1 + y2) * (x2 - x1) / 2
            x = x2

        return area
