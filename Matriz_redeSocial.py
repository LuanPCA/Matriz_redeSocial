import numpy as np
import concurrent.futures
import time

A = np.array([
    [0, 1, 0, 1, 0],
    [0, 1, 1, 0, 1],
    [1, 0, 0, 1, 0],
    [0, 1, 0, 1, 1],
    [1, 0, 1, 0, 0]
])

names = ["Alice", "Bob", "Carol", "David", "Paul"]

def calculate_row(row_index, A, result, times):
    n = len(A)
    start_time = time.time()  # Tempo de início para a linha 
    row_result = []
    for j in range(n):
        value = sum(A[row_index][k] * A[k][j] for k in range(n))
        row_result.append(value)
    end_time = time.time()  # Tempo de fim para a linha 
    result[row_index] = row_result
    times[row_index] = end_time - start_time  # Tempo total para a linha

# Função principal para realizar a multiplicação paralela
def parallel_matrix_multiplication(A):
    n = len(A)
    result = [None] * n
    times = [0] * n  # Lista para armazenar o tempo de execução de cada linha
    start_total_time = time.time()  # Tempo total de início da multiplicação
    
    with concurrent.futures.ThreadPoolExecutor() as executor:
        futures = [executor.submit(calculate_row, i, A, result, times) for i in range(n)]
        concurrent.futures.wait(futures)
    
    total_mult_time = time.time() - start_total_time  # Tempo total de multiplicação
    return np.array(result), times, total_mult_time

# Identifica os influenciadores e amigos em comum a partir de A2
def analyze_influencers(A2):
    start_analysis_time = time.time()  # Tempo de início para a análise
    col_sums = np.sum(A2, axis=0)
    influencers = np.argsort(col_sums)[-2:]  # As duas pessoas com mais amigos
    most_influential = np.argmax(col_sums)   # Pessoa mais influente
    analysis_time = time.time() - start_analysis_time  # Tempo total para a análise
    return influencers, most_influential, analysis_time

# Executa a multiplicação e faz a análise
A2, row_times, total_mult_time = parallel_matrix_multiplication(A)
influencers, most_influential, analysis_time = analyze_influencers(A2)

# Exibe tempos de execução por linha
print("Tempos de execução para cada linha (em segundos):")
for i, t in enumerate(row_times):
    print(f"Linha {i} ({names[i]}): {t:.6f} segundos")

# Exibe a matriz A^2, os influenciadores e a pessoa mais influente com os nomes
print("\nMatriz A^2:\n", A2)
print("\nDuas pessoas com mais amigos em comum:", [names[i] for i in influencers])
print("Pessoa mais influente:", names[most_influential])

# Exibe os tempos totais de execução
print("\nTempo total para calcular a matriz resultante: {:.6f} segundos".format(total_mult_time))
print("Tempo para identificar as pessoas com mais amigos em comum e a pessoa mais influente: {:.6f} segundos".format(analysis_time))

