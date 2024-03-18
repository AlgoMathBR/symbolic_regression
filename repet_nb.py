import time
import subprocess
import os

diretorio_atual = os.path.dirname(os.path.abspath(__file__))

# Defina o nome do seu notebook
notebook_file = "binary_search_algorithm.ipynb"

caminho_notebook = os.path.join(diretorio_atual, notebook_file)
print(caminho_notebook)
# Defina o número de repetições desejado
num_repeticoes = 1

# Execute o notebook e repita o número especificado de vezes
for i in range(num_repeticoes):
    print(f"Executando o notebook: tentativa {i+1}/{num_repeticoes}")
    subprocess.run(["jupyter", "nbconvert", "--to", "notebook", "--execute", caminho_notebook])
    print("Notebook executado.")
    
    # Tempo de espera entre as execuções (em segundos)
    tempo_espera = 10
    print(f"Aguardando {tempo_espera} segundos antes da próxima execução...")
    time.sleep(tempo_espera)
