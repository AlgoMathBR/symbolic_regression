# UFABC - Disciplina: CCM-107 - Tópicos em Inteligência Artificial 2024.1

## Regressão Simbólica

Nesse projeto estou utilizando Regressão Simbólica para encontrar a complexidade assintótica dos algoritmos.

Cada algoritmo terá seu próprio `notebook - ipynb` pois dessa forma o código fica mais simples para realizar testes e criar, ao invés de criar um notebook genérico para diversos tipos de algoritmos.

O motivo dessa escolha foi porque os algoritmos possuem entradas diferentes, alguns recebem arrays, outros strings, outros reecbem numeros, etc. Além de poderem receber quantidades diferentes de variáveis, tipo 2 arrays, ou 2 strings.

O `y -> saída` dos algoritmos foi encontrado usando o *Método de Contagem de Frequencia* onde é contado quantas vezes cada linha do algoritmo foi chamada, e por fim soma-se essa contagem.

Para executar todos os algoritmos diversas vezes e podendo escolher quantas repetições internas ele deverá fazer, pode executar o arquivo `symbolic_regression_bash.py`.

E para visualizar pode executar o arquivo `results.py`.

E para visualizar os melhores resultados dentre os dados, pode executar o arquivo `best_results.py`.

### Instalação

>pip install numpy

>pip install line-profiler

>pip install pysr

Pode instalar usando o comando abaixo na raiz do projeto
`pip install -r requirements.txt`

Para utilizar, basta abrir o notebook no [COLAB](https://colab.research.google.com/) do Google ou no [Jupyter](https://jupyter.org/install), ou no [VSCode](https://code.visualstudio.com/), onde preferir, e depois executar as células do Notebook

---

## Symbolic Regression

In this project, I'm using Symbolic Regression to find the asymptotic complexity of algorithms.

Each algorithm will have its own `notebook - ipynb` because this way the code becomes simpler to test and create, instead of creating a generic notebook for various types of algorithms.

The reason for this choice is because algorithms have different inputs; some receive arrays, others receive strings, others receive numbers, etc. Besides, they can receive different numbers of variables, like 2 arrays or 2 strings.

The `y -> output` of the algorithms was found using the Frequency Counting Method, where it counts how many times each line of the algorithm was called, and finally adds up this count.

To run all the algorithms multiple times and choose how many internal repetitions it should perform, you can execute the file `symbolic_regression_bash.py`.

And to view, you can execute the file ``results.py`.

And to view the best results among the data, you can execute the file `best_results.py`.

### Installation

>pip install numpy

>pip install line-profiler

>pip install pysr

You can install using the command below in the project root
`pip install -r requirements.txt`

To use it, simply open the notebook in Google COLAB or in Jupyter, or in VSCode, wherever you prefer, and then execute the Notebook cells.

To use it, simply open the notebook in Google [COLAB](https://colab.research.google.com/)  or in [Jupyter](https://jupyter.org/install), or in [VSCode](https://code.visualstudio.com/), wherever you prefer, and then execute the Notebook cells.