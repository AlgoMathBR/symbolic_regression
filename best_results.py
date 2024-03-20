import pickle
import math

arquivo = 'dados_v4.pck'

# Carrega os dados do arquivo pickle
with open(arquivo, 'rb') as f:
    dados_carregados = pickle.load(f)

lista_1 = []
for k, v in dados_carregados.items():
    lista_2 = [k]
    for index, item in enumerate(v):
        lista_3 =[]
        loss = item[1]
        score = item[2]
        complexity = item[0]
        funcao = item[3]
        #print(loss, score, complexity)
        w = (loss * score)/complexity
        if math.isnan(w):
            lista_3.append(0)
            lista_3.append(funcao)
        else:
            lista_3.append(w)
            lista_3.append(funcao)
        lista_2.append(lista_3)
    lista_1.append(lista_2)

score = []
for i in lista_1:
    ll = []
    for j in i:
        if isinstance(j, str) == False:
            ll.append(j)
    lista_melhor_valor = max(ll, key=lambda x: x[0])
    score.append([i[0], lista_melhor_valor])


for i in score:
    print(i)
    print('======================')
