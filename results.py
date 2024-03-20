import pickle

arquivo = 'dados_v4.pck'

# Carrega os dados do arquivo pickle
with open(arquivo, 'rb') as f:
    dados_carregados = pickle.load(f)

print("Conteúdo do arquivo pickle:")
for k, v in dados_carregados.items():
    print('\u25CF', k)
    for index, item in enumerate(v):
        if index == len(v)-1:
            print('└─', item)
        else:
            print('├─', item)
    print('==========================')