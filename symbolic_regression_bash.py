import numpy as np
from line_profiler import LineProfiler
from pysr import PySRRegressor
import random
import os
import sys
import math
import pickle
import string



# HELPERS
def generate_pat(n):
  with open('dna.txt', 'r') as f:
    dna = f.read()
    numero_aleatorio = random.randint(1, len(dna)-100)
    pattern = dna[numero_aleatorio: numero_aleatorio + n]
    return dna, pattern

def generate_random_numbers(num_digits):
  # Gerar o limite superior e inferior com base no número de dígitos
  lower_bound = 10 ** (num_digits - 1)
  upper_bound = (10 ** num_digits) - 1
  
  # Gerar dois números aleatórios dentro do intervalo especificado
  random_number1 = random.randint(lower_bound, upper_bound)
  random_number2 = random.randint(lower_bound, upper_bound)
  
  return random_number1, random_number2

def generate_random_string(length):
    return ''.join(random.choice(string.ascii_lowercase) for _ in range(length))

# ALGORITHM

# 01 BINARY SEARCH
def binary_search(l, value): # logN
  low = 0
  high = len(l)-1
  while low <= high: 
    mid = (low+high)//2
    if l[mid] > value: high = mid-1
    elif l[mid] < value: low = mid+1
    else: return mid
  return -1

# 02 BOYER MOORE
def preprocess_strong_suffix(shift, bpos, pat, m):

	# m is the length of pattern
	i = m
	j = m + 1
	bpos[i] = j

	while i > 0:
		
		'''if character at position i-1 is 
		not equivalent to character at j-1, 
		then continue searching to right 
		of the pattern for border '''
		while j <= m and pat[i - 1] != pat[j - 1]:
			
			''' the character preceding the occurrence 
			of t in pattern P is different than the 
			mismatching character in P, we stop skipping
			the occurrences and shift the pattern 
			from i to j '''
			if shift[j] == 0:
				shift[j] = j - i

			# Update the position of next border
			j = bpos[j]
			
		''' p[i-1] matched with p[j-1], border is found. 
		store the beginning position of border '''
		i -= 1
		j -= 1
		bpos[i] = j

# Preprocessing for case 2
def preprocess_case2(shift, bpos, pat, m):
	j = bpos[0]
	for i in range(m + 1):
		
		''' set the border position of the first character 
		of the pattern to all indices in array shift
		having shift[i] = 0 '''
		if shift[i] == 0:
			shift[i] = j
			
		''' suffix becomes shorter than bpos[0], 
		use the position of next widest border
		as value of j '''
		if i == j:
			j = bpos[j]

'''Search for a pattern in given text using 
Boyer Moore algorithm with Good suffix rule '''
def boyer_moore(text, pat): # N (text)+ M (pattern)

	# s is shift of the pattern with respect to text
	s = 0
	m = len(pat)
	n = len(text)

	bpos = [0] * (m + 1)

	# initialize all occurrence of shift to 0
	shift = [0] * (m + 1)

	# do preprocessing
	preprocess_strong_suffix(shift, bpos, pat, m)
	preprocess_case2(shift, bpos, pat, m)

	while s <= n - m:
		j = m - 1
		
		''' Keep reducing index j of pattern while characters of 
			pattern and text are matching at this shift s'''
		while j >= 0 and pat[j] == text[s + j]:
			j -= 1
			
		''' If the pattern is present at the current shift, 
			then index j will become -1 after the above loop '''
		if j < 0:
			s += shift[0]
		else:
			
			'''pat[i] != pat[s+j] so shift the pattern 
			shift[j+1] times '''
			s += shift[j + 1]

# 03 BUBBLESORT
def bubblesort(arr): # x^2
  n = len(arr)
  for i in range(n):
    swapped = False
    for j in range(0, n-i-1):
      if arr[j] > arr[j+1]:
        arr[j], arr[j+1] = arr[j+1], arr[j]
        swapped = True
    if (swapped == False):
      break

# 04 FIBONACCI INTERATIVE
def fibonacci_interative(n): # N
  fib = [0, 1]
  for i in range(2, n+1):
    fib.append(fib[i-1] + fib[i-2])
  return fib[n]

# 05 FIBONACCI RECURSIVE
def fibonacci_recursive(n): # 2^N
  if n == 1 or n == 2:
    return 1
  else:
    return fibonacci_recursive(n-1) + fibonacci_recursive(n-2)

# 06 HEAPSORT
def siftdown(lst, start, end):  
  root = start
  while True:
    child = root * 2 + 1
    if child > end: break
    if child + 1 <= end and lst[child] < lst[child + 1]:
      child += 1
    if lst[root] < lst[child]:
      lst[root], lst[child] = lst[child], lst[root]
      root = child
    else:
      break

def heapsort(lst): # N log N 
  ''' Heapsort. Note: this function sorts in-place (it mutates the list). '''

  # in pseudo-code, heapify only called once, so inline it here
  for start in range(int((len(lst)-2)/2), -1, -1):
    siftdown(lst, start, len(lst)-1)

  for end in range(int(len(lst)-1), 0, -1):
    lst[end], lst[0] = lst[0], lst[end]
    siftdown(lst, 0, end - 1)

# 07 INSERTION SORT
def insertion_sort(L): # x^2
  for i in range(1, len(L)):
    j = i-1 
    key = L[i]
    while j >= 0 and L[j] > key:
        L[j+1] = L[j]
        j -= 1
    L[j+1] = key

# 08 KARATSUBA SORT
def karatsuba(a: int, b: int) -> int:
    """
    >>> karatsuba(15463, 23489) == 15463 * 23489
    True
    >>> karatsuba(3, 9) == 3 * 9
    True
    """
    if len(str(a)) == 1 or len(str(b)) == 1:
        return a * b

    m1 = max(len(str(a)), len(str(b)))
    m2 = m1 // 2

    a1, a2 = divmod(a, 10**m2)
    b1, b2 = divmod(b, 10**m2)

    x = karatsuba(a2, b2)
    y = karatsuba((a1 + a2), (b1 + b2))
    z = karatsuba(a1, b1)

    return (z * 10 ** (2 * m2)) + ((y - z - x) * 10 ** (m2)) + (x)

# 09 LEVENSHTEIN DISTANCE
def levenshtein_distance(str1, str2): # M*N
  m = len(str1)
  n = len(str2)
  d = [[i] for i in range(1, m + 1)]   # d matrix rows
  d.insert(0, list(range(0, n + 1)))   # d matrix columns
  for j in range(1, n + 1): # A 
    for i in range(1, m + 1): # B
      if str1[i - 1] == str2[j - 1]: # C     # Python (string) is 0-based
        substitutionCost = 0
      else: # D
        substitutionCost = 1
      d[i].insert(j, min(d[i - 1][j] + 1,
                          d[i][j - 1] + 1,
                          d[i - 1][j - 1] + substitutionCost))

# 10 LINEAR SEARCH
def linear_search(arr, N, x): # N
  for i in range(0, N):
    if (arr[i] == x):
      return i
  return -1

# 11 MERGESORT
def merge(input_list: list, low: int, mid: int, high: int) -> list: # N Log N
    """
    sorting left-half and right-half individually
    then merging them into result
    """
    result = []
    left, right = input_list[low:mid], input_list[mid : high + 1]
    while left and right:
        result.append((left if left[0] <= right[0] else right).pop(0))
    input_list[low : high + 1] = result + left + right
    return input_list


# iteration over the unsorted list
def iter_merge_sort(input_list: list) -> list:
    if len(input_list) <= 1:
        return input_list
    input_list = list(input_list)

    # iteration for two-way merging
    p = 2
    while p <= len(input_list):
        # getting low, high and middle value for merge-sort of single list
        for i in range(0, len(input_list), p):
            low = i
            high = i + p - 1
            mid = (low + high + 1) // 2
            input_list = merge(input_list, low, mid, high)
        # final merge of last two parts
        if p * 2 >= len(input_list):
            mid = i
            input_list = merge(input_list, 0, mid, len(input_list) - 1)
            break
        p *= 2

# 12 QUICKSORT
def quicksort(sequence): # x^2
  lesser = []
  equal = []
  greater = []
  if len(sequence) <= 1:
      return sequence
  pivot = sequence[0]
  for element in sequence:
      if element < pivot:
          lesser.append(element)
      elif element > pivot:
          greater.append(element)
      else:
          equal.append(element)
  lesser = quicksort(lesser)
  greater = quicksort(greater)

# 13 SELECTION SORT
def selection_sort(array):
  size = len(array)
  for ind in range(size):
      min_index = ind 
      for j in range(ind + 1, size):
          if array[j] < array[min_index]:
              min_index = j
      (array[ind], array[min_index]) = (array[min_index], array[ind])

# 14 SHELLSORT
def shellsort(seq): # N log N ou N^6/5 
  inc = len(seq) // 2
  while inc:
    for i, el in enumerate(seq[inc:], inc):
      while i >= inc and seq[i - inc] > el:
        seq[i] = seq[i - inc]
        i -= inc
      seq[i] = el
    inc = 1 if inc == 2 else inc * 5 // 11

# 15 STRASSEN
def create_matrices(n):
    # Criar duas matrizes de inteiros com dimensão n x n
    matrix1 = [[random.randint(0, 100) for _ in range(n)] for _ in range(n)]
    matrix2 = [[random.randint(0, 100) for _ in range(n)] for _ in range(n)]
    
    return matrix1, matrix2

#Function to add two matrices
def add_matrix(matrix_A, matrix_B, matrix_C, split_index): 
    for i in range(split_index): 
        for j in range(split_index): 
            matrix_C[i][j] = matrix_A[i][j] + matrix_B[i][j] 
            
def multiply_matrix_strassen(matrix_A, matrix_B): 
    col_1 = len(matrix_A[0]) 
    row_1 = len(matrix_A) 
    col_2 = len(matrix_B[0]) 
    row_2 = len(matrix_B) 
 
    if (col_1 != row_2): 
        print("\nError: The number of columns in Matrix A  must be equal to the number of rows in Matrix B\n") 
        return 0
 
    # result_matrix_row = [0] * col_2
    result_matrix = [[0 for x in range(col_2)] for y in range(row_1)] 
 
    if (col_1 == 1): 
        result_matrix[0][0] = matrix_A[0][0] * matrix_B[0][0] 
 
    else: 
        split_index = col_1 // 2
 
        # row_vector = [0] * split_index 
        result_matrix_00 = [[0 for x in range(split_index)] for y in range(split_index)] 
        result_matrix_01 = [[0 for x in range(split_index)] for y in range(split_index)] 
        result_matrix_10 = [[0 for x in range(split_index)] for y in range(split_index)] 
        result_matrix_11 = [[0 for x in range(split_index)] for y in range(split_index)] 
        a00 = [[0 for x in range(split_index)] for y in range(split_index)] 
        a01 = [[0 for x in range(split_index)] for y in range(split_index)] 
        a10 = [[0 for x in range(split_index)] for y in range(split_index)] 
        a11 = [[0 for x in range(split_index)] for y in range(split_index)] 
        b00 = [[0 for x in range(split_index)] for y in range(split_index)] 
        b01 = [[0 for x in range(split_index)] for y in range(split_index)] 
        b10 = [[0 for x in range(split_index)] for y in range(split_index)] 
        b11 = [[0 for x in range(split_index)] for y in range(split_index)] 
 
        for i in range(split_index): 
            for j in range(split_index): 
                a00[i][j] = matrix_A[i][j] 
                a01[i][j] = matrix_A[i][j + split_index] 
                a10[i][j] = matrix_A[split_index + i][j] 
                a11[i][j] = matrix_A[i + split_index][j + split_index] 
                b00[i][j] = matrix_B[i][j] 
                b01[i][j] = matrix_B[i][j + split_index] 
                b10[i][j] = matrix_B[split_index + i][j] 
                b11[i][j] = matrix_B[i + split_index][j + split_index] 
 
        add_matrix(multiply_matrix_strassen(a00, b00),multiply_matrix_strassen(a01, b10),result_matrix_00, split_index)
        add_matrix(multiply_matrix_strassen(a00, b01),multiply_matrix_strassen(a01, b11),result_matrix_01, split_index)
        add_matrix(multiply_matrix_strassen(a10, b00),multiply_matrix_strassen(a11, b10),result_matrix_10, split_index)
        add_matrix(multiply_matrix_strassen(a10, b01),multiply_matrix_strassen(a11, b11),result_matrix_11, split_index)
 
        for i in range(split_index): 
            for j in range(split_index): 
                result_matrix[i][j] = result_matrix_00[i][j] 
                result_matrix[i][j + split_index] = result_matrix_01[i][j] 
                result_matrix[split_index + i][j] = result_matrix_10[i][j] 
                result_matrix[i + split_index][j + split_index] = result_matrix_11[i][j] 
 
    return result_matrix

# 16 TOWER HANOI
def tower_hanoi(ndisks, startPeg=1, endPeg=3): # 2^N
  if ndisks:
    tower_hanoi(ndisks-1, startPeg, 6-startPeg-endPeg)
    # print(f"Move disk {ndisks} from peg {startPeg} to peg {endPeg}")
    tower_hanoi(ndisks-1, 6-startPeg-endPeg, endPeg)


# ==================================
def insumes(n, algorithm_name):
  if algorithm_name == 'binary_search':
    input = random.choices(range(10000), k=n)
    arr = sorted(input)
    return [arr, arr[-1]+1]
  
  if algorithm_name == 'boyer_moore':
    text, pat = generate_pat(n)
    return [text, pat]
  
  if algorithm_name in ['bubblesort', 'heapsort', 'insertion_sort', 'iter_merge_sort', 'quicksort', 'selection_sort', 'shellsort']:
    return [random.choices(range(10000), k=n)]
  
  if algorithm_name in ['fibonacci_interative', 'fibonacci_recursive', 'tower_hanoi']:
    return [n]

  if algorithm_name == 'karatsuba':
    num1, num2 = generate_random_numbers(n)
    return [num1, num2]
  
  if algorithm_name == 'levenshtein_distance':
    string1 = generate_random_string(n)
    string2 = generate_random_string(n*2)
    return [string1, string2]
  
  if algorithm_name == 'linear_search':
    input = random.choices(range(10000), k=n)
    arr = sorted(input)
    return [arr, len(arr), arr[-1]+1]
  
  if algorithm_name == 'multiply_matrix_strassen':
    matrix1, matrix2 = create_matrices(n)
    return [matrix1, matrix2]
  

def frequence_count_method(algorithm_function, algorithm_name, start, end):
  X_y = []

  for n in range(start, end):
    lprofiler = LineProfiler()
    lp_wrapper = lprofiler(algorithm_function)

    input = insumes(n, algorithm_name)

    if len(input) == 3:
      lp_wrapper(input[0], input[1], input[2])
    if len(input) == 2:
      lp_wrapper(input[0], input[1])
    if len(input) == 1:
      lp_wrapper(input[0])

    stats = lprofiler.get_stats()
    line_numbers = []
    hits = []

    for line in stats.timings.values():
      for i in line:
        line_numbers.append(i[0])
        hits.append(i[1])

    X_y.append([n, sum(hits)])

  dados = np.array(X_y)

  X = dados[:, 0]
  y = dados[:, 1]
  X_reshaped = X.reshape(-1, 1)

  return X_reshaped, y


def best_results(X_reshaped, y):
  repeat = 5
  registros = []
  unary_operators_list = ["log", "square", "cube", "sqrt", "round", "exp", "abs"]

  original_stdout = sys.stdout

  with open(os.devnull, 'w') as devnull:
    sys.stdout = devnull

    for i in range(repeat):
      reg1 = PySRRegressor(
      unary_operators = unary_operators_list
      )

      fit1 = reg1.fit(X_reshaped, y)
      best_program1 = fit1.get_best()

      registro1 = []
      for index, value in enumerate(best_program1):
        registro1.append(value)
      
      registros.append(registro1)
      
  sys.stdout = original_stdout

  return registros


def the_best_result(registros):
  for i in registros:
    loss = i[1]
    score = i[2]
    complexity = i[0]
    w = (loss * score)/complexity
    if math.isnan(w):
      i.append(0)
    else:
      i.append(w)

  lista_melhor_valor = max(registros, key=lambda x: x[6])
  novos_dados = lista_melhor_valor[0:3] + [lista_melhor_valor[4]]

  return novos_dados


def salvar_dados(dados, key, arquivo):
  if os.path.exists(arquivo):
    with open(arquivo, 'rb') as f:
      dados_exist = pickle.load(f)
  else:
    dados_exist = {}
    
  valor_original = dados_exist.get(key)
  if valor_original == None:
    dados_exist.update({key: [dados]})
  else:
    if isinstance(valor_original, list):
      valor_original.append(dados)
    else:
      dados_exist.update({key: [dados]})

  with open(arquivo, 'wb') as f:
    pickle.dump(dados_exist, f)


def carregar_dados(arquivo, algorithm_name):
    # Carrega os dados do arquivo pickle
    with open(arquivo, 'rb') as f:
      dados_carregados = pickle.load(f)
    
    print("Conteúdo do arquivo pickle:")
    for k, v in dados_carregados.items():
      if k == algorithm_name:
        print('\u25CF', k)
        for index, item in enumerate(v):
          if index == len(v)-1:
            print('└─', item)
          else:
            print('├─', item)
        print('==========================')

#algorithm_function = [binary_search, boyer_moore, bubblesort, heapsort, insertion_sort, iter_merge_sort, quicksort, selection_sort, shellsort, fibonacci_interative, fibonacci_recursive, tower_hanoi, karatsuba, levenshtein_distance, linear_search, multiply_matrix_strassen]
#algorithm_name = ['binary_search', 'boyer_moore', 'bubblesort', 'heapsort', 'insertion_sort', 'iter_merge_sort', 'quicksort', 'selection_sort', 'shellsort', 'fibonacci_interative', 'fibonacci_recursive', 'tower_hanoi', 'karatsuba', 'levenshtein_distance', 'linear_search', 'multiply_matrix_strassen']

algorithm_function = [selection_sort, shellsort, fibonacci_interative, fibonacci_recursive, tower_hanoi, karatsuba, levenshtein_distance, linear_search, multiply_matrix_strassen]
algorithm_name = ['selection_sort', 'shellsort', 'fibonacci_interative', 'fibonacci_recursive', 'tower_hanoi', 'karatsuba', 'levenshtein_distance', 'linear_search', 'multiply_matrix_strassen']


start_end = {
   'binary_search': [2, 50], 
   'boyer_moore': [50, 55], 
   'bubblesort': [2, 50], 
   'heapsort': [2, 50], 
   'insertion_sort': [2, 50],
   'iter_merge_sort': [2, 50],
   'quicksort': [2, 50],
   'selection_sort': [2, 50],
   'shellsort': [2, 50],
   'fibonacci_interative': [2, 25],
   'fibonacci_recursive': [2, 25],
   'tower_hanoi': [2, 10],
   'karatsuba': [2, 10],
   'levenshtein_distance': [2, 50],
   'linear_search': [2, 50],
   'multiply_matrix_strassen': [2, 5]
}

file_pickle = 'dados_v4.pck'

for i in range(0, len(algorithm_name)):
  for j in range(10):
    X_reshaped, y = frequence_count_method(algorithm_function[i], algorithm_name[i], start_end[algorithm_name[i]][0], start_end[algorithm_name[i]][1])
    registros = best_results(X_reshaped, y)
    dados = the_best_result(registros)
    salvar_dados(dados, algorithm_name[i], file_pickle)
    #dados_carregados = carregar_dados(file_pickle, algorithm_name[i])