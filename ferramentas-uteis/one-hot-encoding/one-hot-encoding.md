# One-Hot Encoding

O one-hot encoding é uma técnica utilizada em *Machine Learning* e processamento de dados para converter variáveis categóricas em uma forma que pode ser fornecida a algoritmos de aprendizado de máquina. Variáveis categóricas são aquelas que têm valores discretos e não ordenados, como "cor" ou "tipo de animal".

A ideia básica do one-hot encoding é transformar cada categoria única em uma nova variável binária (0 ou 1). Esse método cria uma nova coluna para cada valor único na variável categórica original. Em cada coluna, um "1" representa a presença de uma categoria específica, enquanto um "0" representa a ausência dessa categoria.

Vamos usar um exemplo para ilustrar:

## Exemplo

Suponha que temos uma variável categórica chamada "Cor" com três valores possíveis: "Vermelho", "Verde" e "Azul". Com isso, obtemos a seguinte tabela de observações:

|Cor|
|-----|
|Vermelho|
|Verde|
|Azul|
|Vermelho|

Dessa forma, representando "Vermelho" como [1, 0, 0], "Verde" como [0, 1, 0] e "Azul" como [0, 0, 1], podemos fazer a nova tabela:

|Vermelho|Verde|Azul|
|--------|-----|----|
|1       |0    |0   |
|0       |1    |0   |
|0       |0    |1   |
|1       |0    |0   |

Obtendo uma representação numérica de nossas variáveis categóricas.

## Vantagens e Desvantagens do One-Hot Encoding

One-Hot encoding permite a representação de variáveis **sem viés de ordenação implícita**, ou seja, caso substituíssemos cada categoria por um número inteiro, por exemplo, "Vermelho"=1, "Verde"=2 e "Azul"=3, certos modelos de Aprendizado de Máquina interpretariam que a cor vermelha está mais próxima da cor verde do que da cor azul, o que não é necessáriamente verde. Entretanto, uma desvantagem de tal representação é o aumento no número de dimensões vertiginoso à medida que a quantidade de classes aumenta, o que pode causar na criação de modelos muito mais computacionalmente intensivos.

# Contribuições

Fernando Cruz

# Referências

- [One Hot Encoding - O que é?](https://arthurlambletvaz.medium.com/one-hot-encoding-o-que-%C3%A9-cd2e8d302ae0)

- [Geeks for Geeks - One Hot Encoding](https://www.geeksforgeeks.org/ml-one-hot-encoding/)