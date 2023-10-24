# Árvores de Decisão

Árvores de decisão são modelos [não paramétricos][1]. Isto é: o modelo não assume a priori nada sobre a estrutura dos dados, ao contrário de outros modelos paramétricos, como regressão linear, logística ou até mesmo redes neurais. Por isso, árvores de decisão são modelos bastante flexíveis. 

![Exemplo de árvore de decisão](../imagens/mais-algoritmos/arvores-de-decisao/exemplo1.png)

Árvores de decisão são modelos muito intuitivos porque se assemelham a forma com que construímos programas. Suponha que você quer construir um algoritmo para identificar se uma certa fruta é uma maçã ou um tomate. A maneira intuitiva de escrever um programa seria fazer várias perguntas sobre o objeto, e agir sobre as respostas. Por exemplo:

```python
if fruta.tamanho <= 7:
    if fruta.numero_de_folhas <= 1:
        return 'maça'
    else:
        return 'tomate'
else:    
    if ...    

```

O desafio das ávores de decisão é decidir essas perguntas com base nos dados. Existem árvores de decisão e regressão, mas o funcionamento de ambas é similar.

## Definição matemática
Cada nó pode ser de dois tipos: folha ou interno. Nós internos tomam decisões com base em um dos atributos da entrada e um limiar. Ou seja, as regras de decisão são do tipo $$ X_i \leq t $$, para algum i e algum t. Com base nisso, a decisão é encaminhada a um nó diferente. 

Portanto, cada folha representa uma região diferente do espaço de entrada. Resta, portanto, definir uma saída para cada folha (rótulo no caso de classificação ou saída numérica para regressão).

## Ajuste do modelo

Infelizmente, encontrar a árvore de decisão ótima é um problema NP-Completo (inviável computacionalmente, portanto, para grandes conjuntos de dados). Com isso, precisamos utilizar um algoritmo guloso para aproximar a solução. Portanto, iremos crescer a árvore um nó por vez.

Considere que estamos em um nó $$i$$. Queremos encontrar um limiar e um atributo para fazer a divisão de forma ótima. Mas o que é uma "forma ótima"?

Podemos usar diversas métricas para isso. Para regressão, comumente é utilizado o erro quadrático. Para classificação, métricas comuns são ganho de informação e impureza de Gini. 

Com isso, podemos testar todos possíveis limiares e atributos para encontrar aquele que têm a melhor métrica.

### Ganho de informação
Essa é uma métrica que tenta medir a diminuição de entropia ao dividir o dataset em dois. 

Entropia é uma medida, em termos informais, da "desorganização" dos dados. Podemos definí-la, em termos formais, como o nível médio de "informação" (ou incerteza) carregado pelos dados. Uma distribuição uniforme têm entropia máxima, enquanto uam distribuição onde apenas um item têm probabilidade "1" têm entropia mínima.

Se K é conjunto de classes, então temos que a fórmula para calculá-la é:

$$ E = -\sum_{k \in K} P(y=k)log(P(y=k))$$

Agora, seja $E_T$ a entropia do conjunto completo, e $$E_E$$ e $$E_D$$ a entropia dos subconjuntos. Além disso, sejam $$P_E$$ e $$P_D$$ a proporção dos dados que foram para cada subconjunto. Então, o ganho de informação é a diferença entre a entropia total e a entropia média dos conjuntos separados. Podemos calculá-la da seguinte forma:

$$ GI = E_T - P_L E_L - P_R E_R $$ 

O ganho de informação deve ser maximizado.

### Impureza de Gini
A impureza de Gini calcula a probabilidade de que um elemento aleatório dos dados seja classificado incorretamente, se for classificado aleatoriamente. Se tivermos K classes e $$p_k$$ for a proporção da classe $$k$$, para calculá-la:

$$ IG = \sum_{k=1}^K p_k (\sum_{j \not = k}p_j) = \sum_{k=1}^K p_k (1 - p_k) = 1- \sum_{k=1}^K p_k^2 $$

A Impureza de Gini deve ser minimizada.


## Recursos úteis:
- [Modelos não paramétricos - ScienceDirect][1]
- [Implementação em Python](https://scikit-learn.org/stable/modules/tree.html)
- [Ganho de informação](https://machinelearningmastery.com/information-gain-and-mutual-information/)


[1]: <https://www.sciencedirect.com/topics/engineering/nonparametric-model> (Nonparametric Model - ScienceDirect)

