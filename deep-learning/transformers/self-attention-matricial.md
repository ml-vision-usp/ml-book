# Noção matricial de Self-Attention

Primeiramente, vamos considerar o caso sem Multi-headed attention. Assim, queremos computar os vetores contextualizados $$y_i, i = 1, \ldots , n$$ onde $$n$$ é o tamanho da sequencia (formada por tokens - ou palavras). Seja $$X_{n \times d}$$ a matriz de input. Essa matriz tem dimensão $$n \times d$$ onde $n$ é a quantidade de palavras na seuquência e $d$ é o tamanho do embedding. Podemos pensar que cada linha dessa matriz é o embedding de uma palavra.

Assim, como comentado anteriormente, usamos as palavras presentes na sequência de três maneiras distintas, associadas à três matrizes de pesos. Portanto, usaremos a seguinte notação:

- $Q_{(n \times d)} = X_{(n \times d)} W_{Q (n \times d)}$
- $K_{(n \times d)} = X_{(n \times d)} W_{K (n \times d)}$
- $V_{(n \times d)} = X_{(n \times d)} W_{V (n \times d)}$

onde $W_Q, W_K, W_V$ são matrizes de pesos treináveis.

Assim, primeiro relacionamos as palavras via dot product para obter os scores. Então, podemos representar esses scores como:
$$\text{Scores} = QK^T$$

Para entender melhor essa operação, usaremos a notação de colunas e linhas do Python, usando slicing. Então, se temos uma matriz $A$, sua i-ésima linha é $A[i, :]$ e sua j-ésima coluna é $A[:, j]$

Então, voltando à matriz $\text{Scores}$, a matriz $Q$ é usada no sentido de ser as palavras sobre as quais queremos computar self-attention (Query) e a matriz $K$ é usada no sentido de possuir as palavras com as quais, para cada palavra em $Q$, iremos computar a relação por meio do dot product. 

Então, se $Q[i, :]_{(1 \times d)}$ é uma palavra (vetor), $\text{Scores}[i, :]_{(1 \times n)}$ é importância contextual dessa palavra pa para todas as palavras da sequência.

Desse modo, perceba que $$\text{Scores}[i, :] = \begin{Bmatrix}
A[i, :] \cdot K^T[:, 1] & A[i, :] \cdot K^T[:, 2] & \cdots & A[i, :] \cdot K^T[:, n]  \\
\end{Bmatrix}$$

Onde 
$$
A[i, :] \cdot K^T[:, j] = v_i \cdot v_j = s_{ij}
$$

No paper Attention is All you Need, há um fator de escala (que é um escalar) para essa multiplicação de matrizes. Esse fator não é tão importante para o entendimento conceitual desse mecanismo. Então, vamos ignorá-lo. 

Depois disso, devemos aplicar a função softmax para normalizar esses scores. Então, temos:

1. $\text{Scores} = QK^T$
2. $\text{softmax}\left( \text{Scores}\right)$ Os elementos dessa matriz são $w_{i,j}, i = 1, \ldots, n ; j = 1, \ldots, n$

Agora que já temos os scores normalizados que indicam a importância contextual (normalizada) de cada palavra relacionada entre si, entramos em um próximo estágio do processo:

O vetor contextualizado (${\bf y}_i$) de uma palavra ${\bf v}_i$ é a soma do produto de todas as palavras e os seus coeficientes de attention em relação à essa palavra ${\bf v}_i$.

Então para obter essa soma, usamos a matriz de palavras $V$. Essa matriz é denominada value, pois estamos usando o valor de cada palavra (após já termos os coeficientes de attention) para obter os vetores contextualizados.

Seja o vetor contextualizado ${\bf y}_i = \begin{Bmatrix} y_{i,1} & y_{i,2} & \cdots & y_{i,d}
\end{Bmatrix}$

onde

$$
y_{i,j} = \sum^n_{k = 1} w_{i,k} v_{k, j}
$$

Perceba que isso é equivalente à multiplicar a linha $i$ da matriz $\text{Scores}$ pela coluna $j$ da matriz $V$. Disso, temos:

$$
\text{Scores}_{(n \times n)}V_{(n \times d)} = C_{(n \times d)}
$$

Onde o $C[i,:] = {\bf y}_i$

Portanto, temos que:

$$
\text{Attention}\left(Q, K , V\right) = \text{softmax}\left(QK^T\right)V
$$

## Multihead Attention

## Fator de escala do produto escalar $QK^T$

## Adicionando Batches à jogada

Quando falamos de deep learning, o método de otimização mais utilizado é Mini-Batch Gradient Descent. Então, como incorporar batches à esse processo de treinamento?

## Reflexão sobre tamanho máximo de sequência

